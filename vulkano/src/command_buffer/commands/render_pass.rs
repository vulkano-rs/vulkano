// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::pipeline::CheckPipelineError;
use crate::{
    command_buffer::{
        auto::{ClearAttachmentsError, RenderPassState},
        pool::CommandPoolBuilderAlloc,
        synced::{
            Command, CommandBufferState, KeyTy, SyncCommandBufferBuilder,
            SyncCommandBufferBuilderError,
        },
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, BeginRenderPassError,
        PrimaryAutoCommandBuffer, SubpassContents,
    },
    format::{ClearValue, NumericType},
    image::{
        attachment::{ClearAttachment, ClearRect},
        ImageAspects,
    },
    pipeline::GraphicsPipeline,
    render_pass::{Framebuffer, LoadOp},
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStages},
    Version, VulkanObject,
};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::sync::Arc;

/// # Commands for render passes.
///
/// These commands require a graphics queue.
impl<P> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>
where
    P: CommandPoolBuilderAlloc,
{
    /// Adds a command that enters a render pass.
    ///
    /// If `contents` is `SubpassContents::SecondaryCommandBuffers`, then you will only be able to
    /// add secondary command buffers while you're inside the first subpass of the render pass.
    /// If it is `SubpassContents::Inline`, you will only be able to add inline draw commands and
    /// not secondary command buffers.
    ///
    /// C must contain exactly one clear value for each attachment in the framebuffer.
    ///
    /// You must call this before you can add draw commands.
    #[inline]
    pub fn begin_render_pass<I>(
        &mut self,
        framebuffer: Arc<Framebuffer>,
        contents: SubpassContents,
        clear_values: I,
    ) -> Result<&mut Self, BeginRenderPassError>
    where
        I: IntoIterator<Item = ClearValue>,
    {
        unsafe {
            if !self.queue_family().supports_graphics() {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;

            let clear_values: Vec<_> = framebuffer
                .render_pass()
                .convert_clear_values(clear_values)
                .collect();
            let mut clear_values_copy = clear_values.iter().enumerate(); // TODO: Proper errors for clear value errors instead of panics

            for (atch_i, atch_desc) in framebuffer
                .render_pass()
                .attachments()
                .into_iter()
                .enumerate()
            {
                match clear_values_copy.next() {
                    Some((clear_i, clear_value)) => {
                        if atch_desc.load_op == LoadOp::Clear {
                            let aspects = atch_desc
                                .format
                                .map_or(ImageAspects::none(), |f| f.aspects());

                            if aspects.depth && aspects.stencil {
                                assert!(
                                    matches!(clear_value, ClearValue::DepthStencil(_)),
                                    "Bad ClearValue! index: {}, attachment index: {}, expected: DepthStencil, got: {:?}",
                                    clear_i,
                                    atch_i,
                                    clear_value,
                                );
                            } else if aspects.depth {
                                assert!(
                                    matches!(clear_value, ClearValue::Depth(_)),
                                    "Bad ClearValue! index: {}, attachment index: {}, expected: Depth, got: {:?}",
                                    clear_i,
                                    atch_i,
                                    clear_value,
                                );
                            } else if aspects.stencil {
                                assert!(
                                    matches!(clear_value, ClearValue::Stencil(_)),
                                    "Bad ClearValue! index: {}, attachment index: {}, expected: Stencil, got: {:?}",
                                    clear_i,
                                    atch_i,
                                    clear_value,
                                );
                            } else if let Some(numeric_type) =
                                atch_desc.format.and_then(|f| f.type_color())
                            {
                                match numeric_type {
                                    NumericType::SFLOAT
                                    | NumericType::UFLOAT
                                    | NumericType::SNORM
                                    | NumericType::UNORM
                                    | NumericType::SSCALED
                                    | NumericType::USCALED
                                    | NumericType::SRGB => {
                                        assert!(
                                            matches!(clear_value, ClearValue::Float(_)),
                                            "Bad ClearValue! index: {}, attachment index: {}, expected: Float, got: {:?}",
                                            clear_i,
                                            atch_i,
                                            clear_value,
                                        );
                                    }
                                    NumericType::SINT => {
                                        assert!(
                                            matches!(clear_value, ClearValue::Int(_)),
                                            "Bad ClearValue! index: {}, attachment index: {}, expected: Int, got: {:?}",
                                            clear_i,
                                            atch_i,
                                            clear_value,
                                        );
                                    }
                                    NumericType::UINT => {
                                        assert!(
                                            matches!(clear_value, ClearValue::Uint(_)),
                                            "Bad ClearValue! index: {}, attachment index: {}, expected: Uint, got: {:?}",
                                            clear_i,
                                            atch_i,
                                            clear_value,
                                        );
                                    }
                                }
                            } else {
                                panic!("Shouldn't happen!");
                            }
                        } else {
                            assert!(
                                matches!(clear_value, ClearValue::None),
                                "Bad ClearValue! index: {}, attachment index: {}, expected: None, got: {:?}",
                                clear_i,
                                atch_i,
                                clear_value,
                            );
                        }
                    }
                    None => panic!("Not enough clear values"),
                }
            }

            if clear_values_copy.count() != 0 {
                panic!("Too many clear values")
            }

            let render_pass_state = RenderPassState {
                subpass: framebuffer.render_pass().clone().first_subpass(),
                extent: framebuffer.extent(),
                attached_layers_ranges: framebuffer.attached_layers_ranges(),
                contents,
                framebuffer: framebuffer.internal_object(),
            };

            self.inner.begin_render_pass(
                RenderPassBeginInfo {
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                contents,
            )?;

            self.render_pass_state = Some(render_pass_state);
            Ok(self)
        }
    }

    /// Adds a command that ends the current render pass.
    ///
    /// This must be called after you went through all the subpasses and before you can build
    /// the command buffer or add further commands.
    #[inline]
    pub fn end_render_pass(&mut self) -> Result<&mut Self, AutoCommandBufferBuilderContextError> {
        unsafe {
            if let Some(render_pass_state) = self.render_pass_state.as_ref() {
                if !render_pass_state.subpass.is_last_subpass() {
                    return Err(AutoCommandBufferBuilderContextError::NumSubpassesMismatch {
                        actual: render_pass_state.subpass.render_pass().subpasses().len() as u32,
                        current: render_pass_state.subpass.index(),
                    });
                }
            } else {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
            }

            if self.query_state.values().any(|state| state.in_subpass) {
                return Err(AutoCommandBufferBuilderContextError::QueryIsActive);
            }

            debug_assert!(self.queue_family().supports_graphics());

            self.inner.end_render_pass();
            self.render_pass_state = None;
            Ok(self)
        }
    }

    /// Adds a command that jumps to the next subpass of the current render pass.
    #[inline]
    pub fn next_subpass(
        &mut self,
        contents: SubpassContents,
    ) -> Result<&mut Self, AutoCommandBufferBuilderContextError> {
        unsafe {
            if let Some(render_pass_state) = self.render_pass_state.as_mut() {
                if render_pass_state.subpass.try_next_subpass() {
                    render_pass_state.contents = contents;
                } else {
                    return Err(AutoCommandBufferBuilderContextError::NumSubpassesMismatch {
                        actual: render_pass_state.subpass.render_pass().subpasses().len() as u32,
                        current: render_pass_state.subpass.index(),
                    });
                }

                if render_pass_state.subpass.render_pass().views_used() != 0 {
                    // When multiview is enabled, at the beginning of each subpass all non-render pass state is undefined
                    self.inner.reset_state();
                }
            } else {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
            }

            if self.query_state.values().any(|state| state.in_subpass) {
                return Err(AutoCommandBufferBuilderContextError::QueryIsActive);
            }

            debug_assert!(self.queue_family().supports_graphics());

            self.inner.next_subpass(contents);
            Ok(self)
        }
    }

    /// Adds a command that clears specific regions of specific attachments of the framebuffer.
    ///
    /// `attachments` specify the types of attachments and their clear values.
    /// `rects` specify the regions to clear.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). And the command must be inside render pass.
    ///
    /// If the render pass instance this is recorded in uses multiview,
    /// then `ClearRect.base_array_layer` must be zero and `ClearRect.layer_count` must be one.
    ///
    /// The rectangle area must be inside the render area ranges.
    pub fn clear_attachments<A, R>(
        &mut self,
        attachments: A,
        rects: R,
    ) -> Result<&mut Self, ClearAttachmentsError>
    where
        A: IntoIterator<Item = ClearAttachment>,
        R: IntoIterator<Item = ClearRect>,
    {
        let pipeline = check_pipeline_graphics(self.state())?;
        self.ensure_inside_render_pass_inline(pipeline)?;

        let render_pass_state = self.render_pass_state.as_ref().unwrap();
        let subpass = &render_pass_state.subpass;
        let has_depth_stencil_attachment = subpass.has_depth_stencil_attachment();
        let num_color_attachments = subpass.num_color_attachments();
        let attached_layers_ranges = &render_pass_state.attached_layers_ranges;

        let attachments: SmallVec<[ClearAttachment; 3]> = attachments.into_iter().collect();
        let rects: SmallVec<[ClearRect; 4]> = rects.into_iter().collect();

        for attachment in &attachments {
            match attachment {
                ClearAttachment::Color(_, color_attachment) => {
                    if *color_attachment >= num_color_attachments as u32 {
                        return Err(ClearAttachmentsError::InvalidColorAttachmentIndex(
                            *color_attachment,
                        ));
                    }
                }
                ClearAttachment::Depth(_)
                | ClearAttachment::Stencil(_)
                | ClearAttachment::DepthStencil(_) => {
                    if !has_depth_stencil_attachment {
                        return Err(ClearAttachmentsError::DepthStencilAttachmentNotPresent);
                    }
                }
            }
        }

        for rect in &rects {
            if rect.rect_extent[0] == 0 || rect.rect_extent[1] == 0 {
                return Err(ClearAttachmentsError::ZeroRectExtent);
            }
            if rect.rect_offset[0] + rect.rect_extent[0] > render_pass_state.extent[0]
                || rect.rect_offset[1] + rect.rect_extent[1] > render_pass_state.extent[1]
            {
                return Err(ClearAttachmentsError::RectOutOfBounds);
            }

            if rect.layer_count == 0 {
                return Err(ClearAttachmentsError::ZeroLayerCount);
            }
            if subpass.render_pass().views_used() != 0
                && (rect.base_array_layer != 0 || rect.layer_count != 1)
            {
                return Err(ClearAttachmentsError::InvalidMultiviewLayerRange);
            }

            // make sure rect's layers is inside attached layers ranges
            for range in attached_layers_ranges {
                if rect.base_array_layer < range.start
                    || rect.base_array_layer + rect.layer_count > range.end
                {
                    return Err(ClearAttachmentsError::LayersOutOfBounds);
                }
            }
        }

        unsafe {
            self.inner.clear_attachments(attachments, rects);
        }

        Ok(self)
    }
}

fn check_pipeline_graphics(
    current_state: CommandBufferState,
) -> Result<&GraphicsPipeline, CheckPipelineError> {
    let pipeline = match current_state.pipeline_graphics() {
        Some(x) => x,
        None => return Err(CheckPipelineError::PipelineNotBound),
    };

    Ok(pipeline)
}

impl SyncCommandBufferBuilder {
    /// Calls `vkBeginRenderPass` on the builder.
    // TODO: it shouldn't be possible to get an error if the framebuffer checked conflicts already
    // TODO: after begin_render_pass has been called, flushing should be forbidden and an error
    //       returned if conflict
    #[inline]
    pub unsafe fn begin_render_pass(
        &mut self,
        render_pass_begin_info: RenderPassBeginInfo,
        subpass_contents: SubpassContents,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            render_pass_begin_info: Mutex<RenderPassBeginInfo>,
            subpass_contents: SubpassContents,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdBeginRenderPass"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                let mut render_pass_begin_info = self.render_pass_begin_info.lock();
                let clear_values = std::mem::take(&mut render_pass_begin_info.clear_values);

                out.begin_render_pass(
                    RenderPassBeginInfo {
                        framebuffer: render_pass_begin_info.framebuffer.clone(),
                        render_area_offset: render_pass_begin_info.render_area_offset,
                        render_area_extent: render_pass_begin_info.render_area_extent,
                        clear_values,
                        _ne: crate::NonExhaustive(()),
                    },
                    self.subpass_contents,
                );
            }
        }

        let resources = render_pass_begin_info
            .framebuffer
            .render_pass()
            .attachments()
            .iter()
            .enumerate()
            .map(|(num, desc)| {
                (
                    KeyTy::Image(render_pass_begin_info.framebuffer.attachments()[num].image()),
                    format!("attachment {}", num).into(),
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                all_commands: true,
                                ..PipelineStages::none()
                            }, // TODO: wrong!
                            access: AccessFlags {
                                input_attachment_read: true,
                                color_attachment_read: true,
                                color_attachment_write: true,
                                depth_stencil_attachment_read: true,
                                depth_stencil_attachment_write: true,
                                ..AccessFlags::none()
                            }, // TODO: suboptimal
                            exclusive: true, // TODO: suboptimal ; note: remember to always pass true if desc.initial_layout != desc.final_layout
                        },
                        desc.initial_layout,
                        desc.final_layout,
                    )),
                )
            })
            .collect::<Vec<_>>();

        self.append_command(
            Cmd {
                render_pass_begin_info: Mutex::new(render_pass_begin_info),
                subpass_contents,
            },
            resources,
        )?;

        self.latest_render_pass_enter = Some(self.commands.len() - 1);
        Ok(())
    }

    /// Calls `vkCmdNextSubpass` on the builder.
    #[inline]
    pub unsafe fn next_subpass(&mut self, subpass_contents: SubpassContents) {
        struct Cmd {
            subpass_contents: SubpassContents,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdNextSubpass"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.next_subpass(self.subpass_contents);
            }
        }

        self.append_command(Cmd { subpass_contents }, []).unwrap();
    }

    /// Calls `vkCmdEndRenderPass` on the builder.
    #[inline]
    pub unsafe fn end_render_pass(&mut self) {
        struct Cmd;

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdEndRenderPass"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.end_render_pass();
            }
        }

        self.append_command(Cmd, []).unwrap();
        debug_assert!(self.latest_render_pass_enter.is_some());
        self.latest_render_pass_enter = None;
    }

    /// Calls `vkCmdClearAttachments` on the builder.
    ///
    /// Does nothing if the list of attachments or the list of rects is empty, as it would be a
    /// no-op and isn't a valid usage of the command anyway.
    pub unsafe fn clear_attachments<A, R>(&mut self, attachments: A, rects: R)
    where
        A: IntoIterator<Item = ClearAttachment>,
        R: IntoIterator<Item = ClearRect>,
    {
        struct Cmd {
            attachments: Mutex<SmallVec<[ClearAttachment; 3]>>,
            rects: Mutex<SmallVec<[ClearRect; 4]>>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdClearAttachments"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.clear_attachments(
                    self.attachments.lock().drain(..),
                    self.rects.lock().drain(..),
                );
            }
        }
        let attachments: SmallVec<[_; 3]> = attachments.into_iter().collect();
        let rects: SmallVec<[_; 4]> = rects.into_iter().collect();

        self.append_command(
            Cmd {
                attachments: Mutex::new(attachments),
                rects: Mutex::new(rects),
            },
            [],
        )
        .unwrap();
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdBeginRenderPass` on the builder.
    #[inline]
    pub unsafe fn begin_render_pass(
        &mut self,
        render_pass_begin_info: RenderPassBeginInfo,
        subpass_contents: SubpassContents,
    ) {
        let RenderPassBeginInfo {
            framebuffer,
            render_area_offset,
            render_area_extent,
            clear_values,
            _ne: _,
        } = render_pass_begin_info;

        debug_assert!(
            render_area_offset[0] + render_area_extent[0] <= framebuffer.extent()[0]
                && render_area_offset[1] + render_area_extent[1] <= framebuffer.extent()[1]
        );

        let clear_values_vk: SmallVec<[_; 4]> = clear_values
            .into_iter()
            .map(|clear_value| match clear_value {
                ClearValue::None => ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue { float32: [0.0; 4] },
                },
                ClearValue::Float(val) => ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue { float32: val },
                },
                ClearValue::Int(val) => ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue { int32: val },
                },
                ClearValue::Uint(val) => ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue { uint32: val },
                },
                ClearValue::Depth(val) => ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue {
                        depth: val,
                        stencil: 0,
                    },
                },
                ClearValue::Stencil(val) => ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: val,
                    },
                },
                ClearValue::DepthStencil((depth, stencil)) => ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue { depth, stencil },
                },
            })
            .collect();

        let render_pass_begin_info = ash::vk::RenderPassBeginInfo {
            render_pass: framebuffer.render_pass().internal_object(),
            framebuffer: framebuffer.internal_object(),
            render_area: ash::vk::Rect2D {
                offset: ash::vk::Offset2D {
                    x: render_area_offset[0] as i32,
                    y: render_area_offset[1] as i32,
                },
                extent: ash::vk::Extent2D {
                    width: render_area_extent[0],
                    height: render_area_extent[1],
                },
            },
            clear_value_count: clear_values_vk.len() as u32,
            p_clear_values: clear_values_vk.as_ptr(),
            ..Default::default()
        };

        let subpass_begin_info = ash::vk::SubpassBeginInfo {
            contents: subpass_contents.into(),
            ..Default::default()
        };

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_2
            || self.device.enabled_extensions().khr_create_renderpass2
        {
            if self.device.api_version() >= Version::V1_2 {
                fns.v1_2.cmd_begin_render_pass2(
                    self.handle,
                    &render_pass_begin_info,
                    &subpass_begin_info,
                );
            } else {
                fns.khr_create_renderpass2.cmd_begin_render_pass2_khr(
                    self.handle,
                    &render_pass_begin_info,
                    &subpass_begin_info,
                );
            }
        } else {
            debug_assert!(subpass_begin_info.p_next.is_null());

            fns.v1_0.cmd_begin_render_pass(
                self.handle,
                &render_pass_begin_info,
                subpass_begin_info.contents,
            );
        }
    }

    /// Calls `vkCmdNextSubpass` on the builder.
    #[inline]
    pub unsafe fn next_subpass(&mut self, subpass_contents: SubpassContents) {
        let fns = self.device.fns();

        let subpass_begin_info = ash::vk::SubpassBeginInfo {
            contents: subpass_contents.into(),
            ..Default::default()
        };

        let subpass_end_info = ash::vk::SubpassEndInfo::default();

        if self.device.api_version() >= Version::V1_2
            || self.device.enabled_extensions().khr_create_renderpass2
        {
            if self.device.api_version() >= Version::V1_2 {
                fns.v1_2
                    .cmd_next_subpass2(self.handle, &subpass_begin_info, &subpass_end_info);
            } else {
                fns.khr_create_renderpass2.cmd_next_subpass2_khr(
                    self.handle,
                    &subpass_begin_info,
                    &subpass_end_info,
                );
            }
        } else {
            debug_assert!(subpass_begin_info.p_next.is_null());
            debug_assert!(subpass_end_info.p_next.is_null());

            fns.v1_0
                .cmd_next_subpass(self.handle, subpass_begin_info.contents.into());
        }
    }

    /// Calls `vkCmdEndRenderPass` on the builder.
    #[inline]
    pub unsafe fn end_render_pass(&mut self) {
        let fns = self.device.fns();

        let subpass_end_info = ash::vk::SubpassEndInfo::default();

        if self.device.api_version() >= Version::V1_2
            || self.device.enabled_extensions().khr_create_renderpass2
        {
            if self.device.api_version() >= Version::V1_2 {
                fns.v1_2
                    .cmd_end_render_pass2(self.handle, &subpass_end_info);
            } else {
                fns.khr_create_renderpass2
                    .cmd_end_render_pass2_khr(self.handle, &subpass_end_info);
            }
        } else {
            debug_assert!(subpass_end_info.p_next.is_null());

            fns.v1_0.cmd_end_render_pass(self.handle);
        }
    }

    /// Calls `vkCmdClearAttachments` on the builder.
    ///
    /// Does nothing if the list of attachments or the list of rects is empty, as it would be a
    /// no-op and isn't a valid usage of the command anyway.
    #[inline]
    pub unsafe fn clear_attachments(
        &mut self,
        attachments: impl IntoIterator<Item = ClearAttachment>,
        rects: impl IntoIterator<Item = ClearRect>,
    ) {
        let attachments: SmallVec<[_; 3]> = attachments.into_iter().map(|v| v.into()).collect();
        let rects: SmallVec<[_; 4]> = rects
            .into_iter()
            .filter_map(|rect| {
                if rect.layer_count == 0 {
                    return None;
                }

                Some(ash::vk::ClearRect {
                    rect: ash::vk::Rect2D {
                        offset: ash::vk::Offset2D {
                            x: rect.rect_offset[0] as i32,
                            y: rect.rect_offset[1] as i32,
                        },
                        extent: ash::vk::Extent2D {
                            width: rect.rect_extent[0],
                            height: rect.rect_extent[1],
                        },
                    },
                    base_array_layer: rect.base_array_layer,
                    layer_count: rect.layer_count,
                })
            })
            .collect();

        if attachments.is_empty() || rects.is_empty() {
            return;
        }

        let fns = self.device.fns();
        fns.v1_0.cmd_clear_attachments(
            self.handle,
            attachments.len() as u32,
            attachments.as_ptr(),
            rects.len() as u32,
            rects.as_ptr(),
        );
    }
}

/// Parameters to begin a new render pass.
#[derive(Clone, Debug)]
pub struct RenderPassBeginInfo {
    /// The framebuffer to use for rendering.
    ///
    /// There is no default value.
    // TODO: allow passing a different render pass
    pub framebuffer: Arc<Framebuffer>,

    /// The offset from the top left corner of the framebuffer that will be rendered to.
    ///
    /// The default value is `[0, 0]`.
    pub render_area_offset: [u32; 2],

    /// The size of the area that will be rendered to.
    ///
    /// `render_area_offset + render_area_extent` must not be greater than [`framebuffer.extent()`].
    ///
    /// The default value is [`framebuffer.extent()`].
    pub render_area_extent: [u32; 2],

    /// The clear values that should be used for the attachments in the framebuffer.
    ///
    /// There must be exactly [`framebuffer.attachments().len()`] elements provided, and each one
    /// must match the attachment format.
    ///
    /// The default value is empty, which must be overridden if the framebuffer has attachments.
    pub clear_values: Vec<ClearValue>,

    pub _ne: crate::NonExhaustive,
}

impl RenderPassBeginInfo {
    #[inline]
    pub fn framebuffer(framebuffer: Arc<Framebuffer>) -> Self {
        let render_area_extent = framebuffer.extent();

        Self {
            framebuffer,
            render_area_offset: [0, 0],
            render_area_extent,
            clear_values: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}
