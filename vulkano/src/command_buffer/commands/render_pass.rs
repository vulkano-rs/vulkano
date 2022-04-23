// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    command_buffer::{
        auto::RenderPassState,
        pool::CommandPoolBuilderAlloc,
        synced::{Command, Resource, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SubpassContents,
    },
    device::DeviceOwned,
    format::{ClearColorValue, ClearValue, Format, NumericType},
    image::ImageLayout,
    render_pass::{AttachmentDescription, Framebuffer, LoadOp, RenderPass, SubpassDescription},
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStages},
    Version, VulkanObject,
};
use smallvec::SmallVec;
use std::{error, fmt, ops::Range, sync::Arc};

/// # Commands for render passes.
///
/// These commands require a graphics queue.
impl<P> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>
where
    P: CommandPoolBuilderAlloc,
{
    /// Begins a render pass using a render pass object and framebuffer.
    ///
    /// You must call this before you can record draw commands.
    ///
    /// `contents` specifies what kinds of commands will be recorded in the render pass, either
    /// draw commands or executions of secondary command buffers.
    #[inline]
    pub fn begin_render_pass(
        &mut self,
        mut render_pass_begin_info: RenderPassBeginInfo,
        contents: SubpassContents,
    ) -> Result<&mut Self, RenderPassError> {
        self.validate_begin_render_pass(&mut render_pass_begin_info, contents)?;

        unsafe {
            let &RenderPassBeginInfo {
                ref render_pass,
                ref framebuffer,
                render_area_offset,
                render_area_extent,
                clear_values: _,
                _ne: _,
            } = &render_pass_begin_info;

            let render_pass_state = RenderPassState {
                subpass: render_pass.clone().first_subpass(),
                render_area_offset,
                render_area_extent,
                contents,
                framebuffer: Some(framebuffer.clone()),
            };

            self.inner
                .begin_render_pass(render_pass_begin_info, contents)?;

            self.render_pass_state = Some(render_pass_state);
            Ok(self)
        }
    }

    fn validate_begin_render_pass(
        &self,
        render_pass_begin_info: &mut RenderPassBeginInfo,
        contents: SubpassContents,
    ) -> Result<(), RenderPassError> {
        let device = self.device();

        // VUID-vkCmdBeginRenderPass2-commandBuffer-cmdpool
        if !self.queue_family().supports_graphics() {
            return Err(RenderPassError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdBeginRenderPass2-renderpass
        if self.render_pass_state.is_some() {
            return Err(RenderPassError::ForbiddenInsideRenderPass);
        }

        let &mut RenderPassBeginInfo {
            ref render_pass,
            ref framebuffer,
            render_area_offset,
            render_area_extent,
            ref clear_values,
            _ne: _,
        } = render_pass_begin_info;

        // VUID-VkRenderPassBeginInfo-commonparent
        // VUID-vkCmdBeginRenderPass2-framebuffer-02779
        assert_eq!(device, framebuffer.device());

        // VUID-VkRenderPassBeginInfo-renderPass-00904
        if !render_pass.is_compatible_with(framebuffer.render_pass()) {
            return Err(RenderPassError::FramebufferNotCompatible);
        }

        for i in 0..2 {
            // VUID-VkRenderPassBeginInfo-pNext-02852
            // VUID-VkRenderPassBeginInfo-pNext-02853
            if render_area_offset[i] + render_area_extent[i] > framebuffer.extent()[i] {
                return Err(RenderPassError::RenderAreaOutOfBounds);
            }
        }

        for (attachment_index, (attachment_desc, image_view)) in render_pass
            .attachments()
            .iter()
            .zip(framebuffer.attachments())
            .enumerate()
        {
            let attachment_index = attachment_index as u32;
            let &AttachmentDescription {
                initial_layout,
                final_layout,
                ..
            } = attachment_desc;

            for layout in [initial_layout, final_layout] {
                match layout {
                    ImageLayout::ColorAttachmentOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03094
                        if !image_view.usage().color_attachment {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index,
                                usage: "color_attachment",
                            });
                        }
                    }
                    ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03096
                        if !image_view.usage().depth_stencil_attachment {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index,
                                usage: "depth_stencil_attachment",
                            });
                        }
                    }
                    ImageLayout::ShaderReadOnlyOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03097
                        if !(image_view.usage().sampled || image_view.usage().input_attachment) {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index,
                                usage: "sampled or input_attachment",
                            });
                        }
                    }
                    ImageLayout::TransferSrcOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03098
                        if !image_view.usage().transfer_src {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index,
                                usage: "transfer_src",
                            });
                        }
                    }
                    ImageLayout::TransferDstOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03099
                        if !image_view.usage().transfer_dst {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index,
                                usage: "transfer_dst",
                            });
                        }
                    }
                    _ => (),
                }
            }
        }

        for subpass_desc in render_pass.subpasses() {
            let &SubpassDescription {
                view_mask: _,
                ref input_attachments,
                ref color_attachments,
                ref resolve_attachments,
                ref depth_stencil_attachment,
                preserve_attachments: _,
                _ne: _,
            } = subpass_desc;

            for atch_ref in (input_attachments.iter())
                .chain(color_attachments)
                .chain(resolve_attachments)
                .chain([depth_stencil_attachment])
                .flatten()
            {
                let image_view = &framebuffer.attachments()[atch_ref.attachment as usize];

                match atch_ref.layout {
                    ImageLayout::ColorAttachmentOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03094
                        if !image_view.usage().color_attachment {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index: atch_ref.attachment,
                                usage: "color_attachment",
                            });
                        }
                    }
                    ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03096
                        if !image_view.usage().depth_stencil_attachment {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index: atch_ref.attachment,
                                usage: "depth_stencil_attachment",
                            });
                        }
                    }
                    ImageLayout::ShaderReadOnlyOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03097
                        if !(image_view.usage().sampled || image_view.usage().input_attachment) {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index: atch_ref.attachment,
                                usage: "sampled or input_attachment",
                            });
                        }
                    }
                    ImageLayout::TransferSrcOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03098
                        if !image_view.usage().transfer_src {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index: atch_ref.attachment,
                                usage: "transfer_src",
                            });
                        }
                    }
                    ImageLayout::TransferDstOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03099
                        if !image_view.usage().transfer_dst {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index: atch_ref.attachment,
                                usage: "transfer_dst",
                            });
                        }
                    }
                    _ => (),
                }
            }
        }

        // VUID-VkRenderPassBeginInfo-clearValueCount-00902
        if clear_values.len() < render_pass.attachments().len() {
            return Err(RenderPassError::ClearValueMissing {
                attachment_index: clear_values.len() as u32,
            });
        }

        // VUID-VkRenderPassBeginInfo-clearValueCount-04962
        for (attachment_index, (attachment_desc, &clear_value)) in render_pass
            .attachments()
            .iter()
            .zip(clear_values)
            .enumerate()
        {
            let attachment_index = attachment_index as u32;
            let attachment_format = attachment_desc.format.unwrap();

            if attachment_desc.load_op == LoadOp::Clear
                || attachment_desc.stencil_load_op == LoadOp::Clear
            {
                let clear_value = match clear_value {
                    Some(x) => x,
                    None => return Err(RenderPassError::ClearValueMissing { attachment_index }),
                };

                if let (Some(numeric_type), LoadOp::Clear) =
                    (attachment_format.type_color(), attachment_desc.load_op)
                {
                    match numeric_type {
                        NumericType::SFLOAT
                        | NumericType::UFLOAT
                        | NumericType::SNORM
                        | NumericType::UNORM
                        | NumericType::SSCALED
                        | NumericType::USCALED
                        | NumericType::SRGB => {
                            if !matches!(clear_value, ClearValue::Float(_)) {
                                return Err(RenderPassError::ClearValueNotCompatible {
                                    clear_value,
                                    attachment_index,
                                    attachment_format,
                                });
                            }
                        }
                        NumericType::SINT => {
                            if !matches!(clear_value, ClearValue::Int(_)) {
                                return Err(RenderPassError::ClearValueNotCompatible {
                                    clear_value,
                                    attachment_index,
                                    attachment_format,
                                });
                            }
                        }
                        NumericType::UINT => {
                            if !matches!(clear_value, ClearValue::Uint(_)) {
                                return Err(RenderPassError::ClearValueNotCompatible {
                                    clear_value,
                                    attachment_index,
                                    attachment_format,
                                });
                            }
                        }
                    }
                } else {
                    let attachment_aspects = attachment_format.aspects();
                    let need_depth =
                        attachment_aspects.depth && attachment_desc.load_op == LoadOp::Clear;
                    let need_stencil = attachment_aspects.stencil
                        && attachment_desc.stencil_load_op == LoadOp::Clear;

                    if need_depth && need_stencil {
                        if !matches!(clear_value, ClearValue::DepthStencil(_)) {
                            return Err(RenderPassError::ClearValueNotCompatible {
                                clear_value,
                                attachment_index,
                                attachment_format,
                            });
                        }
                    } else if need_depth {
                        if !matches!(clear_value, ClearValue::Depth(_)) {
                            return Err(RenderPassError::ClearValueNotCompatible {
                                clear_value,
                                attachment_index,
                                attachment_format,
                            });
                        }
                    } else if need_stencil {
                        if !matches!(clear_value, ClearValue::Stencil(_)) {
                            return Err(RenderPassError::ClearValueNotCompatible {
                                clear_value,
                                attachment_index,
                                attachment_format,
                            });
                        }
                    }
                }
            }
        }

        // VUID-vkCmdBeginRenderPass2-initialLayout-03100
        // If the initialLayout member of any of the VkAttachmentDescription structures specified when creating the render pass specified in the renderPass member of pRenderPassBegin is not VK_IMAGE_LAYOUT_UNDEFINED, then
        // each such initialLayout must be equal to the current layout of the corresponding attachment image subresource of the framebuffer specified in the framebuffer member of pRenderPassBegin

        // VUID-vkCmdBeginRenderPass2-srcStageMask-06453
        // TODO:

        // VUID-vkCmdBeginRenderPass2-dstStageMask-06454
        // TODO:

        // VUID-vkCmdBeginRenderPass2-framebuffer-02533
        // For any attachment in framebuffer that is used by renderPass and is bound to memory locations that are also bound to another attachment used by renderPass, and if at least one of those uses causes either
        // attachment to be written to, both attachments must have had the VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT set

        Ok(())
    }

    /// Advances to the next subpass of the render pass previously begun with `begin_render_pass`.
    #[inline]
    pub fn next_subpass(
        &mut self,
        contents: SubpassContents,
    ) -> Result<&mut Self, RenderPassError> {
        self.validate_next_subpass(contents)?;

        unsafe {
            if let Some(render_pass_state) = self.render_pass_state.as_mut() {
                render_pass_state.subpass.next_subpass();
                render_pass_state.contents = contents;

                if render_pass_state.subpass.render_pass().views_used() != 0 {
                    // When multiview is enabled, at the beginning of each subpass, all
                    // non-render pass state is undefined.
                    self.inner.reset_state();
                }
            }

            self.inner.next_subpass(contents);
        }

        Ok(self)
    }

    fn validate_next_subpass(&self, contents: SubpassContents) -> Result<(), RenderPassError> {
        // VUID-vkCmdNextSubpass2-renderpass
        let render_pass_state = self
            .render_pass_state
            .as_ref()
            .ok_or_else(|| RenderPassError::ForbiddenOutsideRenderPass)?;

        // VUID-vkCmdNextSubpass2-None-03102
        if render_pass_state.subpass.is_last_subpass() {
            return Err(RenderPassError::NoSubpassesRemaining {
                current_subpass: render_pass_state.subpass.index(),
            });
        }

        // VUID?
        if self.query_state.values().any(|state| state.in_subpass) {
            return Err(RenderPassError::QueryIsActive);
        }

        // VUID-vkCmdNextSubpass2-commandBuffer-cmdpool
        debug_assert!(self.queue_family().supports_graphics());

        // VUID-vkCmdNextSubpass2-bufferlevel
        // Ensured by the type of the impl block

        Ok(())
    }

    /// Ends the render pass previously begun with `begin_render_pass`.
    ///
    /// This must be called after you went through all the subpasses.
    #[inline]
    pub fn end_render_pass(&mut self) -> Result<&mut Self, RenderPassError> {
        self.validate_end_render_pass()?;

        unsafe {
            self.inner.end_render_pass();
            self.render_pass_state = None;
        }

        Ok(self)
    }

    fn validate_end_render_pass(&self) -> Result<(), RenderPassError> {
        // VUID-vkCmdEndRenderPass2-renderpass
        let render_pass_state = self
            .render_pass_state
            .as_ref()
            .ok_or_else(|| RenderPassError::ForbiddenOutsideRenderPass)?;

        // VUID-vkCmdEndRenderPass2-None-03103
        if !render_pass_state.subpass.is_last_subpass() {
            return Err(RenderPassError::SubpassesRemaining {
                current_subpass: render_pass_state.subpass.index(),
                remaining_subpasses: render_pass_state.subpass.render_pass().subpasses().len()
                    as u32
                    - render_pass_state.subpass.index(),
            });
        }

        // VUID?
        if self.query_state.values().any(|state| state.in_subpass) {
            return Err(RenderPassError::QueryIsActive);
        }

        // VUID-vkCmdEndRenderPass2-commandBuffer-cmdpool
        debug_assert!(self.queue_family().supports_graphics());

        // VUID-vkCmdEndRenderPass2-bufferlevel
        // Ensured by the type of the impl block

        Ok(())
    }
}

impl<L, P> AutoCommandBufferBuilder<L, P> {
    /// Clears specific regions of specific attachments of the framebuffer.
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
    pub fn clear_attachments(
        &mut self,
        attachments: impl IntoIterator<Item = ClearAttachment>,
        rects: impl IntoIterator<Item = ClearRect>,
    ) -> Result<&mut Self, RenderPassError> {
        let attachments: SmallVec<[ClearAttachment; 3]> = attachments.into_iter().collect();
        let rects: SmallVec<[ClearRect; 4]> = rects.into_iter().collect();

        self.validate_clear_attachments(&attachments, &rects)?;

        unsafe {
            self.inner.clear_attachments(attachments, rects);
        }

        Ok(self)
    }

    fn validate_clear_attachments(
        &self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> Result<(), RenderPassError> {
        // VUID-vkCmdClearAttachments-renderpass
        let render_pass_state = self
            .render_pass_state
            .as_ref()
            .ok_or(RenderPassError::ForbiddenOutsideRenderPass)?;

        if render_pass_state.contents != SubpassContents::Inline {
            return Err(RenderPassError::ForbiddenWithSubpassContents {
                subpass_contents: render_pass_state.contents,
            });
        }

        let subpass_desc = render_pass_state.subpass.subpass_desc();
        let render_pass = render_pass_state.subpass.render_pass();
        let is_multiview = render_pass.views_used() != 0;

        for attachment in attachments {
            match attachment {
                &ClearAttachment::Color {
                    color_attachment,
                    clear_value,
                } => {
                    // VUID-vkCmdClearAttachments-aspectMask-02501
                    let atch_ref = match subpass_desc
                        .color_attachments
                        .get(color_attachment as usize)
                    {
                        Some(x) => x.as_ref(),
                        None => {
                            return Err(RenderPassError::ColorAttachmentIndexOutOfRange {
                                color_attachment_index: color_attachment,
                                num_color_attachments: subpass_desc.color_attachments.len() as u32,
                            });
                        }
                    };

                    if let Some(atch_ref) = atch_ref {
                        let attachment_desc =
                            render_pass.attachments()[atch_ref.attachment as usize];

                        if let Some(numeric_type) = attachment_desc.format.unwrap().type_color() {
                            match numeric_type {
                                NumericType::SFLOAT
                                | NumericType::UFLOAT
                                | NumericType::SNORM
                                | NumericType::UNORM
                                | NumericType::SSCALED
                                | NumericType::USCALED
                                | NumericType::SRGB => {
                                    if !matches!(clear_value, ClearColorValue::Float(_)) {
                                        return Err(RenderPassError::ClearValueNotCompatible {
                                            clear_value: clear_value.into(),
                                            attachment_index: atch_ref.attachment,
                                            attachment_format: attachment_desc.format.unwrap(),
                                        });
                                    }
                                }
                                NumericType::SINT => {
                                    if !matches!(clear_value, ClearColorValue::Int(_)) {
                                        return Err(RenderPassError::ClearValueNotCompatible {
                                            clear_value: clear_value.into(),
                                            attachment_index: atch_ref.attachment,
                                            attachment_format: attachment_desc.format.unwrap(),
                                        });
                                    }
                                }
                                NumericType::UINT => {
                                    if !matches!(clear_value, ClearColorValue::Uint(_)) {
                                        return Err(RenderPassError::ClearValueNotCompatible {
                                            clear_value: clear_value.into(),
                                            attachment_index: atch_ref.attachment,
                                            attachment_format: attachment_desc.format.unwrap(),
                                        });
                                    }
                                }
                            }
                        } else {
                            unreachable!()
                        }
                    }
                }
                ClearAttachment::Depth(_)
                | ClearAttachment::Stencil(_)
                | ClearAttachment::DepthStencil(_) => {
                    if let Some(atch_ref) = &subpass_desc.depth_stencil_attachment {
                        let attachment_desc =
                            render_pass.attachments()[atch_ref.attachment as usize];
                        let aspects = attachment_desc.format.unwrap().aspects();

                        match attachment {
                            &ClearAttachment::Depth(val) => {
                                // VUID-vkCmdClearAttachments-aspectMask-02502
                                if !aspects.depth {
                                    return Err(RenderPassError::ClearValueNotCompatible {
                                        clear_value: val.into(),
                                        attachment_index: atch_ref.attachment,
                                        attachment_format: attachment_desc.format.unwrap(),
                                    });
                                }
                            }
                            &ClearAttachment::Stencil(val) => {
                                // VUID-vkCmdClearAttachments-aspectMask-02503
                                if !aspects.stencil {
                                    return Err(RenderPassError::ClearValueNotCompatible {
                                        clear_value: val.into(),
                                        attachment_index: atch_ref.attachment,
                                        attachment_format: attachment_desc.format.unwrap(),
                                    });
                                }
                            }
                            &ClearAttachment::DepthStencil(val) => {
                                // VUID-vkCmdClearAttachments-aspectMask-02502
                                // VUID-vkCmdClearAttachments-aspectMask-02503
                                if !(aspects.depth && aspects.stencil) {
                                    return Err(RenderPassError::ClearValueNotCompatible {
                                        clear_value: val.into(),
                                        attachment_index: atch_ref.attachment,
                                        attachment_format: attachment_desc.format.unwrap(),
                                    });
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }

        for (rect_index, rect) in rects.iter().enumerate() {
            for i in 0..2 {
                // VUID-vkCmdClearAttachments-rect-02682
                // VUID-vkCmdClearAttachments-rect-02683
                if rect.extent[i] == 0 {
                    return Err(RenderPassError::RectExtentZero { rect_index });
                }

                // VUID-vkCmdClearAttachments-pRects-00016
                // TODO: This check will always pass in secondary command buffers because of how
                // it's set in `with_level`.
                // It needs to be checked during `execute_commands` instead.
                if rect.offset[i] < render_pass_state.render_area_offset[i]
                    || rect.offset[i] + rect.extent[i]
                        > render_pass_state.render_area_offset[i]
                            + render_pass_state.render_area_extent[i]
                {
                    return Err(RenderPassError::RectOutOfBounds { rect_index });
                }
            }

            // VUID-vkCmdClearAttachments-layerCount-01934
            if rect.array_layers.is_empty() {
                return Err(RenderPassError::RectArrayLayersEmpty { rect_index });
            }

            // TODO: This can't be checked for secondary command buffers if they didn't provide
            // a framebuffer with their inheritance info.
            // It needs to be checked during `execute_commands` instead.
            if let Some(framebuffer) = &render_pass_state.framebuffer {
                // VUID-vkCmdClearAttachments-pRects-00017
                for range in framebuffer.attached_layers_ranges() {
                    if rect.array_layers.start < range.start || rect.array_layers.end > range.end {
                        return Err(RenderPassError::RectArrayLayersOutOfBounds { rect_index });
                    }
                }
            }

            // VUID-vkCmdClearAttachments-baseArrayLayer-00018
            if is_multiview && rect.array_layers != (0..1) {
                return Err(RenderPassError::MultiviewRectArrayLayersInvalid { rect_index });
            }
        }

        // VUID-vkCmdClearAttachments-commandBuffer-cmdpool
        debug_assert!(self.queue_family().supports_graphics());

        Ok(())
    }
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
            render_pass_begin_info: RenderPassBeginInfo,
            subpass_contents: SubpassContents,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "begin_render_pass"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.begin_render_pass(&self.render_pass_begin_info, self.subpass_contents);
            }
        }

        let &RenderPassBeginInfo {
            ref render_pass,
            ref framebuffer,
            render_area_offset,
            render_area_extent,
            ref clear_values,
            _ne: _,
        } = &render_pass_begin_info;

        let resources = render_pass
            .attachments()
            .iter()
            .enumerate()
            .map(|(num, desc)| {
                let image_view = &framebuffer.attachments()[num];

                (
                    format!("attachment {}", num).into(),
                    Resource::Image {
                        image: image_view.image(),
                        subresource_range: image_view.subresource_range().clone(),
                        memory: PipelineMemoryAccess {
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
                        start_layout: desc.initial_layout,
                        end_layout: desc.final_layout,
                    },
                )
            })
            .collect::<Vec<_>>();

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            render_pass_begin_info,
            subpass_contents,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

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
                "next_subpass"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.next_subpass(self.subpass_contents);
            }
        }

        self.commands.push(Box::new(Cmd { subpass_contents }));
    }

    /// Calls `vkCmdEndRenderPass` on the builder.
    #[inline]
    pub unsafe fn end_render_pass(&mut self) {
        struct Cmd;

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "end_render_pass"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.end_render_pass();
            }
        }

        self.commands.push(Box::new(Cmd));
        debug_assert!(self.latest_render_pass_enter.is_some());
        self.latest_render_pass_enter = None;
    }

    /// Calls `vkCmdClearAttachments` on the builder.
    ///
    /// Does nothing if the list of attachments or the list of rects is empty, as it would be a
    /// no-op and isn't a valid usage of the command anyway.
    pub unsafe fn clear_attachments(
        &mut self,
        attachments: impl IntoIterator<Item = ClearAttachment>,
        rects: impl IntoIterator<Item = ClearRect>,
    ) {
        struct Cmd {
            attachments: SmallVec<[ClearAttachment; 3]>,
            rects: SmallVec<[ClearRect; 4]>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "clear_attachments"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.clear_attachments(self.attachments.iter().copied(), self.rects.iter().cloned());
            }
        }
        let attachments: SmallVec<[_; 3]> = attachments.into_iter().collect();
        let rects: SmallVec<[_; 4]> = rects.into_iter().collect();

        self.commands.push(Box::new(Cmd { attachments, rects }));
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdBeginRenderPass` on the builder.
    #[inline]
    pub unsafe fn begin_render_pass(
        &mut self,
        render_pass_begin_info: &RenderPassBeginInfo,
        subpass_contents: SubpassContents,
    ) {
        let &RenderPassBeginInfo {
            ref render_pass,
            ref framebuffer,
            render_area_offset,
            render_area_extent,
            ref clear_values,
            _ne: _,
        } = render_pass_begin_info;

        let clear_values_vk: SmallVec<[_; 4]> = clear_values
            .into_iter()
            .copied()
            .map(|clear_value| clear_value.map(Into::into).unwrap_or_default())
            .collect();

        let render_pass_begin_info = ash::vk::RenderPassBeginInfo {
            render_pass: render_pass.internal_object(),
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
    pub unsafe fn clear_attachments<'a>(
        &mut self,
        attachments: impl IntoIterator<Item = ClearAttachment>,
        rects: impl IntoIterator<Item = ClearRect>,
    ) {
        let attachments: SmallVec<[_; 3]> = attachments.into_iter().map(|v| v.into()).collect();
        let rects: SmallVec<[_; 4]> = rects
            .into_iter()
            .map(|rect| ash::vk::ClearRect {
                rect: ash::vk::Rect2D {
                    offset: ash::vk::Offset2D {
                        x: rect.offset[0] as i32,
                        y: rect.offset[1] as i32,
                    },
                    extent: ash::vk::Extent2D {
                        width: rect.extent[0],
                        height: rect.extent[1],
                    },
                },
                base_array_layer: rect.array_layers.start,
                layer_count: rect.array_layers.end - rect.array_layers.start,
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

/// Error that can happen when recording a render pass command.
#[derive(Clone, Debug)]
pub enum RenderPassError {
    SyncCommandBufferBuilderError(SyncCommandBufferBuilderError),

    /// A framebuffer image did not have the required usage enabled.
    AttachmentImageMissingUsage {
        attachment_index: u32,
        usage: &'static str,
    },

    /// A clear value for a render pass attachment is missing.
    ClearValueMissing {
        attachment_index: u32,
    },

    /// A clear value provided for a render pass attachment is not compatible with the attachment's
    /// format.
    ClearValueNotCompatible {
        clear_value: ClearValue,
        attachment_index: u32,
        attachment_format: Format,
    },

    /// An attachment clear value specifies a `color_attachment` index that is not less than the
    /// number of color attachments in the subpass.
    ColorAttachmentIndexOutOfRange {
        color_attachment_index: u32,
        num_color_attachments: u32,
    },

    /// Operation forbidden inside a render pass.
    ForbiddenInsideRenderPass,

    /// Operation forbidden outside a render pass.
    ForbiddenOutsideRenderPass,

    /// Operation forbidden inside a render subpass with the specified contents.
    ForbiddenWithSubpassContents {
        subpass_contents: SubpassContents,
    },

    /// The framebuffer is not compatible with the render pass.
    FramebufferNotCompatible,

    /// The render pass uses multiview, and in a clear rectangle, `array_layers` was not `0..1`.
    MultiviewRectArrayLayersInvalid {
        rect_index: usize,
    },

    /// Tried to advance to the next subpass, but there are no subpasses remaining in the render
    /// pass.
    NoSubpassesRemaining {
        current_subpass: u32,
    },

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,

    /// A query is active that conflicts with the current operation.
    QueryIsActive,

    /// A clear rectangle's `array_layers` is empty.
    RectArrayLayersEmpty {
        rect_index: usize,
    },

    /// A clear rectangle's `array_layers` is outside the range of layers of the attachments.
    RectArrayLayersOutOfBounds {
        rect_index: usize,
    },

    /// A clear rectangle's `extent` is zero.
    RectExtentZero {
        rect_index: usize,
    },

    /// A clear rectangle's `offset` and `extent` are outside the render area of the render pass
    /// instance.
    RectOutOfBounds {
        rect_index: usize,
    },

    /// The render area's `offset` and `extent` are outside the extent of the framebuffer.
    RenderAreaOutOfBounds,

    /// Tried to end a render pass with subpasses still remaining in the render pass.
    SubpassesRemaining {
        current_subpass: u32,
        remaining_subpasses: u32,
    },
}

impl error::Error for RenderPassError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::SyncCommandBufferBuilderError(err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for RenderPassError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::SyncCommandBufferBuilderError(_) => write!(f, "a SyncCommandBufferBuilderError"),

            Self::AttachmentImageMissingUsage { attachment_index, usage } => write!(
                f,
                "the framebuffer image attached to attachment index {} did not have the required usage {} enabled",
                attachment_index, usage,
            ),
            Self::ClearValueMissing {
                attachment_index,
            } => write!(
                f,
                "a clear value for render pass attachment {} is missing",
                attachment_index,
            ),
            Self::ClearValueNotCompatible {
                clear_value,
                attachment_index,
                attachment_format,
            } => write!(
                f,
                "a clear value ({:?}) provided for render pass attachment {} is not compatible with the attachment's format ({:?})",
                clear_value, attachment_index, attachment_format,
            ),
            Self::ColorAttachmentIndexOutOfRange {
                color_attachment_index,
                num_color_attachments,
            } => write!(
                f,
                "an attachment clear value specifies a `color_attachment` index {} that is not less than the number of color attachments in the subpass ({})",
                color_attachment_index, num_color_attachments,
            ),
            Self::ForbiddenInsideRenderPass => {
                write!(f, "operation forbidden inside a render pass")
            }
            Self::ForbiddenOutsideRenderPass => {
                write!(f, "operation forbidden outside a render pass")
            }
            Self::ForbiddenWithSubpassContents { subpass_contents } => write!(
                f,
                "operation forbidden inside a render subpass with contents {:?}",
                subpass_contents,
            ),
            Self::FramebufferNotCompatible => write!(
                f,
                "the framebuffer is not compatible with the render pass",
            ),
            Self::MultiviewRectArrayLayersInvalid { rect_index } => write!(
                f,
                "the render pass uses multiview, and in clear rectangle index {}, `array_layers` was not `0..1`",
                rect_index,
            ),
            Self::NoSubpassesRemaining {
                current_subpass,
            } => write!(
                f,
                "tried to advance to the next subpass after subpass {}, but there are no subpasses remaining in the render pass",
                current_subpass,
            ),
            Self::NotSupportedByQueueFamily => {
                write!(f, "the queue family doesn't allow this operation")
            }
            Self::QueryIsActive => write!(
                f,
                "a query is active that conflicts with the current operation"
            ),
            Self::RectArrayLayersEmpty { rect_index } => write!(
                f,
                "clear rectangle index {} `array_layers` is empty",
                rect_index,
            ),
            Self::RectArrayLayersOutOfBounds { rect_index } => write!(
                f,
                "clear rectangle index {} `array_layers` is outside the range of layers of the attachments",
                rect_index,
            ),
            Self::RectExtentZero { rect_index } => write!(
                f,
                "clear rectangle index {} `extent` is zero",
                rect_index,
            ),
            Self::RectOutOfBounds { rect_index } => write!(
                f,
                "clear rectangle index {} `offset` and `extent` are outside the render area of the render pass instance",
                rect_index,
            ),
            Self::RenderAreaOutOfBounds => write!(
                f,
                "the render area's `offset` and `extent` are outside the extent of the framebuffer",
            ),
            Self::SubpassesRemaining {
                current_subpass,
                remaining_subpasses,
            } => write!(
                f,
                "tried to end a render pass at subpass {}, with {} subpasses still remaining in the render pass",
                current_subpass, remaining_subpasses,
            ),
        }
    }
}

impl From<SyncCommandBufferBuilderError> for RenderPassError {
    #[inline]
    fn from(err: SyncCommandBufferBuilderError) -> Self {
        Self::SyncCommandBufferBuilderError(err)
    }
}

/// Parameters to begin a new render pass.
#[derive(Clone, Debug)]
pub struct RenderPassBeginInfo {
    /// The render pass to begin.
    ///
    /// If this is not the render pass that `framebuffer` was created with, it must be compatible
    /// with that render pass.
    ///
    /// The default value is the render pass of `framebuffer`.
    pub render_pass: Arc<RenderPass>,

    /// The framebuffer to use for rendering.
    ///
    /// There is no default value.
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

    /// Provides, for each attachment in `render_pass` that has a load operation of
    /// [`LoadOp::Clear`], the clear values that should be used for the attachments in the
    /// framebuffer. There must be exactly [`framebuffer.attachments().len()`] elements provided,
    /// and each one must match the attachment format.
    ///
    /// To skip over an attachment whose load operation is something else, provide `None`.
    ///
    /// The default value is empty, which must be overridden if the framebuffer has attachments.
    pub clear_values: Vec<Option<ClearValue>>,

    pub _ne: crate::NonExhaustive,
}

impl RenderPassBeginInfo {
    #[inline]
    pub fn framebuffer(framebuffer: Arc<Framebuffer>) -> Self {
        let render_area_extent = framebuffer.extent();

        Self {
            render_pass: framebuffer.render_pass().clone(),
            framebuffer,
            render_area_offset: [0, 0],
            render_area_extent,
            clear_values: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Clear attachment type, used in [`clear_attachments`](crate::command_buffer::AutoCommandBufferBuilder::clear_attachments) command.
#[derive(Clone, Copy, Debug)]
pub enum ClearAttachment {
    /// Clear the color attachment at the specified index, with the specified clear value.
    Color {
        color_attachment: u32,
        clear_value: ClearColorValue,
    },

    /// Clear the depth attachment with the specified depth value.
    Depth(f32),

    /// Clear the stencil attachment with the specified stencil value.
    Stencil(u32),

    /// Clear the depth and stencil attachments with the specified depth and stencil values.
    DepthStencil((f32, u32)),
}

impl From<ClearAttachment> for ash::vk::ClearAttachment {
    fn from(v: ClearAttachment) -> Self {
        match v {
            ClearAttachment::Color {
                color_attachment,
                clear_value,
            } => ash::vk::ClearAttachment {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                color_attachment,
                clear_value: ash::vk::ClearValue {
                    color: clear_value.into(),
                },
            },
            ClearAttachment::Depth(depth) => ash::vk::ClearAttachment {
                aspect_mask: ash::vk::ImageAspectFlags::DEPTH,
                color_attachment: 0,
                clear_value: ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue { depth, stencil: 0 },
                },
            },
            ClearAttachment::Stencil(stencil) => ash::vk::ClearAttachment {
                aspect_mask: ash::vk::ImageAspectFlags::STENCIL,
                color_attachment: 0,
                clear_value: ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil,
                    },
                },
            },
            ClearAttachment::DepthStencil((depth, stencil)) => ash::vk::ClearAttachment {
                aspect_mask: ash::vk::ImageAspectFlags::DEPTH | ash::vk::ImageAspectFlags::STENCIL,
                color_attachment: 0,
                clear_value: ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue { depth, stencil },
                },
            },
        }
    }
}

/// Specifies the clear region for the [`clear_attachments`](crate::command_buffer::AutoCommandBufferBuilder::clear_attachments) command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClearRect {
    /// The rectangle offset.
    pub offset: [u32; 2],

    /// The width and height of the rectangle.
    pub extent: [u32; 2],

    /// The range of array layers to be cleared.
    pub array_layers: Range<u32>,
}
