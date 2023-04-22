// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    BeginRenderPassState, BeginRenderingState, ClearAttachment, ClearRect, CommandBufferBuilder,
    RenderPassBeginInfo, RenderPassError, RenderPassState, RenderPassStateAttachmentInfo,
    RenderPassStateAttachmentResolveInfo, RenderPassStateAttachments, RenderPassStateType,
    RenderingAttachmentInfo, RenderingAttachmentResolveInfo, RenderingInfo, ResourcesState,
};
use crate::{
    command_buffer::{
        allocator::CommandBufferAllocator, PrimaryCommandBuffer, ResourceInCommand, ResourceUseRef,
        SubpassContents,
    },
    device::{DeviceOwned, QueueFlags},
    format::{ClearColorValue, ClearValue, NumericType},
    image::{ImageAspects, ImageLayout, ImageUsage, SampleCount},
    pipeline::graphics::subpass::PipelineRenderingCreateInfo,
    render_pass::{AttachmentDescription, LoadOp, ResolveMode, SubpassDescription},
    sync::PipelineStageAccess,
    RequiresOneOf, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::cmp::min;

impl<A> CommandBufferBuilder<PrimaryCommandBuffer<A::Alloc>, A>
where
    A: CommandBufferAllocator,
{
    /// Begins a render pass using a render pass object and framebuffer.
    ///
    /// You must call this or `begin_rendering` before you can record draw commands.
    ///
    /// `contents` specifies what kinds of commands will be recorded in the render pass, either
    /// draw commands or executions of secondary command buffers.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn begin_render_pass(
        &mut self,
        render_pass_begin_info: RenderPassBeginInfo,
        contents: SubpassContents,
    ) -> Result<&mut Self, RenderPassError> {
        self.validate_begin_render_pass(&render_pass_begin_info, contents)?;

        unsafe { Ok(self.begin_render_pass_unchecked(render_pass_begin_info, contents)) }
    }

    fn validate_begin_render_pass(
        &self,
        render_pass_begin_info: &RenderPassBeginInfo,
        contents: SubpassContents,
    ) -> Result<(), RenderPassError> {
        let device = self.device();

        // VUID-VkSubpassBeginInfo-contents-parameter
        contents.validate_device(device)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBeginRenderPass2-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(RenderPassError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdBeginRenderPass2-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(RenderPassError::ForbiddenInsideRenderPass);
        }

        let RenderPassBeginInfo {
            render_pass,
            framebuffer,
            render_area_offset,
            render_area_extent,
            clear_values,
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
                        if !image_view.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index,
                                usage: "color_attachment",
                            });
                        }
                    }
                    ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                    | ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03096
                        if !image_view
                            .usage()
                            .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                        {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index,
                                usage: "depth_stencil_attachment",
                            });
                        }
                    }
                    ImageLayout::ShaderReadOnlyOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03097
                        if !image_view
                            .usage()
                            .intersects(ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT)
                        {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index,
                                usage: "sampled or input_attachment",
                            });
                        }
                    }
                    ImageLayout::TransferSrcOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03098
                        if !image_view.usage().intersects(ImageUsage::TRANSFER_SRC) {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index,
                                usage: "transfer_src",
                            });
                        }
                    }
                    ImageLayout::TransferDstOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03099
                        if !image_view.usage().intersects(ImageUsage::TRANSFER_DST) {
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
            let SubpassDescription {
                view_mask: _,
                input_attachments,
                color_attachments,
                resolve_attachments,
                depth_stencil_attachment,
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
                        if !image_view.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index: atch_ref.attachment,
                                usage: "color_attachment",
                            });
                        }
                    }
                    ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                    | ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03096
                        if !image_view
                            .usage()
                            .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                        {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index: atch_ref.attachment,
                                usage: "depth_stencil_attachment",
                            });
                        }
                    }
                    ImageLayout::ShaderReadOnlyOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03097
                        if !image_view
                            .usage()
                            .intersects(ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT)
                        {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index: atch_ref.attachment,
                                usage: "sampled or input_attachment",
                            });
                        }
                    }
                    ImageLayout::TransferSrcOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03098
                        if !image_view.usage().intersects(ImageUsage::TRANSFER_SRC) {
                            return Err(RenderPassError::AttachmentImageMissingUsage {
                                attachment_index: atch_ref.attachment,
                                usage: "transfer_src",
                            });
                        }
                    }
                    ImageLayout::TransferDstOptimal => {
                        // VUID-vkCmdBeginRenderPass2-initialLayout-03099
                        if !image_view.usage().intersects(ImageUsage::TRANSFER_DST) {
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
                    let need_depth = attachment_aspects.intersects(ImageAspects::DEPTH)
                        && attachment_desc.load_op == LoadOp::Clear;
                    let need_stencil = attachment_aspects.intersects(ImageAspects::STENCIL)
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
        // TODO:

        // VUID-vkCmdBeginRenderPass2-srcStageMask-06453
        // TODO:

        // VUID-vkCmdBeginRenderPass2-dstStageMask-06454
        // TODO:

        // VUID-vkCmdBeginRenderPass2-framebuffer-02533
        // For any attachment in framebuffer that is used by renderPass and is bound to memory locations that are also bound to another attachment used by renderPass, and if at least one of those uses causes either
        // attachment to be written to, both attachments must have had the VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT set

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_render_pass_unchecked(
        &mut self,
        render_pass_begin_info: RenderPassBeginInfo,
        contents: SubpassContents,
    ) -> &mut Self {
        let RenderPassBeginInfo {
            render_pass,
            framebuffer,
            render_area_offset,
            render_area_extent,
            clear_values,
            _ne: _,
        } = render_pass_begin_info;

        let clear_values_vk: SmallVec<[_; 4]> = clear_values
            .iter()
            .copied()
            .map(|clear_value| clear_value.map(Into::into).unwrap_or_default())
            .collect();

        let render_pass_begin_info = ash::vk::RenderPassBeginInfo {
            render_pass: render_pass.handle(),
            framebuffer: framebuffer.handle(),
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
            contents: contents.into(),
            ..Default::default()
        };

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_2
            || self.device().enabled_extensions().khr_create_renderpass2
        {
            if self.device().api_version() >= Version::V1_2 {
                (fns.v1_2.cmd_begin_render_pass2)(
                    self.handle(),
                    &render_pass_begin_info,
                    &subpass_begin_info,
                );
            } else {
                (fns.khr_create_renderpass2.cmd_begin_render_pass2_khr)(
                    self.handle(),
                    &render_pass_begin_info,
                    &subpass_begin_info,
                );
            }
        } else {
            debug_assert!(subpass_begin_info.p_next.is_null());

            (fns.v1_0.cmd_begin_render_pass)(
                self.handle(),
                &render_pass_begin_info,
                subpass_begin_info.contents,
            );
        }

        let command_index = self.next_command_index;
        let command_name = "begin_render_pass";

        // Advance to first subpass
        {
            let subpass = render_pass.clone().first_subpass();
            self.builder_state.render_pass = Some(RenderPassState {
                contents,
                render_area_offset,
                render_area_extent,

                rendering_info: PipelineRenderingCreateInfo::from_subpass(&subpass),
                attachments: Some(RenderPassStateAttachments::from_subpass(
                    &subpass,
                    &framebuffer,
                )),

                render_pass: BeginRenderPassState {
                    subpass,
                    framebuffer: Some(framebuffer.clone()),
                }
                .into(),
            });
        }

        // Start of first subpass
        {
            // TODO: Apply barriers and layout transitions

            let render_pass_state = self.builder_state.render_pass.as_ref().unwrap();
            record_subpass_attachments_load(
                &mut self.resources_usage_state,
                command_index,
                command_name,
                render_pass_state,
            );
        }

        self.resources.push(Box::new(render_pass));
        self.resources.push(Box::new(framebuffer));

        self.next_command_index += 1;
        self
    }

    /// Advances to the next subpass of the render pass previously begun with `begin_render_pass`.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn next_subpass(
        &mut self,
        contents: SubpassContents,
    ) -> Result<&mut Self, RenderPassError> {
        self.validate_next_subpass(contents)?;

        unsafe { Ok(self.next_subpass_unchecked(contents)) }
    }

    fn validate_next_subpass(&self, contents: SubpassContents) -> Result<(), RenderPassError> {
        let device = self.device();

        // VUID-VkSubpassBeginInfo-contents-parameter
        contents.validate_device(device)?;

        // VUID-vkCmdNextSubpass2-renderpass
        let render_pass_state = self
            .builder_state
            .render_pass
            .as_ref()
            .ok_or(RenderPassError::ForbiddenOutsideRenderPass)?;

        let begin_render_pass_state = match &render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(state) => state,
            RenderPassStateType::BeginRendering(_) => {
                return Err(RenderPassError::ForbiddenWithBeginRendering)
            }
        };

        // VUID-vkCmdNextSubpass2-None-03102
        if begin_render_pass_state.subpass.is_last_subpass() {
            return Err(RenderPassError::NoSubpassesRemaining {
                current_subpass: begin_render_pass_state.subpass.index(),
            });
        }

        // VUID?
        if self
            .builder_state
            .queries
            .values()
            .any(|state| state.in_subpass)
        {
            return Err(RenderPassError::QueryIsActive);
        }

        // VUID-vkCmdNextSubpass2-commandBuffer-cmdpool
        debug_assert!(self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS));

        // VUID-vkCmdNextSubpass2-bufferlevel
        // Ensured by the type of the impl block

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn next_subpass_unchecked(&mut self, contents: SubpassContents) -> &mut Self {
        let subpass_begin_info = ash::vk::SubpassBeginInfo {
            contents: contents.into(),
            ..Default::default()
        };

        let subpass_end_info = ash::vk::SubpassEndInfo::default();

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_2
            || self.device().enabled_extensions().khr_create_renderpass2
        {
            if self.device().api_version() >= Version::V1_2 {
                (fns.v1_2.cmd_next_subpass2)(self.handle(), &subpass_begin_info, &subpass_end_info);
            } else {
                (fns.khr_create_renderpass2.cmd_next_subpass2_khr)(
                    self.handle(),
                    &subpass_begin_info,
                    &subpass_end_info,
                );
            }
        } else {
            (fns.v1_0.cmd_next_subpass)(self.handle(), subpass_begin_info.contents);
        }

        let command_index = self.next_command_index;
        let command_name = "next_subpass";

        // End of previous subpass
        {
            let render_pass_state = self.builder_state.render_pass.as_ref().unwrap();
            record_subpass_attachments_resolve(
                &mut self.resources_usage_state,
                command_index,
                command_name,
                render_pass_state,
            );
            record_subpass_attachments_store(
                &mut self.resources_usage_state,
                command_index,
                command_name,
                render_pass_state,
            );
        }

        // Advance to next subpass
        {
            let render_pass_state = self.builder_state.render_pass.as_mut().unwrap();
            let begin_render_pass_state = match &mut render_pass_state.render_pass {
                RenderPassStateType::BeginRenderPass(x) => x,
                _ => unreachable!(),
            };

            begin_render_pass_state.subpass.next_subpass();
            render_pass_state.contents = contents;
            render_pass_state.rendering_info =
                PipelineRenderingCreateInfo::from_subpass(&begin_render_pass_state.subpass);
            render_pass_state.attachments = Some(RenderPassStateAttachments::from_subpass(
                &begin_render_pass_state.subpass,
                begin_render_pass_state.framebuffer.as_ref().unwrap(),
            ));

            if render_pass_state.rendering_info.view_mask != 0 {
                // When multiview is enabled, at the beginning of each subpass, all
                // non-render pass state is undefined.
                self.builder_state = Default::default();
            }
        }

        // Start of next subpass
        {
            // TODO: Apply barriers and layout transitions

            let render_pass_state = self.builder_state.render_pass.as_ref().unwrap();
            record_subpass_attachments_load(
                &mut self.resources_usage_state,
                command_index,
                command_name,
                render_pass_state,
            );
        }

        self.next_command_index += 1;
        self
    }

    /// Ends the render pass previously begun with `begin_render_pass`.
    ///
    /// This must be called after you went through all the subpasses.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn end_render_pass(&mut self) -> Result<&mut Self, RenderPassError> {
        self.validate_end_render_pass()?;

        unsafe { Ok(self.end_render_pass_unchecked()) }
    }

    fn validate_end_render_pass(&self) -> Result<(), RenderPassError> {
        // VUID-vkCmdEndRenderPass2-renderpass
        let render_pass_state = self
            .builder_state
            .render_pass
            .as_ref()
            .ok_or(RenderPassError::ForbiddenOutsideRenderPass)?;

        let begin_render_pass_state = match &render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(state) => state,
            RenderPassStateType::BeginRendering(_) => {
                return Err(RenderPassError::ForbiddenWithBeginRendering)
            }
        };

        // VUID-vkCmdEndRenderPass2-None-03103
        if !begin_render_pass_state.subpass.is_last_subpass() {
            return Err(RenderPassError::SubpassesRemaining {
                current_subpass: begin_render_pass_state.subpass.index(),
                remaining_subpasses: begin_render_pass_state
                    .subpass
                    .render_pass()
                    .subpasses()
                    .len() as u32
                    - begin_render_pass_state.subpass.index(),
            });
        }

        // VUID?
        if self
            .builder_state
            .queries
            .values()
            .any(|state| state.in_subpass)
        {
            return Err(RenderPassError::QueryIsActive);
        }

        // VUID-vkCmdEndRenderPass2-commandBuffer-cmdpool
        debug_assert!(self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS));

        // VUID-vkCmdEndRenderPass2-bufferlevel
        // Ensured by the type of the impl block

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_render_pass_unchecked(&mut self) -> &mut Self {
        let subpass_end_info = ash::vk::SubpassEndInfo::default();

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_2
            || self.device().enabled_extensions().khr_create_renderpass2
        {
            if self.device().api_version() >= Version::V1_2 {
                (fns.v1_2.cmd_end_render_pass2)(self.handle(), &subpass_end_info);
            } else {
                (fns.khr_create_renderpass2.cmd_end_render_pass2_khr)(
                    self.handle(),
                    &subpass_end_info,
                );
            }
        } else {
            (fns.v1_0.cmd_end_render_pass)(self.handle());
        }

        let command_index = self.next_command_index;
        let command_name = "end_render_pass";

        // End of last subpass
        {
            let render_pass_state = self.builder_state.render_pass.as_ref().unwrap();
            record_subpass_attachments_resolve(
                &mut self.resources_usage_state,
                command_index,
                command_name,
                render_pass_state,
            );
            record_subpass_attachments_store(
                &mut self.resources_usage_state,
                command_index,
                command_name,
                render_pass_state,
            );
        }

        // TODO: Apply barriers and layout transitions

        self.builder_state.render_pass = None;

        self.next_command_index += 1;
        self
    }
}

impl<L, A> CommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Begins a render pass without a render pass object or framebuffer.
    ///
    /// You must call this or `begin_render_pass` before you can record draw commands.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn begin_rendering(
        &mut self,
        mut rendering_info: RenderingInfo,
    ) -> Result<&mut Self, RenderPassError> {
        rendering_info.set_extent_layers()?;
        self.validate_begin_rendering(&rendering_info)?;

        unsafe { Ok(self.begin_rendering_unchecked(rendering_info)) }
    }

    fn validate_begin_rendering(
        &self,
        rendering_info: &RenderingInfo,
    ) -> Result<(), RenderPassError> {
        let device = self.device();
        let properties = device.physical_device().properties();

        // VUID-vkCmdBeginRendering-dynamicRendering-06446
        if !device.enabled_features().dynamic_rendering {
            return Err(RenderPassError::RequirementNotMet {
                required_for: "`CommandBufferBuilder::begin_rendering`",
                requires_one_of: RequiresOneOf {
                    features: &["dynamic_rendering"],
                    ..Default::default()
                },
            });
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBeginRendering-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(RenderPassError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdBeginRendering-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(RenderPassError::ForbiddenInsideRenderPass);
        }

        let &RenderingInfo {
            render_area_offset,
            render_area_extent,
            layer_count,
            view_mask,
            ref color_attachments,
            ref depth_attachment,
            ref stencil_attachment,
            contents,
            _ne: _,
        } = rendering_info;

        // VUID-VkRenderingInfo-flags-parameter
        contents.validate_device(device)?;

        // VUID-vkCmdBeginRendering-commandBuffer-06068
        if self.inheritance_info.is_some() && contents == SubpassContents::SecondaryCommandBuffers {
            return Err(RenderPassError::ContentsForbiddenInSecondaryCommandBuffer);
        }

        // No VUID, but for sanity it makes sense to treat this the same as in framebuffers.
        if view_mask != 0 && layer_count != 1 {
            return Err(RenderPassError::MultiviewLayersInvalid);
        }

        // VUID-VkRenderingInfo-multiview-06127
        if view_mask != 0 && !device.enabled_features().multiview {
            return Err(RenderPassError::RequirementNotMet {
                required_for: "`rendering_info.viewmask` is not `0`",
                requires_one_of: RequiresOneOf {
                    features: &["multiview"],
                    ..Default::default()
                },
            });
        }

        let view_count = u32::BITS - view_mask.leading_zeros();

        // VUID-VkRenderingInfo-viewMask-06128
        if view_count > properties.max_multiview_view_count.unwrap_or(0) {
            return Err(RenderPassError::MaxMultiviewViewCountExceeded {
                view_count,
                max: properties.max_multiview_view_count.unwrap_or(0),
            });
        }

        let mut samples = None;

        // VUID-VkRenderingInfo-colorAttachmentCount-06106
        if color_attachments.len() > properties.max_color_attachments as usize {
            return Err(RenderPassError::MaxColorAttachmentsExceeded {
                color_attachment_count: color_attachments.len() as u32,
                max: properties.max_color_attachments,
            });
        }

        for (attachment_index, attachment_info) in
            color_attachments
                .iter()
                .enumerate()
                .filter_map(|(index, attachment_info)| {
                    attachment_info
                        .as_ref()
                        .map(|attachment_info| (index, attachment_info))
                })
        {
            let attachment_index = attachment_index as u32;
            let RenderingAttachmentInfo {
                image_view,
                image_layout,
                resolve_info,
                load_op,
                store_op,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            // VUID-VkRenderingAttachmentInfo-imageLayout-parameter
            image_layout.validate_device(device)?;

            // VUID-VkRenderingAttachmentInfo-loadOp-parameter
            load_op.validate_device(device)?;

            // VUID-VkRenderingAttachmentInfo-storeOp-parameter
            store_op.validate_device(device)?;

            // VUID-VkRenderingInfo-colorAttachmentCount-06087
            if !image_view.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                return Err(RenderPassError::ColorAttachmentMissingUsage { attachment_index });
            }

            let image = image_view.image();
            let image_extent = image.dimensions().width_height_depth();

            for i in 0..2 {
                // VUID-VkRenderingInfo-pNext-06079
                // VUID-VkRenderingInfo-pNext-06080
                if render_area_offset[i] + render_area_extent[i] > image_extent[i] {
                    return Err(RenderPassError::RenderAreaOutOfBounds);
                }
            }

            // VUID-VkRenderingInfo-imageView-06070
            match samples {
                Some(samples) if samples == image.samples() => (),
                Some(_) => {
                    return Err(RenderPassError::ColorAttachmentSamplesMismatch {
                        attachment_index,
                    });
                }
                None => samples = Some(image.samples()),
            }

            // VUID-VkRenderingAttachmentInfo-imageView-06135
            // VUID-VkRenderingAttachmentInfo-imageView-06145
            // VUID-VkRenderingInfo-colorAttachmentCount-06090
            // VUID-VkRenderingInfo-colorAttachmentCount-06096
            if matches!(
                image_layout,
                ImageLayout::Undefined
                    | ImageLayout::ShaderReadOnlyOptimal
                    | ImageLayout::TransferSrcOptimal
                    | ImageLayout::TransferDstOptimal
                    | ImageLayout::Preinitialized
                    | ImageLayout::PresentSrc
                    | ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal
                    | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
            ) {
                return Err(RenderPassError::ColorAttachmentLayoutInvalid { attachment_index });
            }

            if let Some(resolve_info) = resolve_info {
                let &RenderingAttachmentResolveInfo {
                    mode,
                    image_view: ref resolve_image_view,
                    image_layout: resolve_image_layout,
                } = resolve_info;

                // VUID-VkRenderingAttachmentInfo-resolveImageLayout-parameter
                resolve_image_layout.validate_device(device)?;

                // VUID-VkRenderingAttachmentInfo-resolveMode-parameter
                mode.validate_device(device)?;

                let resolve_image = resolve_image_view.image();

                match image_view.format().unwrap().type_color() {
                    Some(
                        NumericType::SFLOAT
                        | NumericType::UFLOAT
                        | NumericType::SNORM
                        | NumericType::UNORM
                        | NumericType::SSCALED
                        | NumericType::USCALED
                        | NumericType::SRGB,
                    ) => {
                        // VUID-VkRenderingAttachmentInfo-imageView-06129
                        if mode != ResolveMode::Average {
                            return Err(RenderPassError::ColorAttachmentResolveModeNotSupported {
                                attachment_index,
                            });
                        }
                    }
                    Some(NumericType::SINT | NumericType::UINT) => {
                        // VUID-VkRenderingAttachmentInfo-imageView-06130
                        if mode != ResolveMode::SampleZero {
                            return Err(RenderPassError::ColorAttachmentResolveModeNotSupported {
                                attachment_index,
                            });
                        }
                    }
                    None => (),
                }

                // VUID-VkRenderingAttachmentInfo-imageView-06132
                if image.samples() == SampleCount::Sample1 {
                    return Err(RenderPassError::ColorAttachmentWithResolveNotMultisampled {
                        attachment_index,
                    });
                }

                // VUID-VkRenderingAttachmentInfo-imageView-06133
                if resolve_image.samples() != SampleCount::Sample1 {
                    return Err(RenderPassError::ColorAttachmentResolveMultisampled {
                        attachment_index,
                    });
                }

                // VUID-VkRenderingAttachmentInfo-imageView-06134
                if image_view.format() != resolve_image_view.format() {
                    return Err(RenderPassError::ColorAttachmentResolveFormatMismatch {
                        attachment_index,
                    });
                }

                // VUID-VkRenderingAttachmentInfo-imageView-06136
                // VUID-VkRenderingAttachmentInfo-imageView-06146
                // VUID-VkRenderingInfo-colorAttachmentCount-06091
                // VUID-VkRenderingInfo-colorAttachmentCount-06097
                if matches!(
                    resolve_image_layout,
                    ImageLayout::Undefined
                        | ImageLayout::ShaderReadOnlyOptimal
                        | ImageLayout::TransferSrcOptimal
                        | ImageLayout::TransferDstOptimal
                        | ImageLayout::Preinitialized
                        | ImageLayout::PresentSrc
                        | ImageLayout::DepthStencilAttachmentOptimal
                        | ImageLayout::DepthStencilReadOnlyOptimal
                        | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                        | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                ) {
                    return Err(RenderPassError::ColorAttachmentResolveLayoutInvalid {
                        attachment_index,
                    });
                }
            }
        }

        if let Some(attachment_info) = depth_attachment {
            let RenderingAttachmentInfo {
                image_view,
                image_layout,
                resolve_info,
                load_op,
                store_op,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            // VUID-VkRenderingAttachmentInfo-imageLayout-parameter
            image_layout.validate_device(device)?;

            // VUID-VkRenderingAttachmentInfo-loadOp-parameter
            load_op.validate_device(device)?;

            // VUID-VkRenderingAttachmentInfo-storeOp-parameter
            store_op.validate_device(device)?;

            let image_aspects = image_view.format().unwrap().aspects();

            // VUID-VkRenderingInfo-pDepthAttachment-06547
            if !image_aspects.intersects(ImageAspects::DEPTH) {
                return Err(RenderPassError::DepthAttachmentFormatUsageNotSupported);
            }

            // VUID-VkRenderingInfo-pDepthAttachment-06088
            if !image_view
                .usage()
                .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(RenderPassError::DepthAttachmentMissingUsage);
            }

            let image = image_view.image();
            let image_extent = image.dimensions().width_height_depth();

            for i in 0..2 {
                // VUID-VkRenderingInfo-pNext-06079
                // VUID-VkRenderingInfo-pNext-06080
                if render_area_offset[i] + render_area_extent[i] > image_extent[i] {
                    return Err(RenderPassError::RenderAreaOutOfBounds);
                }
            }

            // VUID-VkRenderingInfo-imageView-06070
            match samples {
                Some(samples) if samples == image.samples() => (),
                Some(_) => {
                    return Err(RenderPassError::DepthAttachmentSamplesMismatch);
                }
                None => samples = Some(image.samples()),
            }

            // VUID-VkRenderingAttachmentInfo-imageView-06135
            // VUID-VkRenderingAttachmentInfo-imageView-06145
            // VUID-VkRenderingInfo-pDepthAttachment-06092
            if matches!(
                image_layout,
                ImageLayout::Undefined
                    | ImageLayout::ShaderReadOnlyOptimal
                    | ImageLayout::TransferSrcOptimal
                    | ImageLayout::TransferDstOptimal
                    | ImageLayout::Preinitialized
                    | ImageLayout::PresentSrc
                    | ImageLayout::ColorAttachmentOptimal
            ) {
                return Err(RenderPassError::DepthAttachmentLayoutInvalid);
            }

            if let Some(resolve_info) = resolve_info {
                let &RenderingAttachmentResolveInfo {
                    mode,
                    image_view: ref resolve_image_view,
                    image_layout: resolve_image_layout,
                } = resolve_info;

                // VUID-VkRenderingAttachmentInfo-resolveImageLayout-parameter
                resolve_image_layout.validate_device(device)?;

                // VUID-VkRenderingAttachmentInfo-resolveMode-parameter
                mode.validate_device(device)?;

                // VUID-VkRenderingInfo-pDepthAttachment-06102
                if !properties
                    .supported_depth_resolve_modes
                    .map_or(false, |modes| modes.contains_enum(mode))
                {
                    return Err(RenderPassError::DepthAttachmentResolveModeNotSupported);
                }

                let resolve_image = resolve_image_view.image();

                // VUID-VkRenderingAttachmentInfo-imageView-06132
                if image.samples() == SampleCount::Sample1 {
                    return Err(RenderPassError::DepthAttachmentWithResolveNotMultisampled);
                }

                // VUID-VkRenderingAttachmentInfo-imageView-06133
                if resolve_image.samples() != SampleCount::Sample1 {
                    return Err(RenderPassError::DepthAttachmentResolveMultisampled);
                }

                // VUID-VkRenderingAttachmentInfo-imageView-06134
                if image_view.format() != resolve_image_view.format() {
                    return Err(RenderPassError::DepthAttachmentResolveFormatMismatch);
                }

                // VUID-VkRenderingAttachmentInfo-imageView-06136
                // VUID-VkRenderingAttachmentInfo-imageView-06146
                // VUID-VkRenderingInfo-pDepthAttachment-06093
                // VUID-VkRenderingInfo-pDepthAttachment-06098
                if matches!(
                    resolve_image_layout,
                    ImageLayout::Undefined
                        | ImageLayout::DepthStencilReadOnlyOptimal
                        | ImageLayout::ShaderReadOnlyOptimal
                        | ImageLayout::TransferSrcOptimal
                        | ImageLayout::TransferDstOptimal
                        | ImageLayout::Preinitialized
                        | ImageLayout::PresentSrc
                        | ImageLayout::ColorAttachmentOptimal
                        | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                ) {
                    return Err(RenderPassError::DepthAttachmentResolveLayoutInvalid);
                }
            }
        }

        if let Some(attachment_info) = stencil_attachment {
            let RenderingAttachmentInfo {
                image_view,
                image_layout,
                resolve_info,
                load_op,
                store_op,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            // VUID-VkRenderingAttachmentInfo-imageLayout-parameter
            image_layout.validate_device(device)?;

            // VUID-VkRenderingAttachmentInfo-loadOp-parameter
            load_op.validate_device(device)?;

            // VUID-VkRenderingAttachmentInfo-storeOp-parameter
            store_op.validate_device(device)?;

            let image_aspects = image_view.format().unwrap().aspects();

            // VUID-VkRenderingInfo-pStencilAttachment-06548
            if !image_aspects.intersects(ImageAspects::STENCIL) {
                return Err(RenderPassError::StencilAttachmentFormatUsageNotSupported);
            }

            // VUID-VkRenderingInfo-pStencilAttachment-06089
            if !image_view
                .usage()
                .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(RenderPassError::StencilAttachmentMissingUsage);
            }

            let image = image_view.image();
            let image_extent = image.dimensions().width_height_depth();

            for i in 0..2 {
                // VUID-VkRenderingInfo-pNext-06079
                // VUID-VkRenderingInfo-pNext-06080
                if render_area_offset[i] + render_area_extent[i] > image_extent[i] {
                    return Err(RenderPassError::RenderAreaOutOfBounds);
                }
            }

            // VUID-VkRenderingInfo-imageView-06070
            match samples {
                Some(samples) if samples == image.samples() => (),
                Some(_) => {
                    return Err(RenderPassError::StencilAttachmentSamplesMismatch);
                }
                None => (),
            }

            // VUID-VkRenderingAttachmentInfo-imageView-06135
            // VUID-VkRenderingAttachmentInfo-imageView-06145
            // VUID-VkRenderingInfo-pStencilAttachment-06094
            if matches!(
                image_layout,
                ImageLayout::Undefined
                    | ImageLayout::ShaderReadOnlyOptimal
                    | ImageLayout::TransferSrcOptimal
                    | ImageLayout::TransferDstOptimal
                    | ImageLayout::Preinitialized
                    | ImageLayout::PresentSrc
                    | ImageLayout::ColorAttachmentOptimal
            ) {
                return Err(RenderPassError::StencilAttachmentLayoutInvalid);
            }

            if let Some(resolve_info) = resolve_info {
                let &RenderingAttachmentResolveInfo {
                    mode,
                    image_view: ref resolve_image_view,
                    image_layout: resolve_image_layout,
                } = resolve_info;

                // VUID-VkRenderingAttachmentInfo-resolveImageLayout-parameter
                resolve_image_layout.validate_device(device)?;

                // VUID-VkRenderingAttachmentInfo-resolveMode-parameter
                mode.validate_device(device)?;

                // VUID-VkRenderingInfo-pStencilAttachment-06103
                if !properties
                    .supported_stencil_resolve_modes
                    .map_or(false, |modes| modes.contains_enum(mode))
                {
                    return Err(RenderPassError::StencilAttachmentResolveModeNotSupported);
                }

                let resolve_image = resolve_image_view.image();

                // VUID-VkRenderingAttachmentInfo-imageView-06132
                if image.samples() == SampleCount::Sample1 {
                    return Err(RenderPassError::StencilAttachmentWithResolveNotMultisampled);
                }

                // VUID-VkRenderingAttachmentInfo-imageView-06133
                if resolve_image.samples() != SampleCount::Sample1 {
                    return Err(RenderPassError::StencilAttachmentResolveMultisampled);
                }

                // VUID-VkRenderingAttachmentInfo-imageView-06134
                if image_view.format() != resolve_image_view.format() {
                    return Err(RenderPassError::StencilAttachmentResolveFormatMismatch);
                }

                // VUID-VkRenderingAttachmentInfo-imageView-06136
                // VUID-VkRenderingAttachmentInfo-imageView-06146
                // VUID-VkRenderingInfo-pStencilAttachment-06095
                // VUID-VkRenderingInfo-pStencilAttachment-06099
                if matches!(
                    resolve_image_layout,
                    ImageLayout::Undefined
                        | ImageLayout::DepthStencilReadOnlyOptimal
                        | ImageLayout::ShaderReadOnlyOptimal
                        | ImageLayout::TransferSrcOptimal
                        | ImageLayout::TransferDstOptimal
                        | ImageLayout::Preinitialized
                        | ImageLayout::PresentSrc
                        | ImageLayout::ColorAttachmentOptimal
                        | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                ) {
                    return Err(RenderPassError::StencilAttachmentResolveLayoutInvalid);
                }
            }
        }

        if let (Some(depth_attachment_info), Some(stencil_attachment_info)) =
            (depth_attachment, stencil_attachment)
        {
            // VUID-VkRenderingInfo-pDepthAttachment-06085
            if &depth_attachment_info.image_view != &stencil_attachment_info.image_view {
                return Err(RenderPassError::DepthStencilAttachmentImageViewMismatch);
            }

            match (
                &depth_attachment_info.resolve_info,
                &stencil_attachment_info.resolve_info,
            ) {
                (None, None) => (),
                (None, Some(_)) | (Some(_), None) => {
                    // VUID-VkRenderingInfo-pDepthAttachment-06104
                    if !properties.independent_resolve_none.unwrap_or(false) {
                        return Err(
                            RenderPassError::DepthStencilAttachmentResolveModesNotSupported,
                        );
                    }
                }
                (Some(depth_resolve_info), Some(stencil_resolve_info)) => {
                    // VUID-VkRenderingInfo-pDepthAttachment-06105
                    if !properties.independent_resolve.unwrap_or(false)
                        && depth_resolve_info.mode != stencil_resolve_info.mode
                    {
                        return Err(
                            RenderPassError::DepthStencilAttachmentResolveModesNotSupported,
                        );
                    }

                    // VUID-VkRenderingInfo-pDepthAttachment-06086
                    if &depth_resolve_info.image_view != &stencil_resolve_info.image_view {
                        return Err(
                            RenderPassError::DepthStencilAttachmentResolveImageViewMismatch,
                        );
                    }
                }
            }
        }

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_rendering_unchecked(&mut self, rendering_info: RenderingInfo) -> &mut Self {
        {
            let &RenderingInfo {
                render_area_offset,
                render_area_extent,
                layer_count,
                view_mask,
                ref color_attachments,
                ref depth_attachment,
                ref stencil_attachment,
                contents,
                _ne,
            } = &rendering_info;

            let map_attachment_info = |attachment_info: &Option<_>| {
                if let Some(attachment_info) = attachment_info {
                    let &RenderingAttachmentInfo {
                        ref image_view,
                        image_layout,
                        resolve_info: ref resolve,
                        load_op,
                        store_op,
                        clear_value,
                        _ne: _,
                    } = attachment_info;

                    let (resolve_mode, resolve_image_view, resolve_image_layout) =
                        if let Some(resolve) = resolve {
                            let &RenderingAttachmentResolveInfo {
                                mode,
                                ref image_view,
                                image_layout,
                            } = resolve;

                            (mode.into(), image_view.handle(), image_layout.into())
                        } else {
                            (
                                ash::vk::ResolveModeFlags::NONE,
                                Default::default(),
                                Default::default(),
                            )
                        };

                    ash::vk::RenderingAttachmentInfo {
                        image_view: image_view.handle(),
                        image_layout: image_layout.into(),
                        resolve_mode,
                        resolve_image_view,
                        resolve_image_layout,
                        load_op: load_op.into(),
                        store_op: store_op.into(),
                        clear_value: clear_value.map_or_else(Default::default, Into::into),
                        ..Default::default()
                    }
                } else {
                    ash::vk::RenderingAttachmentInfo {
                        image_view: ash::vk::ImageView::null(),
                        ..Default::default()
                    }
                }
            };

            let color_attachments_vk: SmallVec<[_; 2]> =
                color_attachments.iter().map(map_attachment_info).collect();
            let depth_attachment_vk = map_attachment_info(depth_attachment);
            let stencil_attachment_vk = map_attachment_info(stencil_attachment);

            let rendering_info_vk = ash::vk::RenderingInfo {
                flags: contents.into(),
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
                layer_count,
                view_mask,
                color_attachment_count: color_attachments_vk.len() as u32,
                p_color_attachments: color_attachments_vk.as_ptr(),
                p_depth_attachment: &depth_attachment_vk,
                p_stencil_attachment: &stencil_attachment_vk,
                ..Default::default()
            };

            let fns = self.device().fns();

            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_begin_rendering)(self.handle(), &rendering_info_vk);
            } else {
                debug_assert!(self.device().enabled_extensions().khr_dynamic_rendering);
                (fns.khr_dynamic_rendering.cmd_begin_rendering_khr)(
                    self.handle(),
                    &rendering_info_vk,
                );
            }

            self.builder_state.render_pass = Some(RenderPassState {
                contents,
                render_area_offset,
                render_area_extent,

                rendering_info: PipelineRenderingCreateInfo::from_rendering_info(&rendering_info),
                attachments: Some(RenderPassStateAttachments::from_rendering_info(
                    &rendering_info,
                )),

                render_pass: BeginRenderingState {
                    pipeline_used: false,
                }
                .into(),
            });
        }

        let RenderingInfo {
            color_attachments,
            depth_attachment,
            stencil_attachment,
            ..
        } = rendering_info;

        for attachment_info in color_attachments.into_iter().flatten() {
            let RenderingAttachmentInfo {
                image_view,
                image_layout: _,
                resolve_info,
                load_op: _,
                store_op: _,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            self.resources.push(Box::new(image_view));

            if let Some(resolve_info) = resolve_info {
                let RenderingAttachmentResolveInfo {
                    mode: _,
                    image_view,
                    image_layout: _,
                } = resolve_info;

                self.resources.push(Box::new(image_view));
            }
        }

        if let Some(attachment_info) = depth_attachment {
            let RenderingAttachmentInfo {
                image_view,
                image_layout: _,
                resolve_info,
                load_op: _,
                store_op: _,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            self.resources.push(Box::new(image_view));

            if let Some(resolve_info) = resolve_info {
                let RenderingAttachmentResolveInfo {
                    mode: _,
                    image_view,
                    image_layout: _,
                } = resolve_info;

                self.resources.push(Box::new(image_view));
            }
        }

        if let Some(attachment_info) = stencil_attachment {
            let RenderingAttachmentInfo {
                image_view,
                image_layout: _,
                resolve_info,
                load_op: _,
                store_op: _,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            self.resources.push(Box::new(image_view));

            if let Some(resolve_info) = resolve_info {
                let RenderingAttachmentResolveInfo {
                    mode: _,
                    image_view,
                    image_layout: _,
                } = resolve_info;

                self.resources.push(Box::new(image_view));
            }
        }

        self.next_command_index += 1;
        self
    }

    /// Ends the render pass previously begun with `begin_rendering`.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn end_rendering(&mut self) -> Result<&mut Self, RenderPassError> {
        self.validate_end_rendering()?;

        unsafe { Ok(self.end_rendering_unchecked()) }
    }

    fn validate_end_rendering(&self) -> Result<(), RenderPassError> {
        // VUID-vkCmdEndRendering-renderpass
        let render_pass_state = self
            .builder_state
            .render_pass
            .as_ref()
            .ok_or(RenderPassError::ForbiddenOutsideRenderPass)?;

        // VUID?
        if self.inheritance_info.is_some() {
            return Err(RenderPassError::ForbiddenWithInheritedRenderPass);
        }

        // VUID-vkCmdEndRendering-None-06161
        // VUID-vkCmdEndRendering-commandBuffer-06162
        match &render_pass_state.render_pass {
            RenderPassStateType::BeginRenderPass(_) => {
                return Err(RenderPassError::ForbiddenWithBeginRenderPass)
            }
            RenderPassStateType::BeginRendering(_) => (),
        }

        // VUID-vkCmdEndRendering-commandBuffer-cmdpool
        debug_assert!(self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS));

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_rendering_unchecked(&mut self) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_end_rendering)(self.handle());
        } else {
            debug_assert!(self.device().enabled_extensions().khr_dynamic_rendering);
            (fns.khr_dynamic_rendering.cmd_end_rendering_khr)(self.handle());
        }

        let command_index = self.next_command_index;
        let command_name = "end_rendering";
        let render_pass_state = self.builder_state.render_pass.as_ref().unwrap();

        record_subpass_attachments_resolve(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            render_pass_state,
        );
        record_subpass_attachments_store(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            render_pass_state,
        );

        self.builder_state.render_pass = None;

        self.next_command_index += 1;
        self
    }

    /// Clears specific regions of specific attachments of the framebuffer.
    ///
    /// `attachments` specify the types of attachments and their clear values.
    /// `rects` specify the regions to clear.
    ///
    /// A graphics pipeline must have been bound using [`bind_pipeline_graphics`].
    /// And the command must be inside render pass.
    ///
    /// If the render pass instance this is recorded in uses multiview,
    /// then `ClearRect.base_array_layer` must be zero and `ClearRect.layer_count` must be one.
    ///
    /// The rectangle area must be inside the render area ranges.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    ///
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    #[inline]
    pub unsafe fn clear_attachments(
        &mut self,
        attachments: impl IntoIterator<Item = ClearAttachment>,
        rects: impl IntoIterator<Item = ClearRect>,
    ) -> Result<&mut Self, RenderPassError> {
        let attachments: SmallVec<[ClearAttachment; 3]> = attachments.into_iter().collect();
        let rects: SmallVec<[ClearRect; 4]> = rects.into_iter().collect();

        self.validate_clear_attachments(&attachments, &rects)?;

        unsafe { Ok(self.clear_attachments_unchecked(attachments, rects)) }
    }

    fn validate_clear_attachments(
        &self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> Result<(), RenderPassError> {
        // VUID-vkCmdClearAttachments-renderpass
        let render_pass_state = self
            .builder_state
            .render_pass
            .as_ref()
            .ok_or(RenderPassError::ForbiddenOutsideRenderPass)?;

        if render_pass_state.contents != SubpassContents::Inline {
            return Err(RenderPassError::ForbiddenWithSubpassContents {
                contents: render_pass_state.contents,
            });
        }

        //let subpass_desc = begin_render_pass_state.subpass.subpass_desc();
        //let render_pass = begin_render_pass_state.subpass.render_pass();
        let is_multiview = render_pass_state.rendering_info.view_mask != 0;
        let mut layer_count = u32::MAX;

        for &clear_attachment in attachments {
            match clear_attachment {
                ClearAttachment::Color {
                    color_attachment,
                    clear_value,
                } => {
                    let attachment_format = *render_pass_state
                        .rendering_info
                        .color_attachment_formats
                        .get(color_attachment as usize)
                        .ok_or(RenderPassError::ColorAttachmentIndexOutOfRange {
                            color_attachment_index: color_attachment,
                            num_color_attachments: render_pass_state
                                .rendering_info
                                .color_attachment_formats
                                .len() as u32,
                        })?;

                    // VUID-vkCmdClearAttachments-aspectMask-02501
                    if !attachment_format.map_or(false, |format| {
                        matches!(
                            (clear_value, format.type_color().unwrap()),
                            (
                                ClearColorValue::Float(_),
                                NumericType::SFLOAT
                                    | NumericType::UFLOAT
                                    | NumericType::SNORM
                                    | NumericType::UNORM
                                    | NumericType::SSCALED
                                    | NumericType::USCALED
                                    | NumericType::SRGB
                            ) | (ClearColorValue::Int(_), NumericType::SINT)
                                | (ClearColorValue::Uint(_), NumericType::UINT)
                        )
                    }) {
                        return Err(RenderPassError::ClearAttachmentNotCompatible {
                            clear_attachment,
                            attachment_format,
                        });
                    }

                    let image_view = render_pass_state
                        .attachments
                        .as_ref()
                        .and_then(|attachments| {
                            attachments.color_attachments[color_attachment as usize].as_ref()
                        })
                        .map(|attachment_info| &attachment_info.image_view);

                    // We only know the layer count if we have a known attachment image.
                    if let Some(image_view) = image_view {
                        let array_layers = &image_view.subresource_range().array_layers;
                        layer_count = min(layer_count, array_layers.end - array_layers.start);
                    }
                }
                ClearAttachment::Depth(_)
                | ClearAttachment::Stencil(_)
                | ClearAttachment::DepthStencil(_) => {
                    let depth_format = render_pass_state.rendering_info.depth_attachment_format;
                    let stencil_format = render_pass_state.rendering_info.stencil_attachment_format;

                    // VUID-vkCmdClearAttachments-aspectMask-02502
                    if matches!(
                        clear_attachment,
                        ClearAttachment::Depth(_) | ClearAttachment::DepthStencil(_)
                    ) && !depth_format.map_or(false, |format| {
                        format.aspects().intersects(ImageAspects::DEPTH)
                    }) {
                        return Err(RenderPassError::ClearAttachmentNotCompatible {
                            clear_attachment,
                            attachment_format: None,
                        });
                    }

                    // VUID-vkCmdClearAttachments-aspectMask-02503
                    if matches!(
                        clear_attachment,
                        ClearAttachment::Stencil(_) | ClearAttachment::DepthStencil(_)
                    ) && !stencil_format.map_or(false, |format| {
                        format.aspects().intersects(ImageAspects::STENCIL)
                    }) {
                        return Err(RenderPassError::ClearAttachmentNotCompatible {
                            clear_attachment,
                            attachment_format: None,
                        });
                    }

                    let image_view = render_pass_state
                        .attachments
                        .as_ref()
                        .and_then(|attachments| attachments.depth_attachment.as_ref())
                        .map(|attachment_info| &attachment_info.image_view);

                    // We only know the layer count if we have a known attachment image.
                    if let Some(image_view) = image_view {
                        let array_layers = &image_view.subresource_range().array_layers;
                        layer_count = min(layer_count, array_layers.end - array_layers.start);
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

            // VUID-vkCmdClearAttachments-pRects-00017
            if rect.array_layers.end > layer_count {
                return Err(RenderPassError::RectArrayLayersOutOfBounds { rect_index });
            }

            // VUID-vkCmdClearAttachments-baseArrayLayer-00018
            if is_multiview && rect.array_layers != (0..1) {
                return Err(RenderPassError::MultiviewRectArrayLayersInvalid { rect_index });
            }
        }

        // VUID-vkCmdClearAttachments-commandBuffer-cmdpool
        debug_assert!(self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS));

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_attachments_unchecked(
        &mut self,
        attachments: impl IntoIterator<Item = ClearAttachment>,
        rects: impl IntoIterator<Item = ClearRect>,
    ) -> &mut Self {
        let attachments_vk: SmallVec<[_; 3]> = attachments.into_iter().map(|v| v.into()).collect();
        let rects_vk: SmallVec<[_; 4]> = rects
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

        if attachments_vk.is_empty() || rects_vk.is_empty() {
            return self;
        }

        let fns = self.device().fns();
        (fns.v1_0.cmd_clear_attachments)(
            self.handle(),
            attachments_vk.len() as u32,
            attachments_vk.as_ptr(),
            rects_vk.len() as u32,
            rects_vk.as_ptr(),
        );

        // TODO: sync state update

        self.next_command_index += 1;
        self
    }
}

fn record_subpass_attachments_resolve(
    resources_usage_state: &mut ResourcesState,
    command_index: usize,
    command_name: &'static str,
    render_pass_state: &RenderPassState,
) {
    let attachments = render_pass_state.attachments.as_ref().unwrap();

    let record_attachment = |resources_usage_state: &mut ResourcesState,
                             attachment_info,
                             aspects_override,
                             resource_in_command,
                             resolve_resource_in_command| {
        let &RenderPassStateAttachmentInfo {
            ref image_view,
            image_layout,
            ref resolve_info,
            ..
        } = attachment_info;

        let image = image_view.image();
        let image_inner = image.inner();
        let mut subresource_range = image_view.subresource_range().clone();
        subresource_range.array_layers.start += image_inner.first_layer;
        subresource_range.array_layers.end += image_inner.first_layer;
        subresource_range.mip_levels.start += image_inner.first_mipmap_level;
        subresource_range.mip_levels.end += image_inner.first_mipmap_level;

        if let Some(aspects) = aspects_override {
            subresource_range.aspects = aspects;
        }

        let use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command,
            secondary_use_ref: None,
        };

        if let Some(resolve_info) = resolve_info {
            let &RenderPassStateAttachmentResolveInfo {
                image_view: ref resolve_image_view,
                image_layout: resolve_image_layout,
                ..
            } = resolve_info;

            let resolve_image = resolve_image_view.image();
            let resolve_image_inner = resolve_image.inner();
            let mut resolve_subresource_range = resolve_image_view.subresource_range().clone();
            resolve_subresource_range.array_layers.start += resolve_image_inner.first_layer;
            resolve_subresource_range.array_layers.end += resolve_image_inner.first_layer;
            resolve_subresource_range.mip_levels.start += resolve_image_inner.first_mipmap_level;
            resolve_subresource_range.mip_levels.end += resolve_image_inner.first_mipmap_level;

            if let Some(aspects) = aspects_override {
                resolve_subresource_range.aspects = aspects;
            }

            let resolve_use_ref = ResourceUseRef {
                command_index,
                command_name,
                resource_in_command: resolve_resource_in_command,
                secondary_use_ref: None,
            };

            // The resolve operation uses the stages/access for color attachments,
            // even for depth/stencil attachments.
            resources_usage_state.record_image_access(
                &use_ref,
                image_inner.image,
                subresource_range,
                PipelineStageAccess::ColorAttachmentOutput_ColorAttachmentRead,
                image_layout,
            );
            resources_usage_state.record_image_access(
                &resolve_use_ref,
                resolve_image_inner.image,
                resolve_subresource_range,
                PipelineStageAccess::ColorAttachmentOutput_ColorAttachmentWrite,
                resolve_image_layout,
            );
        }
    };

    if let Some(attachment_info) = attachments.depth_attachment.as_ref() {
        record_attachment(
            resources_usage_state,
            attachment_info,
            Some(ImageAspects::DEPTH),
            ResourceInCommand::DepthStencilAttachment,
            ResourceInCommand::DepthStencilResolveAttachment,
        );
    }

    if let Some(attachment_info) = attachments.stencil_attachment.as_ref() {
        record_attachment(
            resources_usage_state,
            attachment_info,
            Some(ImageAspects::STENCIL),
            ResourceInCommand::DepthStencilAttachment,
            ResourceInCommand::DepthStencilResolveAttachment,
        );
    }

    for (index, attachment_info) in (attachments.color_attachments.iter().enumerate())
        .filter_map(|(i, a)| a.as_ref().map(|a| (i as u32, a)))
    {
        record_attachment(
            resources_usage_state,
            attachment_info,
            None,
            ResourceInCommand::ColorAttachment { index },
            ResourceInCommand::ColorResolveAttachment { index },
        );
    }
}

fn record_subpass_attachments_store(
    resources_usage_state: &mut ResourcesState,
    command_index: usize,
    command_name: &'static str,
    render_pass_state: &RenderPassState,
) {
    let attachments = render_pass_state.attachments.as_ref().unwrap();

    let record_attachment = |resources_usage_state: &mut ResourcesState,
                             attachment_info,
                             aspects_override,
                             resource_in_command,
                             resolve_resource_in_command| {
        let &RenderPassStateAttachmentInfo {
            ref image_view,
            image_layout,
            store_access,
            ref resolve_info,
            ..
        } = attachment_info;

        if let Some(access) = store_access {
            let image = image_view.image();
            let image_inner = image.inner();
            let mut subresource_range = image_view.subresource_range().clone();
            subresource_range.array_layers.start += image_inner.first_layer;
            subresource_range.array_layers.end += image_inner.first_layer;
            subresource_range.mip_levels.start += image_inner.first_mipmap_level;
            subresource_range.mip_levels.end += image_inner.first_mipmap_level;

            if let Some(aspects) = aspects_override {
                subresource_range.aspects = aspects;
            }

            let use_ref = ResourceUseRef {
                command_index,
                command_name,
                resource_in_command,
                secondary_use_ref: None,
            };

            resources_usage_state.record_image_access(
                &use_ref,
                image_inner.image,
                subresource_range,
                access,
                image_layout,
            );
        }

        if let Some(resolve_info) = resolve_info {
            let &RenderPassStateAttachmentResolveInfo {
                ref image_view,
                image_layout,
                store_access,
                ..
            } = resolve_info;

            if let Some(access) = store_access {
                let image = image_view.image();
                let image_inner = image.inner();
                let mut subresource_range = image_view.subresource_range().clone();
                subresource_range.array_layers.start += image_inner.first_layer;
                subresource_range.array_layers.end += image_inner.first_layer;
                subresource_range.mip_levels.start += image_inner.first_mipmap_level;
                subresource_range.mip_levels.end += image_inner.first_mipmap_level;

                if let Some(aspects) = aspects_override {
                    subresource_range.aspects = aspects;
                }

                let use_ref = ResourceUseRef {
                    command_index,
                    command_name,
                    resource_in_command: resolve_resource_in_command,
                    secondary_use_ref: None,
                };

                resources_usage_state.record_image_access(
                    &use_ref,
                    image_inner.image,
                    subresource_range,
                    access,
                    image_layout,
                );
            }
        }
    };

    if let Some(attachment_info) = attachments.depth_attachment.as_ref() {
        record_attachment(
            resources_usage_state,
            attachment_info,
            Some(ImageAspects::DEPTH),
            ResourceInCommand::DepthStencilAttachment,
            ResourceInCommand::DepthStencilResolveAttachment,
        );
    }

    if let Some(attachment_info) = attachments.stencil_attachment.as_ref() {
        record_attachment(
            resources_usage_state,
            attachment_info,
            Some(ImageAspects::STENCIL),
            ResourceInCommand::DepthStencilAttachment,
            ResourceInCommand::DepthStencilResolveAttachment,
        );
    }

    for (index, attachment_info) in (attachments.color_attachments.iter().enumerate())
        .filter_map(|(i, a)| a.as_ref().map(|a| (i as u32, a)))
    {
        record_attachment(
            resources_usage_state,
            attachment_info,
            None,
            ResourceInCommand::ColorAttachment { index },
            ResourceInCommand::ColorResolveAttachment { index },
        );
    }
}

fn record_subpass_attachments_load(
    resources_usage_state: &mut ResourcesState,
    command_index: usize,
    command_name: &'static str,
    render_pass_state: &RenderPassState,
) {
    let attachments = render_pass_state.attachments.as_ref().unwrap();

    let record_attachment = |resources_usage_state: &mut ResourcesState,
                             attachment_info,
                             aspects_override,
                             resource_in_command,
                             resolve_resource_in_command| {
        let &RenderPassStateAttachmentInfo {
            ref image_view,
            image_layout,
            load_access,
            ref resolve_info,
            ..
        } = attachment_info;

        if let Some(access) = load_access {
            let image = image_view.image();
            let image_inner = image.inner();
            let mut subresource_range = image_view.subresource_range().clone();
            subresource_range.array_layers.start += image_inner.first_layer;
            subresource_range.array_layers.end += image_inner.first_layer;
            subresource_range.mip_levels.start += image_inner.first_mipmap_level;
            subresource_range.mip_levels.end += image_inner.first_mipmap_level;

            if let Some(aspects) = aspects_override {
                subresource_range.aspects = aspects;
            }

            let use_ref = ResourceUseRef {
                command_index,
                command_name,
                resource_in_command,
                secondary_use_ref: None,
            };

            resources_usage_state.record_image_access(
                &use_ref,
                image_inner.image,
                subresource_range,
                access,
                image_layout,
            );
        }

        if let Some(resolve_info) = resolve_info {
            let &RenderPassStateAttachmentResolveInfo {
                ref image_view,
                image_layout,
                load_access,
                ..
            } = resolve_info;

            if let Some(access) = load_access {
                let image = image_view.image();
                let image_inner = image.inner();
                let mut subresource_range = image_view.subresource_range().clone();
                subresource_range.array_layers.start += image_inner.first_layer;
                subresource_range.array_layers.end += image_inner.first_layer;
                subresource_range.mip_levels.start += image_inner.first_mipmap_level;
                subresource_range.mip_levels.end += image_inner.first_mipmap_level;

                if let Some(aspects) = aspects_override {
                    subresource_range.aspects = aspects;
                }

                let use_ref = ResourceUseRef {
                    command_index,
                    command_name,
                    resource_in_command: resolve_resource_in_command,
                    secondary_use_ref: None,
                };

                resources_usage_state.record_image_access(
                    &use_ref,
                    image_inner.image,
                    subresource_range,
                    access,
                    image_layout,
                );
            }
        }
    };

    if let Some(attachment_info) = attachments.depth_attachment.as_ref() {
        record_attachment(
            resources_usage_state,
            attachment_info,
            Some(ImageAspects::DEPTH),
            ResourceInCommand::DepthStencilAttachment,
            ResourceInCommand::DepthStencilResolveAttachment,
        );
    }

    if let Some(attachment_info) = attachments.stencil_attachment.as_ref() {
        record_attachment(
            resources_usage_state,
            attachment_info,
            Some(ImageAspects::STENCIL),
            ResourceInCommand::DepthStencilAttachment,
            ResourceInCommand::DepthStencilResolveAttachment,
        );
    }

    for (index, attachment_info) in (attachments.color_attachments.iter().enumerate())
        .filter_map(|(i, a)| a.as_ref().map(|a| (i as u32, a)))
    {
        record_attachment(
            resources_usage_state,
            attachment_info,
            None,
            ResourceInCommand::ColorAttachment { index },
            ResourceInCommand::ColorResolveAttachment { index },
        );
    }
}
