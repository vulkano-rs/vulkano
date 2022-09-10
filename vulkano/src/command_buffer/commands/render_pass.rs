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
        auto::{
            BeginRenderPassState, BeginRenderingAttachments, BeginRenderingState, RenderPassState,
            RenderPassStateType,
        },
        pool::CommandPoolBuilderAlloc,
        synced::{Command, Resource, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SubpassContents,
    },
    device::DeviceOwned,
    format::{ClearColorValue, ClearValue, Format, NumericType},
    image::{ImageLayout, ImageViewAbstract, SampleCount},
    render_pass::{
        AttachmentDescription, Framebuffer, LoadOp, RenderPass, ResolveMode, StoreOp,
        SubpassDescription,
    },
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStages},
    RequirementNotMet, RequiresOneOf, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    cmp::min,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    ops::Range,
    sync::Arc,
};

/// # Commands for render passes.
///
/// These commands require a graphics queue.
impl<P> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>
where
    P: CommandPoolBuilderAlloc,
{
    /// Begins a render pass using a render pass object and framebuffer.
    ///
    /// You must call this or `begin_rendering` before you can record draw commands.
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

            let subpass = render_pass.clone().first_subpass();
            let view_mask = subpass.subpass_desc().view_mask;

            let render_pass_state = RenderPassState {
                contents,
                render_area_offset,
                render_area_extent,
                render_pass: BeginRenderPassState {
                    subpass,
                    framebuffer: Some(framebuffer.clone()),
                }
                .into(),
                view_mask,
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

        // VUID-VkSubpassBeginInfo-contents-parameter
        contents.validate_device(device)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBeginRenderPass2-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
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
            let render_pass_state = self.render_pass_state.as_mut().unwrap();
            let begin_render_pass_state = match &mut render_pass_state.render_pass {
                RenderPassStateType::BeginRenderPass(x) => x,
                _ => unreachable!(),
            };

            begin_render_pass_state.subpass.next_subpass();
            render_pass_state.contents = contents;
            render_pass_state.view_mask = begin_render_pass_state.subpass.subpass_desc().view_mask;

            if render_pass_state.view_mask != 0 {
                // When multiview is enabled, at the beginning of each subpass, all
                // non-render pass state is undefined.
                self.inner.reset_state();
            }

            self.inner.next_subpass(contents);
        }

        Ok(self)
    }

    fn validate_next_subpass(&self, contents: SubpassContents) -> Result<(), RenderPassError> {
        let device = self.device();

        // VUID-VkSubpassBeginInfo-contents-parameter
        contents.validate_device(device)?;

        // VUID-vkCmdNextSubpass2-renderpass
        let render_pass_state = self
            .render_pass_state
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
        if self.query_state.values().any(|state| state.in_subpass) {
            return Err(RenderPassError::QueryIsActive);
        }

        // VUID-vkCmdNextSubpass2-commandBuffer-cmdpool
        debug_assert!({
            let queue_family_properties = self.queue_family_properties();
            queue_family_properties.queue_flags.graphics
        });

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
        if self.query_state.values().any(|state| state.in_subpass) {
            return Err(RenderPassError::QueryIsActive);
        }

        // VUID-vkCmdEndRenderPass2-commandBuffer-cmdpool
        debug_assert!({
            let queue_family_properties = self.queue_family_properties();
            queue_family_properties.queue_flags.graphics
        });

        // VUID-vkCmdEndRenderPass2-bufferlevel
        // Ensured by the type of the impl block

        Ok(())
    }
}

impl<L, P> AutoCommandBufferBuilder<L, P> {
    /// Begins a render pass without a render pass object or framebuffer.
    ///
    /// You must call this or `begin_render_pass` before you can record draw commands.
    pub fn begin_rendering(
        &mut self,
        mut rendering_info: RenderingInfo,
    ) -> Result<&mut Self, RenderPassError> {
        {
            let &mut RenderingInfo {
                render_area_offset,
                ref mut render_area_extent,
                ref mut layer_count,
                view_mask,
                ref color_attachments,
                ref depth_attachment,
                ref stencil_attachment,
                contents: _,
                _ne: _,
            } = &mut rendering_info;

            let auto_extent = render_area_extent[0] == 0 || render_area_extent[1] == 0;
            let auto_layers = *layer_count == 0;

            // Set the values based on the attachment sizes.
            if auto_extent || auto_layers {
                if auto_extent {
                    *render_area_extent = [u32::MAX, u32::MAX];
                }

                if auto_layers {
                    if view_mask != 0 {
                        *layer_count = 1;
                    } else {
                        *layer_count = u32::MAX;
                    }
                }

                for image_view in (color_attachments.iter().flatten())
                    .chain(depth_attachment.iter())
                    .chain(stencil_attachment.iter())
                    .flat_map(|attachment_info| {
                        Some(&attachment_info.image_view).into_iter().chain(
                            attachment_info
                                .resolve_info
                                .as_ref()
                                .map(|resolve_info| &resolve_info.image_view),
                        )
                    })
                {
                    if auto_extent {
                        let extent = image_view.dimensions().width_height();

                        for i in 0..2 {
                            render_area_extent[i] = min(render_area_extent[i], extent[i]);
                        }
                    }

                    if auto_layers {
                        let subresource_range = image_view.subresource_range();
                        let array_layers = subresource_range.array_layers.end
                            - subresource_range.array_layers.start;

                        *layer_count = min(*layer_count, array_layers);
                    }
                }

                if auto_extent {
                    if *render_area_extent == [u32::MAX, u32::MAX] {
                        return Err(RenderPassError::AutoExtentAttachmentsEmpty);
                    }

                    // Subtract the offset from the calculated max extent.
                    // If there is an underflow, then the offset is too large, and validation should
                    // catch that later.
                    for i in 0..2 {
                        render_area_extent[i] = render_area_extent[i]
                            .checked_sub(render_area_offset[i])
                            .unwrap_or(1);
                    }
                }

                if auto_layers {
                    if *layer_count == u32::MAX {
                        return Err(RenderPassError::AutoLayersAttachmentsEmpty);
                    }
                }
            }
        }

        self.validate_begin_rendering(&mut rendering_info)?;

        unsafe {
            let &RenderingInfo {
                render_area_offset,
                render_area_extent,
                layer_count: _,
                view_mask,
                ref color_attachments,
                ref depth_attachment,
                ref stencil_attachment,
                contents,
                _ne: _,
            } = &rendering_info;

            let render_pass_state = RenderPassState {
                contents,
                render_area_offset,
                render_area_extent,
                render_pass: BeginRenderingState {
                    attachments: Some(BeginRenderingAttachments {
                        color_attachments: color_attachments.clone(),
                        depth_attachment: depth_attachment.clone(),
                        stencil_attachment: stencil_attachment.clone(),
                    }),
                    color_attachment_formats: color_attachments
                        .iter()
                        .map(|a| a.as_ref().map(|a| a.image_view.format().unwrap()))
                        .collect(),
                    depth_attachment_format: depth_attachment
                        .as_ref()
                        .map(|a| a.image_view.format().unwrap()),
                    stencil_attachment_format: stencil_attachment
                        .as_ref()
                        .map(|a| a.image_view.format().unwrap()),
                    pipeline_used: false,
                }
                .into(),
                view_mask,
            };

            self.inner.begin_rendering(rendering_info)?;

            self.render_pass_state = Some(render_pass_state);
        }

        Ok(self)
    }

    fn validate_begin_rendering(
        &self,
        rendering_info: &mut RenderingInfo,
    ) -> Result<(), RenderPassError> {
        let device = self.device();
        let properties = device.physical_device().properties();

        // VUID-vkCmdBeginRendering-dynamicRendering-06446
        if !device.enabled_features().dynamic_rendering {
            return Err(RenderPassError::RequirementNotMet {
                required_for: "`begin_rendering`",
                requires_one_of: RequiresOneOf {
                    features: &["dynamic_rendering"],
                    ..Default::default()
                },
            });
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBeginRendering-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(RenderPassError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdBeginRendering-renderpass
        if self.render_pass_state.is_some() {
            return Err(RenderPassError::ForbiddenInsideRenderPass);
        }

        let &mut RenderingInfo {
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
            let &RenderingAttachmentInfo {
                ref image_view,
                image_layout,
                ref resolve_info,
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
            if !image_view.usage().color_attachment {
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
                ) {
                    return Err(RenderPassError::ColorAttachmentResolveLayoutInvalid {
                        attachment_index,
                    });
                }
            }
        }

        if let Some(attachment_info) = depth_attachment {
            let &RenderingAttachmentInfo {
                ref image_view,
                image_layout,
                ref resolve_info,
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
            if !image_aspects.depth {
                return Err(RenderPassError::DepthAttachmentFormatUsageNotSupported);
            }

            // VUID-VkRenderingInfo-pDepthAttachment-06088
            if !image_view.usage().depth_stencil_attachment {
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
                    .map_or(false, |modes| modes.contains_mode(mode))
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
                ) {
                    return Err(RenderPassError::DepthAttachmentResolveLayoutInvalid);
                }
            }
        }

        if let Some(attachment_info) = stencil_attachment {
            let &RenderingAttachmentInfo {
                ref image_view,
                image_layout,
                ref resolve_info,
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
            if !image_aspects.stencil {
                return Err(RenderPassError::StencilAttachmentFormatUsageNotSupported);
            }

            // VUID-VkRenderingInfo-pStencilAttachment-06089
            if !image_view.usage().depth_stencil_attachment {
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
                    .map_or(false, |modes| modes.contains_mode(mode))
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

        Ok(())
    }

    /// Ends the render pass previously begun with `begin_rendering`.
    pub fn end_rendering(&mut self) -> Result<&mut Self, RenderPassError> {
        self.validate_end_rendering()?;

        unsafe {
            self.inner.end_rendering();
            self.render_pass_state = None;
        }

        Ok(self)
    }

    fn validate_end_rendering(&self) -> Result<(), RenderPassError> {
        // VUID-vkCmdEndRendering-renderpass
        let render_pass_state = self
            .render_pass_state
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
        debug_assert!({
            let queue_family_properties = self.queue_family_properties();
            queue_family_properties.queue_flags.graphics
        });

        Ok(())
    }

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
                contents: render_pass_state.contents,
            });
        }

        //let subpass_desc = begin_render_pass_state.subpass.subpass_desc();
        //let render_pass = begin_render_pass_state.subpass.render_pass();
        let is_multiview = render_pass_state.view_mask != 0;
        let mut layer_count = u32::MAX;

        for &clear_attachment in attachments {
            match clear_attachment {
                ClearAttachment::Color {
                    color_attachment,
                    clear_value,
                } => {
                    let attachment_format = match &render_pass_state.render_pass {
                        RenderPassStateType::BeginRenderPass(state) => {
                            let color_attachments = &state.subpass.subpass_desc().color_attachments;
                            let atch_ref = color_attachments.get(color_attachment as usize).ok_or(
                                RenderPassError::ColorAttachmentIndexOutOfRange {
                                    color_attachment_index: color_attachment,
                                    num_color_attachments: color_attachments.len() as u32,
                                },
                            )?;

                            atch_ref.as_ref().map(|atch_ref| {
                                state.subpass.render_pass().attachments()
                                    [atch_ref.attachment as usize]
                                    .format
                                    .unwrap()
                            })
                        }
                        RenderPassStateType::BeginRendering(state) => *state
                            .color_attachment_formats
                            .get(color_attachment as usize)
                            .ok_or(RenderPassError::ColorAttachmentIndexOutOfRange {
                                color_attachment_index: color_attachment,
                                num_color_attachments: state.color_attachment_formats.len() as u32,
                            })?,
                    };

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

                    let image_view = match &render_pass_state.render_pass {
                        RenderPassStateType::BeginRenderPass(state) => (state.framebuffer.as_ref())
                            .zip(
                                state.subpass.subpass_desc().color_attachments
                                    [color_attachment as usize]
                                    .as_ref(),
                            )
                            .map(|(framebuffer, atch_ref)| {
                                &framebuffer.attachments()[atch_ref.attachment as usize]
                            }),
                        RenderPassStateType::BeginRendering(state) => state
                            .attachments
                            .as_ref()
                            .and_then(|attachments| {
                                attachments.color_attachments[color_attachment as usize].as_ref()
                            })
                            .map(|attachment_info| &attachment_info.image_view),
                    };

                    // We only know the layer count if we have a known attachment image.
                    if let Some(image_view) = image_view {
                        let array_layers = &image_view.subresource_range().array_layers;
                        layer_count = min(layer_count, array_layers.end - array_layers.start);
                    }
                }
                ClearAttachment::Depth(_)
                | ClearAttachment::Stencil(_)
                | ClearAttachment::DepthStencil(_) => {
                    let (depth_format, stencil_format) = match &render_pass_state.render_pass {
                        RenderPassStateType::BeginRenderPass(state) => state
                            .subpass
                            .subpass_desc()
                            .depth_stencil_attachment
                            .as_ref()
                            .map_or((None, None), |atch_ref| {
                                let format = state.subpass.render_pass().attachments()
                                    [atch_ref.attachment as usize]
                                    .format
                                    .unwrap();
                                (Some(format), Some(format))
                            }),
                        RenderPassStateType::BeginRendering(state) => (
                            state.depth_attachment_format,
                            state.stencil_attachment_format,
                        ),
                    };

                    // VUID-vkCmdClearAttachments-aspectMask-02502
                    if matches!(
                        clear_attachment,
                        ClearAttachment::Depth(_) | ClearAttachment::DepthStencil(_)
                    ) && !depth_format.map_or(false, |format| format.aspects().depth)
                    {
                        return Err(RenderPassError::ClearAttachmentNotCompatible {
                            clear_attachment,
                            attachment_format: None,
                        });
                    }

                    // VUID-vkCmdClearAttachments-aspectMask-02503
                    if matches!(
                        clear_attachment,
                        ClearAttachment::Stencil(_) | ClearAttachment::DepthStencil(_)
                    ) && !stencil_format.map_or(false, |format| format.aspects().stencil)
                    {
                        return Err(RenderPassError::ClearAttachmentNotCompatible {
                            clear_attachment,
                            attachment_format: None,
                        });
                    }

                    let image_view = match &render_pass_state.render_pass {
                        RenderPassStateType::BeginRenderPass(state) => (state.framebuffer.as_ref())
                            .zip(
                                state
                                    .subpass
                                    .subpass_desc()
                                    .depth_stencil_attachment
                                    .as_ref(),
                            )
                            .map(|(framebuffer, atch_ref)| {
                                &framebuffer.attachments()[atch_ref.attachment as usize]
                            }),
                        RenderPassStateType::BeginRendering(state) => state
                            .attachments
                            .as_ref()
                            .and_then(|attachments| attachments.depth_attachment.as_ref())
                            .map(|attachment_info| &attachment_info.image_view),
                    };

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
        debug_assert!({
            let queue_family_properties = self.queue_family_properties();
            queue_family_properties.queue_flags.graphics
        });

        Ok(())
    }
}

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdBeginRenderPass` on the builder.
    // TODO: it shouldn't be possible to get an error if the framebuffer checked conflicts already
    // TODO: after begin_render_pass has been called, flushing should be forbidden and an error
    //       returned if conflict
    #[inline]
    pub unsafe fn begin_render_pass(
        &mut self,
        render_pass_begin_info: RenderPassBeginInfo,
        contents: SubpassContents,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            render_pass_begin_info: RenderPassBeginInfo,
            contents: SubpassContents,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "begin_render_pass"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.begin_render_pass(&self.render_pass_begin_info, self.contents);
            }
        }

        let &RenderPassBeginInfo {
            ref render_pass,
            ref framebuffer,
            render_area_offset: _,
            render_area_extent: _,
            clear_values: _,
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
                                ..PipelineStages::empty()
                            }, // TODO: wrong!
                            access: AccessFlags {
                                input_attachment_read: true,
                                color_attachment_read: true,
                                color_attachment_write: true,
                                depth_stencil_attachment_read: true,
                                depth_stencil_attachment_write: true,
                                ..AccessFlags::empty()
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
            contents,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        self.latest_render_pass_enter = Some(self.commands.len() - 1);

        Ok(())
    }

    /// Calls `vkCmdNextSubpass` on the builder.
    #[inline]
    pub unsafe fn next_subpass(&mut self, contents: SubpassContents) {
        struct Cmd {
            contents: SubpassContents,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "next_subpass"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.next_subpass(self.contents);
            }
        }

        self.commands.push(Box::new(Cmd { contents }));
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

    /// Calls `vkCmdBeginRendering` on the builder.
    #[inline]
    pub unsafe fn begin_rendering(
        &mut self,
        rendering_info: RenderingInfo,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            rendering_info: RenderingInfo,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "begin_rendering"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.begin_rendering(&self.rendering_info);
            }
        }

        let &RenderingInfo {
            render_area_offset: _,
            render_area_extent: _,
            layer_count: _,
            view_mask: _,
            ref color_attachments,
            ref depth_attachment,
            ref stencil_attachment,
            contents: _,
            _ne,
        } = &rendering_info;

        let resources = (color_attachments
            .iter()
            .enumerate()
            .filter_map(|(index, attachment_info)| {
                attachment_info
                    .as_ref()
                    .map(|attachment_info| (index, attachment_info))
            })
            .flat_map(|(index, attachment_info)| {
                let &RenderingAttachmentInfo {
                    ref image_view,
                    image_layout,
                    ref resolve_info,
                    load_op: _,
                    store_op: _,
                    clear_value: _,
                    _ne: _,
                } = attachment_info;

                [
                    Some((
                        format!("color attachment {}", index).into(),
                        Resource::Image {
                            image: image_view.image(),
                            subresource_range: image_view.subresource_range().clone(),
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    all_commands: true,
                                    ..PipelineStages::empty()
                                }, // TODO: wrong!
                                access: AccessFlags {
                                    color_attachment_read: true,
                                    color_attachment_write: true,
                                    ..AccessFlags::empty()
                                }, // TODO: suboptimal
                                exclusive: true, // TODO: suboptimal
                            },
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )),
                    resolve_info.as_ref().map(|resolve_info| {
                        let &RenderingAttachmentResolveInfo {
                            mode: _,
                            ref image_view,
                            image_layout,
                        } = resolve_info;

                        (
                            format!("color resolve attachment {}", index).into(),
                            Resource::Image {
                                image: image_view.image(),
                                subresource_range: image_view.subresource_range().clone(),
                                memory: PipelineMemoryAccess {
                                    stages: PipelineStages {
                                        all_commands: true,
                                        ..PipelineStages::empty()
                                    }, // TODO: wrong!
                                    access: AccessFlags {
                                        color_attachment_read: true,
                                        color_attachment_write: true,
                                        ..AccessFlags::empty()
                                    }, // TODO: suboptimal
                                    exclusive: true, // TODO: suboptimal
                                },
                                start_layout: image_layout,
                                end_layout: image_layout,
                            },
                        )
                    }),
                ]
                .into_iter()
                .flatten()
            }))
        .chain(depth_attachment.iter().flat_map(|attachment_info| {
            let &RenderingAttachmentInfo {
                ref image_view,
                image_layout,
                ref resolve_info,
                load_op: _,
                store_op: _,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            [
                Some((
                    "depth attachment".into(),
                    Resource::Image {
                        image: image_view.image(),
                        subresource_range: image_view.subresource_range().clone(),
                        memory: PipelineMemoryAccess {
                            stages: PipelineStages {
                                all_commands: true,
                                ..PipelineStages::empty()
                            }, // TODO: wrong!
                            access: AccessFlags {
                                depth_stencil_attachment_read: true,
                                depth_stencil_attachment_write: true,
                                ..AccessFlags::empty()
                            }, // TODO: suboptimal
                            exclusive: true, // TODO: suboptimal
                        },
                        start_layout: image_layout,
                        end_layout: image_layout,
                    },
                )),
                resolve_info.as_ref().map(|resolve_info| {
                    let &RenderingAttachmentResolveInfo {
                        mode: _,
                        ref image_view,
                        image_layout,
                    } = resolve_info;

                    (
                        "depth resolve attachment".into(),
                        Resource::Image {
                            image: image_view.image(),
                            subresource_range: image_view.subresource_range().clone(),
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    all_commands: true,
                                    ..PipelineStages::empty()
                                }, // TODO: wrong!
                                access: AccessFlags {
                                    depth_stencil_attachment_read: true,
                                    depth_stencil_attachment_write: true,
                                    ..AccessFlags::empty()
                                }, // TODO: suboptimal
                                exclusive: true, // TODO: suboptimal
                            },
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )
                }),
            ]
            .into_iter()
            .flatten()
        }))
        .chain(stencil_attachment.iter().flat_map(|attachment_info| {
            let &RenderingAttachmentInfo {
                ref image_view,
                image_layout,
                ref resolve_info,
                load_op: _,
                store_op: _,
                clear_value: _,
                _ne: _,
            } = attachment_info;

            [
                Some((
                    "stencil attachment".into(),
                    Resource::Image {
                        image: image_view.image(),
                        subresource_range: image_view.subresource_range().clone(),
                        memory: PipelineMemoryAccess {
                            stages: PipelineStages {
                                all_commands: true,
                                ..PipelineStages::empty()
                            }, // TODO: wrong!
                            access: AccessFlags {
                                depth_stencil_attachment_read: true,
                                depth_stencil_attachment_write: true,
                                ..AccessFlags::empty()
                            }, // TODO: suboptimal
                            exclusive: true, // TODO: suboptimal
                        },
                        start_layout: image_layout,
                        end_layout: image_layout,
                    },
                )),
                resolve_info.as_ref().map(|resolve_info| {
                    let &RenderingAttachmentResolveInfo {
                        mode: _,
                        ref image_view,
                        image_layout,
                    } = resolve_info;

                    (
                        "stencil resolve attachment".into(),
                        Resource::Image {
                            image: image_view.image(),
                            subresource_range: image_view.subresource_range().clone(),
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    all_commands: true,
                                    ..PipelineStages::empty()
                                }, // TODO: wrong!
                                access: AccessFlags {
                                    depth_stencil_attachment_read: true,
                                    depth_stencil_attachment_write: true,
                                    ..AccessFlags::empty()
                                }, // TODO: suboptimal
                                exclusive: true, // TODO: suboptimal
                            },
                            start_layout: image_layout,
                            end_layout: image_layout,
                        },
                    )
                }),
            ]
            .into_iter()
            .flatten()
        }))
        .collect::<Vec<_>>();

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd { rendering_info }));

        for resource in resources {
            self.add_resource(resource);
        }

        self.latest_render_pass_enter = Some(self.commands.len() - 1);

        Ok(())
    }

    /// Calls `vkCmdEndRendering` on the builder.
    #[inline]
    pub unsafe fn end_rendering(&mut self) {
        struct Cmd;

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "end_rendering"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.end_rendering();
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
        contents: SubpassContents,
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
            .iter()
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
            contents: contents.into(),
            ..Default::default()
        };

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_2
            || self.device.enabled_extensions().khr_create_renderpass2
        {
            if self.device.api_version() >= Version::V1_2 {
                (fns.v1_2.cmd_begin_render_pass2)(
                    self.handle,
                    &render_pass_begin_info,
                    &subpass_begin_info,
                );
            } else {
                (fns.khr_create_renderpass2.cmd_begin_render_pass2_khr)(
                    self.handle,
                    &render_pass_begin_info,
                    &subpass_begin_info,
                );
            }
        } else {
            debug_assert!(subpass_begin_info.p_next.is_null());

            (fns.v1_0.cmd_begin_render_pass)(
                self.handle,
                &render_pass_begin_info,
                subpass_begin_info.contents,
            );
        }
    }

    /// Calls `vkCmdNextSubpass` on the builder.
    #[inline]
    pub unsafe fn next_subpass(&mut self, contents: SubpassContents) {
        let fns = self.device.fns();

        let subpass_begin_info = ash::vk::SubpassBeginInfo {
            contents: contents.into(),
            ..Default::default()
        };

        let subpass_end_info = ash::vk::SubpassEndInfo::default();

        if self.device.api_version() >= Version::V1_2
            || self.device.enabled_extensions().khr_create_renderpass2
        {
            if self.device.api_version() >= Version::V1_2 {
                (fns.v1_2.cmd_next_subpass2)(self.handle, &subpass_begin_info, &subpass_end_info);
            } else {
                (fns.khr_create_renderpass2.cmd_next_subpass2_khr)(
                    self.handle,
                    &subpass_begin_info,
                    &subpass_end_info,
                );
            }
        } else {
            debug_assert!(subpass_begin_info.p_next.is_null());
            debug_assert!(subpass_end_info.p_next.is_null());

            (fns.v1_0.cmd_next_subpass)(self.handle, subpass_begin_info.contents);
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
                (fns.v1_2.cmd_end_render_pass2)(self.handle, &subpass_end_info);
            } else {
                (fns.khr_create_renderpass2.cmd_end_render_pass2_khr)(
                    self.handle,
                    &subpass_end_info,
                );
            }
        } else {
            debug_assert!(subpass_end_info.p_next.is_null());

            (fns.v1_0.cmd_end_render_pass)(self.handle);
        }
    }

    /// Calls `vkCmdBeginRendering` on the builder.
    pub unsafe fn begin_rendering(&mut self, rendering_info: &RenderingInfo) {
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

                        (
                            mode.into(),
                            image_view.internal_object(),
                            image_layout.into(),
                        )
                    } else {
                        (
                            ash::vk::ResolveModeFlags::NONE,
                            Default::default(),
                            Default::default(),
                        )
                    };

                ash::vk::RenderingAttachmentInfo {
                    image_view: image_view.internal_object(),
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

        let color_attachments: SmallVec<[_; 2]> =
            color_attachments.iter().map(map_attachment_info).collect();
        let depth_attachment = map_attachment_info(depth_attachment);
        let stencil_attachment = map_attachment_info(stencil_attachment);

        let rendering_info = ash::vk::RenderingInfo {
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
            color_attachment_count: color_attachments.len() as u32,
            p_color_attachments: color_attachments.as_ptr(),
            p_depth_attachment: &depth_attachment,
            p_stencil_attachment: &stencil_attachment,
            ..Default::default()
        };

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_begin_rendering)(self.handle, &rendering_info);
        } else {
            debug_assert!(self.device.enabled_extensions().khr_dynamic_rendering);
            (fns.khr_dynamic_rendering.cmd_begin_rendering_khr)(self.handle, &rendering_info);
        }
    }

    /// Calls `vkCmdEndRendering` on the builder.
    pub unsafe fn end_rendering(&mut self) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_end_rendering)(self.handle);
        } else {
            debug_assert!(self.device.enabled_extensions().khr_dynamic_rendering);
            (fns.khr_dynamic_rendering.cmd_end_rendering_khr)(self.handle);
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
        (fns.v1_0.cmd_clear_attachments)(
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

/// Parameters to begin rendering.
#[derive(Clone, Debug)]
pub struct RenderingInfo {
    /// The offset from the top left corner of the attachments that will be rendered to.
    ///
    /// This value must be smaller than the smallest width and height of the attachment images.
    ///
    /// The default value is `[0, 0]`.
    pub render_area_offset: [u32; 2],

    /// The size of the area that will be rendered to.
    ///
    /// This value plus `render_area_offset` must be no larger than the smallest width and height
    /// of the attachment images.
    /// If one of the elements is set to 0, the extent will be calculated automatically from the
    /// extents of the attachment images to be the largest allowed. At least one attachment image
    /// must be specified in that case.
    ///
    /// The default value is `[0, 0]`.
    pub render_area_extent: [u32; 2],

    /// The number of layers of the attachments that will be rendered to.
    ///
    /// This must be no larger than the smallest number of array layers of the attachment images.
    /// If set to 0, the number of layers will be calculated automatically from the
    /// layer ranges of the attachment images to be the largest allowed. At least one attachment
    /// image must be specified in that case.
    ///
    /// If the render pass uses multiview (`view_mask` is not 0), then this value must be 0 or 1.
    ///
    /// The default value is `0`.
    pub layer_count: u32,

    /// If not `0`, enables multiview rendering, and specifies the view indices that are rendered
    /// to. The value is a bitmask, so that that for example `0b11` will draw to the first two
    /// views and `0b101` will draw to the first and third view.
    ///
    /// If set to a nonzero value, the [`multiview`](crate::device::Features::multiview) feature
    /// must be enabled on the device.
    ///
    /// The default value is `0`.
    pub view_mask: u32,

    /// The color attachments to use for rendering.
    ///
    /// The number of color attachments must be less than the
    /// [`max_color_attachments`](crate::device::Properties::max_color_attachments) limit of the
    /// physical device. All color attachments must have the same `samples` value.
    ///
    /// The default value is empty.
    pub color_attachments: Vec<Option<RenderingAttachmentInfo>>,

    /// The depth attachment to use for rendering.
    ///
    /// If set to `Some`, the image view must have the same `samples` value as those in
    /// `color_attachments`.
    ///
    /// The default value is `None`.
    pub depth_attachment: Option<RenderingAttachmentInfo>,

    /// The stencil attachment to use for rendering.
    ///
    /// If set to `Some`, the image view must have the same `samples` value as those in
    /// `color_attachments`.
    ///
    /// The default value is `None`.
    pub stencil_attachment: Option<RenderingAttachmentInfo>,

    /// What kinds of commands will be recorded in the render pass: either inline draw commands, or
    /// executions of secondary command buffers.
    ///
    /// If recorded in a secondary command buffer, this must be [`SubpassContents::Inline`].
    ///
    /// The default value is [`SubpassContents::Inline`].
    pub contents: SubpassContents,

    pub _ne: crate::NonExhaustive,
}

impl Default for RenderingInfo {
    #[inline]
    fn default() -> Self {
        Self {
            render_area_offset: [0, 0],
            render_area_extent: [0, 0],
            layer_count: 0,
            view_mask: 0,
            color_attachments: Vec::new(),
            depth_attachment: None,
            stencil_attachment: None,
            contents: SubpassContents::Inline,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to specify properties of an attachment.
#[derive(Clone, Debug)]
pub struct RenderingAttachmentInfo {
    /// The image view to use as the attachment.
    ///
    /// There is no default value.
    pub image_view: Arc<dyn ImageViewAbstract>,

    /// The image layout that `image_view` should be in during the resolve operation.
    ///
    /// The default value is [`ImageLayout::ColorAttachmentOptimal`] if `image_view` has a color
    /// format, [`ImageLayout::DepthStencilAttachmentOptimal`] if `image_view` has a depth/stencil
    /// format.
    pub image_layout: ImageLayout,

    /// The resolve operation that should be performed at the end of rendering.
    ///
    /// The default value is `None`.
    pub resolve_info: Option<RenderingAttachmentResolveInfo>,

    /// What the implementation should do with the attachment at the start of rendering.
    ///
    /// The default value is [`LoadOp::DontCare`].
    pub load_op: LoadOp,

    /// What the implementation should do with the attachment at the end of rendering.
    ///
    /// The default value is [`StoreOp::DontCare`].
    pub store_op: StoreOp,

    /// If `load_op` is [`LoadOp::Clear`], specifies the clear value that should be used for the
    /// attachment.
    ///
    /// If `load_op` is something else, provide `None`.
    ///
    /// The default value is `None`.
    pub clear_value: Option<ClearValue>,

    pub _ne: crate::NonExhaustive,
}

impl RenderingAttachmentInfo {
    /// Returns a `RenderingAttachmentInfo` with the specified `image_view`.
    #[inline]
    pub fn image_view(image_view: Arc<dyn ImageViewAbstract>) -> Self {
        let aspects = image_view.format().unwrap().aspects();
        let image_layout = if aspects.depth || aspects.stencil {
            ImageLayout::DepthStencilAttachmentOptimal
        } else {
            ImageLayout::ColorAttachmentOptimal
        };

        Self {
            image_view,
            image_layout,
            resolve_info: None,
            load_op: LoadOp::DontCare,
            store_op: StoreOp::DontCare,
            clear_value: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to specify the resolve behavior of an attachment.
#[derive(Clone, Debug)]
pub struct RenderingAttachmentResolveInfo {
    /// How the resolve operation should be performed.
    ///
    /// The default value is [`ResolveMode::Average`].
    pub mode: ResolveMode,

    /// The image view that the result of the resolve operation should be written to.
    ///
    /// There is no default value.
    pub image_view: Arc<dyn ImageViewAbstract>,

    /// The image layout that `image_view` should be in during the resolve operation.
    ///
    /// The default value is [`ImageLayout::ColorAttachmentOptimal`] if `image_view` has a color
    /// format, [`ImageLayout::DepthStencilAttachmentOptimal`] if `image_view` has a depth/stencil
    /// format.
    pub image_layout: ImageLayout,
}

impl RenderingAttachmentResolveInfo {
    /// Returns a `RenderingAttachmentResolveInfo` with the specified `image_view`.
    #[inline]
    pub fn image_view(image_view: Arc<dyn ImageViewAbstract>) -> Self {
        let aspects = image_view.format().unwrap().aspects();
        let image_layout = if aspects.depth || aspects.stencil {
            ImageLayout::DepthStencilAttachmentOptimal
        } else {
            ImageLayout::ColorAttachmentOptimal
        };

        Self {
            mode: ResolveMode::Average,
            image_view,
            image_layout,
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

/// Error that can happen when recording a render pass command.
#[derive(Clone, Debug)]
pub enum RenderPassError {
    SyncCommandBufferBuilderError(SyncCommandBufferBuilderError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// A framebuffer image did not have the required usage enabled.
    AttachmentImageMissingUsage {
        attachment_index: u32,
        usage: &'static str,
    },

    /// One of the elements of `render_pass_extent` is zero, but no attachment images were given to
    /// calculate the extent from.
    AutoExtentAttachmentsEmpty,

    /// `layer_count` is zero, but no attachment images were given to calculate the number of layers
    /// from.
    AutoLayersAttachmentsEmpty,

    /// A clear attachment value is not compatible with the attachment's format.
    ClearAttachmentNotCompatible {
        clear_attachment: ClearAttachment,
        attachment_format: Option<Format>,
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

    /// A color attachment has a layout that is not supported.
    ColorAttachmentLayoutInvalid {
        attachment_index: u32,
    },

    /// A color attachment is missing the `color_attachment` usage.
    ColorAttachmentMissingUsage {
        attachment_index: u32,
    },

    /// A color resolve attachment has a `format` value different from the corresponding color
    /// attachment.
    ColorAttachmentResolveFormatMismatch {
        attachment_index: u32,
    },

    /// A color resolve attachment has a layout that is not supported.
    ColorAttachmentResolveLayoutInvalid {
        attachment_index: u32,
    },

    /// A color resolve attachment has a resolve mode that is not supported.
    ColorAttachmentResolveModeNotSupported {
        attachment_index: u32,
    },

    /// A color resolve attachment has a `samples` value other than [`SampleCount::Sample1`].
    ColorAttachmentResolveMultisampled {
        attachment_index: u32,
    },

    /// A color attachment has a `samples` value that is different from the first
    /// color attachment.
    ColorAttachmentSamplesMismatch {
        attachment_index: u32,
    },

    /// A color attachment with a resolve attachment has a `samples` value of
    /// [`SampleCount::Sample1`].
    ColorAttachmentWithResolveNotMultisampled {
        attachment_index: u32,
    },

    /// The contents `SubpassContents::SecondaryCommandBuffers` is not allowed inside a secondary command buffer.
    ContentsForbiddenInSecondaryCommandBuffer,

    /// The depth attachment has a format that does not support that usage.
    DepthAttachmentFormatUsageNotSupported,

    /// The depth attachment has a layout that is not supported.
    DepthAttachmentLayoutInvalid,

    /// The depth attachment is missing the `depth_stencil_attachment` usage.
    DepthAttachmentMissingUsage,

    /// The depth resolve attachment has a `format` value different from the corresponding depth
    /// attachment.
    DepthAttachmentResolveFormatMismatch,

    /// The depth resolve attachment has a layout that is not supported.
    DepthAttachmentResolveLayoutInvalid,

    /// The depth resolve attachment has a resolve mode that is not supported.
    DepthAttachmentResolveModeNotSupported,

    /// The depth resolve attachment has a `samples` value other than [`SampleCount::Sample1`].
    DepthAttachmentResolveMultisampled,

    /// The depth attachment has a `samples` value that is different from the first
    /// color attachment.
    DepthAttachmentSamplesMismatch,

    /// The depth attachment has a resolve attachment and has a `samples` value of
    /// [`SampleCount::Sample1`].
    DepthAttachmentWithResolveNotMultisampled,

    /// The depth and stencil attachments have different image views.
    DepthStencilAttachmentImageViewMismatch,

    /// The depth and stencil resolve attachments have different image views.
    DepthStencilAttachmentResolveImageViewMismatch,

    /// The combination of depth and stencil resolve modes is not supported by the device.
    DepthStencilAttachmentResolveModesNotSupported,

    /// Operation forbidden inside a render pass.
    ForbiddenInsideRenderPass,

    /// Operation forbidden outside a render pass.
    ForbiddenOutsideRenderPass,

    /// Operation forbidden inside a render pass instance that was begun with `begin_rendering`.
    ForbiddenWithBeginRendering,

    /// Operation forbidden inside a render pass instance that was begun with `begin_render_pass`.
    ForbiddenWithBeginRenderPass,

    /// Operation forbidden inside a render pass instance that is inherited by a secondary command
    /// buffer.
    ForbiddenWithInheritedRenderPass,

    /// Operation forbidden inside a render subpass with the specified contents.
    ForbiddenWithSubpassContents {
        contents: SubpassContents,
    },

    /// The framebuffer is not compatible with the render pass.
    FramebufferNotCompatible,

    /// The `max_color_attachments` limit has been exceeded.
    MaxColorAttachmentsExceeded {
        color_attachment_count: u32,
        max: u32,
    },

    /// The `max_multiview_view_count` limit has been exceeded.
    MaxMultiviewViewCountExceeded {
        view_count: u32,
        max: u32,
    },

    /// The render pass uses multiview, but `layer_count` was not 0 or 1.
    MultiviewLayersInvalid,

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

    /// The stencil attachment has a format that does not support that usage.
    StencilAttachmentFormatUsageNotSupported,

    /// The stencil attachment has a layout that is not supported.
    StencilAttachmentLayoutInvalid,

    /// The stencil attachment is missing the `depth_stencil_attachment` usage.
    StencilAttachmentMissingUsage,

    /// The stencil resolve attachment has a `format` value different from the corresponding stencil
    /// attachment.
    StencilAttachmentResolveFormatMismatch,

    /// The stencil resolve attachment has a layout that is not supported.
    StencilAttachmentResolveLayoutInvalid,

    /// The stencil resolve attachment has a resolve mode that is not supported.
    StencilAttachmentResolveModeNotSupported,

    /// The stencil resolve attachment has a `samples` value other than [`SampleCount::Sample1`].
    StencilAttachmentResolveMultisampled,

    /// The stencil attachment has a `samples` value that is different from the first
    /// color attachment or the depth attachment.
    StencilAttachmentSamplesMismatch,

    /// The stencil attachment has a resolve attachment and has a `samples` value of
    /// [`SampleCount::Sample1`].
    StencilAttachmentWithResolveNotMultisampled,

    /// Tried to end a render pass with subpasses still remaining in the render pass.
    SubpassesRemaining {
        current_subpass: u32,
        remaining_subpasses: u32,
    },
}

impl Error for RenderPassError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SyncCommandBufferBuilderError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for RenderPassError {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Self::SyncCommandBufferBuilderError(_) => write!(f, "a SyncCommandBufferBuilderError"),

            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),

            Self::AttachmentImageMissingUsage { attachment_index, usage } => write!(
                f,
                "the framebuffer image attached to attachment index {} did not have the required usage {} enabled",
                attachment_index, usage,
            ),
            Self::AutoExtentAttachmentsEmpty => write!(
                f,
                "one of the elements of `render_pass_extent` is zero, but no attachment images were given to calculate the extent from",
            ),
            Self::AutoLayersAttachmentsEmpty => write!(
                f,
                "`layer_count` is zero, but no attachment images were given to calculate the number of layers from",
            ),
            Self::ClearAttachmentNotCompatible {
                clear_attachment,
                attachment_format,
            } => write!(
                f,
                "a clear attachment value ({:?}) is not compatible with the attachment's format ({:?})",
                clear_attachment,
                attachment_format,
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
            Self::ColorAttachmentLayoutInvalid {
                attachment_index,
            } => write!(
                f,
                "color attachment {} has a layout that is not supported",
                attachment_index,
            ),
            Self::ColorAttachmentMissingUsage {
                attachment_index,
            } => write!(
                f,
                "color attachment {} is missing the `color_attachment` usage",
                attachment_index,
            ),
            Self::ColorAttachmentResolveFormatMismatch {
                attachment_index,
            } => write!(
                f,
                "color attachment {} has a `format` value different from the corresponding color attachment",
                attachment_index,
            ),
            Self::ColorAttachmentResolveLayoutInvalid {
                attachment_index,
            } => write!(
                f,
                "color resolve attachment {} has a layout that is not supported",
                attachment_index,
            ),
            Self::ColorAttachmentResolveModeNotSupported {
                attachment_index,
            } => write!(
                f,
                "color resolve attachment {} has a resolve mode that is not supported",
                attachment_index,
            ),
            Self::ColorAttachmentResolveMultisampled {
                attachment_index,
            } => write!(
                f,
                "color resolve attachment {} has a `samples` value other than `SampleCount::Sample1`",
                attachment_index,
            ),
            Self::ColorAttachmentSamplesMismatch {
                attachment_index,
            } => write!(
                f,
                "color attachment {} has a `samples` value that is different from the first color attachment",
                attachment_index,
            ),
            Self::ColorAttachmentWithResolveNotMultisampled {
                attachment_index,
            } => write!(
                f,
                "color attachment {} with a resolve attachment has a `samples` value of `SampleCount::Sample1`",
                attachment_index,
            ),
            Self::ContentsForbiddenInSecondaryCommandBuffer => write!(
                f,
                "the contents `SubpassContents::SecondaryCommandBuffers` is not allowed inside a secondary command buffer",
            ),
            Self::DepthAttachmentFormatUsageNotSupported => write!(
                f,
                "the depth attachment has a format that does not support that usage",
            ),
            Self::DepthAttachmentLayoutInvalid => write!(
                f,
                "the depth attachment has a layout that is not supported",
            ),
            Self::DepthAttachmentMissingUsage => write!(
                f,
                "the depth attachment is missing the `depth_stencil_attachment` usage",
            ),
            Self::DepthAttachmentResolveFormatMismatch => write!(
                f,
                "the depth resolve attachment has a `format` value different from the corresponding depth attachment",
            ),
            Self::DepthAttachmentResolveLayoutInvalid => write!(
                f,
                "the depth resolve attachment has a layout that is not supported",
            ),
            Self::DepthAttachmentResolveModeNotSupported => write!(
                f,
                "the depth resolve attachment has a resolve mode that is not supported",
            ),
            Self::DepthAttachmentResolveMultisampled => write!(
                f,
                "the depth resolve attachment has a `samples` value other than `SampleCount::Sample1`",
            ),
            Self::DepthAttachmentSamplesMismatch => write!(
                f,
                "the depth attachment has a `samples` value that is different from the first color attachment",
            ),
            Self::DepthAttachmentWithResolveNotMultisampled => write!(
                f,
                "the depth attachment has a resolve attachment and has a `samples` value of `SampleCount::Sample1`",
            ),
            Self::DepthStencilAttachmentImageViewMismatch => write!(
                f,
                "the depth and stencil attachments have different image views",
            ),
            Self::DepthStencilAttachmentResolveImageViewMismatch => write!(
                f,
                "the depth and stencil resolve attachments have different image views",
            ),
            Self::DepthStencilAttachmentResolveModesNotSupported => write!(
                f,
                "the combination of depth and stencil resolve modes is not supported by the device",
            ),
            Self::ForbiddenInsideRenderPass => {
                write!(f, "operation forbidden inside a render pass")
            }
            Self::ForbiddenOutsideRenderPass => {
                write!(f, "operation forbidden outside a render pass")
            }
            Self::ForbiddenWithBeginRendering => write!(
                f,
                "operation forbidden inside a render pass instance that was begun with `begin_rendering`",
            ),
            Self::ForbiddenWithBeginRenderPass => write!(
                f,
                "operation forbidden inside a render pass instance that was begun with `begin_render_pass`",
            ),
            Self::ForbiddenWithInheritedRenderPass => write!(
                f,
                "operation forbidden inside a render pass instance that is inherited by a secondary command buffer",
            ),
            Self::ForbiddenWithSubpassContents { contents: subpass_contents } => write!(
                f,
                "operation forbidden inside a render subpass with contents {:?}",
                subpass_contents,
            ),
            Self::FramebufferNotCompatible => write!(
                f,
                "the framebuffer is not compatible with the render pass",
            ),
            Self::MaxColorAttachmentsExceeded { .. } => {
                write!(f, "the `max_color_attachments` limit has been exceeded",)
            }
            Self::MaxMultiviewViewCountExceeded { .. } => {
                write!(f, "the `max_multiview_view_count` limit has been exceeded",)
            },
            Self::MultiviewLayersInvalid => write!(
                f,
                "the render pass uses multiview, but `layer_count` was not 0 or 1",
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
            Self::StencilAttachmentFormatUsageNotSupported => write!(
                f,
                "the stencil attachment has a format that does not support that usage",
            ),
            Self::StencilAttachmentLayoutInvalid => write!(
                f,
                "the stencil attachment has a layout that is not supported",
            ),
            Self::StencilAttachmentMissingUsage => write!(
                f,
                "the stencil attachment is missing the `depth_stencil_attachment` usage",
            ),
            Self::StencilAttachmentResolveFormatMismatch => write!(
                f,
                "the stencil resolve attachment has a `format` value different from the corresponding stencil attachment",
            ),
            Self::StencilAttachmentResolveLayoutInvalid => write!(
                f,
                "the stencil resolve attachment has a layout that is not supported",
            ),
            Self::StencilAttachmentResolveModeNotSupported => write!(
                f,
                "the stencil resolve attachment has a resolve mode that is not supported",
            ),
            Self::StencilAttachmentResolveMultisampled => write!(
                f,
                "the stencil resolve attachment has a `samples` value other than `SampleCount::Sample1`",
            ),
            Self::StencilAttachmentSamplesMismatch => write!(
                f,
                "the stencil attachment has a `samples` value that is different from the first color attachment",
            ),
            Self::StencilAttachmentWithResolveNotMultisampled => write!(
                f,
                "the stencil attachment has a resolve attachment and has a `samples` value of `SampleCount::Sample1`",
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

impl From<RequirementNotMet> for RenderPassError {
    #[inline]
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}
