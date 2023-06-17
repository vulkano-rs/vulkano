// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    command_buffer::{CommandBufferInheritanceRenderingInfo, RenderingInfo},
    format::Format,
    image::ImageAspects,
    render_pass::Subpass,
};

/// Selects the type of subpass that a graphics pipeline is created for.
#[derive(Clone, Debug)]
pub enum PipelineSubpassType {
    BeginRenderPass(Subpass),
    BeginRendering(PipelineRenderingCreateInfo),
}

impl From<Subpass> for PipelineSubpassType {
    #[inline]
    fn from(val: Subpass) -> Self {
        Self::BeginRenderPass(val)
    }
}

impl From<PipelineRenderingCreateInfo> for PipelineSubpassType {
    #[inline]
    fn from(val: PipelineRenderingCreateInfo) -> Self {
        Self::BeginRendering(val)
    }
}

/// The dynamic rendering parameters to create a graphics pipeline.
#[derive(Clone, Debug)]
pub struct PipelineRenderingCreateInfo {
    /// If not `0`, indicates that multiview rendering will be enabled, and specifies the view
    /// indices that are rendered to. The value is a bitmask, so that that for example `0b11` will
    /// draw to the first two views and `0b101` will draw to the first and third view.
    ///
    /// If set to a nonzero value, the [`multiview`](crate::device::Features::multiview) feature
    /// must be enabled on the device.
    ///
    /// The default value is `0`.
    pub view_mask: u32,

    /// The formats of the color attachments that will be used during rendering.
    ///
    /// If an element is `None`, it indicates that the attachment will not be used.
    ///
    /// The default value is empty.
    pub color_attachment_formats: Vec<Option<Format>>,

    /// The format of the depth attachment that will be used during rendering.
    ///
    /// If set to `None`, it indicates that no depth attachment will be used.
    ///
    /// The default value is `None`.
    pub depth_attachment_format: Option<Format>,

    /// The format of the stencil attachment that will be used during rendering.
    ///
    /// If set to `None`, it indicates that no stencil attachment will be used.
    ///
    /// The default value is `None`.
    pub stencil_attachment_format: Option<Format>,

    pub _ne: crate::NonExhaustive,
}

impl Default for PipelineRenderingCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            view_mask: 0,
            color_attachment_formats: Vec::new(),
            depth_attachment_format: None,
            stencil_attachment_format: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl PipelineRenderingCreateInfo {
    pub(crate) fn from_subpass(subpass: &Subpass) -> Self {
        let subpass_desc = subpass.subpass_desc();
        let rp_attachments = subpass.render_pass().attachments();

        Self {
            view_mask: subpass_desc.view_mask,
            color_attachment_formats: (subpass_desc.color_attachments.iter())
                .map(|color_attachment| {
                    color_attachment.as_ref().map(|color_attachment| {
                        rp_attachments[color_attachment.attachment as usize]
                            .format
                            .unwrap()
                    })
                })
                .collect(),
            depth_attachment_format: (subpass_desc.depth_stencil_attachment.as_ref())
                .map(|depth_stencil_attachment| {
                    rp_attachments[depth_stencil_attachment.attachment as usize]
                        .format
                        .unwrap()
                })
                .filter(|format| format.aspects().intersects(ImageAspects::DEPTH)),
            stencil_attachment_format: (subpass_desc.depth_stencil_attachment.as_ref())
                .map(|depth_stencil_attachment| {
                    rp_attachments[depth_stencil_attachment.attachment as usize]
                        .format
                        .unwrap()
                })
                .filter(|format| format.aspects().intersects(ImageAspects::STENCIL)),
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn from_rendering_info(info: &RenderingInfo) -> Self {
        Self {
            view_mask: info.view_mask,
            color_attachment_formats: (info.color_attachments.iter())
                .map(|atch_info| {
                    atch_info
                        .as_ref()
                        .map(|atch_info| atch_info.image_view.format().unwrap())
                })
                .collect(),
            depth_attachment_format: (info.depth_attachment.as_ref())
                .map(|atch_info| atch_info.image_view.format().unwrap()),
            stencil_attachment_format: (info.stencil_attachment.as_ref())
                .map(|atch_info| atch_info.image_view.format().unwrap()),
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn from_inheritance_rendering_info(
        info: &CommandBufferInheritanceRenderingInfo,
    ) -> Self {
        Self {
            view_mask: info.view_mask,
            color_attachment_formats: info.color_attachment_formats.clone(),
            depth_attachment_format: info.depth_attachment_format,
            stencil_attachment_format: info.stencil_attachment_format,
            _ne: crate::NonExhaustive(()),
        }
    }
}
