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
    device::Device,
    format::{Format, FormatFeatures},
    image::ImageAspects,
    render_pass::Subpass,
    Requires, RequiresAllOf, RequiresOneOf, ValidationError,
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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            view_mask,
            ref color_attachment_formats,
            depth_attachment_format,
            stencil_attachment_format,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        if view_mask != 0 && !device.enabled_features().multiview {
            return Err(Box::new(ValidationError {
                context: "view_mask".into(),
                problem: "is not zero".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature("multiview")])]),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-multiview-06577"],
            }));
        }

        let view_count = u32::BITS - view_mask.leading_zeros();

        if view_count > properties.max_multiview_view_count.unwrap_or(0) {
            return Err(Box::new(ValidationError {
                context: "view_mask".into(),
                problem: "the number of views exceeds the \
                    `max_multiview_view_count` limit"
                    .into(),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06578"],
                ..Default::default()
            }));
        }

        for (attachment_index, format) in color_attachment_formats
            .iter()
            .enumerate()
            .flat_map(|(i, f)| f.map(|f| (i, f)))
        {
            let attachment_index = attachment_index as u32;

            format
                .validate_device(device)
                .map_err(|err| ValidationError {
                    context: "format".into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06580"],
                    ..ValidationError::from_requirement(err)
                })?;

            if !unsafe { device.physical_device().format_properties_unchecked(format) }
                .potential_format_features()
                .intersects(FormatFeatures::COLOR_ATTACHMENT)
            {
                return Err(Box::new(ValidationError {
                    context: format!("color_attachment_formats[{}]", attachment_index).into(),
                    problem: "format features do not contain \
                        `FormatFeature::COLOR_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06582"],
                    ..Default::default()
                }));
            }
        }

        if let Some(format) = depth_attachment_format {
            format
                .validate_device(device)
                .map_err(|err| ValidationError {
                    context: "format".into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06583"],
                    ..ValidationError::from_requirement(err)
                })?;

            if !unsafe { device.physical_device().format_properties_unchecked(format) }
                .potential_format_features()
                .intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment_format".into(),
                    problem: "format features do not contain \
                        `FormatFeature::DEPTH_STENCIL_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06585"],
                    ..Default::default()
                }));
            }

            if !format.aspects().intersects(ImageAspects::DEPTH) {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment_format".into(),
                    problem: "does not have a depth aspect".into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06587"],
                    ..Default::default()
                }));
            }
        }

        if let Some(format) = stencil_attachment_format {
            format
                .validate_device(device)
                .map_err(|err| ValidationError {
                    context: "format".into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06584"],
                    ..ValidationError::from_requirement(err)
                })?;

            if !unsafe { device.physical_device().format_properties_unchecked(format) }
                .potential_format_features()
                .intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(Box::new(ValidationError {
                    context: "render_pass.stencil_attachment_format".into(),
                    problem: "format features do not contain \
                        `FormatFeature::DEPTH_STENCIL_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06586"],
                    ..Default::default()
                }));
            }

            if !format.aspects().intersects(ImageAspects::STENCIL) {
                return Err(Box::new(ValidationError {
                    context: "render_pass.stencil_attachment_format".into(),
                    problem: "does not have a stencil aspect".into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06588"],
                    ..Default::default()
                }));
            }
        }

        if let (Some(depth_format), Some(stencil_format)) =
            (depth_attachment_format, stencil_attachment_format)
        {
            if depth_format != stencil_format {
                return Err(Box::new(ValidationError {
                    problem: "`depth_attachment_format` and `stencil_attachment_format` are both \
                        `Some`, but are not equal"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06589"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }
}
