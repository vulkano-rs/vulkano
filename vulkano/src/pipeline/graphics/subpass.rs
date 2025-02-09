use crate::{
    command_buffer::{CommandBufferInheritanceRenderingInfo, RenderingInfo},
    device::Device,
    format::{Format, FormatFeatures},
    image::ImageAspects,
    render_pass::Subpass,
    Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
use ash::vk;
use smallvec::SmallVec;

/// Selects the type of subpass that a graphics pipeline is created for.
#[derive(Clone, Debug)]
pub enum PipelineSubpassType {
    BeginRenderPass(Subpass),
    BeginRendering(PipelineRenderingCreateInfo),
}

impl PipelineSubpassType {
    pub(crate) fn to_vk_rendering<'a>(
        &self,
        fields1_vk: &'a PipelineRenderingCreateInfoFields1Vk,
    ) -> vk::PipelineRenderingCreateInfo<'a> {
        match self {
            PipelineSubpassType::BeginRenderPass(_) => unreachable!(),
            PipelineSubpassType::BeginRendering(rendering_info) => rendering_info.to_vk(fields1_vk),
        }
    }

    pub(crate) fn to_vk_rendering_fields1(&self) -> Option<PipelineRenderingCreateInfoFields1Vk> {
        match self {
            PipelineSubpassType::BeginRenderPass(_) => None,
            PipelineSubpassType::BeginRendering(rendering_info) => {
                Some(rendering_info.to_vk_fields1())
            }
        }
    }
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
    /// If set to a nonzero value, the [`multiview`](crate::device::DeviceFeatures::multiview)
    /// feature must be enabled on the device.
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
        Self::new()
    }
}

impl PipelineRenderingCreateInfo {
    /// Returns a default `PipelineRenderingCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            view_mask: 0,
            color_attachment_formats: Vec::new(),
            depth_attachment_format: None,
            stencil_attachment_format: None,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn from_subpass(subpass: &Subpass) -> Self {
        let subpass_desc = subpass.subpass_desc();
        let rp_attachments = subpass.render_pass().attachments();

        Self {
            view_mask: subpass_desc.view_mask,
            color_attachment_formats: subpass_desc
                .color_attachments
                .iter()
                .map(|color_attachment| {
                    color_attachment.as_ref().map(|color_attachment| {
                        rp_attachments[color_attachment.attachment as usize].format
                    })
                })
                .collect(),
            depth_attachment_format: subpass_desc
                .depth_stencil_attachment
                .as_ref()
                .map(|depth_stencil_attachment| {
                    rp_attachments[depth_stencil_attachment.attachment as usize].format
                })
                .filter(|format| format.aspects().intersects(ImageAspects::DEPTH)),
            stencil_attachment_format: subpass_desc
                .depth_stencil_attachment
                .as_ref()
                .map(|depth_stencil_attachment| {
                    rp_attachments[depth_stencil_attachment.attachment as usize].format
                })
                .filter(|format| format.aspects().intersects(ImageAspects::STENCIL)),
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn from_rendering_info(info: &RenderingInfo) -> Self {
        Self {
            view_mask: info.view_mask,
            color_attachment_formats: info
                .color_attachments
                .iter()
                .map(|atch_info| {
                    atch_info
                        .as_ref()
                        .map(|atch_info| atch_info.image_view.format())
                })
                .collect(),
            depth_attachment_format: info
                .depth_attachment
                .as_ref()
                .map(|atch_info| atch_info.image_view.format()),
            stencil_attachment_format: info
                .stencil_attachment
                .as_ref()
                .map(|atch_info| atch_info.image_view.format()),
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
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "multiview",
                )])]),
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

            format.validate_device(device).map_err(|err| {
                err.add_context(format!("color_attachment_formats[{}]", attachment_index))
                    .set_vuids(&["VUID-VkGraphicsPipelineCreateInfo-renderPass-06580"])
            })?;

            if format == Format::UNDEFINED {
                return Err(Box::new(ValidationError {
                    context: format!("color_attachment_formats[{}]", attachment_index).into(),
                    problem: "is `Format::UNDEFINED`".into(),
                    ..Default::default()
                }));
            }

            let format_properties =
                unsafe { device.physical_device().format_properties_unchecked(format) };

            if !format_properties
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
            format.validate_device(device).map_err(|err| {
                err.add_context("depth_attachment_format")
                    .set_vuids(&["VUID-VkGraphicsPipelineCreateInfo-renderPass-06583"])
            })?;

            if format == Format::UNDEFINED {
                return Err(Box::new(ValidationError {
                    context: "depth_attachment_format".into(),
                    problem: "is `Format::UNDEFINED`".into(),
                    ..Default::default()
                }));
            }

            let format_properties =
                unsafe { device.physical_device().format_properties_unchecked(format) };

            if !format_properties
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
            format.validate_device(device).map_err(|err| {
                err.add_context("stencil_attachment_format")
                    .set_vuids(&["VUID-VkGraphicsPipelineCreateInfo-renderPass-06584"])
            })?;

            if format == Format::UNDEFINED {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment_format".into(),
                    problem: "is `Format::UNDEFINED`".into(),
                    ..Default::default()
                }));
            }

            let format_properties =
                unsafe { device.physical_device().format_properties_unchecked(format) };

            if !format_properties
                .potential_format_features()
                .intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment_format".into(),
                    problem: "format features do not contain \
                        `FormatFeature::DEPTH_STENCIL_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06586"],
                    ..Default::default()
                }));
            }

            if !format.aspects().intersects(ImageAspects::STENCIL) {
                return Err(Box::new(ValidationError {
                    context: "stencil_attachment_format".into(),
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

    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a PipelineRenderingCreateInfoFields1Vk,
    ) -> vk::PipelineRenderingCreateInfo<'a> {
        let &Self {
            view_mask,
            color_attachment_formats: _,
            depth_attachment_format,
            stencil_attachment_format,
            _ne: _,
        } = self;
        let PipelineRenderingCreateInfoFields1Vk {
            color_attachment_formats_vk,
        } = fields1_vk;

        vk::PipelineRenderingCreateInfo::default()
            .view_mask(view_mask)
            .color_attachment_formats(color_attachment_formats_vk)
            .depth_attachment_format(
                depth_attachment_format.map_or(vk::Format::UNDEFINED, Into::into),
            )
            .stencil_attachment_format(
                stencil_attachment_format.map_or(vk::Format::UNDEFINED, Into::into),
            )
    }

    pub(crate) fn to_vk_fields1(&self) -> PipelineRenderingCreateInfoFields1Vk {
        let color_attachment_formats_vk = self
            .color_attachment_formats
            .iter()
            .map(|format| format.map_or(vk::Format::UNDEFINED, Into::into))
            .collect();

        PipelineRenderingCreateInfoFields1Vk {
            color_attachment_formats_vk,
        }
    }
}

pub(crate) struct PipelineRenderingCreateInfoFields1Vk {
    pub(crate) color_attachment_formats_vk: SmallVec<[vk::Format; 4]>,
}
