// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::vertex_input::IncompatibleVertexDefinitionError;
use crate::{
    descriptor_set::layout::DescriptorSetLayoutCreationError,
    format::{Format, NumericType},
    pipeline::layout::{PipelineLayoutCreationError, PipelineLayoutSupersetError},
    shader::ShaderInterfaceMismatchError,
    OomError, VulkanError,
};
use std::{error::Error, fmt};

/// Error that can happen when creating a graphics pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphicsPipelineCreationError {
    /// A device extension that was required for a particular setting on the graphics pipeline was not enabled.
    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },

    /// A device feature that was required for a particular setting on the graphics pipeline was not enabled.
    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
    },

    /// A color attachment has a format that does not support blending.
    ColorAttachmentFormatBlendNotSupported { attachment_index: u32 },

    /// A color attachment has a format that does not support that usage.
    ColorAttachmentFormatUsageNotSupported { attachment_index: u32 },

    /// The depth attachment has a format that does not support that usage.
    DepthAttachmentFormatUsageNotSupported,

    /// The depth and stencil attachments have different formats.
    DepthStencilAttachmentFormatMismatch,

    /// The output of the fragment shader is not compatible with what the render pass subpass
    /// expects.
    FragmentShaderRenderPassIncompatible,

    /// The pipeline layout is not compatible with what the shaders expect.
    IncompatiblePipelineLayout(PipelineLayoutSupersetError),

    /// The provided specialization constants are not compatible with what the shader expects.
    IncompatibleSpecializationConstants,

    /// The vertex definition is not compatible with the input of the vertex shader.
    IncompatibleVertexDefinition(IncompatibleVertexDefinitionError),

    /// Tried to use a patch list without a tessellation shader, or a non-patch-list with a
    /// tessellation shader.
    InvalidPrimitiveTopology,

    /// `patch_control_points` was not greater than 0 and less than or equal to the `max_tessellation_patch_size` limit.
    InvalidNumPatchControlPoints,

    /// The maximum number of discard rectangles has been exceeded.
    MaxDiscardRectanglesExceeded {
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: u32,
    },

    /// The `max_multiview_view_count` limit has been exceeded.
    MaxMultiviewViewCountExceeded { view_count: u32, max: u32 },

    /// The maximum value for the instance rate divisor has been exceeded.
    MaxVertexAttribDivisorExceeded {
        /// Index of the faulty binding.
        binding: u32,
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: u32,
    },

    /// The maximum number of vertex attributes has been exceeded.
    MaxVertexInputAttributesExceeded {
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: usize,
    },

    /// The maximum offset for a vertex attribute has been exceeded. This means that your vertex
    /// struct is too large.
    MaxVertexInputAttributeOffsetExceeded {
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: u32,
    },

    /// The maximum number of vertex sources has been exceeded.
    MaxVertexInputBindingsExceeded {
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: u32,
    },

    /// The maximum stride value for vertex input (ie. the distance between two vertex elements)
    /// has been exceeded.
    MaxVertexInputBindingStrideExceeded {
        /// Index of the faulty binding.
        binding: u32,
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: u32,
    },

    /// The maximum number of viewports has been exceeded.
    MaxViewportsExceeded {
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: u32,
    },

    /// The maximum dimensions of viewports has been exceeded.
    MaxViewportDimensionsExceeded,

    /// The number of attachments specified in the blending does not match the number of
    /// attachments in the subpass.
    MismatchBlendingAttachmentsCount,

    /// The provided `rasterization_samples` does not match the number of samples of the render
    /// subpass.
    MultisampleRasterizationSamplesMismatch,

    /// The depth test requires a depth attachment but render pass has no depth attachment, or
    /// depth writing is enabled and the depth attachment is read-only.
    NoDepthAttachment,

    /// The stencil test requires a stencil attachment but render pass has no stencil attachment, or
    /// stencil writing is enabled and the stencil attachment is read-only.
    NoStencilAttachment,

    /// Not enough memory.
    OomError(OomError),

    /// Error while creating a descriptor set layout object.
    DescriptorSetLayoutCreationError(DescriptorSetLayoutCreationError),

    /// Error while creating the pipeline layout object.
    PipelineLayoutCreationError(PipelineLayoutCreationError),

    /// The output interface of one shader and the input interface of the next shader do not match.
    ShaderStagesMismatch(ShaderInterfaceMismatchError),

    /// The stencil attachment has a format that does not support that usage.
    StencilAttachmentFormatUsageNotSupported,

    /// The [`strict_lines`](crate::device::Properties::strict_lines) device property was `false`.
    StrictLinesNotSupported,

    /// The primitives topology does not match what the geometry shader expects.
    TopologyNotMatchingGeometryShader,

    /// The type of the shader input variable at the given location is not compatible with the
    /// format of the corresponding vertex input attribute.
    VertexInputAttributeIncompatibleFormat {
        location: u32,
        shader_type: NumericType,
        attribute_type: NumericType,
    },

    /// The binding number specified by a vertex input attribute does not exist in the provided list
    /// of binding descriptions.
    VertexInputAttributeInvalidBinding { location: u32, binding: u32 },

    /// The vertex shader expects an input variable at the given location, but no vertex input
    /// attribute exists for that location.
    VertexInputAttributeMissing { location: u32 },

    /// The format specified by a vertex input attribute is not supported for vertex buffers.
    VertexInputAttributeUnsupportedFormat { location: u32, format: Format },

    /// The minimum or maximum bounds of viewports have been exceeded.
    ViewportBoundsExceeded,

    /// The wrong type of shader has been passed.
    ///
    /// For example you passed a vertex shader as the fragment shader.
    WrongShaderType,

    /// The requested stencil test is invalid.
    WrongStencilState,
}

impl Error for GraphicsPipelineCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            Self::PipelineLayoutCreationError(ref err) => Some(err),
            Self::IncompatiblePipelineLayout(ref err) => Some(err),
            Self::ShaderStagesMismatch(ref err) => Some(err),
            Self::IncompatibleVertexDefinition(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for GraphicsPipelineCreationError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::ExtensionNotEnabled { extension, reason } => write!(
                f,
                "the extension {} must be enabled: {}",
                extension, reason
            ),
            Self::FeatureNotEnabled { feature, reason } => write!(
                f,
                "the feature {} must be enabled: {}",
                feature, reason
            ),
            Self::ColorAttachmentFormatBlendNotSupported { attachment_index } => write!(
                f,
                "color attachment {} has a format that does not support blending",
                attachment_index,
            ),
            Self::ColorAttachmentFormatUsageNotSupported { attachment_index } => write!(
                f,
                "color attachment {} has a format that does not support that usage",
                attachment_index,
            ),
            Self::DepthAttachmentFormatUsageNotSupported => write!(
                f,
                "the depth attachment has a format that does not support that usage",
            ),
            Self::DepthStencilAttachmentFormatMismatch => write!(
                f,
                "the depth and stencil attachments have different formats",
            ),
            Self::FragmentShaderRenderPassIncompatible => write!(
                f,
                "the output of the fragment shader is not compatible with what the render pass subpass expects",
            ),
            Self::IncompatiblePipelineLayout(_) => write!(
                f,
                "the pipeline layout is not compatible with what the shaders expect",
            ),
            Self::IncompatibleSpecializationConstants => write!(
                f,
                "the provided specialization constants are not compatible with what the shader expects",
            ),
            Self::IncompatibleVertexDefinition(_) => write!(
                f,
                "the vertex definition is not compatible with the input of the vertex shader",
            ),
            Self::InvalidPrimitiveTopology => write!(
                f,
                "trying to use a patch list without a tessellation shader, or a non-patch-list with a tessellation shader",
            ),
            Self::InvalidNumPatchControlPoints => write!(
                f,
                "patch_control_points was not greater than 0 and less than or equal to the max_tessellation_patch_size limit",
            ),
            Self::MaxDiscardRectanglesExceeded { .. } => write!(
                f,
                "the maximum number of discard rectangles has been exceeded",
            ),
            Self::MaxMultiviewViewCountExceeded { .. } => {
                write!(f, "the `max_multiview_view_count` limit has been exceeded",)
            },
            Self::MaxVertexAttribDivisorExceeded { .. } => write!(
                f,
                "the maximum value for the instance rate divisor has been exceeded",
            ),
            Self::MaxVertexInputAttributesExceeded { .. } => write!(
                f,
                "the maximum number of vertex attributes has been exceeded",
            ),
            Self::MaxVertexInputAttributeOffsetExceeded { .. } => write!(
                f,
                "the maximum offset for a vertex attribute has been exceeded",
            ),
            Self::MaxVertexInputBindingsExceeded { .. } => write!(
                f,
                "the maximum number of vertex sources has been exceeded",
            ),
            Self::MaxVertexInputBindingStrideExceeded { .. } => write!(
                f,
                "the maximum stride value for vertex input (ie. the distance between two vertex elements) has been exceeded",
            ),
            Self::MaxViewportsExceeded { .. } => write!(
                f,
                "the maximum number of viewports has been exceeded",
            ),
            Self::MaxViewportDimensionsExceeded => write!(
                f,
                "the maximum dimensions of viewports has been exceeded",
            ),
            Self::MismatchBlendingAttachmentsCount => write!(
                f,
                "the number of attachments specified in the blending does not match the number of attachments in the subpass",
            ),
            Self::MultisampleRasterizationSamplesMismatch => write!(
                f,
                "the provided `rasterization_samples` does not match the number of samples of the render subpass",
            ),
            Self::NoDepthAttachment => write!(
                f,
                "the depth attachment of the render pass does not match the depth test",
            ),
            Self::NoStencilAttachment => write!(
                f,
                "the stencil attachment of the render pass does not match the stencil test",
            ),
            Self::OomError(_) => write!(
                f,
                "not enough memory available",
            ),
            Self::DescriptorSetLayoutCreationError(_) => write!(
                f,
                "error while creating a descriptor set layout object",
            ),
            Self::PipelineLayoutCreationError(_) => write!(
                f,
                "error while creating the pipeline layout object",
            ),
            Self::ShaderStagesMismatch(_) => write!(
                f,
                "the output interface of one shader and the input interface of the next shader do not match",
            ),
            Self::StencilAttachmentFormatUsageNotSupported => write!(
                f,
                "the stencil attachment has a format that does not support that usage",
            ),
            Self::StrictLinesNotSupported => write!(
                f,
                "the strict_lines device property was false",
            ),
            Self::TopologyNotMatchingGeometryShader => write!(
                f,
                "the primitives topology does not match what the geometry shader expects",
            ),
            Self::VertexInputAttributeIncompatibleFormat {
                location,
                shader_type,
                attribute_type,
            } => write!(
                f,
                "the type of the shader input variable at location {} ({:?}) is not compatible with the format of the corresponding vertex input attribute ({:?})",
                location, shader_type, attribute_type,
            ),
            Self::VertexInputAttributeInvalidBinding { location, binding } => write!(
                f,
                "the binding number {} specified by vertex input attribute location {} does not exist in the provided list of binding descriptions",
                binding, location,
            ),
            Self::VertexInputAttributeMissing { location } => write!(
                f,
                "the vertex shader expects an input variable at location {}, but no vertex input attribute exists for that location",
                location,
            ),
            Self::VertexInputAttributeUnsupportedFormat { location, format } => write!(
                f,
                "the format {:?} specified by vertex input attribute location {} is not supported for vertex buffers",
                format, location,
            ),
            Self::ViewportBoundsExceeded => write!(
                f,
                "the minimum or maximum bounds of viewports have been exceeded",
            ),
            Self::WrongShaderType => write!(
                f,
                "the wrong type of shader has been passed",
            ),
            Self::WrongStencilState => write!(
                f,
                "the requested stencil test is invalid",
            ),
        }
    }
}

impl From<OomError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: OomError) -> GraphicsPipelineCreationError {
        Self::OomError(err)
    }
}

impl From<DescriptorSetLayoutCreationError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: DescriptorSetLayoutCreationError) -> Self {
        Self::DescriptorSetLayoutCreationError(err)
    }
}

impl From<PipelineLayoutCreationError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutCreationError) -> Self {
        Self::PipelineLayoutCreationError(err)
    }
}

impl From<PipelineLayoutSupersetError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutSupersetError) -> Self {
        Self::IncompatiblePipelineLayout(err)
    }
}

impl From<IncompatibleVertexDefinitionError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: IncompatibleVertexDefinitionError) -> Self {
        Self::IncompatibleVertexDefinition(err)
    }
}

impl From<VulkanError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: VulkanError) -> Self {
        match err {
            err @ VulkanError::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ VulkanError::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
