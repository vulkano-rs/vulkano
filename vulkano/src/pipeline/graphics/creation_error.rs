// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::format::Format;
use crate::format::NumericType;
use crate::pipeline::graphics::vertex_input::IncompatibleVertexDefinitionError;
use crate::pipeline::layout::PipelineLayoutCreationError;
use crate::pipeline::layout::PipelineLayoutSupersetError;
use crate::shader::ShaderInterfaceMismatchError;
use crate::Error;
use crate::OomError;
use std::error;
use std::fmt;
use std::u32;

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

    /// The depth test requires a depth attachment but render pass has no depth attachment, or
    /// depth writing is enabled and the depth attachment is read-only.
    NoDepthAttachment,

    /// The stencil test requires a stencil attachment but render pass has no stencil attachment, or
    /// stencil writing is enabled and the stencil attachment is read-only.
    NoStencilAttachment,

    /// Not enough memory.
    OomError(OomError),

    /// Error while creating the pipeline layout object.
    PipelineLayoutCreationError(PipelineLayoutCreationError),

    /// The output interface of one shader and the input interface of the next shader do not match.
    ShaderStagesMismatch(ShaderInterfaceMismatchError),

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

impl error::Error for GraphicsPipelineCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
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
    // TODO: finish
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::ExtensionNotEnabled { extension, reason } => write!(
                fmt,
                "the extension {} must be enabled: {}",
                extension, reason
            ),
            Self::FeatureNotEnabled { feature, reason } => write!(
                fmt,
                "the feature {} must be enabled: {}",
                feature, reason
            ),
            Self::FragmentShaderRenderPassIncompatible => write!(
                fmt,
                "the output of the fragment shader is not compatible with what the render pass subpass expects",
            ),
            Self::IncompatiblePipelineLayout(_) => write!(
                fmt,
                "the pipeline layout is not compatible with what the shaders expect",
            ),
            Self::IncompatibleSpecializationConstants => write!(
                fmt,
                "the provided specialization constants are not compatible with what the shader expects",
            ),
            Self::IncompatibleVertexDefinition(_) => write!(
                fmt,
                "the vertex definition is not compatible with the input of the vertex shader",
            ),
            Self::InvalidPrimitiveTopology => write!(
                fmt,
                "trying to use a patch list without a tessellation shader, or a non-patch-list with a tessellation shader",
            ),
            Self::InvalidNumPatchControlPoints => write!(
                fmt,
                "patch_control_points was not greater than 0 and less than or equal to the max_tessellation_patch_size limit",
            ),
            Self::MaxDiscardRectanglesExceeded { .. } => write!(
                fmt,
                "the maximum number of discard rectangles has been exceeded",
            ),
            Self::MaxVertexAttribDivisorExceeded { .. } => write!(
                fmt,
                "the maximum value for the instance rate divisor has been exceeded",
            ),
            Self::MaxVertexInputAttributesExceeded { .. } => write!(
                fmt,
                "the maximum number of vertex attributes has been exceeded",
            ),
            Self::MaxVertexInputAttributeOffsetExceeded { .. } => write!(
                fmt,
                "the maximum offset for a vertex attribute has been exceeded",
            ),
            Self::MaxVertexInputBindingsExceeded { .. } => write!(
                fmt,
                "the maximum number of vertex sources has been exceeded",
            ),
            Self::MaxVertexInputBindingStrideExceeded { .. } => write!(
                fmt,
                "the maximum stride value for vertex input (ie. the distance between two vertex elements) has been exceeded",
            ),
            Self::MaxViewportsExceeded { .. } => write!(
                fmt,
                "the maximum number of viewports has been exceeded",
            ),
            Self::MaxViewportDimensionsExceeded => write!(
                fmt,
                "the maximum dimensions of viewports has been exceeded",
            ),
            Self::MismatchBlendingAttachmentsCount => write!(
                fmt,
                "the number of attachments specified in the blending does not match the number of attachments in the subpass",
            ),
            Self::NoDepthAttachment => write!(
                fmt,
                "the depth attachment of the render pass does not match the depth test",
            ),
            Self::NoStencilAttachment => write!(
                fmt,
                "the stencil attachment of the render pass does not match the stencil test",
            ),
            Self::OomError(_) => write!(
                fmt,
                "not enough memory available",
            ),
            Self::PipelineLayoutCreationError(_) => write!(
                fmt,
                "error while creating the pipeline layout object",
            ),
            Self::ShaderStagesMismatch(_) => write!(
                fmt,
                "the output interface of one shader and the input interface of the next shader do not match",
            ),
            Self::StrictLinesNotSupported => write!(
                fmt,
                "the strict_lines device property was false",
            ),
            Self::TopologyNotMatchingGeometryShader => write!(
                fmt,
                "the primitives topology does not match what the geometry shader expects",
            ),
            Self::VertexInputAttributeIncompatibleFormat {
                location,
                shader_type,
                attribute_type,
            } => write!(
                fmt,
                "the type of the shader input variable at location {} ({:?}) is not compatible with the format of the corresponding vertex input attribute ({:?})",
                location, shader_type, attribute_type,
            ),
            Self::VertexInputAttributeInvalidBinding { location, binding } => write!(
                fmt,
                "the binding number {} specified by vertex input attribute location {} does not exist in the provided list of binding descriptions",
                binding, location,
            ),
            Self::VertexInputAttributeMissing { location } => write!(
                fmt,
                "the vertex shader expects an input variable at location {}, but no vertex input attribute exists for that location",
                location,
            ),
            Self::VertexInputAttributeUnsupportedFormat { location, format } => write!(
                fmt,
                "the format {:?} specified by vertex input attribute location {} is not supported for vertex buffers",
                format, location,
            ),
            Self::ViewportBoundsExceeded => write!(
                fmt,
                "the minimum or maximum bounds of viewports have been exceeded",
            ),
            Self::WrongShaderType => write!(
                fmt,
                "the wrong type of shader has been passed",
            ),
            Self::WrongStencilState => write!(
                fmt,
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

impl From<PipelineLayoutCreationError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutCreationError) -> GraphicsPipelineCreationError {
        Self::PipelineLayoutCreationError(err)
    }
}

impl From<PipelineLayoutSupersetError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutSupersetError) -> GraphicsPipelineCreationError {
        Self::IncompatiblePipelineLayout(err)
    }
}

impl From<IncompatibleVertexDefinitionError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: IncompatibleVertexDefinitionError) -> GraphicsPipelineCreationError {
        Self::IncompatibleVertexDefinition(err)
    }
}

impl From<Error> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: Error) -> GraphicsPipelineCreationError {
        match err {
            err @ Error::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
