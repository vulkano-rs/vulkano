// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::pipeline::layout::PipelineLayoutCreationError;
use crate::pipeline::layout::PipelineLayoutSupersetError;
use crate::pipeline::shader::ShaderInterfaceMismatchError;
use crate::pipeline::vertex::IncompatibleVertexDefinitionError;
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

    /// The device doesn't support using the `multiview´ feature with geometry shaders.
    MultiviewGeometryShaderNotSupported,

    /// The device doesn't support using the `multiview´ feature with tessellation shaders.
    MultiviewTessellationShaderNotSupported,

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
            GraphicsPipelineCreationError::OomError(ref err) => Some(err),
            GraphicsPipelineCreationError::PipelineLayoutCreationError(ref err) => Some(err),
            GraphicsPipelineCreationError::IncompatiblePipelineLayout(ref err) => Some(err),
            GraphicsPipelineCreationError::ShaderStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::IncompatibleVertexDefinition(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for GraphicsPipelineCreationError {
    // TODO: finish
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            GraphicsPipelineCreationError::ExtensionNotEnabled { extension, reason } => {
                write!(
                    fmt,
                    "the extension {} must be enabled: {}",
                    extension, reason
                )
            }
            GraphicsPipelineCreationError::FeatureNotEnabled { feature, reason } => {
                write!(fmt, "the feature {} must be enabled: {}", feature, reason)
            }
            GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible => {
                write!(fmt, "the output of the fragment shader is not compatible with what the render pass subpass expects")
            }
            GraphicsPipelineCreationError::IncompatiblePipelineLayout(_) => {
                write!(
                    fmt,
                    "the pipeline layout is not compatible with what the shaders expect"
                )
            }
            GraphicsPipelineCreationError::IncompatibleSpecializationConstants => {
                write!(fmt, "the provided specialization constants are not compatible with what the shader expects")
            }
            GraphicsPipelineCreationError::IncompatibleVertexDefinition(_) => {
                write!(
                    fmt,
                    "the vertex definition is not compatible with the input of the vertex shader"
                )
            }
            GraphicsPipelineCreationError::InvalidPrimitiveTopology => {
                write!(fmt, "trying to use a patch list without a tessellation shader, or a non-patch-list with a tessellation shader")
            }
            GraphicsPipelineCreationError::InvalidNumPatchControlPoints => {
                write!(fmt, "patch_control_points was not greater than 0 and less than or equal to the max_tessellation_patch_size limit")
            }
            GraphicsPipelineCreationError::MaxDiscardRectanglesExceeded { .. } => {
                write!(
                    fmt,
                    "the maximum number of discard rectangles has been exceeded"
                )
            }
            GraphicsPipelineCreationError::MaxVertexAttribDivisorExceeded { .. } => {
                write!(
                    fmt,
                    "the maximum value for the instance rate divisor has been exceeded"
                )
            }
            GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded { .. } => {
                write!(
                    fmt,
                    "the maximum number of vertex attributes has been exceeded"
                )
            }
            GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded { .. } => {
                write!(
                    fmt,
                    "the maximum offset for a vertex attribute has been exceeded"
                )
            }
            GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded { .. } => {
                write!(
                    fmt,
                    "the maximum number of vertex sources has been exceeded"
                )
            }
            GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded { .. } => {
                write!(fmt, "the maximum stride value for vertex input (ie. the distance between two vertex elements) has been exceeded")
            }
            GraphicsPipelineCreationError::MaxViewportsExceeded { .. } => {
                write!(fmt, "the maximum number of viewports has been exceeded")
            }
            GraphicsPipelineCreationError::MaxViewportDimensionsExceeded => {
                write!(fmt, "the maximum dimensions of viewports has been exceeded")
            }
            GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount => {
                write!(fmt, "the number of attachments specified in the blending does not match the number of attachments in the subpass")
            }
            GraphicsPipelineCreationError::MultiviewGeometryShaderNotSupported => {
                write!(fmt, "the device doesn't support using the `multiview´ feature with geometry shaders")
            }
            GraphicsPipelineCreationError::MultiviewTessellationShaderNotSupported => {
                write!(fmt, "the device doesn't support using the `multiview´ feature with tessellation shaders")
            }
            GraphicsPipelineCreationError::NoDepthAttachment => {
                write!(
                    fmt,
                    "the depth attachment of the render pass does not match the depth test"
                )
            }
            GraphicsPipelineCreationError::NoStencilAttachment => {
                write!(
                    fmt,
                    "the stencil attachment of the render pass does not match the stencil test"
                )
            }
            GraphicsPipelineCreationError::OomError(_) => {
                write!(fmt, "not enough memory available")
            }
            GraphicsPipelineCreationError::PipelineLayoutCreationError(_) => {
                write!(fmt, "error while creating the pipeline layout object")
            }
            GraphicsPipelineCreationError::ShaderStagesMismatch(_) => {
                write!(fmt, "the output interface of one shader and the input interface of the next shader do not match")
            }
            GraphicsPipelineCreationError::StrictLinesNotSupported => {
                write!(fmt, "the strict_lines device property was false")
            }
            GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader => {
                write!(
                    fmt,
                    "the primitives topology does not match what the geometry shader expects"
                )
            }
            GraphicsPipelineCreationError::ViewportBoundsExceeded => {
                write!(
                    fmt,
                    "the minimum or maximum bounds of viewports have been exceeded"
                )
            }
            GraphicsPipelineCreationError::WrongShaderType => {
                write!(fmt, "the wrong type of shader has been passed")
            }
            GraphicsPipelineCreationError::WrongStencilState => {
                write!(fmt, "the requested stencil test is invalid")
            }
        }
    }
}

impl From<OomError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: OomError) -> GraphicsPipelineCreationError {
        GraphicsPipelineCreationError::OomError(err)
    }
}

impl From<PipelineLayoutCreationError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutCreationError) -> GraphicsPipelineCreationError {
        GraphicsPipelineCreationError::PipelineLayoutCreationError(err)
    }
}

impl From<PipelineLayoutSupersetError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutSupersetError) -> GraphicsPipelineCreationError {
        GraphicsPipelineCreationError::IncompatiblePipelineLayout(err)
    }
}

impl From<IncompatibleVertexDefinitionError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: IncompatibleVertexDefinitionError) -> GraphicsPipelineCreationError {
        GraphicsPipelineCreationError::IncompatibleVertexDefinition(err)
    }
}

impl From<Error> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: Error) -> GraphicsPipelineCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                GraphicsPipelineCreationError::OomError(OomError::from(err))
            }
            err @ Error::OutOfDeviceMemory => {
                GraphicsPipelineCreationError::OomError(OomError::from(err))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
