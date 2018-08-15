// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::u32;

use Error;
use OomError;
use descriptor::pipeline_layout::PipelineLayoutNotSupersetError;
use pipeline::input_assembly::PrimitiveTopology;
use pipeline::shader::ShaderInterfaceMismatchError;
use pipeline::vertex::IncompatibleVertexDefinitionError;

/// Error that can happen when creating a graphics pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphicsPipelineCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// The pipeline layout is not compatible with what the shaders expect.
    IncompatiblePipelineLayout(PipelineLayoutNotSupersetError),

    /// The interface between the vertex shader and the geometry shader mismatches.
    VertexGeometryStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the vertex shader and the tessellation control shader mismatches.
    VertexTessControlStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the vertex shader and the fragment shader mismatches.
    VertexFragmentStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the tessellation control shader and the tessellation evaluation
    /// shader mismatches.
    TessControlTessEvalStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the tessellation evaluation shader and the geometry shader
    /// mismatches.
    TessEvalGeometryStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the tessellation evaluation shader and the fragment shader
    /// mismatches.
    TessEvalFragmentStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the geometry shader and the fragment shader mismatches.
    GeometryFragmentStagesMismatch(ShaderInterfaceMismatchError),

    /// The output of the fragment shader is not compatible with what the render pass subpass
    /// expects.
    FragmentShaderRenderPassIncompatible,

    /// The vertex definition is not compatible with the input of the vertex shader.
    IncompatibleVertexDefinition(IncompatibleVertexDefinitionError),

    /// The maximum stride value for vertex input (ie. the distance between two vertex elements)
    /// has been exceeded.
    MaxVertexInputBindingStrideExceeded {
        /// Index of the faulty binding.
        binding: usize,
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The maximum number of vertex sources has been exceeded.
    MaxVertexInputBindingsExceeded {
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The maximum offset for a vertex attribute has been exceeded. This means that your vertex
    /// struct is too large.
    MaxVertexInputAttributeOffsetExceeded {
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The maximum number of vertex attributes has been exceeded.
    MaxVertexInputAttributesExceeded {
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The user requested to use primitive restart, but the primitive topology doesn't support it.
    PrimitiveDoesntSupportPrimitiveRestart {
        /// The topology that doesn't support primitive restart.
        primitive: PrimitiveTopology,
    },

    /// The `multi_viewport` feature must be enabled in order to use multiple viewports at once.
    MultiViewportFeatureNotEnabled,

    /// The maximum number of viewports has been exceeded.
    MaxViewportsExceeded {
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: u32,
    },

    /// The maximum dimensions of viewports has been exceeded.
    MaxViewportDimensionsExceeded,

    /// The minimum or maximum bounds of viewports have been exceeded.
    ViewportBoundsExceeded,

    /// The `wide_lines` feature must be enabled in order to use a line width greater than 1.0.
    WideLinesFeatureNotEnabled,

    /// The `depth_clamp` feature must be enabled in order to use depth clamping.
    DepthClampFeatureNotEnabled,

    /// The `depth_bias_clamp` feature must be enabled in order to use a depth bias clamp different
    /// from 0.0.
    DepthBiasClampFeatureNotEnabled,

    /// The `fill_mode_non_solid` feature must be enabled in order to use a polygon mode different
    /// from `Fill`.
    FillModeNonSolidFeatureNotEnabled,

    /// The `depth_bounds` feature must be enabled in order to use depth bounds testing.
    DepthBoundsFeatureNotEnabled,

    /// The requested stencil test is invalid.
    WrongStencilState,

    /// The primitives topology does not match what the geometry shader expects.
    TopologyNotMatchingGeometryShader,

    /// The `geometry_shader` feature must be enabled in order to use geometry shaders.
    GeometryShaderFeatureNotEnabled,

    /// The `tessellation_shader` feature must be enabled in order to use tessellation shaders.
    TessellationShaderFeatureNotEnabled,

    /// The number of attachments specified in the blending does not match the number of
    /// attachments in the subpass.
    MismatchBlendingAttachmentsCount,

    /// The `independent_blend` feature must be enabled in order to use different blending
    /// operations per attachment.
    IndependentBlendFeatureNotEnabled,

    /// The `logic_op` feature must be enabled in order to use logic operations.
    LogicOpFeatureNotEnabled,

    /// The depth test requires a depth attachment but render pass has no depth attachment, or
    /// depth writing is enabled and the depth attachment is read-only.
    NoDepthAttachment,

    /// The stencil test requires a stencil attachment but render pass has no stencil attachment, or
    /// stencil writing is enabled and the stencil attachment is read-only.
    NoStencilAttachment,

    /// Tried to use a patch list without a tessellation shader, or a non-patch-list with a
    /// tessellation shader.
    InvalidPrimitiveTopology,

    /// The `maxTessellationPatchSize` limit was exceeded.
    MaxTessellationPatchSizeExceeded,

    /// The wrong type of shader has been passed.
    ///
    /// For example you passed a vertex shader as the fragment shader.
    WrongShaderType,

    /// The `sample_rate_shading` feature must be enabled in order to use sample shading.
    SampleRateShadingFeatureNotEnabled,

    /// The `alpha_to_one` feature must be enabled in order to use alpha-to-one.
    AlphaToOneFeatureNotEnabled,
}

impl error::Error for GraphicsPipelineCreationError {
    #[inline]
    // TODO: finish
    fn description(&self) -> &str {
        match *self {
            GraphicsPipelineCreationError::OomError(_) => "not enough memory available",
            GraphicsPipelineCreationError::VertexGeometryStagesMismatch(_) => {
                "the interface between the vertex shader and the geometry shader mismatches"
            },
            GraphicsPipelineCreationError::VertexTessControlStagesMismatch(_) => {
                "the interface between the vertex shader and the tessellation control shader \
                 mismatches"
            },
            GraphicsPipelineCreationError::VertexFragmentStagesMismatch(_) => {
                "the interface between the vertex shader and the fragment shader mismatches"
            },
            GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(_) => {
                "the interface between the tessellation control shader and the tessellation \
                 evaluation shader mismatches"
            },
            GraphicsPipelineCreationError::TessEvalGeometryStagesMismatch(_) => {
                "the interface between the tessellation evaluation shader and the geometry \
                 shader mismatches"
            },
            GraphicsPipelineCreationError::TessEvalFragmentStagesMismatch(_) => {
                "the interface between the tessellation evaluation shader and the fragment \
                 shader mismatches"
            },
            GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(_) => {
                "the interface between the geometry shader and the fragment shader mismatches"
            },
            GraphicsPipelineCreationError::IncompatiblePipelineLayout(_) => {
                "the pipeline layout is not compatible with what the shaders expect"
            },
            GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible => {
                "the output of the fragment shader is not compatible with what the render pass \
                 subpass expects"
            },
            GraphicsPipelineCreationError::IncompatibleVertexDefinition(_) => {
                "the vertex definition is not compatible with the input of the vertex shader"
            },
            GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded { .. } => {
                "the maximum stride value for vertex input (ie. the distance between two vertex \
                 elements) has been exceeded"
            },
            GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded { .. } => {
                "the maximum number of vertex sources has been exceeded"
            },
            GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded { .. } => {
                "the maximum offset for a vertex attribute has been exceeded"
            },
            GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded { .. } => {
                "the maximum number of vertex attributes has been exceeded"
            },
            GraphicsPipelineCreationError::PrimitiveDoesntSupportPrimitiveRestart { .. } => {
                "the user requested to use primitive restart, but the primitive topology \
                 doesn't support it"
            },
            GraphicsPipelineCreationError::MultiViewportFeatureNotEnabled => {
                "the `multi_viewport` feature must be enabled in order to use multiple viewports \
                 at once"
            },
            GraphicsPipelineCreationError::MaxViewportsExceeded { .. } => {
                "the maximum number of viewports has been exceeded"
            },
            GraphicsPipelineCreationError::MaxViewportDimensionsExceeded => {
                "the maximum dimensions of viewports has been exceeded"
            },
            GraphicsPipelineCreationError::ViewportBoundsExceeded => {
                "the minimum or maximum bounds of viewports have been exceeded"
            },
            GraphicsPipelineCreationError::WideLinesFeatureNotEnabled => {
                "the `wide_lines` feature must be enabled in order to use a line width \
                 greater than 1.0"
            },
            GraphicsPipelineCreationError::DepthClampFeatureNotEnabled => {
                "the `depth_clamp` feature must be enabled in order to use depth clamping"
            },
            GraphicsPipelineCreationError::DepthBiasClampFeatureNotEnabled => {
                "the `depth_bias_clamp` feature must be enabled in order to use a depth bias \
                 clamp different from 0.0."
            },
            GraphicsPipelineCreationError::FillModeNonSolidFeatureNotEnabled => {
                "the `fill_mode_non_solid` feature must be enabled in order to use a polygon mode \
                 different from `Fill`"
            },
            GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled => {
                "the `depth_bounds` feature must be enabled in order to use depth bounds testing"
            },
            GraphicsPipelineCreationError::WrongStencilState => {
                "the requested stencil test is invalid"
            },
            GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader => {
                "the primitives topology does not match what the geometry shader expects"
            },
            GraphicsPipelineCreationError::GeometryShaderFeatureNotEnabled => {
                "the `geometry_shader` feature must be enabled in order to use geometry shaders"
            },
            GraphicsPipelineCreationError::TessellationShaderFeatureNotEnabled => {
                "the `tessellation_shader` feature must be enabled in order to use tessellation \
                 shaders"
            },
            GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount => {
                "the number of attachments specified in the blending does not match the number of \
                 attachments in the subpass"
            },
            GraphicsPipelineCreationError::IndependentBlendFeatureNotEnabled => {
                "the `independent_blend` feature must be enabled in order to use different \
                 blending operations per attachment"
            },
            GraphicsPipelineCreationError::LogicOpFeatureNotEnabled => {
                "the `logic_op` feature must be enabled in order to use logic operations"
            },
            GraphicsPipelineCreationError::NoDepthAttachment => {
                "the depth attachment of the render pass does not match the depth test"
            },
            GraphicsPipelineCreationError::NoStencilAttachment => {
                "the stencil attachment of the render pass does not match the stencil test"
            },
            GraphicsPipelineCreationError::InvalidPrimitiveTopology => {
                "trying to use a patch list without a tessellation shader, or a non-patch-list \
                 with a tessellation shader"
            },
            GraphicsPipelineCreationError::MaxTessellationPatchSizeExceeded => {
                "the maximum tessellation patch size was exceeded"
            },
            GraphicsPipelineCreationError::WrongShaderType => {
                "the wrong type of shader has been passed"
            },
            GraphicsPipelineCreationError::SampleRateShadingFeatureNotEnabled => {
                "the `sample_rate_shading` feature must be enabled in order to use sample shading"
            },
            GraphicsPipelineCreationError::AlphaToOneFeatureNotEnabled => {
                "the `alpha_to_one` feature must be enabled in order to use alpha-to-one"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            GraphicsPipelineCreationError::OomError(ref err) => Some(err),
            GraphicsPipelineCreationError::IncompatiblePipelineLayout(ref err) => Some(err),
            GraphicsPipelineCreationError::VertexGeometryStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::VertexTessControlStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::VertexFragmentStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::TessEvalGeometryStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::TessEvalFragmentStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::IncompatibleVertexDefinition(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for GraphicsPipelineCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: OomError) -> GraphicsPipelineCreationError {
        GraphicsPipelineCreationError::OomError(err)
    }
}

impl From<PipelineLayoutNotSupersetError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutNotSupersetError) -> GraphicsPipelineCreationError {
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
            },
            err @ Error::OutOfDeviceMemory => {
                GraphicsPipelineCreationError::OomError(OomError::from(err))
            },
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
