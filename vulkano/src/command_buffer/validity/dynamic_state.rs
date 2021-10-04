// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::command_buffer::synced::CommandBufferState;
use crate::pipeline::DynamicState;
use crate::pipeline::DynamicStateMode;
use crate::pipeline::GraphicsPipeline;
use std::error;
use std::fmt;

/// Checks whether states that are about to be set are correct.
pub(in super::super) fn check_dynamic_state_validity(
    current_state: CommandBufferState,
    pipeline: &GraphicsPipeline,
) -> Result<(), CheckDynamicStateValidityError> {
    let device = pipeline.device();

    for state in pipeline.dynamic_states().filter_map(|(state, mode)| {
        if matches!(mode, DynamicStateMode::Dynamic) {
            Some(state)
        } else {
            None
        }
    }) {
        match state {
            DynamicState::BlendConstants => {
                if current_state.blend_constants().is_none() {
                    return Err(CheckDynamicStateValidityError::BlendConstantsNotSet);
                }
            }
            DynamicState::ColorWriteEnable => todo!(),
            DynamicState::CullMode => todo!(),
            DynamicState::DepthBias => todo!(),
            DynamicState::DepthBiasEnable => todo!(),
            DynamicState::DepthBounds => {
                if current_state.depth_bounds().is_none() {
                    return Err(CheckDynamicStateValidityError::DepthBoundsNotSet);
                }
            }
            DynamicState::DepthBoundsTestEnable => todo!(),
            DynamicState::DepthCompareOp => todo!(),
            DynamicState::DepthTestEnable => todo!(),
            DynamicState::DepthWriteEnable => todo!(),
            DynamicState::DiscardRectangle => todo!(),
            DynamicState::ExclusiveScissor => todo!(),
            DynamicState::FragmentShadingRate => todo!(),
            DynamicState::FrontFace => todo!(),
            DynamicState::LineStipple => todo!(),
            DynamicState::LineWidth => {
                if current_state.line_width().is_none() {
                    return Err(CheckDynamicStateValidityError::LineWidthNotSet);
                }
            }
            DynamicState::LogicOp => todo!(),
            DynamicState::PatchControlPoints => {
                if current_state.patch_control_points().is_none() {
                    return Err(CheckDynamicStateValidityError::PatchControlPointsNotSet);
                }
            }
            DynamicState::PrimitiveRestartEnable => {
                // TODO: does this have the same restrictions as fixed values at pipeline creation?

                if current_state.primitive_restart_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::PrimitiveRestartEnableNotSet);
                }
            }
            DynamicState::PrimitiveTopology => {
                // TODO: does this have the same restrictions as fixed values at pipeline creation?

                if current_state.primitive_topology().is_none() {
                    return Err(CheckDynamicStateValidityError::PrimitiveTopologyNotSet);
                }
            }
            DynamicState::RasterizerDiscardEnable => todo!(),
            DynamicState::RayTracingPipelineStackSize => unreachable!(
                "RayTracingPipelineStackSize dynamic state should not occur on a graphics pipeline"
            ),
            DynamicState::SampleLocations => todo!(),
            DynamicState::Scissor => {
                for num in 0..pipeline.num_viewports() {
                    if current_state.scissor(num).is_none() {
                        return Err(CheckDynamicStateValidityError::ScissorNotSet { num });
                    }
                }
            }
            DynamicState::ScissorWithCount => todo!(),
            DynamicState::StencilCompareMask => {
                let state = current_state.stencil_compare_mask();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::StencilCompareMaskNotSet);
                }
            }
            DynamicState::StencilOp => todo!(),
            DynamicState::StencilReference => {
                let state = current_state.stencil_reference();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::StencilReferenceNotSet);
                }
            }
            DynamicState::StencilTestEnable => todo!(),
            DynamicState::StencilWriteMask => {
                let state = current_state.stencil_write_mask();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::StencilWriteMaskNotSet);
                }
            }
            DynamicState::VertexInput => todo!(),
            DynamicState::VertexInputBindingStride => todo!(),
            DynamicState::Viewport => {
                for num in 0..pipeline.num_viewports() {
                    if current_state.viewport(num).is_none() {
                        return Err(CheckDynamicStateValidityError::ViewportNotSet { num });
                    }
                }
            }
            DynamicState::ViewportCoarseSampleOrder => todo!(),
            DynamicState::ViewportShadingRatePalette => todo!(),
            DynamicState::ViewportWithCount => todo!(),
            DynamicState::ViewportWScaling => todo!(),
        }
    }

    Ok(())
}

/// Error that can happen when validating dynamic states.
#[derive(Debug, Copy, Clone)]
pub enum CheckDynamicStateValidityError {
    /// The pipeline has dynamic blend constants, but no value was set.
    BlendConstantsNotSet,
    /// The pipeline has dynamic depth bounds, but no value was set.
    DepthBoundsNotSet,
    /// The pipeline has a dynamic line width, but no value was set.
    LineWidthNotSet,
    /// The pipeline has a dynamic number of patch control points, but no value was set.
    PatchControlPointsNotSet,
    /// The pipeline has dynamic primitive restart enable, but no value was set.
    PrimitiveRestartEnableNotSet,
    /// The pipeline has a dynamic primitive topology, but no value was set.
    PrimitiveTopologyNotSet,
    /// The pipeline has a dynamic scissor, but the scissor for a slot used by the pipeline was not set.
    ScissorNotSet { num: u32 },
    /// The pipeline has dynamic stencil compare mask, but no value was set for the front or back face.
    StencilCompareMaskNotSet,
    /// The pipeline has dynamic stencil reference, but no value was set for the front or back face.
    StencilReferenceNotSet,
    /// The pipeline has dynamic stencil write mask, but no value was set for the front or back face.
    StencilWriteMaskNotSet,
    /// The pipeline has a dynamic viewport, but the viewport for a slot used by the pipeline was not set.
    ViewportNotSet { num: u32 },
}

impl error::Error for CheckDynamicStateValidityError {}

impl fmt::Display for CheckDynamicStateValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckDynamicStateValidityError::BlendConstantsNotSet => {
                    "the pipeline has dynamic blend constants, but no value was set"
                }
                CheckDynamicStateValidityError::DepthBoundsNotSet => {
                    "the pipeline has dynamic depth bounds, but no value was set"
                }
                CheckDynamicStateValidityError::LineWidthNotSet => {
                    "the pipeline has a dynamic line width, but no value was set"
                }
                CheckDynamicStateValidityError::PatchControlPointsNotSet => {
                    "the pipeline has a dynamic number of patch control points, but no value was set"
                }
                CheckDynamicStateValidityError::PrimitiveRestartEnableNotSet => {
                    "the pipeline has dynamic primitive restart enable, but no value was set"
                }
                CheckDynamicStateValidityError::PrimitiveTopologyNotSet => {
                    "the pipeline has a dynamic primitive topology, but no value was set"
                }
                CheckDynamicStateValidityError::ScissorNotSet { .. } => {
                    "The pipeline has a dynamic scissor, but the scissor for a slot used by the pipeline was not set"
                }
                CheckDynamicStateValidityError::StencilCompareMaskNotSet => {
                    "the pipeline has dynamic stencil compare mask, but no value was set for the front or back face"
                }
                CheckDynamicStateValidityError::StencilReferenceNotSet => {
                    "the pipeline has dynamic stencil reference, but no value was set for the front or back face"
                }
                CheckDynamicStateValidityError::StencilWriteMaskNotSet => {
                    "the pipeline has dynamic stencil write mask, but no value was set for the front or back face"
                }
                CheckDynamicStateValidityError::ViewportNotSet { .. } => {
                    "the pipeline has a dynamic viewport, but the viewport for a slot used by the pipeline was not set"
                }
            }
        )
    }
}
