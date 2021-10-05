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
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::BlendConstants,
                    });
                }
            }
            DynamicState::ColorWriteEnable => todo!(),
            DynamicState::CullMode => {
                if current_state.cull_mode().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::CullMode,
                    });
                }
            }
            DynamicState::DepthBias => {
                if current_state.depth_bias().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::DepthBias,
                    });
                }
            }
            DynamicState::DepthBiasEnable => {
                if current_state.depth_bias_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::DepthBiasEnable,
                    });
                }
            }
            DynamicState::DepthBounds => {
                if current_state.depth_bounds().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::DepthBounds,
                    });
                }
            }
            DynamicState::DepthBoundsTestEnable => {
                if current_state.depth_bounds_test_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::DepthBoundsTestEnable,
                    });
                }
            }
            DynamicState::DepthCompareOp => {
                if current_state.depth_compare_op().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::DepthCompareOp,
                    });
                }
            }
            DynamicState::DepthTestEnable => {
                if current_state.depth_test_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::DepthTestEnable,
                    });
                }
            }
            DynamicState::DepthWriteEnable => {
                if current_state.depth_write_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::DepthWriteEnable,
                    });
                }

                // TODO: Check if the depth buffer is writable
            }
            DynamicState::DiscardRectangle => todo!(),
            DynamicState::ExclusiveScissor => todo!(),
            DynamicState::FragmentShadingRate => todo!(),
            DynamicState::FrontFace => {
                if current_state.front_face().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::FrontFace,
                    });
                }
            }
            DynamicState::LineStipple => todo!(),
            DynamicState::LineWidth => {
                if current_state.line_width().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::LineWidth,
                    });
                }
            }
            DynamicState::LogicOp => todo!(),
            DynamicState::PatchControlPoints => {
                if current_state.patch_control_points().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::PatchControlPoints,
                    });
                }
            }
            DynamicState::PrimitiveRestartEnable => {
                // TODO: does this have the same restrictions as fixed values at pipeline creation?

                if current_state.primitive_restart_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::PrimitiveRestartEnable,
                    });
                }
            }
            DynamicState::PrimitiveTopology => {
                // TODO: does this have the same restrictions as fixed values at pipeline creation?

                if current_state.primitive_topology().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::PrimitiveTopology,
                    });
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
                        return Err(CheckDynamicStateValidityError::NotSet {
                            dynamic_state: DynamicState::Scissor,
                        });
                    }
                }
            }
            DynamicState::ScissorWithCount => todo!(),
            DynamicState::StencilCompareMask => {
                let state = current_state.stencil_compare_mask();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::StencilCompareMask,
                    });
                }
            }
            DynamicState::StencilOp => {
                let state = current_state.stencil_op();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::StencilOp,
                    });
                }
            }
            DynamicState::StencilReference => {
                let state = current_state.stencil_reference();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::StencilReference,
                    });
                }
            }
            DynamicState::StencilTestEnable => {
                if current_state.stencil_test_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::StencilTestEnable,
                    });
                }

                // TODO: Check if the stencil buffer is writable
            }
            DynamicState::StencilWriteMask => {
                let state = current_state.stencil_write_mask();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::StencilWriteMask,
                    });
                }
            }
            DynamicState::VertexInput => todo!(),
            DynamicState::VertexInputBindingStride => todo!(),
            DynamicState::Viewport => {
                for num in 0..pipeline.num_viewports() {
                    if current_state.viewport(num).is_none() {
                        return Err(CheckDynamicStateValidityError::NotSet {
                            dynamic_state: DynamicState::Viewport,
                        });
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
    /// The pipeline requires a particular state to be set dynamically, but the value was not or
    /// only partially set.
    NotSet { dynamic_state: DynamicState },
}

impl error::Error for CheckDynamicStateValidityError {}

impl fmt::Display for CheckDynamicStateValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            CheckDynamicStateValidityError::NotSet { dynamic_state } => {
                write!(fmt, "the pipeline requires the dynamic state {:?} to be set, but the value was not or only partially set", dynamic_state)
            }
        }
    }
}
