// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::command_buffer::synced::CommandBufferState;
use crate::pipeline::input_assembly::PrimitiveTopology;
use crate::pipeline::shader::ShaderStage;
use crate::pipeline::DynamicState;
use crate::pipeline::GraphicsPipeline;
use crate::pipeline::PartialStateMode;
use std::error;
use std::fmt;

/// Checks whether states that are about to be set are correct.
pub(in super::super) fn check_dynamic_state_validity(
    current_state: CommandBufferState,
    pipeline: &GraphicsPipeline,
) -> Result<(), CheckDynamicStateValidityError> {
    let device = pipeline.device();

    for state in pipeline
        .dynamic_states()
        .filter(|(_, d)| *d)
        .map(|(s, _)| s)
    {
        match state {
            DynamicState::BlendConstants => {
                if current_state.blend_constants().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::BlendConstants,
                    });
                }
            }
            DynamicState::ColorWriteEnable => {
                let enables = if let Some(enables) = current_state.color_write_enable() {
                    enables
                } else {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::CullMode,
                    });
                };

                if enables.len() < pipeline.color_blend_state().unwrap().attachments.len() {
                    return Err(CheckDynamicStateValidityError::NotEnoughColorWriteEnable {
                        color_write_enable_count: enables.len() as u32,
                        attachment_count: pipeline.color_blend_state().unwrap().attachments.len()
                            as u32,
                    });
                }
            }
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
            DynamicState::LogicOp => {
                if current_state.logic_op().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::LogicOp,
                    });
                }
            }
            DynamicState::PatchControlPoints => {
                if current_state.patch_control_points().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::PatchControlPoints,
                    });
                }
            }
            DynamicState::PrimitiveRestartEnable => {
                let primitive_restart_enable =
                    if let Some(enable) = current_state.primitive_restart_enable() {
                        enable
                    } else {
                        return Err(CheckDynamicStateValidityError::NotSet {
                            dynamic_state: DynamicState::PrimitiveRestartEnable,
                        });
                    };

                if primitive_restart_enable {
                    let topology = match pipeline.input_assembly_state().topology {
                        PartialStateMode::Fixed(topology) => topology,
                        PartialStateMode::Dynamic(_) => {
                            if let Some(topology) = current_state.primitive_topology() {
                                topology
                            } else {
                                return Err(CheckDynamicStateValidityError::NotSet {
                                    dynamic_state: DynamicState::PrimitiveTopology,
                                });
                            }
                        }
                    };

                    match topology {
                        PrimitiveTopology::PointList
                        | PrimitiveTopology::LineList
                        | PrimitiveTopology::TriangleList
                        | PrimitiveTopology::LineListWithAdjacency
                        | PrimitiveTopology::TriangleListWithAdjacency => {
                            if !device.enabled_features().primitive_topology_list_restart {
                                return Err(CheckDynamicStateValidityError::FeatureNotEnabled {
                                    feature: "primitive_topology_list_restart",
                                    reason: "the PrimitiveRestartEnable dynamic state was true in combination with a List PrimitiveTopology",
                                });
                            }
                        }
                        PrimitiveTopology::PatchList => {
                            if !device
                                .enabled_features()
                                .primitive_topology_patch_list_restart
                            {
                                return Err(CheckDynamicStateValidityError::FeatureNotEnabled {
                                    feature: "primitive_topology_patch_list_restart",
                                    reason: "the PrimitiveRestartEnable dynamic state was true in combination with PrimitiveTopology::PatchList",
                                });
                            }
                        }
                        _ => (),
                    }
                }
            }
            DynamicState::PrimitiveTopology => {
                let topology = if let Some(topology) = current_state.primitive_topology() {
                    topology
                } else {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::PrimitiveTopology,
                    });
                };

                if pipeline.shader(ShaderStage::TessellationControl).is_some() {
                    if !matches!(topology, PrimitiveTopology::PatchList) {
                        return Err(CheckDynamicStateValidityError::InvalidPrimitiveTopology {
                            topology,
                            reason: "the graphics pipeline includes tessellation shaders, so the topology must be PatchList",
                        });
                    }
                } else {
                    if matches!(topology, PrimitiveTopology::PatchList) {
                        return Err(CheckDynamicStateValidityError::InvalidPrimitiveTopology {
                            topology,
                            reason: "the graphics pipeline doesn't include tessellation shaders",
                        });
                    }
                }

                let topology_class = match pipeline.input_assembly_state().topology {
                    PartialStateMode::Dynamic(topology_class) => topology_class,
                    _ => unreachable!(),
                };

                if topology.class() != topology_class {
                    return Err(CheckDynamicStateValidityError::InvalidPrimitiveTopology {
                        topology,
                        reason: "the topology class does not match the class the pipeline was created for",
                    });
                }

                // TODO: check that the topology matches the geometry shader
            }
            DynamicState::RasterizerDiscardEnable => todo!(),
            DynamicState::RayTracingPipelineStackSize => unreachable!(
                "RayTracingPipelineStackSize dynamic state should not occur on a graphics pipeline"
            ),
            DynamicState::SampleLocations => todo!(),
            DynamicState::Scissor => {
                for num in 0..pipeline.viewport_state().unwrap().count().unwrap() {
                    if current_state.scissor(num).is_none() {
                        return Err(CheckDynamicStateValidityError::NotSet {
                            dynamic_state: DynamicState::Scissor,
                        });
                    }
                }
            }
            DynamicState::ScissorWithCount => {
                let scissor_count = if let Some(scissors) = current_state.scissor_with_count() {
                    scissors.len() as u32
                } else {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::ScissorWithCount,
                    });
                };

                // Check if the counts match, but only if the viewport count is fixed.
                // If the viewport count is also dynamic, then the DynamicState::ViewportWithCount
                // match arm will handle it.
                if let Some(viewport_count) = pipeline.viewport_state().unwrap().count() {
                    if viewport_count != scissor_count {
                        return Err(
                            CheckDynamicStateValidityError::ViewportScissorCountMismatch {
                                viewport_count,
                                scissor_count,
                            },
                        );
                    }
                }
            }
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
                for num in 0..pipeline.viewport_state().unwrap().count().unwrap() {
                    if current_state.viewport(num).is_none() {
                        return Err(CheckDynamicStateValidityError::NotSet {
                            dynamic_state: DynamicState::CullMode,
                        });
                    }
                }
            }
            DynamicState::ViewportCoarseSampleOrder => todo!(),
            DynamicState::ViewportShadingRatePalette => todo!(),
            DynamicState::ViewportWithCount => {
                let viewport_count = if let Some(viewports) = current_state.viewport_with_count() {
                    viewports.len() as u32
                } else {
                    return Err(CheckDynamicStateValidityError::NotSet {
                        dynamic_state: DynamicState::ViewportWithCount,
                    });
                };

                let scissor_count =
                    if let Some(scissor_count) = pipeline.viewport_state().unwrap().count() {
                        // The scissor count is fixed.
                        scissor_count
                    } else {
                        // The scissor count is also dynamic.
                        if let Some(scissors) = current_state.scissor_with_count() {
                            scissors.len() as u32
                        } else {
                            return Err(CheckDynamicStateValidityError::NotSet {
                                dynamic_state: DynamicState::ScissorWithCount,
                            });
                        }
                    };

                if viewport_count != scissor_count {
                    return Err(
                        CheckDynamicStateValidityError::ViewportScissorCountMismatch {
                            viewport_count,
                            scissor_count,
                        },
                    );
                }

                // TODO: VUID-vkCmdDrawIndexed-primitiveFragmentShadingRateWithMultipleViewports-04552
                // If the primitiveFragmentShadingRateWithMultipleViewports limit is not supported,
                // the bound graphics pipeline was created with the
                // VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT dynamic state enabled, and any of the
                // shader stages of the bound graphics pipeline write to the PrimitiveShadingRateKHR
                // built-in, then vkCmdSetViewportWithCountEXT must have been called in the current
                // command buffer prior to this drawing command, and the viewportCount parameter of
                // vkCmdSetViewportWithCountEXT must be 1
            }
            DynamicState::ViewportWScaling => todo!(),
        }
    }

    Ok(())
}

/// Error that can happen when validating dynamic states.
#[derive(Debug, Copy, Clone)]
pub enum CheckDynamicStateValidityError {
    /// A device feature that was required for a particular dynamic state value was not enabled.
    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
    },

    /// The provided dynamic primitive topology is not compatible with the pipeline.
    InvalidPrimitiveTopology {
        topology: PrimitiveTopology,
        reason: &'static str,
    },

    /// The number of ColorWriteEnable values was less than the number of attachments in the
    /// color blend state of the pipeline.
    NotEnoughColorWriteEnable {
        color_write_enable_count: u32,
        attachment_count: u32,
    },

    /// The pipeline requires a particular state to be set dynamically, but the value was not or
    /// only partially set.
    NotSet { dynamic_state: DynamicState },

    /// The viewport count and scissor count do not match.
    ViewportScissorCountMismatch {
        viewport_count: u32,
        scissor_count: u32,
    },
}

impl error::Error for CheckDynamicStateValidityError {}

impl fmt::Display for CheckDynamicStateValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::FeatureNotEnabled { feature, reason } => {
                write!(fmt, "the feature {} must be enabled: {}", feature, reason)
            }
            Self::InvalidPrimitiveTopology { topology, reason } => {
                write!(
                    fmt,
                    "invalid dynamic PrimitiveTypology::{:?}: {}",
                    topology, reason
                )
            }
            Self::NotEnoughColorWriteEnable {
                color_write_enable_count,
                attachment_count,
            } => {
                write!(fmt, "the number of ColorWriteEnable values ({}) was less than the number of attachments ({}) in the color blend state of the pipeline", color_write_enable_count, attachment_count)
            }
            Self::NotSet { dynamic_state } => {
                write!(fmt, "the pipeline requires the dynamic state {:?} to be set, but the value was not or only partially set", dynamic_state)
            }
            Self::ViewportScissorCountMismatch {
                viewport_count,
                scissor_count,
            } => {
                write!(fmt, "the viewport count and scissor count do not match; viewport count is {}, scissor count is {}", viewport_count, scissor_count)
            }
        }
    }
}
