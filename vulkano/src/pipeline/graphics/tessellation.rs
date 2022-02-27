// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Subdivides primitives into smaller primitives.

use crate::device::Device;
use crate::pipeline::graphics::input_assembly::{
    InputAssemblyState, PrimitiveTopology, PrimitiveTopologyClass,
};
use crate::pipeline::graphics::GraphicsPipelineCreationError;
use crate::pipeline::{DynamicState, PartialStateMode, StateMode};
use std::collections::HashMap;

/// The state in a graphics pipeline describing the tessellation shader execution of a graphics
/// pipeline.
#[derive(Clone, Copy, Debug)]
pub struct TessellationState {
    /// The number of patch control points to use.
    ///
    /// If set to `Dynamic`, the
    /// [`extended_dynamic_state2_patch_control_points`](crate::device::Features::extended_dynamic_state2_patch_control_points)
    /// feature must be enabled on the device.
    pub patch_control_points: StateMode<u32>,
}

impl TessellationState {
    /// Creates a new `TessellationState` with 3 patch control points.
    #[inline]
    pub fn new() -> Self {
        Self {
            patch_control_points: StateMode::Fixed(3),
        }
    }

    /// Sets the number of patch control points.
    #[inline]
    pub fn patch_control_points(mut self, num: u32) -> Self {
        self.patch_control_points = StateMode::Fixed(num);
        self
    }

    /// Sets the number of patch control points to dynamic.
    pub fn patch_control_points_dynamic(mut self) -> Self {
        self.patch_control_points = StateMode::Dynamic;
        self
    }

    pub(crate) fn to_vulkan(
        &self,
        device: &Device,
        dynamic_state_modes: &mut HashMap<DynamicState, bool>,
        input_assembly_state: &InputAssemblyState,
    ) -> Result<ash::vk::PipelineTessellationStateCreateInfo, GraphicsPipelineCreationError> {
        if !matches!(
            input_assembly_state.topology,
            PartialStateMode::Dynamic(PrimitiveTopologyClass::Patch)
                | PartialStateMode::Fixed(PrimitiveTopology::PatchList)
        ) {
            return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
        }

        let patch_control_points = match self.patch_control_points {
            StateMode::Fixed(patch_control_points) => {
                if patch_control_points <= 0
                    || patch_control_points
                        > device
                            .physical_device()
                            .properties()
                            .max_tessellation_patch_size
                {
                    return Err(GraphicsPipelineCreationError::InvalidNumPatchControlPoints);
                }
                dynamic_state_modes.insert(DynamicState::PatchControlPoints, false);
                patch_control_points
            }
            StateMode::Dynamic => {
                if !device
                    .enabled_features()
                    .extended_dynamic_state2_patch_control_points
                {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "extended_dynamic_state2_patch_control_points",
                        reason: "TessellationState::patch_control_points was set to Dynamic",
                    });
                }
                dynamic_state_modes.insert(DynamicState::PatchControlPoints, true);
                Default::default()
            }
        };

        Ok(ash::vk::PipelineTessellationStateCreateInfo {
            flags: ash::vk::PipelineTessellationStateCreateFlags::empty(),
            patch_control_points,
            ..Default::default()
        })
    }
}

impl Default for TessellationState {
    /// Returns [`TessellationState::new()`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
