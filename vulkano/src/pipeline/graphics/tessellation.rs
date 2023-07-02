// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Subdivides primitives into smaller primitives.

use crate::{
    device::Device, macros::vulkan_enum, pipeline::StateMode, Requires, RequiresAllOf,
    RequiresOneOf, ValidationError, Version,
};

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

    /// The origin to use for the tessellation domain.
    ///
    /// If this is not [`TessellationDomainOrigin::UpperLeft`], the device API version must be at
    /// least 1.1, or the [`khr_maintenance2`](crate::device::DeviceExtensions::khr_maintenance2)
    /// extension must be enabled on the device.
    ///
    /// The default value is [`TessellationDomainOrigin::UpperLeft`].
    pub domain_origin: TessellationDomainOrigin,

    pub _ne: crate::NonExhaustive,
}

impl TessellationState {
    /// Creates a new `TessellationState` with 3 patch control points.
    #[inline]
    pub fn new() -> Self {
        Self {
            patch_control_points: StateMode::Fixed(3),
            domain_origin: TessellationDomainOrigin::default(),
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Sets the number of patch control points.
    #[inline]
    pub fn patch_control_points(mut self, num: u32) -> Self {
        self.patch_control_points = StateMode::Fixed(num);
        self
    }

    /// Sets the number of patch control points to dynamic.
    #[inline]
    pub fn patch_control_points_dynamic(mut self) -> Self {
        self.patch_control_points = StateMode::Dynamic;
        self
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            patch_control_points,
            domain_origin,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        match patch_control_points {
            StateMode::Fixed(patch_control_points) => {
                if patch_control_points == 0 {
                    return Err(ValidationError {
                        context: "patch_control_points".into(),
                        problem: "is zero".into(),
                        vuids: &[
                            "VUID-VkPipelineTessellationStateCreateInfo-patchControlPoints-01214",
                        ],
                        ..Default::default()
                    });
                }

                if patch_control_points > properties.max_tessellation_patch_size {
                    return Err(ValidationError {
                        context: "patch_control_points".into(),
                        problem: "exceeds the `max_tessellation_patch_size` limit".into(),
                        vuids: &[
                            "VUID-VkPipelineTessellationStateCreateInfo-patchControlPoints-01214",
                        ],
                        ..Default::default()
                    });
                }
            }
            StateMode::Dynamic => {
                if !device
                    .enabled_features()
                    .extended_dynamic_state2_patch_control_points
                {
                    return Err(ValidationError {
                        context: "patch_control_points".into(),
                        problem: "is dynamic".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "extended_dynamic_state2_patch_control_points",
                        )])]),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-04870"],
                    });
                }
            }
        };

        domain_origin
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "domain_origin".into(),
                vuids: &[
                    "VUID-VkPipelineTessellationDomainOriginStateCreateInfo-domainOrigin-parameter",
                ],
                ..ValidationError::from_requirement(err)
            })?;

        if domain_origin != TessellationDomainOrigin::UpperLeft
            && !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_maintenance2)
        {
            return Err(ValidationError {
                context: "domain_origin".into(),
                problem: "is not `TessellationDomainOrigin::UpperLeft`".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                    RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance2")]),
                ]),
                ..Default::default()
            });
        }

        Ok(())
    }
}

impl Default for TessellationState {
    /// Returns [`TessellationState::new()`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The origin of the tessellation domain.
    TessellationDomainOrigin = TessellationDomainOrigin(i32);

    /// The origin is in the upper left corner.
    ///
    /// This is the default.
    UpperLeft = UPPER_LEFT,

    /// The origin is in the lower left corner.
    LowerLeft = LOWER_LEFT,
}

impl Default for TessellationDomainOrigin {
    #[inline]
    fn default() -> Self {
        Self::UpperLeft
    }
}
