// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Subdivides primitives into smaller primitives.

use crate::{macros::vulkan_enum, pipeline::StateMode};

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
    pub domain_origin: TessellationDomainOrigin,
}

impl TessellationState {
    /// Creates a new `TessellationState` with 3 patch control points.
    #[inline]
    pub fn new() -> Self {
        Self {
            patch_control_points: StateMode::Fixed(3),
            domain_origin: TessellationDomainOrigin::default(),
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
