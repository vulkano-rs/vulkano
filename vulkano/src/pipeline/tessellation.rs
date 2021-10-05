// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Subdividing primitives.
//!
//! A tessellation shader divides primitives into smaller primitives.

use crate::pipeline::StateMode;

/// The state in a graphics pipeline describing the tessellation shader execution of a graphics
/// pipeline.
#[derive(Clone, Copy, Debug)]
pub struct TessellationState {
    /// The number of patch control points to use.
    pub patch_control_points: StateMode<u32>,
}

impl Default for TessellationState {
    /// Creates a new `TessellationState` with dynamic patch control points.
    #[inline]
    fn default() -> Self {
        Self {
            patch_control_points: StateMode::Dynamic,
        }
    }
}
