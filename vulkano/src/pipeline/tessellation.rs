// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

/// The tessellation state of a graphics pipeline.
#[derive(Clone, Copy, Debug, Default)]
pub struct TessellationState {
    /// The number of patch control points to use. `None` indicates that it is set dynamically.
    pub patch_control_points: Option<u32>,
}
