// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A test to discard pixels that would be written to certain areas of a framebuffer.
//!
//! The discard rectangle test is similar to, but separate from the scissor test.

use crate::pipeline::{graphics::viewport::Scissor, PartialStateMode};

/// The state in a graphics pipeline describing how the discard rectangle test should behave.
#[derive(Clone, Debug)]
pub struct DiscardRectangleState {
    /// Sets whether the discard rectangle test operates inclusively or exclusively.
    pub mode: DiscardRectangleMode,

    /// Specifies the discard rectangles. If set to `Dynamic`, it specifies only the number of
    /// rectangles used from the dynamic state.
    ///
    /// If set to `Dynamic` or to `Fixed` with a non-empty list, the
    /// [`ext_discard_rectangles`](crate::device::DeviceExtensions::ext_discard_rectangles)
    /// extension must be enabled on the device.
    pub rectangles: PartialStateMode<Vec<Scissor>, u32>,
}

impl DiscardRectangleState {
    /// Creates a `DiscardRectangleState` in exclusive mode with zero rectangles.
    #[inline]
    pub fn new() -> Self {
        Self {
            mode: DiscardRectangleMode::Exclusive,
            rectangles: PartialStateMode::Fixed(Vec::new()),
        }
    }
}

impl Default for DiscardRectangleState {
    /// Returns [`DiscardRectangleState::new`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// The mode in which the discard rectangle test operates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DiscardRectangleMode {
    /// Samples that are inside a rectangle are kept, samples that are outside all rectangles
    /// are discarded.
    Inclusive = ash::vk::DiscardRectangleModeEXT::INCLUSIVE.as_raw(),

    /// Samples that are inside a rectangle are discarded, samples that are outside all rectangles
    /// are kept.
    Exclusive = ash::vk::DiscardRectangleModeEXT::EXCLUSIVE.as_raw(),
}

impl From<DiscardRectangleMode> for ash::vk::DiscardRectangleModeEXT {
    #[inline]
    fn from(val: DiscardRectangleMode) -> Self {
        Self::from_raw(val as i32)
    }
}
