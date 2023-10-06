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

use crate::{
    device::Device, macros::vulkan_enum, pipeline::graphics::viewport::Scissor, ValidationError,
};

/// The state in a graphics pipeline describing how the discard rectangle test should behave.
#[derive(Clone, Debug)]
pub struct DiscardRectangleState {
    /// Sets whether the discard rectangle test operates inclusively or exclusively.
    ///
    /// The default value is [`DiscardRectangleMode::Exclusive`].
    pub mode: DiscardRectangleMode,

    /// Specifies the discard rectangles.
    ///
    /// When [`DynamicState::DiscardRectangle`] is used, the values of each rectangle are ignored
    /// and must be set dynamically, but the number of discard rectangles is fixed and
    /// must be matched when setting the dynamic value.
    ///
    /// If this not not empty, then the
    /// [`ext_discard_rectangles`](crate::device::DeviceExtensions::ext_discard_rectangles)
    /// extension must be enabled on the device.
    ///
    /// The default value is empty.
    ///
    /// [`DynamicState::DiscardRectangle`]: crate::pipeline::DynamicState::DiscardRectangle
    pub rectangles: Vec<Scissor>,

    pub _ne: crate::NonExhaustive,
}

impl Default for DiscardRectangleState {
    #[inline]
    fn default() -> Self {
        Self {
            mode: DiscardRectangleMode::Exclusive,
            rectangles: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl DiscardRectangleState {
    /// Creates a `DiscardRectangleState` in exclusive mode with zero rectangles.
    #[inline]
    #[deprecated(
        since = "0.34.0",
        note = "Use `DiscardRectangleState::default` instead."
    )]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            mode,
            ref rectangles,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        mode.validate_device(device).map_err(|err| {
            err.add_context("mode").set_vuids(&[
                "VUID-VkPipelineDiscardRectangleStateCreateInfoEXT-discardRectangleMode-parameter",
            ])
        })?;

        if rectangles.len() as u32 > properties.max_discard_rectangles.unwrap() {
            return Err(Box::new(ValidationError {
                context: "rectangles".into(),
                problem: "the length exceeds the `max_discard_rectangles` limit".into(),
                vuids: &[
                    "VUID-VkPipelineDiscardRectangleStateCreateInfoEXT-discardRectangleCount-00582",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The mode in which the discard rectangle test operates.
    DiscardRectangleMode = DiscardRectangleModeEXT(i32);

    /// Samples that are inside a rectangle are kept, samples that are outside all rectangles
    /// are discarded.
    Inclusive = INCLUSIVE,

    /// Samples that are inside a rectangle are discarded, samples that are outside all rectangles
    /// are kept.
    Exclusive = EXCLUSIVE,
}
