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

use crate::device::Device;
use crate::pipeline::viewport::Scissor;
use crate::pipeline::{DynamicState, GraphicsPipelineCreationError, PartialStateMode};
use fnv::FnvHashMap;
use smallvec::SmallVec;

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

    pub(crate) fn to_vulkan_rectangles(
        &self,
        device: &Device,
        dynamic_state_modes: &mut FnvHashMap<DynamicState, bool>,
    ) -> Result<SmallVec<[ash::vk::Rect2D; 2]>, GraphicsPipelineCreationError> {
        Ok(match &self.rectangles {
            PartialStateMode::Fixed(rectangles) => {
                dynamic_state_modes.insert(DynamicState::DiscardRectangle, false);
                rectangles.iter().map(|&rect| rect.into()).collect()
            }
            PartialStateMode::Dynamic(_) => {
                dynamic_state_modes.insert(DynamicState::DiscardRectangle, true);
                Default::default()
            }
        })
    }

    pub(crate) fn to_vulkan(
        &self,
        device: &Device,
        dynamic_state_modes: &mut FnvHashMap<DynamicState, bool>,
        discard_rectangles: &[ash::vk::Rect2D],
    ) -> Result<
        Option<ash::vk::PipelineDiscardRectangleStateCreateInfoEXT>,
        GraphicsPipelineCreationError,
    > {
        Ok(if device.enabled_extensions().ext_discard_rectangles {
            if discard_rectangles.len()
                > device
                    .physical_device()
                    .properties()
                    .max_discard_rectangles
                    .unwrap() as usize
            {
                return Err(
                    GraphicsPipelineCreationError::MaxDiscardRectanglesExceeded {
                        max: device
                            .physical_device()
                            .properties()
                            .max_discard_rectangles
                            .unwrap(),
                        obtained: discard_rectangles.len() as u32,
                    },
                );
            }

            let discard_rectangle_count = match &self.rectangles {
                PartialStateMode::Dynamic(count) => *count,
                PartialStateMode::Fixed(_) => discard_rectangles.len() as u32,
            };

            Some(ash::vk::PipelineDiscardRectangleStateCreateInfoEXT {
                flags: ash::vk::PipelineDiscardRectangleStateCreateFlagsEXT::empty(),
                discard_rectangle_mode: self.mode.into(),
                discard_rectangle_count,
                p_discard_rectangles: discard_rectangles.as_ptr(),
                ..Default::default()
            })
        } else {
            let error = match &self.rectangles {
                PartialStateMode::Dynamic(_) => true,
                PartialStateMode::Fixed(rectangles) => !rectangles.is_empty(),
            };

            if error {
                return Err(GraphicsPipelineCreationError::ExtensionNotEnabled {
                    extension: "ext_discard_rectangles",
                    reason: "DiscardRectangleState::rectangles was not Fixed with an empty list",
                });
            }

            None
        })
    }
}

impl Default for DiscardRectangleState {
    /// Returns [`DiscardRectangleState::new`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

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
