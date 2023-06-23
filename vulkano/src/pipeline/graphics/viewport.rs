// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Configures the area of the framebuffer that pixels will be written to.
//!
//! There are two different concepts to determine where things will be drawn:
//!
//! - The viewport is the region of the image which corresponds to the vertex coordinates `-1.0` to
//!   `1.0`.
//! - Any pixel outside of the scissor box will be discarded.
//!
//! In other words, modifying the viewport will stretch the image, while modifying the scissor
//! box acts like a filter.
//!
//! It is legal and sensible to use a viewport that is larger than the target image or that
//! only partially overlaps the target image.
//!
//! # Multiple viewports
//!
//! In most situations, you only need a single viewport and a single scissor box.
//!
//! If, however, you use a geometry shader, you can specify multiple viewports and scissor boxes.
//! Then in your geometry shader you can specify in which viewport and scissor box the primitive
//! should be written to. In GLSL this is done by writing to the special variable
//! `gl_ViewportIndex`.
//!
//! If you don't use a geometry shader or use a geometry shader where don't set which viewport to
//! use, then the first viewport and scissor box will be used.
//!
//! # Dynamic and fixed
//!
//! Vulkan allows four different setups:
//!
//! - The state of both the viewports and scissor boxes is known at pipeline creation.
//! - The state of viewports is known at pipeline creation, but the state of scissor boxes is
//!   only known when submitting the draw command.
//! - The state of scissor boxes is known at pipeline creation, but the state of viewports is
//!   only known when submitting the draw command.
//! - The state of both the viewports and scissor boxes is only known when submitting the
//!   draw command.
//!
//! In all cases the number of viewports and scissor boxes must be the same.
//!

use crate::{
    device::Device,
    pipeline::{PartialStateMode, StateMode},
    Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version,
};
use smallvec::{smallvec, SmallVec};
use std::ops::RangeInclusive;

/// List of viewports and scissors that are used when creating a graphics pipeline object.
///
/// Note that the number of viewports and scissors must be the same.
#[derive(Clone, Debug)]
pub struct ViewportState {
    /// Specifies the viewport transforms.
    ///
    /// If the value is `PartialStateMode::Dynamic`, then the viewports are set dynamically, but
    /// the number of viewports is still fixed.
    ///
    /// If the value is `PartialStateMode::Dynamic(StateMode::Dynamic)`, then the number of
    /// viewports is also dynamic. The device API version must be at least 1.3, or the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must
    /// be enabled on the device.
    ///
    /// If neither the number of viewports nor the number of scissors is dynamic, then the
    /// number of both must be identical.
    ///
    /// The default value is `PartialStateMode::Fixed` with no viewports.
    pub viewports: PartialStateMode<SmallVec<[Viewport; 1]>, StateMode<u32>>,

    /// Specifies the scissor rectangles.
    ///
    /// If the value is `PartialStateMode::Dynamic`, then the scissors are set dynamically, but
    /// the number of scissors is still fixed.
    ///
    /// If the value is `PartialStateMode::Dynamic(StateMode::Dynamic)`, then the number of
    /// scissors is also dynamic. The device API version must be at least 1.3, or the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must
    /// be enabled on the device.
    ///
    /// If neither the number of viewports nor the number of scissors is dynamic, then the
    /// number of both must be identical.
    ///
    /// The default value is `PartialStateMode::Fixed` with no scissors.
    pub scissors: PartialStateMode<SmallVec<[Scissor; 1]>, StateMode<u32>>,

    pub _ne: crate::NonExhaustive,
}

impl ViewportState {
    /// Creates a `ViewportState` with fixed state and no viewports or scissors.
    #[inline]
    pub fn new() -> Self {
        Self {
            viewports: PartialStateMode::Fixed(SmallVec::new()),
            scissors: PartialStateMode::Fixed(SmallVec::new()),
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Creates a `ViewportState` with fixed state from the given viewports and scissors.
    pub fn viewport_fixed_scissor_fixed(
        data: impl IntoIterator<Item = (Viewport, Scissor)>,
    ) -> Self {
        let (viewports, scissors) = data.into_iter().unzip();
        Self {
            viewports: PartialStateMode::Fixed(viewports),
            scissors: PartialStateMode::Fixed(scissors),
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Creates a `ViewportState` with fixed state from the given viewports, and matching scissors
    /// that cover the whole viewport.
    pub fn viewport_fixed_scissor_irrelevant(data: impl IntoIterator<Item = Viewport>) -> Self {
        let viewports: SmallVec<_> = data.into_iter().collect();
        let scissors = smallvec![Scissor::irrelevant(); viewports.len()];
        Self {
            viewports: PartialStateMode::Fixed(viewports),
            scissors: PartialStateMode::Fixed(scissors),
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Creates a `ViewportState` with dynamic viewport, and a single scissor that always covers
    /// the whole viewport.
    #[inline]
    pub fn viewport_dynamic_scissor_irrelevant() -> Self {
        Self {
            viewports: PartialStateMode::Dynamic(StateMode::Fixed(1)),
            scissors: PartialStateMode::Fixed(smallvec![Scissor::irrelevant(); 1]),
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Creates a `ViewportState` with dynamic viewports and scissors, but a fixed count.
    #[inline]
    pub fn viewport_dynamic_scissor_dynamic(count: u32) -> Self {
        Self {
            viewports: PartialStateMode::Dynamic(StateMode::Fixed(count)),
            scissors: PartialStateMode::Dynamic(StateMode::Fixed(count)),
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Creates a `ViewportState` with dynamic viewport count and scissor count.
    #[inline]
    pub fn viewport_count_dynamic_scissor_count_dynamic() -> Self {
        Self {
            viewports: PartialStateMode::Dynamic(StateMode::Dynamic),
            scissors: PartialStateMode::Dynamic(StateMode::Dynamic),
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Returns the number of viewports and scissors.
    ///
    /// `None` is returned if both counts are dynamic.
    #[inline]
    pub(crate) fn count(&self) -> Option<u32> {
        match &self.viewports {
            PartialStateMode::Fixed(viewports) => return Some(viewports.len() as u32),
            PartialStateMode::Dynamic(StateMode::Fixed(count)) => return Some(*count),
            PartialStateMode::Dynamic(StateMode::Dynamic) => (),
        }

        match &self.scissors {
            PartialStateMode::Fixed(scissors) => return Some(scissors.len() as u32),
            PartialStateMode::Dynamic(StateMode::Fixed(count)) => return Some(*count),
            PartialStateMode::Dynamic(StateMode::Dynamic) => (),
        }

        None
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let Self {
            viewports,
            scissors,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        let viewport_count = match viewports {
            PartialStateMode::Fixed(viewports) => {
                if viewports.is_empty() {
                    return Err(ValidationError {
                        context: "viewports".into(),
                        problem: "is empty".into(),
                        vuids: &["VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135"],
                        ..Default::default()
                    });
                }

                for (index, viewport) in viewports.iter().enumerate() {
                    viewport
                        .validate(device)
                        .map_err(|err| err.add_context(format!("viewports[{}].0", index)))?;
                }

                viewports.len() as u32
            }
            PartialStateMode::Dynamic(StateMode::Fixed(count)) => {
                if *count == 0 {
                    return Err(ValidationError {
                        context: "viewports".into(),
                        problem: "is empty".into(),
                        vuids: &["VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135"],
                        ..Default::default()
                    });
                }

                *count
            }
            PartialStateMode::Dynamic(StateMode::Dynamic) => {
                // VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135
                0
            }
        };

        let scissor_count = match scissors {
            PartialStateMode::Fixed(scissors) => {
                if scissors.is_empty() {
                    return Err(ValidationError {
                        context: "scissors".into(),
                        problem: "is empty".into(),
                        vuids: &["VUID-VkPipelineViewportStateCreateInfo-scissorCount-04136"],
                        ..Default::default()
                    });
                }

                for (index, scissor) in scissors.iter().enumerate() {
                    let &Scissor { offset, extent } = scissor;

                    // VUID-VkPipelineViewportStateCreateInfo-x-02821
                    // Ensured by the use of an unsigned integer.

                    if (i32::try_from(offset[0]).ok())
                        .zip(i32::try_from(extent[0]).ok())
                        .and_then(|(o, e)| o.checked_add(e))
                        .is_none()
                    {
                        return Err(ValidationError {
                            context: format!("scissors[{}]", index).into(),
                            problem: "`offset[0] + extent[0]` is greater than `i32::MAX`".into(),
                            vuids: &["VUID-VkPipelineViewportStateCreateInfo-offset-02822"],
                            ..Default::default()
                        });
                    }

                    if (i32::try_from(offset[1]).ok())
                        .zip(i32::try_from(extent[1]).ok())
                        .and_then(|(o, e)| o.checked_add(e))
                        .is_none()
                    {
                        return Err(ValidationError {
                            context: format!("scissors[{}]", index).into(),
                            problem: "`offset[1] + extent[1]` is greater than `i32::MAX`".into(),
                            vuids: &["VUID-VkPipelineViewportStateCreateInfo-offset-02823"],
                            ..Default::default()
                        });
                    }
                }

                scissors.len() as u32
            }
            PartialStateMode::Dynamic(StateMode::Fixed(count)) => {
                if *count == 0 {
                    return Err(ValidationError {
                        context: "scissors".into(),
                        problem: "is empty".into(),
                        vuids: &["VUID-VkPipelineViewportStateCreateInfo-scissorCount-04136"],
                        ..Default::default()
                    });
                }

                *count
            }
            PartialStateMode::Dynamic(StateMode::Dynamic) => {
                // VUID-VkPipelineViewportStateCreateInfo-scissorCount-04136
                0
            }
        };

        if viewport_count != 0 && scissor_count != 0 && viewport_count != scissor_count {
            return Err(ValidationError {
                problem: "the length of `viewports` and the length of `scissors` are not equal"
                    .into(),
                vuids: &["VUID-VkPipelineViewportStateCreateInfo-scissorCount-04134"],
                ..Default::default()
            });
        }

        if viewport_count > 1 && !device.enabled_features().multi_viewport {
            return Err(ValidationError {
                context: "viewports".into(),
                problem: "the length is greater than 1".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "multi_viewport",
                )])]),
                vuids: &["VUID-VkPipelineViewportStateCreateInfo-viewportCount-01216"],
            });
        }

        if scissor_count > 1 && !device.enabled_features().multi_viewport {
            return Err(ValidationError {
                context: "scissors".into(),
                problem: "the length is greater than 1".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "multi_viewport",
                )])]),
                vuids: &["VUID-VkPipelineViewportStateCreateInfo-scissorCount-01217"],
            });
        }

        if viewport_count > properties.max_viewports {
            return Err(ValidationError {
                context: "viewports".into(),
                problem: "the length exceeds the `max_viewports` limit".into(),
                vuids: &["VUID-VkPipelineViewportStateCreateInfo-viewportCount-01218"],
                ..Default::default()
            });
        }

        if scissor_count > properties.max_viewports {
            return Err(ValidationError {
                context: "scissors".into(),
                problem: "the length exceeds the `max_viewports` limit".into(),
                vuids: &["VUID-VkPipelineViewportStateCreateInfo-scissorCount-01219"],
                ..Default::default()
            });
        }

        Ok(())
    }
}

impl Default for ViewportState {
    /// Returns [`ViewportState::new()`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// State of a single viewport.
#[derive(Debug, Clone, PartialEq)]
pub struct Viewport {
    /// Coordinates in pixels of the top-left hand corner of the viewport.
    pub offset: [f32; 2],

    /// Dimensions in pixels of the viewport.
    pub extent: [f32; 2],

    /// Minimum and maximum values of the depth.
    ///
    /// The values `0.0` to `1.0` of each vertex's Z coordinate will be mapped to this
    /// `depth_range` before being compared to the existing depth value.
    ///
    /// This is equivalents to `glDepthRange` in OpenGL, except that OpenGL uses the Z coordinate
    /// range from `-1.0` to `1.0` instead.
    pub depth_range: RangeInclusive<f32>,
}

impl Viewport {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            offset,
            extent,
            ref depth_range,
        } = self;

        let properties = device.physical_device().properties();

        if extent[0] <= 0.0 {
            return Err(ValidationError {
                context: "extent[0]".into(),
                problem: "is not greater than zero".into(),
                vuids: &["VUID-VkViewport-width-01770"],
                ..Default::default()
            });
        }

        if extent[0] > properties.max_viewport_dimensions[0] as f32 {
            return Err(ValidationError {
                context: "extent[0]".into(),
                problem: "exceeds the `max_viewport_dimensions[0]` limit".into(),
                vuids: &["VUID-VkViewport-width-01771"],
                ..Default::default()
            });
        }

        if extent[1] <= 0.0
            && !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_maintenance1)
        {
            return Err(ValidationError {
                context: "extent[1]".into(),
                problem: "is not greater than zero".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance1")]),
                ]),
                vuids: &["VUID-VkViewport-apiVersion-07917"],
            });
        }

        if extent[1].abs() > properties.max_viewport_dimensions[1] as f32 {
            return Err(ValidationError {
                context: "extent[1]".into(),
                problem: "exceeds the `max_viewport_dimensions[1]` limit".into(),
                vuids: &["VUID-VkViewport-height-01773"],
                ..Default::default()
            });
        }

        if offset[0] < properties.viewport_bounds_range[0] {
            return Err(ValidationError {
                problem: "`offset[0]` is less than the `viewport_bounds_range[0]` property".into(),
                vuids: &["VUID-VkViewport-x-01774"],
                ..Default::default()
            });
        }

        if offset[0] + extent[0] > properties.viewport_bounds_range[1] {
            return Err(ValidationError {
                problem: "`offset[0] + extent[0]` is greater than the \
                    `viewport_bounds_range[1]` property"
                    .into(),
                vuids: &["VUID-VkViewport-x-01232"],
                ..Default::default()
            });
        }

        if offset[1] < properties.viewport_bounds_range[0] {
            return Err(ValidationError {
                problem: "`offset[1]` is less than the `viewport_bounds_range[0]` property".into(),
                vuids: &["VUID-VkViewport-y-01775"],
                ..Default::default()
            });
        }

        if offset[1] > properties.viewport_bounds_range[1] {
            return Err(ValidationError {
                problem: "`offset[1]` is greater than the `viewport_bounds_range[1]` property"
                    .into(),
                vuids: &["VUID-VkViewport-y-01776"],
                ..Default::default()
            });
        }

        if offset[1] + extent[1] < properties.viewport_bounds_range[0] {
            return Err(ValidationError {
                problem: "`offset[1] + extent[1]` is less than the \
                    `viewport_bounds_range[0]` property"
                    .into(),
                vuids: &["VUID-VkViewport-y-01777"],
                ..Default::default()
            });
        }

        if offset[1] + extent[1] > properties.viewport_bounds_range[1] {
            return Err(ValidationError {
                problem: "`offset[1] + extent[1]` is greater than the \
                    `viewport_bounds_range[1]` property"
                    .into(),
                vuids: &["VUID-VkViewport-y-01233"],
                ..Default::default()
            });
        }

        if !device.enabled_extensions().ext_depth_range_unrestricted {
            if *depth_range.start() < 0.0 || *depth_range.start() > 1.0 {
                return Err(ValidationError {
                    problem: "`depth_range.start` is not between 0.0 and 1.0 inclusive".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_depth_range_unrestricted",
                    )])]),
                    vuids: &["VUID-VkViewport-minDepth-01234"],
                    ..Default::default()
                });
            }

            if *depth_range.end() < 0.0 || *depth_range.end() > 1.0 {
                return Err(ValidationError {
                    problem: "`depth_range.end` is not between 0.0 and 1.0 inclusive".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_depth_range_unrestricted",
                    )])]),
                    vuids: &["VUID-VkViewport-maxDepth-01235"],
                    ..Default::default()
                });
            }
        }

        Ok(())
    }
}

impl From<&Viewport> for ash::vk::Viewport {
    #[inline]
    fn from(val: &Viewport) -> Self {
        ash::vk::Viewport {
            x: val.offset[0],
            y: val.offset[1],
            width: val.extent[0],
            height: val.extent[1],
            min_depth: *val.depth_range.start(),
            max_depth: *val.depth_range.end(),
        }
    }
}

/// State of a single scissor box.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Scissor {
    /// Coordinates in pixels of the top-left hand corner of the box.
    pub offset: [u32; 2],

    /// Dimensions in pixels of the box.
    pub extent: [u32; 2],
}

impl Scissor {
    /// Returns a scissor that, when used, will instruct the pipeline to draw to the entire
    /// framebuffer no matter its size.
    #[inline]
    pub fn irrelevant() -> Scissor {
        Scissor {
            offset: [0, 0],
            extent: [0x7fffffff, 0x7fffffff],
        }
    }
}

impl Default for Scissor {
    #[inline]
    fn default() -> Scissor {
        Scissor::irrelevant()
    }
}

impl From<&Scissor> for ash::vk::Rect2D {
    #[inline]
    fn from(val: &Scissor) -> Self {
        ash::vk::Rect2D {
            offset: ash::vk::Offset2D {
                x: val.offset[0] as i32,
                y: val.offset[1] as i32,
            },
            extent: ash::vk::Extent2D {
                width: val.extent[0],
                height: val.extent[1],
            },
        }
    }
}
