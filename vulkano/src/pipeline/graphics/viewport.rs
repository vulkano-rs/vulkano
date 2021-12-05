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

use crate::device::Device;
use crate::pipeline::graphics::GraphicsPipelineCreationError;
use crate::pipeline::DynamicState;
use fnv::FnvHashMap;
use smallvec::SmallVec;
use std::{ops::Range, ptr};

/// List of viewports and scissors that are used when creating a graphics pipeline object.
///
/// Note that the number of viewports and scissors must be the same.
#[derive(Debug, Clone)]
pub enum ViewportState {
    /// The state is known in advance.
    Fixed {
        /// State of the viewports and scissors.
        data: Vec<(Viewport, Scissor)>,
    },

    /// The state of viewports is known in advance, but the state of scissors is dynamic and will
    /// be set when drawing.
    FixedViewport {
        /// State of the viewports.
        viewports: Vec<Viewport>,

        /// Sets whether the scissor count is also dynamic, or only the scissors themselves.
        ///
        /// If set to `true`, the
        /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must
        /// be enabled on the device.
        scissor_count_dynamic: bool,
    },

    /// The state of scissors is known in advance, but the state of viewports is dynamic and will
    /// be set when drawing.
    FixedScissor {
        /// State of the scissors.
        scissors: Vec<Scissor>,

        /// Sets whether the viewport count is also dynamic, or only the viewports themselves.
        ///
        /// If set to `true`, the
        /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must
        /// be enabled on the device.
        viewport_count_dynamic: bool,
    },

    /// The state of both the viewports and scissors is dynamic and will be set when drawing.
    Dynamic {
        /// Number of viewports and scissors.
        ///
        /// This is ignored if both `viewport_count_dynamic` and `scissor_count_dynamic` are `true`.
        count: u32,

        /// Sets whether the viewport count is also dynamic, or only the viewports themselves.
        ///
        /// If set to `true`, the
        /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must
        /// be enabled on the device.
        viewport_count_dynamic: bool,

        /// Sets whether the scissor count is also dynamic, or only the scissors themselves.
        ///
        /// If set to `true`, the
        /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must
        /// be enabled on the device.
        scissor_count_dynamic: bool,
    },
}

impl ViewportState {
    /// Creates a `ViewportState` with fixed state and no viewports or scissors.
    #[inline]
    pub fn new() -> Self {
        Self::Fixed { data: Vec::new() }
    }

    /// Creates a `ViewportState` with fixed state from the given viewports and scissors.
    #[inline]
    pub fn viewport_fixed_scissor_fixed(
        data: impl IntoIterator<Item = (Viewport, Scissor)>,
    ) -> Self {
        Self::Fixed {
            data: data.into_iter().collect(),
        }
    }

    /// Creates a `ViewportState` with fixed state from the given viewports, and matching scissors
    /// that cover the whole viewport.
    #[inline]
    pub fn viewport_fixed_scissor_irrelevant(data: impl IntoIterator<Item = Viewport>) -> Self {
        Self::Fixed {
            data: data
                .into_iter()
                .map(|viewport| (viewport, Scissor::irrelevant()))
                .collect(),
        }
    }

    /// Creates a `ViewportState` with dynamic viewport, and a single scissor that always covers
    /// the whole viewport.
    #[inline]
    pub fn viewport_dynamic_scissor_irrelevant() -> Self {
        Self::FixedScissor {
            scissors: vec![Scissor::irrelevant()],
            viewport_count_dynamic: false,
        }
    }

    /// Creates a `ViewportState` with dynamic viewports and scissors, but a fixed count.
    #[inline]
    pub fn viewport_dynamic_scissor_dynamic(count: u32) -> Self {
        Self::Dynamic {
            count,
            viewport_count_dynamic: false,
            scissor_count_dynamic: false,
        }
    }

    /// Creates a `ViewportState` with dynamic viewport count and scissor count.
    #[inline]
    pub fn viewport_count_dynamic_scissor_count_dynamic() -> Self {
        Self::Dynamic {
            count: 0,
            viewport_count_dynamic: true,
            scissor_count_dynamic: true,
        }
    }

    /// Returns the number of viewports and scissors.
    ///
    /// `None` is returned if both `viewport_count_dynamic` and `scissor_count_dynamic` are `true`.
    pub fn count(&self) -> Option<u32> {
        Some(match *self {
            ViewportState::Fixed { ref data } => data.len() as u32,
            ViewportState::FixedViewport { ref viewports, .. } => viewports.len() as u32,
            ViewportState::FixedScissor { ref scissors, .. } => scissors.len() as u32,
            ViewportState::Dynamic {
                viewport_count_dynamic: true,
                scissor_count_dynamic: true,
                ..
            } => return None,
            ViewportState::Dynamic { count, .. } => count,
        })
    }

    pub(crate) fn to_vulkan_viewports_scissors(
        &self,
        device: &Device,
        dynamic_state_modes: &mut FnvHashMap<DynamicState, bool>,
    ) -> Result<
        (
            u32,
            SmallVec<[ash::vk::Viewport; 2]>,
            u32,
            SmallVec<[ash::vk::Rect2D; 2]>,
        ),
        GraphicsPipelineCreationError,
    > {
        Ok(match self {
            ViewportState::Fixed { data } => {
                let count = data.len() as u32;
                assert!(count != 0); // TODO: return error?
                let viewports = data
                    .iter()
                    .map(|e| e.0.clone().into())
                    .collect::<SmallVec<[ash::vk::Viewport; 2]>>();
                dynamic_state_modes.insert(DynamicState::Viewport, false);
                dynamic_state_modes.insert(DynamicState::ViewportWithCount, false);

                let scissors = data
                    .iter()
                    .map(|e| e.1.clone().into())
                    .collect::<SmallVec<[ash::vk::Rect2D; 2]>>();
                dynamic_state_modes.insert(DynamicState::Scissor, false);
                dynamic_state_modes.insert(DynamicState::ScissorWithCount, false);

                (count, viewports, count, scissors)
            }
            ViewportState::FixedViewport {
                viewports,
                scissor_count_dynamic,
            } => {
                let viewport_count = viewports.len() as u32;
                assert!(viewport_count != 0); // TODO: return error?
                let viewports = viewports
                    .iter()
                    .map(|e| e.clone().into())
                    .collect::<SmallVec<[ash::vk::Viewport; 2]>>();
                dynamic_state_modes.insert(DynamicState::Viewport, false);
                dynamic_state_modes.insert(DynamicState::ViewportWithCount, false);

                let scissor_count = if *scissor_count_dynamic {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "ViewportState::FixedViewport::scissor_count_dynamic was set to true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::Scissor, false);
                    dynamic_state_modes.insert(DynamicState::ScissorWithCount, true);
                    0
                } else {
                    dynamic_state_modes.insert(DynamicState::Scissor, true);
                    dynamic_state_modes.insert(DynamicState::ScissorWithCount, false);
                    viewport_count
                };

                (viewport_count, viewports, scissor_count, SmallVec::new())
            }
            ViewportState::FixedScissor {
                scissors,
                viewport_count_dynamic,
            } => {
                let scissor_count = scissors.len() as u32;
                assert!(scissor_count != 0); // TODO: return error?
                let scissors = scissors
                    .iter()
                    .map(|e| e.clone().into())
                    .collect::<SmallVec<[ash::vk::Rect2D; 2]>>();
                dynamic_state_modes.insert(DynamicState::Scissor, false);
                dynamic_state_modes.insert(DynamicState::ScissorWithCount, false);

                let viewport_count = if *viewport_count_dynamic {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "ViewportState::FixedScissor::viewport_count_dynamic was set to true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::Viewport, false);
                    dynamic_state_modes.insert(DynamicState::ViewportWithCount, true);
                    0
                } else {
                    dynamic_state_modes.insert(DynamicState::Viewport, true);
                    dynamic_state_modes.insert(DynamicState::ViewportWithCount, false);
                    scissor_count
                };

                (viewport_count, SmallVec::new(), scissor_count, scissors)
            }
            ViewportState::Dynamic {
                count,
                viewport_count_dynamic,
                scissor_count_dynamic,
            } => {
                if !(*viewport_count_dynamic && *scissor_count_dynamic) {
                    assert!(*count != 0); // TODO: return error?
                }

                let viewport_count = if *viewport_count_dynamic {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason:
                                "ViewportState::Dynamic::viewport_count_dynamic was set to true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::Viewport, false);
                    dynamic_state_modes.insert(DynamicState::ViewportWithCount, true);
                    0
                } else {
                    dynamic_state_modes.insert(DynamicState::Viewport, true);
                    dynamic_state_modes.insert(DynamicState::ViewportWithCount, false);
                    *count
                };

                let scissor_count = if *scissor_count_dynamic {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "ViewportState::Dynamic::scissor_count_dynamic was set to true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::Scissor, false);
                    dynamic_state_modes.insert(DynamicState::ScissorWithCount, true);
                    0
                } else {
                    dynamic_state_modes.insert(DynamicState::Scissor, true);
                    dynamic_state_modes.insert(DynamicState::ScissorWithCount, false);
                    *count
                };

                (
                    viewport_count,
                    SmallVec::new(),
                    scissor_count,
                    SmallVec::new(),
                )
            }
        })
    }

    pub(crate) fn to_vulkan(
        &self,
        device: &Device,
        dynamic_state_modes: &mut FnvHashMap<DynamicState, bool>,
        viewport_count: u32,
        viewports: &[ash::vk::Viewport],
        scissor_count: u32,
        scissors: &[ash::vk::Rect2D],
    ) -> Result<ash::vk::PipelineViewportStateCreateInfo, GraphicsPipelineCreationError> {
        let viewport_scissor_count = u32::max(viewport_count, scissor_count);

        if viewport_scissor_count > 1 && !device.enabled_features().multi_viewport {
            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                feature: "multi_viewport",
                reason: "ViewportState viewport/scissor count was greater than 1",
            });
        }

        if viewport_scissor_count > device.physical_device().properties().max_viewports {
            return Err(GraphicsPipelineCreationError::MaxViewportsExceeded {
                obtained: viewport_scissor_count,
                max: device.physical_device().properties().max_viewports,
            });
        }

        for vp in viewports.iter() {
            if vp.width
                > device
                    .physical_device()
                    .properties()
                    .max_viewport_dimensions[0] as f32
                || vp.height
                    > device
                        .physical_device()
                        .properties()
                        .max_viewport_dimensions[1] as f32
            {
                return Err(GraphicsPipelineCreationError::MaxViewportDimensionsExceeded);
            }

            if vp.x < device.physical_device().properties().viewport_bounds_range[0]
                || vp.x + vp.width > device.physical_device().properties().viewport_bounds_range[1]
                || vp.y < device.physical_device().properties().viewport_bounds_range[0]
                || vp.y + vp.height > device.physical_device().properties().viewport_bounds_range[1]
            {
                return Err(GraphicsPipelineCreationError::ViewportBoundsExceeded);
            }
        }

        Ok(ash::vk::PipelineViewportStateCreateInfo {
            flags: ash::vk::PipelineViewportStateCreateFlags::empty(),
            viewport_count,
            p_viewports: if viewports.is_empty() {
                ptr::null()
            } else {
                viewports.as_ptr()
            }, // validation layer crashes if you just pass the pointer
            scissor_count,
            p_scissors: if scissors.is_empty() {
                ptr::null()
            } else {
                scissors.as_ptr()
            }, // validation layer crashes if you just pass the pointer
            ..Default::default()
        })
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
// FIXME: check that:
//        x + width must be less than or equal to viewportBoundsRange[0]
//        y + height must be less than or equal to viewportBoundsRange[1]
#[derive(Debug, Clone, PartialEq)]
pub struct Viewport {
    /// Coordinates in pixels of the top-left hand corner of the viewport.
    pub origin: [f32; 2],

    /// Dimensions in pixels of the viewport.
    pub dimensions: [f32; 2],

    /// Minimum and maximum values of the depth.
    ///
    /// The values `0.0` to `1.0` of each vertex's Z coordinate will be mapped to this
    /// `depth_range` before being compared to the existing depth value.
    ///
    /// This is equivalents to `glDepthRange` in OpenGL, except that OpenGL uses the Z coordinate
    /// range from `-1.0` to `1.0` instead.
    pub depth_range: Range<f32>,
}

impl From<Viewport> for ash::vk::Viewport {
    #[inline]
    fn from(val: Viewport) -> Self {
        ash::vk::Viewport {
            x: val.origin[0],
            y: val.origin[1],
            width: val.dimensions[0],
            height: val.dimensions[1],
            min_depth: val.depth_range.start,
            max_depth: val.depth_range.end,
        }
    }
}

/// State of a single scissor box.
// FIXME: add a check:
//      Evaluation of (offset.x + extent.width) must not cause a signed integer addition overflow
//      Evaluation of (offset.y + extent.height) must not cause a signed integer addition overflow
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Scissor {
    /// Coordinates in pixels of the top-left hand corner of the box.
    pub origin: [u32; 2],

    /// Dimensions in pixels of the box.
    pub dimensions: [u32; 2],
}

impl Scissor {
    /// Returns a scissor that, when used, will instruct the pipeline to draw to the entire
    /// framebuffer no matter its size.
    #[inline]
    pub fn irrelevant() -> Scissor {
        Scissor {
            origin: [0, 0],
            dimensions: [0x7fffffff, 0x7fffffff],
        }
    }
}

impl Default for Scissor {
    #[inline]
    fn default() -> Scissor {
        Scissor::irrelevant()
    }
}

impl From<Scissor> for ash::vk::Rect2D {
    #[inline]
    fn from(val: Scissor) -> Self {
        ash::vk::Rect2D {
            offset: ash::vk::Offset2D {
                x: val.origin[0] as i32,
                y: val.origin[1] as i32,
            },
            extent: ash::vk::Extent2D {
                width: val.dimensions[0],
                height: val.dimensions[1],
            },
        }
    }
}
