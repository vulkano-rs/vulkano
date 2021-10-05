// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Stage when triangles are turned into pixels.
//!
//! The rasterization is the stage when collections of triangles are turned into collections
//! of pixels or samples.

use crate::pipeline::StateMode;

/// The state in a graphics pipeline describing how the rasterization stage should behave.
#[derive(Clone, Debug)]
pub struct RasterizationState {
    /// If true, then the depth value of the vertices will be clamped to the range [0.0, 1.0]. If
    /// false, fragments whose depth is outside of this range will be discarded.
    ///
    /// If enabled, the [`depth_clamp`](crate::device::Features::depth_clamp) feature must be
    /// enabled on the device.
    pub depth_clamp_enable: bool,

    /// If true, all the fragments will be discarded, and the fragment shader will not be run. This
    /// is usually used when your vertex shader has some side effects and you don't need to run the
    /// fragment shader.
    ///
    /// If set to `Dynamic`, the
    /// [`extended_dynamic_state2`](crate::device::Features::extended_dynamic_state2) feature must
    /// be enabled on the device.
    pub rasterizer_discard_enable: StateMode<bool>,

    /// This setting can ask the rasterizer to downgrade triangles into lines or points, or lines
    /// into points.
    ///
    /// If set to a value other than `Fill`, the
    /// [`fill_mode_non_solid`](crate::device::Features::fill_mode_non_solid) feature must be
    /// enabled on the device.
    pub polygon_mode: PolygonMode,

    /// Specifies whether front faces or back faces should be discarded, or none, or both.
    ///
    /// If set to `Dynamic`, the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub cull_mode: StateMode<CullMode>,

    /// Specifies which triangle orientation is considered to be the front of the triangle.
    ///
    /// If set to `Dynamic`, the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub front_face: StateMode<FrontFace>,

    /// Sets how to modify depth values in the rasterization stage.
    ///
    /// If set to `None`, depth biasing is disabled, the depth values will pass to the fragment
    /// shader unmodified.
    pub depth_bias: Option<DepthBiasState>,

    /// Width, in pixels, of lines when drawing lines.
    ///
    /// Setting this to a value other than 1.0 requires the
    /// [`wide_lines`](crate::device::Features::wide_lines) feature to be enabled on
    /// the device.
    pub line_width: StateMode<f32>,
}

impl Default for RasterizationState {
    /// Creates a `RasterizationState` with depth clamping, discard and depth biasing disabled,
    /// filled polygons, no culling, counterclockwise front face, and the default line width.
    #[inline]
    fn default() -> Self {
        Self {
            depth_clamp_enable: false,
            rasterizer_discard_enable: StateMode::Fixed(false),
            polygon_mode: Default::default(),
            cull_mode: StateMode::Fixed(Default::default()),
            front_face: StateMode::Fixed(Default::default()),
            depth_bias: None,
            line_width: StateMode::Fixed(1.0),
        }
    }
}

/// The state in a graphics pipeline describing how depth biasing should behave when enabled.
#[derive(Clone, Copy, Debug)]
pub struct DepthBiasState {
    /// Sets whether depth biasing should be enabled and disabled dynamically. If set to `false`,
    /// depth biasing is always enabled.
    ///
    /// If set to `true`, the
    /// [`extended_dynamic_state2`](crate::device::Features::extended_dynamic_state2) feature must
    /// be enabled on the device.
    pub enable_dynamic: bool,

    /// The values to use when depth biasing is enabled.
    pub bias: StateMode<DepthBias>,
}

/// The values to use for depth biasing.
#[derive(Clone, Copy, Debug)]
pub struct DepthBias {
    /// Specifies a constant factor to be added to every depth value.
    pub constant_factor: f32,

    /// The maximum (or minimum) depth bias of a fragment.
    ///
    /// Setting this to a value other than 0.0 requires the
    /// [`depth_bias_clamp`](crate::device::Features::depth_bias_clamp) feature to be enabled on
    /// the device.
    pub clamp: f32,

    /// A scalar factor applied to a fragment's slope in depth bias calculations.
    pub slope_factor: f32,
}

/// Specifies the culling mode.
///
/// This setting works in pair with `front_face`. The `front_face` setting tells the GPU whether
/// clockwise or counter-clockwise correspond to the front and the back of each triangle. Then
/// `cull_mode` lets you specify whether front faces should be discarded, back faces should be
/// discarded, or none, or both.
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum CullMode {
    /// No culling.
    None = ash::vk::CullModeFlags::NONE.as_raw(),
    /// The faces facing the front of the screen (ie. facing the user) will be removed.
    Front = ash::vk::CullModeFlags::FRONT.as_raw(),
    /// The faces facing the back of the screen will be removed.
    Back = ash::vk::CullModeFlags::BACK.as_raw(),
    /// All faces will be removed.
    FrontAndBack = ash::vk::CullModeFlags::FRONT_AND_BACK.as_raw(),
}

impl From<CullMode> for ash::vk::CullModeFlags {
    #[inline]
    fn from(val: CullMode) -> Self {
        Self::from_raw(val as u32)
    }
}

impl Default for CullMode {
    #[inline]
    fn default() -> CullMode {
        CullMode::None
    }
}

/// Specifies which triangle orientation corresponds to the front or the triangle.
#[derive(Copy, Clone, Debug)]
#[repr(i32)]
pub enum FrontFace {
    /// Triangles whose vertices are oriented counter-clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    CounterClockwise = ash::vk::FrontFace::COUNTER_CLOCKWISE.as_raw(),

    /// Triangles whose vertices are oriented clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    Clockwise = ash::vk::FrontFace::CLOCKWISE.as_raw(),
}

impl From<FrontFace> for ash::vk::FrontFace {
    #[inline]
    fn from(val: FrontFace) -> Self {
        Self::from_raw(val as i32)
    }
}

impl Default for FrontFace {
    #[inline]
    fn default() -> FrontFace {
        FrontFace::CounterClockwise
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum PolygonMode {
    Fill = ash::vk::PolygonMode::FILL.as_raw(),
    Line = ash::vk::PolygonMode::LINE.as_raw(),
    Point = ash::vk::PolygonMode::POINT.as_raw(),
}

impl From<PolygonMode> for ash::vk::PolygonMode {
    #[inline]
    fn from(val: PolygonMode) -> Self {
        Self::from_raw(val as i32)
    }
}

impl Default for PolygonMode {
    #[inline]
    fn default() -> PolygonMode {
        PolygonMode::Fill
    }
}
