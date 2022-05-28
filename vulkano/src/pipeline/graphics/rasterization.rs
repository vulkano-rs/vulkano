// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Configures how primitives should be converted into collections of fragments.

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
    /// If set to `Dynamic`, the device API version must be at least 1.3, or the
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
    /// If set to `Dynamic`, the device API version must be at least 1.3, or the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub cull_mode: StateMode<CullMode>,

    /// Specifies which triangle orientation is considered to be the front of the triangle.
    ///
    /// If set to `Dynamic`, the device API version must be at least 1.3, or the
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

    /// The rasterization mode for lines.
    ///
    /// If this is not set to `Default`, the
    /// [`ext_line_rasterization`](crate::device::DeviceExtensions::ext_line_rasterization)
    /// extension and an additional feature must be enabled on the device.
    pub line_rasterization_mode: LineRasterizationMode,

    /// Enables and sets the parameters for line stippling.
    ///
    /// If this is set to `Some`, the
    /// [`ext_line_rasterization`](crate::device::DeviceExtensions::ext_line_rasterization)
    /// extension and an additional feature must be enabled on the device.
    pub line_stipple: Option<StateMode<LineStipple>>,
}

impl RasterizationState {
    /// Creates a `RasterizationState` with depth clamping, discard, depth biasing and line
    /// stippling disabled, filled polygons, no culling, counterclockwise front face, and the
    /// default line width and line rasterization mode.
    #[inline]
    pub fn new() -> Self {
        Self {
            depth_clamp_enable: false,
            rasterizer_discard_enable: StateMode::Fixed(false),
            polygon_mode: Default::default(),
            cull_mode: StateMode::Fixed(Default::default()),
            front_face: StateMode::Fixed(Default::default()),
            depth_bias: None,
            line_width: StateMode::Fixed(1.0),
            line_rasterization_mode: Default::default(),
            line_stipple: None,
        }
    }

    /// Sets the polygon mode.
    #[inline]
    pub fn polygon_mode(mut self, polygon_mode: PolygonMode) -> Self {
        self.polygon_mode = polygon_mode;
        self
    }

    /// Sets the cull mode.
    #[inline]
    pub fn cull_mode(mut self, cull_mode: CullMode) -> Self {
        self.cull_mode = StateMode::Fixed(cull_mode);
        self
    }

    /// Sets the cull mode to dynamic.
    #[inline]
    pub fn cull_mode_dynamic(mut self) -> Self {
        self.cull_mode = StateMode::Dynamic;
        self
    }

    /// Sets the front face.
    #[inline]
    pub fn front_face(mut self, front_face: FrontFace) -> Self {
        self.front_face = StateMode::Fixed(front_face);
        self
    }

    /// Sets the front face to dynamic.
    #[inline]
    pub fn front_face_dynamic(mut self) -> Self {
        self.front_face = StateMode::Dynamic;
        self
    }
}

impl Default for RasterizationState {
    /// Returns [`RasterizationState::new()`].
    #[inline]
    fn default() -> Self {
        Self::new()
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

/// The rasterization mode to use for lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum LineRasterizationMode {
    /// If the [`strict_lines`](crate::device::Properties::strict_lines) device property is `true`,
    /// then this is the same as `Rectangular`. Otherwise, lines are drawn as parallelograms.
    ///
    /// If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`strict_lines`](crate::device::Properties::strict_lines) property must be `true` and the
    /// [`stippled_rectangular_lines`](crate::device::Features::stippled_rectangular_lines) feature
    /// must be enabled on the device.
    Default = ash::vk::LineRasterizationModeEXT::DEFAULT.as_raw(),

    /// Lines are drawn as if they were rectangles extruded from the line.
    ///
    /// The [`rectangular_lines`](crate::device::Features::rectangular_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_rectangular_lines`](crate::device::Features::stippled_rectangular_lines) must
    /// also be enabled.
    Rectangular = ash::vk::LineRasterizationModeEXT::RECTANGULAR.as_raw(),

    /// Lines are drawn by determining which pixel diamonds the line intersects and exits.
    ///
    /// The [`bresenham_lines`](crate::device::Features::bresenham_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_bresenham_lines`](crate::device::Features::stippled_bresenham_lines) must
    /// also be enabled.
    Bresenham = ash::vk::LineRasterizationModeEXT::BRESENHAM.as_raw(),

    /// As `Rectangular`, but with alpha falloff.
    ///
    /// The [`smooth_lines`](crate::device::Features::smooth_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_smooth_lines`](crate::device::Features::stippled_smooth_lines) must
    /// also be enabled.
    RectangularSmooth = ash::vk::LineRasterizationModeEXT::RECTANGULAR_SMOOTH.as_raw(),
}

impl Default for LineRasterizationMode {
    /// Returns `LineRasterizationMode::Default`.
    fn default() -> Self {
        Self::Default
    }
}

impl From<LineRasterizationMode> for ash::vk::LineRasterizationModeEXT {
    fn from(val: LineRasterizationMode) -> Self {
        Self::from_raw(val as i32)
    }
}

/// The parameters of a stippled line.
#[derive(Clone, Copy, Debug)]
pub struct LineStipple {
    /// The repeat factor used in stippled line rasterization. Must be between 1 and 256 inclusive.
    pub factor: u32,
    /// The bit pattern used in stippled line rasterization.
    pub pattern: u16,
}
