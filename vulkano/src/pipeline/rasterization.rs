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

use crate::{
    device::Device,
    pipeline::{DynamicState, GraphicsPipelineCreationError, StateMode},
};
use fnv::FnvHashMap;

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

    pub(crate) fn to_vulkan_line_state(
        &self,
        device: &Device,
        dynamic_state_modes: &mut FnvHashMap<DynamicState, bool>,
    ) -> Result<
        Option<ash::vk::PipelineRasterizationLineStateCreateInfoEXT>,
        GraphicsPipelineCreationError,
    > {
        Ok(if device.enabled_extensions().ext_line_rasterization {
            let line_rasterization_mode = {
                match self.line_rasterization_mode {
                    LineRasterizationMode::Default => (),
                    LineRasterizationMode::Rectangular => {
                        if !device.enabled_features().rectangular_lines {
                            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                feature: "rectangular_lines",
                                reason:
                                    "RasterizationState::line_rasterization_mode was Rectangular",
                            });
                        }
                    }
                    LineRasterizationMode::Bresenham => {
                        if !device.enabled_features().bresenham_lines {
                            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                feature: "bresenham_lines",
                                reason: "RasterizationState::line_rasterization_mode was Bresenham",
                            });
                        }
                    }
                    LineRasterizationMode::RectangularSmooth => {
                        if !device.enabled_features().smooth_lines {
                            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                feature: "smooth_lines",
                                reason:
                                    "RasterizationState::line_rasterization_mode was RectangularSmooth",
                            });
                        }
                    }
                }

                self.line_rasterization_mode.into()
            };

            let (stippled_line_enable, line_stipple_factor, line_stipple_pattern) = if let Some(
                line_stipple,
            ) =
                self.line_stipple
            {
                match self.line_rasterization_mode {
                    LineRasterizationMode::Default => {
                        if !device.enabled_features().stippled_rectangular_lines {
                            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                feature: "stippled_rectangular_lines",
                                reason:
                                    "RasterizationState::line_stipple was Some and line_rasterization_mode was Default",
                            });
                        }

                        if !device.physical_device().properties().strict_lines {
                            return Err(GraphicsPipelineCreationError::StrictLinesNotSupported);
                        }
                    }
                    LineRasterizationMode::Rectangular => {
                        if !device.enabled_features().stippled_rectangular_lines {
                            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                feature: "stippled_rectangular_lines",
                                reason:
                                    "RasterizationState::line_stipple was Some and line_rasterization_mode was Rectangular",
                            });
                        }
                    }
                    LineRasterizationMode::Bresenham => {
                        if !device.enabled_features().stippled_bresenham_lines {
                            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                feature: "stippled_bresenham_lines",
                                reason: "RasterizationState::line_stipple was Some and line_rasterization_mode was Bresenham",
                            });
                        }
                    }
                    LineRasterizationMode::RectangularSmooth => {
                        if !device.enabled_features().stippled_smooth_lines {
                            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                    feature: "stippled_smooth_lines",
                                    reason:
                                        "RasterizationState::line_stipple was Some and line_rasterization_mode was RectangularSmooth",
                                });
                        }
                    }
                }

                let (factor, pattern) = match line_stipple {
                    StateMode::Fixed(line_stipple) => {
                        assert!(line_stipple.factor >= 1 && line_stipple.factor <= 256); // TODO: return error?
                        dynamic_state_modes.insert(DynamicState::LineStipple, false);
                        (line_stipple.factor, line_stipple.pattern)
                    }
                    StateMode::Dynamic => {
                        dynamic_state_modes.insert(DynamicState::LineStipple, true);
                        (1, 0)
                    }
                };

                (ash::vk::TRUE, factor, pattern)
            } else {
                (ash::vk::FALSE, 1, 0)
            };

            Some(ash::vk::PipelineRasterizationLineStateCreateInfoEXT {
                line_rasterization_mode,
                stippled_line_enable,
                line_stipple_factor,
                line_stipple_pattern,
                ..Default::default()
            })
        } else {
            if self.line_rasterization_mode != LineRasterizationMode::Default {
                return Err(GraphicsPipelineCreationError::ExtensionNotEnabled {
                    extension: "ext_line_rasterization",
                    reason: "RasterizationState::line_rasterization_mode was not Default",
                });
            }

            if self.line_stipple.is_some() {
                return Err(GraphicsPipelineCreationError::ExtensionNotEnabled {
                    extension: "ext_line_rasterization",
                    reason: "RasterizationState::line_stipple was not None",
                });
            }

            None
        })
    }

    pub(crate) fn to_vulkan(
        &self,
        device: &Device,
        dynamic_state_modes: &mut FnvHashMap<DynamicState, bool>,
        rasterization_line_state: Option<&mut ash::vk::PipelineRasterizationLineStateCreateInfoEXT>,
    ) -> Result<ash::vk::PipelineRasterizationStateCreateInfo, GraphicsPipelineCreationError> {
        if self.depth_clamp_enable && !device.enabled_features().depth_clamp {
            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                feature: "depth_clamp",
                reason: "RasterizationState::depth_clamp_enable was true",
            });
        }

        let rasterizer_discard_enable = match self.rasterizer_discard_enable {
            StateMode::Fixed(rasterizer_discard_enable) => {
                dynamic_state_modes.insert(DynamicState::RasterizerDiscardEnable, false);
                rasterizer_discard_enable as ash::vk::Bool32
            }
            StateMode::Dynamic => {
                if !device.enabled_features().extended_dynamic_state2 {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "extended_dynamic_state2",
                        reason: "RasterizationState::rasterizer_discard_enable was set to Dynamic",
                    });
                }
                dynamic_state_modes.insert(DynamicState::RasterizerDiscardEnable, true);
                ash::vk::FALSE
            }
        };

        if self.polygon_mode != PolygonMode::Fill && !device.enabled_features().fill_mode_non_solid
        {
            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                feature: "fill_mode_non_solid",
                reason: "RasterizationState::polygon_mode was not Fill",
            });
        }

        let cull_mode = match self.cull_mode {
            StateMode::Fixed(cull_mode) => {
                dynamic_state_modes.insert(DynamicState::CullMode, false);
                cull_mode.into()
            }
            StateMode::Dynamic => {
                if !device.enabled_features().extended_dynamic_state {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "extended_dynamic_state",
                        reason: "RasterizationState::cull_mode was set to Dynamic",
                    });
                }
                dynamic_state_modes.insert(DynamicState::CullMode, true);
                CullMode::default().into()
            }
        };

        let front_face = match self.front_face {
            StateMode::Fixed(front_face) => {
                dynamic_state_modes.insert(DynamicState::FrontFace, false);
                front_face.into()
            }
            StateMode::Dynamic => {
                if !device.enabled_features().extended_dynamic_state {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "extended_dynamic_state",
                        reason: "RasterizationState::front_face was set to Dynamic",
                    });
                }
                dynamic_state_modes.insert(DynamicState::FrontFace, true);
                FrontFace::default().into()
            }
        };

        let (
            depth_bias_enable,
            depth_bias_constant_factor,
            depth_bias_clamp,
            depth_bias_slope_factor,
        ) = if let Some(depth_bias_state) = self.depth_bias {
            if depth_bias_state.enable_dynamic {
                if !device.enabled_features().extended_dynamic_state2 {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "extended_dynamic_state2",
                        reason: "DepthBiasState::enable_dynamic was true",
                    });
                }
                dynamic_state_modes.insert(DynamicState::DepthTestEnable, true);
            } else {
                dynamic_state_modes.insert(DynamicState::DepthBiasEnable, false);
            }

            let (constant_factor, clamp, slope_factor) = match depth_bias_state.bias {
                StateMode::Fixed(bias) => {
                    if bias.clamp != 0.0 && !device.enabled_features().depth_bias_clamp {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "depth_bias_clamp",
                            reason: "DepthBias::clamp was not 0.0",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::DepthBias, false);
                    (bias.constant_factor, bias.clamp, bias.slope_factor)
                }
                StateMode::Dynamic => {
                    dynamic_state_modes.insert(DynamicState::DepthBias, true);
                    (0.0, 0.0, 0.0)
                }
            };

            (ash::vk::TRUE, constant_factor, clamp, slope_factor)
        } else {
            (ash::vk::FALSE, 0.0, 0.0, 0.0)
        };

        let line_width = match self.line_width {
            StateMode::Fixed(line_width) => {
                if line_width != 1.0 && !device.enabled_features().wide_lines {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "wide_lines",
                        reason: "RasterizationState::line_width was not 1.0",
                    });
                }
                dynamic_state_modes.insert(DynamicState::LineWidth, false);
                line_width
            }
            StateMode::Dynamic => {
                dynamic_state_modes.insert(DynamicState::LineWidth, true);
                1.0
            }
        };

        let mut rasterization_state = ash::vk::PipelineRasterizationStateCreateInfo {
            flags: ash::vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: self.depth_clamp_enable as ash::vk::Bool32,
            rasterizer_discard_enable,
            polygon_mode: self.polygon_mode.into(),
            cull_mode,
            front_face,
            depth_bias_enable,
            depth_bias_constant_factor,
            depth_bias_clamp,
            depth_bias_slope_factor,
            line_width,
            ..Default::default()
        };

        if let Some(rasterization_line_state) = rasterization_line_state {
            rasterization_line_state.p_next = rasterization_state.p_next;
            rasterization_state.p_next = rasterization_line_state as *const _ as *const _;
        }

        Ok(rasterization_state)
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
