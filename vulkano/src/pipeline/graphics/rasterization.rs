// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Configures how primitives should be converted into collections of fragments.

use crate::{
    device::Device, macros::vulkan_enum, pipeline::StateMode, Requires, RequiresAllOf,
    RequiresOneOf, ValidationError, Version,
};

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

    pub _ne: crate::NonExhaustive,
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
            _ne: crate::NonExhaustive(()),
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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            depth_clamp_enable,
            rasterizer_discard_enable,
            polygon_mode,
            cull_mode,
            front_face,
            ref depth_bias,
            line_width,
            line_rasterization_mode,
            ref line_stipple,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        polygon_mode
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "polygon_mode".into(),
                vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        line_rasterization_mode
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "line_rasterization_mode".into(),
                vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if depth_clamp_enable && !device.enabled_features().depth_clamp {
            return Err(Box::new(ValidationError {
                context: "depth_clamp_enable".into(),
                problem: "is `true`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "depth_clamp",
                )])]),
                vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-depthClampEnable-00782"],
            }));
        }

        if polygon_mode != PolygonMode::Fill && !device.enabled_features().fill_mode_non_solid {
            return Err(Box::new(ValidationError {
                context: "polygon_mode".into(),
                problem: "is not `PolygonMode::Fill`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "fill_mode_non_solid",
                )])]),
                vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-01507"],
            }));
        }

        match rasterizer_discard_enable {
            StateMode::Fixed(false) => {
                if device.enabled_extensions().khr_portability_subset
                    && !device.enabled_features().point_polygons
                    && polygon_mode == PolygonMode::Point
                {
                    return Err(Box::new(ValidationError {
                        problem: "this device is a portability subset device, \
                            `rasterizer_discard_enable` is `false`, and \
                            `polygon_mode` is `PolygonMode::Point`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "point_polygons",
                        )])]),
                        vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-pointPolygons-04458"],
                        ..Default::default()
                    }));
                }
            }
            StateMode::Dynamic => {
                if !(device.api_version() >= Version::V1_3
                    || device.enabled_features().extended_dynamic_state2)
                {
                    return Err(Box::new(ValidationError {
                        context: "rasterizer_discard_enable".into(),
                        problem: "is dynamic".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                            RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                        ]),
                        // vuids?
                        ..Default::default()
                    }));
                }
            }
            _ => (),
        }

        match cull_mode {
            StateMode::Fixed(cull_mode) => {
                cull_mode
                    .validate_device(device)
                    .map_err(|err| ValidationError {
                        context: "cull_mode".into(),
                        vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-cullMode-parameter"],
                        ..ValidationError::from_requirement(err)
                    })?;
            }
            StateMode::Dynamic => {
                if !(device.api_version() >= Version::V1_3
                    || device.enabled_features().extended_dynamic_state)
                {
                    return Err(Box::new(ValidationError {
                        context: "cull_mode".into(),
                        problem: "is dynamic".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                            RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                        ]),
                        // vuids?
                        ..Default::default()
                    }));
                }
            }
        }

        match front_face {
            StateMode::Fixed(front_face) => {
                front_face
                    .validate_device(device)
                    .map_err(|err| ValidationError {
                        context: "front_face".into(),
                        vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-frontFace-parameter"],
                        ..ValidationError::from_requirement(err)
                    })?;
            }
            StateMode::Dynamic => {
                if !(device.api_version() >= Version::V1_3
                    || device.enabled_features().extended_dynamic_state)
                {
                    return Err(Box::new(ValidationError {
                        context: "front_face".into(),
                        problem: "is dynamic".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                            RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                        ]),
                        // vuids?
                        ..Default::default()
                    }));
                }
            }
        }

        if let Some(depth_bias_state) = depth_bias {
            let &DepthBiasState {
                enable_dynamic,
                bias,
            } = depth_bias_state;

            if enable_dynamic
                && !(device.api_version() >= Version::V1_3
                    || device.enabled_features().extended_dynamic_state2)
            {
                return Err(Box::new(ValidationError {
                    context: "depth_bias.enable_dynamic".into(),
                    problem: "is `true`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                        RequiresAllOf(&[Requires::Feature("extended_dynamic_state2")]),
                    ]),
                    // vuids?
                    ..Default::default()
                }));
            }

            if matches!(bias, StateMode::Fixed(bias) if bias.clamp != 0.0)
                && !device.enabled_features().depth_bias_clamp
            {
                return Err(Box::new(ValidationError {
                    context: "depth_bias.bias.clamp".into(),
                    problem: "is not 0.0".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "depth_bias_clamp",
                    )])]),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-00754"],
                }));
            }
        }

        if matches!(line_width, StateMode::Fixed(line_width) if line_width != 1.0)
            && !device.enabled_features().wide_lines
        {
            return Err(Box::new(ValidationError {
                context: "line_width".into(),
                problem: "is not 1.0".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "wide_lines",
                )])]),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-00749"],
            }));
        }

        if line_rasterization_mode != LineRasterizationMode::Default {
            if !device.enabled_extensions().ext_line_rasterization {
                return Err(Box::new(ValidationError {
                    context: "line_rasterization_mode".into(),
                    problem: "`is not `LineRasterizationMode::Default`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_line_rasterization",
                    )])]),
                    ..Default::default()
                }));
            }

            match line_rasterization_mode {
                LineRasterizationMode::Default => (),
                LineRasterizationMode::Rectangular => {
                    if !device.enabled_features().rectangular_lines {
                        return Err(Box::new(ValidationError {
                            context: "line_rasterization_mode".into(),
                            problem: "is `LineRasterizationMode::Rectangular`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "rectangular_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-02768"],
                        }));
                    }
                }
                LineRasterizationMode::Bresenham => {
                    if !device.enabled_features().bresenham_lines {
                        return Err(Box::new(ValidationError {
                            context: "line_rasterization_mode".into(),
                            problem: "is `LineRasterizationMode::Bresenham`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "bresenham_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-02769"],
                        }));
                    }
                }
                LineRasterizationMode::RectangularSmooth => {
                    if !device.enabled_features().smooth_lines {
                        return Err(Box::new(ValidationError {
                            context: "line_rasterization_mode".into(),
                            problem: "is `LineRasterizationMode::RectangularSmooth`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "smooth_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-02770"],
                        }));
                    }
                }
            }
        }

        if let Some(line_stipple) = line_stipple {
            if !device.enabled_extensions().ext_line_rasterization {
                return Err(Box::new(ValidationError {
                    context: "line_stipple".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_line_rasterization",
                    )])]),
                    ..Default::default()
                }));
            }

            if let StateMode::Fixed(line_stipple) = line_stipple {
                let &LineStipple { factor, pattern: _ } = line_stipple;

                if !(1..=256).contains(&factor) {
                    return Err(Box::new(ValidationError {
                        context: "line_stipple.factor".into(),
                        problem: "is not between 1 and 256 inclusive".into(),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-stippledLineEnable-02767"],
                        ..Default::default()
                    }));
                }
            }

            match line_rasterization_mode {
                LineRasterizationMode::Default => {
                    if !device.enabled_features().stippled_rectangular_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::Default`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "stippled_rectangular_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02774"],
                            ..Default::default()
                        }));
                    }

                    if !properties.strict_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::Default`, \
                                but the `strict_lines` property is `false`".into(),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02774"],
                            ..Default::default()
                        }));
                    }
                }
                LineRasterizationMode::Rectangular => {
                    if !device.enabled_features().stippled_rectangular_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::Rectangular`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "stippled_rectangular_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02771"],
                            ..Default::default()
                        }));
                    }
                }
                LineRasterizationMode::Bresenham => {
                    if !device.enabled_features().stippled_bresenham_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::Bresenham`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "stippled_bresenham_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02772"],
                            ..Default::default()
                        }));
                    }
                }
                LineRasterizationMode::RectangularSmooth => {
                    if !device.enabled_features().stippled_smooth_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::RectangularSmooth`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "stippled_smooth_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02773"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
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

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies the culling mode.
    ///
    /// This setting works in pair with `front_face`. The `front_face` setting tells the GPU whether
    /// clockwise or counter-clockwise correspond to the front and the back of each triangle. Then
    /// `cull_mode` lets you specify whether front faces should be discarded, back faces should be
    /// discarded, or none, or both.
    CullMode = CullModeFlags(u32);

    /// No culling.
    None = NONE,

    /// The faces facing the front of the screen (ie. facing the user) will be removed.
    Front = FRONT,

    /// The faces facing the back of the screen will be removed.
    Back = BACK,

    /// All faces will be removed.
    FrontAndBack = FRONT_AND_BACK,
}

impl Default for CullMode {
    #[inline]
    fn default() -> CullMode {
        CullMode::None
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies which triangle orientation corresponds to the front or the triangle.
    FrontFace = FrontFace(i32);

    /// Triangles whose vertices are oriented counter-clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    CounterClockwise = COUNTER_CLOCKWISE,


    /// Triangles whose vertices are oriented clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    Clockwise = CLOCKWISE,
}

impl Default for FrontFace {
    #[inline]
    fn default() -> FrontFace {
        FrontFace::CounterClockwise
    }
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    PolygonMode = PolygonMode(i32);

    // TODO: document
    Fill = FILL,

    // TODO: document
    Line = LINE,

    // TODO: document further
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, unless `rasterizer_discard_enable` is active, the
    /// [`point_polygons`](crate::device::Features::point_polygons)
    /// feature must be enabled on the device.
    Point = POINT,

    /* TODO: enable
    // TODO: document
    FillRectangle = FILL_RECTANGLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_fill_rectangle)]),
    ]),*/
}

impl Default for PolygonMode {
    #[inline]
    fn default() -> PolygonMode {
        PolygonMode::Fill
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The rasterization mode to use for lines.
    LineRasterizationMode = LineRasterizationModeEXT(i32);

    /// If the [`strict_lines`](crate::device::Properties::strict_lines) device property is `true`,
    /// then this is the same as `Rectangular`. Otherwise, lines are drawn as parallelograms.
    ///
    /// If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`strict_lines`](crate::device::Properties::strict_lines) property must be `true` and the
    /// [`stippled_rectangular_lines`](crate::device::Features::stippled_rectangular_lines) feature
    /// must be enabled on the device.
    Default = DEFAULT,

    /// Lines are drawn as if they were rectangles extruded from the line.
    ///
    /// The [`rectangular_lines`](crate::device::Features::rectangular_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_rectangular_lines`](crate::device::Features::stippled_rectangular_lines) must
    /// also be enabled.
    Rectangular = RECTANGULAR,

    /// Lines are drawn by determining which pixel diamonds the line intersects and exits.
    ///
    /// The [`bresenham_lines`](crate::device::Features::bresenham_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_bresenham_lines`](crate::device::Features::stippled_bresenham_lines) must
    /// also be enabled.
    Bresenham = BRESENHAM,

    /// As `Rectangular`, but with alpha falloff.
    ///
    /// The [`smooth_lines`](crate::device::Features::smooth_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_smooth_lines`](crate::device::Features::stippled_smooth_lines) must
    /// also be enabled.
    RectangularSmooth = RECTANGULAR_SMOOTH,
}

impl Default for LineRasterizationMode {
    /// Returns `LineRasterizationMode::Default`.
    #[inline]
    fn default() -> Self {
        Self::Default
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
