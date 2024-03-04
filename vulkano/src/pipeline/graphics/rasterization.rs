//! Configures how primitives should be converted into collections of fragments.

use crate::{
    device::Device, macros::vulkan_enum, Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};

/// The state in a graphics pipeline describing how the rasterization stage should behave.
#[derive(Clone, Debug)]
pub struct RasterizationState {
    /// If true, then the depth value of the vertices will be clamped to the range [0.0, 1.0]. If
    /// false, fragments whose depth is outside of this range will be discarded.
    ///
    /// If enabled, the [`depth_clamp`](crate::device::DeviceFeatures::depth_clamp) feature must be
    /// enabled on the device.
    ///
    /// The default value is `false`.
    pub depth_clamp_enable: bool,

    /// If true, all the fragments will be discarded, and the fragment shader will not be run. This
    /// is usually used when your vertex shader has some side effects and you don't need to run the
    /// fragment shader.
    ///
    /// The default value is `false`.
    pub rasterizer_discard_enable: bool,

    /// This setting can ask the rasterizer to downgrade triangles into lines or points, or lines
    /// into points.
    ///
    /// If set to a value other than `Fill`, the
    /// [`fill_mode_non_solid`](crate::device::DeviceFeatures::fill_mode_non_solid) feature must be
    /// enabled on the device.
    ///
    /// The default value is [`PolygonMode::Fill`].
    pub polygon_mode: PolygonMode,

    /// Specifies whether front faces or back faces should be discarded, or none, or both.
    ///
    /// The default value is [`CullMode::None`].
    pub cull_mode: CullMode,

    /// Specifies which triangle orientation is considered to be the front of the triangle.
    ///
    /// The default value is [`FrontFace::CounterClockwise`].
    pub front_face: FrontFace,

    /// Sets how to modify depth values in the rasterization stage.
    ///
    /// If set to `None`, depth biasing is disabled, the depth values will pass to the fragment
    /// shader unmodified.
    ///
    /// The default value is `None`.
    pub depth_bias: Option<DepthBiasState>,

    /// Width, in pixels, of lines when drawing lines.
    ///
    /// Setting this to a value other than 1.0 requires the
    /// [`wide_lines`](crate::device::DeviceFeatures::wide_lines) feature to be enabled on
    /// the device.
    ///
    /// The default value is `1.0`.
    pub line_width: f32,

    /// The rasterization mode for lines.
    ///
    /// If this is not set to `Default`, the
    /// [`ext_line_rasterization`](crate::device::DeviceExtensions::ext_line_rasterization)
    /// extension and an additional feature must be enabled on the device.
    ///
    /// The default value is [`LineRasterizationMode::Default`].
    pub line_rasterization_mode: LineRasterizationMode,

    /// Enables and sets the parameters for line stippling.
    ///
    /// If this is set to `Some`, the
    /// [`ext_line_rasterization`](crate::device::DeviceExtensions::ext_line_rasterization)
    /// extension and an additional feature must be enabled on the device.
    ///
    /// The default value is `None`.
    pub line_stipple: Option<LineStipple>,

    pub _ne: crate::NonExhaustive,
}

impl Default for RasterizationState {
    #[inline]
    fn default() -> Self {
        Self {
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            polygon_mode: Default::default(),
            cull_mode: Default::default(),
            front_face: Default::default(),
            depth_bias: None,
            line_width: 1.0,
            line_rasterization_mode: Default::default(),
            line_stipple: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl RasterizationState {
    /// Creates a `RasterizationState` with depth clamping, discard, depth biasing and line
    /// stippling disabled, filled polygons, no culling, counterclockwise front face, and the
    /// default line width and line rasterization mode.
    #[inline]
    #[deprecated(since = "0.34.0", note = "use `RasterizationState::default` instead")]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the polygon mode.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn polygon_mode(mut self, polygon_mode: PolygonMode) -> Self {
        self.polygon_mode = polygon_mode;
        self
    }

    /// Sets the cull mode.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn cull_mode(mut self, cull_mode: CullMode) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    /// Sets the front face.
    #[inline]
    #[deprecated(since = "0.34.0")]
    pub fn front_face(mut self, front_face: FrontFace) -> Self {
        self.front_face = front_face;
        self
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            depth_clamp_enable,
            rasterizer_discard_enable,
            polygon_mode,
            cull_mode,
            front_face,
            depth_bias: _,
            line_width: _,
            line_rasterization_mode,
            ref line_stipple,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        polygon_mode.validate_device(device).map_err(|err| {
            err.add_context("polygon_mode")
                .set_vuids(&["VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-parameter"])
        })?;

        line_rasterization_mode
            .validate_device(device)
            .map_err(|err| err.add_context("line_rasterization_mode").set_vuids(&["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-parameter"]))?;

        if depth_clamp_enable && !device.enabled_features().depth_clamp {
            return Err(Box::new(ValidationError {
                context: "depth_clamp_enable".into(),
                problem: "is `true`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "depth_clamp",
                )])]),
                vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-depthClampEnable-00782"],
            }));
        }

        if polygon_mode != PolygonMode::Fill && !device.enabled_features().fill_mode_non_solid {
            return Err(Box::new(ValidationError {
                context: "polygon_mode".into(),
                problem: "is not `PolygonMode::Fill`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "fill_mode_non_solid",
                )])]),
                vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-01507"],
            }));
        }

        if device.enabled_extensions().khr_portability_subset
            && !device.enabled_features().point_polygons
            && !rasterizer_discard_enable
            && polygon_mode == PolygonMode::Point
        {
            return Err(Box::new(ValidationError {
                problem: "this device is a portability subset device, \
                    `rasterizer_discard_enable` is `false`, and \
                    `polygon_mode` is `PolygonMode::Point`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "point_polygons",
                )])]),
                vuids: &["VUID-VkPipelineRasterizationStateCreateInfo-pointPolygons-04458"],
                ..Default::default()
            }));
        }

        cull_mode.validate_device(device).map_err(|err| {
            err.add_context("cull_mode")
                .set_vuids(&["VUID-VkPipelineRasterizationStateCreateInfo-cullMode-parameter"])
        })?;

        front_face.validate_device(device).map_err(|err| {
            err.add_context("front_face")
                .set_vuids(&["VUID-VkPipelineRasterizationStateCreateInfo-frontFace-parameter"])
        })?;

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
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
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
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
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
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "smooth_lines",
                            )])]),
                            vuids: &["VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-02770"],
                        }));
                    }
                }
            }
        }

        if line_stipple.is_some() {
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

            match line_rasterization_mode {
                LineRasterizationMode::Default => {
                    if !device.enabled_features().stippled_rectangular_lines {
                        return Err(Box::new(ValidationError {
                            problem: "`line_stipple` is `Some`, and \
                                `line_rasterization_mode` is \
                                `LineRasterizationMode::Default`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
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
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
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
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
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
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
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

/// The values to use for depth biasing.
#[derive(Clone, Copy, Debug)]
pub struct DepthBiasState {
    /// Specifies a constant factor to be multiplied to every depth value.
    ///
    /// The default value is `1.0`.
    pub constant_factor: f32,

    /// The maximum (or minimum) depth bias of a fragment.
    ///
    /// Setting this to a value other than 0.0 requires the
    /// [`depth_bias_clamp`](crate::device::DeviceFeatures::depth_bias_clamp) feature to be enabled
    /// on the device.
    ///
    /// The default value is `0.0`.
    pub clamp: f32,

    /// A scalar factor to multiply with a fragment's slope in depth bias calculations.
    ///
    /// The default value is `1.0`.
    pub slope_factor: f32,
}

impl Default for DepthBiasState {
    #[inline]
    fn default() -> Self {
        Self {
            constant_factor: 1.0,
            clamp: 0.0,
            slope_factor: 1.0,
        }
    }
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
    /// [`point_polygons`](crate::device::DeviceFeatures::point_polygons)
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

    /// If the [`strict_lines`](crate::device::DeviceProperties::strict_lines) device property is `true`,
    /// then this is the same as `Rectangular`. Otherwise, lines are drawn as parallelograms.
    ///
    /// If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`strict_lines`](crate::device::DeviceProperties::strict_lines) property must be `true` and the
    /// [`stippled_rectangular_lines`](crate::device::DeviceFeatures::stippled_rectangular_lines) feature
    /// must be enabled on the device.
    Default = DEFAULT,

    /// Lines are drawn as if they were rectangles extruded from the line.
    ///
    /// The [`rectangular_lines`](crate::device::DeviceFeatures::rectangular_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_rectangular_lines`](crate::device::DeviceFeatures::stippled_rectangular_lines) must
    /// also be enabled.
    Rectangular = RECTANGULAR,

    /// Lines are drawn by determining which pixel diamonds the line intersects and exits.
    ///
    /// The [`bresenham_lines`](crate::device::DeviceFeatures::bresenham_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_bresenham_lines`](crate::device::DeviceFeatures::stippled_bresenham_lines) must
    /// also be enabled.
    Bresenham = BRESENHAM,

    /// As `Rectangular`, but with alpha falloff.
    ///
    /// The [`smooth_lines`](crate::device::DeviceFeatures::smooth_lines) feature must be
    /// enabled on the device. If [`RasterizationState::line_stipple`] is `Some`, then the
    /// [`stippled_smooth_lines`](crate::device::DeviceFeatures::stippled_smooth_lines) must
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
