// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Configures how the color output of the fragment shader is written to the attachment.
//!
//! # Blending in details
//!
//! There are three kinds of color attachments for the purpose of blending:
//!
//! - Attachments with a floating-point or fixed point format.
//! - Attachments with a (non-normalized) integer format.
//! - Attachments with a normalized integer format.
//!
//! For floating-point and fixed-point formats, the blending operation is applied. For integer
//! formats, the logic operation is applied. For normalized integer formats, the logic operation
//! will take precedence if it is activated, otherwise the blending operation is applied.

use crate::{
    device::Device,
    macros::{vulkan_bitflags, vulkan_enum},
    pipeline::StateMode,
    Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};

/// Describes how the color output of the fragment shader is written to the attachment. See the
/// documentation of the `blend` module for more info.
#[derive(Clone, Debug)]
pub struct ColorBlendState {
    /// Additional properties of the color blend state.
    ///
    /// The default value is empty.
    pub flags: ColorBlendStateFlags,

    /// Sets the logical operation to perform between the incoming fragment color and the existing
    /// fragment in the framebuffer attachment.
    ///
    /// If set to `Some`, the [`logic_op`](crate::device::Features::logic_op) feature must be
    /// enabled on the device. If set to `Some(Dynamic)`, then the
    /// [`extended_dynamic_state2_logic_op`](crate::device::Features::extended_dynamic_state2_logic_op)
    /// feature must also be enabled on the device.
    pub logic_op: Option<StateMode<LogicOp>>,

    /// Sets the blend and output state for each color attachment. The number of elements must match
    /// the number of color attachments in the framebuffer.
    ///
    /// If there are multiple elements, and the `blend` and `color_write_mask` members of each
    /// element differ, then the [`independent_blend`](crate::device::Features::independent_blend)
    /// feature must be enabled on the device.
    pub attachments: Vec<ColorBlendAttachmentState>,

    /// The constant color to use for some of the `BlendFactor` variants.
    pub blend_constants: StateMode<[f32; 4]>,
}

impl ColorBlendState {
    /// Creates a `ColorBlendState` with logical operations disabled, blend constants set to zero,
    /// and `num` attachment entries that have blending disabled, and color write and all color
    /// components enabled.
    #[inline]
    pub fn new(num: u32) -> Self {
        Self {
            flags: ColorBlendStateFlags::empty(),
            logic_op: None,
            attachments: (0..num)
                .map(|_| ColorBlendAttachmentState {
                    blend: None,
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: StateMode::Fixed(true),
                })
                .collect(),
            blend_constants: StateMode::Fixed([0.0, 0.0, 0.0, 0.0]),
        }
    }

    /// Enables logical operations with the given logical operation.
    #[inline]
    pub fn logic_op(mut self, logic_op: LogicOp) -> Self {
        self.logic_op = Some(StateMode::Fixed(logic_op));
        self
    }

    /// Enables logical operations with a dynamic logical operation.
    #[inline]
    pub fn logic_op_dynamic(mut self) -> Self {
        self.logic_op = Some(StateMode::Dynamic);
        self
    }

    /// Enables blending for all attachments, with the given parameters.
    #[inline]
    pub fn blend(mut self, blend: AttachmentBlend) -> Self {
        self.attachments
            .iter_mut()
            .for_each(|attachment_state| attachment_state.blend = Some(blend));
        self
    }

    /// Enables blending for all attachments, with alpha blending.
    #[inline]
    pub fn blend_alpha(mut self) -> Self {
        self.attachments
            .iter_mut()
            .for_each(|attachment_state| attachment_state.blend = Some(AttachmentBlend::alpha()));
        self
    }

    /// Enables blending for all attachments, with additive blending.
    #[inline]
    pub fn blend_additive(mut self) -> Self {
        self.attachments.iter_mut().for_each(|attachment_state| {
            attachment_state.blend = Some(AttachmentBlend::additive())
        });
        self
    }

    /// Sets the color write mask for all attachments.
    #[inline]
    pub fn color_write_mask(mut self, color_write_mask: ColorComponents) -> Self {
        self.attachments
            .iter_mut()
            .for_each(|attachment_state| attachment_state.color_write_mask = color_write_mask);
        self
    }

    /// Sets the blend constants.
    #[inline]
    pub fn blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.blend_constants = StateMode::Fixed(constants);
        self
    }

    /// Sets the blend constants as dynamic.
    #[inline]
    pub fn blend_constants_dynamic(mut self) -> Self {
        self.blend_constants = StateMode::Dynamic;
        self
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            flags,
            logic_op,
            ref attachments,
            blend_constants: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkPipelineColorBlendStateCreateInfo-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if let Some(logic_op) = logic_op {
            if !device.enabled_features().logic_op {
                return Err(ValidationError {
                    context: "logic_op".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "logic_op",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendStateCreateInfo-logicOpEnable-00606"],
                });
            }

            match logic_op {
                StateMode::Fixed(logic_op) => {
                    logic_op
                        .validate_device(device)
                        .map_err(|err| ValidationError {
                            context: "logic_op".into(),
                            vuids: &[
                                "VUID-VkPipelineColorBlendStateCreateInfo-logicOpEnable-00607",
                            ],
                            ..ValidationError::from_requirement(err)
                        })?
                }
                StateMode::Dynamic => {
                    if !device.enabled_features().extended_dynamic_state2_logic_op {
                        return Err(ValidationError {
                            context: "logic_op".into(),
                            problem: "is dynamic".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "extended_dynamic_state2_logic_op",
                            )])]),
                            vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-04869"],
                        });
                    }
                }
            }
        }

        if device.enabled_features().independent_blend {
            for (index, state) in attachments.iter().enumerate() {
                state
                    .validate(device)
                    .map_err(|err| err.add_context(format!("attachments[{}]", index)))?;
            }
        } else if let Some(first) = attachments.first() {
            first
                .validate(device)
                .map_err(|err| err.add_context("attachments[0]"))?;

            for (index, state) in attachments.iter().enumerate().skip(1) {
                if state != first {
                    return Err(ValidationError {
                        problem: format!(
                            "`attachments[{}]` does not equal `attachments[0]`",
                            index
                        )
                        .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "independent_blend",
                        )])]),
                        vuids: &["VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605"],
                        ..Default::default()
                    });
                }
            }
        }

        Ok(())
    }
}

impl Default for ColorBlendState {
    /// Returns [`ColorBlendState::new(1)`].
    #[inline]
    fn default() -> Self {
        Self::new(1)
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of the color blend state.
    ColorBlendStateFlags = PipelineColorBlendStateCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    RASTERIZATION_ORDER_ATTACHMENT_ACCESS = RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_rasterization_order_attachment_access)]),
        RequiresAllOf([DeviceExtension(arm_rasterization_order_attachment_access)]),
    ]), */
}

vulkan_enum! {
    #[non_exhaustive]
    /// Which logical operation to apply to the output values.
    ///
    /// The operation is applied individually for each channel (red, green, blue and alpha).
    ///
    /// Only relevant for integer or unsigned attachments.
    ///
    /// Also note that some implementations don't support logic operations.
    LogicOp = LogicOp(i32);

    /// Returns `0`.
    Clear = CLEAR,

    /// Returns `source & destination`.
    And = AND,

    /// Returns `source & !destination`.
    AndReverse = AND_REVERSE,

    /// Returns `source`.
    Copy = COPY,

    /// Returns `!source & destination`.
    AndInverted = AND_INVERTED,

    /// Returns `destination`.
    Noop = NO_OP,

    /// Returns `source ^ destination`.
    Xor = XOR,

    /// Returns `source | destination`.
    Or = OR,

    /// Returns `!(source | destination)`.
    Nor = NOR,

    /// Returns `!(source ^ destination)`.
    Equivalent = EQUIVALENT,

    /// Returns `!destination`.
    Invert = INVERT,

    /// Returns `source | !destination.
    OrReverse = OR_REVERSE,

    /// Returns `!source`.
    CopyInverted = COPY_INVERTED,

    /// Returns `!source | destination`.
    OrInverted = OR_INVERTED,

    /// Returns `!(source & destination)`.
    Nand = NAND,

    /// Returns `!0` (all bits set to 1).
    Set = SET,

}

impl Default for LogicOp {
    #[inline]
    fn default() -> LogicOp {
        LogicOp::Noop
    }
}

/// Describes how a framebuffer color attachment is handled in the pipeline during the color
/// blend stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorBlendAttachmentState {
    /// The blend parameters for the attachment.
    ///
    /// If set to `None`, blending is disabled, and all incoming pixels will be used directly.
    pub blend: Option<AttachmentBlend>,

    /// Sets which components of the final pixel value are written to the attachment.
    pub color_write_mask: ColorComponents,

    /// Sets whether anything at all is written to the attachment. If enabled, the pixel data
    /// that is written is determined by the `color_write_mask`. If disabled, the mask is ignored
    /// and nothing is written.
    ///
    /// If set to anything other than `Fixed(true)`, the
    /// [`color_write_enable`](crate::device::Features::color_write_enable) feature must be enabled
    /// on the device.
    pub color_write_enable: StateMode<bool>,
}

impl ColorBlendAttachmentState {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            ref blend,
            color_write_mask: _,
            color_write_enable,
        } = self;

        if let Some(blend) = blend {
            blend
                .validate(device)
                .map_err(|err| err.add_context("blend"))?;
        }

        match color_write_enable {
            StateMode::Fixed(enable) => {
                if !enable && !device.enabled_features().color_write_enable {
                    return Err(ValidationError {
                        context: "color_write_enable".into(),
                        problem: "is `false`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "color_write_enable",
                        )])]),
                        vuids: &["VUID-VkPipelineColorWriteCreateInfoEXT-pAttachments-04801"],
                    });
                }
            }
            StateMode::Dynamic => {
                if !device.enabled_features().color_write_enable {
                    return Err(ValidationError {
                        context: "color_write_enable".into(),
                        problem: "is dynamic".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "color_write_enable",
                        )])]),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-04800"],
                    });
                }
            }
        }

        Ok(())
    }
}

/// Describes how the blending system should behave for an attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttachmentBlend {
    /// The operation to apply to the source color component before applying `color_op`.
    pub src_color_blend_factor: BlendFactor,

    /// The operation to apply to the destination color component before applying `color_op`.
    pub dst_color_blend_factor: BlendFactor,

    /// The operation to apply between the color components of the source and destination pixels,
    /// to produce the final pixel value.
    pub color_blend_op: BlendOp,

    /// The operation to apply to the source alpha component before applying `alpha_op`.
    pub src_alpha_blend_factor: BlendFactor,

    /// The operation to apply to the destination alpha component before applying `alpha_op`.
    pub dst_alpha_blend_factor: BlendFactor,

    /// The operation to apply between the alpha component of the source and destination pixels,
    /// to produce the final pixel value.
    pub alpha_blend_op: BlendOp,
}

impl AttachmentBlend {
    /// Builds an `AttachmentBlend` where the output of the fragment shader is ignored and the
    /// destination is untouched.
    #[inline]
    pub fn ignore_source() -> Self {
        Self {
            src_color_blend_factor: BlendFactor::Zero,
            dst_color_blend_factor: BlendFactor::DstColor,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::Zero,
            dst_alpha_blend_factor: BlendFactor::DstColor,
            alpha_blend_op: BlendOp::Add,
        }
    }

    /// Builds an `AttachmentBlend` where the output will be merged with the existing value
    /// based on the alpha of the source.
    #[inline]
    pub fn alpha() -> Self {
        Self {
            src_color_blend_factor: BlendFactor::SrcAlpha,
            dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::SrcAlpha,
            dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
            alpha_blend_op: BlendOp::Add,
        }
    }

    /// Builds an `AttachmentBlend` where the colors are added, and alpha is set to the maximum of
    /// the two.
    #[inline]
    pub fn additive() -> Self {
        Self {
            src_color_blend_factor: BlendFactor::One,
            dst_color_blend_factor: BlendFactor::One,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::One,
            alpha_blend_op: BlendOp::Max,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            src_color_blend_factor,
            dst_color_blend_factor,
            color_blend_op,
            src_alpha_blend_factor,
            dst_alpha_blend_factor,
            alpha_blend_op,
        } = self;

        src_color_blend_factor
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "src_color_blend_factor".into(),
                vuids: &["VUID-VkPipelineColorBlendAttachmentState-srcColorBlendFactor-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        dst_color_blend_factor
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "dst_color_blend_factor".into(),
                vuids: &["VUID-VkPipelineColorBlendAttachmentState-dstColorBlendFactor-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        color_blend_op
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "color_blend_op".into(),
                vuids: &["VUID-VkPipelineColorBlendAttachmentState-colorBlendOp-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        src_alpha_blend_factor
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "src_alpha_blend_factor".into(),
                vuids: &["VUID-VkPipelineColorBlendAttachmentState-srcAlphaBlendFactor-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        dst_alpha_blend_factor
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "dst_alpha_blend_factor".into(),
                vuids: &["VUID-VkPipelineColorBlendAttachmentState-dstAlphaBlendFactor-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        alpha_blend_op
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "alpha_blend_op".into(),
                vuids: &["VUID-VkPipelineColorBlendAttachmentState-alphaBlendOp-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if !device.enabled_features().dual_src_blend {
            if matches!(
                src_color_blend_factor,
                BlendFactor::Src1Color
                    | BlendFactor::OneMinusSrc1Color
                    | BlendFactor::Src1Alpha
                    | BlendFactor::OneMinusSrc1Alpha
            ) {
                return Err(ValidationError {
                    context: "src_color_blend_factor".into(),
                    problem: "is `BlendFactor::Src1*`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "dual_src_blend",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-srcColorBlendFactor-00608"],
                });
            }

            if matches!(
                dst_color_blend_factor,
                BlendFactor::Src1Color
                    | BlendFactor::OneMinusSrc1Color
                    | BlendFactor::Src1Alpha
                    | BlendFactor::OneMinusSrc1Alpha
            ) {
                return Err(ValidationError {
                    context: "dst_color_blend_factor".into(),
                    problem: "is `BlendFactor::Src1*`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "dual_src_blend",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-dstColorBlendFactor-00609"],
                });
            }

            if matches!(
                src_alpha_blend_factor,
                BlendFactor::Src1Color
                    | BlendFactor::OneMinusSrc1Color
                    | BlendFactor::Src1Alpha
                    | BlendFactor::OneMinusSrc1Alpha
            ) {
                return Err(ValidationError {
                    context: "src_alpha_blend_factor".into(),
                    problem: "is `BlendFactor::Src1*`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "dual_src_blend",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-srcAlphaBlendFactor-00610"],
                });
            }

            if matches!(
                dst_alpha_blend_factor,
                BlendFactor::Src1Color
                    | BlendFactor::OneMinusSrc1Color
                    | BlendFactor::Src1Alpha
                    | BlendFactor::OneMinusSrc1Alpha
            ) {
                return Err(ValidationError {
                    context: "dst_alpha_blend_factor".into(),
                    problem: "is `BlendFactor::Src1*`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "dual_src_blend",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-dstAlphaBlendFactor-00611"],
                });
            }
        }

        if device.enabled_extensions().khr_portability_subset
            && !device.enabled_features().constant_alpha_color_blend_factors
        {
            if matches!(
                src_color_blend_factor,
                BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha
            ) {
                return Err(ValidationError {
                    problem: "this device is a portability subset device, and \
                        `src_color_blend_factor` is `BlendFactor::ConstantAlpha` or \
                        `BlendFactor::OneMinusConstantAlpha`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "constant_alpha_color_blend_factors",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-constantAlphaColorBlendFactors-04454"],
                    ..Default::default()
                });
            }

            if matches!(
                dst_color_blend_factor,
                BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha
            ) {
                return Err(ValidationError {
                    problem: "this device is a portability subset device, and \
                        `dst_color_blend_factor` is `BlendFactor::ConstantAlpha` or \
                        `BlendFactor::OneMinusConstantAlpha`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "constant_alpha_color_blend_factors",
                    )])]),
                    vuids: &["VUID-VkPipelineColorBlendAttachmentState-constantAlphaColorBlendFactors-04455"],
                    ..Default::default()
                });
            }
        }

        Ok(())
    }
}

impl From<AttachmentBlend> for ash::vk::PipelineColorBlendAttachmentState {
    #[inline]
    fn from(val: AttachmentBlend) -> Self {
        ash::vk::PipelineColorBlendAttachmentState {
            blend_enable: ash::vk::TRUE,
            src_color_blend_factor: val.src_color_blend_factor.into(),
            dst_color_blend_factor: val.dst_color_blend_factor.into(),
            color_blend_op: val.color_blend_op.into(),
            src_alpha_blend_factor: val.src_alpha_blend_factor.into(),
            dst_alpha_blend_factor: val.dst_alpha_blend_factor.into(),
            alpha_blend_op: val.alpha_blend_op.into(),
            color_write_mask: ash::vk::ColorComponentFlags::empty(),
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The operation that takes `source` (output from the fragment shader), `destination` (value
    /// currently in the framebuffer attachment) and `blend_constant` input values,
    /// and produces new inputs to be fed to `BlendOp`.
    ///
    /// Some operations take `source1` as an input, representing the second source value. The
    /// [`dual_src_blend`](crate::device::Features::dual_src_blend) feature must be enabled on the
    /// device when these are used.
    BlendFactor = BlendFactor(i32);

    /// Always `0`.
    Zero = ZERO,

    /// Always `1`.
    One = ONE,

    /// `source` component-wise.
    SrcColor = SRC_COLOR,

    /// `1 - source` component-wise.
    OneMinusSrcColor = ONE_MINUS_SRC_COLOR,

    /// `destination` component-wise.
    DstColor = DST_COLOR,

    /// `1 - destination` component-wise.
    OneMinusDstColor = ONE_MINUS_DST_COLOR,

    /// `source.a` for all components.
    SrcAlpha = SRC_ALPHA,

    /// `1 - source.a` for all components.
    OneMinusSrcAlpha = ONE_MINUS_SRC_ALPHA,

    /// `destination.a` for all components.
    DstAlpha = DST_ALPHA,

    /// `1 - destination.a` for all components.
    OneMinusDstAlpha = ONE_MINUS_DST_ALPHA,

    /// `blend_constants` component-wise.
    ConstantColor = CONSTANT_COLOR,

    /// `1 - blend_constants` component-wise.
    OneMinusConstantColor = ONE_MINUS_CONSTANT_COLOR,

    /// `blend_constants.a` for all components.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if this value is used for the `color_source` or `color_destination` blend factors,
    /// then the
    /// [`constant_alpha_color_blend_factors`](crate::device::Features::constant_alpha_color_blend_factors)
    /// feature must be enabled on the device.
    ConstantAlpha = CONSTANT_ALPHA,

    /// `1 - blend_constants.a` for all components.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if this value is used for the `color_source` or `color_destination` blend factors,
    /// then the
    /// [`constant_alpha_color_blend_factors`](crate::device::Features::constant_alpha_color_blend_factors)
    /// feature must be enabled on the device.
    OneMinusConstantAlpha = ONE_MINUS_CONSTANT_ALPHA,

    /// For the alpha component, always `1`. For the color components,
    /// `min(source.a, 1 - destination.a)` for all components.
    SrcAlphaSaturate = SRC_ALPHA_SATURATE,

    /// `source1` component-wise.
    Src1Color = SRC1_COLOR,

    /// `1 - source1` component-wise.
    OneMinusSrc1Color = ONE_MINUS_SRC1_COLOR,

    /// `source1.a` for all components.
    Src1Alpha = SRC1_ALPHA,

    /// `1 - source1.a` for all components.
    OneMinusSrc1Alpha = ONE_MINUS_SRC1_ALPHA,
}

vulkan_enum! {
    #[non_exhaustive]

    /// The arithmetic operation that is applied between the `source` and `destination` component
    /// values, after the appropriate `BlendFactor` is applied to both.
    BlendOp = BlendOp(i32);

    /// `source + destination`.
    Add = ADD,

    /// `source - destination`.
    Subtract = SUBTRACT,

    /// `destination - source`.
    ReverseSubtract = REVERSE_SUBTRACT,

    /// `min(source, destination)`.
    Min = MIN,

    /// `max(source, destination)`.
    Max = MAX,

    /* TODO: enable
    // TODO: document
    Zero = ZERO_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Src = SRC_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Dst = DST_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SrcOver = SRC_OVER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DstOver = DST_OVER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SrcIn = SRC_IN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DstIn = DST_IN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SrcOut = SRC_OUT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DstOut = DST_OUT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SrcAtop = SRC_ATOP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DstAtop = DST_ATOP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Xor = XOR_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Multiply = MULTIPLY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Screen = SCREEN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Overlay = OVERLAY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Darken = DARKEN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Lighten = LIGHTEN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Colordodge = COLORDODGE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Colorburn = COLORBURN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Hardlight = HARDLIGHT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Softlight = SOFTLIGHT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Difference = DIFFERENCE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Exclusion = EXCLUSION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Invert = INVERT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    InvertRgb = INVERT_RGB_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Lineardodge = LINEARDODGE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Linearburn = LINEARBURN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Vividlight = VIVIDLIGHT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Linearlight = LINEARLIGHT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Pinlight = PINLIGHT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Hardmix = HARDMIX_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    HslHue = HSL_HUE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    HslSaturation = HSL_SATURATION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    HslColor = HSL_COLOR_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    HslLuminosity = HSL_LUMINOSITY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Plus = PLUS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    PlusClamped = PLUS_CLAMPED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    PlusClampedAlpha = PLUS_CLAMPED_ALPHA_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    PlusDarker = PLUS_DARKER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Minus = MINUS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    MinusClamped = MINUS_CLAMPED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Contrast = CONTRAST_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    InvertOvg = INVERT_OVG_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Red = RED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Green = GREEN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Blue = BLUE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),*/
}

vulkan_bitflags! {
    /// A mask specifying color components that can be written to a framebuffer attachment.
    ColorComponents = ColorComponentFlags(u32);

    /// The red component.
    R = R,

    /// The green component.
    G = G,

    /// The blue component.
    B = B,

    /// The alpha component.
    A = A,
}
