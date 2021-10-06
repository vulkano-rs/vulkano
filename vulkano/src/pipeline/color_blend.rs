// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Defines how the color output of the fragment shader is written to the attachment.
//!
//! # Blending in details
//!
//! There are three kinds of color attachments for the purpose of blending:
//!
//! - Attachments with a floating-point, fixed point format.
//! - Attachments with a (non-normalized) integer format.
//! - Attachments with a normalized integer format.
//!
//! For floating-point and fixed-point formats, the blending operation is applied. For integer
//! formats, the logic operation is applied. For normalized integer formats, the logic operation
//! will take precedence if it is activated, otherwise the blending operation is applied.

use crate::pipeline::StateMode;

/// Describes how the color output of the fragment shader is written to the attachment. See the
/// documentation of the `blend` module for more info.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorBlendState {
    /// Sets the logical operation to perform between the incoming fragment color and the existing
    /// fragment in the framebuffer attachment.
    ///
    /// If set to `Some`, the [`logic_op`](crate::device::Features::logic_op) feature must be
    /// enabled on the device. If set to `Some(Dynamic)`, then the
    /// [`extended_dynamic_state2`](crate::device::Features::extended_dynamic_state2_logic_op)
    /// feature must also be enabled on the device.
    pub logic_op: Option<StateMode<LogicOp>>,

    pub attachments: AttachmentsBlend,

    /// The constant color to use for the `Constant*` blending operation.
    pub blend_constants: StateMode<[f32; 4]>,
}

impl ColorBlendState {
    /// Creates a `ColorBlendState` with attachment passthrough, logical operations disabled and
    /// blend constants set to zero.
    #[inline]
    pub fn new() -> Self {
        Self {
            logic_op: None,
            attachments: AttachmentsBlend::Collective(AttachmentBlend::pass_through()),
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

    /// Sets the attachments to collective alpha blending.
    #[inline]
    pub fn alpha_blending(mut self) -> Self {
        self.attachments = AttachmentsBlend::Collective(AttachmentBlend::alpha_blending());
        self
    }

    /// Sets the attachments to collective blending with the given parameters.
    #[inline]
    pub fn collective(mut self, params: AttachmentBlend) -> Self {
        self.attachments = AttachmentsBlend::Collective(params);
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
}

impl Default for ColorBlendState {
    /// Returns [`ColorBlendState::new()`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Describes how the blending system should behave.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttachmentsBlend {
    /// All the framebuffer attachments will use the same blending.
    Collective(AttachmentBlend),

    /// Each attachment will behave differently. The
    /// [`independent_blend`](crate::device::Features::independent_blend) feature must be enabled
    /// on the device.
    Individual(Vec<AttachmentBlend>),
}

/// Describes how the blending system should behave for an individual attachment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachmentBlend {
    // TODO: could be automatically determined from the other params
    /// If false, blending is ignored and the output is directly written to the attachment.
    pub enabled: bool,

    pub color_op: BlendOp,
    pub color_source: BlendFactor,
    pub color_destination: BlendFactor,

    pub alpha_op: BlendOp,
    pub alpha_source: BlendFactor,
    pub alpha_destination: BlendFactor,

    pub mask_red: bool,
    pub mask_green: bool,
    pub mask_blue: bool,
    pub mask_alpha: bool,
}

impl AttachmentBlend {
    /// Builds an `AttachmentBlend` where blending is disabled.
    #[inline]
    pub fn pass_through() -> AttachmentBlend {
        AttachmentBlend {
            enabled: false,
            color_op: BlendOp::Add,
            color_source: BlendFactor::Zero,
            color_destination: BlendFactor::One,
            alpha_op: BlendOp::Add,
            alpha_source: BlendFactor::Zero,
            alpha_destination: BlendFactor::One,
            mask_red: true,
            mask_green: true,
            mask_blue: true,
            mask_alpha: true,
        }
    }

    /// Builds an `AttachmentBlend` where the output of the fragment shader is ignored and the
    /// destination is untouched.
    #[inline]
    pub fn ignore_source() -> AttachmentBlend {
        AttachmentBlend {
            enabled: true,
            color_op: BlendOp::Add,
            color_source: BlendFactor::Zero,
            color_destination: BlendFactor::DstColor,
            alpha_op: BlendOp::Add,
            alpha_source: BlendFactor::Zero,
            alpha_destination: BlendFactor::DstColor,
            mask_red: true,
            mask_green: true,
            mask_blue: true,
            mask_alpha: true,
        }
    }

    /// Builds an `AttachmentBlend` where the output will be merged with the existing value
    /// based on the alpha of the source.
    #[inline]
    pub fn alpha_blending() -> AttachmentBlend {
        AttachmentBlend {
            enabled: true,
            color_op: BlendOp::Add,
            color_source: BlendFactor::SrcAlpha,
            color_destination: BlendFactor::OneMinusSrcAlpha,
            alpha_op: BlendOp::Add,
            alpha_source: BlendFactor::SrcAlpha,
            alpha_destination: BlendFactor::OneMinusSrcAlpha,
            mask_red: true,
            mask_green: true,
            mask_blue: true,
            mask_alpha: true,
        }
    }
}

impl From<AttachmentBlend> for ash::vk::PipelineColorBlendAttachmentState {
    #[inline]
    fn from(val: AttachmentBlend) -> Self {
        ash::vk::PipelineColorBlendAttachmentState {
            blend_enable: if val.enabled {
                ash::vk::TRUE
            } else {
                ash::vk::FALSE
            },
            src_color_blend_factor: val.color_source.into(),
            dst_color_blend_factor: val.color_destination.into(),
            color_blend_op: val.color_op.into(),
            src_alpha_blend_factor: val.alpha_source.into(),
            dst_alpha_blend_factor: val.alpha_destination.into(),
            alpha_blend_op: val.alpha_op.into(),
            color_write_mask: {
                let mut mask = ash::vk::ColorComponentFlags::empty();
                if val.mask_red {
                    mask |= ash::vk::ColorComponentFlags::R;
                }
                if val.mask_green {
                    mask |= ash::vk::ColorComponentFlags::G;
                }
                if val.mask_blue {
                    mask |= ash::vk::ColorComponentFlags::B;
                }
                if val.mask_alpha {
                    mask |= ash::vk::ColorComponentFlags::A;
                }
                mask
            },
        }
    }
}

/// Which logical operation to apply to the output values.
///
/// The operation is applied individually for each channel (red, green, blue and alpha).
///
/// Only relevant for integer or unsigned attachments.
///
/// Also note that some implementations don't support logic operations.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum LogicOp {
    /// Returns `0`.
    Clear = ash::vk::LogicOp::CLEAR.as_raw(),
    /// Returns `source & destination`.
    And = ash::vk::LogicOp::AND.as_raw(),
    /// Returns `source & !destination`.
    AndReverse = ash::vk::LogicOp::AND_REVERSE.as_raw(),
    /// Returns `source`.
    Copy = ash::vk::LogicOp::COPY.as_raw(),
    /// Returns `!source & destination`.
    AndInverted = ash::vk::LogicOp::AND_INVERTED.as_raw(),
    /// Returns `destination`.
    Noop = ash::vk::LogicOp::NO_OP.as_raw(),
    /// Returns `source ^ destination`.
    Xor = ash::vk::LogicOp::XOR.as_raw(),
    /// Returns `source | destination`.
    Or = ash::vk::LogicOp::OR.as_raw(),
    /// Returns `!(source | destination)`.
    Nor = ash::vk::LogicOp::NOR.as_raw(),
    /// Returns `!(source ^ destination)`.
    Equivalent = ash::vk::LogicOp::EQUIVALENT.as_raw(),
    /// Returns `!destination`.
    Invert = ash::vk::LogicOp::INVERT.as_raw(),
    /// Returns `source | !destination.
    OrReverse = ash::vk::LogicOp::OR_REVERSE.as_raw(),
    /// Returns `!source`.
    CopyInverted = ash::vk::LogicOp::COPY_INVERTED.as_raw(),
    /// Returns `!source | destination`.
    OrInverted = ash::vk::LogicOp::OR_INVERTED.as_raw(),
    /// Returns `!(source & destination)`.
    Nand = ash::vk::LogicOp::NAND.as_raw(),
    /// Returns `!0` (all bits set to 1).
    Set = ash::vk::LogicOp::SET.as_raw(),
}

impl From<LogicOp> for ash::vk::LogicOp {
    #[inline]
    fn from(val: LogicOp) -> Self {
        Self::from_raw(val as i32)
    }
}

impl Default for LogicOp {
    #[inline]
    fn default() -> LogicOp {
        LogicOp::Noop
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum BlendOp {
    Add = ash::vk::BlendOp::ADD.as_raw(),
    Subtract = ash::vk::BlendOp::SUBTRACT.as_raw(),
    ReverseSubtract = ash::vk::BlendOp::REVERSE_SUBTRACT.as_raw(),
    Min = ash::vk::BlendOp::MIN.as_raw(),
    Max = ash::vk::BlendOp::MAX.as_raw(),
}

impl From<BlendOp> for ash::vk::BlendOp {
    #[inline]
    fn from(val: BlendOp) -> Self {
        Self::from_raw(val as i32)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum BlendFactor {
    Zero = ash::vk::BlendFactor::ZERO.as_raw(),
    One = ash::vk::BlendFactor::ONE.as_raw(),
    SrcColor = ash::vk::BlendFactor::SRC_COLOR.as_raw(),
    OneMinusSrcColor = ash::vk::BlendFactor::ONE_MINUS_SRC_COLOR.as_raw(),
    DstColor = ash::vk::BlendFactor::DST_COLOR.as_raw(),
    OneMinusDstColor = ash::vk::BlendFactor::ONE_MINUS_DST_COLOR.as_raw(),
    SrcAlpha = ash::vk::BlendFactor::SRC_ALPHA.as_raw(),
    OneMinusSrcAlpha = ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA.as_raw(),
    DstAlpha = ash::vk::BlendFactor::DST_ALPHA.as_raw(),
    OneMinusDstAlpha = ash::vk::BlendFactor::ONE_MINUS_DST_ALPHA.as_raw(),
    ConstantColor = ash::vk::BlendFactor::CONSTANT_COLOR.as_raw(),
    OneMinusConstantColor = ash::vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR.as_raw(),
    ConstantAlpha = ash::vk::BlendFactor::CONSTANT_ALPHA.as_raw(),
    OneMinusConstantAlpha = ash::vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA.as_raw(),
    SrcAlphaSaturate = ash::vk::BlendFactor::SRC_ALPHA_SATURATE.as_raw(),
    Src1Color = ash::vk::BlendFactor::SRC1_COLOR.as_raw(),
    OneMinusSrc1Color = ash::vk::BlendFactor::ONE_MINUS_SRC1_COLOR.as_raw(),
    Src1Alpha = ash::vk::BlendFactor::SRC1_ALPHA.as_raw(),
    OneMinusSrc1Alpha = ash::vk::BlendFactor::ONE_MINUS_SRC1_ALPHA.as_raw(),
}

impl From<BlendFactor> for ash::vk::BlendFactor {
    #[inline]
    fn from(val: BlendFactor) -> Self {
        Self::from_raw(val as i32)
    }
}
