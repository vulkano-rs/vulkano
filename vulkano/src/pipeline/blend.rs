// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
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
//!

use vk;

/// Describes how the color output of the fragment shader is written to the attachment. See the
/// documentation of the `blend` module for more info.
#[derive(Debug, Clone, PartialEq)]
pub struct Blend {
    pub logic_op: Option<LogicOp>,

    pub attachments: AttachmentsBlend,

    /// The constant color to use for the `Constant*` blending operation.
    ///
    /// If you pass `None`, then this state will be considered as dynamic and the blend constants
    /// will need to be set when you build the command buffer.
    pub blend_constants: Option<[f32; 4]>,
}

impl Blend {
    /// Returns a `Blend` object that directly writes colors and alpha on the surface.
    #[inline]
    pub fn pass_through() -> Blend {
        Blend {
            logic_op: None,
            attachments: AttachmentsBlend::Collective(AttachmentBlend::pass_through()),
            blend_constants: Some([0.0, 0.0, 0.0, 0.0]),
        }
    }

    /// Returns a `Blend` object that adds transparent objects over others.
    #[inline]
    pub fn alpha_blending() -> Blend {
        Blend {
            logic_op: None,
            attachments: AttachmentsBlend::Collective(AttachmentBlend::alpha_blending()),
            blend_constants: Some([0.0, 0.0, 0.0, 0.0]),
        }
    }
}

/// Describes how the blending system should behave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentsBlend {
    /// All the framebuffer attachments will use the same blending.
    Collective(AttachmentBlend),

    /// Each attachment will behave differently. Note that this requires enabling the
    /// `independent_blend` feature.
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

    #[inline]
    pub(crate) fn into_vulkan_state(self) -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState {
            blendEnable: if self.enabled { vk::TRUE } else { vk::FALSE },
            srcColorBlendFactor: self.color_source as u32,
            dstColorBlendFactor: self.color_destination as u32,
            colorBlendOp: self.color_op as u32,
            srcAlphaBlendFactor: self.alpha_source as u32,
            dstAlphaBlendFactor: self.alpha_destination as u32,
            alphaBlendOp: self.alpha_op as u32,
            colorWriteMask: {
                let mut mask = 0;
                if self.mask_red {
                    mask |= vk::COLOR_COMPONENT_R_BIT;
                }
                if self.mask_green {
                    mask |= vk::COLOR_COMPONENT_G_BIT;
                }
                if self.mask_blue {
                    mask |= vk::COLOR_COMPONENT_B_BIT;
                }
                if self.mask_alpha {
                    mask |= vk::COLOR_COMPONENT_A_BIT;
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
#[repr(u32)]
pub enum LogicOp {
    /// Returns `0`.
    Clear = vk::LOGIC_OP_CLEAR,
    /// Returns `source & destination`.
    And = vk::LOGIC_OP_AND,
    /// Returns `source & !destination`.
    AndReverse = vk::LOGIC_OP_AND_REVERSE,
    /// Returns `source`.
    Copy = vk::LOGIC_OP_COPY,
    /// Returns `!source & destination`.
    AndInverted = vk::LOGIC_OP_AND_INVERTED,
    /// Returns `destination`.
    Noop = vk::LOGIC_OP_NO_OP,
    /// Returns `source ^ destination`.
    Xor = vk::LOGIC_OP_XOR,
    /// Returns `source | destination`.
    Or = vk::LOGIC_OP_OR,
    /// Returns `!(source | destination)`.
    Nor = vk::LOGIC_OP_NOR,
    /// Returns `!(source ^ destination)`.
    Equivalent = vk::LOGIC_OP_EQUIVALENT,
    /// Returns `!destination`.
    Invert = vk::LOGIC_OP_INVERT,
    /// Returns `source | !destination.
    OrReverse = vk::LOGIC_OP_OR_REVERSE,
    /// Returns `!source`.
    CopyInverted = vk::LOGIC_OP_COPY_INVERTED,
    /// Returns `!source | destination`.
    OrInverted = vk::LOGIC_OP_OR_INVERTED,
    /// Returns `!(source & destination)`.
    Nand = vk::LOGIC_OP_NAND,
    /// Returns `!0` (all bits set to 1).
    Set = vk::LOGIC_OP_SET,
}

impl Default for LogicOp {
    #[inline]
    fn default() -> LogicOp {
        LogicOp::Noop
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum BlendOp {
    Add = vk::BLEND_OP_ADD,
    Subtract = vk::BLEND_OP_SUBTRACT,
    ReverseSubtract = vk::BLEND_OP_REVERSE_SUBTRACT,
    Min = vk::BLEND_OP_MIN,
    Max = vk::BLEND_OP_MAX,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum BlendFactor {
    Zero = vk::BLEND_FACTOR_ZERO,
    One = vk::BLEND_FACTOR_ONE,
    SrcColor = vk::BLEND_FACTOR_SRC_COLOR,
    OneMinusSrcColor = vk::BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
    DstColor = vk::BLEND_FACTOR_DST_COLOR,
    OneMinusDstColor = vk::BLEND_FACTOR_ONE_MINUS_DST_COLOR,
    SrcAlpha = vk::BLEND_FACTOR_SRC_ALPHA,
    OneMinusSrcAlpha = vk::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    DstAlpha = vk::BLEND_FACTOR_DST_ALPHA,
    OneMinusDstAlpha = vk::BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
    ConstantColor = vk::BLEND_FACTOR_CONSTANT_COLOR,
    OneMinusConstantColor = vk::BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
    ConstantAlpha = vk::BLEND_FACTOR_CONSTANT_ALPHA,
    OneMinusConstantAlpha = vk::BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
    SrcAlphaSaturate = vk::BLEND_FACTOR_SRC_ALPHA_SATURATE,
    Src1Color = vk::BLEND_FACTOR_SRC1_COLOR,
    OneMinusSrc1Color = vk::BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
    Src1Alpha = vk::BLEND_FACTOR_SRC1_ALPHA,
    OneMinusSrc1Alpha = vk::BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
}
