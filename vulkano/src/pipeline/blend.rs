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
            attachments: AttachmentsBlend::Collective(AttachmentBlend {
                enabled: false,
                color_op: BlendOp::Add,
                color_src: BlendFactor::Zero,
                color_dst: BlendFactor::One,
                alpha_op: BlendOp::Add,
                alpha_src: BlendFactor::Zero,
                alpha_dst: BlendFactor::One,
                mask_red: true,
                mask_green: true,
                mask_blue: true,
                mask_alpha: true,
            }),
            blend_constants: Some([0.0, 0.0, 0.0, 0.0]),
        }
    }

    /// Returns a `Blend` object that adds transparent objects over others.
    #[inline]
    pub fn alpha_blending() -> Blend {
        Blend {
            logic_op: None,
            attachments: AttachmentsBlend::Collective(AttachmentBlend {
                enabled: true,
                color_op: BlendOp::Add,
                color_src: BlendFactor::SrcAlpha,
                color_dst: BlendFactor::OneMinusSrcAlpha,
                alpha_op: BlendOp::Add,
                alpha_src: BlendFactor::SrcAlpha,
                alpha_dst: BlendFactor::OneMinusSrcAlpha,
                mask_red: true,
                mask_green: true,
                mask_blue: true,
                mask_alpha: true,
            }),
            blend_constants: Some([0.0, 0.0, 0.0, 0.0]),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentsBlend {
    Collective(AttachmentBlend),
    Individual(Vec<AttachmentBlend>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachmentBlend {
    // TODO: could be automatically determined from the other params
    pub enabled: bool,

    pub color_op: BlendOp,
    pub color_src: BlendFactor,
    pub color_dst: BlendFactor,

    pub alpha_op: BlendOp,
    pub alpha_src: BlendFactor,
    pub alpha_dst: BlendFactor,

    pub mask_red: bool,
    pub mask_green: bool,
    pub mask_blue: bool,
    pub mask_alpha: bool,
}

#[doc(hidden)]
impl Into<vk::PipelineColorBlendAttachmentState> for AttachmentBlend {
    #[inline]
    fn into(self) -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState {
            blendEnable: if self.enabled { vk::TRUE } else { vk::FALSE },
            srcColorBlendFactor: self.color_src as u32,
            dstColorBlendFactor: self.color_dst as u32,
            colorBlendOp: self.color_op as u32,
            srcAlphaBlendFactor: self.alpha_src as u32,
            dstAlphaBlendFactor: self.alpha_dst as u32,
            alphaBlendOp: self.alpha_op as u32,
            colorWriteMask: {
                let mut mask = 0;
                if self.mask_red { mask |= vk::COLOR_COMPONENT_R_BIT; }
                if self.mask_green { mask |= vk::COLOR_COMPONENT_G_BIT; }
                if self.mask_blue { mask |= vk::COLOR_COMPONENT_B_BIT; }
                if self.mask_alpha { mask |= vk::COLOR_COMPONENT_A_BIT; }
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
    /// Returns `src & dest`.
    And = vk::LOGIC_OP_AND,
    /// Returns `src & !dest`.
    AndReverse = vk::LOGIC_OP_AND_REVERSE,
    /// Returns `src`.
    Copy = vk::LOGIC_OP_COPY,
    /// Returns `!src & dest`.
    AndInverted = vk::LOGIC_OP_AND_INVERTED,
    /// Returns `dest`.
    Noop = vk::LOGIC_OP_NO_OP,
    /// Returns `src ^ dest`.
    Xor = vk::LOGIC_OP_XOR,
    /// Returns `src | dest`.
    Or = vk::LOGIC_OP_OR,
    /// Returns `!(src | dest)`.
    Nor = vk::LOGIC_OP_NOR,
    /// Returns `!(src ^ dest)`.
    Equivalent = vk::LOGIC_OP_EQUIVALENT,
    /// Returns `!dest`.
    Invert = vk::LOGIC_OP_INVERT,
    /// Returns `src | !dest.
    OrReverse = vk::LOGIC_OP_OR_REVERSE,
    /// Returns `!src`.
    CopyInverted = vk::LOGIC_OP_COPY_INVERTED,
    /// Returns `!src | dest`.
    OrInverted = vk::LOGIC_OP_OR_INVERTED,
    /// Returns `!(src & dest)`.
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
