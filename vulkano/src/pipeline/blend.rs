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
//! will take precedence if it is activated. Otherwise the blending operation is applied.
//!

use vk;

pub struct Blend {
    pub logic_op: Option<LogicOp>,

    

    /// The constant color to use for the `Constant*` blending operation.
    ///
    /// If you pass `None`, then this state will be considered as dynamic and the blend constants
    /// will need to be set when you build the command buffer.
    pub blend_constants: Option<[f32; 4]>,
}

/*
VkStructureType                             sType;
    const void*                                 pNext;
    VkPipelineColorBlendStateCreateFlags        flags;
    VkBool32                                    logicOpEnable;
    VkLogicOp                                   logicOp;
    uint32_t                                    attachmentCount;
    const VkPipelineColorBlendAttachmentState*  pAttachments;
    float                                       blendConstants[4];
} VkPipelineColorBlendStateCreateInfo;

typedef struct {
    VkBool32                                    blendEnable;
    VkBlend                                     srcBlendColor;
    VkBlend                                     dstBlendColor;
    VkBlendOp                                   blendOpColor;
    VkBlend                                     srcBlendAlpha;
    VkBlend                                     dstBlendAlpha;
    VkBlendOp                                   blendOpAlpha;
    VkChannelFlags                              channelWriteMask;
} VkPipelineColorBlendAttachmentState;
*/

/// Which logical operation to apply to the output values.
///
/// The operation is applied individually for each channel (red, green, blue and alpha).
///
/// Only relevant for integer or unsigned attachments.
///
/// Also note that some implementations don't support logic operations.
#[derive(Debug, Copy, Clone)]
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

///
#[derive(Debug, Copy, Clone)]
#[repr(u32)]
pub enum BlendOp {
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
