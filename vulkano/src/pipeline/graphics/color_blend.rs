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

use super::{DynamicState, GraphicsPipelineCreationError};
use crate::{device::Device, pipeline::StateMode, render_pass::Subpass};
use fnv::FnvHashMap;
use smallvec::SmallVec;

/// Describes how the color output of the fragment shader is written to the attachment. See the
/// documentation of the `blend` module for more info.
#[derive(Clone, Debug)]
pub struct ColorBlendState {
    /// Sets the logical operation to perform between the incoming fragment color and the existing
    /// fragment in the framebuffer attachment.
    ///
    /// If set to `Some`, the [`logic_op`](crate::device::Features::logic_op) feature must be
    /// enabled on the device. If set to `Some(Dynamic)`, then the
    /// [`extended_dynamic_state2`](crate::device::Features::extended_dynamic_state2_logic_op)
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

    pub(crate) fn to_vulkan_attachments(
        &mut self, // TODO: make non-mut
        device: &Device,
        dynamic_state_modes: &mut FnvHashMap<DynamicState, bool>,
        subpass: &Subpass,
    ) -> Result<
        (
            SmallVec<[ash::vk::PipelineColorBlendAttachmentState; 4]>,
            SmallVec<[ash::vk::Bool32; 4]>,
        ),
        GraphicsPipelineCreationError,
    > {
        let num_atch = subpass.num_color_attachments();

        // If there is one element, duplicate it for all attachments.
        // TODO: this is undocumented and only exists for compatibility with some of the
        // deprecated builder methods. Remove it when those methods are gone.
        if self.attachments.len() == 1 {
            let element = self.attachments.pop().unwrap();
            self.attachments
                .extend(std::iter::repeat(element).take(num_atch as usize));
        } else {
            if self.attachments.len() != num_atch as usize {
                return Err(GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount);
            }

            if self.attachments.len() > 1 && !device.enabled_features().independent_blend {
                // Ensure that all `blend` and `color_write_mask` are identical.
                let mut iter = self
                    .attachments
                    .iter()
                    .map(|state| (&state.blend, &state.color_write_mask));
                let first = iter.next().unwrap();

                if !iter.all(|state| state == first) {
                    return Err(
                        GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "independent_blend",
                            reason: "The blend and color_write_mask members of all elements of ColorBlendState::attachments were not identical",
                        },
                    );
                }
            }
        }

        let mut color_blend_attachments = SmallVec::with_capacity(self.attachments.len());
        let mut color_write_enables = SmallVec::with_capacity(self.attachments.len());

        for state in self.attachments.iter() {
            let blend = if let Some(blend) = &state.blend {
                if !device.enabled_features().dual_src_blend
                    && std::array::IntoIter::new([
                        blend.color_source,
                        blend.color_destination,
                        blend.alpha_source,
                        blend.alpha_destination,
                    ])
                    .any(|blend_factor| {
                        matches!(
                            blend_factor,
                            BlendFactor::Src1Color
                                | BlendFactor::OneMinusSrc1Color
                                | BlendFactor::Src1Alpha
                                | BlendFactor::OneMinusSrc1Alpha
                        )
                    })
                {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "dual_src_blend",
                        reason: "One of the BlendFactor members of AttachmentBlend was set to Src1",
                    });
                }

                blend.clone().into()
            } else {
                Default::default()
            };

            let color_blend_attachment_state = ash::vk::PipelineColorBlendAttachmentState {
                color_write_mask: state.color_write_mask.into(),
                ..blend
            };

            let color_write_enable = match state.color_write_enable {
                StateMode::Fixed(enable) => {
                    if !enable && !device.enabled_features().color_write_enable {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "color_write_enable",
                            reason:
                                "ColorBlendAttachmentState::color_blend_enable was set to Fixed(false)",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::ColorWriteEnable, false);
                    enable as ash::vk::Bool32
                }
                StateMode::Dynamic => {
                    if !device.enabled_features().color_write_enable {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "color_write_enable",
                            reason:
                                "ColorBlendAttachmentState::color_blend_enable was set to Dynamic",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::ColorWriteEnable, true);
                    ash::vk::TRUE
                }
            };

            color_blend_attachments.push(color_blend_attachment_state);
            color_write_enables.push(color_write_enable);
        }

        debug_assert_eq!(color_blend_attachments.len(), color_write_enables.len());
        Ok((color_blend_attachments, color_write_enables))
    }

    pub(crate) fn to_vulkan_color_write(
        &self,
        device: &Device,
        dynamic_state_modes: &mut FnvHashMap<DynamicState, bool>,
        color_write_enables: &[ash::vk::Bool32],
    ) -> Result<Option<ash::vk::PipelineColorWriteCreateInfoEXT>, GraphicsPipelineCreationError>
    {
        Ok(if device.enabled_extensions().ext_color_write_enable {
            Some(ash::vk::PipelineColorWriteCreateInfoEXT {
                attachment_count: color_write_enables.len() as u32,
                p_color_write_enables: color_write_enables.as_ptr(),
                ..Default::default()
            })
        } else {
            None
        })
    }

    pub(crate) fn to_vulkan(
        &self,
        device: &Device,
        dynamic_state_modes: &mut FnvHashMap<DynamicState, bool>,
        color_blend_attachments: &[ash::vk::PipelineColorBlendAttachmentState],
        color_write: Option<&mut ash::vk::PipelineColorWriteCreateInfoEXT>,
    ) -> Result<ash::vk::PipelineColorBlendStateCreateInfo, GraphicsPipelineCreationError> {
        let (logic_op_enable, logic_op) = if let Some(logic_op) = self.logic_op {
            if !device.enabled_features().logic_op {
                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                    feature: "logic_op",
                    reason: "ColorBlendState::logic_op was set to Some",
                });
            }

            let logic_op = match logic_op {
                StateMode::Fixed(logic_op) => {
                    dynamic_state_modes.insert(DynamicState::LogicOp, false);
                    logic_op.into()
                }
                StateMode::Dynamic => {
                    if !device.enabled_features().extended_dynamic_state2_logic_op {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state2_logic_op",
                            reason: "ColorBlendState::logic_op was set to Some(Dynamic)",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::LogicOp, true);
                    Default::default()
                }
            };

            (ash::vk::TRUE, logic_op)
        } else {
            (ash::vk::FALSE, Default::default())
        };

        let blend_constants = match self.blend_constants {
            StateMode::Fixed(blend_constants) => {
                dynamic_state_modes.insert(DynamicState::BlendConstants, false);
                blend_constants
            }
            StateMode::Dynamic => {
                dynamic_state_modes.insert(DynamicState::BlendConstants, true);
                Default::default()
            }
        };

        let mut color_blend_state = ash::vk::PipelineColorBlendStateCreateInfo {
            flags: ash::vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable,
            logic_op,
            attachment_count: color_blend_attachments.len() as u32,
            p_attachments: color_blend_attachments.as_ptr(),
            blend_constants,
            ..Default::default()
        };

        if let Some(color_write) = color_write {
            color_write.p_next = color_blend_state.p_next;
            color_blend_state.p_next = color_write as *const _ as *const _;
        }

        Ok(color_blend_state)
    }
}

impl Default for ColorBlendState {
    /// Returns [`ColorBlendState::new(1)`].
    #[inline]
    fn default() -> Self {
        Self::new(1)
    }
}

/// Which logical operation to apply to the output values.
///
/// The operation is applied individually for each channel (red, green, blue and alpha).
///
/// Only relevant for integer or unsigned attachments.
///
/// Also note that some implementations don't support logic operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Describes how a framebuffer color attachment is handled in the pipeline during the color
/// blend stage.
#[derive(Clone, Debug)]
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

/// Describes how the blending system should behave for an attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttachmentBlend {
    /// The operation to apply between the color components of the source and destination pixels,
    /// to produce the final pixel value.
    pub color_op: BlendOp,

    /// The operation to apply to the source color component before applying `color_op`.
    pub color_source: BlendFactor,

    /// The operation to apply to the destination color component before applying `color_op`.
    pub color_destination: BlendFactor,

    /// The operation to apply between the alpha component of the source and destination pixels,
    /// to produce the final pixel value.
    pub alpha_op: BlendOp,

    /// The operation to apply to the source alpha component before applying `alpha_op`.
    pub alpha_source: BlendFactor,

    /// The operation to apply to the destination alpha component before applying `alpha_op`.
    pub alpha_destination: BlendFactor,
}

impl AttachmentBlend {
    /// Builds an `AttachmentBlend` where the output of the fragment shader is ignored and the
    /// destination is untouched.
    #[inline]
    pub fn ignore_source() -> Self {
        Self {
            color_op: BlendOp::Add,
            color_source: BlendFactor::Zero,
            color_destination: BlendFactor::DstColor,
            alpha_op: BlendOp::Add,
            alpha_source: BlendFactor::Zero,
            alpha_destination: BlendFactor::DstColor,
        }
    }

    /// Builds an `AttachmentBlend` where the output will be merged with the existing value
    /// based on the alpha of the source.
    #[inline]
    pub fn alpha() -> Self {
        Self {
            color_op: BlendOp::Add,
            color_source: BlendFactor::SrcAlpha,
            color_destination: BlendFactor::OneMinusSrcAlpha,
            alpha_op: BlendOp::Add,
            alpha_source: BlendFactor::SrcAlpha,
            alpha_destination: BlendFactor::OneMinusSrcAlpha,
        }
    }

    /// Builds an `AttachmentBlend` where the colors are added, and alpha is set to the maximum of
    /// the two.
    #[inline]
    pub fn additive() -> Self {
        Self {
            color_op: BlendOp::Add,
            color_source: BlendFactor::One,
            color_destination: BlendFactor::One,
            alpha_op: BlendOp::Max,
            alpha_source: BlendFactor::One,
            alpha_destination: BlendFactor::One,
        }
    }
}

impl From<AttachmentBlend> for ash::vk::PipelineColorBlendAttachmentState {
    #[inline]
    fn from(val: AttachmentBlend) -> Self {
        ash::vk::PipelineColorBlendAttachmentState {
            blend_enable: ash::vk::TRUE,
            src_color_blend_factor: val.color_source.into(),
            dst_color_blend_factor: val.color_destination.into(),
            color_blend_op: val.color_op.into(),
            src_alpha_blend_factor: val.alpha_source.into(),
            dst_alpha_blend_factor: val.alpha_destination.into(),
            alpha_blend_op: val.alpha_op.into(),
            color_write_mask: ash::vk::ColorComponentFlags::empty(), // Overwritten by GraphicsPipelineBuilder
        }
    }
}

/// The operation that takes `source` (output from the fragment shader), `destination` (value
/// currently in the framebuffer attachment) and `blend_constant` input values,
/// and produces new inputs to be fed to `BlendOp`.
///
/// Some operations take `source1` as an input, representing the second source value. The
/// [`dual_src_blend`](crate::device::Features::dual_src_blend) feature must be enabled on the
/// device when these are used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum BlendFactor {
    /// Always `0`.
    Zero = ash::vk::BlendFactor::ZERO.as_raw(),

    /// Always `1`.
    One = ash::vk::BlendFactor::ONE.as_raw(),

    /// `source` component-wise.
    SrcColor = ash::vk::BlendFactor::SRC_COLOR.as_raw(),

    /// `1 - source` component-wise.
    OneMinusSrcColor = ash::vk::BlendFactor::ONE_MINUS_SRC_COLOR.as_raw(),

    /// `destination` component-wise.
    DstColor = ash::vk::BlendFactor::DST_COLOR.as_raw(),

    /// `1 - destination` component-wise.
    OneMinusDstColor = ash::vk::BlendFactor::ONE_MINUS_DST_COLOR.as_raw(),

    /// `source.a` for all components.
    SrcAlpha = ash::vk::BlendFactor::SRC_ALPHA.as_raw(),

    /// `1 - source.a` for all components.
    OneMinusSrcAlpha = ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA.as_raw(),

    /// `destination.a` for all components.
    DstAlpha = ash::vk::BlendFactor::DST_ALPHA.as_raw(),

    /// `1 - destination.a` for all components.
    OneMinusDstAlpha = ash::vk::BlendFactor::ONE_MINUS_DST_ALPHA.as_raw(),

    /// `blend_constants` component-wise.
    ConstantColor = ash::vk::BlendFactor::CONSTANT_COLOR.as_raw(),

    /// `1 - blend_constants` component-wise.
    OneMinusConstantColor = ash::vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR.as_raw(),

    /// `blend_constants.a` for all components.
    ConstantAlpha = ash::vk::BlendFactor::CONSTANT_ALPHA.as_raw(),

    /// `1 - blend_constants.a` for all components.
    OneMinusConstantAlpha = ash::vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA.as_raw(),

    /// For the alpha component, always `1`. For the color components,
    /// `min(source.a, 1 - destination.a)` for all components.
    SrcAlphaSaturate = ash::vk::BlendFactor::SRC_ALPHA_SATURATE.as_raw(),

    /// `source1` component-wise.
    Src1Color = ash::vk::BlendFactor::SRC1_COLOR.as_raw(),

    /// `1 - source1` component-wise.
    OneMinusSrc1Color = ash::vk::BlendFactor::ONE_MINUS_SRC1_COLOR.as_raw(),

    /// `source1.a` for all components.
    Src1Alpha = ash::vk::BlendFactor::SRC1_ALPHA.as_raw(),

    /// `1 - source1.a` for all components.
    OneMinusSrc1Alpha = ash::vk::BlendFactor::ONE_MINUS_SRC1_ALPHA.as_raw(),
}

impl From<BlendFactor> for ash::vk::BlendFactor {
    #[inline]
    fn from(val: BlendFactor) -> Self {
        Self::from_raw(val as i32)
    }
}

/// The arithmetic operation that is applied between the `source` and `destination` component
/// values, after the appropriate `BlendFactor` is applied to both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum BlendOp {
    /// `source + destination`.
    Add = ash::vk::BlendOp::ADD.as_raw(),

    /// `source - destination`.
    Subtract = ash::vk::BlendOp::SUBTRACT.as_raw(),

    /// `destination - source`.
    ReverseSubtract = ash::vk::BlendOp::REVERSE_SUBTRACT.as_raw(),

    /// `min(source, destination)`.
    Min = ash::vk::BlendOp::MIN.as_raw(),

    /// `max(source, destination)`.
    Max = ash::vk::BlendOp::MAX.as_raw(),
}

impl From<BlendOp> for ash::vk::BlendOp {
    #[inline]
    fn from(val: BlendOp) -> Self {
        Self::from_raw(val as i32)
    }
}

/// A mask specifying color components that can be written to a framebuffer attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorComponents {
    #[allow(missing_docs)]
    pub r: bool,
    #[allow(missing_docs)]
    pub g: bool,
    #[allow(missing_docs)]
    pub b: bool,
    #[allow(missing_docs)]
    pub a: bool,
}

impl ColorComponents {
    /// Returns a mask that specifies no components.
    #[inline]
    pub fn none() -> Self {
        Self {
            r: false,
            g: false,
            b: false,
            a: false,
        }
    }

    /// Returns a mask that specifies all components.
    #[inline]
    pub fn all() -> Self {
        Self {
            r: true,
            g: true,
            b: true,
            a: true,
        }
    }
}

impl From<ColorComponents> for ash::vk::ColorComponentFlags {
    fn from(val: ColorComponents) -> Self {
        let mut result = Self::empty();
        if val.r {
            result |= ash::vk::ColorComponentFlags::R;
        }
        if val.g {
            result |= ash::vk::ColorComponentFlags::G;
        }
        if val.b {
            result |= ash::vk::ColorComponentFlags::B;
        }
        if val.a {
            result |= ash::vk::ColorComponentFlags::A;
        }
        result
    }
}
