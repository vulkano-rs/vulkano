// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Depth and stencil operations description.
//!
//! After the fragment shader has finished running, each fragment goes through the depth
//! and stencil tests.
//!
//! The depth test passes of fails depending on how the depth value of each fragment compares
//! to the existing depth value in the depth buffer at that fragment's location. Depth values
//! are always between 0.0 and 1.0.
//!
//! The stencil test passes or fails depending on how a reference value compares to the existing
//! value in the stencil buffer at each fragment's location. Depending on the outcome of the
//! depth and stencil tests, the value of the stencil buffer at that location can be updated.

use std::ops::Range;
use std::u32;
use vk;

/// Configuration of the depth and stencil tests.
#[derive(Debug, Clone)]
pub struct DepthStencil {
    /// Comparison to use between the depth value of each fragment and the depth value currently
    /// in the depth buffer.
    pub depth_compare: Compare,

    /// If `true`, then the value in the depth buffer will be updated when the depth test succeeds.
    pub depth_write: bool,

    /// Allows you to ask the GPU to exclude fragments that are outside of a certain range. This is
    /// done in addition to the regular depth test.
    pub depth_bounds_test: DepthBounds,

    /// Stencil operations to use for points, lines and triangles whose front is facing the user.
    pub stencil_front: Stencil,

    /// Stencil operations to use for triangles whose back is facing the user.
    pub stencil_back: Stencil,
}

impl DepthStencil {
    /// Creates a `DepthStencil` where both the depth and stencil tests are disabled and have
    /// no effect.
    #[inline]
    pub fn disabled() -> DepthStencil {
        DepthStencil {
            depth_write: false,
            depth_compare: Compare::Always,
            depth_bounds_test: DepthBounds::Disabled,
            stencil_front: Default::default(),
            stencil_back: Default::default(),
        }
    }

    /// Creates a `DepthStencil` with a `Less` depth test, `depth_write` set to true, and stencil
    /// testing disabled.
    #[inline]
    pub fn simple_depth_test() -> DepthStencil {
        DepthStencil {
            depth_write: true,
            depth_compare: Compare::Less,
            depth_bounds_test: DepthBounds::Disabled,
            stencil_front: Default::default(),
            stencil_back: Default::default(),
        }
    }
}

impl Default for DepthStencil {
    #[inline]
    fn default() -> DepthStencil {
        DepthStencil::disabled()
    }
}

/// Configuration of a stencil test.
#[derive(Debug, Copy, Clone)]
pub struct Stencil {
    /// The comparison to perform between the existing stencil value in the stencil buffer, and
    /// the reference value (given by `reference`).
    pub compare: Compare,

    /// The operation to perform when both the depth test and the stencil test passed.
    pub pass_op: StencilOp,

    /// The operation to perform when the stencil test failed.
    pub fail_op: StencilOp,

    /// The operation to perform when the stencil test passed but the depth test failed.
    pub depth_fail_op: StencilOp,

    /// Selects the bits of the unsigned integer stencil values participating in the stencil test.
    ///
    /// Ignored if `compare` is `Never` or `Always`.
    ///
    /// If `None`, then this value is dynamic and will need to be set when drawing. Doesn't apply
    /// if `compare` is `Never` or `Always`.
    ///
    /// Note that if this value is `Some` in `stencil_front`, it must also be `Some` in
    /// `stencil_back` (but the content can be different). If this value is `None` in
    /// `stencil_front`, then it must also be `None` in `stencil_back`. This rule doesn't apply
    /// if `compare` is `Never` or `Always`.
    pub compare_mask: Option<u32>,

    /// Selects the bits of the unsigned integer stencil values updated by the stencil test in the
    /// stencil framebuffer attachment.
    ///
    /// If `None`, then this value is dynamic and will need to be set when drawing.
    ///
    /// Note that if this value is `Some` in `stencil_front`, it must also be `Some` in
    /// `stencil_back` (but the content can be different). If this value is `None` in
    /// `stencil_front`, then it must also be `None` in `stencil_back`.
    pub write_mask: Option<u32>,

    /// Reference value that is used in the unsigned stencil comparison.
    ///
    /// If `None`, then this value is dynamic and will need to be set when drawing.
    ///
    /// Note that if this value is `Some` in `stencil_front`, it must also be `Some` in
    /// `stencil_back` (but the content can be different). If this value is `None` in
    /// `stencil_front`, then it must also be `None` in `stencil_back`.
    pub reference: Option<u32>,
}

impl Stencil {
    /// Returns true if the stencil operation will always result in `Keep`.
    #[inline]
    pub fn always_keep(&self) -> bool {
        match self.compare {
            Compare::Always => self.pass_op == StencilOp::Keep &&
                self.depth_fail_op == StencilOp::Keep,
            Compare::Never => self.fail_op == StencilOp::Keep,
            _ => self.pass_op == StencilOp::Keep && self.fail_op == StencilOp::Keep &&
                self.depth_fail_op == StencilOp::Keep,
        }
    }
}

impl Default for Stencil {
    #[inline]
    fn default() -> Stencil {
        Stencil {
            compare: Compare::Never,
            pass_op: StencilOp::Keep,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            compare_mask: Some(u32::MAX),
            write_mask: Some(u32::MAX),
            reference: Some(u32::MAX),
        }
    }
}

/// Operation to perform after the depth and stencil tests.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum StencilOp {
    Keep = vk::STENCIL_OP_KEEP,
    Zero = vk::STENCIL_OP_ZERO,
    Replace = vk::STENCIL_OP_REPLACE,
    IncrementAndClamp = vk::STENCIL_OP_INCREMENT_AND_CLAMP,
    DecrementAndClamp = vk::STENCIL_OP_DECREMENT_AND_CLAMP,
    Invert = vk::STENCIL_OP_INVERT,
    IncrementAndWrap = vk::STENCIL_OP_INCREMENT_AND_WRAP,
    DecrementAndWrap = vk::STENCIL_OP_DECREMENT_AND_WRAP,
}

/// Enum to specify which stencil state to use
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum StencilFaceFlags {
    StencilFaceFrontBit = vk::STENCIL_FACE_FRONT_BIT,
    StencilFaceBackBit = vk::STENCIL_FACE_BACK_BIT,
    StencilFrontAndBack = vk::STENCIL_FRONT_AND_BACK,
}

/// Container for dynamic StencilFaceFlags and value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DynamicStencilValue {
    pub face: StencilFaceFlags,
    pub value: u32,
}

/// Allows you to ask the GPU to exclude fragments that are outside of a certain range.
#[derive(Debug, Clone, PartialEq)]
pub enum DepthBounds {
    /// The test is disabled. All fragments pass the depth bounds test.
    Disabled,

    /// Fragments that are within the given range do pass the test. Values are depth values
    /// between 0.0 and 1.0.
    Fixed(Range<f32>),

    /// The depth bounds test is enabled, but the range will need to specified when you submit
    /// a draw command.
    Dynamic,
}

impl DepthBounds {
    /// Returns true if equal to `DepthBounds::Dynamic`.
    #[inline]
    pub fn is_dynamic(&self) -> bool {
        match self {
            &DepthBounds::Dynamic => true,
            _ => false,
        }
    }
}

/// Specifies how two values should be compared to decide whether a test passes or fails.
///
/// Used for both depth testing and stencil testing.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum Compare {
    /// The test never passes.
    Never = vk::COMPARE_OP_NEVER,
    /// The test passes if `value < reference_value`.
    Less = vk::COMPARE_OP_LESS,
    /// The test passes if `value == reference_value`.
    Equal = vk::COMPARE_OP_EQUAL,
    /// The test passes if `value <= reference_value`.
    LessOrEqual = vk::COMPARE_OP_LESS_OR_EQUAL,
    /// The test passes if `value > reference_value`.
    Greater = vk::COMPARE_OP_GREATER,
    /// The test passes if `value != reference_value`.
    NotEqual = vk::COMPARE_OP_NOT_EQUAL,
    /// The test passes if `value >= reference_value`.
    GreaterOrEqual = vk::COMPARE_OP_GREATER_OR_EQUAL,
    /// The test always passes.
    Always = vk::COMPARE_OP_ALWAYS,
}
