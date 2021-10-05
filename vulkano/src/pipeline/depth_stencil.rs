// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Depth and stencil operations description.
//!
//! After the fragment shader has finished running, each fragment goes through the depth, depth
//! bounds and stencil tests.
//!
//! The depth test passes of fails depending on how the depth value of each fragment compares
//! to the existing depth value in the depth buffer at that fragment's location. Depth values
//! are always between 0.0 and 1.0.
//!
//! The depth bounds test allows you to ask the GPU to exclude fragments that are outside of a
//! certain range. This is done in addition to the regular depth test.
//!
//! The stencil test passes or fails depending on how a reference value compares to the existing
//! value in the stencil buffer at each fragment's location. Depending on the outcome of the
//! depth and stencil tests, the value of the stencil buffer at that location can be updated.

use crate::pipeline::StateMode;
use std::ops::RangeInclusive;
use std::u32;

/// The state in a graphics pipeline describing how the depth, depth bounds and stencil tests
/// should behave.
#[derive(Clone, Debug)]
pub struct DepthStencilState {
    /// The state of the depth test.
    ///
    /// If set to `None`, the depth test is disabled, all fragments will pass and no depth writes
    /// are performed.
    pub depth: Option<DepthState>,

    /// The state of the depth bounds test.
    ///
    /// If set to `None`, the depth bounds test is disabled, all fragments will pass.
    pub depth_bounds: Option<DepthBoundsState>,

    /// The state of the stencil test.
    ///
    /// If set to `None`, the stencil test is disabled, all fragments will pass and no stencil
    /// writes are performed.
    pub stencil: Option<StencilState>,
}

impl DepthStencilState {
    /// Creates a `DepthStencilState` where all tests are disabled and have no effect.
    #[inline]
    pub fn disabled() -> Self {
        Self {
            depth: Default::default(),
            depth_bounds: Default::default(),
            stencil: Default::default(),
        }
    }

    /// Creates a `DepthStencilState` with a `Less` depth test, `depth_write` set to true, and other
    /// tests disabled.
    #[inline]
    pub fn simple_depth_test() -> Self {
        Self {
            depth: Some(DepthState {
                enable_dynamic: false,
                compare_op: StateMode::Fixed(CompareOp::Less),
                write_enable: StateMode::Fixed(true),
            }),
            depth_bounds: Default::default(),
            stencil: Default::default(),
        }
    }
}

impl Default for DepthStencilState {
    /// Creates a `DepthStencilState` with all tests disabled.
    #[inline]
    fn default() -> Self {
        DepthStencilState::disabled()
    }
}

/// The state in a graphics pipeline describing how the depth test should behave when enabled.
#[derive(Clone, Copy, Debug)]
pub struct DepthState {
    /// Sets whether depth testing should be enabled and disabled dynamically. If set to `false`,
    /// depth testing is always enabled.
    ///
    /// If set to `true`, the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub enable_dynamic: bool,

    /// Sets whether the value in the depth buffer will be updated when the depth test succeeds.
    ///
    /// If set to `Dynamic`, the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub write_enable: StateMode<bool>,

    /// Comparison operation to use between the depth value of each incoming fragment and the depth
    /// value currently in the depth buffer.
    ///
    /// If set to `Dynamic`, the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub compare_op: StateMode<CompareOp>,
}

impl Default for DepthState {
    /// Creates a `DepthState` with no dynamic state, depth writes disabled and `compare_op` set
    /// to always pass.
    #[inline]
    fn default() -> Self {
        Self {
            enable_dynamic: false,
            write_enable: StateMode::Fixed(false),
            compare_op: StateMode::Fixed(CompareOp::Always),
        }
    }
}

/// The state in a graphics pipeline describing how the depth bounds test should behave when
/// enabled.
#[derive(Clone, Debug)]
pub struct DepthBoundsState {
    /// Sets whether depth bounds testing should be enabled and disabled dynamically. If set to
    /// `false`, depth bounds testing is always enabled.
    ///
    /// If set to `true`, the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub enable_dynamic: bool,

    /// The minimum and maximum depth values to use for the test. Fragments with values outside this
    /// range are discarded.
    ///
    /// If set to `Dynamic`, the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub bounds: StateMode<RangeInclusive<f32>>,
}

impl Default for DepthBoundsState {
    /// Creates a `DepthBoundsState` with no dynamic state and the bounds set to `0.0..=1.0`.
    #[inline]
    fn default() -> Self {
        Self {
            enable_dynamic: false,
            bounds: StateMode::Fixed(0.0..=1.0),
        }
    }
}

/// The state in a graphics pipeline describing how the stencil test should behave when enabled.
///
/// Dynamic state can only be enabled or disabled for both faces at once. Therefore, the dynamic
/// state values in `StencilOpState`, must match: the values for `front` and `back` must either both
/// be `Fixed` or both be `Dynamic`.
#[derive(Clone, Debug)]
pub struct StencilState {
    /// Sets whether stencil testing should be enabled and disabled dynamically. If set to
    /// `false`, stencil testing is always enabled.
    ///
    /// If set to `true`, the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub enable_dynamic: bool,

    /// The stencil operation state to use for points and lines, and for triangles whose front is
    /// facing the user.
    pub front: StencilOpState,

    /// The stencil operation state to use for triangles whose back is facing the user.
    pub back: StencilOpState,
}

/// Stencil test operations for a single face.
#[derive(Clone, Copy, Debug)]
pub struct StencilOpState {
    /// The stencil operations to perform.
    ///
    /// If set to `Dynamic`, the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub ops: StateMode<StencilOps>,

    /// A bitmask that selects the bits of the unsigned integer stencil values participating in the
    /// stencil test. Ignored if `compare_op` is `Never` or `Always`.
    pub compare_mask: StateMode<u32>,

    /// A bitmask that selects the bits of the unsigned integer stencil values updated by the
    /// stencil test in the stencil framebuffer attachment. Ignored if the relevant operation is
    /// `Keep`.
    pub write_mask: StateMode<u32>,

    /// Reference value that is used in the unsigned stencil comparison. The stencil test is
    /// considered to pass if the `compare_op` between the stencil buffer value and this reference
    /// value yields true.
    pub reference: StateMode<u32>,
}

impl Default for StencilOpState {
    /// Creates a `StencilOpState` with no dynamic state, `compare_op` set to `Never`, the stencil
    /// operations set to `Keep`, and the masks and reference values set to `u32::MAX`.
    #[inline]
    fn default() -> StencilOpState {
        StencilOpState {
            ops: StateMode::Fixed(Default::default()),
            compare_mask: StateMode::Fixed(u32::MAX),
            write_mask: StateMode::Fixed(u32::MAX),
            reference: StateMode::Fixed(u32::MAX),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StencilOps {
    /// The operation to perform when the stencil test failed.
    pub fail_op: StencilOp,

    /// The operation to perform when both the depth test and the stencil test passed.
    pub pass_op: StencilOp,

    /// The operation to perform when the stencil test passed but the depth test failed.
    pub depth_fail_op: StencilOp,

    /// The comparison to perform between the existing stencil value in the stencil buffer, and
    /// the reference value (given by `reference`).
    pub compare_op: CompareOp,
}

impl Default for StencilOps {
    /// Creates a `StencilOps` with no dynamic state, `compare_op` set to `Never` and the stencil
    /// operations set to `Keep`.
    #[inline]
    fn default() -> Self {
        Self {
            pass_op: StencilOp::Keep,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            compare_op: CompareOp::Never,
        }
    }
}

/// Operation to perform after the depth and stencil tests.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum StencilOp {
    Keep = ash::vk::StencilOp::KEEP.as_raw(),
    Zero = ash::vk::StencilOp::ZERO.as_raw(),
    Replace = ash::vk::StencilOp::REPLACE.as_raw(),
    IncrementAndClamp = ash::vk::StencilOp::INCREMENT_AND_CLAMP.as_raw(),
    DecrementAndClamp = ash::vk::StencilOp::DECREMENT_AND_CLAMP.as_raw(),
    Invert = ash::vk::StencilOp::INVERT.as_raw(),
    IncrementAndWrap = ash::vk::StencilOp::INCREMENT_AND_WRAP.as_raw(),
    DecrementAndWrap = ash::vk::StencilOp::DECREMENT_AND_WRAP.as_raw(),
}

impl From<StencilOp> for ash::vk::StencilOp {
    #[inline]
    fn from(val: StencilOp) -> Self {
        Self::from_raw(val as i32)
    }
}

/// Specifies a face for stencil operations.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum StencilFaces {
    Front = ash::vk::StencilFaceFlags::FRONT.as_raw(),
    Back = ash::vk::StencilFaceFlags::BACK.as_raw(),
    FrontAndBack = ash::vk::StencilFaceFlags::FRONT_AND_BACK.as_raw(),
}

impl From<StencilFaces> for ash::vk::StencilFaceFlags {
    #[inline]
    fn from(val: StencilFaces) -> Self {
        Self::from_raw(val as u32)
    }
}

/// Specifies a dynamic state value for the front and back faces.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DynamicStencilValue {
    pub front: u32,
    pub back: u32,
}

/// Specifies how two values should be compared to decide whether a test passes or fails.
///
/// Used for both depth testing and stencil testing.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum CompareOp {
    /// The test never passes.
    Never = ash::vk::CompareOp::NEVER.as_raw(),
    /// The test passes if `value < reference_value`.
    Less = ash::vk::CompareOp::LESS.as_raw(),
    /// The test passes if `value == reference_value`.
    Equal = ash::vk::CompareOp::EQUAL.as_raw(),
    /// The test passes if `value <= reference_value`.
    LessOrEqual = ash::vk::CompareOp::LESS_OR_EQUAL.as_raw(),
    /// The test passes if `value > reference_value`.
    Greater = ash::vk::CompareOp::GREATER.as_raw(),
    /// The test passes if `value != reference_value`.
    NotEqual = ash::vk::CompareOp::NOT_EQUAL.as_raw(),
    /// The test passes if `value >= reference_value`.
    GreaterOrEqual = ash::vk::CompareOp::GREATER_OR_EQUAL.as_raw(),
    /// The test always passes.
    Always = ash::vk::CompareOp::ALWAYS.as_raw(),
}

impl From<CompareOp> for ash::vk::CompareOp {
    #[inline]
    fn from(val: CompareOp) -> Self {
        Self::from_raw(val as i32)
    }
}
