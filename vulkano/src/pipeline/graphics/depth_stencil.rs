// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Configures the operation of the depth, stencil and depth bounds tests.
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

use crate::{
    device::Device,
    macros::{vulkan_bitflags, vulkan_enum},
    pipeline::StateMode,
    Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version,
};
use std::ops::RangeInclusive;

/// The state in a graphics pipeline describing how the depth, depth bounds and stencil tests
/// should behave.
#[derive(Clone, Debug)]
pub struct DepthStencilState {
    /// Additional properties of the depth/stencil state.
    ///
    /// The default value is empty.
    pub flags: DepthStencilStateFlags,

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

    pub _ne: crate::NonExhaustive,
}

impl DepthStencilState {
    /// Creates a `DepthStencilState` where all tests are disabled and have no effect.
    #[inline]
    pub fn disabled() -> Self {
        Self {
            flags: DepthStencilStateFlags::empty(),
            depth: Default::default(),
            depth_bounds: Default::default(),
            stencil: Default::default(),
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Creates a `DepthStencilState` with a `Less` depth test, `depth_write` set to true, and other
    /// tests disabled.
    #[inline]
    pub fn simple_depth_test() -> Self {
        Self {
            flags: DepthStencilStateFlags::empty(),
            depth: Some(DepthState {
                enable_dynamic: false,
                compare_op: StateMode::Fixed(CompareOp::Less),
                write_enable: StateMode::Fixed(true),
            }),
            depth_bounds: Default::default(),
            stencil: Default::default(),
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            flags,
            ref depth,
            ref depth_bounds,
            ref stencil,
            _ne: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkPipelineDepthStencilStateCreateInfo-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if let Some(depth_state) = depth {
            depth_state
                .validate(device)
                .map_err(|err| err.add_context("depth"))?;
        }

        if let Some(depth_bounds_state) = depth_bounds {
            if !device.enabled_features().depth_bounds {
                return Err(ValidationError {
                    context: "depth_bounds".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "depth_bounds",
                    )])]),
                    vuids: &[
                        "VUID-VkPipelineDepthStencilStateCreateInfo-depthBoundsTestEnable-00598",
                    ],
                });
            }

            depth_bounds_state
                .validate(device)
                .map_err(|err| err.add_context("depth_bounds"))?;
        }

        if let Some(stencil_state) = stencil {
            stencil_state
                .validate(device)
                .map_err(|err| err.add_context("stencil"))?;
        }

        Ok(())
    }
}

impl Default for DepthStencilState {
    /// Returns [`DepthStencilState::disabled()`].
    #[inline]
    fn default() -> Self {
        DepthStencilState::disabled()
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of the depth/stencil state.
    DepthStencilStateFlags = PipelineDepthStencilStateCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS = RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_rasterization_order_attachment_access)]),
        RequiresAllOf([DeviceExtension(arm_rasterization_order_attachment_access)]),
    ]), */

    /* TODO: enable
    // TODO: document
    RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS = RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_rasterization_order_attachment_access)]),
        RequiresAllOf([DeviceExtension(arm_rasterization_order_attachment_access)]),
    ]), */
}

/// The state in a graphics pipeline describing how the depth test should behave when enabled.
#[derive(Clone, Copy, Debug)]
pub struct DepthState {
    /// Sets whether depth testing should be enabled and disabled dynamically. If set to `false`,
    /// depth testing is always enabled.
    ///
    /// If set to `true`, the device API version must be at least 1.3, or the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub enable_dynamic: bool,

    /// Sets whether the value in the depth buffer will be updated when the depth test succeeds.
    ///
    /// If set to `Dynamic`, the device API version must be at least 1.3, or the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub write_enable: StateMode<bool>,

    /// Comparison operation to use between the depth value of each incoming fragment and the depth
    /// value currently in the depth buffer.
    ///
    /// If set to `Dynamic`, the device API version must be at least 1.3, or the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub compare_op: StateMode<CompareOp>,
}

impl DepthState {
    pub(crate) fn validate(self, device: &Device) -> Result<(), ValidationError> {
        let Self {
            enable_dynamic,
            write_enable,
            compare_op,
        } = self;

        if enable_dynamic
            && !(device.api_version() >= Version::V1_3
                || device.enabled_features().extended_dynamic_state)
        {
            return Err(ValidationError {
                context: "enable_dynamic".into(),
                problem: "is `true`".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                // vuids?
                ..Default::default()
            });
        }

        match write_enable {
            StateMode::Fixed(_) => (),
            StateMode::Dynamic => {
                if !(device.api_version() >= Version::V1_3
                    || device.enabled_features().extended_dynamic_state)
                {
                    return Err(ValidationError {
                        context: "write_enable".into(),
                        problem: "is dynamic".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                            RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                        ]),
                        // vuids?
                        ..Default::default()
                    });
                }
            }
        }

        match compare_op {
            StateMode::Fixed(compare_op) => {
                compare_op
                    .validate_device(device)
                    .map_err(|err| ValidationError {
                        context: "compare_op".into(),
                        vuids: &[
                            "VUID-VkPipelineDepthStencilStateCreateInfo-depthCompareOp-parameter",
                        ],
                        ..ValidationError::from_requirement(err)
                    })?;
            }
            StateMode::Dynamic => {
                if !(device.api_version() >= Version::V1_3
                    || device.enabled_features().extended_dynamic_state)
                {
                    return Err(ValidationError {
                        context: "compare_op".into(),
                        problem: "is dynamic".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                            RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                        ]),
                        // vuids?
                        ..Default::default()
                    });
                }
            }
        }

        Ok(())
    }
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
    /// If set to `true`, the device API version must be at least 1.3, or the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub enable_dynamic: bool,

    /// The minimum and maximum depth values to use for the test. Fragments with values outside this
    /// range are discarded.
    ///
    /// If set to `Dynamic`, the device API version must be at least 1.3, or the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub bounds: StateMode<RangeInclusive<f32>>,
}

impl DepthBoundsState {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            enable_dynamic,
            ref bounds,
        } = self;

        if enable_dynamic
            && !(device.api_version() >= Version::V1_3
                || device.enabled_features().extended_dynamic_state)
        {
            return Err(ValidationError {
                context: "enable_dynamic".into(),
                problem: "is `true`".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                // vuids?
                ..Default::default()
            });
        }

        if let StateMode::Fixed(bounds) = bounds {
            if !device.enabled_extensions().ext_depth_range_unrestricted {
                if !(0.0..1.0).contains(bounds.start()) {
                    return Err(ValidationError {
                        context: "bounds.start".into(),
                        problem: "is not between 0.0 and 1.0 inclusive".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceExtension("ext_depth_range_unrestricted"),
                        ])]),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-02510"],
                    });
                }

                if !(0.0..1.0).contains(bounds.end()) {
                    return Err(ValidationError {
                        context: "bounds.end".into(),
                        problem: "is not between 0.0 and 1.0 inclusive".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceExtension("ext_depth_range_unrestricted"),
                        ])]),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-02510"],
                    });
                }
            }
        }

        Ok(())
    }
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
    /// If set to `true`, the device API version must be at least 1.3, or the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub enable_dynamic: bool,

    /// The stencil operation state to use for points and lines, and for triangles whose front is
    /// facing the user.
    pub front: StencilOpState,

    /// The stencil operation state to use for triangles whose back is facing the user.
    pub back: StencilOpState,
}

impl StencilState {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &StencilState {
            enable_dynamic,
            ref front,
            ref back,
        } = self;

        if enable_dynamic
            && !(device.api_version() >= Version::V1_3
                || device.enabled_features().extended_dynamic_state)
        {
            return Err(ValidationError {
                context: "enable_dynamic".into(),
                problem: "is `true`".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                // vuids?
                ..Default::default()
            });
        }

        match (front.ops, back.ops) {
            (StateMode::Fixed(front_ops), StateMode::Fixed(back_ops)) => {
                front_ops
                    .validate(device)
                    .map_err(|err| err.add_context("front.ops"))?;
                back_ops
                    .validate(device)
                    .map_err(|err| err.add_context("back.ops"))?;
            }
            (StateMode::Dynamic, StateMode::Dynamic) => {
                if !(device.api_version() >= Version::V1_3
                    || device.enabled_features().extended_dynamic_state)
                {
                    return Err(ValidationError {
                        problem: "`front.ops` and `back.ops` are dynamic".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                            RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                        ]),
                        // vuids?
                        ..Default::default()
                    });
                }
            }
            _ => {
                return Err(ValidationError {
                    problem: "`front.ops` and `back.ops` are \
                        not both fixed or both dynamic"
                        .into(),
                    // vuids?
                    ..Default::default()
                });
            }
        }

        if !matches!(
            (front.compare_mask, back.compare_mask),
            (StateMode::Fixed(_), StateMode::Fixed(_)) | (StateMode::Dynamic, StateMode::Dynamic)
        ) {
            return Err(ValidationError {
                problem: "`front.compare_mask` and `back.compare_mask` are \
                    not both fixed or both dynamic"
                    .into(),
                // vuids?
                ..Default::default()
            });
        }

        if !matches!(
            (front.write_mask, back.write_mask),
            (StateMode::Fixed(_), StateMode::Fixed(_)) | (StateMode::Dynamic, StateMode::Dynamic)
        ) {
            return Err(ValidationError {
                problem: "`front.write_mask` and `back.write_mask` are \
                    not both fixed or both dynamic"
                    .into(),
                // vuids?
                ..Default::default()
            });
        }

        if !matches!(
            (front.reference, back.reference),
            (StateMode::Fixed(_), StateMode::Fixed(_)) | (StateMode::Dynamic, StateMode::Dynamic)
        ) {
            return Err(ValidationError {
                problem: "`front.reference` and `back.reference` are \
                    not both fixed or both dynamic"
                    .into(),
                // vuids?
                ..Default::default()
            });
        }

        Ok(())
    }
}

/// Stencil test operations for a single face.
#[derive(Clone, Copy, Debug)]
pub struct StencilOpState {
    /// The stencil operations to perform.
    ///
    /// If set to `Dynamic`, the device API version must be at least 1.3, or the
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
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if culling is disabled, and the `reference` values of the front and back face
    /// are not equal, then the
    /// [`separate_stencil_mask_ref`](crate::device::Features::separate_stencil_mask_ref)
    /// feature must be enabled on the device.
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

impl StencilOps {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            fail_op,
            pass_op,
            depth_fail_op,
            compare_op,
        } = self;

        fail_op
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "fail_op".into(),
                vuids: &["VUID-VkStencilOpState-failOp-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        pass_op
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "pass_op".into(),
                vuids: &["VUID-VkStencilOpState-passOp-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        depth_fail_op
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "depth_fail_op".into(),
                vuids: &["VUID-VkStencilOpState-depthFailOp-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        compare_op
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "compare_op".into(),
                vuids: &["VUID-VkStencilOpState-compareOp-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        Ok(())
    }
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

vulkan_enum! {
    #[non_exhaustive]

    /// Operation to perform after the depth and stencil tests.
    StencilOp = StencilOp(i32);

    // TODO: document
    Keep = KEEP,

    // TODO: document
    Zero = ZERO,

    // TODO: document
    Replace = REPLACE,

    // TODO: document
    IncrementAndClamp = INCREMENT_AND_CLAMP,

    // TODO: document
    DecrementAndClamp = DECREMENT_AND_CLAMP,

    // TODO: document
    Invert = INVERT,

    // TODO: document
    IncrementAndWrap = INCREMENT_AND_WRAP,

    // TODO: document
    DecrementAndWrap = DECREMENT_AND_WRAP,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies a face for stencil operations.
    StencilFaces = StencilFaceFlags(u32);

    // TODO: document
    Front = FRONT,

    // TODO: document
    Back = BACK,

    // TODO: document
    FrontAndBack = FRONT_AND_BACK,
}

/// Specifies a dynamic state value for the front and back faces.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DynamicStencilValue {
    pub front: u32,
    pub back: u32,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies how two values should be compared to decide whether a test passes or fails.
    ///
    /// Used for both depth testing and stencil testing.
    CompareOp = CompareOp(i32);

    /// The test never passes.
    Never = NEVER,

    /// The test passes if `value < reference_value`.
    Less = LESS,

    /// The test passes if `value == reference_value`.
    Equal = EQUAL,

    /// The test passes if `value <= reference_value`.
    LessOrEqual = LESS_OR_EQUAL,

    /// The test passes if `value > reference_value`.
    Greater = GREATER,

    /// The test passes if `value != reference_value`.
    NotEqual = NOT_EQUAL,

    /// The test passes if `value >= reference_value`.
    GreaterOrEqual = GREATER_OR_EQUAL,

    /// The test always passes.
    Always = ALWAYS,
}
