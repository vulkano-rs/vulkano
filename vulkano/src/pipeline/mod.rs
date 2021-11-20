// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Describes a processing operation that will execute on the Vulkan device.
//!
//! In Vulkan, before you can add a draw or a compute command to a command buffer you have to
//! create a *pipeline object* that describes this command.
//!
//! When you create a pipeline object, the implementation will usually generate some GPU machine
//! code that will execute the operation (similar to a compiler that generates an executable for
//! the CPU). Consequently it is a CPU-intensive operation that should be performed at
//! initialization or during a loading screen.

pub use self::compute::ComputePipeline;
pub use self::graphics::GraphicsPipeline;
pub use self::layout::PipelineLayout;
use std::sync::Arc;

pub mod cache;
pub mod compute;
pub mod graphics;
pub mod layout;

/// A trait for operations shared between pipeline types.
pub trait Pipeline {
    /// Returns the bind point of this pipeline.
    fn bind_point(&self) -> PipelineBindPoint;

    /// Returns the pipeline layout used in this pipeline.
    fn layout(&self) -> &Arc<PipelineLayout>;

    /// Returns the number of descriptor sets actually accessed by this pipeline. This may be less
    /// than the number of sets in the pipeline layout.
    fn num_used_descriptor_sets(&self) -> u32;
}

/// The type of a pipeline.
///
/// When binding a pipeline or descriptor sets in a command buffer, the state for each bind point
/// is independent from the others. This means that it is possible, for example, to bind a graphics
/// pipeline without disturbing any bound compute pipeline. Likewise, binding descriptor sets for
/// the `Compute` bind point does not affect sets that were bound to the `Graphics` bind point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum PipelineBindPoint {
    Compute = ash::vk::PipelineBindPoint::COMPUTE.as_raw(),
    Graphics = ash::vk::PipelineBindPoint::GRAPHICS.as_raw(),
}

impl From<PipelineBindPoint> for ash::vk::PipelineBindPoint {
    #[inline]
    fn from(val: PipelineBindPoint) -> Self {
        Self::from_raw(val as i32)
    }
}

/// A particular state value within a graphics pipeline that can be dynamically set by a command
/// buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DynamicState {
    Viewport = ash::vk::DynamicState::VIEWPORT.as_raw(),
    Scissor = ash::vk::DynamicState::SCISSOR.as_raw(),
    LineWidth = ash::vk::DynamicState::LINE_WIDTH.as_raw(),
    DepthBias = ash::vk::DynamicState::DEPTH_BIAS.as_raw(),
    BlendConstants = ash::vk::DynamicState::BLEND_CONSTANTS.as_raw(),
    DepthBounds = ash::vk::DynamicState::DEPTH_BOUNDS.as_raw(),
    StencilCompareMask = ash::vk::DynamicState::STENCIL_COMPARE_MASK.as_raw(),
    StencilWriteMask = ash::vk::DynamicState::STENCIL_WRITE_MASK.as_raw(),
    StencilReference = ash::vk::DynamicState::STENCIL_REFERENCE.as_raw(),
    ViewportWScaling = ash::vk::DynamicState::VIEWPORT_W_SCALING_NV.as_raw(),
    DiscardRectangle = ash::vk::DynamicState::DISCARD_RECTANGLE_EXT.as_raw(),
    SampleLocations = ash::vk::DynamicState::SAMPLE_LOCATIONS_EXT.as_raw(),
    RayTracingPipelineStackSize =
        ash::vk::DynamicState::RAY_TRACING_PIPELINE_STACK_SIZE_KHR.as_raw(),
    ViewportShadingRatePalette = ash::vk::DynamicState::VIEWPORT_SHADING_RATE_PALETTE_NV.as_raw(),
    ViewportCoarseSampleOrder = ash::vk::DynamicState::VIEWPORT_COARSE_SAMPLE_ORDER_NV.as_raw(),
    ExclusiveScissor = ash::vk::DynamicState::EXCLUSIVE_SCISSOR_NV.as_raw(),
    FragmentShadingRate = ash::vk::DynamicState::FRAGMENT_SHADING_RATE_KHR.as_raw(),
    LineStipple = ash::vk::DynamicState::LINE_STIPPLE_EXT.as_raw(),
    CullMode = ash::vk::DynamicState::CULL_MODE_EXT.as_raw(),
    FrontFace = ash::vk::DynamicState::FRONT_FACE_EXT.as_raw(),
    PrimitiveTopology = ash::vk::DynamicState::PRIMITIVE_TOPOLOGY_EXT.as_raw(),
    ViewportWithCount = ash::vk::DynamicState::VIEWPORT_WITH_COUNT_EXT.as_raw(),
    ScissorWithCount = ash::vk::DynamicState::SCISSOR_WITH_COUNT_EXT.as_raw(),
    VertexInputBindingStride = ash::vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE_EXT.as_raw(),
    DepthTestEnable = ash::vk::DynamicState::DEPTH_TEST_ENABLE_EXT.as_raw(),
    DepthWriteEnable = ash::vk::DynamicState::DEPTH_WRITE_ENABLE_EXT.as_raw(),
    DepthCompareOp = ash::vk::DynamicState::DEPTH_COMPARE_OP_EXT.as_raw(),
    DepthBoundsTestEnable = ash::vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE_EXT.as_raw(),
    StencilTestEnable = ash::vk::DynamicState::STENCIL_TEST_ENABLE_EXT.as_raw(),
    StencilOp = ash::vk::DynamicState::STENCIL_OP_EXT.as_raw(),
    VertexInput = ash::vk::DynamicState::VERTEX_INPUT_EXT.as_raw(),
    PatchControlPoints = ash::vk::DynamicState::PATCH_CONTROL_POINTS_EXT.as_raw(),
    RasterizerDiscardEnable = ash::vk::DynamicState::RASTERIZER_DISCARD_ENABLE_EXT.as_raw(),
    DepthBiasEnable = ash::vk::DynamicState::DEPTH_BIAS_ENABLE_EXT.as_raw(),
    LogicOp = ash::vk::DynamicState::LOGIC_OP_EXT.as_raw(),
    PrimitiveRestartEnable = ash::vk::DynamicState::PRIMITIVE_RESTART_ENABLE_EXT.as_raw(),
    ColorWriteEnable = ash::vk::DynamicState::COLOR_WRITE_ENABLE_EXT.as_raw(),
}

impl From<DynamicState> for ash::vk::DynamicState {
    #[inline]
    fn from(val: DynamicState) -> Self {
        Self::from_raw(val as i32)
    }
}

/// Specifies how a dynamic state is handled by a graphics pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateMode<F> {
    /// The pipeline has a fixed value for this state. Previously set dynamic state will be lost
    /// when binding it, and will have to be re-set after binding a pipeline that uses it.
    Fixed(F),
    /// The pipeline expects a dynamic value to be set by a command buffer. Previously set dynamic
    /// state is not disturbed when binding it.
    Dynamic,
}

impl<T> From<Option<T>> for StateMode<T> {
    #[inline]
    fn from(val: Option<T>) -> Self {
        match val {
            Some(x) => StateMode::Fixed(x),
            None => StateMode::Dynamic,
        }
    }
}

impl<T> From<StateMode<T>> for Option<T> {
    #[inline]
    fn from(val: StateMode<T>) -> Self {
        match val {
            StateMode::Fixed(x) => Some(x),
            StateMode::Dynamic => None,
        }
    }
}

/// A variant of `StateMode` that is used for cases where some value is still needed when the state
/// is dynamic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartialStateMode<F, D> {
    Fixed(F),
    Dynamic(D),
}
