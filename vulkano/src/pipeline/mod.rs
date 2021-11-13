// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Describes a graphical or compute operation.
//!
//! In Vulkan, before you can add a draw or a compute command to a command buffer you have to
//! create a *pipeline object* that describes this command.
//!
//! When you create a pipeline object, the implementation will usually generate some GPU machine
//! code that will execute the operation (similar to a compiler that generates an executable for
//! the CPU). Consequently it is a CPU-intensive operation that should be performed at
//! initialization or during a loading screen.
//!
//! There are two kinds of pipelines:
//!
//! - `ComputePipeline`s, for compute operations (general-purpose operations that read/write data
//!   in buffers or raw pixels in images).
//! - `GraphicsPipeline`s, for graphical operations (operations that take vertices as input and
//!   write pixels to a framebuffer).
//!
//! # Creating a compute pipeline.
//!
//! In order to create a compute pipeline, you first need a *shader entry point*.
//!
//! TODO: write the rest
//! For now vulkano has no "clean" way to create shaders ; everything's a bit hacky
//!
//! # Creating a graphics pipeline
//!
//! A graphics operation takes vertices or vertices and indices as input, and writes pixels to a
//! framebuffer. It consists of multiple steps:
//!
//! - A *shader* named the *vertex shader* is run once for each vertex of the input.
//! - Vertices are assembled into primitives.
//! - Optionally, a shader named the *tessellation control shader* is run once for each primitive
//!   and indicates the tessellation level to apply for this primitive.
//! - Optionally, a shader named the *tessellation evaluation shader* is run once for each vertex,
//!   including the ones newly created by the tessellation.
//! - Optionally, a shader named the *geometry shader* is run once for each line or triangle.
//! - The vertex coordinates (as outputted by the geometry shader, or by the tessellation
//!   evaluation shader if there's no geometry shader, or by the vertex shader if there's no
//!   geometry shader nor tessellation evaluation shader) are turned into screen-space coordinates.
//! - The list of pixels that cover each triangle are determined.
//! - A shader named the fragment shader is run once for each pixel that covers one of the
//!   triangles.
//! - The depth test and/or the stencil test are performed.
//! - The output of the fragment shader is written to the framebuffer attachments, possibly by
//!   mixing it with the existing values.
//!
//! All the sub-modules of this module (with the exception of `cache`) correspond to the various
//! stages of graphical pipelines.
//!
//! > **Note**: With the exception of the addition of the tessellation shaders and the geometry
//! > shader, these steps haven't changed in the past decade. If you are familiar with shaders in
//! > OpenGL 2 for example, don't worry as it works in the same in Vulkan.
//!
//! > **Note**: All the stages that consist in executing a shader are performed by a microprocessor
//! > (unless you happen to use a software implementation of Vulkan). As for the other stages,
//! > some hardware (usually desktop graphics cards) have dedicated chips that will execute them
//! > while some other hardware (usually mobile) perform them with the microprocessor as well. In
//! > the latter situation, the implementation will usually glue these steps to your shaders.
//!
//! Creating a graphics pipeline follows the same principle as a compute pipeline, except that
//! you must pass multiple shaders alongside with configuration for the other steps.
//!
//! TODO: add an example

// TODO: graphics pipeline params are deprecated, but are still the primary implementation in order
// to avoid duplicating code, so we hide the warnings for now
#![allow(deprecated)]

pub use self::compute_pipeline::ComputePipeline;
pub use self::compute_pipeline::ComputePipelineCreationError;
pub use self::graphics_pipeline::GraphicsPipeline;
pub use self::graphics_pipeline::GraphicsPipelineBuilder;
pub use self::graphics_pipeline::GraphicsPipelineCreationError;
use self::layout::PipelineLayout;
use std::sync::Arc;

pub mod cache;
pub mod color_blend;
mod compute_pipeline;
pub mod depth_stencil;
pub mod discard_rectangle;
mod graphics_pipeline;
pub mod input_assembly;
pub mod layout;
pub mod multisample;
pub mod rasterization;
pub mod shader;
pub mod tessellation;
pub mod vertex;
pub mod viewport;

// A trait for operations shared between pipeline types.
pub trait Pipeline {
    /// Returns the bind point of this pipeline.
    fn bind_point(&self) -> PipelineBindPoint;

    /// Returns the pipeline layout used in this pipeline.
    fn layout(&self) -> &Arc<PipelineLayout>;

    /// Returns the number of descriptor sets actually accessed by this pipeline. This may be less
    /// than the number of sets in the pipeline layout.
    fn num_used_descriptor_sets(&self) -> u32;
}

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
