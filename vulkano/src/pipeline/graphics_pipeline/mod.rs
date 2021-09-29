// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub use self::builder::GraphicsPipelineBuilder;
pub use self::creation_error::GraphicsPipelineCreationError;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::vertex::BuffersDefinition;
use crate::pipeline::vertex::VertexInput;
use crate::render_pass::Subpass;
use crate::VulkanObject;
use fnv::FnvHashMap;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;
use std::u32;

mod builder;
mod creation_error;
// FIXME: restore
//mod tests;

/// Defines how the implementation should perform a draw operation.
///
/// This object contains the shaders and the various fixed states that describe how the
/// implementation should perform the various operations needed by a draw command.
pub struct GraphicsPipeline {
    inner: Inner,
    layout: Arc<PipelineLayout>,
    subpass: Subpass,
    vertex_input: VertexInput,
    dynamic_state: FnvHashMap<DynamicState, DynamicStateMode>,
    num_viewports: u32,
}

#[derive(PartialEq, Eq, Hash)]
struct Inner {
    pipeline: ash::vk::Pipeline,
    device: Arc<Device>,
}

impl GraphicsPipeline {
    /// Starts the building process of a graphics pipeline. Returns a builder object that you can
    /// fill with the various parameters.
    pub fn start<'a>() -> GraphicsPipelineBuilder<
        'static,
        'static,
        'static,
        'static,
        'static,
        BuffersDefinition,
        (),
        (),
        (),
        (),
        (),
    > {
        GraphicsPipelineBuilder::new()
    }

    /// Returns the device used to create this pipeline.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }

    /// Returns the pipeline layout used to create this pipeline.
    #[inline]
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }

    /// Returns the subpass this graphics pipeline is rendering to.
    #[inline]
    pub fn subpass(&self) -> &Subpass {
        &self.subpass
    }

    /// Returns the vertex input description of the graphics pipeline.
    #[inline]
    pub fn vertex_input(&self) -> &VertexInput {
        &self.vertex_input
    }

    /// Returns the number of viewports and scissors of this pipeline.
    #[inline]
    pub fn num_viewports(&self) -> u32 {
        self.num_viewports
    }

    /// Returns the mode of a particular dynamic state.
    ///
    /// `None` is returned if the pipeline does not contain this state. Previously set dynamic
    /// state is not disturbed when binding it.
    pub fn dynamic_state(&self, state: DynamicState) -> Option<DynamicStateMode> {
        self.dynamic_state.get(&state).copied()
    }

    /// Returns all dynamic states and their modes.
    pub fn dynamic_states(
        &self,
    ) -> impl ExactSizeIterator<Item = (DynamicState, DynamicStateMode)> + '_ {
        self.dynamic_state.iter().map(|(k, v)| (*k, *v))
    }
}

unsafe impl DeviceOwned for GraphicsPipeline {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }
}

impl fmt::Debug for GraphicsPipeline {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan graphics pipeline {:?}>", self.inner.pipeline)
    }
}

unsafe impl VulkanObject for GraphicsPipeline {
    type Object = ash::vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> ash::vk::Pipeline {
        self.inner.pipeline
    }
}

impl Drop for Inner {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_pipeline(self.device.internal_object(), self.pipeline, ptr::null());
        }
    }
}

impl PartialEq for GraphicsPipeline {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for GraphicsPipeline {}

impl Hash for GraphicsPipeline {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

/// Opaque object that represents the inside of the graphics pipeline.
#[derive(Debug, Copy, Clone)]
pub struct GraphicsPipelineSys<'a>(ash::vk::Pipeline, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for GraphicsPipelineSys<'a> {
    type Object = ash::vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> ash::vk::Pipeline {
        self.0
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
pub enum DynamicStateMode {
    /// The pipeline has a fixed value for this state. Previously set dynamic state will be lost
    /// when binding it, and will have to be re-set after binding a pipeline that uses it.
    Fixed,
    /// The pipeline expects a dynamic value to be set by a command buffer. Previously set dynamic
    /// state is not disturbed when binding it.
    Dynamic,
}
