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
use crate::device::{Device, DeviceOwned};
use crate::pipeline::color_blend::ColorBlendState;
use crate::pipeline::depth_stencil::DepthStencilState;
use crate::pipeline::discard_rectangle::DiscardRectangleState;
use crate::pipeline::input_assembly::InputAssemblyState;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::multisample::MultisampleState;
use crate::pipeline::rasterization::RasterizationState;
use crate::pipeline::shader::{DescriptorRequirements, ShaderStage};
use crate::pipeline::tessellation::TessellationState;
use crate::pipeline::vertex::{BuffersDefinition, VertexInput};
use crate::pipeline::viewport::ViewportState;
use crate::pipeline::DynamicState;
use crate::render_pass::Subpass;
use crate::VulkanObject;
use fnv::FnvHashMap;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

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
    // TODO: replace () with an object that describes the shaders in some way.
    shaders: FnvHashMap<ShaderStage, ()>,
    descriptor_requirements: FnvHashMap<(u32, u32), DescriptorRequirements>,

    vertex_input: VertexInput,
    input_assembly_state: InputAssemblyState,
    tessellation_state: Option<TessellationState>,
    viewport_state: Option<ViewportState>,
    discard_rectangle_state: Option<DiscardRectangleState>,
    rasterization_state: RasterizationState,
    multisample_state: Option<MultisampleState>,
    depth_stencil_state: Option<DepthStencilState>,
    color_blend_state: Option<ColorBlendState>,
    dynamic_state: FnvHashMap<DynamicState, bool>,
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

    /// Returns information about a particular shader.
    ///
    /// `None` is returned if the pipeline does not contain this shader.
    ///
    /// Compatibility note: `()` is temporary, it will be replaced with something else in the future.
    // TODO: ^ implement and make this public
    #[inline]
    pub(crate) fn shader(&self, stage: ShaderStage) -> Option<()> {
        self.shaders.get(&stage).copied()
    }

    /// Returns an iterator over the descriptor requirements for this pipeline.
    #[inline]
    pub fn descriptor_requirements(
        &self,
    ) -> impl ExactSizeIterator<Item = ((u32, u32), &DescriptorRequirements)> {
        self.descriptor_requirements
            .iter()
            .map(|(loc, reqs)| (*loc, reqs))
    }

    /// Returns the vertex input state used to create this pipeline.
    #[inline]
    pub fn vertex_input(&self) -> &VertexInput {
        &self.vertex_input
    }

    /// Returns the input assembly state used to create this pipeline.
    #[inline]
    pub fn input_assembly_state(&self) -> &InputAssemblyState {
        &self.input_assembly_state
    }

    /// Returns the tessellation state used to create this pipeline.
    #[inline]
    pub fn tessellation_state(&self) -> Option<&TessellationState> {
        self.tessellation_state.as_ref()
    }

    /// Returns the viewport state used to create this pipeline.
    #[inline]
    pub fn viewport_state(&self) -> Option<&ViewportState> {
        self.viewport_state.as_ref()
    }

    /// Returns the discard rectangle state used to create this pipeline.
    #[inline]
    pub fn discard_rectangle_state(&self) -> Option<&DiscardRectangleState> {
        self.discard_rectangle_state.as_ref()
    }

    /// Returns the rasterization state used to create this pipeline.
    #[inline]
    pub fn rasterization_state(&self) -> &RasterizationState {
        &self.rasterization_state
    }

    /// Returns the multisample state used to create this pipeline.
    #[inline]
    pub fn multisample_state(&self) -> Option<&MultisampleState> {
        self.multisample_state.as_ref()
    }

    /// Returns the depth/stencil state used to create this pipeline.
    #[inline]
    pub fn depth_stencil_state(&self) -> Option<&DepthStencilState> {
        self.depth_stencil_state.as_ref()
    }

    /// Returns the color blend state used to create this pipeline.
    #[inline]
    pub fn color_blend_state(&self) -> Option<&ColorBlendState> {
        self.color_blend_state.as_ref()
    }

    /// Returns whether a particular state is must be dynamically set.
    ///
    /// `None` is returned if the pipeline does not contain this state. Previously set dynamic
    /// state is not disturbed when binding it.
    pub fn dynamic_state(&self, state: DynamicState) -> Option<bool> {
        self.dynamic_state.get(&state).copied()
    }

    /// Returns all potentially dynamic states in the pipeline, and whether they are dynamic or not.
    pub fn dynamic_states(&self) -> impl ExactSizeIterator<Item = (DynamicState, bool)> + '_ {
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
