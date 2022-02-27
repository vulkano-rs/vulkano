// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A pipeline that performs graphics processing operations.
//!
//! Unlike a compute pipeline, which performs general-purpose work, a graphics pipeline is geared
//! specifically towards doing graphical processing. To that end, it consists of several shaders,
//! with additional state and glue logic in between.
//!
//! A graphics pipeline performs many separate steps, that execute more or less in sequence.
//! Due to the parallel nature of a GPU, no strict ordering guarantees may exist.
//!
//! 1. Vertex input and assembly: vertex input data is read from data buffers and then assembled
//!    into primitives (points, lines, triangles etc.).
//! 2. Vertex shader invocations: the vertex data of each primitive is fed as input to the vertex
//!    shader, which performs transformations on the data and generates new data as output.
//! 3. (Optional) Tessellation: primitives are subdivided by the operations of two shaders, the
//!    tessellation control and tessellation evaluation shaders. The control shader produces the
//!    tessellation level to apply for the primitive, while the evaluation shader postprocesses the
//!    newly created vertices.
//! 4. (Optional) Geometry shading: whole primitives are fed as input and processed into a new set
//!    of output primitives.
//! 5. Vertex post-processing, including:
//!    - Clipping primitives to the view frustum and user-defined clipping planes.
//!    - Perspective division.
//!    - Viewport mapping.
//! 6. Rasterization: converting primitives into a two-dimensional representation. Primitives may be
//!    discarded depending on their orientation, and are then converted into a collection of
//!    fragments that are processed further.
//! 7. Fragment operations. These include invocations of the fragment shader, which generates the
//!    values to be written to the color attachment. Various testing and discarding operations can
//!    be performed both before and after the fragment shader ("early" and "late" fragment tests),
//!    including:
//!    - Discard rectangle test
//!    - Scissor test
//!    - Sample mask test
//!    - Depth bounds test
//!    - Stencil test
//!    - Depth test
//! 8. Color attachment output: the final pixel data is written to a framebuffer. Blending and
//!    logical operations can be applied to combine incoming pixel data with data already present
//!    in the framebuffer.
//!
//! A graphics pipeline contains many configuration options, which are grouped into collections of
//! "state". Often, these directly correspond to one or more steps in the graphics pipeline. Each
//! state collection has a dedicated submodule.
//!
//! Once a graphics pipeline has been created, you can execute it by first *binding* it in a command
//! buffer, binding the necessary vertex buffers, binding any descriptor sets, setting push
//! constants, and setting any dynamic state that the pipeline may need. Then you issue a `draw`
//! command.

pub use self::builder::GraphicsPipelineBuilder;
pub use self::creation_error::GraphicsPipelineCreationError;
use crate::device::{Device, DeviceOwned};
use crate::pipeline::graphics::color_blend::ColorBlendState;
use crate::pipeline::graphics::depth_stencil::DepthStencilState;
use crate::pipeline::graphics::discard_rectangle::DiscardRectangleState;
use crate::pipeline::graphics::input_assembly::InputAssemblyState;
use crate::pipeline::graphics::multisample::MultisampleState;
use crate::pipeline::graphics::rasterization::RasterizationState;
use crate::pipeline::graphics::tessellation::TessellationState;
use crate::pipeline::graphics::vertex_input::VertexInputState;
use crate::pipeline::graphics::viewport::ViewportState;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::{DynamicState, Pipeline, PipelineBindPoint};
use crate::render_pass::Subpass;
use crate::shader::{DescriptorRequirements, ShaderStage};
use crate::VulkanObject;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::ptr;
use std::sync::Arc;

mod builder;
pub mod color_blend;
mod creation_error;
pub mod depth_stencil;
pub mod discard_rectangle;
pub mod input_assembly;
pub mod multisample;
pub mod rasterization;
pub mod tessellation;
pub mod vertex_input;
pub mod viewport;
// FIXME: restore
//mod tests;

/// Defines how the implementation should perform a draw operation.
///
/// This object contains the shaders and the various fixed states that describe how the
/// implementation should perform the various operations needed by a draw command.
pub struct GraphicsPipeline {
    handle: ash::vk::Pipeline,
    device: Arc<Device>,
    layout: Arc<PipelineLayout>,
    subpass: Subpass,
    // TODO: replace () with an object that describes the shaders in some way.
    shaders: HashMap<ShaderStage, ()>,
    descriptor_requirements: HashMap<(u32, u32), DescriptorRequirements>,
    num_used_descriptor_sets: u32,

    vertex_input_state: VertexInputState,
    input_assembly_state: InputAssemblyState,
    tessellation_state: Option<TessellationState>,
    viewport_state: Option<ViewportState>,
    discard_rectangle_state: Option<DiscardRectangleState>,
    rasterization_state: RasterizationState,
    multisample_state: Option<MultisampleState>,
    depth_stencil_state: Option<DepthStencilState>,
    color_blend_state: Option<ColorBlendState>,
    dynamic_state: HashMap<DynamicState, bool>,
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
        VertexInputState,
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
        &self.device
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
    pub fn vertex_input_state(&self) -> &VertexInputState {
        &self.vertex_input_state
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

impl Pipeline for GraphicsPipeline {
    #[inline]
    fn bind_point(&self) -> PipelineBindPoint {
        PipelineBindPoint::Graphics
    }

    #[inline]
    fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }

    #[inline]
    fn num_used_descriptor_sets(&self) -> u32 {
        self.num_used_descriptor_sets
    }
}

unsafe impl DeviceOwned for GraphicsPipeline {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl fmt::Debug for GraphicsPipeline {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan graphics pipeline {:?}>", self.handle)
    }
}

unsafe impl VulkanObject for GraphicsPipeline {
    type Object = ash::vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> ash::vk::Pipeline {
        self.handle
    }
}

impl Drop for GraphicsPipeline {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_pipeline(self.device.internal_object(), self.handle, ptr::null());
        }
    }
}

impl PartialEq for GraphicsPipeline {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device && self.handle == other.handle
    }
}

impl Eq for GraphicsPipeline {}

impl Hash for GraphicsPipeline {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device.hash(state);
    }
}
