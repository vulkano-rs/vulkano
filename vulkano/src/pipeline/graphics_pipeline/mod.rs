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
use crate::buffer::BufferAccess;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::shader::ShaderInterface;
use crate::pipeline::vertex::BufferlessDefinition;
use crate::pipeline::vertex::IncompatibleVertexDefinitionError;
use crate::pipeline::vertex::VertexDefinition;
use crate::pipeline::vertex::VertexSource;
use crate::render_pass::RenderPass;
use crate::render_pass::Subpass;
use crate::SafeDeref;
use crate::VulkanObject;
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
pub struct GraphicsPipeline<VertexDefinition> {
    inner: Inner,
    layout: Arc<PipelineLayout>,

    subpass: Subpass,

    vertex_definition: VertexDefinition,

    dynamic_line_width: bool,
    dynamic_viewport: bool,
    dynamic_scissor: bool,
    dynamic_depth_bias: bool,
    dynamic_depth_bounds: bool,
    dynamic_stencil_compare_mask: bool,
    dynamic_stencil_write_mask: bool,
    dynamic_stencil_reference: bool,
    dynamic_blend_constants: bool,

    num_viewports: u32,
}

#[derive(PartialEq, Eq, Hash)]
struct Inner {
    pipeline: ash::vk::Pipeline,
    device: Arc<Device>,
}

impl GraphicsPipeline<()> {
    /// Starts the building process of a graphics pipeline. Returns a builder object that you can
    /// fill with the various parameters.
    pub fn start<'a>() -> GraphicsPipelineBuilder<
        'static,
        'static,
        'static,
        'static,
        'static,
        BufferlessDefinition,
        (),
        (),
        (),
        (),
        (),
    > {
        GraphicsPipelineBuilder::new()
    }
}

impl<Mv> GraphicsPipeline<Mv> {
    /// Returns the vertex definition used in the constructor.
    #[inline]
    pub fn vertex_definition(&self) -> &Mv {
        &self.vertex_definition
    }

    /// Returns the device used to create this pipeline.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }

    /// Returns the pass used in the constructor.
    #[inline]
    pub fn subpass(&self) -> Subpass {
        self.subpass.clone()
    }

    /// Returns the render pass used in the constructor.
    #[inline]
    pub fn render_pass(&self) -> &Arc<RenderPass> {
        self.subpass.render_pass()
    }

    /// Returns true if the line width used by this pipeline is dynamic.
    #[inline]
    pub fn has_dynamic_line_width(&self) -> bool {
        self.dynamic_line_width
    }

    /// Returns the number of viewports and scissors of this pipeline.
    #[inline]
    pub fn num_viewports(&self) -> u32 {
        self.num_viewports
    }

    /// Returns true if the viewports used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_viewports(&self) -> bool {
        self.dynamic_viewport
    }

    /// Returns true if the scissors used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_scissors(&self) -> bool {
        self.dynamic_scissor
    }

    /// Returns true if the depth bounds used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_depth_bounds(&self) -> bool {
        self.dynamic_depth_bounds
    }

    /// Returns true if the stencil compare masks used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_stencil_compare_mask(&self) -> bool {
        self.dynamic_stencil_compare_mask
    }

    /// Returns true if the stencil write masks used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_stencil_write_mask(&self) -> bool {
        self.dynamic_stencil_write_mask
    }

    /// Returns true if the stencil references used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_stencil_reference(&self) -> bool {
        self.dynamic_stencil_reference
    }
}

unsafe impl<Mv> DeviceOwned for GraphicsPipeline<Mv> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }
}

impl<Mv> fmt::Debug for GraphicsPipeline<Mv> {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan graphics pipeline {:?}>", self.inner.pipeline)
    }
}

unsafe impl<Mv> VulkanObject for GraphicsPipeline<Mv> {
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

/// Trait implemented on objects that reference a graphics pipeline. Can be made into a trait
/// object.
/// When using this trait `AutoCommandBufferBuilder::draw*` calls will need the buffers to be
/// wrapped in a `vec!()`.
pub unsafe trait GraphicsPipelineAbstract:
    VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>> + DeviceOwned
{
    /// Returns an opaque object that represents the inside of the graphics pipeline.
    fn inner(&self) -> GraphicsPipelineSys;

    /// Returns the pipeline layout used in the constructor.
    fn layout(&self) -> &Arc<PipelineLayout>;

    /// Returns the subpass this graphics pipeline is rendering to.
    fn subpass(&self) -> &Subpass;

    /// Returns true if the line width used by this pipeline is dynamic.
    fn has_dynamic_line_width(&self) -> bool;

    /// Returns the number of viewports and scissors of this pipeline.
    fn num_viewports(&self) -> u32;

    /// Returns true if the viewports used by this pipeline are dynamic.
    fn has_dynamic_viewports(&self) -> bool;

    /// Returns true if the scissors used by this pipeline are dynamic.
    fn has_dynamic_scissors(&self) -> bool;

    /// Returns true if the depth bounds used by this pipeline are dynamic.
    fn has_dynamic_depth_bounds(&self) -> bool;

    /// Returns true if the stencil compare masks used by this pipeline are dynamic.
    fn has_dynamic_stencil_compare_mask(&self) -> bool;

    /// Returns true if the stencil write masks used by this pipeline are dynamic.
    fn has_dynamic_stencil_write_mask(&self) -> bool;

    /// Returns true if the stencil references used by this pipeline are dynamic.
    fn has_dynamic_stencil_reference(&self) -> bool;
}

unsafe impl<Mv> GraphicsPipelineAbstract for GraphicsPipeline<Mv>
where
    Mv: VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>>,
{
    #[inline]
    fn inner(&self) -> GraphicsPipelineSys {
        GraphicsPipelineSys(self.inner.pipeline, PhantomData)
    }

    /// Returns the pipeline layout used in the constructor.
    #[inline]
    fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }

    #[inline]
    fn subpass(&self) -> &Subpass {
        &self.subpass
    }

    #[inline]
    fn has_dynamic_line_width(&self) -> bool {
        self.dynamic_line_width
    }

    #[inline]
    fn num_viewports(&self) -> u32 {
        self.num_viewports
    }

    #[inline]
    fn has_dynamic_viewports(&self) -> bool {
        self.dynamic_viewport
    }

    #[inline]
    fn has_dynamic_scissors(&self) -> bool {
        self.dynamic_scissor
    }

    #[inline]
    fn has_dynamic_depth_bounds(&self) -> bool {
        self.dynamic_depth_bounds
    }

    #[inline]
    fn has_dynamic_stencil_compare_mask(&self) -> bool {
        self.dynamic_stencil_compare_mask
    }

    #[inline]
    fn has_dynamic_stencil_write_mask(&self) -> bool {
        self.dynamic_stencil_write_mask
    }

    #[inline]
    fn has_dynamic_stencil_reference(&self) -> bool {
        self.dynamic_stencil_reference
    }
}

unsafe impl<T> GraphicsPipelineAbstract for T
where
    T: SafeDeref,
    T::Target: GraphicsPipelineAbstract,
{
    #[inline]
    fn inner(&self) -> GraphicsPipelineSys {
        GraphicsPipelineAbstract::inner(&**self)
    }

    #[inline]
    fn layout(&self) -> &Arc<PipelineLayout> {
        (**self).layout()
    }

    #[inline]
    fn subpass(&self) -> &Subpass {
        (**self).subpass()
    }

    #[inline]
    fn has_dynamic_line_width(&self) -> bool {
        (**self).has_dynamic_line_width()
    }

    #[inline]
    fn num_viewports(&self) -> u32 {
        (**self).num_viewports()
    }

    #[inline]
    fn has_dynamic_viewports(&self) -> bool {
        (**self).has_dynamic_viewports()
    }

    #[inline]
    fn has_dynamic_scissors(&self) -> bool {
        (**self).has_dynamic_scissors()
    }

    #[inline]
    fn has_dynamic_depth_bounds(&self) -> bool {
        (**self).has_dynamic_depth_bounds()
    }

    #[inline]
    fn has_dynamic_stencil_compare_mask(&self) -> bool {
        (**self).has_dynamic_stencil_compare_mask()
    }

    #[inline]
    fn has_dynamic_stencil_write_mask(&self) -> bool {
        (**self).has_dynamic_stencil_write_mask()
    }

    #[inline]
    fn has_dynamic_stencil_reference(&self) -> bool {
        (**self).has_dynamic_stencil_reference()
    }
}

impl<Mv> PartialEq for GraphicsPipeline<Mv>
where
    Mv: VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<Mv> Eq for GraphicsPipeline<Mv> where Mv: VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>>
{}

impl<Mv> Hash for GraphicsPipeline<Mv>
where
    Mv: VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>>,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl PartialEq for dyn GraphicsPipelineAbstract + Send + Sync {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        GraphicsPipelineAbstract::inner(self).0 == GraphicsPipelineAbstract::inner(other).0
            && DeviceOwned::device(self) == DeviceOwned::device(other)
    }
}

impl Eq for dyn GraphicsPipelineAbstract + Send + Sync {}

impl Hash for dyn GraphicsPipelineAbstract + Send + Sync {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        GraphicsPipelineAbstract::inner(self).0.hash(state);
        DeviceOwned::device(self).hash(state);
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

unsafe impl<Mv> VertexDefinition for GraphicsPipeline<Mv>
where
    Mv: VertexDefinition,
{
    type BuffersIter = <Mv as VertexDefinition>::BuffersIter;
    type AttribsIter = <Mv as VertexDefinition>::AttribsIter;

    #[inline]
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        self.vertex_definition.definition(interface)
    }
}

unsafe impl<Mv, S> VertexSource<S> for GraphicsPipeline<Mv>
where
    Mv: VertexSource<S>,
{
    #[inline]
    fn decode(&self, s: S) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
        self.vertex_definition.decode(s)
    }
}
