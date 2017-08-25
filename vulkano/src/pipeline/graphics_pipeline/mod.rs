// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;
use std::u32;

use SafeDeref;
use VulkanObject;
use buffer::BufferAccess;
use descriptor::PipelineLayoutAbstract;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use descriptor::pipeline_layout::PipelineLayoutSys;
use device::Device;
use device::DeviceOwned;
use format::ClearValue;
use framebuffer::LayoutAttachmentDescription;
use framebuffer::LayoutPassDependencyDescription;
use framebuffer::LayoutPassDescription;
use framebuffer::RenderPassAbstract;
use framebuffer::RenderPassDesc;
use framebuffer::RenderPassDescClearValues;
use framebuffer::RenderPassSys;
use framebuffer::Subpass;
use vk;

use pipeline::blend::Blend;
use pipeline::depth_stencil::DepthStencil;
use pipeline::input_assembly::InputAssembly;
use pipeline::multisample::Multisample;
use pipeline::raster::Rasterization;
use pipeline::shader::EmptyEntryPointDummy;
use pipeline::vertex::IncompatibleVertexDefinitionError;
use pipeline::vertex::SingleBufferDefinition;
use pipeline::vertex::VertexDefinition;
use pipeline::vertex::VertexSource;
use pipeline::viewport::ViewportsState;

pub use self::builder::GraphicsPipelineBuilder;
pub use self::creation_error::GraphicsPipelineCreationError;

mod builder;
mod creation_error;
// FIXME: restore
//mod tests;

/// Description of a `GraphicsPipeline`.
pub(crate) struct GraphicsPipelineParams<
 Vdef,
 Vs,
 Tcs,
 Tes,
 Gs,
 Fs,
 Rp>
{
    /// Describes the layout of the vertex input.
    ///
    /// For example if you want to pass a vertex buffer and an instance buffer, this parameter
    /// should describe it as well as the offsets and data type of various vertex attributes.
    ///
    /// Must implement the `VertexDefinition` trait.
    pub vertex_input: Vdef,

    /// The entry point of the vertex shader that will be run on the vertex input.
    pub vertex_shader: Vs,

    /// Describes how vertices should be assembled into primitives. Essentially contains the type
    /// of primitives.
    pub input_assembly: InputAssembly,

    /// Parameters of the tessellation stage. `None` if you don't want to use tessellation.
    /// If you use tessellation, you must enable the `tessellation_shader` feature on the device.
    pub tessellation: Option<GraphicsPipelineParamsTess<Tcs, Tes>>,

    /// The entry point of the geometry shader. `None` if you don't want a geometry shader.
    /// If you use a geometry shader, you must enable the `geometry_shader` feature on the device.
    pub geometry_shader: Option<Gs>,

    /// Describes the subsection of the framebuffer attachments where the scene will be drawn.
    /// You can use one or multiple viewports, but using multiple viewports is only relevant with
    /// a geometry shader.
    pub viewport: ViewportsState,

    /// Describes how the implementation determines which pixels are covered by the shape.
    pub raster: Rasterization,

    // TODO: document
    pub multisample: Multisample,

    /// The entry point of the fragment shader that will be run on the pixels.
    pub fragment_shader: Fs,

    /// Describes how the implementation should perform the depth and stencil tests.
    pub depth_stencil: DepthStencil,

    /// Describes how the implementation should merge the color output of the fragment shader with
    /// the existing value in the attachments.
    pub blend: Blend,

    /// Which subpass of which render pass this pipeline will run on. It is an error to run a
    /// graphics pipeline on a different subpass.
    pub render_pass: Subpass<Rp>,
}

/// Additional parameters if you use tessellation.
#[derive(Copy, Clone)]
pub(crate) struct GraphicsPipelineParamsTess<Tcs, Tes> {
    /// The entry point of the tessellation control shader.
    pub tessellation_control_shader: Tcs,
    /// The entry point of the tessellation evaluation shader.
    pub tessellation_evaluation_shader: Tes,
}

/// Defines how the implementation should perform a draw operation.
///
/// This object contains the shaders and the various fixed states that describe how the
/// implementation should perform the various operations needed by a draw command.
pub struct GraphicsPipeline<VertexDefinition, Layout, RenderP> {
    inner: Inner,
    layout: Layout,

    render_pass: RenderP,
    render_pass_subpass: u32,

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

struct Inner {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
}

impl GraphicsPipeline<(), (), ()> {
    /// Starts the building process of a graphics pipeline. Returns a builder object that you can
    /// fill with the various parameters.
    pub fn start<'a>()
        -> GraphicsPipelineBuilder<SingleBufferDefinition<()>,
                                   EmptyEntryPointDummy,
                                   EmptyEntryPointDummy,
                                   EmptyEntryPointDummy,
                                   EmptyEntryPointDummy,
                                   EmptyEntryPointDummy,
                                   ()>
    {
        GraphicsPipelineBuilder::new()
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp> {
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
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp>
    where L: PipelineLayoutAbstract
{
    /// Returns the pipeline layout used in the constructor.
    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp>
    where Rp: RenderPassDesc
{
    /// Returns the pass used in the constructor.
    #[inline]
    pub fn subpass(&self) -> Subpass<&Rp> {
        Subpass::from(&self.render_pass, self.render_pass_subpass).unwrap()
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp> {
    /// Returns the render pass used in the constructor.
    #[inline]
    pub fn render_pass(&self) -> &Rp {
        &self.render_pass
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

unsafe impl<Mv, L, Rp> PipelineLayoutAbstract for GraphicsPipeline<Mv, L, Rp>
    where L: PipelineLayoutAbstract
{
    #[inline]
    fn sys(&self) -> PipelineLayoutSys {
        self.layout.sys()
    }

    #[inline]
    fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
        self.layout.descriptor_set_layout(index)
    }
}

unsafe impl<Mv, L, Rp> PipelineLayoutDesc for GraphicsPipeline<Mv, L, Rp>
    where L: PipelineLayoutDesc
{
    #[inline]
    fn num_sets(&self) -> usize {
        self.layout.num_sets()
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        self.layout.num_bindings_in_set(set)
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        self.layout.descriptor(set, binding)
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        self.layout.num_push_constants_ranges()
    }

    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        self.layout.push_constants_range(num)
    }
}

unsafe impl<Mv, L, Rp> DeviceOwned for GraphicsPipeline<Mv, L, Rp> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }
}

impl<Mv, L, Rp> fmt::Debug for GraphicsPipeline<Mv, L, Rp> {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan graphics pipeline {:?}>", self.inner.pipeline)
    }
}

unsafe impl<Mv, L, Rp> RenderPassAbstract for GraphicsPipeline<Mv, L, Rp>
    where Rp: RenderPassAbstract
{
    #[inline]
    fn inner(&self) -> RenderPassSys {
        self.render_pass.inner()
    }
}

unsafe impl<Mv, L, Rp> RenderPassDesc for GraphicsPipeline<Mv, L, Rp>
    where Rp: RenderPassDesc
{
    #[inline]
    fn num_attachments(&self) -> usize {
        self.render_pass.num_attachments()
    }

    #[inline]
    fn attachment_desc(&self, num: usize) -> Option<LayoutAttachmentDescription> {
        self.render_pass.attachment_desc(num)
    }

    #[inline]
    fn num_subpasses(&self) -> usize {
        self.render_pass.num_subpasses()
    }

    #[inline]
    fn subpass_desc(&self, num: usize) -> Option<LayoutPassDescription> {
        self.render_pass.subpass_desc(num)
    }

    #[inline]
    fn num_dependencies(&self) -> usize {
        self.render_pass.num_dependencies()
    }

    #[inline]
    fn dependency_desc(&self, num: usize) -> Option<LayoutPassDependencyDescription> {
        self.render_pass.dependency_desc(num)
    }
}

unsafe impl<C, Mv, L, Rp> RenderPassDescClearValues<C> for GraphicsPipeline<Mv, L, Rp>
    where Rp: RenderPassDescClearValues<C>
{
    #[inline]
    fn convert_clear_values(&self, vals: C) -> Box<Iterator<Item = ClearValue>> {
        self.render_pass.convert_clear_values(vals)
    }
}

unsafe impl<Mv, L, Rp> VulkanObject for GraphicsPipeline<Mv, L, Rp> {
    type Object = vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.inner.pipeline
    }
}

impl Drop for Inner {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipeline(self.device.internal_object(), self.pipeline, ptr::null());
        }
    }
}

/// Trait implemented on objects that reference a graphics pipeline. Can be made into a trait
/// object.
pub unsafe trait GraphicsPipelineAbstract: PipelineLayoutAbstract + RenderPassAbstract + VertexSource<Vec<Arc<BufferAccess + Send + Sync>>> {
    /// Returns an opaque object that represents the inside of the graphics pipeline.
    fn inner(&self) -> GraphicsPipelineSys;

    /// Returns the index of the subpass this graphics pipeline is rendering to.
    fn subpass_index(&self) -> u32;

    /// Returns the subpass this graphics pipeline is rendering to.
    #[inline]
    fn subpass(self) -> Subpass<Self> where Self: Sized {
        let index = self.subpass_index();
        Subpass::from(self, index).expect("Wrong subpass index in GraphicsPipelineAbstract::subpass")
    }

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

unsafe impl<Mv, L, Rp> GraphicsPipelineAbstract for GraphicsPipeline<Mv, L, Rp>
    where L: PipelineLayoutAbstract,
          Rp: RenderPassAbstract,
          Mv: VertexSource<Vec<Arc<BufferAccess + Send + Sync>>>
{
    #[inline]
    fn inner(&self) -> GraphicsPipelineSys {
        GraphicsPipelineSys(self.inner.pipeline, PhantomData)
    }

    #[inline]
    fn subpass_index(&self) -> u32 {
        self.render_pass_subpass
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
    where T: SafeDeref,
          T::Target: GraphicsPipelineAbstract
{
    #[inline]
    fn inner(&self) -> GraphicsPipelineSys {
        GraphicsPipelineAbstract::inner(&**self)
    }

    #[inline]
    fn subpass_index(&self) -> u32 {
        (**self).subpass_index()
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

/// Opaque object that represents the inside of the graphics pipeline.
#[derive(Debug, Copy, Clone)]
pub struct GraphicsPipelineSys<'a>(vk::Pipeline, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for GraphicsPipelineSys<'a> {
    type Object = vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.0
    }
}

unsafe impl<Mv, L, Rp, I> VertexDefinition<I> for GraphicsPipeline<Mv, L, Rp>
    where Mv: VertexDefinition<I>
{
    type BuffersIter = <Mv as VertexDefinition<I>>::BuffersIter;
    type AttribsIter = <Mv as VertexDefinition<I>>::AttribsIter;

    #[inline]
    fn definition(
        &self, interface: &I)
        -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        self.vertex_definition.definition(interface)
    }
}

unsafe impl<Mv, L, Rp, S> VertexSource<S> for GraphicsPipeline<Mv, L, Rp>
    where Mv: VertexSource<S>
{
    #[inline]
    fn decode(&self, s: S) -> (Vec<Box<BufferAccess + Send + Sync>>, usize, usize) {
        self.vertex_definition.decode(s)
    }
}
