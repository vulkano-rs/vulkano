// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::iter;
use std::mem;
use std::slice;
use std::sync::Arc;

use buffer::BufferAccess;
use buffer::TypedBufferAccess;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferExecError;
use command_buffer::DrawIndirectCommand;
use command_buffer::DynamicState;
use command_buffer::StateCacher;
use command_buffer::StateCacherOutcome;
use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolBuilderAlloc;
use command_buffer::pool::standard::StandardCommandPool;
use command_buffer::pool::standard::StandardCommandPoolAlloc;
use command_buffer::pool::standard::StandardCommandPoolBuilder;
use command_buffer::synced::SyncCommandBuffer;
use command_buffer::synced::SyncCommandBufferBuilder;
use command_buffer::synced::SyncCommandBufferBuilderError;
use command_buffer::synced::SyncCommandBufferBuilderBindVertexBuffer;
use command_buffer::sys::Flags;
use command_buffer::sys::Kind;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilderBufferImageCopy;
use command_buffer::sys::UnsafeCommandBufferBuilderImageAspect;
use command_buffer::validity;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use framebuffer::FramebufferAbstract;
use framebuffer::RenderPassDescClearValues;
use framebuffer::RenderPassAbstract;
use image::ImageLayout;
use image::ImageAccess;
use instance::QueueFamily;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use pipeline::input_assembly::Index;
use pipeline::vertex::VertexSource;
use sync::AccessCheckError;
use sync::AccessFlagBits;
use sync::PipelineStages;
use sync::GpuFuture;
use OomError;

///
///
/// Note that command buffers allocated from the default command pool (`Arc<StandardCommandPool>`)
/// don't implement the `Send` and `Sync` traits. If you use this pool, then the
/// `AutoCommandBufferBuilder` will not implement `Send` and `Sync` either. Once a command buffer
/// is built, however, it *does* implement `Send` and `Sync`.
///
pub struct AutoCommandBufferBuilder<P = StandardCommandPoolBuilder> {
    inner: SyncCommandBufferBuilder<P>,
    state_cacher: StateCacher,
}

impl AutoCommandBufferBuilder<StandardCommandPoolBuilder> {
    pub fn new(device: Arc<Device>, queue_family: QueueFamily)
               -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
    {
        unsafe {
            let pool = Device::standard_command_pool(&device, queue_family);
            let inner = SyncCommandBufferBuilder::new(&pool, Kind::primary(), Flags::None);
            let state_cacher = StateCacher::new();

            Ok(AutoCommandBufferBuilder {
                inner: inner?,
                state_cacher: state_cacher,
            })
        }
    }
}

impl<P> AutoCommandBufferBuilder<P> {
    /// Builds the command buffer.
    #[inline]
    pub fn build(self) -> Result<AutoCommandBuffer<P::Alloc>, OomError>
        where P: CommandPoolBuilderAlloc
    {
        // TODO: error if we're inside a render pass
        Ok(AutoCommandBuffer {
            inner: self.inner.build()?
        })
    }

    /// Adds a command that enters a render pass.
    ///
    /// If `secondary` is true, then you will only be able to add secondary command buffers while
    /// you're inside the first subpass of the render pass. If `secondary` is false, you will only
    /// be able to add inline draw commands and not secondary command buffers.
    ///
    /// You must call this before you can add draw commands.
    #[inline]
    pub fn begin_render_pass<F, C>(mut self, framebuffer: F, secondary: bool,
                                   clear_values: C)
                                   -> Result<Self, AutoCommandBufferBuilderContextError>
        where F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static
    {
        unsafe {
            let clear_values = framebuffer.convert_clear_values(clear_values);
            let clear_values = clear_values.collect::<Vec<_>>().into_iter();        // TODO: necessary for Send + Sync ; needs an API rework of convert_clear_values
            self.inner.begin_render_pass(framebuffer, secondary, clear_values);
            Ok(self)
        }
    }

    /// Adds a command that copies from a buffer to another.
    ///
    /// This command will copy from the source to the destination. If their size is not equal, then
    /// the amount of data copied is equal to the smallest of the two.
    #[inline]
    pub fn copy_buffer<S, D>(mut self, src: S, dest: D) -> Result<Self, validity::CheckCopyBufferError>
        where S: BufferAccess + Send + Sync + 'static,
              D: BufferAccess + Send + Sync + 'static,
    {
        unsafe {
            // TODO: check that we're not in a render pass

            validity::check_copy_buffer(self.device(), &src, &dest)?;
            let size = src.size();
            self.inner.copy_buffer(src, dest, iter::once((0, 0, size)));
            Ok(self)
        }
    }

    /// Adds a command that copies from a buffer to an image.
    pub fn copy_buffer_to_image<S, D>(mut self, src: S, dest: D)
                                      -> Result<Self, AutoCommandBufferBuilderContextError>
        where S: BufferAccess + Send + Sync + 'static,
              D: ImageAccess + Send + Sync + 'static,
    {
        let dims = dest.dimensions().width_height_depth();
        self.copy_buffer_to_image_dimensions(src, dest, [0, 0, 0], dims, 0, 1, 0)
    }

    /// Adds a command that copies from a buffer to an image.
    pub fn copy_buffer_to_image_dimensions<S, D>(mut self, src: S, dest: D, offset: [u32; 3],
                                                 size: [u32; 3], first_layer: u32, num_layers: u32,
                                                 mipmap: u32)
                                                 -> Result<Self, AutoCommandBufferBuilderContextError>
        where S: BufferAccess + Send + Sync + 'static,
              D: ImageAccess + Send + Sync + 'static,
    {
        unsafe {
            // TODO: check that we're not in a render pass
            // TODO: check validity
            // TODO: hastily implemented

            let copy = UnsafeCommandBufferBuilderBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_aspect: if dest.has_color() {
                    UnsafeCommandBufferBuilderImageAspect { color: true, depth: false, stencil: false }
                } else {
                    unimplemented!()
                },
                image_mip_level: mipmap,
                image_base_array_layer: first_layer,
                image_layer_count: num_layers,
                image_offset: [offset[0] as i32, offset[1] as i32, offset[2] as i32],
                image_extent: size,
            };

            let size = src.size();
            self.inner.copy_buffer_to_image(src, dest, ImageLayout::TransferDstOptimal,     // TODO: let choose layout
                                            iter::once(copy));
            Ok(self)
        }
    }

    #[inline]
    pub fn dispatch<Cp, S, Pc>(mut self, dimensions: [u32; 3], pipeline: Cp, sets: S, constants: Pc)
                               -> Result<Self, AutoCommandBufferBuilderContextError>
        where Cp: ComputePipelineAbstract + Send + Sync + 'static + Clone,    // TODO: meh for Clone
              S: DescriptorSetsCollection,
    {
        unsafe {
            // TODO: missing checks

            if let StateCacherOutcome::NeedChange = self.state_cacher.bind_compute_pipeline(&pipeline) {
                self.inner.bind_pipeline_compute(pipeline.clone());
            }

            push_constants(&mut self.inner, pipeline.clone(), constants);
            descriptor_sets(&mut self.inner, false, pipeline.clone(), sets);

            self.inner.dispatch(dimensions);
            Ok(self)
        }
    }

    #[inline]
    pub fn draw<V, Gp, S, Pc>(mut self, pipeline: Gp, dynamic: DynamicState, vertices: V, sets: S,
                              constants: Pc) -> Result<Self, AutoCommandBufferBuilderContextError>
        where Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone,    // TODO: meh for Clone
              S: DescriptorSetsCollection,
    {
        unsafe {
            // TODO: missing checks

            if let StateCacherOutcome::NeedChange = self.state_cacher.bind_graphics_pipeline(&pipeline) {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            push_constants(&mut self.inner, pipeline.clone(), constants);
            set_state(&mut self.inner, dynamic);
            descriptor_sets(&mut self.inner, true, pipeline.clone(), sets);
            let (vertex_count, instance_count) = vertex_buffers(&mut self.inner, &pipeline,
                                                                vertices);

            self.inner.draw(vertex_count as u32, instance_count as u32, 0, 0);
            Ok(self)
        }
    }

    #[inline]
    pub fn draw_indexed<V, Gp, S, Pc, Ib, I>(mut self, pipeline: Gp, dynamic: DynamicState,
                                             vertices: V, index_buffer: Ib, sets: S,
                                             constants: Pc) -> Result<Self, AutoCommandBufferBuilderContextError>
        where Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone,    // TODO: meh for Clone
              S: DescriptorSetsCollection,
              Ib: BufferAccess + TypedBufferAccess<Content = [I]> + Send + Sync + 'static,
              I: Index + 'static,
    {
        unsafe {
            // TODO: missing checks

            let index_count = index_buffer.len();

            if let StateCacherOutcome::NeedChange = self.state_cacher.bind_graphics_pipeline(&pipeline) {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            self.inner.bind_index_buffer(index_buffer, I::ty());
            push_constants(&mut self.inner, pipeline.clone(), constants);
            set_state(&mut self.inner, dynamic);
            descriptor_sets(&mut self.inner, true, pipeline.clone(), sets);
            vertex_buffers(&mut self.inner, &pipeline, vertices);
            // TODO: how to handle an index out of range of the vertex buffers?

            self.inner.draw_indexed(index_count as u32, 1, 0, 0, 0);
            Ok(self)
        }
    }

    #[inline]
    pub fn draw_indirect<V, Gp, S, Pc, Ib>(mut self, pipeline: Gp, dynamic: DynamicState,
                                           vertices: V, indirect_buffer: Ib, sets: S,
                                           constants: Pc) -> Result<Self, AutoCommandBufferBuilderContextError>
        where Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone,    // TODO: meh for Clone
              S: DescriptorSetsCollection,
              Ib: BufferAccess + TypedBufferAccess<Content = [DrawIndirectCommand]> + Send + Sync + 'static,
    {
        unsafe {
            // TODO: missing checks

            let draw_count = indirect_buffer.len() as u32;

            if let StateCacherOutcome::NeedChange = self.state_cacher.bind_graphics_pipeline(&pipeline) {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            push_constants(&mut self.inner, pipeline.clone(), constants);
            set_state(&mut self.inner, dynamic);
            descriptor_sets(&mut self.inner, true, pipeline.clone(), sets);
            vertex_buffers(&mut self.inner, &pipeline, vertices);

            self.inner.draw_indirect(indirect_buffer, draw_count,
                                     mem::size_of::<DrawIndirectCommand>() as u32);
            Ok(self)
        }
    }

    /// Adds a command that ends the current render pass.
    ///
    /// This must be called after you went through all the subpasses and before you can build
    /// the command buffer or add further commands.
    #[inline]
    pub fn end_render_pass(mut self) -> Result<Self, AutoCommandBufferBuilderContextError> {
        unsafe {
            // TODO: check
            self.inner.end_render_pass();
            Ok(self)
        }
    }

    /// Adds a command that writes the content of a buffer.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatidely written through the entire buffer.
    ///
    /// > **Note**: This function is technically safe because buffers can only contain integers or
    /// > floating point numbers, which are always valid whatever their memory representation is.
    /// > But unless your buffer actually contains only 32-bits integers, you are encouraged to use
    /// > this function only for zeroing the content of a buffer by passing `0` for the data.
    // TODO: not safe because of signalling NaNs
    #[inline]
    pub fn fill_buffer<B>(mut self, buffer: B, data: u32) -> Result<Self, validity::CheckFillBufferError>
        where B: BufferAccess + Send + Sync + 'static,
    {
        unsafe {
            // TODO: check that we're not in a render pass
            validity::check_fill_buffer(self.device(), &buffer)?;
            self.inner.fill_buffer(buffer, data);
            Ok(self)
        }
    }

    /// Adds a command that jumps to the next subpass of the current render pass.
    #[inline]
    pub fn next_subpass(mut self, secondary: bool)
                        -> Result<Self, AutoCommandBufferBuilderContextError>
    {
        unsafe {
            // TODO: check
            self.inner.next_subpass(secondary);
            Ok(self)
        }
    }

    /// Adds a command that writes data to a buffer.
    ///
    /// If `data` is larger than the buffer, only the part of `data` that fits is written. If the
    /// buffer is larger than `data`, only the start of the buffer is written.
    #[inline]
    pub fn update_buffer<B, D>(mut self, buffer: B, data: D)
                               -> Result<Self, validity::CheckUpdateBufferError>
        where B: BufferAccess + Send + Sync + 'static,
              D: Send + Sync + 'static
    {
        unsafe {
            // TODO: check that we're not in a render pass
            validity::check_update_buffer(self.device(), &buffer, &data)?;

            let size_of_data = mem::size_of_val(&data);
            if buffer.size() > size_of_data {
                self.inner.update_buffer(buffer, data);
            } else {
                unimplemented!()        // TODO:
                //self.inner.update_buffer(buffer.slice(0 .. size_of_data), data);
            }

            Ok(self)
        }
    }
}

unsafe impl<P> DeviceOwned for AutoCommandBufferBuilder<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

// Shortcut function to set the push constants.
unsafe fn push_constants<P, Pl, Pc>(dest: &mut SyncCommandBufferBuilder<P>, pipeline: Pl,
                                    push_constants: Pc)
    where Pl: PipelineLayoutAbstract + Send + Sync + Clone + 'static
{
    for num_range in 0 .. pipeline.num_push_constants_ranges() {
        let range = match pipeline.push_constants_range(num_range) {
            Some(r) => r,
            None => continue
        };

        debug_assert_eq!(range.offset % 4, 0);
        debug_assert_eq!(range.size % 4, 0);

        let data = slice::from_raw_parts((&push_constants as *const Pc as *const u8)
                                            .offset(range.offset as isize),
                                         range.size as usize);

        dest.push_constants::<_, [u8]>(pipeline.clone(), range.stages,
                                       range.offset as u32, range.size as u32,
                                       data);
    }
}

// Shortcut function to change the state of the pipeline.
unsafe fn set_state<P>(dest: &mut SyncCommandBufferBuilder<P>, dynamic: DynamicState) {
    if let Some(line_width) = dynamic.line_width {
        dest.set_line_width(line_width);
    }

    if let Some(ref viewports) = dynamic.viewports {
        dest.set_viewport(0, viewports.iter().cloned().collect::<Vec<_>>().into_iter());        // TODO: don't collect
    }

    if let Some(ref scissors) = dynamic.scissors {
        dest.set_scissor(0, scissors.iter().cloned().collect::<Vec<_>>().into_iter());      // TODO: don't collect
    }
}

// Shortcut function to bind vertex buffers.
unsafe fn vertex_buffers<P, Gp, V>(dest: &mut SyncCommandBufferBuilder<P>, pipeline: &Gp,
                                   vertices: V) -> (u32, u32)
    where Gp: VertexSource<V>,
{
    let (vertex_buffers, vertex_count, instance_count) = pipeline.decode(vertices);

    let mut binder = dest.bind_vertex_buffers();
    for vb in vertex_buffers {
        binder.add(vb);
    }
    binder.submit(0);

    (vertex_count as u32, instance_count as u32)
}

unsafe fn descriptor_sets<P, Pl, S>(dest: &mut SyncCommandBufferBuilder<P>, gfx: bool,
                                    pipeline: Pl, sets: S)
    where Pl: PipelineLayoutAbstract + Send + Sync + Clone + 'static,
          S: DescriptorSetsCollection
{
    let mut sets_binder = dest.bind_descriptor_sets();

    for set in sets.into_vec() {
        sets_binder.add(set);
    }

    sets_binder.submit(gfx, pipeline.clone(), 0, iter::empty());
}

pub struct AutoCommandBuffer<P = StandardCommandPoolAlloc> {
    inner: SyncCommandBuffer<P>,
}

unsafe impl<P> CommandBuffer for AutoCommandBuffer<P> {
    type PoolAlloc = P;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<P> {
        self.inner.inner()
    }

    #[inline]
    fn prepare_submit(&self, future: &GpuFuture, queue: &Queue) -> Result<(), CommandBufferExecError> {
        self.inner.prepare_submit(future, queue)
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        self.inner.check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        self.inner.check_image_access(image, layout, exclusive, queue)
    }
}

unsafe impl<P> DeviceOwned for AutoCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

macro_rules! err_gen {
    ($name:ident) => (
        pub enum $name {
            SyncCommandBufferBuilderError(SyncCommandBufferBuilderError),
        }
    );
}

err_gen!(Foo);

#[derive(Debug, Copy, Clone)]
pub enum AutoCommandBufferBuilderContextError {
    Forbidden,
}

impl error::Error for AutoCommandBufferBuilderContextError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            AutoCommandBufferBuilderContextError::Forbidden => {
                "operation forbidden inside or outside of a render pass"
            },
        }
    }
}

impl fmt::Display for AutoCommandBufferBuilderContextError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
