// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::Range;
use std::sync::Arc;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferSlice;
use buffer::TypedBuffer;
use command_buffer::DrawIndirectCommand;
use command_buffer::inner::InnerCommandBufferBuilder;
use command_buffer::inner::InnerCommandBuffer;
use command_buffer::inner::Submission;
use command_buffer::inner::submit as inner_submit;
use command_buffer::pool::CommandPool;
use command_buffer::pool::StandardCommandPool;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::PipelineLayout;
use device::Device;
use device::Queue;
use framebuffer::Framebuffer;
use framebuffer::UnsafeRenderPass;
use framebuffer::RenderPassCompatible;
use framebuffer::RenderPass;
use framebuffer::RenderPassDesc;
use framebuffer::RenderPassClearValues;
use framebuffer::Subpass;
use image::traits::Image;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::Source as VertexSource;
use pipeline::viewport::Viewport;
use pipeline::viewport::Scissor;

use OomError;

/// A prototype of a primary command buffer.
///
/// # Usage
///
/// ```ignore   // TODO: change that
/// let commands_buffer =
///     PrimaryCommandBufferBuilder::new(&device)
///         .copy_memory(..., ...)
///         .draw(...)
///         .build();
/// 
/// ```
///
pub struct PrimaryCommandBufferBuilder<P = Arc<StandardCommandPool>> where P: CommandPool {
    inner: InnerCommandBufferBuilder<P>,
}

impl PrimaryCommandBufferBuilder<Arc<StandardCommandPool>> {
    /// Builds a new primary command buffer and start recording commands in it.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    /// - Panicks if the device and queue family do not belong to the same physical device.
    ///
    #[inline]
    pub fn new(device: &Arc<Device>, queue_family: QueueFamily)
               -> PrimaryCommandBufferBuilder<Arc<StandardCommandPool>>
    {
        PrimaryCommandBufferBuilder::raw(Device::standard_command_pool(device, queue_family)).unwrap()
    }
}

impl<P> PrimaryCommandBufferBuilder<P> where P: CommandPool {
    /// See the docs of new().
    #[inline]
    pub fn raw(pool: P) -> Result<PrimaryCommandBufferBuilder<P>, OomError> {
        let inner = try!(InnerCommandBufferBuilder::new::<UnsafeRenderPass>(pool, false, None, None));
        Ok(PrimaryCommandBufferBuilder { inner: inner })
    }

    /// Writes data to a buffer.
    ///
    /// The data is stored inside the command buffer and written to the given buffer slice.
    /// This function is intended to be used for small amounts of data (only 64kB is allowed). if
    /// you want to transfer large amounts of data, use copies between buffers.
    ///
    /// # Panic
    ///
    /// - Panicks if the size of `data` is not the same as the size of the buffer slice.
    /// - Panicks if the size of `data` is superior to 65536 bytes.
    /// - Panicks if the offset or size is not a multiple of 4.
    /// - Panicks if the buffer wasn't created with the right usage.
    /// - Panicks if the queue family doesn't support transfer operations.
    ///
    #[inline]
    pub fn update_buffer<'a, B, T, Bb>(self, buffer: B, data: &T) -> PrimaryCommandBufferBuilder<P>
        where B: Into<BufferSlice<'a, T, Bb>>, Bb: Buffer + 'static, T: Clone + 'static + Send + Sync
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.update_buffer(buffer, data)
            }
        }
    }

    /// Fills a buffer with data.
    ///
    /// The data is repeated until it fills the range from `offset` to `offset + size`.
    /// Since the data is a u32, the offset and the size must be multiples of 4.
    ///
    /// # Panic
    ///
    /// - Panicks if `offset + data` is superior to the size of the buffer.
    /// - Panicks if the offset or size is not a multiple of 4.
    /// - Panicks if the buffer wasn't created with the right usage.
    /// - Panicks if the queue family doesn't support transfer operations.
    ///
    /// # Safety
    ///
    /// - Type safety is not enforced by the API.
    ///
    pub unsafe fn fill_buffer<B>(self, buffer: &Arc<B>, offset: usize,
                                 size: usize, data: u32) -> PrimaryCommandBufferBuilder<P>
        where B: Buffer + 'static
    {
        PrimaryCommandBufferBuilder {
            inner: self.inner.fill_buffer(buffer, offset, size, data)
        }
    }

    pub fn copy_buffer<T: ?Sized + 'static, Bs, Bd>(self, source: &Arc<Bs>, destination: &Arc<Bd>)
                                                    -> PrimaryCommandBufferBuilder<P>
        where Bs: TypedBuffer<Content = T> + 'static, Bd: TypedBuffer<Content = T> + 'static
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.copy_buffer(source, destination),
            }
        }
    }

    pub fn copy_buffer_to_color_image<'a, Pi, S, Img, Sb>(self, source: S, destination: &Arc<Img>, mip_level: u32, array_layers_range: Range<u32>,
                                                         offset: [u32; 3], extent: [u32; 3])
                                                    -> PrimaryCommandBufferBuilder<P>
        where S: Into<BufferSlice<'a, [Pi], Sb>>, Sb: Buffer + 'static,
              Img: ImageContent<Pi> + 'static
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.copy_buffer_to_color_image(source, destination, mip_level,
                                                             array_layers_range, offset, extent),
            }
        }
    }

    pub fn copy_color_image_to_buffer<'a, Pi, S, Img, Sb>(self, dest: S, destination: &Arc<Img>, mip_level: u32, array_layers_range: Range<u32>,
                                                         offset: [u32; 3], extent: [u32; 3])
                                                    -> PrimaryCommandBufferBuilder<P>
        where S: Into<BufferSlice<'a, [Pi], Sb>>, Sb: Buffer + 'static,
              Img: ImageContent<Pi> + 'static
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.copy_color_image_to_buffer(dest, destination, mip_level,
                                                             array_layers_range, offset, extent),
            }
        }
    }

    pub fn blit<Si, Di>(self, source: &Arc<Si>, source_mip_level: u32,
                        source_array_layers: Range<u32>, src_coords: [Range<i32>; 3],
                        destination: &Arc<Di>, dest_mip_level: u32,
                        dest_array_layers: Range<u32>, dest_coords: [Range<i32>; 3])
                        -> PrimaryCommandBufferBuilder<P>
        where Si: Image + 'static, Di: Image + 'static
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.blit(source, source_mip_level, source_array_layers, src_coords,
                                       destination, dest_mip_level, dest_array_layers, dest_coords),
            }
        }
    }

    ///
    /// Note that compressed formats are not supported.
    pub fn clear_color_image<'a, I, V>(self, image: &Arc<I>, color: V)
                                       -> PrimaryCommandBufferBuilder<P>
        where I: ImageClearValue<V> + 'static
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.clear_color_image(image, color),
            }
        }
    }

    /// Executes secondary compute command buffers within this primary command buffer.
    #[inline]
    pub fn execute_commands<S>(self, cb: &Arc<SecondaryComputeCommandBuffer<S>>)
                               -> PrimaryCommandBufferBuilder<P>
        where S: CommandPool + 'static,
              S::Finished: Send + Sync + 'static,
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.execute_commands(cb.clone() as Arc<_>, &cb.inner)
            }
        }
    }

    /// Executes a compute pipeline.
    #[inline]
    pub fn dispatch<Pl, L, Pc>(self, pipeline: &Arc<ComputePipeline<Pl>>, sets: L,
                           dimensions: [u32; 3], push_constants: &Pc) -> PrimaryCommandBufferBuilder<P>
        where L: DescriptorSetsCollection + Send + Sync,
              Pl: 'static + PipelineLayout + Send + Sync,
              Pc: 'static + Clone + Send + Sync
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.dispatch(pipeline, sets, dimensions, push_constants)
            }
        }
    }

    /// Start drawing on a framebuffer.
    //
    /// This function returns an object that can be used to submit draw commands on the first
    /// subpass of the renderpass.
    ///
    /// # Panic
    ///
    /// - Panicks if the framebuffer is not compatible with the renderpass.
    ///
    // FIXME: rest of the parameters (render area and clear attachment values)
    #[inline]
    pub fn draw_inline<R, F, C>(self, renderpass: &Arc<R>,
                                framebuffer: &Arc<Framebuffer<F>>, clear_values: C)
                                -> PrimaryCommandBufferBuilderInlineDraw<P>
        where F: RenderPass + RenderPassDesc + RenderPassClearValues<C> + 'static,
              R: RenderPass + RenderPassDesc + 'static
    {
        // FIXME: check for compatibility

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let clear_values = framebuffer.render_pass().convert_clear_values(clear_values)
                                      .collect::<SmallVec<[_; 16]>>();

        unsafe {
            let inner = self.inner.begin_renderpass(renderpass, framebuffer, false, &clear_values);

            PrimaryCommandBufferBuilderInlineDraw {
                inner: inner,
                current_subpass: 0,
                num_subpasses: framebuffer.render_pass().num_subpasses(),
            }
        }
    }

    /// Start drawing on a framebuffer.
    //
    /// This function returns an object that can be used to submit secondary graphics command
    /// buffers that will operate on the first subpass of the renderpass.
    ///
    /// # Panic
    ///
    /// - Panicks if the framebuffer is not compatible with the renderpass.
    ///
    // FIXME: rest of the parameters (render area and clear attachment values)
    #[inline]
    pub fn draw_secondary<R, F, C>(self, renderpass: &Arc<R>,
                                   framebuffer: &Arc<Framebuffer<F>>, clear_values: C)
                                   -> PrimaryCommandBufferBuilderSecondaryDraw<P>
        where F: RenderPass + RenderPassDesc + RenderPassClearValues<C> + 'static,
              R: RenderPass + RenderPassDesc + 'static
    {
        // FIXME: check for compatibility

        let clear_values = framebuffer.render_pass().convert_clear_values(clear_values)
                                      .collect::<SmallVec<[_; 16]>>();

        unsafe {
            let inner = self.inner.begin_renderpass(renderpass, framebuffer, true, &clear_values);

            PrimaryCommandBufferBuilderSecondaryDraw {
                inner: inner,
                current_subpass: 0,
                num_subpasses: framebuffer.render_pass().num_subpasses(),
            }
        }
    }

    /// See the docs of build().
    #[inline]
    pub fn build_raw(self) -> Result<PrimaryCommandBuffer<P>, OomError> {
        let inner = try!(self.inner.build());
        Ok(PrimaryCommandBuffer { inner: inner })
    }

    /// Finish recording commands and build the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    ///
    #[inline]
    pub fn build(self) -> Arc<PrimaryCommandBuffer<P>> {
        Arc::new(self.build_raw().unwrap())
    }
}

/// Object that you obtain when calling `draw_inline` or `next_subpass_inline`.
pub struct PrimaryCommandBufferBuilderInlineDraw<P = Arc<StandardCommandPool>>
    where P: CommandPool
{
    inner: InnerCommandBufferBuilder<P>,
    current_subpass: u32,
    num_subpasses: u32,
}

impl<P> PrimaryCommandBufferBuilderInlineDraw<P> where P: CommandPool {
    /// Calls `vkCmdDraw`.
    // FIXME: push constants
    pub fn draw<V, L, Pv, Pl, Rp, Pc>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                              vertices: V, dynamic: &DynamicState, sets: L, push_constants: &Pc)
                              -> PrimaryCommandBufferBuilderInlineDraw<P>
        where Pv: VertexSource<V> + 'static, Pl: PipelineLayout + 'static + Send + Sync, Rp: 'static + Send + Sync,
              L: DescriptorSetsCollection + Send + Sync, Pc: 'static + Clone + Send + Sync
    {
        // FIXME: check subpass

        unsafe {
            PrimaryCommandBufferBuilderInlineDraw {
                inner: self.inner.draw(pipeline, vertices, dynamic, sets, push_constants),
                num_subpasses: self.num_subpasses,
                current_subpass: self.current_subpass,
            }
        }
    }

    /// Calls `vkCmdDrawIndexed`.
    pub fn draw_indexed<'a, V, L, Pv, Pl, Rp, I, Ib, Ibb, Pc>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                                              vertices: V, indices: Ib, dynamic: &DynamicState,
                                              sets: L, push_constants: &Pc) -> PrimaryCommandBufferBuilderInlineDraw<P>
        where Pv: 'static + VertexSource<V> + Send + Sync, Pl: 'static + PipelineLayout + Send + Sync, Rp: 'static + Send + Sync,
              Ib: Into<BufferSlice<'a, [I], Ibb>>, I: 'static + Index, Ibb: Buffer + 'static + Send + Sync,
              L: DescriptorSetsCollection + Send + Sync, Pc: 'static + Clone + Send + Sync
    {
        // FIXME: check subpass

        unsafe {
            PrimaryCommandBufferBuilderInlineDraw {
                inner: self.inner.draw_indexed(pipeline, vertices, indices, dynamic, sets, push_constants),
                num_subpasses: self.num_subpasses,
                current_subpass: self.current_subpass,
            }
        }
    }

    /// Switches to the next subpass of the current renderpass.
    ///
    /// This function is similar to `draw_inline` on the builder.
    ///
    /// # Panic
    ///
    /// - Panicks if no more subpasses remain.
    ///
    #[inline]
    pub fn next_subpass_inline(self) -> PrimaryCommandBufferBuilderInlineDraw<P> {
        assert!(self.current_subpass + 1 < self.num_subpasses);

        unsafe {
            let inner = self.inner.next_subpass(false);

            PrimaryCommandBufferBuilderInlineDraw {
                inner: inner,
                num_subpasses: self.num_subpasses,
                current_subpass: self.current_subpass + 1,
            }
        }
    }

    /// Switches to the next subpass of the current renderpass.
    ///
    /// This function is similar to `draw_secondary` on the builder.
    ///
    /// # Panic
    ///
    /// - Panicks if no more subpasses remain.
    ///
    #[inline]
    pub fn next_subpass_secondary(self) -> PrimaryCommandBufferBuilderSecondaryDraw<P> {
        assert!(self.current_subpass + 1 < self.num_subpasses);

        unsafe {
            let inner = self.inner.next_subpass(true);

            PrimaryCommandBufferBuilderSecondaryDraw {
                inner: inner,
                num_subpasses: self.num_subpasses,
                current_subpass: self.current_subpass + 1,
            }
        }
    }

    /// Finish drawing this renderpass and get back the builder.
    ///
    /// # Panic
    ///
    /// - Panicks if not at the last subpass.
    ///
    #[inline]
    pub fn draw_end(self) -> PrimaryCommandBufferBuilder<P> {
        assert!(self.current_subpass + 1 == self.num_subpasses);

        unsafe {
            let inner = self.inner.end_renderpass();
            PrimaryCommandBufferBuilder {
                inner: inner,
            }
        }
    }
}

/// Object that you obtain when calling `draw_secondary` or `next_subpass_secondary`.
pub struct PrimaryCommandBufferBuilderSecondaryDraw<P = Arc<StandardCommandPool>>
    where P: CommandPool
{
    inner: InnerCommandBufferBuilder<P>,
    current_subpass: u32,
    num_subpasses: u32,
}

impl<P> PrimaryCommandBufferBuilderSecondaryDraw<P> where P: CommandPool {
    /// Switches to the next subpass of the current renderpass.
    ///
    /// This function is similar to `draw_inline` on the builder.
    ///
    /// # Panic
    ///
    /// - Panicks if no more subpasses remain.
    ///
    #[inline]
    pub fn next_subpass_inline(self) -> PrimaryCommandBufferBuilderInlineDraw<P> {
        assert!(self.current_subpass + 1 < self.num_subpasses);

        unsafe {
            let inner = self.inner.next_subpass(false);

            PrimaryCommandBufferBuilderInlineDraw {
                inner: inner,
                num_subpasses: self.num_subpasses,
                current_subpass: self.current_subpass + 1,
            }
        }
    }

    /// Switches to the next subpass of the current renderpass.
    ///
    /// This function is similar to `draw_secondary` on the builder.
    ///
    /// # Panic
    ///
    /// - Panicks if no more subpasses remain.
    ///
    #[inline]
    pub fn next_subpass_secondary(self) -> PrimaryCommandBufferBuilderSecondaryDraw<P> {
        assert!(self.current_subpass + 1 < self.num_subpasses);

        unsafe {
            let inner = self.inner.next_subpass(true);

            PrimaryCommandBufferBuilderSecondaryDraw {
                inner: inner,
                num_subpasses: self.num_subpasses,
                current_subpass: self.current_subpass + 1,
            }
        }
    }

    /// Executes secondary graphics command buffers within this primary command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the secondary command buffers wasn't created with a compatible
    ///   renderpass or is using the wrong subpass.
    #[inline]
    pub fn execute_commands<R, Ps>(mut self, cb: &Arc<SecondaryGraphicsCommandBuffer<R, Ps>>)
                                   -> PrimaryCommandBufferBuilderSecondaryDraw<P>
        where R: 'static + Send + Sync,
              Ps: CommandPool + 'static,
              Ps::Finished: Send + Sync + 'static,
    {
        // FIXME: check renderpass, subpass and framebuffer

        unsafe {
            self.inner = self.inner.execute_commands(cb.clone() as Arc<_>, &cb.inner);
            self
        }
    }

    /// Finish drawing this renderpass and get back the builder.
    ///
    /// # Panic
    ///
    /// - Panicks if not at the last subpass.
    ///
    #[inline]
    pub fn draw_end(self) -> PrimaryCommandBufferBuilder<P> {
        assert!(self.current_subpass + 1 == self.num_subpasses);

        unsafe {
            let inner = self.inner.end_renderpass();
            PrimaryCommandBufferBuilder {
                inner: inner,
            }
        }
    }
}

/// Represents a collection of commands to be executed by the GPU.
///
/// A primary command buffer can contain any command.
pub struct PrimaryCommandBuffer<P = Arc<StandardCommandPool>> where P: CommandPool {
    inner: InnerCommandBuffer<P>,
}

/// Submits the command buffer to a queue so that it is executed.
///
/// Fences and semaphores are automatically handled.
///
/// # Panic
///
/// - Panicks if the queue doesn't belong to the device this command buffer was created with.
/// - Panicks if the queue doesn't belong to the family the pool was created with.
///
#[inline]
pub fn submit<P>(cmd: &Arc<PrimaryCommandBuffer<P>>, queue: &Arc<Queue>)
                 -> Result<Arc<Submission>, OomError>
    where P: CommandPool + 'static,
          P::Finished: Send + Sync + 'static
{       // TODO: wrong error type
    inner_submit(&cmd.inner, cmd.clone() as Arc<_>, queue)
}

/// A prototype of a secondary compute command buffer.
pub struct SecondaryGraphicsCommandBufferBuilder<R, P = Arc<StandardCommandPool>>
    where P: CommandPool
{
    inner: InnerCommandBufferBuilder<P>,
    render_pass: Arc<R>,
    render_pass_subpass: u32,
    framebuffer: Option<Arc<Framebuffer<R>>>,
}

impl<R> SecondaryGraphicsCommandBufferBuilder<R, Arc<StandardCommandPool>>
    where R: RenderPass + RenderPassDesc + 'static
{
    /// Builds a new secondary command buffer and start recording commands in it.
    ///
    /// The `framebuffer` parameter is optional and can be used as an optimisation.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    /// - Panicks if the device and queue family do not belong to the same physical device.
    ///
    #[inline]
    pub fn new(device: &Arc<Device>, queue_family: QueueFamily, subpass: Subpass<R>,
               framebuffer: Option<&Arc<Framebuffer<R>>>)
               -> SecondaryGraphicsCommandBufferBuilder<R, Arc<StandardCommandPool>>
        where R: 'static + Send + Sync
    {
        SecondaryGraphicsCommandBufferBuilder::raw(Device::standard_command_pool(device,
                                                   queue_family), subpass, framebuffer).unwrap()
    }
}

impl<R, P> SecondaryGraphicsCommandBufferBuilder<R, P>
    where R: RenderPass + RenderPassDesc + 'static,
          P: CommandPool
{
    /// See the docs of new().
    #[inline]
    pub fn raw(pool: P, subpass: Subpass<R>, framebuffer: Option<&Arc<Framebuffer<R>>>)
               -> Result<SecondaryGraphicsCommandBufferBuilder<R, P>, OomError>
        where R: 'static + Send + Sync
    {
        let inner = try!(InnerCommandBufferBuilder::new(pool, true, Some(subpass), framebuffer.clone()));
        Ok(SecondaryGraphicsCommandBufferBuilder {
            inner: inner,
            render_pass: subpass.render_pass().clone(),
            render_pass_subpass: subpass.index(),
            framebuffer: framebuffer.map(|fb| fb.clone()),
        })
    }

    /// Calls `vkCmdDraw`.
    // FIXME: push constants
    pub fn draw<V, L, Pv, Pl, Rp, Pc>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                              vertices: V, dynamic: &DynamicState, sets: L, push_constants: &Pc)
                              -> SecondaryGraphicsCommandBufferBuilder<R, P>
        where Pv: VertexSource<V> + 'static, Pl: PipelineLayout + 'static + Send + Sync,
              Rp: RenderPass + RenderPassDesc + 'static + Send + Sync, L: DescriptorSetsCollection + Send + Sync,
              R: RenderPassCompatible<Rp>, Pc: 'static + Clone + Send + Sync
    {
        assert!(self.render_pass.is_compatible_with(pipeline.subpass().render_pass()));
        assert_eq!(self.render_pass_subpass, pipeline.subpass().index());

        unsafe {
            SecondaryGraphicsCommandBufferBuilder {
                inner: self.inner.draw(pipeline, vertices, dynamic, sets, push_constants),
                render_pass: self.render_pass,
                render_pass_subpass: self.render_pass_subpass,
                framebuffer: self.framebuffer,
            }
        }
    }

    /// Calls `vkCmdDrawIndexed`.
    pub fn draw_indexed<'a, V, L, Pv, Pl, Rp, I, Ib, Ibb, Pc>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                                              vertices: V, indices: Ib, dynamic: &DynamicState,
                                              sets: L, push_constants: &Pc) -> SecondaryGraphicsCommandBufferBuilder<R, P>
        where Pv: 'static + VertexSource<V>, Pl: 'static + PipelineLayout + Send + Sync,
              Rp: RenderPass + RenderPassDesc + 'static + Send + Sync,
              Ib: Into<BufferSlice<'a, [I], Ibb>>, I: 'static + Index, Ibb: Buffer + 'static,
              L: DescriptorSetsCollection + Send + Sync, Pc: 'static + Clone + Send + Sync
    {
        assert!(self.render_pass.is_compatible_with(pipeline.subpass().render_pass()));
        assert_eq!(self.render_pass_subpass, pipeline.subpass().index());

        unsafe {
            SecondaryGraphicsCommandBufferBuilder {
                inner: self.inner.draw_indexed(pipeline, vertices, indices, dynamic, sets, push_constants),
                render_pass: self.render_pass,
                render_pass_subpass: self.render_pass_subpass,
                framebuffer: self.framebuffer,
            }
        }
    }

    /// Calls `vkCmdDrawIndirect`.
    pub fn draw_indirect<I, V, Pv, Pl, L, Rp, Pc>(self, buffer: &Arc<I>, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                             vertices: V, dynamic: &DynamicState,
                             sets: L, push_constants: &Pc) -> SecondaryGraphicsCommandBufferBuilder<R, P>
        where Pv: 'static + VertexSource<V>, L: DescriptorSetsCollection + Send + Sync,
              Pl: 'static + PipelineLayout + Send + Sync, Rp: RenderPass + RenderPassDesc + 'static + Send + Sync,
              Pc: 'static + Clone + Send + Sync,
              I: 'static + TypedBuffer<Content = [DrawIndirectCommand]>
    {
        assert!(self.render_pass.is_compatible_with(pipeline.subpass().render_pass()));
        assert_eq!(self.render_pass_subpass, pipeline.subpass().index());

        unsafe {
            SecondaryGraphicsCommandBufferBuilder {
                inner: self.inner.draw_indirect(buffer, pipeline, vertices, dynamic, sets, push_constants),
                render_pass: self.render_pass,
                render_pass_subpass: self.render_pass_subpass,
                framebuffer: self.framebuffer,
            }
        }
    }

    /// See the docs of build().
    #[inline]
    pub fn build_raw(self) -> Result<SecondaryGraphicsCommandBuffer<R, P>, OomError> {
        let inner = try!(self.inner.build());

        Ok(SecondaryGraphicsCommandBuffer {
            inner: inner,
            render_pass: self.render_pass,
            render_pass_subpass: self.render_pass_subpass,
        })
    }

    /// Finish recording commands and build the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    ///
    #[inline]
    pub fn build(self) -> Arc<SecondaryGraphicsCommandBuffer<R, P>> {
        Arc::new(self.build_raw().unwrap())
    }
}

/// Represents a collection of commands to be executed by the GPU.
///
/// A secondary graphics command buffer contains draw commands and non-draw commands. Secondary
/// command buffers can't specify which framebuffer they are drawing to. Instead you must create
/// a primary command buffer, specify a framebuffer, and then call the secondary command buffer.
///
/// A secondary graphics command buffer can't be called outside of a renderpass.
pub struct SecondaryGraphicsCommandBuffer<R, P = Arc<StandardCommandPool>> where P: CommandPool {
    inner: InnerCommandBuffer<P>,
    render_pass: Arc<R>,
    render_pass_subpass: u32,
}

/// A prototype of a secondary compute command buffer.
pub struct SecondaryComputeCommandBufferBuilder<P = Arc<StandardCommandPool>> where P: CommandPool {
    inner: InnerCommandBufferBuilder<P>,
}

impl SecondaryComputeCommandBufferBuilder<Arc<StandardCommandPool>> {
    /// Builds a new secondary command buffer and start recording commands in it.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    /// - Panicks if the device and queue family do not belong to the same physical device.
    ///
    #[inline]
    pub fn new(device: &Arc<Device>, queue_family: QueueFamily)
               -> SecondaryComputeCommandBufferBuilder<Arc<StandardCommandPool>>
    {
        SecondaryComputeCommandBufferBuilder::raw(Device::standard_command_pool(device,
                                                  queue_family)).unwrap()
    }
}

impl<P> SecondaryComputeCommandBufferBuilder<P> where P: CommandPool {
    /// See the docs of new().
    #[inline]
    pub fn raw(pool: P) -> Result<SecondaryComputeCommandBufferBuilder<P>, OomError> {
        let inner = try!(InnerCommandBufferBuilder::new::<UnsafeRenderPass>(pool, true, None, None));
        Ok(SecondaryComputeCommandBufferBuilder { inner: inner })
    }

    /// Writes data to a buffer.
    ///
    /// The data is stored inside the command buffer and written to the given buffer slice.
    /// This function is intended to be used for small amounts of data (only 64kB is allowed). if
    /// you want to transfer large amounts of data, use copies between buffers.
    ///
    /// # Panic
    ///
    /// - Panicks if the size of `data` is not the same as the size of the buffer slice.
    /// - Panicks if the size of `data` is superior to 65536 bytes.
    /// - Panicks if the offset or size is not a multiple of 4.
    /// - Panicks if the buffer wasn't created with the right usage.
    /// - Panicks if the queue family doesn't support transfer operations.
    ///
    #[inline]
    pub fn update_buffer<'a, B, T, Bb>(self, buffer: B, data: &T) -> SecondaryComputeCommandBufferBuilder<P>
        where B: Into<BufferSlice<'a, T, Bb>>, Bb: Buffer + 'static, T: Clone + 'static + Send + Sync
    {
        unsafe {
            SecondaryComputeCommandBufferBuilder {
                inner: self.inner.update_buffer(buffer, data)
            }
        }
    }

    /// Fills a buffer with data.
    ///
    /// The data is repeated until it fills the range from `offset` to `offset + size`.
    /// Since the data is a u32, the offset and the size must be multiples of 4.
    ///
    /// # Panic
    ///
    /// - Panicks if `offset + data` is superior to the size of the buffer.
    /// - Panicks if the offset or size is not a multiple of 4.
    /// - Panicks if the buffer wasn't created with the right usage.
    /// - Panicks if the queue family doesn't support transfer operations.
    ///
    /// # Safety
    ///
    /// - Type safety is not enforced by the API.
    pub unsafe fn fill_buffer<B>(self, buffer: &Arc<B>, offset: usize, size: usize, data: u32)
                                 -> SecondaryComputeCommandBufferBuilder<P>
        where B: Buffer + 'static
    {
        SecondaryComputeCommandBufferBuilder {
            inner: self.inner.fill_buffer(buffer, offset, size, data)
        }
    }

    /// See the docs of build().
    #[inline]
    pub fn build_raw(self) -> Result<SecondaryComputeCommandBuffer<P>, OomError> {
        let inner = try!(self.inner.build());
        Ok(SecondaryComputeCommandBuffer { inner: inner })
    }

    /// Finish recording commands and build the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    ///
    #[inline]
    pub fn build(self) -> Arc<SecondaryComputeCommandBuffer<P>> {
        Arc::new(self.build_raw().unwrap())
    }
}

/// Represents a collection of commands to be executed by the GPU.
///
/// A secondary compute command buffer contains non-draw commands (like copy commands, compute
/// shader execution, etc.). It can only be called outside of a renderpass.
pub struct SecondaryComputeCommandBuffer<P = Arc<StandardCommandPool>>
    where P: CommandPool
{
    inner: InnerCommandBuffer<P>,
}

/// The dynamic state to use for a draw command.
#[derive(Debug, Clone)]
pub struct DynamicState {
    pub line_width: Option<f32>,
    pub viewports: Option<Vec<Viewport>>,
    pub scissors: Option<Vec<Scissor>>,
}

impl DynamicState {
    #[inline]
    pub fn none() -> DynamicState {
        DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
        }
    }
}

impl Default for DynamicState {
    #[inline]
    fn default() -> DynamicState {
        DynamicState::none()
    }
}
