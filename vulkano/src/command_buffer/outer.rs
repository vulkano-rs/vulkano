use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::CommandBufferPool;
use command_buffer::inner::InnerCommandBufferBuilder;
use command_buffer::inner::InnerCommandBuffer;
use descriptor_set::PipelineLayoutDesc;
use device::Queue;
use framebuffer::Framebuffer;
use framebuffer::RenderPass;
use framebuffer::RenderPassLayout;
use memory::MemorySourceChunk;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::MultiVertex;

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
pub struct PrimaryCommandBufferBuilder {
    inner: InnerCommandBufferBuilder,
}

impl PrimaryCommandBufferBuilder {
    /// Builds a new primary command buffer and start recording commands in it.
    #[inline]
    pub fn new(pool: &Arc<CommandBufferPool>)
               -> Result<PrimaryCommandBufferBuilder, OomError>
    {
        let inner = try!(InnerCommandBufferBuilder::new(pool, false));
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
    pub fn update_buffer<'a, B, T: 'a, M: 'a>(self, buffer: B, data: &T)
                                              -> PrimaryCommandBufferBuilder
        where B: Into<BufferSlice<'a, T, M>>
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
    pub unsafe fn fill_buffer<T: 'static, M>(self, buffer: &Arc<Buffer<T, M>>, offset: usize,
                                             size: usize, data: u32) -> PrimaryCommandBufferBuilder
        where M: MemorySourceChunk + 'static
    {
        PrimaryCommandBufferBuilder {
            inner: self.inner.fill_buffer(buffer, offset, size, data)
        }
    }

    pub fn copy_buffer<T: ?Sized + 'static, Ms, Md>(self, source: &Arc<Buffer<T, Ms>>,
                                                    destination: &Arc<Buffer<T, Md>>)
                                                    -> PrimaryCommandBufferBuilder
        where Ms: MemorySourceChunk + 'static, Md: MemorySourceChunk + 'static
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.copy_buffer(source, destination),
            }
        }
    }

    /// Executes secondary compute command buffers within this primary command buffer.
    #[inline]
    pub fn execute_commands<'a, I>(self, iter: I) -> PrimaryCommandBufferBuilder
        where I: Iterator<Item = &'a SecondaryComputeCommandBuffer>
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.execute_commands(iter.map(|cb| &cb.inner))
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
    pub fn draw_inline<R, F>(self, renderpass: &Arc<RenderPass<R>>,
                             framebuffer: &Arc<Framebuffer<F>>, clear_values: F::ClearValues)
                             -> PrimaryCommandBufferBuilderInlineDraw
        where F: RenderPassLayout, R: RenderPassLayout
    {
        // FIXME: check for compatibility

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let clear_values = framebuffer.renderpass().layout().convert_clear_values(clear_values)
                                      .collect::<Vec<_>>();

        unsafe {
            let inner = self.inner.begin_renderpass(renderpass, framebuffer, false, &clear_values);

            PrimaryCommandBufferBuilderInlineDraw {
                inner: inner,
                current_subpass: 0,
                num_subpasses: 1,   // FIXME:
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
    pub fn draw_secondary<R, F>(self, renderpass: &Arc<RenderPass<R>>,
                                framebuffer: &Arc<Framebuffer<F>>, clear_values: F::ClearValues)
                                -> PrimaryCommandBufferBuilderSecondaryDraw
        where F: RenderPassLayout, R: RenderPassLayout
    {
        // FIXME: check for compatibility

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let clear_values = framebuffer.renderpass().layout().convert_clear_values(clear_values)
                                      .collect::<Vec<_>>();

        unsafe {
            let inner = self.inner.begin_renderpass(renderpass, framebuffer, true, &clear_values);

            PrimaryCommandBufferBuilderSecondaryDraw {
                inner: inner,
                current_subpass: 0,
                num_subpasses: 1,   // FIXME:
            }
        }
    }

    /// Finish recording commands and build the command buffer.
    #[inline]
    pub fn build(self) -> Result<Arc<PrimaryCommandBuffer>, OomError> {
        let inner = try!(self.inner.build());
        Ok(Arc::new(PrimaryCommandBuffer { inner: inner }))
    }
}

/// Object that you obtain when calling `draw_inline` or `next_subpass_inline`.
pub struct PrimaryCommandBufferBuilderInlineDraw {
    inner: InnerCommandBufferBuilder,
    current_subpass: u32,
    num_subpasses: u32,
}

impl PrimaryCommandBufferBuilderInlineDraw {
    /// Calls `vkCmdDraw`.
    // FIXME: push constants
    pub fn draw<V, L>(self, pipeline: &Arc<GraphicsPipeline<V, L>>,
                      vertices: V, dynamic: &DynamicState, sets: L::DescriptorSets)
                      -> PrimaryCommandBufferBuilderInlineDraw
        where V: MultiVertex + 'static, L: PipelineLayoutDesc + 'static
    {
        unsafe {
            PrimaryCommandBufferBuilderInlineDraw {
                inner: self.inner.draw(pipeline, vertices, dynamic, sets),
                num_subpasses: self.num_subpasses,
                current_subpass: self.current_subpass,
            }
        }
    }

    /// Calls `vkCmdDrawIndexed`.
    pub fn draw_indexed<'a, V, L, I, Ib, IbM>(mut self, pipeline: &Arc<GraphicsPipeline<V, L>>,
                                              vertices: V, indices: Ib, dynamic: &DynamicState,
                                              sets: L::DescriptorSets) -> PrimaryCommandBufferBuilderInlineDraw
        where V: 'static + MultiVertex, L: 'static + PipelineLayoutDesc,
              Ib: Into<BufferSlice<'a, [I], IbM>>, I: 'static + Index, IbM: 'static
    {
        unsafe {
            PrimaryCommandBufferBuilderInlineDraw {
                inner: self.inner.draw_indexed(pipeline, vertices, indices, dynamic, sets),
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
    pub fn next_subpass_inline(self) -> PrimaryCommandBufferBuilderInlineDraw {
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
    pub fn next_subpass_secondary(self) -> PrimaryCommandBufferBuilderSecondaryDraw {
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
    #[inline]
    pub fn draw_end(mut self) -> PrimaryCommandBufferBuilder {
        unsafe {
            // skipping the remaining subpasses
            for _ in 0 .. (self.num_subpasses - self.current_subpass - 1) {
                self.inner = self.inner.next_subpass(false);
            }

            let inner = self.inner.end_renderpass();

            PrimaryCommandBufferBuilder {
                inner: inner,
            }
        }
    }
}

/// Object that you obtain when calling `draw_secondary` or `next_subpass_secondary`.
pub struct PrimaryCommandBufferBuilderSecondaryDraw {
    inner: InnerCommandBufferBuilder,
    num_subpasses: u32,
    current_subpass: u32,
}

impl PrimaryCommandBufferBuilderSecondaryDraw {
    /// Switches to the next subpass of the current renderpass.
    ///
    /// This function is similar to `draw_inline` on the builder.
    ///
    /// # Panic
    ///
    /// - Panicks if no more subpasses remain.
    ///
    #[inline]
    pub fn next_subpass_inline(self) -> PrimaryCommandBufferBuilderInlineDraw {
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
    pub fn next_subpass_secondary(self) -> PrimaryCommandBufferBuilderSecondaryDraw {
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
    /// - Panicks if one of the secondary command buffers wasn't created with a compatible
    ///   renderpass or is using the wrong subpass.
    #[inline]
    pub fn execute_commands<'a, I>(mut self, iter: I) -> PrimaryCommandBufferBuilderSecondaryDraw
        where I: Iterator<Item = &'a SecondaryGraphicsCommandBuffer>
    {
        // FIXME: check renderpass and subpass

        unsafe {
            self.inner = self.inner.execute_commands(iter.map(|cb| &cb.inner));
            self
        }
    }

    /// Finish drawing this renderpass and get back the builder.
    #[inline]
    pub fn draw_end(mut self) -> PrimaryCommandBufferBuilder {
        unsafe {
            // skipping the remaining subpasses
            for _ in 0 .. (self.num_subpasses - self.current_subpass - 1) {
                self.inner = self.inner.next_subpass(false);
            }

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
pub struct PrimaryCommandBuffer {
    inner: InnerCommandBuffer,
}

impl PrimaryCommandBuffer {
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
    pub fn submit(&self, queue: &mut Queue) -> Result<(), OomError> {       // TODO: wrong error type
        self.inner.submit(queue)
    }
}

/// A prototype of a secondary compute command buffer.
pub struct SecondaryGraphicsCommandBufferBuilder {
    inner: InnerCommandBufferBuilder,
}

impl SecondaryGraphicsCommandBufferBuilder {
    /// Builds a new secondary command buffer and start recording commands in it.
    #[inline]
    pub fn new(pool: &Arc<CommandBufferPool>)
               -> Result<SecondaryGraphicsCommandBufferBuilder, OomError>
    {
        let inner = try!(InnerCommandBufferBuilder::new(pool, true));
        Ok(SecondaryGraphicsCommandBufferBuilder { inner: inner })
    }

    /// Finish recording commands and build the command buffer.
    #[inline]
    pub fn build(self) -> Result<Arc<SecondaryGraphicsCommandBuffer>, OomError> {
        let inner = try!(self.inner.build());
        Ok(Arc::new(SecondaryGraphicsCommandBuffer { inner: inner }))
    }
}

/// Represents a collection of commands to be executed by the GPU.
///
/// A secondary graphics command buffer contains draw commands and non-draw commands. Secondary
/// command buffers can't specify which framebuffer they are drawing to. Instead you must create
/// a primary command buffer, specify a framebuffer, and then call the secondary command buffer.
///
/// A secondary graphics command buffer can't be called outside of a renderpass.
pub struct SecondaryGraphicsCommandBuffer {
    inner: InnerCommandBuffer,
}

/// A prototype of a secondary compute command buffer.
pub struct SecondaryComputeCommandBufferBuilder {
    inner: InnerCommandBufferBuilder,
}

impl SecondaryComputeCommandBufferBuilder {
    /// Builds a new secondary command buffer and start recording commands in it.
    #[inline]
    pub fn new(pool: &Arc<CommandBufferPool>)
               -> Result<SecondaryComputeCommandBufferBuilder, OomError>
    {
        let inner = try!(InnerCommandBufferBuilder::new(pool, true));
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
    pub fn update_buffer<'a, B, T: 'a, M: 'a>(self, buffer: B, data: &T)
                                              -> SecondaryComputeCommandBufferBuilder
        where B: Into<BufferSlice<'a, T, M>>
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
    pub unsafe fn fill_buffer<T: 'static, M>(self, buffer: &Arc<Buffer<T, M>>, offset: usize,
                                             size: usize, data: u32)
                                             -> SecondaryComputeCommandBufferBuilder
        where M: MemorySourceChunk + 'static
    {
        SecondaryComputeCommandBufferBuilder {
            inner: self.inner.fill_buffer(buffer, offset, size, data)
        }
    }

    /// Finish recording commands and build the command buffer.
    #[inline]
    pub fn build(self) -> Result<Arc<SecondaryComputeCommandBuffer>, OomError> {
        let inner = try!(self.inner.build());
        Ok(Arc::new(SecondaryComputeCommandBuffer { inner: inner }))
    }
}

/// Represents a collection of commands to be executed by the GPU.
///
/// A secondary compute command buffer contains non-draw commands (like copy commands, compute
/// shader execution, etc.). It can only be called outside of a renderpass.
pub struct SecondaryComputeCommandBuffer {
    inner: InnerCommandBuffer,
}

/// The dynamic state to use for a draw command.
#[derive(Debug, Copy, Clone)]
pub struct DynamicState {
    pub line_width: Option<f32>,
}

impl DynamicState {
    #[inline]
    pub fn none() -> DynamicState {
        DynamicState {
            line_width: None,
        }
    }
}

impl Default for DynamicState {
    #[inline]
    fn default() -> DynamicState {
        DynamicState::none()
    }
}
