use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::AbstractCommandBuffer;
use command_buffer::CommandBufferPool;
use command_buffer::inner::InnerCommandBufferBuilder;
use command_buffer::inner::InnerCommandBuffer;
use command_buffer::inner::Submission;
use command_buffer::inner::submit as inner_submit;
use descriptor_set::Layout as PipelineLayoutDesc;
use descriptor_set::DescriptorSetsCollection;
use device::Queue;
use format::PossibleFloatOrCompressedFormatDesc;
use format::PossibleFloatFormatDesc;
use format::StrongStorage;
use framebuffer::Framebuffer;
use framebuffer::RenderPass;
use framebuffer::CompatibleLayout as RenderPassCompatibleLayout;
use framebuffer::Layout as RenderPassLayout;
use framebuffer::LayoutClearValues as RenderPassLayoutClearValues;
use framebuffer::Subpass;
use image::Image;
use image::ImageTypeMarker;
use memory::MemorySourceChunk;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::Definition as VertexDefinition;
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
pub struct PrimaryCommandBufferBuilder {
    inner: InnerCommandBufferBuilder,
}

impl PrimaryCommandBufferBuilder {
    /// Builds a new primary command buffer and start recording commands in it.
    #[inline]
    pub fn new(pool: &Arc<CommandBufferPool>)
               -> Result<PrimaryCommandBufferBuilder, OomError>
    {
        let inner = try!(InnerCommandBufferBuilder::new::<()>(pool, false, None));
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
    pub fn update_buffer<'a, B, T, Bo: ?Sized + 'static, Bm: 'static>(self, buffer: B, data: &T) -> PrimaryCommandBufferBuilder
        where B: Into<BufferSlice<'a, T, Bo, Bm>>, Bm: MemorySourceChunk
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

    pub fn copy_buffer_to_color_image<'a, S, Ty, F, Im, So: ?Sized, Sm>(self, source: S, destination: &Arc<Image<Ty, F, Im>>)
                                                    -> PrimaryCommandBufferBuilder
        where S: Into<BufferSlice<'a, [F::Pixel], So, Sm>>, F: StrongStorage + PossibleFloatOrCompressedFormatDesc,
              Ty: ImageTypeMarker, So: 'static, Sm: MemorySourceChunk + 'static
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.copy_buffer_to_color_image(source, destination),
            }
        }
    }

    ///
    /// Note that compressed formats are not supported.
    pub fn clear_color_image<'a, Ty, F, M>(self, image: &Arc<Image<Ty, F, M>>,
                                           color: F::ClearValue) -> PrimaryCommandBufferBuilder
        where Ty: ImageTypeMarker, F: PossibleFloatFormatDesc
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.clear_color_image(image, color),
            }
        }
    }

    /// Executes secondary compute command buffers within this primary command buffer.
    #[inline]
    pub fn execute_commands(self, cb: &Arc<SecondaryComputeCommandBuffer>)
                            -> PrimaryCommandBufferBuilder
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.execute_commands(cb.clone() as Arc<_>, &cb.inner)
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
    pub fn draw_inline<R, F, C>(self, renderpass: &Arc<RenderPass<R>>,
                                framebuffer: &Arc<Framebuffer<F>>, clear_values: C)
                                -> PrimaryCommandBufferBuilderInlineDraw
        where F: RenderPassLayout + RenderPassLayoutClearValues<C> + 'static, R: RenderPassLayout + 'static
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
                num_subpasses: framebuffer.renderpass().num_subpasses(),
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
    pub fn draw_secondary<R, F, C>(self, renderpass: &Arc<RenderPass<R>>,
                                   framebuffer: &Arc<Framebuffer<F>>, clear_values: C)
                                   -> PrimaryCommandBufferBuilderSecondaryDraw
        where F: RenderPassLayout + RenderPassLayoutClearValues<C> + 'static,
              R: RenderPassLayout + 'static
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
                num_subpasses: framebuffer.renderpass().num_subpasses(),
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
    pub fn draw<V, L, Pv, Pl, Rp>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                              vertices: V, dynamic: &DynamicState, sets: L)
                              -> PrimaryCommandBufferBuilderInlineDraw
        where Pv: VertexDefinition + VertexSource<V> + 'static, Pl: PipelineLayoutDesc + 'static, Rp: 'static,
              L: DescriptorSetsCollection + 'static
    {
        // FIXME: check subpass

        unsafe {
            PrimaryCommandBufferBuilderInlineDraw {
                inner: self.inner.draw(pipeline, vertices, dynamic, sets),
                num_subpasses: self.num_subpasses,
                current_subpass: self.current_subpass,
            }
        }
    }

    /// Calls `vkCmdDrawIndexed`.
    pub fn draw_indexed<'a, V, L, Pv, Pl, Rp, I, Ib, Ibo: ?Sized + 'static, Ibm: 'static>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                                              vertices: V, indices: Ib, dynamic: &DynamicState,
                                              sets: L) -> PrimaryCommandBufferBuilderInlineDraw
        where Pv: 'static + VertexDefinition + VertexSource<V>, Pl: 'static + PipelineLayoutDesc, Rp: 'static,
              Ib: Into<BufferSlice<'a, [I], Ibo, Ibm>>, I: 'static + Index,
              L: DescriptorSetsCollection + 'static,
              Ibm: MemorySourceChunk
    {
        // FIXME: check subpass

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
    ///
    /// # Panic
    ///
    /// - Panicks if not at the last subpass.
    ///
    #[inline]
    pub fn draw_end(mut self) -> PrimaryCommandBufferBuilder {
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
    /// - Panicks if the secondary command buffers wasn't created with a compatible
    ///   renderpass or is using the wrong subpass.
    #[inline]
    pub fn execute_commands<R: 'static>(mut self, cb: &Arc<SecondaryGraphicsCommandBuffer<R>>)
                                        -> PrimaryCommandBufferBuilderSecondaryDraw
    {
        // FIXME: check renderpass and subpass

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
    pub fn draw_end(mut self) -> PrimaryCommandBufferBuilder {
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
pub struct PrimaryCommandBuffer {
    inner: InnerCommandBuffer,
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
pub fn submit(cmd: &Arc<PrimaryCommandBuffer>, queue: &Arc<Queue>) -> Result<Submission, OomError> {       // TODO: wrong error type
    inner_submit(&cmd.inner, cmd.clone() as Arc<_>, queue)
}

impl AbstractCommandBuffer for PrimaryCommandBuffer {}

/// A prototype of a secondary compute command buffer.
pub struct SecondaryGraphicsCommandBufferBuilder<R> {
    inner: InnerCommandBufferBuilder,
    renderpass_layout: R,
    renderpass_subpass: u32,
}

impl<R> SecondaryGraphicsCommandBufferBuilder<R>
    where R: RenderPassLayout
{
    /// Builds a new secondary command buffer and start recording commands in it.
    #[inline]
    pub fn new(pool: &Arc<CommandBufferPool>, subpass: Subpass<R>)
               -> Result<SecondaryGraphicsCommandBufferBuilder<R>, OomError>
        where R: Clone + 'static
    {
        let inner = try!(InnerCommandBufferBuilder::new(pool, true, Some(subpass)));
        Ok(SecondaryGraphicsCommandBufferBuilder {
            inner: inner,
            renderpass_layout: subpass.render_pass().layout().clone(),
            renderpass_subpass: subpass.index(),
        })
    }

    /// Calls `vkCmdDraw`.
    // FIXME: push constants
    pub fn draw<V, L, Pv, Pl, Rp>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                              vertices: V, dynamic: &DynamicState, sets: L)
                              -> SecondaryGraphicsCommandBufferBuilder<R>
        where Pv: VertexDefinition + VertexSource<V> + 'static, Pl: PipelineLayoutDesc + 'static,
              Rp: RenderPassLayout + 'static, L: DescriptorSetsCollection + 'static,
              R: RenderPassCompatibleLayout<Rp>
    {
        assert!(self.renderpass_layout.is_compatible_with(pipeline.subpass().render_pass().layout()));
        assert_eq!(self.renderpass_subpass, pipeline.subpass().index());

        unsafe {
            SecondaryGraphicsCommandBufferBuilder {
                inner: self.inner.draw(pipeline, vertices, dynamic, sets),
                renderpass_layout: self.renderpass_layout,
                renderpass_subpass: self.renderpass_subpass,
            }
        }
    }

    /// Calls `vkCmdDrawIndexed`.
    pub fn draw_indexed<'a, V, L, Pv, Pl, Rp, I, Ib, Ibo: ?Sized + 'static, Ibm: 'static>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                                              vertices: V, indices: Ib, dynamic: &DynamicState,
                                              sets: L) -> SecondaryGraphicsCommandBufferBuilder<R>
        where Pv: 'static + VertexDefinition + VertexSource<V>, Pl: 'static + PipelineLayoutDesc,
              Rp: RenderPassLayout + 'static,
              Ib: Into<BufferSlice<'a, [I], Ibo, Ibm>>, I: 'static + Index,
              L: DescriptorSetsCollection + 'static,
              Ibm: MemorySourceChunk
    {
        assert!(self.renderpass_layout.is_compatible_with(pipeline.subpass().render_pass().layout()));
        assert_eq!(self.renderpass_subpass, pipeline.subpass().index());

        unsafe {
            SecondaryGraphicsCommandBufferBuilder {
                inner: self.inner.draw_indexed(pipeline, vertices, indices, dynamic, sets),
                renderpass_layout: self.renderpass_layout,
                renderpass_subpass: self.renderpass_subpass,
            }
        }
    }

    /// Finish recording commands and build the command buffer.
    #[inline]
    pub fn build(self) -> Result<Arc<SecondaryGraphicsCommandBuffer<R>>, OomError> {
        let inner = try!(self.inner.build());

        Ok(Arc::new(SecondaryGraphicsCommandBuffer {
            inner: inner,
            renderpass_layout: self.renderpass_layout,
            renderpass_subpass: self.renderpass_subpass,
        }))
    }
}

/// Represents a collection of commands to be executed by the GPU.
///
/// A secondary graphics command buffer contains draw commands and non-draw commands. Secondary
/// command buffers can't specify which framebuffer they are drawing to. Instead you must create
/// a primary command buffer, specify a framebuffer, and then call the secondary command buffer.
///
/// A secondary graphics command buffer can't be called outside of a renderpass.
pub struct SecondaryGraphicsCommandBuffer<R> {
    inner: InnerCommandBuffer,
    renderpass_layout: R,
    renderpass_subpass: u32,
}

impl<R> AbstractCommandBuffer for SecondaryGraphicsCommandBuffer<R> {}

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
        let inner = try!(InnerCommandBufferBuilder::new::<()>(pool, true, None));
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
    pub fn update_buffer<'a, B, T, Bo: ?Sized + 'static, Bm: 'static>(self, buffer: B, data: &T) -> SecondaryComputeCommandBufferBuilder
        where B: Into<BufferSlice<'a, T, Bo, Bm>>, Bm: MemorySourceChunk
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

impl AbstractCommandBuffer for SecondaryComputeCommandBuffer {}

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
