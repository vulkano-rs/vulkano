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
use command_buffer::CommandBufferPool;
use command_buffer::inner::InnerCommandBufferBuilder;
use command_buffer::inner::InnerCommandBuffer;
use command_buffer::inner::Submission;
use command_buffer::inner::submit as inner_submit;
use descriptor_set::Layout as PipelineLayoutDesc;
use descriptor_set::DescriptorSetsCollection;
use device::Queue;
use framebuffer::Framebuffer;
use framebuffer::UnsafeRenderPass;
use framebuffer::RenderPassCompatible;
use framebuffer::RenderPass;
use framebuffer::RenderPassClearValues;
use framebuffer::Subpass;
use image::traits::Image;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use pipeline::ComputePipeline;
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
    pub fn update_buffer<'a, B, T, Bb>(self, buffer: B, data: &T) -> PrimaryCommandBufferBuilder
        where B: Into<BufferSlice<'a, T, Bb>>, Bb: Buffer + 'static, T: Clone + 'static
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
                                 size: usize, data: u32) -> PrimaryCommandBufferBuilder
        where B: Buffer + 'static
    {
        PrimaryCommandBufferBuilder {
            inner: self.inner.fill_buffer(buffer, offset, size, data)
        }
    }

    pub fn copy_buffer<T: ?Sized + 'static, Bs, Bd>(self, source: &Arc<Bs>, destination: &Arc<Bd>)
                                                    -> PrimaryCommandBufferBuilder
        where Bs: TypedBuffer<Content = T> + 'static, Bd: TypedBuffer<Content = T> + 'static
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.copy_buffer(source, destination),
            }
        }
    }

    pub fn copy_buffer_to_color_image<'a, P, S, Img, Sb>(self, source: S, destination: &Arc<Img>, mip_level: u32, array_layers_range: Range<u32>,
                                                                offset: [u32; 3], extent: [u32; 3])
                                                    -> PrimaryCommandBufferBuilder
        where S: Into<BufferSlice<'a, [P], Sb>>, Sb: Buffer + 'static,
              Img: ImageContent<P> + 'static
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.copy_buffer_to_color_image(source, destination, mip_level,
                                                             array_layers_range, offset, extent),
            }
        }
    }

    pub fn blit<Si, Di>(self, source: &Arc<Si>, source_mip_level: u32,
                        source_array_layers: Range<u32>, src_coords: [Range<i32>; 3],
                        destination: &Arc<Di>, dest_mip_level: u32,
                        dest_array_layers: Range<u32>, dest_coords: [Range<i32>; 3])
                        -> PrimaryCommandBufferBuilder
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
                                       -> PrimaryCommandBufferBuilder
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
    pub fn execute_commands(self, cb: &Arc<SecondaryComputeCommandBuffer>)
                            -> PrimaryCommandBufferBuilder
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.execute_commands(cb.clone() as Arc<_>, &cb.inner)
            }
        }
    }

    /// Executes a compute pipeline.
    #[inline]
    pub fn dispatch<Pl, L>(self, pipeline: &Arc<ComputePipeline<Pl>>, sets: L,
                           dimensions: [u32; 3]) -> PrimaryCommandBufferBuilder
        where L: 'static + DescriptorSetsCollection,
              Pl: 'static + PipelineLayoutDesc
    {
        unsafe {
            PrimaryCommandBufferBuilder {
                inner: self.inner.dispatch(pipeline, sets, dimensions)
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
                                -> PrimaryCommandBufferBuilderInlineDraw
        where F: RenderPass + RenderPassClearValues<C> + 'static, R: RenderPass + 'static
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
                                   -> PrimaryCommandBufferBuilderSecondaryDraw
        where F: RenderPass + RenderPassClearValues<C> + 'static,
              R: RenderPass + 'static
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
    pub fn draw_indexed<'a, V, L, Pv, Pl, Rp, I, Ib, Ibb>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                                              vertices: V, indices: Ib, dynamic: &DynamicState,
                                              sets: L) -> PrimaryCommandBufferBuilderInlineDraw
        where Pv: 'static + VertexDefinition + VertexSource<V>, Pl: 'static + PipelineLayoutDesc, Rp: 'static,
              Ib: Into<BufferSlice<'a, [I], Ibb>>, I: 'static + Index, Ibb: Buffer + 'static,
              L: DescriptorSetsCollection + 'static
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
    pub fn draw_end(self) -> PrimaryCommandBufferBuilder {
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
    current_subpass: u32,
    num_subpasses: u32,
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
    pub fn draw_end(self) -> PrimaryCommandBufferBuilder {
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
pub fn submit(cmd: &Arc<PrimaryCommandBuffer>, queue: &Arc<Queue>)
              -> Result<Arc<Submission>, OomError>
{       // TODO: wrong error type
    inner_submit(&cmd.inner, cmd.clone() as Arc<_>, queue)
}

/// A prototype of a secondary compute command buffer.
pub struct SecondaryGraphicsCommandBufferBuilder<R> {
    inner: InnerCommandBufferBuilder,
    render_pass: Arc<R>,
    render_pass_subpass: u32,
    framebuffer: Option<Arc<Framebuffer<R>>>,
}

impl<R> SecondaryGraphicsCommandBufferBuilder<R>
    where R: RenderPass + 'static
{
    /// Builds a new secondary command buffer and start recording commands in it.
    ///
    /// The `framebuffer` parameter is optional and can be used as an optimisation.
    #[inline]
    pub fn new(pool: &Arc<CommandBufferPool>, subpass: Subpass<R>,
               framebuffer: Option<&Arc<Framebuffer<R>>>)
               -> Result<SecondaryGraphicsCommandBufferBuilder<R>, OomError>
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
    pub fn draw<V, L, Pv, Pl, Rp>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                              vertices: V, dynamic: &DynamicState, sets: L)
                              -> SecondaryGraphicsCommandBufferBuilder<R>
        where Pv: VertexDefinition + VertexSource<V> + 'static, Pl: PipelineLayoutDesc + 'static,
              Rp: RenderPass + 'static, L: DescriptorSetsCollection + 'static,
              R: RenderPassCompatible<Rp>
    {
        assert!(self.render_pass.is_compatible_with(pipeline.subpass().render_pass()));
        assert_eq!(self.render_pass_subpass, pipeline.subpass().index());

        unsafe {
            SecondaryGraphicsCommandBufferBuilder {
                inner: self.inner.draw(pipeline, vertices, dynamic, sets),
                render_pass: self.render_pass,
                render_pass_subpass: self.render_pass_subpass,
                framebuffer: self.framebuffer,
            }
        }
    }

    /// Calls `vkCmdDrawIndexed`.
    pub fn draw_indexed<'a, V, L, Pv, Pl, Rp, I, Ib, Ibb>(self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                                              vertices: V, indices: Ib, dynamic: &DynamicState,
                                              sets: L) -> SecondaryGraphicsCommandBufferBuilder<R>
        where Pv: 'static + VertexDefinition + VertexSource<V>, Pl: 'static + PipelineLayoutDesc,
              Rp: RenderPass + 'static,
              Ib: Into<BufferSlice<'a, [I], Ibb>>, I: 'static + Index, Ibb: Buffer + 'static,
              L: DescriptorSetsCollection + 'static
    {
        assert!(self.render_pass.is_compatible_with(pipeline.subpass().render_pass()));
        assert_eq!(self.render_pass_subpass, pipeline.subpass().index());

        unsafe {
            SecondaryGraphicsCommandBufferBuilder {
                inner: self.inner.draw_indexed(pipeline, vertices, indices, dynamic, sets),
                render_pass: self.render_pass,
                render_pass_subpass: self.render_pass_subpass,
                framebuffer: self.framebuffer,
            }
        }
    }

    /// Finish recording commands and build the command buffer.
    #[inline]
    pub fn build(self) -> Result<Arc<SecondaryGraphicsCommandBuffer<R>>, OomError> {
        let inner = try!(self.inner.build());

        Ok(Arc::new(SecondaryGraphicsCommandBuffer {
            inner: inner,
            render_pass: self.render_pass,
            render_pass_subpass: self.render_pass_subpass,
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
    render_pass: Arc<R>,
    render_pass_subpass: u32,
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
    pub fn update_buffer<'a, B, T, Bb>(self, buffer: B, data: &T) -> SecondaryComputeCommandBufferBuilder
        where B: Into<BufferSlice<'a, T, Bb>>, Bb: Buffer + 'static, T: Clone + 'static
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
                                 -> SecondaryComputeCommandBufferBuilder
        where B: Buffer + 'static
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
