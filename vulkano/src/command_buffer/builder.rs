// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::Buffer;
use device::DeviceOwned;
use command_buffer::DynamicState;
use command_buffer::cb::AddCommand;
use command_buffer::cb::CommandBufferBuild;
use command_buffer::cmd;
use descriptor::descriptor_set::DescriptorSetsCollection;
use framebuffer::FramebufferAbstract;
use framebuffer::RenderPassAbstract;
use framebuffer::RenderPassDescClearValues;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use pipeline::vertex::VertexSource;

///
/// > **Note**: This trait is just a utility trait. Do not implement it yourself. Instead
/// > implement the `AddCommand` and `CommandBufferBuild` traits.
pub unsafe trait CommandBufferBuilder: DeviceOwned {
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
    fn fill_buffer<B, O>(self, buffer: B, data: u32) -> Result<O, cmd::CmdFillBufferError>
        where Self: Sized + AddCommand<cmd::CmdFillBuffer<B>, Out = O>,
              B: Buffer
    {
        let cmd = cmd::CmdFillBuffer::new(buffer, data)?;
        Ok(self.add(cmd))
    }

    /// Adds a command that starts a render pass.
    ///
    /// If `secondary` is true, then you will only be able to add secondary command buffers while
    /// you're inside the first subpass of the render pass. If `secondary` is false, you will only
    /// be able to add inline draw commands and not secondary command buffers.
    ///
    /// You must call this before you can add draw commands.
    #[inline]
    fn begin_render_pass<F, C, O>(self, framebuffer: F, secondary: bool, clear_values: C)
                                  -> O
        where Self: Sized + AddCommand<cmd::CmdBeginRenderPass<Arc<RenderPassAbstract>, F>, Out = O>,
              F: FramebufferAbstract + RenderPassDescClearValues<C>
    {
        let cmd = cmd::CmdBeginRenderPass::new(framebuffer, secondary, clear_values);
        self.add(cmd)
    }

    /// Adds a command that jumps to the next subpass of the current render pass.
    #[inline]
    fn next_subpass<O>(self, secondary: bool) -> O
        where Self: Sized + AddCommand<cmd::CmdNextSubpass, Out = O>
    {
        let cmd = cmd::CmdNextSubpass::new(secondary);
        self.add(cmd)
    }

    /// Adds a command that ends the current render pass.
    ///
    /// This must be called after you went through all the subpasses and before you can build
    /// the command buffer or add further commands.
    #[inline]
    fn end_render_pass<O>(self) -> O
        where Self: Sized + AddCommand<cmd::CmdEndRenderPass, Out = O>
    {
        let cmd = cmd::CmdEndRenderPass::new();
        self.add(cmd)
    }

    /// Adds a command that draws.
    ///
    /// Can only be used from inside a render pass.
    #[inline]
    fn draw<P, S, Pc, V, O>(self, pipeline: P, dynamic: DynamicState, vertices: V, sets: S,
                            push_constants: Pc) -> O
        where Self: Sized + AddCommand<cmd::CmdDraw<V, P, S, Pc>, Out = O>,
              S: DescriptorSetsCollection,
              P: VertexSource<V> + GraphicsPipelineAbstract + Clone
    {
        let cmd = cmd::CmdDraw::new(pipeline, dynamic, vertices, sets, push_constants);
        self.add(cmd)
    }

    /// Executes a compute shader.
    fn dispatch<P, S, Pc, O>(self, dimensions: [u32; 3], pipeline: P, sets: S, push_constants: Pc)
                             -> Result<O, cmd::CmdDispatchError>
        where Self: Sized + AddCommand<cmd::CmdDispatch<P, S, Pc>, Out = O>,
              S: DescriptorSetsCollection,
              P: Clone + ComputePipelineAbstract,
    {
        let cmd = try!(cmd::CmdDispatch::new(dimensions, pipeline, sets, push_constants));
        Ok(self.add(cmd))
    }

    #[inline]
    fn build<O>(self) -> O
        where Self: Sized + CommandBufferBuild<Out = O>
    {
        CommandBufferBuild::build(self)
    }
}

pub unsafe trait CommandBufferBuilderBuffered {
    /// Adds a pipeline barrier to the underlying command buffer and bypasses the flushing
    /// mechanism.
    fn add_non_buffered_pipeline_barrier(&mut self, cmd: &cmd::CmdPipelineBarrier);

    /// Flushes all the commands that haven't been flushed to the inner builder.
    fn flush(&mut self);
}
