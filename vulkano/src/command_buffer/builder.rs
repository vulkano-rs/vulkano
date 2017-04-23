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
use std::sync::Arc;

use buffer::Buffer;
use buffer::TypedBuffer;
use device::DeviceOwned;
use command_buffer::DrawIndirectCommand;
use command_buffer::DynamicState;
use command_buffer::cb::AddCommand;
use command_buffer::cb::CommandBufferBuild;
use command_buffer::commands_extra;
use command_buffer::commands_raw;
use descriptor::descriptor_set::DescriptorSetsCollection;
use framebuffer::FramebufferAbstract;
use framebuffer::RenderPassAbstract;
use framebuffer::RenderPassDescClearValues;
use image::Image;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use pipeline::vertex::VertexSource;
use pipeline::input_assembly::Index;

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
    fn fill_buffer<B, O>(self, buffer: B, data: u32) -> Result<O, CommandBufferBuilderError<commands_raw::CmdFillBufferError>>
        where Self: Sized + AddCommand<commands_raw::CmdFillBuffer<B::Access>, Out = O>,
              B: Buffer
    {
        let cmd = match commands_raw::CmdFillBuffer::new(buffer.access(), data) {
            Ok(cmd) => cmd,
            Err(err) => return Err(CommandBufferBuilderError::CommandBuildError(err)),
        };

        Ok(self.add(cmd)?)
    }

    /// Adds a command that writes data to a buffer.
    #[inline]
    fn update_buffer<B, D, O>(self, buffer: B, data: D) -> Result<O, CommandBufferBuilderError<commands_raw::CmdUpdateBufferError>>
        where Self: Sized + AddCommand<commands_raw::CmdUpdateBuffer<B::Access, D>, Out = O>,
              B: Buffer
    {
        let cmd = match commands_raw::CmdUpdateBuffer::new(buffer.access(), data) {
            Ok(cmd) => cmd,
            Err(err) => return Err(CommandBufferBuilderError::CommandBuildError(err)),
        };

        Ok(self.add(cmd)?)
    }

    /// Adds a command that copies from a buffer to another.
    #[inline]
    fn copy_buffer<S, D, O>(self, src: S, dest: D) -> Result<O, CommandBufferBuilderError<commands_raw::CmdCopyBufferError>>
        where Self: Sized + AddCommand<commands_raw::CmdCopyBuffer<S::Access, D::Access>, Out = O>,
              S: Buffer,
              D: Buffer
    {
        let cmd = match commands_raw::CmdCopyBuffer::new(src.access(), dest.access()) {
            Ok(cmd) => cmd,
            Err(err) => return Err(CommandBufferBuilderError::CommandBuildError(err)),
        };

        Ok(self.add(cmd)?)
    }

    /// Adds a command that copies the content of a buffer to an image.
    ///
    /// For color images (ie. all formats except depth and/or stencil formats) this command does
    /// not perform any conversion. The data inside the buffer must already have the right format.
    /// TODO: talk about depth/stencil
    ///
    /// > **Note**: This function is technically safe because buffers can only contain integers or
    /// > floating point numbers, which are always valid whatever their memory representation is.
    /// > But unless your buffer actually contains only 32-bits integers, you are encouraged to use
    /// > this function only for zeroing the content of a buffer by passing `0` for the data.
    // TODO: not safe because of signalling NaNs
    #[inline]
    fn copy_buffer_to_image<B, I, O>(self, buffer: B, image: I)
                                     -> Result<O, CommandBufferBuilderError<commands_raw::CmdCopyBufferToImageError>>
        where Self: Sized + AddCommand<commands_raw::CmdCopyBufferToImage<B::Access, I::Access>, Out = O>,
              B: Buffer, I: Image
    {
        let cmd = match commands_raw::CmdCopyBufferToImage::new(buffer.access(), image.access()) {
            Ok(cmd) => cmd,
            Err(err) => return Err(CommandBufferBuilderError::CommandBuildError(err)),
        };

        Ok(self.add(cmd)?)
    }

    /// Same as `copy_buffer_to_image` but lets you specify a range for the destination image.
    #[inline]
    fn copy_buffer_to_image_dimensions<B, I, O>(self, buffer: B, image: I, offset: [u32; 3],
                                                size: [u32; 3], first_layer: u32, num_layers: u32,
                                                mipmap: u32) -> Result<O, CommandBufferBuilderError<commands_raw::CmdCopyBufferToImageError>>
        where Self: Sized + AddCommand<commands_raw::CmdCopyBufferToImage<B::Access, I::Access>, Out = O>,
              B: Buffer, I: Image
    {
        let cmd = match commands_raw::CmdCopyBufferToImage::with_dimensions(buffer.access(),
                                                                            image.access(), offset, size,
                                                                            first_layer, num_layers, mipmap)
        {
            Ok(cmd) => cmd,
            Err(err) => return Err(CommandBufferBuilderError::CommandBuildError(err)),
        };

        Ok(self.add(cmd)?)
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
                                  -> Result<O, CommandAddError>
        where Self: Sized + AddCommand<commands_raw::CmdBeginRenderPass<Arc<RenderPassAbstract + Send + Sync>, F>, Out = O>,
              F: FramebufferAbstract + RenderPassDescClearValues<C>
    {
        let cmd = commands_raw::CmdBeginRenderPass::new(framebuffer, secondary, clear_values);
        self.add(cmd)
    }

    /// Adds a command that jumps to the next subpass of the current render pass.
    #[inline]
    fn next_subpass<O>(self, secondary: bool) -> Result<O, CommandAddError>
        where Self: Sized + AddCommand<commands_raw::CmdNextSubpass, Out = O>
    {
        let cmd = commands_raw::CmdNextSubpass::new(secondary);
        self.add(cmd)
    }

    /// Adds a command that ends the current render pass.
    ///
    /// This must be called after you went through all the subpasses and before you can build
    /// the command buffer or add further commands.
    #[inline]
    fn end_render_pass<O>(self) -> Result<O, CommandAddError>
        where Self: Sized + AddCommand<commands_raw::CmdEndRenderPass, Out = O>
    {
        let cmd = commands_raw::CmdEndRenderPass::new();
        self.add(cmd)
    }

    /// Adds a command that draws.
    ///
    /// Can only be used from inside a render pass.
    #[inline]
    fn draw<P, S, Pc, V, O>(self, pipeline: P, dynamic: DynamicState, vertices: V, sets: S,
                            push_constants: Pc) -> Result<O, CommandAddError>
        where Self: Sized + AddCommand<commands_extra::CmdDraw<V, P, S, Pc>, Out = O>,
              S: DescriptorSetsCollection,
              P: VertexSource<V> + GraphicsPipelineAbstract + Clone
    {
        let cmd = commands_extra::CmdDraw::new(pipeline, dynamic, vertices, sets, push_constants);
        self.add(cmd)
    }

    /// Adds a command that draws indexed vertices.
    ///
    /// Can only be used from inside a render pass.
    #[inline]
    fn draw_indexed<P, S, Pc, V, Ib, I, O>(self, pipeline: P, dynamic: DynamicState,
        vertices: V, index_buffer: Ib, sets: S, push_constants: Pc) -> Result<O, CommandAddError>
        where Self: Sized + AddCommand<commands_extra::CmdDrawIndexed<V, Ib::Access, P, S, Pc>, Out = O>,
              S: DescriptorSetsCollection,
              P: VertexSource<V> + GraphicsPipelineAbstract + Clone,
              Ib: Buffer,
              Ib::Access: TypedBuffer<Content = [I]>,
              I: Index + 'static
    {
        let cmd = commands_extra::CmdDrawIndexed::new(pipeline, dynamic, vertices, index_buffer.access(),
                                           sets, push_constants);
        self.add(cmd)
    }

    /// Adds an indirect draw command.
    ///
    /// Can only be used from inside a render pass.
    #[inline]
    fn draw_indirect<P, S, Pc, V, B, I, O>(self, pipeline: P, dynamic: DynamicState,
        vertices: V, indirect_buffer: B, sets: S, push_constants: Pc) -> Result<O, CommandAddError>
        where Self: Sized + AddCommand<commands_extra::CmdDrawIndirect<V, B::Access, P, S, Pc>, Out = O>,
              S: DescriptorSetsCollection,
              P: VertexSource<V> + GraphicsPipelineAbstract + Clone,
              B: Buffer,
              B::Access: TypedBuffer<Content = [DrawIndirectCommand]>,
              I: Index + 'static
    {
        let cmd = commands_extra::CmdDrawIndirect::new(pipeline, dynamic, vertices, indirect_buffer.access(),
                                           sets, push_constants);
        self.add(cmd)
    }

    /// Executes a compute shader.
    fn dispatch<P, S, Pc, O>(self, dimensions: [u32; 3], pipeline: P, sets: S, push_constants: Pc)
                             -> Result<O, CommandBufferBuilderError<commands_extra::CmdDispatchError>>
        where Self: Sized + AddCommand<commands_extra::CmdDispatch<P, S, Pc>, Out = O>,
              S: DescriptorSetsCollection,
              P: Clone + ComputePipelineAbstract,
    {
        let cmd = match commands_extra::CmdDispatch::new(dimensions, pipeline, sets, push_constants) {
            Ok(cmd) => cmd,
            Err(err) => return Err(CommandBufferBuilderError::CommandBuildError(err)),
        };

        Ok(self.add(cmd)?)
    }

    /// Builds the actual command buffer.
    ///
    /// You must call this function after you have finished adding commands to the command buffer
    /// builder. A command buffer will returned, which you can then submit or use in an "execute
    /// commands" command.
    #[inline]
    fn build(self) -> Result<Self::Out, Self::Err>
        where Self: Sized + CommandBufferBuild
    {
        CommandBufferBuild::build(self)
    }

    /// Returns true if the pool of the builder supports graphics operations.
    fn supports_graphics(&self) -> bool;

    /// Returns true if the pool of the builder supports compute operations.
    fn supports_compute(&self) -> bool;
}

/// Error that can happen when adding a command to a command buffer builder.
#[derive(Debug, Copy, Clone)]
pub enum CommandBufferBuilderError<E> {
    /// Error while creating the command.
    CommandBuildError(E),

    /// Error while adding the command to the builder.
    CommandAddError(CommandAddError),
}

impl<E> From<CommandAddError> for CommandBufferBuilderError<E> {
    #[inline]
    fn from(err: CommandAddError) -> CommandBufferBuilderError<E> {
        CommandBufferBuilderError::CommandAddError(err)
    }
}

impl<E> error::Error for CommandBufferBuilderError<E> where E: error::Error {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CommandBufferBuilderError::CommandBuildError(_) => {
                "error while creating a command to add to a builder"
            },
            CommandBufferBuilderError::CommandAddError(_) => {
                "error while adding a command to the builder"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            CommandBufferBuilderError::CommandBuildError(ref err) => {
                Some(err)
            },
            CommandBufferBuilderError::CommandAddError(ref err) => {
                Some(err)
            },
        }
    }
}

impl<E> fmt::Display for CommandBufferBuilderError<E> where E: error::Error {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

/// Error that can happen when adding a command to a command buffer builder.
#[derive(Debug, Copy, Clone)]
pub enum CommandAddError {
    /// This command is forbidden when inside a render pass.
    ForbiddenInsideRenderPass,

    /// This command is forbidden when outside of a render pass.
    ForbiddenOutsideRenderPass,

    /// This command is forbidden in a secondary command buffer.
    ForbiddenInSecondaryCommandBuffer,

    /// The queue family doesn't support graphics operations.
    GraphicsOperationsNotSupported,

    /// The queue family doesn't support compute operations.
    ComputeOperationsNotSupported,
}

impl error::Error for CommandAddError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CommandAddError::ForbiddenInsideRenderPass => {
                "this command is forbidden when inside a render pass"
            },
            CommandAddError::ForbiddenOutsideRenderPass => {
                "this command is forbidden when outside of a render pass"
            },
            CommandAddError::ForbiddenInSecondaryCommandBuffer => {
                "this command is forbidden in a secondary command buffer"
            },
            CommandAddError::GraphicsOperationsNotSupported => {
                "the queue family doesn't support graphics operations"
            },
            CommandAddError::ComputeOperationsNotSupported => {
                "the queue family doesn't support compute operations"
            },
        }
    }
}

impl fmt::Display for CommandAddError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
