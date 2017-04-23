// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error::Error;
use std::sync::Arc;

use buffer::BufferAccess;
use command_buffer::cb::AddCommand;
use command_buffer::cb::CommandBufferBuild;
use command_buffer::cb::UnsafeCommandBuffer;
use command_buffer::CommandAddError;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferBuilder;
use command_buffer::commands_raw;
use image::ImageAccess;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use sync::AccessFlagBits;
use sync::PipelineStages;
use sync::GpuFuture;

/// Layers that ensures that synchronization with buffers and images between command buffers is
/// properly handled.
///
/// The following are handled:
///
/// - Return an error when submitting if the user didn't provide the guarantees for proper
///   synchronization.
///
/// - Automatically generate pipeline barriers between command buffers if necessary to handle
///   the transition between command buffers.
///
pub struct SubmitSyncBuilderLayer<I> {
    inner: I,
    buffers: Vec<(Box<BufferAccess + Send + Sync>, bool)>,
    images: Vec<(Box<ImageAccess + Send + Sync>, bool)>,
}

impl<I> SubmitSyncBuilderLayer<I> {
    /// Builds a new layer that wraps around an existing builder.
    #[inline]
    pub fn new(inner: I) -> SubmitSyncBuilderLayer<I> {
        SubmitSyncBuilderLayer {
            inner: inner,
            buffers: Vec::new(),
            images: Vec::new(),
        }
    }

    // Adds a buffer to the list.
    fn add_buffer<B>(&mut self, buffer: &B, exclusive: bool)
        where B: BufferAccess + Send + Sync + Clone + 'static
    {
        for &mut (ref existing_buf, ref mut existing_exclusive) in self.buffers.iter_mut() {
            if existing_buf.conflicts_buffer(0, existing_buf.size(), buffer, 0, buffer.size()) {
                *existing_exclusive = *existing_exclusive || exclusive;
                return;
            }
        }

        // FIXME: compare with images as well

        self.buffers.push((Box::new(buffer.clone()), exclusive));
    }

    // Adds an image to the list.
    fn add_image<T>(&mut self, image: &T, exclusive: bool)
        where T: ImageAccess + Send + Sync + Clone + 'static
    {
        // FIXME: actually implement
        self.images.push((Box::new(image.clone()), exclusive));
    }
}

unsafe impl<I, O, E> CommandBufferBuild for SubmitSyncBuilderLayer<I>
    where I: CommandBufferBuild<Out = O, Err = E>
{
    type Out = SubmitSyncLayer<O>;
    type Err = E;

    #[inline]
    fn build(self) -> Result<Self::Out, E> {
        Ok(SubmitSyncLayer {
            inner: try!(self.inner.build()),
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I> DeviceOwned for SubmitSyncBuilderLayer<I>
    where I: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I> CommandBufferBuilder for SubmitSyncBuilderLayer<I>
    where I: CommandBufferBuilder
{
    #[inline]
    fn supports_graphics(&self) -> bool {
        self.inner.supports_graphics()
    }

    #[inline]
    fn supports_compute(&self) -> bool {
        self.inner.supports_compute()
    }
}

// FIXME: implement manually
macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for SubmitSyncBuilderLayer<I>
            where I: AddCommand<$cmd, Out = O>
        {
            type Out = SubmitSyncBuilderLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Result<Self::Out, CommandAddError> {
                Ok(SubmitSyncBuilderLayer {
                    inner: AddCommand::add(self.inner, command)?,
                    buffers: self.buffers,
                    images: self.images,
                })
            }
        }
    }
}

// FIXME: implement manually
pass_through!((Rp, F), commands_raw::CmdBeginRenderPass<Rp, F>);
pass_through!((S, Pl), commands_raw::CmdBindDescriptorSets<S, Pl>);
pass_through!((V), commands_raw::CmdBindVertexBuffers<V>);
pass_through!((C), commands_raw::CmdExecuteCommands<C>);

unsafe impl<I, O, B> AddCommand<commands_raw::CmdBindIndexBuffer<B>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdBindIndexBuffer<B>, Out = O>,
          B: BufferAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdBindIndexBuffer<B>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.buffer(), false);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O, P> AddCommand<commands_raw::CmdBindPipeline<P>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdBindPipeline<P>, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdBindPipeline<P>) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O, S, D> AddCommand<commands_raw::CmdBlitImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdBlitImage<S, D>, Out = O>,
          S: ImageAccess + Send + Sync + Clone + 'static,
          D: ImageAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdBlitImage<S, D>) -> Result<Self::Out, CommandAddError> {
        self.add_image(command.source(), false);
        self.add_image(command.destination(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdClearAttachments> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdClearAttachments, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdClearAttachments) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O, S, D> AddCommand<commands_raw::CmdCopyBuffer<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdCopyBuffer<S, D>, Out = O>,
          S: BufferAccess + Send + Sync + Clone + 'static,
          D: BufferAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdCopyBuffer<S, D>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.source(), false);
        self.add_buffer(command.destination(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O, S, D> AddCommand<commands_raw::CmdCopyBufferToImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdCopyBufferToImage<S, D>, Out = O>,
          S: BufferAccess + Send + Sync + Clone + 'static,
          D: ImageAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdCopyBufferToImage<S, D>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.source(), false);
        self.add_image(command.destination(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O, S, D> AddCommand<commands_raw::CmdCopyImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdCopyImage<S, D>, Out = O>,
          S: ImageAccess + Send + Sync + Clone + 'static,
          D: ImageAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdCopyImage<S, D>) -> Result<Self::Out, CommandAddError> {
        self.add_image(command.source(), false);
        self.add_image(command.destination(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdDispatchRaw> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdDispatchRaw, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdDispatchRaw) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdDrawRaw> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdDrawRaw, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdDrawRaw) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdDrawIndexedRaw> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdDrawIndexedRaw, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdDrawIndexedRaw) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O, B> AddCommand<commands_raw::CmdDrawIndirectRaw<B>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdDrawIndirectRaw<B>, Out = O>,
          B: BufferAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdDrawIndirectRaw<B>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.buffer(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdEndRenderPass> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdEndRenderPass, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdEndRenderPass) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O, B> AddCommand<commands_raw::CmdFillBuffer<B>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdFillBuffer<B>, Out = O>,
          B: BufferAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdFillBuffer<B>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.buffer(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdNextSubpass> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdNextSubpass, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdNextSubpass) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O, Pc, Pl> AddCommand<commands_raw::CmdPushConstants<Pc, Pl>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdPushConstants<Pc, Pl>, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdPushConstants<Pc, Pl>) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O, S, D> AddCommand<commands_raw::CmdResolveImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdResolveImage<S, D>, Out = O>,
          S: ImageAccess + Send + Sync + Clone + 'static,
          D: ImageAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdResolveImage<S, D>) -> Result<Self::Out, CommandAddError> {
        self.add_image(command.source(), false);
        self.add_image(command.destination(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdSetEvent> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdSetEvent, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdSetEvent) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O> AddCommand<commands_raw::CmdSetState> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdSetState, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: commands_raw::CmdSetState) -> Result<Self::Out, CommandAddError> {
        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

unsafe impl<I, O, B, D> AddCommand<commands_raw::CmdUpdateBuffer<B, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<commands_raw::CmdUpdateBuffer<B, D>, Out = O>,
          B: BufferAccess + Send + Sync + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdUpdateBuffer<B, D>) -> Result<Self::Out, CommandAddError> {
        self.add_buffer(command.buffer(), true);

        Ok(SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command)?,
            buffers: self.buffers,
            images: self.images,
        })
    }
}

/// Layer around a command buffer that handles synchronization between command buffers.
pub struct SubmitSyncLayer<I> {
    inner: I,
    buffers: Vec<(Box<BufferAccess + Send + Sync>, bool)>,
    images: Vec<(Box<ImageAccess + Send + Sync>, bool)>,
}

unsafe impl<I> CommandBuffer for SubmitSyncLayer<I> where I: CommandBuffer {
    type Pool = I::Pool;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<I::Pool> {
        self.inner.inner()
    }

    fn submit_check(&self, future: &GpuFuture, queue: &Queue) -> Result<(), Box<Error>> {
        for &(ref buffer, exclusive) in self.buffers.iter() {
            if future.check_buffer_access(buffer, exclusive, queue).is_ok() {
                unsafe { buffer.increase_gpu_lock(); }
                continue;
            }

            if !buffer.try_gpu_lock(exclusive, queue) {
                panic!()    // FIXME: return Err();
            }
        }

        for &(ref image, exclusive) in self.images.iter() {
            if future.check_image_access(image, exclusive, queue).is_ok() {
                unsafe { image.increase_gpu_lock(); }
                continue;
            }

            if !image.try_gpu_lock(exclusive, queue) {
                panic!()    // FIXME: return Err();
            }
        }

        // FIXME: pipeline barriers if necessary

        Ok(())
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        // FIXME: implement
        Err(())
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        // FIXME: implement
        Err(())
    }
}

unsafe impl<I> DeviceOwned for SubmitSyncLayer<I> where I: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}
