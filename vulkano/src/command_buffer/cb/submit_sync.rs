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

use buffer::Buffer;
use command_buffer::cb::AddCommand;
use command_buffer::cb::CommandBufferBuild;
use command_buffer::cb::UnsafeCommandBuffer;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferBuilder;
use command_buffer::cmd;
use image::Image;
use device::Device;
use device::DeviceOwned;
use device::Queue;
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
    buffers: Vec<(Box<Buffer>, bool)>,
    images: Vec<(Box<Image>, bool)>,
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
        where B: Buffer + Clone + 'static
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
        where T: Image + Clone + 'static
    {
        // FIXME: actually implement
        self.images.push((Box::new(image.clone()), exclusive));
    }
}

unsafe impl<I, O> CommandBufferBuild for SubmitSyncBuilderLayer<I>
    where I: CommandBufferBuild<Out = O>
{
    type Out = SubmitSyncLayer<O>;

    #[inline]
    fn build(self) -> Self::Out {
        SubmitSyncLayer {
            inner: self.inner.build(),
            buffers: self.buffers,
            images: self.images,
        }
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
}

// FIXME: implement manually
macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for SubmitSyncBuilderLayer<I>
            where I: AddCommand<$cmd, Out = O>
        {
            type Out = SubmitSyncBuilderLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Self::Out {
                SubmitSyncBuilderLayer {
                    inner: AddCommand::add(self.inner, command),
                    buffers: self.buffers,
                    images: self.images,
                }
            }
        }
    }
}

// FIXME: implement manually
pass_through!((Rp, F), cmd::CmdBeginRenderPass<Rp, F>);
pass_through!((S, Pl), cmd::CmdBindDescriptorSets<S, Pl>);
pass_through!((V), cmd::CmdBindVertexBuffers<V>);
pass_through!((C), cmd::CmdExecuteCommands<C>);

unsafe impl<I, O, B> AddCommand<cmd::CmdBindIndexBuffer<B>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdBindIndexBuffer<B>, Out = O>,
          B: Buffer + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdBindIndexBuffer<B>) -> Self::Out {
        self.add_buffer(command.buffer(), false);

        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O, P> AddCommand<cmd::CmdBindPipeline<P>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdBindPipeline<P>, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: cmd::CmdBindPipeline<P>) -> Self::Out {
        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O, S, D> AddCommand<cmd::CmdBlitImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdBlitImage<S, D>, Out = O>,
          S: Image + Clone + 'static,
          D: Image + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdBlitImage<S, D>) -> Self::Out {
        self.add_image(command.source(), false);
        self.add_image(command.destination(), true);

        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O> AddCommand<cmd::CmdClearAttachments> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdClearAttachments, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: cmd::CmdClearAttachments) -> Self::Out {
        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O, S, D> AddCommand<cmd::CmdCopyBuffer<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdCopyBuffer<S, D>, Out = O>,
          S: Buffer + Clone + 'static,
          D: Buffer + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdCopyBuffer<S, D>) -> Self::Out {
        self.add_buffer(command.source(), false);
        self.add_buffer(command.destination(), true);

        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O, S, D> AddCommand<cmd::CmdCopyBufferToImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdCopyBufferToImage<S, D>, Out = O>,
          S: Buffer + Clone + 'static,
          D: Image + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdCopyBufferToImage<S, D>) -> Self::Out {
        self.add_buffer(command.source(), false);
        self.add_image(command.destination(), true);

        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O, S, D> AddCommand<cmd::CmdCopyImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdCopyImage<S, D>, Out = O>,
          S: Image + Clone + 'static,
          D: Image + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdCopyImage<S, D>) -> Self::Out {
        self.add_image(command.source(), false);
        self.add_image(command.destination(), true);

        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O> AddCommand<cmd::CmdDispatchRaw> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdDispatchRaw, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: cmd::CmdDispatchRaw) -> Self::Out {
        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O> AddCommand<cmd::CmdDrawRaw> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdDrawRaw, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: cmd::CmdDrawRaw) -> Self::Out {
        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O> AddCommand<cmd::CmdEndRenderPass> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdEndRenderPass, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: cmd::CmdEndRenderPass) -> Self::Out {
        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O, B> AddCommand<cmd::CmdFillBuffer<B>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdFillBuffer<B>, Out = O>,
          B: Buffer + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdFillBuffer<B>) -> Self::Out {
        self.add_buffer(command.buffer(), true);

        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O> AddCommand<cmd::CmdNextSubpass> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdNextSubpass, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: cmd::CmdNextSubpass) -> Self::Out {
        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O, Pc, Pl> AddCommand<cmd::CmdPushConstants<Pc, Pl>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdPushConstants<Pc, Pl>, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: cmd::CmdPushConstants<Pc, Pl>) -> Self::Out {
        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O, S, D> AddCommand<cmd::CmdResolveImage<S, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdResolveImage<S, D>, Out = O>,
          S: Image + Clone + 'static,
          D: Image + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdResolveImage<S, D>) -> Self::Out {
        self.add_image(command.source(), false);
        self.add_image(command.destination(), true);

        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O> AddCommand<cmd::CmdSetEvent> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdSetEvent, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: cmd::CmdSetEvent) -> Self::Out {
        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<I, O> AddCommand<cmd::CmdSetState> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdSetState, Out = O>
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(self, command: cmd::CmdSetState) -> Self::Out {
        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

unsafe impl<'a, I, O, B, D> AddCommand<cmd::CmdUpdateBuffer<'a, B, D>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdUpdateBuffer<'a, B, D>, Out = O>,
          B: Buffer + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdUpdateBuffer<'a, B, D>) -> Self::Out {
        self.add_buffer(command.buffer(), true);

        SubmitSyncBuilderLayer {
            inner: AddCommand::add(self.inner, command),
            buffers: self.buffers,
            images: self.images,
        }
    }
}

/// Layer around a command buffer that handles synchronization between command buffers.
pub struct SubmitSyncLayer<I> {
    inner: I,
    buffers: Vec<(Box<Buffer>, bool)>,
    images: Vec<(Box<Image>, bool)>,
}

unsafe impl<I> CommandBuffer for SubmitSyncLayer<I> where I: CommandBuffer {
    type Pool = I::Pool;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<I::Pool> {
        self.inner.inner()
    }

    fn submit_check(&self, future: &GpuFuture, queue: &Queue) -> Result<(), Box<Error>> {
        for &(ref buffer, exclusive) in self.buffers.iter() {
            if future.check_buffer_access(buffer, exclusive, queue) {
                continue;
            }

            if !buffer.gpu_access(exclusive, queue) {
                panic!()    // FIXME: return Err();
            }
        }

        for &(ref image, exclusive) in self.images.iter() {
            if future.check_image_access(image, exclusive, queue) {
                continue;
            }

            if !image.gpu_access(exclusive, queue) {
                panic!()    // FIXME: return Err();
            }
        }

        // FIXME: pipeline barriers if necessary

        Ok(())
    }
}

unsafe impl<I> DeviceOwned for SubmitSyncLayer<I> where I: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}
