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
    #[inline]
    pub fn new(inner: I) -> SubmitSyncBuilderLayer<I> {
        SubmitSyncBuilderLayer {
            inner: inner,
            buffers: Vec::new(),
            images: Vec::new(),
        }
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

pass_through!((Rp, F), cmd::CmdBeginRenderPass<Rp, F>);
pass_through!((S, Pl), cmd::CmdBindDescriptorSets<S, Pl>);
pass_through!((Pl), cmd::CmdBindPipeline<Pl>);
pass_through!((V), cmd::CmdBindVertexBuffers<V>);
pass_through!((S, D), cmd::CmdBlitImage<S, D>);
pass_through!((), cmd::CmdClearAttachments);
pass_through!((S, D), cmd::CmdCopyBuffer<S, D>);
pass_through!((S, D), cmd::CmdCopyBufferToImage<S, D>);
pass_through!((S, D), cmd::CmdCopyImage<S, D>);
pass_through!((), cmd::CmdDispatchRaw);
pass_through!((), cmd::CmdDrawRaw);
pass_through!((), cmd::CmdEndRenderPass);
pass_through!((C), cmd::CmdExecuteCommands<C>);
pass_through!((B), cmd::CmdFillBuffer<B>);
pass_through!((), cmd::CmdNextSubpass);
pass_through!((Pc, Pl), cmd::CmdPushConstants<Pc, Pl>);
pass_through!((S, D), cmd::CmdResolveImage<S, D>);
pass_through!((), cmd::CmdSetEvent);
pass_through!((B, D), cmd::CmdUpdateBuffer<'a, B, D>);

unsafe impl<I, O, B> AddCommand<cmd::CmdBindIndexBuffer<B>> for SubmitSyncBuilderLayer<I>
    where I: AddCommand<cmd::CmdBindIndexBuffer<B>, Out = O>,
          B: Buffer + Clone + 'static
{
    type Out = SubmitSyncBuilderLayer<O>;

    #[inline]
    fn add(mut self, command: cmd::CmdBindIndexBuffer<B>) -> Self::Out {
        self.buffers.push((Box::new(command.buffer().clone()), false));

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
