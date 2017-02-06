// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use command_buffer::cb::AddCommand;
use command_buffer::cb::CommandBufferBuild;
use command_buffer::CommandBufferBuilder;
use command_buffer::cmd;
use device::Device;
use device::DeviceOwned;

/// Layer around a command buffer builder that checks whether the commands added to it match the
/// type of the queue family of the underlying builder.
///
/// Commands that perform graphical or compute operations can only be executed on queue families
/// that support graphical or compute operations. This is what this layer verifies.
pub struct QueueTyCheckLayer<I> {
    inner: I,
}

impl<I> QueueTyCheckLayer<I> {
    /// Builds a new `QueueTyCheckLayer`.
    #[inline]
    pub fn new(inner: I) -> QueueTyCheckLayer<I> {
        QueueTyCheckLayer {
            inner: inner,
        }
    }

    /// Destroys the layer and returns the underlying command buffer.
    #[inline]
    pub fn into_inner(self) -> I {
        self.inner
    }
}

unsafe impl<I> DeviceOwned for QueueTyCheckLayer<I>
    where I: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I> CommandBufferBuilder for QueueTyCheckLayer<I>
    where I: CommandBufferBuilder
{
}

unsafe impl<I, O> CommandBufferBuild for QueueTyCheckLayer<I>
    where I: CommandBufferBuild<Out = O>
{
    type Out = O;

    #[inline]
    fn build(self) -> O {
        self.inner.build()
    }
}

// TODO: actually implement

macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for QueueTyCheckLayer<I>
            where I: AddCommand<$cmd, Out = O>
        {
            type Out = QueueTyCheckLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Self::Out {
                QueueTyCheckLayer {
                    inner: self.inner.add(command),
                }
            }
        }
    }
}

pass_through!((Rp, F), cmd::CmdBeginRenderPass<Rp, F>);
pass_through!((S, Pl), cmd::CmdBindDescriptorSets<S, Pl>);
pass_through!((B), cmd::CmdBindIndexBuffer<B>);
pass_through!((Pl), cmd::CmdBindPipeline<Pl>);
pass_through!((V), cmd::CmdBindVertexBuffers<V>);
pass_through!((), cmd::CmdClearAttachments);
pass_through!((S, D), cmd::CmdCopyBuffer<S, D>);
pass_through!((), cmd::CmdDispatchRaw);
pass_through!((), cmd::CmdDrawIndexedRaw);
pass_through!((B), cmd::CmdDrawIndirectRaw<B>);
pass_through!((), cmd::CmdDrawRaw);
pass_through!((), cmd::CmdEndRenderPass);
pass_through!((C), cmd::CmdExecuteCommands<C>);
pass_through!((B), cmd::CmdFillBuffer<B>);
pass_through!((), cmd::CmdNextSubpass);
pass_through!((Pc, Pl), cmd::CmdPushConstants<Pc, Pl>);
pass_through!((), cmd::CmdSetState);
pass_through!((B, D), cmd::CmdUpdateBuffer<'a, B, D>);
