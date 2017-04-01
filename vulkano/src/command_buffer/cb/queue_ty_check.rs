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
use command_buffer::commands_raw;
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

pass_through!((Rp, F), commands_raw::CmdBeginRenderPass<Rp, F>);
pass_through!((S, Pl), commands_raw::CmdBindDescriptorSets<S, Pl>);
pass_through!((B), commands_raw::CmdBindIndexBuffer<B>);
pass_through!((Pl), commands_raw::CmdBindPipeline<Pl>);
pass_through!((V), commands_raw::CmdBindVertexBuffers<V>);
pass_through!((S, D), commands_raw::CmdBlitImage<S, D>);
pass_through!((), commands_raw::CmdClearAttachments);
pass_through!((S, D), commands_raw::CmdCopyBuffer<S, D>);
pass_through!((S, D), commands_raw::CmdCopyBufferToImage<S, D>);
pass_through!((S, D), commands_raw::CmdCopyImage<S, D>);
pass_through!((), commands_raw::CmdDispatchRaw);
pass_through!((), commands_raw::CmdDrawIndexedRaw);
pass_through!((B), commands_raw::CmdDrawIndirectRaw<B>);
pass_through!((), commands_raw::CmdDrawRaw);
pass_through!((), commands_raw::CmdEndRenderPass);
pass_through!((C), commands_raw::CmdExecuteCommands<C>);
pass_through!((B), commands_raw::CmdFillBuffer<B>);
pass_through!((), commands_raw::CmdNextSubpass);
pass_through!((Pc, Pl), commands_raw::CmdPushConstants<Pc, Pl>);
pass_through!((S, D), commands_raw::CmdResolveImage<S, D>);
pass_through!((), commands_raw::CmdSetEvent);
pass_through!((), commands_raw::CmdSetState);
pass_through!((B, D), commands_raw::CmdUpdateBuffer<B, D>);
