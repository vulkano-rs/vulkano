// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::any::Any;
use std::error::Error;
use std::sync::Arc;

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBuffer;
use command_buffer::cmd;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;

/// Layer that stores commands in an abstract way.
pub struct AbstractStorageLayer<I> {
    inner: I,
    commands: Vec<Box<Any>>,
}

unsafe impl<I> CommandBuffer for AbstractStorageLayer<I> where I: CommandBuffer {
    type Pool = I::Pool;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<I::Pool> {
        self.inner.inner()
    }
}

unsafe impl<I> DeviceOwned for AbstractStorageLayer<I> where I: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I> CommandBufferBuilder for AbstractStorageLayer<I> where I: DeviceOwned {
}

macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<I $(, $param)*> AddCommand<$cmd> for AbstractStorageLayer<I>
            where I: for<'r> AddCommand<&'r $cmd, Out = I>, $cmd: 'static
        {
            type Out = AbstractStorageLayer<I>;

            #[inline]
            fn add(mut self, command: $cmd) -> Self::Out {
                let new_inner = AddCommand::add(self.inner, &command);
                // TODO: should store a lightweight version of the command
                self.commands.push(Box::new(command) as Box<_>);
                
                AbstractStorageLayer {
                    inner: new_inner,
                    commands: self.commands,
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
pass_through!((S, D), cmd::CmdBlitImage<S, D>);
pass_through!((), cmd::CmdClearAttachments);
pass_through!((S, D), cmd::CmdCopyBuffer<S, D>);
pass_through!((S, D), cmd::CmdCopyImage<S, D>);
pass_through!((), cmd::CmdDispatchRaw);
pass_through!((), cmd::CmdDrawIndexedRaw);
pass_through!((B), cmd::CmdDrawIndirectRaw<B>);
pass_through!((), cmd::CmdDrawRaw);
pass_through!((), cmd::CmdEndRenderPass);
pass_through!((C), cmd::CmdExecuteCommands<C>);
pass_through!((B), cmd::CmdFillBuffer<B>);
pass_through!((), cmd::CmdNextSubpass);
pass_through!((Pc, Pl), cmd::CmdPushConstants<Pc, Pl>);
pass_through!((S, D), cmd::CmdResolveImage<S, D>);
pass_through!((), cmd::CmdSetEvent);
pass_through!((), cmd::CmdSetState);
pass_through!((B, D), cmd::CmdUpdateBuffer<'static, B, D>);
