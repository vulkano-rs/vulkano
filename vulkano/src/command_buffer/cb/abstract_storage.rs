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
use command_buffer::cb::CommandBufferBuild;
use command_buffer::cb::UnsafeCommandBuffer;
use command_buffer::cmd;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use sync::GpuFuture;

/// Layer that stores commands in an abstract way.
pub struct AbstractStorageLayer<I> {
    inner: I,
    commands: Vec<Box<Any>>,
}

impl<I> AbstractStorageLayer<I> {
    /// Builds a new `AbstractStorageLayer`.
    #[inline]
    pub fn new(inner: I) -> AbstractStorageLayer<I> {
        AbstractStorageLayer {
            inner: inner,
            commands: Vec::new(),
        }
    }
}

unsafe impl<I> CommandBuffer for AbstractStorageLayer<I> where I: CommandBuffer {
    type Pool = I::Pool;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<I::Pool> {
        self.inner.inner()
    }

    #[inline]
    fn submit_check(&self, future: &GpuFuture, queue: &Queue) -> Result<(), Box<Error>> {
        self.inner.submit_check(future, queue)
    }
}

unsafe impl<I> DeviceOwned for AbstractStorageLayer<I> where I: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I, O> CommandBufferBuild for AbstractStorageLayer<I>
    where I: CommandBufferBuild<Out = O>
{
    type Out = AbstractStorageLayer<O>;

    #[inline]
    fn build(mut self) -> Self::Out {
        let inner = self.inner.build();

        AbstractStorageLayer {
            inner: inner,
            commands: self.commands,
        }
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
pass_through!((S, D), cmd::CmdCopyBufferToImage<S, D>);
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
