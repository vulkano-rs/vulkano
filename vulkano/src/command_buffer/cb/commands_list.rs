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

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBuffer;
use command_buffer::cmd;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;

pub struct CommandsListLayer<I, L> {
    inner: I,
    commands: L,
}

///
// TODO: consider changing this to a more flexible API because right now we're forcing
// implementations to hold a tuple of commands
pub unsafe trait CommandsList {
    type List;

    ///
    /// # Safety
    ///
    /// This function is unsafe because the commands must not be modified through
    /// interior mutability.
    unsafe fn list(&self) -> &Self::List;
}

unsafe impl<I, L> CommandsList for CommandsListLayer<I, L> {
    type List = L;

    #[inline]
    unsafe fn list(&self) -> &L {
        &self.commands
    }
}

impl<I> CommandsListLayer<I, ()> {
    #[inline]
    pub fn new(inner: I) -> CommandsListLayer<I, ()> {
        CommandsListLayer {
            inner: inner,
            commands: (),
        }
    }
}

// TODO: implement CommandBufferBuild

unsafe impl<I, L> CommandBuffer for CommandsListLayer<I, L> where I: CommandBuffer {
    type Pool = I::Pool;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<I::Pool> {
        self.inner.inner()
    }
}

unsafe impl<I, L> DeviceOwned for CommandsListLayer<I, L> where I: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I, L> CommandBufferBuilder for CommandsListLayer<I, L> where I: DeviceOwned {
}

macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, L $(, $param)*> AddCommand<$cmd> for CommandsListLayer<I, L>
            where I: for<'r> AddCommand<&'r $cmd, Out = I>
        {
            type Out = CommandsListLayer<I, (L, $cmd)>;

            #[inline]
            fn add(self, command: $cmd) -> Self::Out {
                CommandsListLayer {
                    inner: AddCommand::add(self.inner, &command),
                    commands: (self.commands, command),
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
pass_through!((B, D), cmd::CmdUpdateBuffer<'a, B, D>);
