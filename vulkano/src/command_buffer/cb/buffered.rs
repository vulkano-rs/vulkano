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
use command_buffer::cb::CommandBufferBuild;
use command_buffer::cb::CommandsList;
use command_buffer::CommandBufferBuilder;
use command_buffer::CommandBufferBuilderBuffered;
use command_buffer::cmd;
use command_buffer::Submit;
use command_buffer::SubmitBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;

/// Layer around a command buffer builder or a command buffer that stores the commands and has a
/// buffering mechanism.
///
/// Whenever you add a command (with the `AddCommand` trait), the command is not immediately added
/// to the underlying builder. Pending commands are added for real when you call `flush()` or when
/// you build the builder into a real command buffer.
///
/// The purpose of this buffering mechanism is to allow inserting pipeline barrier commands between
/// commands that are already submitted and commands that are pending thanks to the
/// `add_non_buffered_pipeline_barrier` method.
pub struct BufferedCommandsListLayer<I, L> {
    inner: Option<I>,
    commands: L,
    // Number of commands in the list that haven't been flushed. Since the latest commands appear
    // first in the list, it is more convenient to store the number of commands that haven't been
    // flushed rather than the number of commands that have been flushed.
    non_flushed: u32,
}

/// Helper trait for `BufferedCommandsListLayer`.
///
/// Whenever you manipulate a `BufferedCommandsListLayer<I, L>`, the template parameter `L` should
/// implement `BufferedCommandsListLayerCommands<I>`.
pub unsafe trait BufferedCommandsListLayerCommands<I> {
    /// Sends the `num` last commands of the list to `dest`.
    fn flush(&self, num: u32, dest: I) -> I;
}

unsafe impl<I> BufferedCommandsListLayerCommands<I> for () {
    #[inline]
    fn flush(&self, num: u32, dest: I) -> I {
        debug_assert_eq!(num, 0);
        dest
    }
}

unsafe impl<I, L, C> BufferedCommandsListLayerCommands<I> for (L, C)
    where I: for<'r> AddCommand<&'r C, Out = I>,
          L: BufferedCommandsListLayerCommands<I>,
{
    #[inline]
    fn flush(&self, num: u32, dest: I) -> I {
        if num == 0 {
            dest
        } else {
            self.0.flush(num - 1, dest).add(&self.1)
        }
    }
}

unsafe impl<I, L> CommandsList for BufferedCommandsListLayer<I, L> {
    type List = L;

    #[inline]
    unsafe fn list(&self) -> &L {
        &self.commands
    }
}

impl<I> BufferedCommandsListLayer<I, ()> {
    /// Builds a new `BufferedCommandsListLayer`.
    #[inline]
    pub fn new(inner: I) -> BufferedCommandsListLayer<I, ()> {
        BufferedCommandsListLayer {
            inner: Some(inner),
            commands: (),
            non_flushed: 0,
        }
    }
}

impl<I, L> BufferedCommandsListLayer<I, L>
    where L: BufferedCommandsListLayerCommands<I>,
{
    #[inline]
    fn flush_inner(&mut self) {
        let inner = self.inner.take().unwrap();
        self.inner = Some(self.commands.flush(self.non_flushed, inner));
        self.non_flushed = 0;
    }
}

unsafe impl<I, L> CommandBufferBuilderBuffered for BufferedCommandsListLayer<I, L>
    where I: for<'r> AddCommand<&'r cmd::CmdPipelineBarrier<'r>, Out = I>,
          L: BufferedCommandsListLayerCommands<I>,
{
    #[inline]
    fn add_non_buffered_pipeline_barrier(&mut self, cmd: &cmd::CmdPipelineBarrier) {
        let inner = self.inner.take().unwrap();
        self.inner = Some(inner.add(cmd));
    }

    #[inline]
    fn flush(&mut self) {
        self.flush_inner();
    }
}

unsafe impl<I, L, O> CommandBufferBuild for BufferedCommandsListLayer<I, L>
    where I: CommandBufferBuild<Out = O>,
          L: BufferedCommandsListLayerCommands<I>       // Necessary in order to flush
{
    type Out = BufferedCommandsListLayer<O, L>;

    #[inline]
    fn build(mut self) -> Self::Out {
        self.flush_inner();
        debug_assert_eq!(self.non_flushed, 0);

        let inner = self.inner.take().unwrap().build();

        BufferedCommandsListLayer {
            inner: Some(inner),
            commands: self.commands,
            non_flushed: 0,
        }
    }
}

unsafe impl<I, L> Submit for BufferedCommandsListLayer<I, L> where I: Submit {
    #[inline]
    unsafe fn append_submission<'a>(&'a self, base: SubmitBuilder<'a>, queue: &Arc<Queue>)
                                    -> Result<SubmitBuilder<'a>, Box<Error>>
    {
        self.inner.as_ref().unwrap().append_submission(base, queue)
    }
}

unsafe impl<I, L> DeviceOwned for BufferedCommandsListLayer<I, L> where I: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.as_ref().unwrap().device()
    }
}

unsafe impl<I, L> CommandBufferBuilder for BufferedCommandsListLayer<I, L> where I: DeviceOwned {
}

macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, L $(, $param)*> AddCommand<$cmd> for BufferedCommandsListLayer<I, L>
            where I: for<'r> AddCommand<&'r $cmd, Out = I>
        {
            type Out = BufferedCommandsListLayer<I, (L, $cmd)>;

            #[inline]
            fn add(self, command: $cmd) -> Self::Out {
                debug_assert!(self.inner.is_some());
                BufferedCommandsListLayer {
                    inner: self.inner,
                    commands: (self.commands, command),
                    non_flushed: self.non_flushed + 1,
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
