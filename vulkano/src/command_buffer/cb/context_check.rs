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

/// Layer around a command buffer builder that checks whether the commands can be executed in the
/// given context.
///
/// "The given context" here means being inside/outside a render pass or a secondary command
/// buffer.
pub struct ContextCheckLayer<I> {
    inner: I,
}

impl<I> ContextCheckLayer<I> {
    /// Builds a new `ContextCheckLayer`.
    #[inline]
    pub fn new(inner: I) -> ContextCheckLayer<I> {
        ContextCheckLayer {
            inner: inner,
        }
    }

    /// Destroys the layer and returns the underlying command buffer.
    #[inline]
    pub fn into_inner(self) -> I {
        self.inner
    }
}

unsafe impl<I, O, E> CommandBufferBuild for ContextCheckLayer<I>
    where I: CommandBufferBuild<Out = O, Err = E>
{
    type Out = O;
    type Err = E;

    #[inline]
    fn build(self) -> Result<O, E> {
        self.inner.build()
    }
}

unsafe impl<I> DeviceOwned for ContextCheckLayer<I>
    where I: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I> CommandBufferBuilder for ContextCheckLayer<I>
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

// TODO: actually implement

macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<'a, I, O $(, $param)*> AddCommand<$cmd> for ContextCheckLayer<I>
            where I: AddCommand<$cmd, Out = O>
        {
            type Out = ContextCheckLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Self::Out {
                ContextCheckLayer {
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
