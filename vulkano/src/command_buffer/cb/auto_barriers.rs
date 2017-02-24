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

pub struct AutoPipelineBarriersLayer<I> {
    inner: I,
}

impl<I> AutoPipelineBarriersLayer<I> {
    #[inline]
    pub fn new(inner: I) -> AutoPipelineBarriersLayer<I> {
        AutoPipelineBarriersLayer {
            inner: inner,
        }
    }
}

/*unsafe impl<C, I, L> AddCommand<C> for AutoPipelineBarriersLayer<I, L>
    where I: for<'r> AddCommand<&'r C, Out = I>
{
    type Out = AutoPipelineBarriersLayer<I, (L, C)>;

    #[inline]
    fn add(self, command: C) -> Self::Out {
        AutoPipelineBarriersLayer {
            inner: AddCommand::add(self.inner, command),
        }
    }
}*/

unsafe impl<I, O> CommandBufferBuild for AutoPipelineBarriersLayer<I>
    where I: CommandBufferBuild<Out = O>
{
    type Out = O;

    #[inline]
    fn build(self) -> O {
        self.inner.build()
    }
}

unsafe impl<I> DeviceOwned for AutoPipelineBarriersLayer<I>
    where I: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I> CommandBufferBuilder for AutoPipelineBarriersLayer<I>
    where I: CommandBufferBuilder
{
}

macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<I, O $(, $param)*> AddCommand<$cmd> for AutoPipelineBarriersLayer<I>
            where I: for<'r> AddCommand<$cmd, Out = O>
        {
            type Out = AutoPipelineBarriersLayer<O>;

            #[inline]
            fn add(self, command: $cmd) -> Self::Out {
                AutoPipelineBarriersLayer {
                    inner: AddCommand::add(self.inner, command),
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
pass_through!((), cmd::CmdDrawRaw);
pass_through!((), cmd::CmdEndRenderPass);
pass_through!((C), cmd::CmdExecuteCommands<C>);
pass_through!((B), cmd::CmdFillBuffer<B>);
pass_through!((), cmd::CmdNextSubpass);
pass_through!((Pc, Pl), cmd::CmdPushConstants<Pc, Pl>);
pass_through!((S, D), cmd::CmdResolveImage<S, D>);
pass_through!((), cmd::CmdSetEvent);
pass_through!((), cmd::CmdSetState);
//pass_through!((B), cmd::CmdUpdateBuffer<B>);

