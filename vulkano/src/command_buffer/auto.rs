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

use command_buffer::cb;
use command_buffer::cmd;
use command_buffer::cb::AddCommand;
use command_buffer::cb::CommandBufferBuild;
use command_buffer::CommandBufferBuilder;
use command_buffer::pool::CommandPool;
use command_buffer::pool::StandardCommandPool;
use command_buffer::Submit;
use command_buffer::SubmitBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use instance::QueueFamily;
use OomError;

type Cb<L, P> = cb::DeviceCheckLayer<cb::QueueTyCheckLayer<cb::ContextCheckLayer<cb::StateCacheLayer<cb::SubmitSyncBuilderLayer<cb::AutoPipelineBarriersLayer<cb::UnsafeCommandBufferBuilder<P>, L>>>>>>;

pub struct AutoCommandBufferBuilder<L, P = Arc<StandardCommandPool>> where P: CommandPool {
    inner: Cb<L, P>
}

impl AutoCommandBufferBuilder<Arc<StandardCommandPool>> {
    pub fn new(device: Arc<Device>, queue_family: QueueFamily)
               -> Result<AutoCommandBufferBuilder<(), Arc<StandardCommandPool>>, OomError>
    {
        let pool = Device::standard_command_pool(&device, queue_family);

        let cmd = unsafe {
            let c = try!(cb::UnsafeCommandBufferBuilder::new(pool, cb::Kind::primary(), cb::Flags::SimultaneousUse /* TODO: */));
            let c = cb::BufferedCommandsListLayer::new(c);
            let c = cb::AutoPipelineBarriersLayer::new(c);
            let c = cb::SubmitSyncBuilderLayer::new(c);
            let c = cb::StateCacheLayer::new(c);
            let c = cb::ContextCheckLayer::new(c);
            let c = cb::QueueTyCheckLayer::new(c);
            let c = cb::DeviceCheckLayer::new(c);
            c
        };

        Ok(AutoCommandBufferBuilder {
            inner: cmd,
        })
    }
}

unsafe impl<L, P, O> CommandBufferBuild for AutoCommandBufferBuilder<L, P>
    where Cb<L, P>: CommandBufferBuild<Out = O>,
          P: CommandPool
{
    type Out = O;

    #[inline]
    fn build(self) -> Self::Out {
        // TODO: wrap around?
        CommandBufferBuild::build(self.inner)
    }
}

unsafe impl<L, P> Submit for AutoCommandBufferBuilder<L, P>
    where Cb<L, P>: Submit,
          P: CommandPool
{
    #[inline]
    unsafe fn append_submission<'a>(&'a self, base: SubmitBuilder<'a>, queue: &Arc<Queue>)
                                    -> Result<SubmitBuilder<'a>, Box<Error>>
    {
        self.inner.append_submission(base, queue)
    }
}

unsafe impl<L, P> DeviceOwned for AutoCommandBufferBuilder<L, P>
    where Cb<L, P>: DeviceOwned,
          P: CommandPool
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<L, P> CommandBufferBuilder for AutoCommandBufferBuilder<L, P>
    where Cb<L, P>: CommandBufferBuilder,
          P: CommandPool
{
}

macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<L, P $(, $param)*> AddCommand<$cmd> for AutoCommandBufferBuilder<L, P>
            where P: CommandPool,
                  Cb<L, P>: AddCommand<$cmd, Out = Cb<(L, $cmd), P>>
        {
            type Out = AutoCommandBufferBuilder<(L, $cmd), P>;

            #[inline]
            fn add(self, command: $cmd) -> Self::Out {
                AutoCommandBufferBuilder {
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
pass_through!((), cmd::CmdDrawRaw);
pass_through!((), cmd::CmdEndRenderPass);
pass_through!((C), cmd::CmdExecuteCommands<C>);
pass_through!((B), cmd::CmdFillBuffer<B>);
pass_through!((), cmd::CmdNextSubpass);
pass_through!((Pc, Pl), cmd::CmdPushConstants<Pc, Pl>);
pass_through!((), cmd::CmdSetState);
//pass_through!((B), cmd::CmdUpdateBuffer<B>);
