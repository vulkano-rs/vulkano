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

use buffer::BufferAccess;
use command_buffer::cb::AddCommand;
use command_buffer::cb::CommandBufferBuild;
use command_buffer::cb::UnsafeCommandBuffer;
use command_buffer::commands_raw;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use sync::AccessFlagBits;
use sync::GpuFuture;
use sync::PipelineStages;

/// Layer that stores commands in an abstract way.
pub struct AbstractStorageLayer<I> {
    inner: I,
    commands: Vec<Box<Any + Send + Sync>>,
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

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        self.inner.check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        self.inner.check_image_access(image, exclusive, queue)
    }
}

unsafe impl<I> DeviceOwned for AbstractStorageLayer<I> where I: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<I, O, E> CommandBufferBuild for AbstractStorageLayer<I>
    where I: CommandBufferBuild<Out = O, Err = E>
{
    type Out = AbstractStorageLayer<O>;
    type Err = E;

    #[inline]
    fn build(self) -> Result<Self::Out, E> {
        let inner = try!(self.inner.build());

        Ok(AbstractStorageLayer {
            inner: inner,
            commands: self.commands,
        })
    }
}

unsafe impl<I> CommandBufferBuilder for AbstractStorageLayer<I> where I: DeviceOwned {
}

macro_rules! pass_through {
    (($($param:ident),*), $cmd:ty) => {
        unsafe impl<I $(, $param)*> AddCommand<$cmd> for AbstractStorageLayer<I>
            where I: for<'r> AddCommand<&'r $cmd, Out = I>, $cmd: Send + Sync + 'static
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
