// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::TrackedBuffer;
use command_buffer::cb::Flags;
use command_buffer::cb::Kind;
use command_buffer::cb::UnsyncedCommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use command_buffer::CommandsListSinkCaller;
use command_buffer::SecondaryCommandBuffer;
use device::Device;
use image::Layout;
use image::TrackedImage;
use sync::AccessFlagBits;
use sync::PipelineStages;
use VulkanObject;
use vk;

use OomError;

pub struct AutobarriersCommandBuffer<L, P> where P: CommandPool {
    // The actual command buffer. 
    inner: UnsyncedCommandBuffer<WrappedCommandsList<L>, P>
}

impl<L, P> AutobarriersCommandBuffer<L, P> where L: CommandsList, P: CommandPool {
    pub fn primary(list: L, pool: P) -> Result<AutobarriersCommandBuffer<L, P>, OomError> {
        let kind = Kind::primary();
        let flags = Flags::SimultaneousUse;

        let cmd = unsafe {
            try!(UnsyncedCommandBuffer::new(WrappedCommandsList(list), pool, kind, flags))
        };

        Ok(AutobarriersCommandBuffer {
            inner: cmd,
        })
    }
}

// TODO: we're not necessarily a secondary command buffer
unsafe impl<L, P> SecondaryCommandBuffer for AutobarriersCommandBuffer<L, P>
    where L: CommandsList, P: CommandPool
{
    #[inline]
    fn inner(&self) -> vk::CommandBuffer {
        self.inner.internal_object()
    }

    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }

    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.inner.commands_list().append(builder);
    }
}

struct WrappedCommandsList<L>(L);
unsafe impl<L> CommandsList for WrappedCommandsList<L> where L: CommandsList {
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        /*let device = pool.device().clone();

        self.0.append(&mut Sink {
            output: builder,
            device: &device,
        });*/
        unimplemented!()
    }
}

// Helper object for AutobarriersCommandBuffer. Implementation detail.
struct Sink<'c: 'o, 'o> {
    output: &'o mut CommandsListSink<'c>,
    device: &'o Arc<Device>,
}

impl<'c: 'o, 'o> CommandsListSink<'c> for Sink<'c, 'o> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device
    }

    #[inline]
    fn add_command(&mut self, f: Box<CommandsListSinkCaller<'c> + 'c>) {
        self.output.add_command(f);
    }

    #[inline]
    fn add_buffer_transition(&mut self, _: &TrackedBuffer, _: usize, _: usize, _: bool,
                             _: PipelineStages, _: AccessFlagBits)
    {
    }

    #[inline]
    fn add_image_transition(&mut self, _: &TrackedImage, _: u32, _: u32, _: u32, _: u32,
                            _: bool, _: Layout, _: PipelineStages, _: AccessFlagBits)
    {
    }

    #[inline]
    fn add_image_transition_notification(&mut self, _: &TrackedImage, _: u32, _: u32, _: u32,
                                         _: u32, _: Layout, _: PipelineStages, _: AccessFlagBits)
    {
    }
}
