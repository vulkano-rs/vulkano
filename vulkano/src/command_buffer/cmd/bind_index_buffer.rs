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
use buffer::TypedBuffer;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use device::Device;
use pipeline::input_assembly::Index;
use sync::AccessFlagBits;
use sync::PipelineStages;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds a command that binds an index buffer at the end of it.
pub struct CmdBindIndexBuffer<L, B> where L: CommandsList {
    // Parent commands list.
    previous: L,
    // Raw handle of the buffer to bind.
    raw_buffer: vk::Buffer,
    // Raw offset of the buffer to bind.
    offset: vk::DeviceSize,
    // Type of index.
    index_type: vk::IndexType,
    // The device of the buffer, so that we can compare it with the command buffer's device.
    device: Arc<Device>,
    // The buffer to bind. Unused, but we need to keep it alive.
    buffer: B,
}

impl<L, B, I> CmdBindIndexBuffer<L, B>
    where L: CommandsList,
          B: TrackedBuffer + TypedBuffer<Content = [I]>,
          I: Index + 'static
{
    /// Builds the command.
    #[inline]
    pub fn new(previous: L, buffer: B) -> CmdBindIndexBuffer<L, B> {
        let device;
        let raw_buffer;
        let offset;

        {
            let inner = buffer.inner();
            debug_assert!(inner.offset < inner.buffer.size());
            // TODO: check > The sum of offset and the address of the range of VkDeviceMemory object that is backing buffer, must be a multiple of the type indicated by indexType
            assert!(inner.buffer.usage_index_buffer());     // TODO: error
            device = inner.buffer.device().clone();
            raw_buffer = inner.buffer.internal_object();
            offset = inner.offset as vk::DeviceSize;
        }

        CmdBindIndexBuffer {
            previous: previous,
            raw_buffer: raw_buffer,
            offset: offset,
            index_type: I::ty() as vk::IndexType,
            device: device,
            buffer: buffer,
        }
    }
}

unsafe impl<L, B> CommandsList for CmdBindIndexBuffer<L, B>
    where L: CommandsList, B: TrackedBuffer
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        assert_eq!(self.device.internal_object(), builder.device().internal_object());

        {
            let stages = PipelineStages { vertex_input: true, .. PipelineStages::none() };
            let access = AccessFlagBits { index_read: true, .. AccessFlagBits::none() };
            builder.add_buffer_transition(&self.buffer, 0, self.buffer.size(), false,
                                          stages, access);
        }

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            let params = (self.raw_buffer, self.offset, self.index_type);
            if raw.bound_index_buffer == params {
                return;
            }

            raw.bound_index_buffer = params;

            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();
                vk.CmdBindIndexBuffer(cmd, self.raw_buffer, self.offset, self.index_type);
            }
        }));
    }
}
