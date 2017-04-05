// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::BufferAccess;
use buffer::TypedBuffer;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use pipeline::input_assembly::Index;
use VulkanObject;
use VulkanPointers;
use vk;

/// Command that binds an index buffer to a command buffer.
pub struct CmdBindIndexBuffer<B> {
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

impl<B, I> CmdBindIndexBuffer<B>
    where B: BufferAccess + TypedBuffer<Content = [I]>,
          I: Index + 'static
{
    /// Builds the command.
    #[inline]
    pub fn new(buffer: B) -> CmdBindIndexBuffer<B> {
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
            raw_buffer: raw_buffer,
            offset: offset,
            index_type: I::ty() as vk::IndexType,
            device: device,
            buffer: buffer,
        }
    }
}

impl<B> CmdBindIndexBuffer<B> {
    /// Returns the index buffer to bind.
    #[inline]
    pub fn buffer(&self) -> &B {
        &self.buffer
    }
}

unsafe impl<B> DeviceOwned for CmdBindIndexBuffer<B>
    where B: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

unsafe impl<'a, P, B> AddCommand<&'a CmdBindIndexBuffer<B>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdBindIndexBuffer<B>) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdBindIndexBuffer(cmd, command.raw_buffer, command.offset, command.index_type);
        }

        self
    }
}
