// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::pool::CommandPool;
use command_buffer::sys::KeepAlive;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use pipeline::input_assembly::Index;

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a command that binds an index buffer.
pub struct IndexBufferBindCommand {
    // Buffer to keep alive.
    keep_alive: Arc<KeepAlive + 'static>,

    // The device of the buffer, or 0 if the list of buffers is empty.
    device: vk::Device,

    // List of parameters for `vkCmdBindIndexBuffer`.
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    index_type: vk::IndexType,
}

impl IndexBufferBindCommand {
    /// Builds a command that bind an index buffer to a command buffer.
    pub fn new<'a, S, I, B>(buffer: S) -> Result<IndexBufferBindCommand, IndexBufferBindError>
        where S: Into<BufferSlice<'a, [I], B>>,
              I: 'static + Index,
              B: Buffer
    {
        let buffer = buffer.into();

        if !buffer.buffer().inner_buffer().usage_index_buffer() {
            return Err(IndexBufferBindError::WrongUsageFlag);
        }

        // FIXME: > The sum of offset, and the address of the range of VkDeviceMemory object that’s
        //          backing buffer, must be a multiple of the type indicated by indexType

        Ok(IndexBufferBindCommand {
            keep_alive: buffer.buffer().clone() as Arc<_>,
            device: buffer.buffer().inner_buffer().device().internal_object(),
            buffer: buffer.buffer().inner_buffer().internal_object(),
            offset: buffer.offset() as vk::DeviceSize,
            index_type: <I as Index>::ty() as u32,
        })
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the buffer was not allocated with the same device as the command buffer.
    /// - Panicks if the queue doesn't not support graphics operations.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            let _pool_lock = cb.pool().lock();

            // Various checks.
            // Note that surprisingly the specs allow this function to be called from outside a
            // render pass.
            assert_eq!(self.device, cb.device().internal_object());
            assert!(cb.pool().queue_family().supports_graphics());

            // Nothing to do.
            let ref_value = Some((self.buffer, self.offset, self.index_type));
            if cb.current_index_buffer == ref_value {
                return cb;
            }

            cb.keep_alive.push(self.keep_alive.clone());
            cb.current_index_buffer = ref_value;

            // Now binding.
            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();

                vk.CmdBindIndexBuffer(cmd, self.buffer, self.offset, self.index_type);
            }

            cb
        }
    }
}

error_ty!{IndexBufferBindError => "Error that can happen when binding vertex buffers.",
    WrongUsageFlag => "one of the buffers was not created with the correct usage flags",
}

#[cfg(test)]
mod tests {
    use buffer::BufferSlice;
    use buffer::BufferUsage;
    use buffer::device_local::DeviceLocalBuffer;
    use command_buffer::sys::bind_ib::IndexBufferBindCommand;
    use command_buffer::sys::bind_ib::IndexBufferBindError;

    #[test]
    fn wrong_usage() {
        let (device, queue) = gfx_dev_and_queue!();
        let buffer = DeviceLocalBuffer::<u16>::new(&device, &BufferUsage::none(),
                                                   Some(queue.family())).unwrap();

        match IndexBufferBindCommand::new(BufferSlice::from(&buffer)) {
            Err(IndexBufferBindError::WrongUsageFlag) => (),
            _ => panic!()
        }
    }
}
