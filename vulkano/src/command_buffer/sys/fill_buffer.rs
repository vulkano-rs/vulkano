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

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a command that fills a buffer with data.
pub struct BufferFillCommand {
    keep_alive: Arc<KeepAlive + 'static>,

    device: vk::Device,
    dst_buffer: vk::Buffer,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
    data: u32,
}

impl BufferFillCommand {
    /// Adds a command that fills a buffer with some data. The data is a u32 whose value will be
    /// repeatidely written in the buffer.
    ///
    /// If the size of the slice is 0, no command is added.
    pub fn untyped<'a, S, T: ?Sized, B>(buffer: S, data: u32)
                                        -> Result<BufferFillCommand, BufferFillError>
        where S: Into<BufferSlice<'a, T, B>>,
              B: Buffer + Send + Sync + 'static
    {
        let buffer = buffer.into();

        // Performing checks.
        if !buffer.buffer().inner_buffer().usage_transfer_src() {
            return Err(BufferFillError::WrongUsageFlag);
        }
        if (buffer.offset() % 4) != 0 {
            return Err(BufferFillError::WrongAlignment);
        }
        if buffer.size() != buffer.buffer().size() && (buffer.size() % 4) != 0 {
            return Err(BufferFillError::WrongAlignment);
        }

        Ok(BufferFillCommand {
            keep_alive: buffer.buffer().clone() as Arc<_>,
            device: buffer.buffer().inner_buffer().device().internal_object(),
            dst_buffer: buffer.buffer().inner_buffer().internal_object(),
            dst_offset: buffer.offset() as vk::DeviceSize,
            size: if buffer.size() == buffer.buffer().size() { vk::WHOLE_SIZE }
                  else { buffer.size() as vk::DeviceSize },
            data: data,
        })
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the command buffer is within a render pass.
    /// - Panicks if the buffer was not allocated with the same device as the command buffer.
    /// - Panicks if the queue family does not support graphics or compute operations.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            let _pool_lock = cb.pool().lock();

            // Various checks.
            assert!(!cb.within_render_pass);
            assert_eq!(self.device, cb.device().internal_object());
            assert!(cb.pool().queue_family().supports_graphics() ||
                    cb.pool().queue_family().supports_compute());

            // Vulkan requires that the size must be >= 1.
            if self.size == 0 { return cb; }

            cb.keep_alive.push(self.keep_alive.clone());

            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();
                vk.CmdFillBuffer(cmd, self.dst_buffer, self.dst_offset, self.size, self.data);
            }

            cb
        }
    }
}

error_ty!{BufferFillError => "Error that can happen when filling a buffer.",
    WrongUsageFlag => "one of the buffers doesn't have the correct usage flag",
    WrongAlignment => "the offset and size must be multiples of 4",
}
