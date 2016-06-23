// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::pool::CommandPool;
use command_buffer::sys::KeepAlive;
use command_buffer::sys::UnsafeCommandBufferBuilder;

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a command that writes data to a buffer.
pub struct BufferUpdateCommand {
    keep_alive: Arc<KeepAlive + 'static>,

    device: vk::Device,
    dst_buffer: vk::Buffer,
    dst_offset: vk::DeviceSize,
    data: Vec<u32>,
}

impl BufferUpdateCommand {
    /// Adds a command that writes the content of a buffer.
    ///
    /// The size of the buffer slice is what is taken into account when determining the size. If
    /// `data` is larger than the slice, then the remaining will be ignored. If `data` is smaller,
    /// an error is returned.
    ///
    /// If the size of the slice is 0, no command is added.
    #[inline]
    pub fn new<'a, S, T: ?Sized, B>(buffer: S, data: &T)
                                    -> Result<BufferUpdateCommand, BufferUpdateError>
        where S: Into<BufferSlice<'a, T, B>>,
              B: Buffer + Send + Sync + 'static,
              T: 'static    // TODO: this bound should be removed eventually, because it would be enforced by the BufferSlice
    {
        BufferUpdateCommand::untyped(buffer, data)
    }

    /// Same as `new`, except that the type of data does not have to match the content of the
    /// buffer.
    ///
    /// Since this only concerns plain data, the function is not unsafe.
    pub fn untyped<'a, S, T: ?Sized, B, D: ?Sized>(buffer: S, data: &D)
                                                   -> Result<BufferUpdateCommand, BufferUpdateError>
        where S: Into<BufferSlice<'a, T, B>>,
              B: Buffer + Send + Sync + 'static,
              D: 'static    // FIXME: needs BufferContent bound or something
    {
        let buffer = buffer.into();

        // Performing checks.
        if !buffer.buffer().inner_buffer().usage_transfer_src() {
            return Err(BufferUpdateError::WrongUsageFlag);
        }
        if (buffer.offset() % 4) != 0 || (buffer.size() % 4) != 0 {
            return Err(BufferUpdateError::WrongAlignment);
        }
        if buffer.size() >= 0x10000 { return Err(BufferUpdateError::RegionTooLarge); }
        if mem::size_of_val(data) < buffer.size() {
            return Err(BufferUpdateError::DataTooSmall);
        }

        let data_size = cmp::min(buffer.size(), mem::size_of_val(data));
        debug_assert!((data_size % 4) == 0);

        Ok(BufferUpdateCommand {
            keep_alive: buffer.buffer().clone() as Arc<_>,
            device: buffer.buffer().inner_buffer().device().internal_object(),
            dst_buffer: buffer.buffer().inner_buffer().internal_object(),
            dst_offset: buffer.offset() as vk::DeviceSize,
            data: unsafe {
                let mut d = Vec::with_capacity(data_size / mem::size_of::<u32>());
                ptr::copy_nonoverlapping(data as *const D as *const u32,
                                         d.as_mut_ptr(), d.capacity());
                d.set_len(data_size / mem::size_of::<u32>());
                d
            },
        })
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the command buffer is within a render pass.
    /// - Panicks if the buffer was not allocated with the same device as the command buffer.
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

            // Vulkan requires that the size must be >= 1.
            if self.data.is_empty() { return cb; }

            cb.keep_alive.push(self.keep_alive.clone());

            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();
                vk.CmdUpdateBuffer(cmd, self.dst_buffer, self.dst_offset,
                                   self.data.len() as vk::DeviceSize, self.data.as_ptr());
            }

            cb
        }
    }
}

error_ty!{BufferUpdateError => "Error that can happen when updating a buffer.",
    WrongUsageFlag => "one of the buffers doesn't have the correct usage flag",
    WrongAlignment => "the offset and size must be multiples of 4",
    RegionTooLarge => "the size of the region exceeds the allowed limits",
    DataTooSmall => "the data is smaller than the region to copy to",
}
