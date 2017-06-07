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

use buffer::BufferAccess;
use buffer::BufferInner;
use command_buffer::CommandAddError;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;
use vk;

pub struct CmdFillBuffer<B> {
    // The buffer to update.
    buffer: B,
    // Raw buffer handle.
    buffer_handle: vk::Buffer,
    // Offset of the update.
    offset: vk::DeviceSize,
    // Size of the update.
    size: vk::DeviceSize,
    // The data to write to the buffer.
    data: u32,
}

unsafe impl<B> DeviceOwned for CmdFillBuffer<B>
    where B: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

impl<B> CmdFillBuffer<B>
    where B: BufferAccess
{
    /// Builds a command that writes data to a buffer.
    // TODO: not safe because of signalling NaNs
    pub fn new(buffer: B, data: u32) -> Result<CmdFillBuffer<B>, CmdFillBufferError> {
        let size = buffer.size();

        let (buffer_handle, offset) = {
            let BufferInner { buffer: buffer_inner, offset } = buffer.inner();
            if !buffer_inner.usage_transfer_dest() {
                return Err(CmdFillBufferError::BufferMissingUsage);
            }
            if offset % 4 != 0 {
                return Err(CmdFillBufferError::WrongAlignment);
            }
            (buffer_inner.internal_object(), offset)
        };

        Ok(CmdFillBuffer {
            buffer: buffer,
            buffer_handle: buffer_handle,
            offset: offset as vk::DeviceSize,
            size: size as vk::DeviceSize,
            data: data,
        })
    }
}

impl<B> CmdFillBuffer<B> {
    /// Returns the buffer that is going to be filled.
    #[inline]
    pub fn buffer(&self) -> &B {
        &self.buffer
    }
}

unsafe impl<'a, P, B> AddCommand<&'a CmdFillBuffer<B>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdFillBuffer<B>) -> Result<Self::Out, CommandAddError> {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdFillBuffer(cmd, command.buffer_handle, command.offset,
                             command.size, command.data);
        }

        Ok(self)
    }
}

/// Error that can happen when creating a `CmdFillBuffer`.
#[derive(Debug, Copy, Clone)]
pub enum CmdFillBufferError {
    /// The "transfer destination" usage must be enabled on the buffer.
    BufferMissingUsage,
    /// The data or size must be 4-bytes aligned.
    WrongAlignment,
}

impl error::Error for CmdFillBufferError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdFillBufferError::BufferMissingUsage => {
                "the transfer destination usage must be enabled on the buffer"
            },
            CmdFillBufferError::WrongAlignment => {
                "the offset or size are not aligned to 4 bytes"
            },
        }
    }
}

impl fmt::Display for CmdFillBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

/* TODO: restore
#[cfg(test)]
mod tests {
    use std::time::Duration;
    use buffer::BufferUsage;
    use buffer::CpuAccessibleBuffer;
    use command_buffer::commands_raw::PrimaryCbBuilder;
    use command_buffer::commands_raw::CommandsList;
    use command_buffer::submit::CommandBuffer;

    #[test]
    fn basic() {
        let (device, queue) = gfx_dev_and_queue!();

        let buffer = CpuAccessibleBuffer::from_data(&device, BufferUsage::transfer_dest(),
                                                    Some(queue.family()), 0u32).unwrap();

        let _ = PrimaryCbBuilder::new(&device, queue.family())
                    .fill_buffer(buffer.clone(), 128u32)
                    .build()
                    .submit(&queue);

        let content = buffer.read(Duration::from_secs(0)).unwrap();
        assert_eq!(*content, 128);
    }
}*/
