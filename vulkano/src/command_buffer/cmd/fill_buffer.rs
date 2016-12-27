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

use buffer::Buffer;
use buffer::BufferInner;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::cmd::CommandsListPossibleOutsideRenderPass;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use sync::AccessFlagBits;
use sync::PipelineStages;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds an update buffer command at the end of it.
pub struct CmdFillBuffer<L, B>
    where B: Buffer, L: CommandsList
{
    // Parent commands list.
    previous: L,
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

impl<L, B> CmdFillBuffer<L, B>
    where B: Buffer,
          L: CommandsList + CommandsListPossibleOutsideRenderPass
{
    /// Builds a command that writes data to a buffer.
    pub fn new(previous: L, buffer: B, data: u32)
               -> Result<CmdFillBuffer<L, B>, CmdFillBufferError>
    {
        assert!(previous.is_outside_render_pass());     // TODO: error

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
            previous: previous,
            buffer: buffer,
            buffer_handle: buffer_handle,
            offset: offset as vk::DeviceSize,
            size: size as vk::DeviceSize,
            data: data,
        })
    }
}

unsafe impl<L, B> CommandsList for CmdFillBuffer<L, B>
    where B: Buffer,
          L: CommandsList,
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        assert_eq!(self.buffer.inner().buffer.device().internal_object(),
                   builder.device().internal_object());

        {
            let stages = PipelineStages { transfer: true, .. PipelineStages::none() };
            let access = AccessFlagBits { transfer_write: true, .. AccessFlagBits::none() };
            builder.add_buffer_transition(&self.buffer, 0, self.buffer.size(), true,
                                          stages, access);
        }

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();
                vk.CmdFillBuffer(cmd, self.buffer_handle, self.offset, self.size, self.data);
            }
        }));
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
    use command_buffer::cmd::PrimaryCbBuilder;
    use command_buffer::cmd::CommandsList;
    use command_buffer::submit::CommandBuffer;

    #[test]
    fn basic() {
        let (device, queue) = gfx_dev_and_queue!();

        let buffer = CpuAccessibleBuffer::from_data(&device, &BufferUsage::transfer_dest(),
                                                    Some(queue.family()), 0u32).unwrap();

        let _ = PrimaryCbBuilder::new(&device, queue.family())
                    .fill_buffer(buffer.clone(), 128u32)
                    .build()
                    .submit(&queue);

        let content = buffer.read(Duration::from_secs(0)).unwrap();
        assert_eq!(*content, 128);
    }
}*/
