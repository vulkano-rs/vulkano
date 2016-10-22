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
use command_buffer::CommandBufferBuilder;
use command_buffer::cmd::CommandsListPossibleOutsideRenderPass;
use command_buffer::cmd::CommandsList;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds an update buffer command at the end of it.
pub struct CmdUpdateBufferUnsynced<'a, L, B, D: ?Sized>
    where B: Buffer, L: CommandsList, D: 'static
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
    data: &'a D,
}

impl<'a, L, B, D: ?Sized> CmdUpdateBufferUnsynced<'a, L, B, D>
    where B: Buffer,
          L: CommandsList + CommandsListPossibleOutsideRenderPass,
          D: Copy + 'static,
{
    /// Builds a command that writes data to a buffer.
    ///
    /// If the size of the data and the size of the buffer mismatch, then only the intersection
    /// of both will be written.
    ///
    /// The size of the modification must not exceed 65536 bytes. The offset and size must be
    /// multiples of four.
    pub unsafe fn new(previous: L, buffer: B, data: &'a D)
                      -> Result<CmdUpdateBufferUnsynced<'a, L, B, D>, CmdUpdateBufferUnsyncedError>
    {
        assert!(previous.is_outside_render_pass());     // TODO: error

        let size = buffer.size();

        let (buffer_handle, offset) = {
            let BufferInner { buffer: buffer_inner, offset } = buffer.inner();
            if offset % 4 != 0 {
                return Err(CmdUpdateBufferUnsyncedError::WrongAlignment);
            }
            (buffer_inner.internal_object(), offset)
        };

        if size % 4 != 0 {
            return Err(CmdUpdateBufferUnsyncedError::WrongAlignment);
        }

        if size > 65536 {
            return Err(CmdUpdateBufferUnsyncedError::DataTooLarge);
        }

        Ok(CmdUpdateBufferUnsynced {
            previous: previous,
            buffer: buffer,
            buffer_handle: buffer_handle,
            offset: offset as vk::DeviceSize,
            size: size as vk::DeviceSize,
            data: data,
        })
    }
}

unsafe impl<'d, L, B, D: ?Sized> CommandsList for CmdUpdateBufferUnsynced<'d, L, B, D>
    where B: Buffer,
          L: CommandsList,
          D: Copy + 'static,
{
    #[inline]
    fn append<'a>(&'a self, builder: CommandBufferBuilder<'a>) -> CommandBufferBuilder<'a> {
        let builder = self.previous.append(builder);

        assert_eq!(self.buffer.inner().buffer.device().internal_object(),
                   builder.device.internal_object());

        unsafe {
            let vk = builder.device.pointers();
            let cmd = builder.command_buffer.clone().take().unwrap();
            vk.CmdUpdateBuffer(cmd, self.buffer_handle, self.offset, self.size,
                               self.data as *const D as *const _);
        }

        builder
    }
}

unsafe impl<'a, L, B, D: ?Sized> CommandsListPossibleOutsideRenderPass
    for CmdUpdateBufferUnsynced<'a, L, B, D>
    where B: Buffer,
          L: CommandsList,
          D: Copy + 'static,
{
    #[inline]
    fn is_outside_render_pass(&self) -> bool {
        true
    }
}

/// Error that can happen when creating a `CmdUpdateBufferUnsynced`.
#[derive(Debug, Copy, Clone)]
pub enum CmdUpdateBufferUnsyncedError {
    /// The data or size must be 4-bytes aligned.
    WrongAlignment,
    /// The data must not be larger than 64k bytes.
    DataTooLarge,
}

impl error::Error for CmdUpdateBufferUnsyncedError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdUpdateBufferUnsyncedError::WrongAlignment => {
                "the offset or size are not aligned to 4 bytes"
            },
            CmdUpdateBufferUnsyncedError::DataTooLarge => "data is too large",
        }
    }
}

impl fmt::Display for CmdUpdateBufferUnsyncedError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
