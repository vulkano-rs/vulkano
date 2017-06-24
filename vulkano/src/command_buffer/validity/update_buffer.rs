// Copyright (c) 2017 The vulkano developers
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

use buffer::BufferAccess;
use device::Device;
use device::DeviceOwned;
use VulkanObject;

/// Checks whether an update buffer command is valid.
///
/// # Panic
///
/// - Panics if the buffer not created with `device`.
///
pub fn check_update_buffer<B, D>(device: &Device, buffer: &B, data: &D)
                                 -> Result<(), CheckUpdateBufferError>
    where B: ?Sized + BufferAccess,
          D: ?Sized
{
    assert_eq!(buffer.inner().buffer.device().internal_object(), device.internal_object());

    if !buffer.inner().buffer.usage_transfer_dest() {
        return Err(CheckUpdateBufferError::BufferMissingUsage);
    }

    if buffer.inner().offset % 4 != 0 {
        return Err(CheckUpdateBufferError::WrongAlignment);
    }

    let size = cmp::min(buffer.size(), mem::size_of_val(data));

    if size % 4 != 0 {
        return Err(CheckUpdateBufferError::WrongAlignment);
    }

    if size > 65536 {
        return Err(CheckUpdateBufferError::DataTooLarge);
    }

    Ok(())
}

/// Error that can happen when attempting to add an `update_buffer` command.
#[derive(Debug, Copy, Clone)]
pub enum CheckUpdateBufferError {
    /// The "transfer destination" usage must be enabled on the buffer.
    BufferMissingUsage,
    /// The data or size must be 4-bytes aligned.
    WrongAlignment,
    /// The data must not be larger than 64k bytes.
    DataTooLarge,
}

impl error::Error for CheckUpdateBufferError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CheckUpdateBufferError::BufferMissingUsage => {
                "the transfer destination usage must be enabled on the buffer"
            },
            CheckUpdateBufferError::WrongAlignment => {
                "the offset or size are not aligned to 4 bytes"
            },
            CheckUpdateBufferError::DataTooLarge => "data is too large",
        }
    }
}

impl fmt::Display for CheckUpdateBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
