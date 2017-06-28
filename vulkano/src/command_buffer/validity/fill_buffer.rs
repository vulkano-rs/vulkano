// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

use VulkanObject;
use buffer::BufferAccess;
use device::Device;
use device::DeviceOwned;

/// Checks whether a fill buffer command is valid.
///
/// # Panic
///
/// - Panics if the buffer not created with `device`.
///
pub fn check_fill_buffer<B>(device: &Device, buffer: &B) -> Result<(), CheckFillBufferError>
    where B: ?Sized + BufferAccess
{
    assert_eq!(buffer.inner().buffer.device().internal_object(),
               device.internal_object());

    if !buffer.inner().buffer.usage_transfer_dest() {
        return Err(CheckFillBufferError::BufferMissingUsage);
    }

    if buffer.inner().offset % 4 != 0 {
        return Err(CheckFillBufferError::WrongAlignment);
    }

    Ok(())
}

/// Error that can happen when attempting to add a `fill_buffer` command.
#[derive(Debug, Copy, Clone)]
pub enum CheckFillBufferError {
    /// The "transfer destination" usage must be enabled on the buffer.
    BufferMissingUsage,
    /// The data or size must be 4-bytes aligned.
    WrongAlignment,
}

impl error::Error for CheckFillBufferError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CheckFillBufferError::BufferMissingUsage => {
                "the transfer destination usage must be enabled on the buffer"
            },
            CheckFillBufferError::WrongAlignment => {
                "the offset or size are not aligned to 4 bytes"
            },
        }
    }
}

impl fmt::Display for CheckFillBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
