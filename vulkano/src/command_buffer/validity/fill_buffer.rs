// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

use buffer::BufferAccess;
use device::Device;
use device::DeviceOwned;
use VulkanObject;

/// Checks whether a fill buffer command is valid.
///
/// # Panic
///
/// - Panics if the buffer not created with `device`.
///
pub fn check_fill_buffer<B>(device: &Device, buffer: &B) -> Result<(), CheckFillBufferError>
where
    B: ?Sized + BufferAccess,
{
    assert_eq!(
        buffer.inner().buffer.device().internal_object(),
        device.internal_object()
    );

    if !buffer.inner().buffer.usage_transfer_destination() {
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

impl error::Error for CheckFillBufferError {}

impl fmt::Display for CheckFillBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckFillBufferError::BufferMissingUsage => {
                    "the transfer destination usage must be enabled on the buffer"
                }
                CheckFillBufferError::WrongAlignment =>
                    "the offset or size are not aligned to 4 bytes",
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use buffer::BufferUsage;
    use buffer::CpuAccessibleBuffer;

    #[test]
    fn missing_usage() {
        let (device, queue) = gfx_dev_and_queue!();
        let buffer = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            0u32,
        )
        .unwrap();

        match check_fill_buffer(&device, &buffer) {
            Err(CheckFillBufferError::BufferMissingUsage) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn wrong_device() {
        let (dev1, queue) = gfx_dev_and_queue!();
        let (dev2, _) = gfx_dev_and_queue!();
        let buffer = CpuAccessibleBuffer::from_data(dev1, BufferUsage::all(), false, 0u32).unwrap();

        assert_should_panic!({
            let _ = check_fill_buffer(&dev2, &buffer);
        });
    }
}
