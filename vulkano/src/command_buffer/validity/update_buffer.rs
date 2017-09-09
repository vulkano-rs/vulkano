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

use VulkanObject;
use buffer::TypedBufferAccess;
use device::Device;
use device::DeviceOwned;

/// Checks whether an update buffer command is valid.
///
/// # Panic
///
/// - Panics if the buffer not created with `device`.
///
pub fn check_update_buffer<B, D>(device: &Device, buffer: &B, data: &D)
                                 -> Result<(), CheckUpdateBufferError>
    where B: ?Sized + TypedBufferAccess<Content = D>,
          D: ?Sized
{
    assert_eq!(buffer.inner().buffer.device().internal_object(),
               device.internal_object());

    if !buffer.inner().buffer.usage_transfer_destination() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use buffer::BufferAccess;
    use buffer::BufferUsage;
    use buffer::CpuAccessibleBuffer;

    #[test]
    fn missing_usage() {
        let (device, queue) = gfx_dev_and_queue!();
        let buffer =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::vertex_buffer(), 0u32)
                .unwrap();

        match check_update_buffer(&device, &buffer, &0) {
            Err(CheckUpdateBufferError::BufferMissingUsage) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn data_too_large() {
        let (device, queue) = gfx_dev_and_queue!();
        let buffer = CpuAccessibleBuffer::from_iter(device.clone(),
                                                    BufferUsage::transfer_destination(),
                                                    0 .. 65536)
            .unwrap();
        let data = (0 .. 65536).collect::<Vec<u32>>();

        match check_update_buffer(&device, &buffer, &data[..]) {
            Err(CheckUpdateBufferError::DataTooLarge) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn data_just_large_enough() {
        let (device, queue) = gfx_dev_and_queue!();
        let buffer = CpuAccessibleBuffer::from_iter(device.clone(),
                                                    BufferUsage::transfer_destination(),
                                                    (0 .. 100000).map(|_| 0))
            .unwrap();
        let data = (0 .. 65536).map(|_| 0).collect::<Vec<u8>>();

        match check_update_buffer(&device, &buffer, &data[..]) {
            Ok(_) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn wrong_alignment() {
        let (device, queue) = gfx_dev_and_queue!();
        let buffer = CpuAccessibleBuffer::from_iter(device.clone(),
                                                    BufferUsage::transfer_destination(),
                                                    0 .. 100)
            .unwrap();
        let data = (0 .. 30).collect::<Vec<u8>>();

        match check_update_buffer(&device, &buffer.slice(1 .. 50).unwrap(), &data[..]) {
            Err(CheckUpdateBufferError::WrongAlignment) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn wrong_device() {
        let (dev1, queue) = gfx_dev_and_queue!();
        let (dev2, _) = gfx_dev_and_queue!();
        let buffer = CpuAccessibleBuffer::from_data(dev1, BufferUsage::all(), 0u32).unwrap();

        assert_should_panic!({
                                 let _ = check_update_buffer(&dev2, &buffer, &0);
                             });
    }
}
