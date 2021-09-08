// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::VulkanObject;
use std::error;
use std::fmt;

/// Checks whether an indirect buffer can be bound.
pub fn check_indirect_buffer<Inb>(
    device: &Device,
    buffer: &Inb,
) -> Result<(), CheckIndirectBufferError>
where
    Inb: BufferAccess + 'static,
{
    assert_eq!(
        buffer.inner().buffer.device().internal_object(),
        device.internal_object()
    );

    if !buffer.inner().buffer.usage().indirect_buffer {
        return Err(CheckIndirectBufferError::BufferMissingUsage);
    }

    Ok(())
}

/// Error that can happen when checking whether binding an indirect buffer is valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckIndirectBufferError {
    /// The "indirect buffer" usage must be enabled on the indirect buffer.
    BufferMissingUsage,
    /// The maximum number of indirect draws has been exceeded.
    MaxDrawIndirectCountLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },
}

impl error::Error for CheckIndirectBufferError {}

impl fmt::Display for CheckIndirectBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckIndirectBufferError::BufferMissingUsage => {
                    "the indirect buffer usage must be enabled on the indirect buffer"
                }
                CheckIndirectBufferError::MaxDrawIndirectCountLimitExceeded {
                    limit,
                    requested,
                } => {
                    "the maximum number of indirect draws has been exceeded"
                }
            }
        )
    }
}
