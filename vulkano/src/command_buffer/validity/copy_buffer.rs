// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::error;
use std::fmt;

use buffer::TypedBufferAccess;
use device::Device;
use device::DeviceOwned;
use VulkanObject;

/// Checks whether a copy buffer command is valid.
///
/// # Panic
///
/// - Panics if the source and destination were not created with `device`.
///
pub fn check_copy_buffer<S, D, T>(
    device: &Device,
    source: &S,
    destination: &D,
) -> Result<CheckCopyBuffer, CheckCopyBufferError>
where
    S: ?Sized + TypedBufferAccess<Content = T>,
    D: ?Sized + TypedBufferAccess<Content = T>,
    T: ?Sized,
{
    assert_eq!(
        source.inner().buffer.device().internal_object(),
        device.internal_object()
    );
    assert_eq!(
        destination.inner().buffer.device().internal_object(),
        device.internal_object()
    );

    if !source.inner().buffer.usage_transfer_source() {
        return Err(CheckCopyBufferError::SourceMissingTransferUsage);
    }

    if !destination.inner().buffer.usage_transfer_destination() {
        return Err(CheckCopyBufferError::DestinationMissingTransferUsage);
    }

    let copy_size = cmp::min(source.size(), destination.size());

    if source.conflicts_buffer(&destination) {
        return Err(CheckCopyBufferError::OverlappingRanges);
    } else {
        debug_assert!(!destination.conflicts_buffer(&source));
    }

    Ok(CheckCopyBuffer { copy_size })
}

/// Information returned if `check_copy_buffer` succeeds.
pub struct CheckCopyBuffer {
    /// Size of the transfer in bytes.
    ///
    /// If the size of the source and destination are not equal, then the value is equal to the
    /// smallest of the two.
    pub copy_size: usize,
}

/// Error that can happen from `check_copy_buffer`.
#[derive(Debug, Copy, Clone)]
pub enum CheckCopyBufferError {
    /// The source buffer is missing the transfer source usage.
    SourceMissingTransferUsage,
    /// The destination buffer is missing the transfer destination usage.
    DestinationMissingTransferUsage,
    /// The source and destination are overlapping.
    OverlappingRanges,
}

impl error::Error for CheckCopyBufferError {}

impl fmt::Display for CheckCopyBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckCopyBufferError::SourceMissingTransferUsage => {
                    "the source buffer is missing the transfer source usage"
                }
                CheckCopyBufferError::DestinationMissingTransferUsage => {
                    "the destination buffer is missing the transfer destination usage"
                }
                CheckCopyBufferError::OverlappingRanges =>
                    "the source and destination are overlapping",
            }
        )
    }
}
