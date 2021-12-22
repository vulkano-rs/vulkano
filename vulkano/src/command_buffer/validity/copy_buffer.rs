// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::TypedBufferAccess;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::DeviceSize;
use crate::VulkanObject;
use std::error;
use std::fmt;

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
    source_offset: DeviceSize,
    destination_offset: DeviceSize,
    size: DeviceSize,
) -> Result<(), CheckCopyBufferError>
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

    if !source.inner().buffer.usage().transfer_source {
        return Err(CheckCopyBufferError::SourceMissingTransferUsage);
    }

    if !destination.inner().buffer.usage().transfer_destination {
        return Err(CheckCopyBufferError::DestinationMissingTransferUsage);
    }

    if source_offset + size > source.size() {
        return Err(CheckCopyBufferError::SourceOutOfBounds);
    }

    if destination_offset + size > destination.size() {
        return Err(CheckCopyBufferError::DestinationOutOfBounds);
    }

    if source.conflict_key() == destination.conflict_key()
        && (destination_offset < source_offset + size)
        && (source_offset < destination_offset + size)
    {
        return Err(CheckCopyBufferError::OverlappingRanges);
    }

    Ok(())
}

/// Error that can happen from `check_copy_buffer`.
#[derive(Debug, Copy, Clone)]
pub enum CheckCopyBufferError {
    /// The source buffer is missing the transfer source usage.
    SourceMissingTransferUsage,
    /// The destination buffer is missing the transfer destination usage.
    DestinationMissingTransferUsage,
    /// The source and destination ranges are overlapping.
    OverlappingRanges,
    /// The source range is out of bounds.
    SourceOutOfBounds,
    /// The destination range is out of bounds.
    DestinationOutOfBounds,
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
                    "the source and destination ranges are overlapping",
                CheckCopyBufferError::SourceOutOfBounds => "the source range is out of bounds",
                CheckCopyBufferError::DestinationOutOfBounds => {
                    "the destination range is out of bounds"
                }
            }
        )
    }
}
