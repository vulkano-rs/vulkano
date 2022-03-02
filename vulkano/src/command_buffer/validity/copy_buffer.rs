// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::ranges::is_overlapping_ranges;
use crate::{
    buffer::BufferAccess,
    device::{Device, DeviceOwned},
    DeviceSize, VulkanObject,
};
use std::{error, fmt};

/// Checks whether a copy buffer command is valid.
///
/// # Panic
///
/// - Panics if the source and destination were not created with `device`.
///
pub fn check_copy_buffer(
    device: &Device,
    source: &dyn BufferAccess,
    destination: &dyn BufferAccess,
    source_offset: DeviceSize,
    destination_offset: DeviceSize,
    size: DeviceSize,
) -> Result<(), CheckCopyBufferError> {
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
        && is_overlapping_ranges(source_offset, size, destination_offset, size)
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
