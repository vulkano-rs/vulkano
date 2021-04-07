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
use crate::query::GetResultsError;
use crate::query::QueryPool;
use crate::query::QueryResultElement;
use crate::query::QueryResultFlags;
use crate::VulkanObject;
use std::error;
use std::fmt;
use std::ops::Range;

/// Checks whether a copy query pool results command is valid.
///
/// # Panic
///
/// - Panics if the buffer was not created with `device`.
///
pub fn check_copy_query_pool_results<D, T>(
    device: &Device,
    query_pool: &QueryPool,
    queries: Range<u32>,
    destination: &D,
    flags: QueryResultFlags,
) -> Result<usize, CheckCopyQueryPoolResultsError>
where
    D: ?Sized + TypedBufferAccess<Content = [T]>,
    T: QueryResultElement,
{
    let buffer_inner = destination.inner();
    assert_eq!(
        device.internal_object(),
        buffer_inner.buffer.device().internal_object(),
    );
    assert_eq!(
        device.internal_object(),
        query_pool.device().internal_object(),
    );

    if !buffer_inner.buffer.usage_transfer_destination() {
        return Err(CheckCopyQueryPoolResultsError::DestinationMissingTransferUsage);
    }

    let queries_range = query_pool
        .queries_range(queries)
        .ok_or(CheckCopyQueryPoolResultsError::OutOfRange)?;

    Ok(queries_range.check_query_pool_results::<T>(
        buffer_inner.offset,
        destination.len(),
        flags,
    )?)
}

/// Error that can happen from `check_copy_query_pool_results`.
#[derive(Debug, Copy, Clone)]
pub enum CheckCopyQueryPoolResultsError {
    /// The buffer is too small for the copy operation.
    BufferTooSmall {
        /// Required number of elements in the buffer.
        required_len: usize,
        /// Actual number of elements in the buffer.
        actual_len: usize,
    },
    /// The destination buffer is missing the transfer destination usage.
    DestinationMissingTransferUsage,
    /// The provided flags are not allowed for this type of query.
    InvalidFlags,
    /// The provided queries range is not valid for this pool.
    OutOfRange,
}

impl From<GetResultsError> for CheckCopyQueryPoolResultsError {
    #[inline]
    fn from(value: GetResultsError) -> Self {
        match value {
            GetResultsError::BufferTooSmall {
                required_len,
                actual_len,
            } => CheckCopyQueryPoolResultsError::BufferTooSmall {
                required_len,
                actual_len,
            },
            GetResultsError::InvalidFlags => CheckCopyQueryPoolResultsError::InvalidFlags,
            GetResultsError::DeviceLost | GetResultsError::OomError(_) => unreachable!(),
        }
    }
}

impl error::Error for CheckCopyQueryPoolResultsError {}

impl fmt::Display for CheckCopyQueryPoolResultsError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::BufferTooSmall { .. } => {
                    "the buffer is too small for the copy operation"
                }
                Self::DestinationMissingTransferUsage => {
                    "the destination buffer is missing the transfer destination usage"
                }
                Self::InvalidFlags => {
                    "the provided flags are not allowed for this type of query"
                }
                Self::OutOfRange => {
                    "the provided queries range is not valid for this pool"
                }
            }
        )
    }
}

pub fn check_reset_query_pool(
    device: &Device,
    query_pool: &QueryPool,
    queries: Range<u32>,
) -> Result<(), CheckResetQueryPoolError> {
    assert_eq!(
        device.internal_object(),
        query_pool.device().internal_object(),
    );
    query_pool
        .queries_range(queries)
        .ok_or(CheckResetQueryPoolError::OutOfRange)?;
    Ok(())
}

/// Error that can happen from `check_reset_query_pool`.
#[derive(Debug, Copy, Clone)]
pub enum CheckResetQueryPoolError {
    /// The provided queries range is not valid for this pool.
    OutOfRange,
}

impl error::Error for CheckResetQueryPoolError {}

impl fmt::Display for CheckResetQueryPoolError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::OutOfRange => {
                    "the provided queries range is not valid for this pool"
                }
            }
        )
    }
}
