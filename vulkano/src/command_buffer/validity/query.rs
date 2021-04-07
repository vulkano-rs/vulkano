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
use crate::query::QueryControlFlags;
use crate::query::QueryPool;
use crate::query::QueryResultElement;
use crate::query::QueryResultFlags;
use crate::query::QueryType;
use crate::sync::PipelineStage;
use crate::VulkanObject;
use std::error;
use std::fmt;
use std::ops::Range;

/// Checks whether a `begin_query` command is valid.
///
/// # Panic
///
/// - Panics if the query pool was not created with `device`.
pub fn check_begin_query(
    device: &Device,
    query_pool: &QueryPool,
    query: u32,
    flags: QueryControlFlags,
) -> Result<(), CheckBeginQueryError> {
    assert_eq!(
        device.internal_object(),
        query_pool.device().internal_object(),
    );
    query_pool
        .query(query)
        .ok_or(CheckBeginQueryError::OutOfRange)?;

    match query_pool.ty() {
        QueryType::Occlusion => {
            if flags.precise && !device.enabled_features().occlusion_query_precise {
                return Err(CheckBeginQueryError::OcclusionQueryPreciseFeatureNotEnabled);
            }
        }
        QueryType::PipelineStatistics(_) => {
            if flags.precise {
                return Err(CheckBeginQueryError::InvalidFlags);
            }
        }
        QueryType::Timestamp => return Err(CheckBeginQueryError::NotPermitted),
    }

    Ok(())
}

/// Error that can happen from `check_begin_query`.
#[derive(Debug, Copy, Clone)]
pub enum CheckBeginQueryError {
    /// The provided flags are not allowed for this type of query.
    InvalidFlags,
    /// This operation is not permitted on this query type.
    NotPermitted,
    /// `QueryControlFlags::precise` was requested, but the `occlusion_query_precise` feature was not enabled.
    OcclusionQueryPreciseFeatureNotEnabled,
    /// The provided query index is not valid for this pool.
    OutOfRange,
}

impl error::Error for CheckBeginQueryError {}

impl fmt::Display for CheckBeginQueryError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::InvalidFlags => {
                    "the provided flags are not allowed for this type of query"
                }
                Self::NotPermitted => {
                    "this operation is not permitted on this query type"
                }
                Self::OcclusionQueryPreciseFeatureNotEnabled => {
                    "QueryControlFlags::precise was requested, but the occlusion_query_precise feature was not enabled"
                }
                Self::OutOfRange => {
                    "the provided query index is not valid for this pool"
                }
            }
        )
    }
}

/// Checks whether a `end_query` command is valid.
///
/// # Panic
///
/// - Panics if the query pool was not created with `device`.
pub fn check_end_query(
    device: &Device,
    query_pool: &QueryPool,
    query: u32,
) -> Result<(), CheckEndQueryError> {
    assert_eq!(
        device.internal_object(),
        query_pool.device().internal_object(),
    );

    query_pool
        .query(query)
        .ok_or(CheckEndQueryError::OutOfRange)?;

    Ok(())
}

/// Error that can happen from `check_end_query`.
#[derive(Debug, Copy, Clone)]
pub enum CheckEndQueryError {
    /// The provided query index is not valid for this pool.
    OutOfRange,
}

impl error::Error for CheckEndQueryError {}

impl fmt::Display for CheckEndQueryError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::OutOfRange => {
                    "the provided query index is not valid for this pool"
                }
            }
        )
    }
}

/// Checks whether a `write_timestamp` command is valid.
///
/// # Panic
///
/// - Panics if the query pool was not created with `device`.
pub fn check_write_timestamp(
    device: &Device,
    query_pool: &QueryPool,
    query: u32,
    stage: PipelineStage,
) -> Result<(), CheckWriteTimestampError> {
    assert_eq!(
        device.internal_object(),
        query_pool.device().internal_object(),
    );

    if !matches!(query_pool.ty(), QueryType::Timestamp) {
        return Err(CheckWriteTimestampError::NotPermitted);
    }

    query_pool
        .query(query)
        .ok_or(CheckWriteTimestampError::OutOfRange)?;

    match stage {
        PipelineStage::GeometryShader => {
            if !device.enabled_features().geometry_shader {
                return Err(CheckWriteTimestampError::GeometryShaderFeatureNotEnabled);
            }
        }
        PipelineStage::TessellationControlShader | PipelineStage::TessellationEvaluationShader => {
            if !device.enabled_features().tessellation_shader {
                return Err(CheckWriteTimestampError::TessellationShaderFeatureNotEnabled);
            }
        }
        _ => (),
    }

    Ok(())
}

/// Error that can happen from `check_write_timestamp`.
#[derive(Debug, Copy, Clone)]
pub enum CheckWriteTimestampError {
    /// The geometry shader stage was requested, but the `geometry_shader` feature was not enabled.
    GeometryShaderFeatureNotEnabled,
    /// This operation is not permitted on this query type.
    NotPermitted,
    /// The provided query index is not valid for this pool.
    OutOfRange,
    /// A tessellation shader stage was requested, but the `tessellation_shader` feature was not enabled.
    TessellationShaderFeatureNotEnabled,
}

impl error::Error for CheckWriteTimestampError {}

impl fmt::Display for CheckWriteTimestampError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::GeometryShaderFeatureNotEnabled => {
                    "the geometry shader stage was requested, but the geometry_shader feature was not enabled"
                }
                Self::NotPermitted => {
                    "this operation is not permitted on this query type"
                }
                Self::OutOfRange => {
                    "the provided query index is not valid for this pool"
                }
                Self::TessellationShaderFeatureNotEnabled => {
                    "a tessellation shader stage was requested, but the tessellation_shader feature was not enabled"
                }
            }
        )
    }
}

/// Checks whether a `copy_query_pool_results` command is valid.
///
/// # Panic
///
/// - Panics if the query pool or buffer was not created with `device`.
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

/// Checks whether a `reset_query_pool` command is valid.
///
/// # Panic
///
/// - Panics if the query pool was not created with `device`.
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
