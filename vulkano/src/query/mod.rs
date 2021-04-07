// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Gather information about rendering, held in query pools.
//!
//! In Vulkan, queries are not created individually. Instead you manipulate **query pools**, which
//! represent a collection of queries. Whenever you use a query, you have to specify both the query
//! pool and the slot id within that query pool.

use crate::check_errors;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::vk;
use crate::Error;
use crate::OomError;
use crate::Success;
use crate::VulkanObject;
use std::error;
use std::ffi::c_void;
use std::fmt;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

/// A collection of one or more queries of a particular type.
#[derive(Debug)]
pub struct QueryPool {
    pool: vk::QueryPool,
    device: Arc<Device>,
    num_slots: u32,
    ty: QueryType,
}

impl QueryPool {
    /// Builds a new query pool.
    pub fn new(
        device: Arc<Device>,
        ty: QueryType,
        num_slots: u32,
    ) -> Result<QueryPool, QueryPoolCreationError> {
        let (vk_ty, statistics) = match ty {
            QueryType::Occlusion => (vk::QUERY_TYPE_OCCLUSION, 0),
            QueryType::Timestamp => (vk::QUERY_TYPE_TIMESTAMP, 0),
            QueryType::PipelineStatistics(flags) => {
                if !device.enabled_features().pipeline_statistics_query {
                    return Err(QueryPoolCreationError::PipelineStatisticsQueryFeatureNotEnabled);
                }

                (vk::QUERY_TYPE_PIPELINE_STATISTICS, flags.into())
            }
        };

        let pool = unsafe {
            let infos = vk::QueryPoolCreateInfo {
                sType: vk::STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                queryType: vk_ty,
                queryCount: num_slots,
                pipelineStatistics: statistics,
            };

            let mut output = MaybeUninit::uninit();
            let vk = device.pointers();
            check_errors(vk.CreateQueryPool(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(QueryPool {
            pool,
            device,
            num_slots,
            ty,
        })
    }

    /// Returns the [`QueryType`] that this query pool was created with.
    #[inline]
    pub fn ty(&self) -> QueryType {
        self.ty
    }

    /// Returns the number of query slots of this query pool.
    #[inline]
    pub fn num_slots(&self) -> u32 {
        self.num_slots
    }

    /// Returns a reference to a single query slot, or `None` if the index is out of range.
    #[inline]
    pub fn query(&self, index: u32) -> Option<Query> {
        if index < self.num_slots() {
            Some(Query { pool: self, index })
        } else {
            None
        }
    }

    /// Returns a reference to a range of queries, or `None` if out of range.
    ///
    /// # Panic
    ///
    /// Panics if the range is empty.
    #[inline]
    pub fn queries_range(&self, range: Range<u32>) -> Option<QueriesRange> {
        assert!(!range.is_empty());

        if range.end <= self.num_slots() {
            Some(QueriesRange { pool: self, range })
        } else {
            None
        }
    }
}

unsafe impl VulkanObject for QueryPool {
    type Object = vk::QueryPool;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_QUERY_POOL;

    #[inline]
    fn internal_object(&self) -> vk::QueryPool {
        self.pool
    }
}

unsafe impl DeviceOwned for QueryPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Drop for QueryPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyQueryPool(self.device.internal_object(), self.pool, ptr::null());
        }
    }
}

/// Error that can happen when creating a query pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryPoolCreationError {
    /// Not enough memory.
    OomError(OomError),
    /// A pipeline statistics pool was requested but the corresponding feature wasn't enabled.
    PipelineStatisticsQueryFeatureNotEnabled,
}

impl error::Error for QueryPoolCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            QueryPoolCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for QueryPoolCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                QueryPoolCreationError::OomError(_) => "not enough memory available",
                QueryPoolCreationError::PipelineStatisticsQueryFeatureNotEnabled => {
                    "a pipeline statistics pool was requested but the corresponding feature \
                 wasn't enabled"
                }
            }
        )
    }
}

impl From<OomError> for QueryPoolCreationError {
    #[inline]
    fn from(err: OomError) -> QueryPoolCreationError {
        QueryPoolCreationError::OomError(err)
    }
}

impl From<Error> for QueryPoolCreationError {
    #[inline]
    fn from(err: Error) -> QueryPoolCreationError {
        match err {
            err @ Error::OutOfHostMemory => QueryPoolCreationError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => QueryPoolCreationError::OomError(OomError::from(err)),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// A reference to a single query slot.
///
/// This is created through [`QueryPool::query`].
#[derive(Clone, Debug)]
pub struct Query<'a> {
    pool: &'a QueryPool,
    index: u32,
}

impl<'a> Query<'a> {
    /// Returns a reference to the query pool.
    #[inline]
    pub fn pool(&self) -> &'a QueryPool {
        &self.pool
    }

    /// Returns the index of the query represented.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

/// A reference to a range of queries.
///
/// This is created through [`QueryPool::queries_range`].
#[derive(Clone, Debug)]
pub struct QueriesRange<'a> {
    pool: &'a QueryPool,
    range: Range<u32>,
}

impl<'a> QueriesRange<'a> {
    /// Returns a reference to the query pool.
    #[inline]
    pub fn pool(&self) -> &'a QueryPool {
        &self.pool
    }

    /// Returns the range of queries represented.
    #[inline]
    pub fn range(&self) -> Range<u32> {
        self.range.clone()
    }

    /// Copies the results of this range of queries to a buffer on the CPU.
    ///
    /// [`self.pool().ty().data_size()`](QueryType::data_size) elements
    /// will be written for each query in the range, plus 1 extra element per query if
    /// [`QueryResultFlags::with_availability`] is enabled.
    /// The provided buffer must be large enough to hold the data.
    ///
    /// `true` is returned if every result was available and written to the buffer. `false`
    /// is returned if some results were not yet available; these will not be written to the buffer.
    ///
    /// See also [`copy_query_pool_results`](crate::command_buffer::AutoCommandBufferBuilder::copy_query_pool_results).
    pub fn get_results<T>(
        &self,
        destination: &mut [T],
        flags: QueryResultFlags,
    ) -> Result<bool, GetResultsError>
    where
        T: QueryResultElement,
    {
        let stride = self.check_query_pool_results::<T>(
            destination.as_ptr() as usize,
            destination.len(),
            flags,
        )?;

        let result = unsafe {
            let vk = self.pool.device.pointers();
            check_errors(vk.GetQueryPoolResults(
                self.pool.device.internal_object(),
                self.pool.internal_object(),
                self.range.start,
                self.range.end - self.range.start,
                std::mem::size_of_val(destination),
                destination.as_mut_ptr() as *mut c_void,
                stride as vk::DeviceSize,
                vk::QueryResultFlags::from(flags) | T::FLAG,
            ))?
        };

        Ok(match result {
            Success::Success => true,
            Success::NotReady => false,
            s => panic!("unexpected success value: {:?}", s),
        })
    }

    pub(crate) fn check_query_pool_results<T>(
        &self,
        buffer_start: usize,
        buffer_len: usize,
        flags: QueryResultFlags,
    ) -> Result<usize, GetResultsError>
    where
        T: QueryResultElement,
    {
        assert!(buffer_len > 0);
        debug_assert!(buffer_start % std::mem::size_of::<T>() == 0);

        let count = self.range.end - self.range.start;
        let per_query_len = self.pool.ty.data_size() + flags.with_availability as usize;
        let required_len = per_query_len * count as usize;

        if buffer_len < required_len {
            return Err(GetResultsError::BufferTooSmall {
                required_len,
                actual_len: buffer_len,
            });
        }

        match self.pool.ty {
            QueryType::Occlusion => (),
            QueryType::PipelineStatistics(_) => (),
            QueryType::Timestamp => {
                if flags.partial {
                    return Err(GetResultsError::InvalidFlags);
                }
            }
        }

        Ok(per_query_len * std::mem::size_of::<T>())
    }
}

/// Error that can happen when calling [`QueriesRange::get_results`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GetResultsError {
    /// The buffer is too small for the operation.
    BufferTooSmall {
        /// Required number of elements in the buffer.
        required_len: usize,
        /// Actual number of elements in the buffer.
        actual_len: usize,
    },
    /// The connection to the device has been lost.
    DeviceLost,
    /// The provided flags are not allowed for this type of query.
    InvalidFlags,
    /// Not enough memory.
    OomError(OomError),
}

impl From<Error> for GetResultsError {
    #[inline]
    fn from(err: Error) -> Self {
        match err {
            Error::OutOfHostMemory | Error::OutOfDeviceMemory => {
                Self::OomError(OomError::from(err))
            }
            Error::DeviceLost => Self::DeviceLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for GetResultsError {
    #[inline]
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl fmt::Display for GetResultsError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::BufferTooSmall { .. } => {
                    "the buffer is too small for the operation"
                }
                Self::DeviceLost => "the connection to the device has been lost",
                Self::InvalidFlags => {
                    "the provided flags are not allowed for this type of query"
                }
                Self::OomError(_) => "not enough memory available",
            }
        )
    }
}

impl error::Error for GetResultsError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

/// A trait for elements of buffers that can be used as a destination for query results.
///
/// # Safety
/// This is implemented for `u32` and `u64`. Unless you really know what you're doing, you should
/// not implement this trait for any other type.
pub unsafe trait QueryResultElement {
    const FLAG: vk::QueryResultFlags;
}

unsafe impl QueryResultElement for u32 {
    const FLAG: vk::QueryResultFlags = 0;
}

unsafe impl QueryResultElement for u64 {
    const FLAG: vk::QueryResultFlags = vk::QUERY_RESULT_64_BIT;
}

/// The type of query that a query pool should perform.
#[derive(Debug, Copy, Clone)]
pub enum QueryType {
    /// Tracks the number of samples that pass per-fragment tests (e.g. the depth test).
    Occlusion,
    /// Tracks statistics on pipeline invocations and their input data.
    PipelineStatistics(QueryPipelineStatisticFlags),
    /// Writes timestamps at chosen points in a command buffer.
    Timestamp,
}

impl QueryType {
    /// Returns the number of [`QueryResultElement`]s that are needed to hold the result of a
    /// single query of this type.
    ///
    /// - For `Occlusion` and `Timestamp` queries, this returns 1.
    /// - For `PipelineStatistics` queries, this returns the number of statistics flags enabled.
    ///
    /// If the results are retrieved with [`QueryResultFlags::with_availability`] enabled, then
    /// an additional element is required per query.
    #[inline]
    pub fn data_size(&self) -> usize {
        match self {
            Self::Occlusion | Self::Timestamp => 1,
            Self::PipelineStatistics(flags) => flags.count(),
        }
    }
}

/// Flags that control how a query is to be executed.
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryControlFlags {
    /// For occlusion queries, specifies that the result must reflect the exact number of
    /// tests passed. If not enabled, the query may return a result of 1 even if more fragments
    /// passed the test.
    pub precise: bool,
}

/// For pipeline statistics queries, the statistics that should be gathered.
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryPipelineStatisticFlags {
    /// Count the number of vertices processed by the input assembly.
    pub input_assembly_vertices: bool,
    /// Count the number of primitives processed by the input assembly.
    pub input_assembly_primitives: bool,
    /// Count the number of times a vertex shader is invoked.
    pub vertex_shader_invocations: bool,
    /// Count the number of times a geometry shader is invoked.
    pub geometry_shader_invocations: bool,
    /// Count the number of primitives generated by geometry shaders.
    pub geometry_shader_primitives: bool,
    /// Count the number of times the clipping stage is invoked on a primitive.
    pub clipping_invocations: bool,
    /// Count the number of primitives that are output by the clipping stage.
    pub clipping_primitives: bool,
    /// Count the number of times a fragment shader is invoked.
    pub fragment_shader_invocations: bool,
    /// Count the number of patches processed by a tessellation control shader.
    pub tessellation_control_shader_patches: bool,
    /// Count the number of times a tessellation evaluation shader is invoked.
    pub tessellation_evaluation_shader_invocations: bool,
    /// Count the number of times a compute shader is invoked.
    pub compute_shader_invocations: bool,
}

impl QueryPipelineStatisticFlags {
    #[inline]
    pub fn none() -> QueryPipelineStatisticFlags {
        QueryPipelineStatisticFlags {
            input_assembly_vertices: false,
            input_assembly_primitives: false,
            vertex_shader_invocations: false,
            geometry_shader_invocations: false,
            geometry_shader_primitives: false,
            clipping_invocations: false,
            clipping_primitives: false,
            fragment_shader_invocations: false,
            tessellation_control_shader_patches: false,
            tessellation_evaluation_shader_invocations: false,
            compute_shader_invocations: false,
        }
    }

    /// Returns the number of flags that are set to `true`.
    #[inline]
    pub fn count(&self) -> usize {
        let &Self {
            input_assembly_vertices,
            input_assembly_primitives,
            vertex_shader_invocations,
            geometry_shader_invocations,
            geometry_shader_primitives,
            clipping_invocations,
            clipping_primitives,
            fragment_shader_invocations,
            tessellation_control_shader_patches,
            tessellation_evaluation_shader_invocations,
            compute_shader_invocations,
        } = self;
        input_assembly_vertices as usize
            + input_assembly_primitives as usize
            + vertex_shader_invocations as usize
            + geometry_shader_invocations as usize
            + geometry_shader_primitives as usize
            + clipping_invocations as usize
            + clipping_primitives as usize
            + fragment_shader_invocations as usize
            + tessellation_control_shader_patches as usize
            + tessellation_evaluation_shader_invocations as usize
            + compute_shader_invocations as usize
    }
}

impl From<QueryPipelineStatisticFlags> for vk::QueryPipelineStatisticFlags {
    fn from(value: QueryPipelineStatisticFlags) -> vk::QueryPipelineStatisticFlags {
        let mut result = 0;
        if value.input_assembly_vertices {
            result |= vk::QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT;
        }
        if value.input_assembly_primitives {
            result |= vk::QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT;
        }
        if value.vertex_shader_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT;
        }
        if value.geometry_shader_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT;
        }
        if value.geometry_shader_primitives {
            result |= vk::QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT;
        }
        if value.clipping_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
        }
        if value.clipping_primitives {
            result |= vk::QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT;
        }
        if value.fragment_shader_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
        }
        if value.tessellation_control_shader_patches {
            result |= vk::QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT;
        }
        if value.tessellation_evaluation_shader_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT;
        }
        if value.compute_shader_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
        }
        result
    }
}

/// Flags to control how the results of a query should be retrieved.
///
/// `VK_QUERY_RESULT_64_BIT` is not included, as it is determined automatically via the
/// [`QueryResultElement`] trait.
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryResultFlags {
    /// Wait for the results to become available before writing the results.
    pub wait: bool,
    /// Write an additional element to the end of each query's results, indicating the availability
    /// of the results:
    /// - Nonzero: The results are available, and have been written to the element(s) preceding.
    /// - Zero: The results are not yet available, and have not been written.
    pub with_availability: bool,
    /// Allow writing partial results to the buffer, instead of waiting until they are fully
    /// available.
    pub partial: bool,
}

impl From<QueryResultFlags> for vk::QueryResultFlags {
    #[inline]
    fn from(value: QueryResultFlags) -> Self {
        let mut result = 0;
        if value.wait {
            result |= vk::QUERY_RESULT_WAIT_BIT;
        }
        if value.with_availability {
            result |= vk::QUERY_RESULT_WITH_AVAILABILITY_BIT;
        }
        if value.partial {
            result |= vk::QUERY_RESULT_PARTIAL_BIT;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::query::QueryPipelineStatisticFlags;
    use crate::query::QueryPool;
    use crate::query::QueryPoolCreationError;
    use crate::query::QueryType;

    #[test]
    fn pipeline_statistics_feature() {
        let (device, _) = gfx_dev_and_queue!();

        let ty = QueryType::PipelineStatistics(QueryPipelineStatisticFlags::none());
        match QueryPool::new(device, ty, 256) {
            Err(QueryPoolCreationError::PipelineStatisticsQueryFeatureNotEnabled) => (),
            _ => panic!(),
        };
    }
}
