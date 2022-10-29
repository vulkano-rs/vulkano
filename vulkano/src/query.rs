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

use crate::{
    device::{Device, DeviceOwned},
    macros::vulkan_bitflags,
    DeviceSize, OomError, RequirementNotMet, RequiresOneOf, VulkanError, VulkanObject,
};
use std::{
    error::Error,
    ffi::c_void,
    fmt::{Display, Error as FmtError, Formatter},
    mem::{size_of_val, MaybeUninit},
    num::NonZeroU64,
    ops::Range,
    ptr,
    sync::Arc,
};

/// A collection of one or more queries of a particular type.
#[derive(Debug)]
pub struct QueryPool {
    handle: ash::vk::QueryPool,
    device: Arc<Device>,
    id: NonZeroU64,

    query_type: QueryType,
    query_count: u32,
}

impl QueryPool {
    /// Creates a new `QueryPool`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.query_count` is `0`.
    pub fn new(
        device: Arc<Device>,
        create_info: QueryPoolCreateInfo,
    ) -> Result<Arc<QueryPool>, QueryPoolCreationError> {
        let QueryPoolCreateInfo {
            query_type,
            query_count,
            _ne: _,
        } = create_info;

        // VUID-VkQueryPoolCreateInfo-queryCount-02763
        assert!(query_count != 0);

        let pipeline_statistics = match query_type {
            QueryType::PipelineStatistics(flags) => {
                // VUID-VkQueryPoolCreateInfo-queryType-00791
                if !device.enabled_features().pipeline_statistics_query {
                    return Err(QueryPoolCreationError::PipelineStatisticsQueryFeatureNotEnabled);
                }

                // VUID-VkQueryPoolCreateInfo-queryType-00792
                flags.into()
            }
            QueryType::Occlusion | QueryType::Timestamp => {
                ash::vk::QueryPipelineStatisticFlags::empty()
            }
        };

        let create_info = ash::vk::QueryPoolCreateInfo {
            flags: ash::vk::QueryPoolCreateFlags::empty(),
            query_type: query_type.into(),
            query_count,
            pipeline_statistics,
            ..Default::default()
        };

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_query_pool)(
                device.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(QueryPool {
            handle,
            device,
            id: Self::next_id(),
            query_type,
            query_count,
        }))
    }

    /// Creates a new `QueryPool` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::QueryPool,
        create_info: QueryPoolCreateInfo,
    ) -> Arc<QueryPool> {
        let QueryPoolCreateInfo {
            query_type,
            query_count,
            _ne: _,
        } = create_info;

        Arc::new(QueryPool {
            handle,
            device,
            id: Self::next_id(),
            query_type,
            query_count,
        })
    }

    /// Returns the query type of the pool.
    #[inline]
    pub fn query_type(&self) -> QueryType {
        self.query_type
    }

    /// Returns the number of query slots of this query pool.
    #[inline]
    pub fn query_count(&self) -> u32 {
        self.query_count
    }

    /// Returns a reference to a single query slot, or `None` if the index is out of range.
    #[inline]
    pub fn query(&self, index: u32) -> Option<Query<'_>> {
        if index < self.query_count {
            Some(Query { pool: self, index })
        } else {
            None
        }
    }

    /// Returns a reference to a range of queries, or `None` if out of range.
    ///
    /// # Panics
    ///
    /// - Panics if the range is empty.
    #[inline]
    pub fn queries_range(&self, range: Range<u32>) -> Option<QueriesRange<'_>> {
        assert!(!range.is_empty());

        if range.end <= self.query_count {
            Some(QueriesRange { pool: self, range })
        } else {
            None
        }
    }
}

impl Drop for QueryPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_query_pool)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for QueryPool {
    type Handle = ash::vk::QueryPool;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for QueryPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

crate::impl_id_counter!(QueryPool);

/// Parameters to create a new `QueryPool`.
#[derive(Clone, Debug)]
pub struct QueryPoolCreateInfo {
    /// The type of query that the pool should be for.
    ///
    /// There is no default value.
    pub query_type: QueryType,

    /// The number of queries to create in the pool.
    ///
    /// The default value is `0`, which must be overridden.
    pub query_count: u32,

    pub _ne: crate::NonExhaustive,
}

impl QueryPoolCreateInfo {
    /// Returns a `QueryPoolCreateInfo` with the specified `query_type`.
    #[inline]
    pub fn query_type(query_type: QueryType) -> Self {
        Self {
            query_type,
            query_count: 0,
            _ne: crate::NonExhaustive(()),
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

impl Error for QueryPoolCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            QueryPoolCreationError::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for QueryPoolCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
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
    fn from(err: OomError) -> QueryPoolCreationError {
        QueryPoolCreationError::OomError(err)
    }
}

impl From<VulkanError> for QueryPoolCreationError {
    fn from(err: VulkanError) -> QueryPoolCreationError {
        match err {
            err @ VulkanError::OutOfHostMemory => {
                QueryPoolCreationError::OomError(OomError::from(err))
            }
            err @ VulkanError::OutOfDeviceMemory => {
                QueryPoolCreationError::OomError(OomError::from(err))
            }
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
        self.pool
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
        self.pool
    }

    /// Returns the range of queries represented.
    #[inline]
    pub fn range(&self) -> Range<u32> {
        self.range.clone()
    }

    /// Copies the results of this range of queries to a buffer on the CPU.
    ///
    /// [`self.pool().ty().result_len()`](QueryType::result_len) elements
    /// will be written for each query in the range, plus 1 extra element per query if
    /// [`QueryResultFlags::with_availability`] is enabled.
    /// The provided buffer must be large enough to hold the data.
    ///
    /// `true` is returned if every result was available and written to the buffer. `false`
    /// is returned if some results were not yet available; these will not be written to the buffer.
    ///
    /// See also [`copy_query_pool_results`](crate::command_buffer::AutoCommandBufferBuilder::copy_query_pool_results).
    #[inline]
    pub fn get_results<T>(
        &self,
        destination: &mut [T],
        flags: QueryResultFlags,
    ) -> Result<bool, GetResultsError>
    where
        T: QueryResultElement,
    {
        let stride = self.check_query_pool_results::<T>(
            destination.as_ptr() as DeviceSize,
            destination.len() as DeviceSize,
            flags,
        )?;

        let result = unsafe {
            let fns = self.pool.device.fns();
            (fns.v1_0.get_query_pool_results)(
                self.pool.device.handle(),
                self.pool.handle(),
                self.range.start,
                self.range.end - self.range.start,
                size_of_val(destination),
                destination.as_mut_ptr() as *mut c_void,
                stride,
                ash::vk::QueryResultFlags::from(flags) | T::FLAG,
            )
        };

        match result {
            ash::vk::Result::SUCCESS => Ok(true),
            ash::vk::Result::NOT_READY => Ok(false),
            err => Err(VulkanError::from(err).into()),
        }
    }

    pub(crate) fn check_query_pool_results<T>(
        &self,
        buffer_start: DeviceSize,
        buffer_len: DeviceSize,
        flags: QueryResultFlags,
    ) -> Result<DeviceSize, GetResultsError>
    where
        T: QueryResultElement,
    {
        // VUID-vkGetQueryPoolResults-flags-parameter
        // VUID-vkCmdCopyQueryPoolResults-flags-parameter
        flags.validate_device(&self.pool.device)?;

        assert!(buffer_len > 0);

        // VUID-vkGetQueryPoolResults-flags-02828
        // VUID-vkGetQueryPoolResults-flags-00815
        debug_assert!(buffer_start % std::mem::size_of::<T>() as DeviceSize == 0);

        let count = self.range.end - self.range.start;
        let per_query_len =
            self.pool.query_type.result_len() + flags.with_availability as DeviceSize;
        let required_len = per_query_len * count as DeviceSize;

        // VUID-vkGetQueryPoolResults-dataSize-00817
        if buffer_len < required_len {
            return Err(GetResultsError::BufferTooSmall {
                required_len: required_len as DeviceSize,
                actual_len: buffer_len as DeviceSize,
            });
        }

        match self.pool.query_type {
            QueryType::Occlusion => (),
            QueryType::PipelineStatistics(_) => (),
            QueryType::Timestamp => {
                // VUID-vkGetQueryPoolResults-queryType-00818
                if flags.partial {
                    return Err(GetResultsError::InvalidFlags);
                }
            }
        }

        Ok(per_query_len * std::mem::size_of::<T>() as DeviceSize)
    }
}

/// Error that can happen when calling [`QueriesRange::get_results`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GetResultsError {
    /// The connection to the device has been lost.
    DeviceLost,

    /// Not enough memory.
    OomError(OomError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The buffer is too small for the operation.
    BufferTooSmall {
        /// Required number of elements in the buffer.
        required_len: DeviceSize,
        /// Actual number of elements in the buffer.
        actual_len: DeviceSize,
    },

    /// The provided flags are not allowed for this type of query.
    InvalidFlags,
}

impl Error for GetResultsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for GetResultsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::DeviceLost => write!(f, "the connection to the device has been lost"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::BufferTooSmall { .. } => write!(f, "the buffer is too small for the operation"),
            Self::InvalidFlags => write!(
                f,
                "the provided flags are not allowed for this type of query"
            ),
        }
    }
}

impl From<VulkanError> for GetResultsError {
    fn from(err: VulkanError) -> Self {
        match err {
            VulkanError::OutOfHostMemory | VulkanError::OutOfDeviceMemory => {
                Self::OomError(OomError::from(err))
            }
            VulkanError::DeviceLost => Self::DeviceLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for GetResultsError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<RequirementNotMet> for GetResultsError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

/// A trait for elements of buffers that can be used as a destination for query results.
///
/// # Safety
/// This is implemented for `u32` and `u64`. Unless you really know what you're doing, you should
/// not implement this trait for any other type.
pub unsafe trait QueryResultElement {
    const FLAG: ash::vk::QueryResultFlags;
}

unsafe impl QueryResultElement for u32 {
    const FLAG: ash::vk::QueryResultFlags = ash::vk::QueryResultFlags::empty();
}

unsafe impl QueryResultElement for u64 {
    const FLAG: ash::vk::QueryResultFlags = ash::vk::QueryResultFlags::TYPE_64;
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
    pub const fn result_len(&self) -> DeviceSize {
        match self {
            Self::Occlusion | Self::Timestamp => 1,
            Self::PipelineStatistics(flags) => flags.count(),
        }
    }
}

impl From<QueryType> for ash::vk::QueryType {
    #[inline]
    fn from(value: QueryType) -> Self {
        match value {
            QueryType::Occlusion => ash::vk::QueryType::OCCLUSION,
            QueryType::PipelineStatistics(_) => ash::vk::QueryType::PIPELINE_STATISTICS,
            QueryType::Timestamp => ash::vk::QueryType::TIMESTAMP,
        }
    }
}

vulkan_bitflags! {
    /// Flags that control how a query is to be executed.
    #[non_exhaustive]
    QueryControlFlags = QueryControlFlags(u32);

    /// For occlusion queries, specifies that the result must reflect the exact number of
    /// tests passed. If not enabled, the query may return a result of 1 even if more fragments
    /// passed the test.
    precise = PRECISE,
}

vulkan_bitflags! {
    /// For pipeline statistics queries, the statistics that should be gathered.
    #[non_exhaustive]
    QueryPipelineStatisticFlags = QueryPipelineStatisticFlags(u32);

    /// Count the number of vertices processed by the input assembly.
    input_assembly_vertices = INPUT_ASSEMBLY_VERTICES,

    /// Count the number of primitives processed by the input assembly.
    input_assembly_primitives = INPUT_ASSEMBLY_PRIMITIVES,

    /// Count the number of times a vertex shader is invoked.
    vertex_shader_invocations = VERTEX_SHADER_INVOCATIONS,

    /// Count the number of times a geometry shader is invoked.
    geometry_shader_invocations = GEOMETRY_SHADER_INVOCATIONS,

    /// Count the number of primitives generated by geometry shaders.
    geometry_shader_primitives = GEOMETRY_SHADER_PRIMITIVES,

    /// Count the number of times the clipping stage is invoked on a primitive.
    clipping_invocations = CLIPPING_INVOCATIONS,

    /// Count the number of primitives that are output by the clipping stage.
    clipping_primitives = CLIPPING_PRIMITIVES,

    /// Count the number of times a fragment shader is invoked.
    fragment_shader_invocations = FRAGMENT_SHADER_INVOCATIONS,

    /// Count the number of patches processed by a tessellation control shader.
    tessellation_control_shader_patches = TESSELLATION_CONTROL_SHADER_PATCHES,

    /// Count the number of times a tessellation evaluation shader is invoked.
    tessellation_evaluation_shader_invocations = TESSELLATION_EVALUATION_SHADER_INVOCATIONS,

    /// Count the number of times a compute shader is invoked.
    compute_shader_invocations = COMPUTE_SHADER_INVOCATIONS,

    /*
    // TODO: document
    task_shader_invocations = TASK_SHADER_INVOCATIONS_NV {
        device_extensions: [nv_mesh_shader],
    },

    // TODO: document
    mesh_shader_invocations = MESH_SHADER_INVOCATIONS_NV {
        device_extensions: [nv_mesh_shader],
    },
     */
}

impl QueryPipelineStatisticFlags {
    /// Returns the number of flags that are set to `true`.
    #[inline]
    pub const fn count(&self) -> DeviceSize {
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
            _ne: _,
        } = self;
        input_assembly_vertices as DeviceSize
            + input_assembly_primitives as DeviceSize
            + vertex_shader_invocations as DeviceSize
            + geometry_shader_invocations as DeviceSize
            + geometry_shader_primitives as DeviceSize
            + clipping_invocations as DeviceSize
            + clipping_primitives as DeviceSize
            + fragment_shader_invocations as DeviceSize
            + tessellation_control_shader_patches as DeviceSize
            + tessellation_evaluation_shader_invocations as DeviceSize
            + compute_shader_invocations as DeviceSize
    }

    /// Returns `true` if any flags referring to compute operations are set to `true`.
    #[inline]
    pub const fn is_compute(&self) -> bool {
        let &Self {
            compute_shader_invocations,
            ..
        } = self;
        compute_shader_invocations
    }

    /// Returns `true` if any flags referring to graphics operations are set to `true`.
    #[inline]
    pub const fn is_graphics(&self) -> bool {
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
            ..
        } = self;
        input_assembly_vertices
            || input_assembly_primitives
            || vertex_shader_invocations
            || geometry_shader_invocations
            || geometry_shader_primitives
            || clipping_invocations
            || clipping_primitives
            || fragment_shader_invocations
            || tessellation_control_shader_patches
            || tessellation_evaluation_shader_invocations
    }
}

vulkan_bitflags! {
    /// Flags to control how the results of a query should be retrieved.
    ///
    /// `VK_QUERY_RESULT_64_BIT` is not included, as it is determined automatically via the
    /// [`QueryResultElement`] trait.
    #[non_exhaustive]
    QueryResultFlags = QueryResultFlags(u32);

    /// Wait for the results to become available before writing the results.
    wait = WAIT,

    /// Write an additional element to the end of each query's results, indicating the availability
    /// of the results:
    /// - Nonzero: The results are available, and have been written to the element(s) preceding.
    /// - Zero: The results are not yet available, and have not been written.
    with_availability = WITH_AVAILABILITY,

    /// Allow writing partial results to the buffer, instead of waiting until they are fully
    /// available.
    partial = PARTIAL,

    /*
    // TODO: document
    with_status = WITH_STATUS_KHR {
        device_extensions: [khr_video_queue],
    },
     */
}

#[cfg(test)]
mod tests {
    use super::QueryPoolCreateInfo;
    use crate::query::{QueryPipelineStatisticFlags, QueryPool, QueryPoolCreationError, QueryType};

    #[test]
    fn pipeline_statistics_feature() {
        let (device, _) = gfx_dev_and_queue!();
        let query_type = QueryType::PipelineStatistics(QueryPipelineStatisticFlags::empty());
        match QueryPool::new(
            device,
            QueryPoolCreateInfo {
                query_count: 256,
                ..QueryPoolCreateInfo::query_type(query_type)
            },
        ) {
            Err(QueryPoolCreationError::PipelineStatisticsQueryFeatureNotEnabled) => (),
            _ => panic!(),
        };
    }
}
