// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! This module provides support for query pools.
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
use std::ptr;
use std::sync::Arc;

#[derive(Debug)]
pub struct UnsafeQueryPool {
    pool: vk::QueryPool,
    device: Arc<Device>,
    num_slots: u32,
    ty: QueryType,
}

impl UnsafeQueryPool {
    /// Builds a new query pool.
    pub fn new(
        device: Arc<Device>,
        ty: QueryType,
        num_slots: u32,
    ) -> Result<UnsafeQueryPool, QueryPoolCreationError> {
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

        Ok(UnsafeQueryPool {
            pool,
            device,
            num_slots,
            ty,
        })
    }

    #[inline]
    pub fn ty(&self) -> QueryType {
        self.ty
    }

    /// Returns the number of slots of that query pool.
    #[inline]
    pub fn num_slots(&self) -> u32 {
        self.num_slots
    }

    #[inline]
    pub fn query(&self, index: u32) -> Option<UnsafeQuery> {
        if index < self.num_slots() {
            Some(UnsafeQuery { pool: self, index })
        } else {
            None
        }
    }

    ///
    /// # Panic
    ///
    /// Panics if `count` is 0.
    #[inline]
    pub fn queries_range(&self, first_index: u32, count: u32) -> Option<UnsafeQueriesRange> {
        assert!(count >= 1);

        if first_index + count < self.num_slots() {
            Some(UnsafeQueriesRange {
                pool: self,
                first: first_index,
                count,
            })
        } else {
            None
        }
    }
}

unsafe impl VulkanObject for UnsafeQueryPool {
    type Object = vk::QueryPool;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_QUERY_POOL;

    #[inline]
    fn internal_object(&self) -> vk::QueryPool {
        self.pool
    }
}

unsafe impl DeviceOwned for UnsafeQueryPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Drop for UnsafeQueryPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyQueryPool(self.device.internal_object(), self.pool, ptr::null());
        }
    }
}

/// Error that can happen when creating a buffer.
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

pub struct UnsafeQuery<'a> {
    pool: &'a UnsafeQueryPool,
    index: u32,
}

impl<'a> UnsafeQuery<'a> {
    #[inline]
    pub fn pool(&self) -> &'a UnsafeQueryPool {
        &self.pool
    }

    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

pub struct UnsafeQueriesRange<'a> {
    pool: &'a UnsafeQueryPool,
    first: u32,
    count: u32,
}

impl<'a> UnsafeQueriesRange<'a> {
    #[inline]
    pub fn pool(&self) -> &'a UnsafeQueryPool {
        &self.pool
    }

    #[inline]
    pub fn first_index(&self) -> u32 {
        self.first
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn get_results<T>(
        &self,
        data: &mut [T],
        flags: QueryResultFlags,
    ) -> Result<bool, GetResultsError>
    where
        T: QueryResultElement,
    {
        assert!(!data.is_empty());
        debug_assert_eq!(std::mem::align_of_val(data), std::mem::size_of::<T>());

        let per_query_len = self.pool.ty.data_size() + flags.with_availability as usize;
        let required_len = per_query_len * self.count as usize;

        if data.len() < required_len {
            return Err(GetResultsError::BufferTooSmall {
                required_len,
                actual_len: data.len(),
            });
        }

        let stride = per_query_len * std::mem::size_of::<T>();

        let result = unsafe {
            let vk = self.pool.device.pointers();
            check_errors(vk.GetQueryPoolResults(
                self.pool.device.internal_object(),
                self.pool.internal_object(),
                self.first,
                self.count,
                std::mem::size_of_val(data),
                data as *mut _ as *mut c_void,
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
}

/// Error that can happen when calling `UnsafeQueriesRange::get_results`.
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

pub unsafe trait QueryResultElement {
    const FLAG: vk::QueryResultFlags;
}

unsafe impl QueryResultElement for u32 {
    const FLAG: vk::QueryResultFlags = 0;
}

unsafe impl QueryResultElement for u64 {
    const FLAG: vk::QueryResultFlags = vk::QUERY_RESULT_64_BIT;
}

#[derive(Debug, Copy, Clone)]
pub enum QueryType {
    Occlusion,
    PipelineStatistics(QueryPipelineStatisticFlags),
    Timestamp,
}

impl QueryType {
    #[inline]
    pub fn data_size(&self) -> usize {
        match self {
            Self::Occlusion | Self::Timestamp => 1,
            Self::PipelineStatistics(flags) => flags.count_bits(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QueryControlFlags {
    pub precise: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QueryPipelineStatisticFlags {
    pub input_assembly_vertices: bool,
    pub input_assembly_primitives: bool,
    pub vertex_shader_invocations: bool,
    pub geometry_shader_invocations: bool,
    pub geometry_shader_primitives: bool,
    pub clipping_invocations: bool,
    pub clipping_primitives: bool,
    pub fragment_shader_invocations: bool,
    pub tessellation_control_shader_patches: bool,
    pub tessellation_evaluation_shader_invocations: bool,
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

    #[inline]
    pub fn count_bits(&self) -> usize {
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

#[derive(Clone, Copy, Debug, Default)]
pub struct QueryResultFlags {
    pub wait: bool,
    pub with_availability: bool,
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

pub struct OcclusionQueriesPool {
    inner: UnsafeQueryPool,
}

impl OcclusionQueriesPool {
    /// See the docs of new().
    pub fn raw(device: Arc<Device>, num_slots: u32) -> Result<OcclusionQueriesPool, OomError> {
        Ok(OcclusionQueriesPool {
            inner: match UnsafeQueryPool::new(device, QueryType::Occlusion, num_slots) {
                Ok(q) => q,
                Err(QueryPoolCreationError::OomError(err)) => return Err(err),
                Err(QueryPoolCreationError::PipelineStatisticsQueryFeatureNotEnabled) => {
                    unreachable!()
                }
            },
        })
    }

    /// Builds a new query pool.
    ///
    /// # Panic
    ///
    /// - Panics if the device or host ran out of memory.
    ///
    #[inline]
    pub fn new(device: Arc<Device>, num_slots: u32) -> Arc<OcclusionQueriesPool> {
        Arc::new(OcclusionQueriesPool::raw(device, num_slots).unwrap())
    }

    /// Returns the number of slots of that query pool.
    #[inline]
    pub fn num_slots(&self) -> u32 {
        self.inner.num_slots()
    }
}

unsafe impl DeviceOwned for OcclusionQueriesPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

#[cfg(test)]
mod tests {
    use crate::query::OcclusionQueriesPool;
    use crate::query::QueryPipelineStatisticFlags;
    use crate::query::QueryPoolCreationError;
    use crate::query::QueryType;
    use crate::query::UnsafeQueryPool;

    #[test]
    fn occlusion_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = OcclusionQueriesPool::new(device, 256);
    }

    #[test]
    fn pipeline_statistics_feature() {
        let (device, _) = gfx_dev_and_queue!();

        let ty = QueryType::PipelineStatistics(QueryPipelineStatisticFlags::none());
        match UnsafeQueryPool::new(device, ty, 256) {
            Err(QueryPoolCreationError::PipelineStatisticsQueryFeatureNotEnabled) => (),
            _ => panic!(),
        };
    }
}
