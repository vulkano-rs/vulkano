// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! This module provides support for query pools.
//!
//! In Vulkan, queries are not created individually. Instead you manipulate **query pools**, which
//! represent a collection of queries. Whenever you use a query, you have to specify both the query
//! pool and the slot id within that query pool.

use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;

use device::Device;
use device::DeviceOwned;

use Error;
use OomError;
use VulkanObject;
use check_errors;
use vk;

pub struct UnsafeQueryPool {
    pool: vk::QueryPool,
    device: Arc<Device>,
    num_slots: u32,
}

impl UnsafeQueryPool {
    /// Builds a new query pool.
    pub fn new(device: Arc<Device>, ty: QueryType, num_slots: u32)
               -> Result<UnsafeQueryPool, QueryPoolCreationError> {
        let (vk_ty, statistics) = match ty {
            QueryType::Occlusion => (vk::QUERY_TYPE_OCCLUSION, 0),
            QueryType::Timestamp => (vk::QUERY_TYPE_TIMESTAMP, 0),
            QueryType::PipelineStatistics(flags) => {
                if !device.enabled_features().pipeline_statistics_query {
                    return Err(QueryPoolCreationError::PipelineStatisticsQueryFeatureNotEnabled);
                }

                (vk::QUERY_TYPE_PIPELINE_STATISTICS, flags.into())
            },
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

            let mut output = mem::uninitialized();
            let vk = device.pointers();
            check_errors(vk.CreateQueryPool(device.internal_object(),
                                            &infos,
                                            ptr::null(),
                                            &mut output))?;
            output
        };

        Ok(UnsafeQueryPool {
               pool: pool,
               device: device,
               num_slots: num_slots,
           })
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

    const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT;

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
}

#[derive(Debug, Copy, Clone)]
pub enum QueryType {
    Occlusion,
    PipelineStatistics(QueryPipelineStatisticFlags),
    Timestamp,
}

#[derive(Debug, Copy, Clone)]
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
}

impl Into<vk::QueryPipelineStatisticFlags> for QueryPipelineStatisticFlags {
    fn into(self) -> vk::QueryPipelineStatisticFlags {
        let mut result = 0;
        if self.input_assembly_vertices {
            result |= vk::QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT;
        }
        if self.input_assembly_primitives {
            result |= vk::QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT;
        }
        if self.vertex_shader_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT;
        }
        if self.geometry_shader_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT;
        }
        if self.geometry_shader_primitives {
            result |= vk::QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT;
        }
        if self.clipping_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
        }
        if self.clipping_primitives {
            result |= vk::QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT;
        }
        if self.fragment_shader_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
        }
        if self.tessellation_control_shader_patches {
            result |= vk::QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT;
        }
        if self.tessellation_evaluation_shader_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT;
        }
        if self.compute_shader_invocations {
            result |= vk::QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
        }
        result
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
    fn description(&self) -> &str {
        match *self {
            QueryPoolCreationError::OomError(_) => "not enough memory available",
            QueryPoolCreationError::PipelineStatisticsQueryFeatureNotEnabled => {
                "a pipeline statistics pool was requested but the corresponding feature \
                 wasn't enabled"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            QueryPoolCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for QueryPoolCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
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
                   },
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
    use query::OcclusionQueriesPool;
    use query::QueryPipelineStatisticFlags;
    use query::QueryPoolCreationError;
    use query::QueryType;
    use query::UnsafeQueryPool;

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
