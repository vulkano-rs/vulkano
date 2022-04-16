// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::TypedBufferAccess,
    command_buffer::{
        auto::QueryState,
        synced::{Command, Resource, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, BeginQueryError,
        CopyQueryPoolResultsError, EndQueryError, ResetQueryPoolError, WriteTimestampError,
    },
    device::{physical::QueueFamily, Device, DeviceOwned},
    query::{
        GetResultsError, QueriesRange, Query, QueryControlFlags, QueryPool, QueryResultElement,
        QueryResultFlags, QueryType,
    },
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStage, PipelineStages},
    DeviceSize, VulkanObject,
};
use std::{error, fmt, mem::size_of, ops::Range, sync::Arc};

/// # Commands related to queries.
impl<L, P> AutoCommandBufferBuilder<L, P> {
    /// Adds a command that begins a query.
    ///
    /// The query will be active until [`end_query`](Self::end_query) is called for the same query.
    ///
    /// # Safety
    /// The query must be unavailable, ensured by calling [`reset_query_pool`](Self::reset_query_pool).
    pub unsafe fn begin_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<&mut Self, BeginQueryError> {
        check_begin_query(self.device(), &query_pool, query, flags)?;

        match query_pool.query_type() {
            QueryType::Occlusion => {
                if !self.queue_family().supports_graphics() {
                    return Err(
                        AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into(),
                    );
                }
            }
            QueryType::PipelineStatistics(flags) => {
                if flags.is_compute() && !self.queue_family().supports_compute()
                    || flags.is_graphics() && !self.queue_family().supports_graphics()
                {
                    return Err(
                        AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into(),
                    );
                }
            }
            QueryType::Timestamp => unreachable!(),
        }

        let ty = query_pool.query_type();
        let raw_ty = ty.into();
        let raw_query_pool = query_pool.internal_object();
        if self.query_state.contains_key(&raw_ty) {
            return Err(AutoCommandBufferBuilderContextError::QueryIsActive.into());
        }

        // TODO: validity checks
        self.inner.begin_query(query_pool, query, flags);
        self.query_state.insert(
            raw_ty,
            QueryState {
                query_pool: raw_query_pool,
                query,
                ty,
                flags,
                in_subpass: self.render_pass_state.is_some(),
            },
        );

        Ok(self)
    }

    /// Adds a command that ends an active query.
    pub fn end_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
    ) -> Result<&mut Self, EndQueryError> {
        unsafe {
            check_end_query(self.device(), &query_pool, query)?;

            let raw_ty = query_pool.query_type().into();
            let raw_query_pool = query_pool.internal_object();
            if !self.query_state.get(&raw_ty).map_or(false, |state| {
                state.query_pool == raw_query_pool && state.query == query
            }) {
                return Err(AutoCommandBufferBuilderContextError::QueryNotActive.into());
            }

            self.inner.end_query(query_pool, query);
            self.query_state.remove(&raw_ty);
        }

        Ok(self)
    }

    /// Adds a command that writes a timestamp to a timestamp query.
    ///
    /// # Safety
    /// The query must be unavailable, ensured by calling [`reset_query_pool`](Self::reset_query_pool).
    pub unsafe fn write_timestamp(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        stage: PipelineStage,
    ) -> Result<&mut Self, WriteTimestampError> {
        check_write_timestamp(
            self.device(),
            self.queue_family(),
            &query_pool,
            query,
            stage,
        )?;

        if !(self.queue_family().supports_graphics()
            || self.queue_family().supports_compute()
            || self.queue_family().explicitly_supports_transfers())
        {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        // TODO: validity checks
        self.inner.write_timestamp(query_pool, query, stage);

        Ok(self)
    }

    /// Adds a command that copies the results of a range of queries to a buffer on the GPU.
    ///
    /// [`query_pool.ty().result_len()`](crate::query::QueryType::result_len) elements
    /// will be written for each query in the range, plus 1 extra element per query if
    /// [`QueryResultFlags::with_availability`] is enabled.
    /// The provided buffer must be large enough to hold the data.
    ///
    /// See also [`get_results`](crate::query::QueriesRange::get_results).
    pub fn copy_query_pool_results<D, T>(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
        destination: Arc<D>,
        flags: QueryResultFlags,
    ) -> Result<&mut Self, CopyQueryPoolResultsError>
    where
        D: TypedBufferAccess<Content = [T]> + 'static,
        T: QueryResultElement,
    {
        unsafe {
            self.ensure_outside_render_pass()?;
            let stride = check_copy_query_pool_results(
                self.device(),
                &query_pool,
                queries.clone(),
                destination.as_ref(),
                flags,
            )?;
            self.inner
                .copy_query_pool_results(query_pool, queries, destination, stride, flags)?;
        }

        Ok(self)
    }

    /// Adds a command to reset a range of queries on a query pool.
    ///
    /// The affected queries will be marked as "unavailable" after this command runs, and will no
    /// longer return any results. They will be ready to have new results recorded for them.
    ///
    /// # Safety
    /// The queries in the specified range must not be active in another command buffer.
    pub unsafe fn reset_query_pool(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
    ) -> Result<&mut Self, ResetQueryPoolError> {
        self.ensure_outside_render_pass()?;
        check_reset_query_pool(self.device(), &query_pool, queries.clone())?;

        let raw_query_pool = query_pool.internal_object();
        if self
            .query_state
            .values()
            .any(|state| state.query_pool == raw_query_pool && queries.contains(&state.query))
        {
            return Err(AutoCommandBufferBuilderContextError::QueryIsActive.into());
        }

        // TODO: validity checks
        // Do other command buffers actually matter here? Not sure on the Vulkan spec.
        self.inner.reset_query_pool(query_pool, queries);

        Ok(self)
    }
}

/// Checks whether a `begin_query` command is valid.
///
/// # Panic
///
/// - Panics if the query pool was not created with `device`.
fn check_begin_query(
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

    match query_pool.query_type() {
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
fn check_end_query(
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
fn check_write_timestamp(
    device: &Device,
    queue_family: QueueFamily,
    query_pool: &QueryPool,
    query: u32,
    stage: PipelineStage,
) -> Result<(), CheckWriteTimestampError> {
    assert_eq!(
        device.internal_object(),
        query_pool.device().internal_object(),
    );

    if !matches!(query_pool.query_type(), QueryType::Timestamp) {
        return Err(CheckWriteTimestampError::NotPermitted);
    }

    query_pool
        .query(query)
        .ok_or(CheckWriteTimestampError::OutOfRange)?;

    if queue_family.timestamp_valid_bits().is_none() {
        return Err(CheckWriteTimestampError::NoTimestampValidBits);
    }

    if !queue_family.supports_stage(stage) {
        return Err(CheckWriteTimestampError::StageNotSupported);
    }

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
    /// The queue family's `timestamp_valid_bits` value is `None`.
    NoTimestampValidBits,
    /// This operation is not permitted on this query type.
    NotPermitted,
    /// The provided query index is not valid for this pool.
    OutOfRange,
    /// The provided stage is not supported by the queue family.
    StageNotSupported,
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
                Self::NoTimestampValidBits => {
                    "the queue family's timestamp_valid_bits value is None"
                }
                Self::NotPermitted => {
                    "this operation is not permitted on this query type"
                }
                Self::OutOfRange => {
                    "the provided query index is not valid for this pool"
                }
                Self::StageNotSupported => {
                    "the provided stage is not supported by the queue family"
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
fn check_copy_query_pool_results<D, T>(
    device: &Device,
    query_pool: &QueryPool,
    queries: Range<u32>,
    destination: &D,
    flags: QueryResultFlags,
) -> Result<DeviceSize, CheckCopyQueryPoolResultsError>
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

    if !buffer_inner.buffer.usage().transfer_dst {
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
        required_len: DeviceSize,
        /// Actual number of elements in the buffer.
        actual_len: DeviceSize,
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
fn check_reset_query_pool(
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

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdBeginQuery` on the builder.
    #[inline]
    pub unsafe fn begin_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        flags: QueryControlFlags,
    ) {
        struct Cmd {
            query_pool: Arc<QueryPool>,
            query: u32,
            flags: QueryControlFlags,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "begin_query"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.begin_query(self.query_pool.query(self.query).unwrap(), self.flags);
            }
        }

        self.commands.push(Box::new(Cmd {
            query_pool,
            query,
            flags,
        }));
    }

    /// Calls `vkCmdEndQuery` on the builder.
    #[inline]
    pub unsafe fn end_query(&mut self, query_pool: Arc<QueryPool>, query: u32) {
        struct Cmd {
            query_pool: Arc<QueryPool>,
            query: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "end_query"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.end_query(self.query_pool.query(self.query).unwrap());
            }
        }

        self.commands.push(Box::new(Cmd { query_pool, query }));
    }

    /// Calls `vkCmdWriteTimestamp` on the builder.
    #[inline]
    pub unsafe fn write_timestamp(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        stage: PipelineStage,
    ) {
        struct Cmd {
            query_pool: Arc<QueryPool>,
            query: u32,
            stage: PipelineStage,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "write_timestamp"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.write_timestamp(self.query_pool.query(self.query).unwrap(), self.stage);
            }
        }

        self.commands.push(Box::new(Cmd {
            query_pool,
            query,
            stage,
        }));
    }

    /// Calls `vkCmdCopyQueryPoolResults` on the builder.
    ///
    /// # Safety
    /// `stride` must be at least the number of bytes that will be written by each query.
    pub unsafe fn copy_query_pool_results<D, T>(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
        destination: Arc<D>,
        stride: DeviceSize,
        flags: QueryResultFlags,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        D: TypedBufferAccess<Content = [T]> + 'static,
        T: QueryResultElement,
    {
        struct Cmd<D> {
            query_pool: Arc<QueryPool>,
            queries: Range<u32>,
            destination: Arc<D>,
            stride: DeviceSize,
            flags: QueryResultFlags,
        }

        impl<D, T> Command for Cmd<D>
        where
            D: TypedBufferAccess<Content = [T]> + 'static,
            T: QueryResultElement,
        {
            fn name(&self) -> &'static str {
                "copy_query_pool_results"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_query_pool_results(
                    self.query_pool.queries_range(self.queries.clone()).unwrap(),
                    self.destination.as_ref(),
                    self.stride,
                    self.flags,
                );
            }
        }

        let resources = [(
            "destination".into(),
            Resource::Buffer {
                buffer: destination.clone(),
                range: 0..destination.size(), // TODO:
                memory: PipelineMemoryAccess {
                    stages: PipelineStages {
                        transfer: true,
                        ..PipelineStages::none()
                    },
                    access: AccessFlags {
                        transfer_write: true,
                        ..AccessFlags::none()
                    },
                    exclusive: true,
                },
            },
        )];

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            query_pool,
            queries,
            destination,
            stride,
            flags,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdResetQueryPool` on the builder.
    #[inline]
    pub unsafe fn reset_query_pool(&mut self, query_pool: Arc<QueryPool>, queries: Range<u32>) {
        struct Cmd {
            query_pool: Arc<QueryPool>,
            queries: Range<u32>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "reset_query_pool"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.reset_query_pool(self.query_pool.queries_range(self.queries.clone()).unwrap());
            }
        }

        self.commands.push(Box::new(Cmd {
            query_pool,
            queries,
        }));
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdBeginQuery` on the builder.
    #[inline]
    pub unsafe fn begin_query(&mut self, query: Query, flags: QueryControlFlags) {
        let fns = self.device.fns();
        let flags = if flags.precise {
            ash::vk::QueryControlFlags::PRECISE
        } else {
            ash::vk::QueryControlFlags::empty()
        };
        fns.v1_0.cmd_begin_query(
            self.handle,
            query.pool().internal_object(),
            query.index(),
            flags,
        );
    }

    /// Calls `vkCmdEndQuery` on the builder.
    #[inline]
    pub unsafe fn end_query(&mut self, query: Query) {
        let fns = self.device.fns();
        fns.v1_0
            .cmd_end_query(self.handle, query.pool().internal_object(), query.index());
    }

    /// Calls `vkCmdWriteTimestamp` on the builder.
    #[inline]
    pub unsafe fn write_timestamp(&mut self, query: Query, stage: PipelineStage) {
        let fns = self.device.fns();
        fns.v1_0.cmd_write_timestamp(
            self.handle,
            stage.into(),
            query.pool().internal_object(),
            query.index(),
        );
    }

    /// Calls `vkCmdCopyQueryPoolResults` on the builder.
    #[inline]
    pub unsafe fn copy_query_pool_results<D, T>(
        &mut self,
        queries: QueriesRange,
        destination: &D,
        stride: DeviceSize,
        flags: QueryResultFlags,
    ) where
        D: TypedBufferAccess<Content = [T]>,
        T: QueryResultElement,
    {
        let destination = destination.inner();
        let range = queries.range();
        debug_assert!(destination.offset < destination.buffer.size());
        debug_assert!(destination.buffer.usage().transfer_dst);
        debug_assert!(destination.offset % size_of::<T>() as DeviceSize == 0);
        debug_assert!(stride % size_of::<T>() as DeviceSize == 0);

        let fns = self.device.fns();
        fns.v1_0.cmd_copy_query_pool_results(
            self.handle,
            queries.pool().internal_object(),
            range.start,
            range.end - range.start,
            destination.buffer.internal_object(),
            destination.offset,
            stride,
            ash::vk::QueryResultFlags::from(flags) | T::FLAG,
        );
    }

    /// Calls `vkCmdResetQueryPool` on the builder.
    #[inline]
    pub unsafe fn reset_query_pool(&mut self, queries: QueriesRange) {
        let range = queries.range();
        let fns = self.device.fns();
        fns.v1_0.cmd_reset_query_pool(
            self.handle,
            queries.pool().internal_object(),
            range.start,
            range.end - range.start,
        );
    }
}
