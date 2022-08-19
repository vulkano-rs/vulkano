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
        allocator::CommandBufferAllocator,
        auto::{QueryState, RenderPassStateType},
        synced::{Command, Resource, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, CommandBufferInheritanceRenderPassType,
    },
    device::{physical::QueueFamily, DeviceOwned},
    query::{
        QueriesRange, Query, QueryControlFlags, QueryPool, QueryResultElement, QueryResultFlags,
        QueryType,
    },
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStage, PipelineStages},
    DeviceSize, VulkanObject,
};
use std::{error::Error, fmt, mem::size_of, ops::Range, sync::Arc};

/// # Commands related to queries.
impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Begins a query.
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
    ) -> Result<&mut Self, QueryError> {
        self.validate_begin_query(&query_pool, query, flags)?;

        let ty = query_pool.query_type();
        let raw_query_pool = query_pool.internal_object();

        self.inner.begin_query(query_pool, query, flags);
        self.query_state.insert(
            ty.into(),
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

    fn validate_begin_query(
        &self,
        query_pool: &QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<(), QueryError> {
        // VUID-vkCmdBeginQuery-commandBuffer-cmdpool
        if !(self.queue_family().supports_graphics() || self.queue_family().supports_compute()) {
            return Err(QueryError::NotSupportedByQueueFamily);
        }

        let device = self.device();

        // VUID-vkCmdBeginQuery-commonparent
        assert_eq!(device, query_pool.device());

        // VUID-vkCmdBeginQuery-query-00802
        query_pool.query(query).ok_or(QueryError::OutOfRange)?;

        match query_pool.query_type() {
            QueryType::Occlusion => {
                // VUID-vkCmdBeginQuery-commandBuffer-cmdpool
                // // VUID-vkCmdBeginQuery-queryType-00803
                if !self.queue_family().supports_graphics() {
                    return Err(QueryError::NotSupportedByQueueFamily);
                }

                // VUID-vkCmdBeginQuery-queryType-00800
                if flags.precise && !device.enabled_features().occlusion_query_precise {
                    return Err(QueryError::FeatureNotEnabled {
                        feature: "occlusion_query_precise",
                        reason: "flags.precise was enabled",
                    });
                }
            }
            QueryType::PipelineStatistics(statistic_flags) => {
                // VUID-vkCmdBeginQuery-commandBuffer-cmdpool
                // VUID-vkCmdBeginQuery-queryType-00804
                // VUID-vkCmdBeginQuery-queryType-00805
                if statistic_flags.is_compute() && !self.queue_family().supports_compute()
                    || statistic_flags.is_graphics() && !self.queue_family().supports_graphics()
                {
                    return Err(QueryError::NotSupportedByQueueFamily);
                }

                // VUID-vkCmdBeginQuery-queryType-00800
                if flags.precise {
                    return Err(QueryError::InvalidFlags);
                }
            }
            // VUID-vkCmdBeginQuery-queryType-02804
            QueryType::Timestamp => return Err(QueryError::NotPermitted),
        }

        // VUID-vkCmdBeginQuery-queryPool-01922
        if self
            .query_state
            .contains_key(&query_pool.query_type().into())
        {
            return Err(QueryError::QueryIsActive);
        }

        if let Some(state) = &self.render_pass_state {
            let view_mask = match &state.render_pass {
                RenderPassStateType::BeginRenderPass(state) => {
                    state.subpass.subpass_desc().view_mask
                }
                RenderPassStateType::BeginRendering(state) => state.view_mask,
                RenderPassStateType::Inherited => match self
                    .inheritance_info
                    .as_ref()
                    .unwrap()
                    .render_pass
                    .as_ref()
                    .unwrap()
                {
                    CommandBufferInheritanceRenderPassType::BeginRenderPass(info) => {
                        info.subpass.subpass_desc().view_mask
                    }
                    CommandBufferInheritanceRenderPassType::BeginRendering(info) => info.view_mask,
                },
            };

            // VUID-vkCmdBeginQuery-query-00808
            if query + view_mask.count_ones() > query_pool.query_count() {
                return Err(QueryError::OutOfRangeMultiview);
            }
        }

        // VUID-vkCmdBeginQuery-None-00807
        // Not checked, therefore unsafe.
        // TODO: add check.

        Ok(())
    }

    /// Ends an active query.
    pub fn end_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
    ) -> Result<&mut Self, QueryError> {
        self.validate_end_query(&query_pool, query)?;

        unsafe {
            let raw_ty = query_pool.query_type().into();
            self.inner.end_query(query_pool, query);
            self.query_state.remove(&raw_ty);
        }

        Ok(self)
    }

    fn validate_end_query(&self, query_pool: &QueryPool, query: u32) -> Result<(), QueryError> {
        // VUID-vkCmdEndQuery-commandBuffer-cmdpool
        if !(self.queue_family().supports_graphics() || self.queue_family().supports_compute()) {
            return Err(QueryError::NotSupportedByQueueFamily);
        }

        let device = self.device();

        // VUID-vkCmdEndQuery-commonparent
        assert_eq!(device, query_pool.device());

        // VUID-vkCmdEndQuery-None-01923
        if !self
            .query_state
            .get(&query_pool.query_type().into())
            .map_or(false, |state| {
                state.query_pool == query_pool.internal_object() && state.query == query
            })
        {
            return Err(QueryError::QueryNotActive);
        }

        // VUID-vkCmdEndQuery-query-00810
        query_pool.query(query).ok_or(QueryError::OutOfRange)?;

        if let Some(state) = &self.render_pass_state {
            let view_mask = match &state.render_pass {
                RenderPassStateType::BeginRenderPass(state) => {
                    state.subpass.subpass_desc().view_mask
                }
                RenderPassStateType::BeginRendering(state) => state.view_mask,
                RenderPassStateType::Inherited => match self
                    .inheritance_info
                    .as_ref()
                    .unwrap()
                    .render_pass
                    .as_ref()
                    .unwrap()
                {
                    CommandBufferInheritanceRenderPassType::BeginRenderPass(info) => {
                        info.subpass.subpass_desc().view_mask
                    }
                    CommandBufferInheritanceRenderPassType::BeginRendering(info) => info.view_mask,
                },
            };

            // VUID-vkCmdEndQuery-query-00812
            if query + view_mask.count_ones() > query_pool.query_count() {
                return Err(QueryError::OutOfRangeMultiview);
            }
        }

        Ok(())
    }

    /// Writes a timestamp to a timestamp query.
    ///
    /// # Safety
    /// The query must be unavailable, ensured by calling [`reset_query_pool`](Self::reset_query_pool).
    pub unsafe fn write_timestamp(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        stage: PipelineStage,
    ) -> Result<&mut Self, QueryError> {
        self.validate_write_timestamp(self.queue_family(), &query_pool, query, stage)?;

        self.inner.write_timestamp(query_pool, query, stage);

        Ok(self)
    }

    fn validate_write_timestamp(
        &self,
        queue_family: QueueFamily,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
    ) -> Result<(), QueryError> {
        // VUID-vkCmdWriteTimestamp-commandBuffer-cmdpool
        if !(self.queue_family().explicitly_supports_transfers()
            || self.queue_family().supports_graphics()
            || self.queue_family().supports_compute())
        {
            return Err(QueryError::NotSupportedByQueueFamily);
        }

        let device = self.device();

        // VUID-vkCmdWriteTimestamp-commonparent
        assert_eq!(device, query_pool.device());

        // VUID-vkCmdWriteTimestamp-pipelineStage-04074
        if !queue_family.supports_stage(stage) {
            return Err(QueryError::StageNotSupported);
        }

        match stage {
            PipelineStage::GeometryShader => {
                // VUID-vkCmdWriteTimestamp-pipelineStage-04075
                if !device.enabled_features().geometry_shader {
                    return Err(QueryError::FeatureNotEnabled {
                        feature: "geometry_shader",
                        reason: "stage was GeometryShader",
                    });
                }
            }
            PipelineStage::TessellationControlShader
            | PipelineStage::TessellationEvaluationShader => {
                // VUID-vkCmdWriteTimestamp-pipelineStage-04076
                if !device.enabled_features().tessellation_shader {
                    return Err(QueryError::FeatureNotEnabled {
                        feature: "tessellation_shader",
                        reason:
                            "stage was TessellationControlShader or TessellationEvaluationShader",
                    });
                }
            }
            _ => (),
        }

        // VUID-vkCmdWriteTimestamp-queryPool-01416
        if !matches!(query_pool.query_type(), QueryType::Timestamp) {
            return Err(QueryError::NotPermitted);
        }

        // VUID-vkCmdWriteTimestamp-timestampValidBits-00829
        if queue_family.timestamp_valid_bits().is_none() {
            return Err(QueryError::NoTimestampValidBits);
        }

        // VUID-vkCmdWriteTimestamp-query-04904
        query_pool.query(query).ok_or(QueryError::OutOfRange)?;

        if let Some(state) = &self.render_pass_state {
            let view_mask = match &state.render_pass {
                RenderPassStateType::BeginRenderPass(state) => {
                    state.subpass.subpass_desc().view_mask
                }
                RenderPassStateType::BeginRendering(state) => state.view_mask,
                RenderPassStateType::Inherited => match self
                    .inheritance_info
                    .as_ref()
                    .unwrap()
                    .render_pass
                    .as_ref()
                    .unwrap()
                {
                    CommandBufferInheritanceRenderPassType::BeginRenderPass(info) => {
                        info.subpass.subpass_desc().view_mask
                    }
                    CommandBufferInheritanceRenderPassType::BeginRendering(info) => info.view_mask,
                },
            };

            // VUID-vkCmdWriteTimestamp-query-00831
            if query + view_mask.count_ones() > query_pool.query_count() {
                return Err(QueryError::OutOfRangeMultiview);
            }
        }

        // VUID-vkCmdWriteTimestamp-queryPool-00828
        // VUID-vkCmdWriteTimestamp-None-00830
        // Not checked, therefore unsafe.
        // TODO: add check.

        Ok(())
    }

    /// Copies the results of a range of queries to a buffer on the GPU.
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
    ) -> Result<&mut Self, QueryError>
    where
        D: TypedBufferAccess<Content = [T]> + 'static,
        T: QueryResultElement,
    {
        self.validate_copy_query_pool_results(
            &query_pool,
            queries.clone(),
            destination.as_ref(),
            flags,
        )?;

        unsafe {
            let per_query_len =
                query_pool.query_type().result_len() + flags.with_availability as DeviceSize;
            let stride = per_query_len * std::mem::size_of::<T>() as DeviceSize;
            self.inner
                .copy_query_pool_results(query_pool, queries, destination, stride, flags)?;
        }

        Ok(self)
    }

    fn validate_copy_query_pool_results<D, T>(
        &self,
        query_pool: &QueryPool,
        queries: Range<u32>,
        destination: &D,
        flags: QueryResultFlags,
    ) -> Result<(), QueryError>
    where
        D: ?Sized + TypedBufferAccess<Content = [T]>,
        T: QueryResultElement,
    {
        // VUID-vkCmdCopyQueryPoolResults-commandBuffer-cmdpool
        if !(self.queue_family().supports_graphics() || self.queue_family().supports_compute()) {
            return Err(QueryError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdCopyQueryPoolResults-renderpass
        if self.render_pass_state.is_some() {
            return Err(QueryError::ForbiddenInsideRenderPass);
        }

        let device = self.device();
        let buffer_inner = destination.inner();

        // VUID-vkCmdCopyQueryPoolResults-commonparent
        assert_eq!(device, buffer_inner.buffer.device());
        assert_eq!(device, query_pool.device());

        assert!(destination.len() > 0);

        // VUID-vkCmdCopyQueryPoolResults-flags-00822
        // VUID-vkCmdCopyQueryPoolResults-flags-00823
        debug_assert!(buffer_inner.offset % std::mem::size_of::<T>() as DeviceSize == 0);

        // VUID-vkCmdCopyQueryPoolResults-firstQuery-00820
        // VUID-vkCmdCopyQueryPoolResults-firstQuery-00821
        query_pool
            .queries_range(queries.clone())
            .ok_or(QueryError::OutOfRange)?;

        let count = queries.end - queries.start;
        let per_query_len =
            query_pool.query_type().result_len() + flags.with_availability as DeviceSize;
        let required_len = per_query_len * count as DeviceSize;

        // VUID-vkCmdCopyQueryPoolResults-dstBuffer-00824
        if destination.len() < required_len {
            return Err(QueryError::BufferTooSmall {
                required_len,
                actual_len: destination.len(),
            });
        }

        // VUID-vkCmdCopyQueryPoolResults-dstBuffer-00825
        if !buffer_inner.buffer.usage().transfer_dst {
            return Err(QueryError::DestinationMissingUsage);
        }

        // VUID-vkCmdCopyQueryPoolResults-queryType-00827
        if matches!(query_pool.query_type(), QueryType::Timestamp) && flags.partial {
            return Err(QueryError::InvalidFlags);
        }

        Ok(())
    }

    /// Resets a range of queries on a query pool.
    ///
    /// The affected queries will be marked as "unavailable" after this command runs, and will no
    /// longer return any results. They will be ready to have new results recorded for them.
    ///
    /// # Safety
    /// The queries in the specified range must not be active in another command buffer.
    // TODO: Do other command buffers actually matter here? Not sure on the Vulkan spec.
    pub unsafe fn reset_query_pool(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
    ) -> Result<&mut Self, QueryError> {
        self.validate_reset_query_pool(&query_pool, queries.clone())?;

        self.inner.reset_query_pool(query_pool, queries);

        Ok(self)
    }

    fn validate_reset_query_pool(
        &self,
        query_pool: &QueryPool,
        queries: Range<u32>,
    ) -> Result<(), QueryError> {
        // VUID-vkCmdResetQueryPool-renderpass
        if self.render_pass_state.is_some() {
            return Err(QueryError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdResetQueryPool-commandBuffer-cmdpool
        if !(self.queue_family().supports_graphics() || self.queue_family().supports_compute()) {
            return Err(QueryError::NotSupportedByQueueFamily);
        }

        let device = self.device();

        // VUID-vkCmdResetQueryPool-commonparent
        assert_eq!(device, query_pool.device());

        // VUID-vkCmdResetQueryPool-firstQuery-00796
        // VUID-vkCmdResetQueryPool-firstQuery-00797
        query_pool
            .queries_range(queries.clone())
            .ok_or(QueryError::OutOfRange)?;

        // VUID-vkCmdResetQueryPool-None-02841
        if self.query_state.values().any(|state| {
            state.query_pool == query_pool.internal_object() && queries.contains(&state.query)
        }) {
            return Err(QueryError::QueryIsActive);
        }

        Ok(())
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
        (fns.v1_0.cmd_begin_query)(
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
        (fns.v1_0.cmd_end_query)(self.handle, query.pool().internal_object(), query.index());
    }

    /// Calls `vkCmdWriteTimestamp` on the builder.
    #[inline]
    pub unsafe fn write_timestamp(&mut self, query: Query, stage: PipelineStage) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_write_timestamp)(
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
        (fns.v1_0.cmd_copy_query_pool_results)(
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
        (fns.v1_0.cmd_reset_query_pool)(
            self.handle,
            queries.pool().internal_object(),
            range.start,
            range.end - range.start,
        );
    }
}

/// Error that can happen when calling a query command.
#[derive(Clone, Debug)]
pub enum QueryError {
    SyncCommandBufferBuilderError(SyncCommandBufferBuilderError),

    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
    },

    /// The buffer is too small for the copy operation.
    BufferTooSmall {
        /// Required number of elements in the buffer.
        required_len: DeviceSize,
        /// Actual number of elements in the buffer.
        actual_len: DeviceSize,
    },

    /// The destination buffer is missing the `transfer_dst` usage.
    DestinationMissingUsage,

    /// Operation forbidden inside of a render pass.
    ForbiddenInsideRenderPass,

    /// The provided flags are not allowed for this type of query.
    InvalidFlags,

    /// The queue family's `timestamp_valid_bits` value is `None`.
    NoTimestampValidBits,

    /// This operation is not permitted on this query type.
    NotPermitted,

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,

    /// The provided query index is not valid for this pool.
    OutOfRange,

    /// The provided query index plus the number of views in the current render subpass is greater
    /// than the number of queries in the pool.
    OutOfRangeMultiview,

    /// A query is active that conflicts with the current operation.
    QueryIsActive,

    /// This query was not active.
    QueryNotActive,

    /// The provided stage is not supported by the queue family.
    StageNotSupported,
}

impl Error for QueryError {}

impl fmt::Display for QueryError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::SyncCommandBufferBuilderError(_) => write!(
                f,
                "a SyncCommandBufferBuilderError",
            ),

            Self::FeatureNotEnabled { feature, reason } => write!(
                f,
                "the feature {} must be enabled: {}",
                feature, reason,
            ),

            Self::BufferTooSmall { .. } => {
                write!(f, "the buffer is too small for the copy operation")
            }
            Self::DestinationMissingUsage => write!(
                f,
                "the destination buffer is missing the `transfer_dst` usage",
            ),
            Self::ForbiddenInsideRenderPass => {
                write!(f, "operation forbidden inside of a render pass")
            }
            Self::InvalidFlags => write!(
                f,
                "the provided flags are not allowed for this type of query",
            ),
            Self::NoTimestampValidBits => {
                write!(f, "the queue family's timestamp_valid_bits value is None")
            }
            Self::NotPermitted => write!(f, "this operation is not permitted on this query type"),
            Self::NotSupportedByQueueFamily => {
                write!(f, "the queue family doesn't allow this operation")
            }
            Self::OutOfRange => write!(f, "the provided query index is not valid for this pool"),
            Self::OutOfRangeMultiview => write!(
                f,
                "the provided query index plus the number of views in the current render subpass is greater than the number of queries in the pool",
            ),
            Self::QueryIsActive => write!(
                f,
                "a query is active that conflicts with the current operation"
            ),
            Self::QueryNotActive => write!(f, "this query was not active"),
            Self::StageNotSupported => {
                write!(f, "the provided stage is not supported by the queue family")
            }
        }
    }
}

impl From<SyncCommandBufferBuilderError> for QueryError {
    #[inline]
    fn from(err: SyncCommandBufferBuilderError) -> Self {
        Self::SyncCommandBufferBuilderError(err)
    }
}
