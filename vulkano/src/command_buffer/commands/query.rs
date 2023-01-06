// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator,
        auto::QueryState,
        synced::{Command, Resource, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, ResourceInCommand, ResourceUseRef,
    },
    device::{DeviceOwned, QueueFlags},
    query::{
        QueriesRange, Query, QueryControlFlags, QueryPool, QueryResultElement, QueryResultFlags,
        QueryType,
    },
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStage, PipelineStages},
    DeviceSize, RequirementNotMet, RequiresOneOf, Version, VulkanObject,
};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::size_of,
    ops::Range,
    sync::Arc,
};

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
    ///
    /// The query must be unavailable, ensured by calling
    /// [`reset_query_pool`](Self::reset_query_pool).
    pub unsafe fn begin_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<&mut Self, QueryError> {
        self.validate_begin_query(&query_pool, query, flags)?;

        let ty = query_pool.query_type();
        let raw_query_pool = query_pool.handle();

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
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBeginQuery-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(QueryError::NotSupportedByQueueFamily);
        }

        let device = self.device();

        // VUID-vkCmdBeginQuery-flags-parameter
        flags.validate_device(device)?;

        // VUID-vkCmdBeginQuery-commonparent
        assert_eq!(device, query_pool.device());

        // VUID-vkCmdBeginQuery-query-00802
        query_pool.query(query).ok_or(QueryError::OutOfRange)?;

        match query_pool.query_type() {
            QueryType::Occlusion => {
                // VUID-vkCmdBeginQuery-commandBuffer-cmdpool
                // // VUID-vkCmdBeginQuery-queryType-00803
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(QueryError::NotSupportedByQueueFamily);
                }

                // VUID-vkCmdBeginQuery-queryType-00800
                if flags.intersects(QueryControlFlags::PRECISE)
                    && !device.enabled_features().occlusion_query_precise
                {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`flags` contains `QueryControlFlags::PRECISE`",
                        requires_one_of: RequiresOneOf {
                            features: &["occlusion_query_precise"],
                            ..Default::default()
                        },
                    });
                }
            }
            QueryType::PipelineStatistics(statistic_flags) => {
                // VUID-vkCmdBeginQuery-commandBuffer-cmdpool
                // VUID-vkCmdBeginQuery-queryType-00804
                // VUID-vkCmdBeginQuery-queryType-00805
                if statistic_flags.is_compute()
                    && !queue_family_properties
                        .queue_flags
                        .intersects(QueueFlags::COMPUTE)
                    || statistic_flags.is_graphics()
                        && !queue_family_properties
                            .queue_flags
                            .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(QueryError::NotSupportedByQueueFamily);
                }

                // VUID-vkCmdBeginQuery-queryType-00800
                if flags.intersects(QueryControlFlags::PRECISE) {
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

        if let Some(render_pass_state) = &self.render_pass_state {
            // VUID-vkCmdBeginQuery-query-00808
            if query + render_pass_state.view_mask.count_ones() > query_pool.query_count() {
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
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdEndQuery-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
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
                state.query_pool == query_pool.handle() && state.query == query
            })
        {
            return Err(QueryError::QueryNotActive);
        }

        // VUID-vkCmdEndQuery-query-00810
        query_pool.query(query).ok_or(QueryError::OutOfRange)?;

        if let Some(render_pass_state) = &self.render_pass_state {
            // VUID-vkCmdEndQuery-query-00812
            if query + render_pass_state.view_mask.count_ones() > query_pool.query_count() {
                return Err(QueryError::OutOfRangeMultiview);
            }
        }

        Ok(())
    }

    /// Writes a timestamp to a timestamp query.
    ///
    /// # Safety
    ///
    /// The query must be unavailable, ensured by calling
    /// [`reset_query_pool`](Self::reset_query_pool).
    pub unsafe fn write_timestamp(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        stage: PipelineStage,
    ) -> Result<&mut Self, QueryError> {
        self.validate_write_timestamp(&query_pool, query, stage)?;

        self.inner.write_timestamp(query_pool, query, stage);

        Ok(self)
    }

    fn validate_write_timestamp(
        &self,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
    ) -> Result<(), QueryError> {
        let device = self.device();

        if !device.enabled_features().synchronization2 && PipelineStages::from(stage).is_2() {
            return Err(QueryError::RequirementNotMet {
                required_for: "`stage` has flags set from `VkPipelineStageFlagBits2`",
                requires_one_of: RequiresOneOf {
                    features: &["synchronization2"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdWriteTimestamp2-stage-parameter
        stage.validate_device(device)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdWriteTimestamp2-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::TRANSFER
                | QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(QueryError::NotSupportedByQueueFamily);
        }

        let device = self.device();

        // VUID-vkCmdWriteTimestamp2-commonparent
        assert_eq!(device, query_pool.device());

        // VUID-vkCmdWriteTimestamp2-stage-03860
        if !PipelineStages::from(queue_family_properties.queue_flags).contains_enum(stage) {
            return Err(QueryError::StageNotSupported);
        }

        match stage {
            PipelineStage::GeometryShader => {
                // VUID-vkCmdWriteTimestamp2-stage-03929
                if !device.enabled_features().geometry_shader {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`stage` is `PipelineStage::GeometryShader`",
                        requires_one_of: RequiresOneOf {
                            features: &["geometry_shadere"],
                            ..Default::default()
                        },
                    });
                }
            }
            PipelineStage::TessellationControlShader
            | PipelineStage::TessellationEvaluationShader => {
                // VUID-vkCmdWriteTimestamp2-stage-03930
                if !device.enabled_features().tessellation_shader {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`stage` is `PipelineStage::TessellationControlShader` or \
                            `PipelineStage::TessellationEvaluationShader`",
                        requires_one_of: RequiresOneOf {
                            features: &["tessellation_shader"],
                            ..Default::default()
                        },
                    });
                }
            }
            PipelineStage::ConditionalRendering => {
                // VUID-vkCmdWriteTimestamp2-stage-03931
                if !device.enabled_features().conditional_rendering {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`stage` is `PipelineStage::ConditionalRendering`",
                        requires_one_of: RequiresOneOf {
                            features: &["conditional_rendering"],
                            ..Default::default()
                        },
                    });
                }
            }
            PipelineStage::FragmentDensityProcess => {
                // VUID-vkCmdWriteTimestamp2-stage-03932
                if !device.enabled_features().fragment_density_map {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`stage` is `PipelineStage::FragmentDensityProcess`",
                        requires_one_of: RequiresOneOf {
                            features: &["fragment_density_map"],
                            ..Default::default()
                        },
                    });
                }
            }
            PipelineStage::TransformFeedback => {
                // VUID-vkCmdWriteTimestamp2-stage-03933
                if !device.enabled_features().transform_feedback {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`stage` is `PipelineStage::TransformFeedback`",
                        requires_one_of: RequiresOneOf {
                            features: &["transform_feedback"],
                            ..Default::default()
                        },
                    });
                }
            }
            PipelineStage::MeshShader => {
                // VUID-vkCmdWriteTimestamp2-stage-03934
                if !device.enabled_features().mesh_shader {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`stage` is `PipelineStage::MeshShader`",
                        requires_one_of: RequiresOneOf {
                            features: &["mesh_shader"],
                            ..Default::default()
                        },
                    });
                }
            }
            PipelineStage::TaskShader => {
                // VUID-vkCmdWriteTimestamp2-stage-03935
                if !device.enabled_features().task_shader {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`stage` is `PipelineStage::TaskShader`",
                        requires_one_of: RequiresOneOf {
                            features: &["task_shader"],
                            ..Default::default()
                        },
                    });
                }
            }
            PipelineStage::FragmentShadingRateAttachment => {
                // VUID-vkCmdWriteTimestamp2-shadingRateImage-07316
                if !(device.enabled_features().attachment_fragment_shading_rate
                    || device.enabled_features().shading_rate_image)
                {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`stage` is `PipelineStage::FragmentShadingRateAttachment`",
                        requires_one_of: RequiresOneOf {
                            features: &["attachment_fragment_shading_rate", "shading_rate_image"],
                            ..Default::default()
                        },
                    });
                }
            }
            PipelineStage::SubpassShading => {
                // VUID-vkCmdWriteTimestamp2-stage-04957
                if !device.enabled_features().subpass_shading {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`stage` is `PipelineStage::SubpassShading`",
                        requires_one_of: RequiresOneOf {
                            features: &["subpass_shading"],
                            ..Default::default()
                        },
                    });
                }
            }
            PipelineStage::InvocationMask => {
                // VUID-vkCmdWriteTimestamp2-stage-04995
                if !device.enabled_features().invocation_mask {
                    return Err(QueryError::RequirementNotMet {
                        required_for: "`stage` is `PipelineStage::InvocationMask`",
                        requires_one_of: RequiresOneOf {
                            features: &["invocation_mask"],
                            ..Default::default()
                        },
                    });
                }
            }
            _ => (),
        }

        // VUID-vkCmdWriteTimestamp2-queryPool-03861
        if !matches!(query_pool.query_type(), QueryType::Timestamp) {
            return Err(QueryError::NotPermitted);
        }

        // VUID-vkCmdWriteTimestamp2-timestampValidBits-03863
        if queue_family_properties.timestamp_valid_bits.is_none() {
            return Err(QueryError::NoTimestampValidBits);
        }

        // VUID-vkCmdWriteTimestamp2-query-04903
        query_pool.query(query).ok_or(QueryError::OutOfRange)?;

        if let Some(render_pass_state) = &self.render_pass_state {
            // VUID-vkCmdWriteTimestamp2-query-03865
            if query + render_pass_state.view_mask.count_ones() > query_pool.query_count() {
                return Err(QueryError::OutOfRangeMultiview);
            }
        }

        // VUID-vkCmdWriteTimestamp2-queryPool-03862
        // VUID-vkCmdWriteTimestamp2-None-03864
        // Not checked, therefore unsafe.
        // TODO: add check.

        Ok(())
    }

    /// Copies the results of a range of queries to a buffer on the GPU.
    ///
    /// [`query_pool.ty().result_len()`] elements will be written for each query in the range, plus
    /// 1 extra element per query if [`QueryResultFlags::WITH_AVAILABILITY`] is enabled.
    /// The provided buffer must be large enough to hold the data.
    ///
    /// See also [`get_results`].
    ///
    /// [`query_pool.ty().result_len()`]: crate::query::QueryType::result_len
    /// [`QueryResultFlags::WITH_AVAILABILITY`]: crate::query::QueryResultFlags::WITH_AVAILABILITY
    /// [`get_results`]: crate::query::QueriesRange::get_results
    pub fn copy_query_pool_results<T>(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
        destination: Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> Result<&mut Self, QueryError>
    where
        T: QueryResultElement,
    {
        self.validate_copy_query_pool_results(&query_pool, queries.clone(), &destination, flags)?;

        unsafe {
            let per_query_len = query_pool.query_type().result_len()
                + flags.intersects(QueryResultFlags::WITH_AVAILABILITY) as DeviceSize;
            let stride = per_query_len * std::mem::size_of::<T>() as DeviceSize;
            self.inner
                .copy_query_pool_results(query_pool, queries, destination, stride, flags)?;
        }

        Ok(self)
    }

    fn validate_copy_query_pool_results<T>(
        &self,
        query_pool: &QueryPool,
        queries: Range<u32>,
        destination: &Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> Result<(), QueryError>
    where
        T: QueryResultElement,
    {
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdCopyQueryPoolResults-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(QueryError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdCopyQueryPoolResults-renderpass
        if self.render_pass_state.is_some() {
            return Err(QueryError::ForbiddenInsideRenderPass);
        }

        let device = self.device();

        // VUID-vkCmdCopyQueryPoolResults-commonparent
        assert_eq!(device, destination.buffer().device());
        assert_eq!(device, query_pool.device());

        assert!(destination.len() > 0);

        // VUID-vkCmdCopyQueryPoolResults-flags-00822
        // VUID-vkCmdCopyQueryPoolResults-flags-00823
        debug_assert!(destination.offset() % std::mem::size_of::<T>() as DeviceSize == 0);

        // VUID-vkCmdCopyQueryPoolResults-firstQuery-00820
        // VUID-vkCmdCopyQueryPoolResults-firstQuery-00821
        query_pool
            .queries_range(queries.clone())
            .ok_or(QueryError::OutOfRange)?;

        let count = queries.end - queries.start;
        let per_query_len = query_pool.query_type().result_len()
            + flags.intersects(QueryResultFlags::WITH_AVAILABILITY) as DeviceSize;
        let required_len = per_query_len * count as DeviceSize;

        // VUID-vkCmdCopyQueryPoolResults-dstBuffer-00824
        if destination.len() < required_len {
            return Err(QueryError::BufferTooSmall {
                required_len,
                actual_len: destination.len(),
            });
        }

        // VUID-vkCmdCopyQueryPoolResults-dstBuffer-00825
        if !destination
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_DST)
        {
            return Err(QueryError::DestinationMissingUsage);
        }

        // VUID-vkCmdCopyQueryPoolResults-queryType-00827
        if matches!(query_pool.query_type(), QueryType::Timestamp)
            && flags.intersects(QueryResultFlags::PARTIAL)
        {
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

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdResetQueryPool-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
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
        if self
            .query_state
            .values()
            .any(|state| state.query_pool == query_pool.handle() && queries.contains(&state.query))
        {
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
    pub unsafe fn copy_query_pool_results(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
        destination: Subbuffer<[impl QueryResultElement]>,
        stride: DeviceSize,
        flags: QueryResultFlags,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd<T> {
            query_pool: Arc<QueryPool>,
            queries: Range<u32>,
            destination: Subbuffer<[T]>,
            stride: DeviceSize,
            flags: QueryResultFlags,
        }

        impl<T> Command for Cmd<T>
        where
            T: QueryResultElement,
        {
            fn name(&self) -> &'static str {
                "copy_query_pool_results"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_query_pool_results(
                    self.query_pool.queries_range(self.queries.clone()).unwrap(),
                    &self.destination,
                    self.stride,
                    self.flags,
                );
            }
        }

        let command_index = self.commands.len();
        let command_name = "copy_query_pool_results";
        let resources = [(
            ResourceUseRef {
                command_index,
                command_name,
                resource_in_command: ResourceInCommand::Destination,
                secondary_use_ref: None,
            },
            Resource::Buffer {
                buffer: destination.as_bytes().clone(),
                range: 0..destination.size(), // TODO:
                memory: PipelineMemoryAccess {
                    stages: PipelineStages::ALL_TRANSFER,
                    access: AccessFlags::TRANSFER_WRITE,
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
    pub unsafe fn begin_query(&mut self, query: Query<'_>, flags: QueryControlFlags) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_begin_query)(
            self.handle,
            query.pool().handle(),
            query.index(),
            flags.into(),
        );
    }

    /// Calls `vkCmdEndQuery` on the builder.
    #[inline]
    pub unsafe fn end_query(&mut self, query: Query<'_>) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_end_query)(self.handle, query.pool().handle(), query.index());
    }

    /// Calls `vkCmdWriteTimestamp` on the builder.
    #[inline]
    pub unsafe fn write_timestamp(&mut self, query: Query<'_>, stage: PipelineStage) {
        let fns = self.device.fns();

        if self.device.enabled_features().synchronization2 {
            if self.device.api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_write_timestamp2)(
                    self.handle,
                    stage.into(),
                    query.pool().handle(),
                    query.index(),
                );
            } else {
                debug_assert!(self.device.enabled_extensions().khr_synchronization2);
                (fns.khr_synchronization2.cmd_write_timestamp2_khr)(
                    self.handle,
                    stage.into(),
                    query.pool().handle(),
                    query.index(),
                );
            }
        } else {
            (fns.v1_0.cmd_write_timestamp)(
                self.handle,
                stage.into(),
                query.pool().handle(),
                query.index(),
            );
        }
    }

    /// Calls `vkCmdCopyQueryPoolResults` on the builder.
    pub unsafe fn copy_query_pool_results<T>(
        &mut self,
        queries: QueriesRange<'_>,
        destination: &Subbuffer<[T]>,
        stride: DeviceSize,
        flags: QueryResultFlags,
    ) where
        T: QueryResultElement,
    {
        let range = queries.range();
        debug_assert!(destination.offset() < destination.buffer().size());
        debug_assert!(destination
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_DST));
        debug_assert!(destination.offset() % size_of::<T>() as DeviceSize == 0);
        debug_assert!(stride % size_of::<T>() as DeviceSize == 0);

        let fns = self.device.fns();
        (fns.v1_0.cmd_copy_query_pool_results)(
            self.handle,
            queries.pool().handle(),
            range.start,
            range.end - range.start,
            destination.buffer().handle(),
            destination.offset(),
            stride,
            ash::vk::QueryResultFlags::from(flags) | T::FLAG,
        );
    }

    /// Calls `vkCmdResetQueryPool` on the builder.
    #[inline]
    pub unsafe fn reset_query_pool(&mut self, queries: QueriesRange<'_>) {
        let range = queries.range();
        let fns = self.device.fns();
        (fns.v1_0.cmd_reset_query_pool)(
            self.handle,
            queries.pool().handle(),
            range.start,
            range.end - range.start,
        );
    }
}

/// Error that can happen when recording a query command.
#[derive(Clone, Debug)]
pub enum QueryError {
    SyncCommandBufferBuilderError(SyncCommandBufferBuilderError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
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

impl Display for QueryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::SyncCommandBufferBuilderError(_) => write!(f, "a SyncCommandBufferBuilderError"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
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
                "the provided query index plus the number of views in the current render subpass \
                is greater than the number of queries in the pool",
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
    fn from(err: SyncCommandBufferBuilderError) -> Self {
        Self::SyncCommandBufferBuilderError(err)
    }
}

impl From<RequirementNotMet> for QueryError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}
