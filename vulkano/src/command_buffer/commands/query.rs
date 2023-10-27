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
        auto::{QueryState, Resource},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, ResourceInCommand,
    },
    device::{DeviceOwned, QueueFlags},
    query::{QueryControlFlags, QueryPool, QueryResultElement, QueryResultFlags, QueryType},
    sync::{PipelineStage, PipelineStageAccessFlags, PipelineStages},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use std::{ops::Range, sync::Arc};

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
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_begin_query(&query_pool, query, flags)?;

        Ok(self.begin_query_unchecked(query_pool, query, flags))
    }

    fn validate_begin_query(
        &self,
        query_pool: &QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_begin_query(query_pool, query, flags)?;

        if self
            .builder_state
            .queries
            .contains_key(&query_pool.query_type().into())
        {
            return Err(Box::new(ValidationError {
                problem: "a query with the same type as `query_pool.query_type()` is \
                    already active"
                    .into(),
                vuids: &["VUID-vkCmdBeginQuery-queryPool-01922"],
                ..Default::default()
            }));
        }

        if let Some(render_pass_state) = &self.builder_state.render_pass {
            if query + render_pass_state.rendering_info.view_mask.count_ones()
                > query_pool.query_count()
            {
                return Err(Box::new(ValidationError {
                    problem: "a render subpass with a non-zero `view_mask` is active, but \
                        `query` + the number of views in `view_mask` is greater than \
                        `query_pool.query_count()`"
                        .into(),
                    vuids: &["VUID-vkCmdBeginQuery-query-00808"],
                    ..Default::default()
                }));
            }
        }

        // VUID-vkCmdBeginQuery-None-00807
        // Not checked, therefore unsafe.
        // TODO: add check.

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_query_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        flags: QueryControlFlags,
    ) -> &mut Self {
        self.builder_state.queries.insert(
            query_pool.query_type().into(),
            QueryState {
                query_pool: query_pool.clone(),
                query,
                flags,
                in_subpass: self.builder_state.render_pass.is_some(),
            },
        );

        self.add_command(
            "begin_query",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.begin_query_unchecked(&query_pool, query, flags);
            },
        );

        self
    }

    /// Ends an active query.
    pub fn end_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_query(&query_pool, query)?;

        unsafe { Ok(self.end_query_unchecked(query_pool, query)) }
    }

    fn validate_end_query(
        &self,
        query_pool: &QueryPool,
        query: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_end_query(query_pool, query)?;

        if !self
            .builder_state
            .queries
            .get(&query_pool.query_type().into())
            .map_or(false, |state| {
                *state.query_pool == *query_pool && state.query == query
            })
        {
            return Err(Box::new(ValidationError {
                problem: "no query with the same type as `query_pool.query_type()` is active"
                    .into(),
                vuids: &["VUID-vkCmdEndQuery-None-01923"],
                ..Default::default()
            }));
        }

        if let Some(render_pass_state) = &self.builder_state.render_pass {
            if query + render_pass_state.rendering_info.view_mask.count_ones()
                > query_pool.query_count()
            {
                return Err(Box::new(ValidationError {
                    problem: "a render subpass with a non-zero `view_mask` is active, but \
                        `query` + the number of views in `view_mask` is greater than \
                        `query_pool.query_count()`"
                        .into(),
                    vuids: &["VUID-vkCmdEndQuery-query-00812"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_query_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
    ) -> &mut Self {
        let raw_ty = query_pool.query_type().into();
        self.builder_state.queries.remove(&raw_ty);

        self.add_command(
            "end_query",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.end_query_unchecked(&query_pool, query);
            },
        );

        self
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
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_write_timestamp(&query_pool, query, stage)?;

        Ok(self.write_timestamp_unchecked(query_pool, query, stage))
    }

    fn validate_write_timestamp(
        &self,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_write_timestamp(query_pool, query, stage)?;

        if let Some(render_pass_state) = &self.builder_state.render_pass {
            if query + render_pass_state.rendering_info.view_mask.count_ones()
                > query_pool.query_count()
            {
                return Err(Box::new(ValidationError {
                    problem: "a render subpass with a non-zero `view_mask` is active, but \
                        `query` + the number of views in `view_mask` is greater than \
                        `query_pool.query_count()`"
                        .into(),
                    vuids: &["VUID-vkCmdWriteTimestamp2-query-03865"],
                    ..Default::default()
                }));
            }
        }

        // VUID-vkCmdWriteTimestamp2-queryPool-03862
        // VUID-vkCmdWriteTimestamp2-None-03864
        // Not checked, therefore unsafe.
        // TODO: add check.

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn write_timestamp_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        stage: PipelineStage,
    ) -> &mut Self {
        self.add_command(
            "write_timestamp",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.write_timestamp_unchecked(&query_pool, query, stage);
            },
        );

        self
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
    /// [`get_results`]: crate::query::QueryPool::get_results
    pub fn copy_query_pool_results<T>(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
        destination: Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> Result<&mut Self, Box<ValidationError>>
    where
        T: QueryResultElement,
    {
        self.validate_copy_query_pool_results(&query_pool, queries.clone(), &destination, flags)?;

        unsafe {
            Ok(self.copy_query_pool_results_unchecked(query_pool, queries, destination, flags))
        }
    }

    fn validate_copy_query_pool_results<T>(
        &self,
        query_pool: &QueryPool,
        queries: Range<u32>,
        destination: &Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> Result<(), Box<ValidationError>>
    where
        T: QueryResultElement,
    {
        self.inner
            .validate_copy_query_pool_results(query_pool, queries, destination, flags)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_query_pool_results_unchecked<T>(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
        destination: Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> &mut Self
    where
        T: QueryResultElement,
    {
        self.add_command(
            "copy_query_pool_results",
            [(
                ResourceInCommand::Destination.into(),
                Resource::Buffer {
                    buffer: destination.as_bytes().clone(),
                    range: 0..destination.size(), // TODO:
                    memory_access: PipelineStageAccessFlags::Copy_TransferWrite,
                },
            )]
            .into_iter()
            .collect(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.copy_query_pool_results_unchecked(
                    &query_pool,
                    queries.clone(),
                    &destination,
                    flags,
                );
            },
        );

        self
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
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_reset_query_pool(&query_pool, queries.clone())?;

        Ok(self.reset_query_pool_unchecked(query_pool, queries))
    }

    fn validate_reset_query_pool(
        &self,
        query_pool: &QueryPool,
        queries: Range<u32>,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_reset_query_pool(query_pool, queries.clone())?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdResetQueryPool-renderpass"],
                ..Default::default()
            }));
        }

        if self
            .builder_state
            .queries
            .values()
            .any(|state| *state.query_pool == *query_pool && queries.contains(&state.query))
        {
            return Err(Box::new(ValidationError {
                problem: "one of the `queries` in `query_pool` is currently active".into(),
                vuids: &["VUID-vkCmdResetQueryPool-None-02841"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_query_pool_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
    ) -> &mut Self {
        self.add_command(
            "reset_query_pool",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.reset_query_pool_unchecked(&query_pool, queries.clone());
            },
        );

        self
    }
}

impl<A> UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    pub unsafe fn begin_query(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_begin_query(query_pool, query, flags)?;

        Ok(self.begin_query_unchecked(query_pool, query, flags))
    }

    fn validate_begin_query(
        &self,
        query_pool: &QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdBeginQuery-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-vkCmdBeginQuery-flags-parameter"])
        })?;

        // VUID-vkCmdBeginQuery-commonparent
        assert_eq!(device, query_pool.device());

        if query > query_pool.query_count() {
            return Err(Box::new(ValidationError {
                problem: "`query` is greater than `query_pool.query_count()`".into(),
                vuids: &["VUID-vkCmdBeginQuery-query-00802"],
                ..Default::default()
            }));
        }

        match query_pool.query_type() {
            QueryType::Occlusion => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`query_pool.query_type()` is `QueryType::Occlusion`, but \
                            the queue family of the command buffer does not support \
                            graphics operations"
                            .into(),
                        vuids: &["VUID-vkCmdBeginQuery-queryType-00803"],
                        ..Default::default()
                    }));
                }
            }
            QueryType::PipelineStatistics(statistic_flags) => {
                if statistic_flags.is_graphics()
                    && !queue_family_properties
                        .queue_flags
                        .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(Box::new(ValidationError {
                        context: "query_pool.query_type()".into(),
                        problem: "is `QueryType::PipelineStatistics`, and the \
                            pipeline statistics flags include a graphics flag, but \
                            the queue family of the command buffer does not support \
                            graphics operations"
                            .into(),
                        vuids: &["VUID-vkCmdBeginQuery-queryType-00804"],
                        ..Default::default()
                    }));
                }

                if statistic_flags.is_compute()
                    && !queue_family_properties
                        .queue_flags
                        .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        context: "query_pool.query_type()".into(),
                        problem: "is `QueryType::PipelineStatistics`, and the \
                            pipeline statistics flags include a compute flag, but \
                            the queue family of the command buffer does not support \
                            compute operations"
                            .into(),
                        vuids: &["VUID-vkCmdBeginQuery-queryType-00805"],
                        ..Default::default()
                    }));
                }
            }
            QueryType::ResultStatusOnly => {
                if !queue_family_properties.supports_result_status_query {
                    return Err(Box::new(ValidationError {
                        problem: "`query_pool.query_type()` is `QueryType::ResultStatusOnly`, but \
                            the queue family of the command buffer does not support \
                            result status queries"
                            .into(),
                        vuids: &["VUID-vkCmdBeginQuery-queryType-07126"],
                        ..Default::default()
                    }));
                }
            }
            QueryType::Timestamp
            | QueryType::AccelerationStructureCompactedSize
            | QueryType::AccelerationStructureSerializationSize
            | QueryType::AccelerationStructureSerializationBottomLevelPointers
            | QueryType::AccelerationStructureSize => {
                return Err(Box::new(ValidationError {
                    context: "query_pool.query_type()".into(),
                    problem: "is not allowed for this command".into(),
                    vuids: &[
                        "VUID-vkCmdBeginQuery-queryType-02804",
                        "VUID-vkCmdBeginQuery-queryType-04728",
                        "VUID-vkCmdBeginQuery-queryType-06741",
                    ],
                    ..Default::default()
                }));
            }
        }

        if flags.intersects(QueryControlFlags::PRECISE) {
            if !device.enabled_features().occlusion_query_precise {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `QueryControlFlags::PRECISE`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "occlusion_query_precise",
                    )])]),
                    vuids: &["VUID-vkCmdBeginQuery-queryType-00800"],
                }));
            }

            if !matches!(query_pool.query_type(), QueryType::Occlusion) {
                return Err(Box::new(ValidationError {
                    problem: "`flags` contains `QueryControlFlags::PRECISE`, but \
                        `query_pool.query_type()` is not `QueryType::Occlusion`"
                        .into(),
                    vuids: &["VUID-vkCmdBeginQuery-queryType-00800"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_query_unchecked(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_begin_query)(self.handle(), query_pool.handle(), query, flags.into());

        self
    }

    pub unsafe fn end_query(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_query(query_pool, query)?;

        Ok(self.end_query_unchecked(query_pool, query))
    }

    fn validate_end_query(
        &self,
        query_pool: &QueryPool,
        query: u32,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdEndQuery-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        // VUID-vkCmdEndQuery-commonparent
        assert_eq!(device, query_pool.device());

        if query > query_pool.query_count() {
            return Err(Box::new(ValidationError {
                problem: "`query` is greater than `query_pool.query_count()`".into(),
                vuids: &["VUID-vkCmdEndQuery-query-00810"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_query_unchecked(&mut self, query_pool: &QueryPool, query: u32) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_end_query)(self.handle(), query_pool.handle(), query);

        self
    }

    pub unsafe fn write_timestamp(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_write_timestamp(query_pool, query, stage)?;

        Ok(self.write_timestamp_unchecked(query_pool, query, stage))
    }

    fn validate_write_timestamp(
        &self,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::TRANSFER
                | QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    transfer, graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdWriteTimestamp2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        stage.validate_device(device).map_err(|err| {
            err.add_context("stage")
                .set_vuids(&["VUID-vkCmdWriteTimestamp2-stage-parameter"])
        })?;

        if !device.enabled_features().synchronization2
            && PipelineStages::from(stage).contains_flags2()
        {
            return Err(Box::new(ValidationError {
                context: "stage".into(),
                problem: "is a stage flag from `VkPipelineStageFlagBits2`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "synchronization2",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-vkCmdWriteTimestamp2-commonparent
        assert_eq!(device, query_pool.device());

        if !PipelineStages::from(queue_family_properties.queue_flags).contains_enum(stage) {
            return Err(Box::new(ValidationError {
                context: "stage".into(),
                problem: "is not supported by the queue family of the command buffer".into(),
                vuids: &["VUID-vkCmdWriteTimestamp2-stage-03860"],
                ..Default::default()
            }));
        }

        match stage {
            PipelineStage::GeometryShader => {
                if !device.enabled_features().geometry_shader {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::GeometryShader`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "geometry_shadere",
                        )])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03929"],
                    }));
                }
            }
            PipelineStage::TessellationControlShader
            | PipelineStage::TessellationEvaluationShader => {
                if !device.enabled_features().tessellation_shader {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::TessellationControlShader` or \
                            `PipelineStage::TessellationEvaluationShader`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "tessellation_shader",
                        )])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03930"],
                    }));
                }
            }
            PipelineStage::ConditionalRendering => {
                if !device.enabled_features().conditional_rendering {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::ConditionalRendering`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "conditional_rendering",
                        )])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03931"],
                    }));
                }
            }
            PipelineStage::FragmentDensityProcess => {
                if !device.enabled_features().fragment_density_map {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::FragmentDensityProcess`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "fragment_density_map",
                        )])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03932"],
                    }));
                }
            }
            PipelineStage::TransformFeedback => {
                if !device.enabled_features().transform_feedback {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::TransformFeedback`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "transform_feedback",
                        )])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03933"],
                    }));
                }
            }
            PipelineStage::MeshShader => {
                if !device.enabled_features().mesh_shader {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::MeshShader`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "mesh_shader",
                        )])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03934"],
                    }));
                }
            }
            PipelineStage::TaskShader => {
                if !device.enabled_features().task_shader {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::TaskShader`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "task_shader",
                        )])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-03935"],
                    }));
                }
            }
            PipelineStage::FragmentShadingRateAttachment => {
                if !(device.enabled_features().attachment_fragment_shading_rate
                    || device.enabled_features().shading_rate_image)
                {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::FragmentShadingRateAttachment`".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[Requires::Feature("attachment_fragment_shading_rate")]),
                            RequiresAllOf(&[Requires::Feature("shading_rate_image")]),
                        ]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-shadingRateImage-07316"],
                    }));
                }
            }
            PipelineStage::SubpassShading => {
                if !device.enabled_features().subpass_shading {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::SubpassShading`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "subpass_shading",
                        )])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-04957"],
                    }));
                }
            }
            PipelineStage::InvocationMask => {
                if !device.enabled_features().invocation_mask {
                    return Err(Box::new(ValidationError {
                        context: "stage".into(),
                        problem: "is `PipelineStage::InvocationMask`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "invocation_mask",
                        )])]),
                        vuids: &["VUID-vkCmdWriteTimestamp2-stage-04995"],
                    }));
                }
            }
            _ => (),
        }

        if !matches!(query_pool.query_type(), QueryType::Timestamp) {
            return Err(Box::new(ValidationError {
                context: "query_pool.query_type()".into(),
                problem: "is not `QueryType::Timestamp`".into(),
                vuids: &["VUID-vkCmdWriteTimestamp2-queryPool-03861"],
                ..Default::default()
            }));
        }

        if queue_family_properties.timestamp_valid_bits.is_none() {
            return Err(Box::new(ValidationError {
                problem: "the `timestamp_valid_bits` value of the queue family properties of \
                    the command buffer is `None`"
                    .into(),
                vuids: &["VUID-vkCmdWriteTimestamp2-timestampValidBits-03863"],
                ..Default::default()
            }));
        }

        if query > query_pool.query_count() {
            return Err(Box::new(ValidationError {
                problem: "`query` is greater than `query_pool.query_count()`".into(),
                vuids: &["VUID-vkCmdWriteTimestamp2-query-04903"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn write_timestamp_unchecked(
        &mut self,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_write_timestamp2)(
                    self.handle(),
                    stage.into(),
                    query_pool.handle(),
                    query,
                );
            } else {
                (fns.khr_synchronization2.cmd_write_timestamp2_khr)(
                    self.handle(),
                    stage.into(),
                    query_pool.handle(),
                    query,
                );
            }
        } else {
            (fns.v1_0.cmd_write_timestamp)(self.handle(), stage.into(), query_pool.handle(), query);
        }

        self
    }

    pub unsafe fn copy_query_pool_results<T>(
        &mut self,
        query_pool: &QueryPool,
        queries: Range<u32>,
        destination: &Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> Result<&mut Self, Box<ValidationError>>
    where
        T: QueryResultElement,
    {
        self.validate_copy_query_pool_results(query_pool, queries.clone(), destination, flags)?;

        Ok(self.copy_query_pool_results_unchecked(query_pool, queries, destination, flags))
    }

    fn validate_copy_query_pool_results<T>(
        &self,
        query_pool: &QueryPool,
        queries: Range<u32>,
        destination: &Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> Result<(), Box<ValidationError>>
    where
        T: QueryResultElement,
    {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        // VUID-vkCmdCopyQueryPoolResults-commonparent
        assert_eq!(device, destination.buffer().device());
        assert_eq!(device, query_pool.device());

        // VUID-vkCmdCopyQueryPoolResults-flags-00822
        // VUID-vkCmdCopyQueryPoolResults-flags-00823
        debug_assert!(destination.offset() % std::mem::size_of::<T>() as DeviceSize == 0);

        if queries.end < queries.start {
            return Err(Box::new(ValidationError {
                context: "queries".into(),
                problem: "`end` is less than `start`".into(),
                ..Default::default()
            }));
        }

        if queries.end > query_pool.query_count() {
            return Err(Box::new(ValidationError {
                problem: "`queries.end` is greater than `query_pool.query_count()`".into(),
                vuids: &[
                    "VUID-vkCmdCopyQueryPoolResults-firstQuery-00820",
                    "VUID-vkCmdCopyQueryPoolResults-firstQuery-00821",
                ],
                ..Default::default()
            }));
        }

        let count = queries.end - queries.start;
        let per_query_len = query_pool.query_type().result_len()
            + flags.intersects(QueryResultFlags::WITH_AVAILABILITY) as DeviceSize;
        let required_len = per_query_len * count as DeviceSize;

        if destination.len() < required_len {
            return Err(Box::new(ValidationError {
                problem: "`destination` is smaller than the size required to write the results"
                    .into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-dstBuffer-00824"],
                ..Default::default()
            }));
        }

        if !destination
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_DST)
        {
            return Err(Box::new(ValidationError {
                context: "destination.usage()".into(),
                problem: "does not contain `BufferUsage::TRANSFER_DST`".into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-dstBuffer-00825"],
                ..Default::default()
            }));
        }

        if matches!(query_pool.query_type(), QueryType::Timestamp)
            && flags.intersects(QueryResultFlags::PARTIAL)
        {
            return Err(Box::new(ValidationError {
                problem: "`query_pool.query_type()` is `QueryType::Timestamp`, but \
                    `flags` contains `QueryResultFlags::PARTIAL`"
                    .into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-queryType-00827"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_query_pool_results_unchecked<T>(
        &mut self,
        query_pool: &QueryPool,
        queries: Range<u32>,
        destination: &Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> &mut Self
    where
        T: QueryResultElement,
    {
        let per_query_len = query_pool.query_type().result_len()
            + flags.intersects(QueryResultFlags::WITH_AVAILABILITY) as DeviceSize;
        let stride = per_query_len * std::mem::size_of::<T>() as DeviceSize;

        let fns = self.device().fns();
        (fns.v1_0.cmd_copy_query_pool_results)(
            self.handle(),
            query_pool.handle(),
            queries.start,
            queries.end - queries.start,
            destination.buffer().handle(),
            destination.offset(),
            stride,
            ash::vk::QueryResultFlags::from(flags) | T::FLAG,
        );

        self
    }

    pub unsafe fn reset_query_pool(
        &mut self,
        query_pool: &QueryPool,
        queries: Range<u32>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_reset_query_pool(query_pool, queries.clone())?;

        Ok(self.reset_query_pool_unchecked(query_pool, queries))
    }

    fn validate_reset_query_pool(
        &self,
        query_pool: &QueryPool,
        queries: Range<u32>,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE
                | QueueFlags::OPTICAL_FLOW,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics, compute, video decode, video encode or optical flow operations"
                    .into(),
                vuids: &["VUID-vkCmdResetQueryPool-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        // VUID-vkCmdResetQueryPool-commonparent
        assert_eq!(device, query_pool.device());

        if queries.end < queries.start {
            return Err(Box::new(ValidationError {
                context: "queries".into(),
                problem: "`end` is less than `start`".into(),
                ..Default::default()
            }));
        }

        if queries.end > query_pool.query_count() {
            return Err(Box::new(ValidationError {
                problem: "`queries.end` is greater than `query_pool.query_count()`".into(),
                vuids: &[
                    "VUID-vkCmdResetQueryPool-firstQuery-00796",
                    "VUID-vkCmdResetQueryPool-firstQuery-00797",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_query_pool_unchecked(
        &mut self,
        query_pool: &QueryPool,
        queries: Range<u32>,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_reset_query_pool)(
            self.handle(),
            query_pool.handle(),
            queries.start,
            queries.end - queries.start,
        );

        self
    }
}
