// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{CommandBufferBuilder, QueryError, QueryState};
use crate::{
    buffer::{BufferUsage, TypedBufferAccess},
    command_buffer::allocator::CommandBufferAllocator,
    device::{DeviceOwned, QueueFlags},
    query::{QueryControlFlags, QueryPool, QueryResultElement, QueryResultFlags, QueryType},
    sync::{PipelineStage, PipelineStages},
    DeviceSize, RequiresOneOf, Version, VulkanObject,
};
use std::{ops::Range, sync::Arc};

impl<L, A> CommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Begins a query.
    ///
    /// The query will be active until [`end_query`] is called for the same query.
    ///
    /// # Safety
    ///
    /// - The query must be unavailable, ensured by calling [`reset_query_pool`].
    ///
    /// [`end_query`]: Self::end_query
    /// [`reset_query_pool`]: Self::reset_query_pool
    #[inline]
    pub unsafe fn begin_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<&mut Self, QueryError> {
        self.validate_begin_query(&query_pool, query, flags)?;

        Ok(self.begin_query_unchecked(query_pool, query, flags))
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
            .current_state
            .queries
            .contains_key(&query_pool.query_type().into())
        {
            return Err(QueryError::QueryIsActive);
        }

        if let Some(render_pass_state) = &self.current_state.render_pass {
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

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_query_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        flags: QueryControlFlags,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_begin_query)(self.handle(), query_pool.handle(), query, flags.into());

        let ty = query_pool.query_type();
        self.current_state.queries.insert(
            ty.into(),
            QueryState {
                query_pool: query_pool.handle(),
                query,
                ty,
                flags,
                in_subpass: self.current_state.render_pass.is_some(),
            },
        );

        self.resources.push(Box::new(query_pool));

        self
    }

    /// Ends an active query.
    #[inline]
    pub fn end_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
    ) -> Result<&mut Self, QueryError> {
        self.validate_end_query(&query_pool, query)?;

        unsafe { Ok(self.end_query_unchecked(query_pool, query)) }
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
            .current_state
            .queries
            .get(&query_pool.query_type().into())
            .map_or(false, |state| {
                state.query_pool == query_pool.handle() && state.query == query
            })
        {
            return Err(QueryError::QueryNotActive);
        }

        // VUID-vkCmdEndQuery-query-00810
        query_pool.query(query).ok_or(QueryError::OutOfRange)?;

        if let Some(render_pass_state) = &self.current_state.render_pass {
            // VUID-vkCmdEndQuery-query-00812
            if query + render_pass_state.view_mask.count_ones() > query_pool.query_count() {
                return Err(QueryError::OutOfRangeMultiview);
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
        let fns = self.device().fns();
        (fns.v1_0.cmd_end_query)(self.handle(), query_pool.handle(), query);

        self.current_state
            .queries
            .remove(&query_pool.query_type().into());

        self.resources.push(Box::new(query_pool));

        self
    }

    /// Writes a timestamp to a timestamp query.
    ///
    /// # Safety
    ///
    /// - The query must be unavailable, ensured by calling [`reset_query_pool`].
    ///
    /// [`reset_query_pool`]: Self::reset_query_pool
    pub unsafe fn write_timestamp(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        stage: PipelineStage,
    ) -> Result<&mut Self, QueryError> {
        self.validate_write_timestamp(&query_pool, query, stage)?;

        Ok(self.write_timestamp_unchecked(query_pool, query, stage))
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

        if let Some(render_pass_state) = &self.current_state.render_pass {
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

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn write_timestamp_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
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
                debug_assert!(self.device().enabled_extensions().khr_synchronization2);
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

        self.resources.push(Box::new(query_pool));

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
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers
    ///   that are accessed by the command.
    ///
    /// [`query_pool.ty().result_len()`]: crate::query::QueryType::result_len
    /// [`QueryResultFlags::WITH_AVAILABILITY`]: crate::query::QueryResultFlags::WITH_AVAILABILITY
    /// [`get_results`]: crate::query::QueriesRange::get_results
    pub unsafe fn copy_query_pool_results<D, T>(
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
            let per_query_len = query_pool.query_type().result_len()
                + flags.intersects(QueryResultFlags::WITH_AVAILABILITY) as DeviceSize;
            let stride = per_query_len * std::mem::size_of::<T>() as DeviceSize;
            Ok(self.copy_query_pool_results_unchecked(
                query_pool,
                queries,
                destination,
                stride,
                flags,
            ))
        }
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
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdCopyQueryPoolResults-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(QueryError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdCopyQueryPoolResults-renderpass
        if self.current_state.render_pass.is_some() {
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
        if !buffer_inner
            .buffer
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

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_query_pool_results_unchecked<D, T>(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
        destination: Arc<D>,
        stride: DeviceSize,
        flags: QueryResultFlags,
    ) -> &mut Self
    where
        D: TypedBufferAccess<Content = [T]> + 'static,
        T: QueryResultElement,
    {
        let destination_inner = destination.inner();

        let fns = self.device().fns();
        (fns.v1_0.cmd_copy_query_pool_results)(
            self.handle(),
            query_pool.handle(),
            queries.start,
            queries.end - queries.start,
            destination_inner.buffer.handle(),
            destination_inner.offset,
            stride,
            ash::vk::QueryResultFlags::from(flags) | T::FLAG,
        );

        self.resources.push(Box::new(query_pool));
        self.resources.push(Box::new(destination));

        // TODO: sync state update

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
    ) -> Result<&mut Self, QueryError> {
        self.validate_reset_query_pool(&query_pool, queries.clone())?;

        Ok(self.reset_query_pool_unchecked(query_pool, queries))
    }

    fn validate_reset_query_pool(
        &self,
        query_pool: &QueryPool,
        queries: Range<u32>,
    ) -> Result<(), QueryError> {
        // VUID-vkCmdResetQueryPool-renderpass
        if self.current_state.render_pass.is_some() {
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
            .current_state
            .queries
            .values()
            .any(|state| state.query_pool == query_pool.handle() && queries.contains(&state.query))
        {
            return Err(QueryError::QueryIsActive);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_query_pool_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_reset_query_pool)(
            self.handle(),
            query_pool.handle(),
            queries.start,
            queries.end - queries.start,
        );

        self.resources.push(Box::new(query_pool));

        self
    }
}
