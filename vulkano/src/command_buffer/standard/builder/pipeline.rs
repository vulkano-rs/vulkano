// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    CommandBufferBuilder, DescriptorSetState, PipelineExecutionError, RenderPassState,
    RenderPassStateType, ResourcesState,
};
use crate::{
    buffer::{view::BufferViewAbstract, BufferAccess, BufferUsage, TypedBufferAccess},
    command_buffer::{
        allocator::CommandBufferAllocator, commands::pipeline::DescriptorResourceInvalidError,
        DispatchIndirectCommand, DrawIndexedIndirectCommand, DrawIndirectCommand,
        ResourceInCommand, ResourceUseRef, SubpassContents,
    },
    descriptor_set::{layout::DescriptorType, DescriptorBindingResources},
    device::{DeviceOwned, QueueFlags},
    format::FormatFeatures,
    image::{ImageAccess, ImageAspects, ImageViewAbstract, SampleCount},
    pipeline::{
        graphics::{
            input_assembly::{IndexType, PrimitiveTopology},
            render_pass::PipelineRenderPassType,
            vertex_input::VertexInputRate,
        },
        DynamicState, GraphicsPipeline, PartialStateMode, Pipeline, PipelineBindPoint,
        PipelineLayout,
    },
    sampler::Sampler,
    shader::{DescriptorBindingRequirements, ShaderScalarType, ShaderStage, ShaderStages},
    sync::PipelineStageAccess,
    RequiresOneOf, VulkanObject,
};
use ahash::HashMap;
use std::{cmp::min, mem::size_of, sync::Arc};

impl<L, A> CommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Perform a single compute operation using a compute pipeline.
    ///
    /// A compute pipeline must have been bound using [`bind_pipeline_compute`]. Any resources used
    /// by the compute pipeline, such as descriptor sets, must have been set beforehand.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers and images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    ///
    /// [`bind_pipeline_compute`]: Self::bind_pipeline_compute
    #[inline]
    pub unsafe fn dispatch(
        &mut self,
        group_counts: [u32; 3],
    ) -> Result<&mut Self, PipelineExecutionError> {
        self.validate_dispatch(group_counts)?;

        unsafe { Ok(self.dispatch_unchecked(group_counts)) }
    }

    fn validate_dispatch(&self, group_counts: [u32; 3]) -> Result<(), PipelineExecutionError> {
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdDispatch-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(PipelineExecutionError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdDispatch-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(PipelineExecutionError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdDispatch-None-02700
        let pipeline = self
            .builder_state
            .pipeline_compute
            .as_ref()
            .ok_or(PipelineExecutionError::PipelineNotBound)?
            .as_ref();

        self.validate_pipeline_descriptor_sets(pipeline)?;
        self.validate_pipeline_push_constants(pipeline.layout())?;

        let max = self
            .device()
            .physical_device()
            .properties()
            .max_compute_work_group_count;

        // VUID-vkCmdDispatch-groupCountX-00386
        // VUID-vkCmdDispatch-groupCountY-00387
        // VUID-vkCmdDispatch-groupCountZ-00388
        if group_counts[0] > max[0] || group_counts[1] > max[1] || group_counts[2] > max[2] {
            return Err(PipelineExecutionError::MaxComputeWorkGroupCountExceeded {
                requested: group_counts,
                max,
            });
        }

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn dispatch_unchecked(&mut self, group_counts: [u32; 3]) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_dispatch)(
            self.handle(),
            group_counts[0],
            group_counts[1],
            group_counts[2],
        );

        let command_index = self.next_command_index;
        let command_name = "dispatch";
        let pipeline = self
            .builder_state
            .pipeline_compute
            .as_ref()
            .unwrap()
            .as_ref();
        record_descriptor_sets_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.descriptor_sets,
            pipeline,
        );

        self.next_command_index += 1;
        self
    }

    /// Perform multiple compute operations using a compute pipeline. One dispatch is performed for
    /// each [`DispatchIndirectCommand`] struct in `indirect_buffer`.
    ///
    /// A compute pipeline must have been bound using [`bind_pipeline_compute`]. Any resources used
    /// by the compute pipeline, such as descriptor sets, must have been set beforehand.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers and images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    ///
    /// [`bind_pipeline_compute`]: Self::bind_pipeline_compute
    #[inline]
    pub unsafe fn dispatch_indirect(
        &mut self,
        indirect_buffer: Arc<impl TypedBufferAccess<Content = [DispatchIndirectCommand]> + 'static>,
    ) -> Result<&mut Self, PipelineExecutionError> {
        self.validate_dispatch_indirect(&indirect_buffer)?;

        unsafe { Ok(self.dispatch_indirect_unchecked(indirect_buffer)) }
    }

    fn validate_dispatch_indirect(
        &self,
        indirect_buffer: &dyn BufferAccess,
    ) -> Result<(), PipelineExecutionError> {
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdDispatchIndirect-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(PipelineExecutionError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdDispatchIndirect-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(PipelineExecutionError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdDispatchIndirect-None-02700
        let pipeline = self
            .builder_state
            .pipeline_compute
            .as_ref()
            .ok_or(PipelineExecutionError::PipelineNotBound)?
            .as_ref();

        self.validate_pipeline_descriptor_sets(pipeline)?;
        self.validate_pipeline_push_constants(pipeline.layout())?;
        self.validate_indirect_buffer(indirect_buffer)?;

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn dispatch_indirect_unchecked(
        &mut self,
        indirect_buffer: Arc<dyn BufferAccess>,
    ) -> &mut Self {
        let indirect_buffer_inner = indirect_buffer.inner();

        let fns = self.device().fns();
        (fns.v1_0.cmd_dispatch_indirect)(
            self.handle(),
            indirect_buffer_inner.buffer.handle(),
            indirect_buffer_inner.offset,
        );

        let command_index = self.next_command_index;
        let command_name = "dispatch_indirect";
        let pipeline = self
            .builder_state
            .pipeline_compute
            .as_ref()
            .unwrap()
            .as_ref();
        record_descriptor_sets_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.descriptor_sets,
            pipeline,
        );
        record_indirect_buffer_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &indirect_buffer,
        );

        self.resources.push(Box::new(indirect_buffer));

        self.next_command_index += 1;
        self
    }

    /// Perform a single draw operation using a graphics pipeline.
    ///
    /// The parameters specify the first vertex and the number of vertices to draw, and the first
    /// instance and number of instances. For non-instanced drawing, specify `instance_count` as 1
    /// and `first_instance` as 0.
    ///
    /// A graphics pipeline must have been bound using [`bind_pipeline_graphics`]. Any resources
    /// used by the graphics pipeline, such as descriptor sets, vertex buffers and dynamic state,
    /// must have been set beforehand. If the bound graphics pipeline uses vertex buffers, then the
    /// provided vertex and instance ranges must be in range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers and images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    ///
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    #[inline]
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<&mut Self, PipelineExecutionError> {
        self.validate_draw(vertex_count, instance_count, first_vertex, first_instance)?;

        unsafe {
            Ok(self.draw_unchecked(vertex_count, instance_count, first_vertex, first_instance))
        }
    }

    fn validate_draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), PipelineExecutionError> {
        // VUID-vkCmdDraw-renderpass
        let render_pass_state = self
            .builder_state
            .render_pass
            .as_ref()
            .ok_or(PipelineExecutionError::ForbiddenOutsideRenderPass)?;

        // VUID-vkCmdDraw-None-02700
        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or(PipelineExecutionError::PipelineNotBound)?
            .as_ref();

        self.validate_pipeline_descriptor_sets(pipeline)?;
        self.validate_pipeline_push_constants(pipeline.layout())?;
        self.validate_pipeline_graphics_dynamic_state(pipeline)?;
        self.validate_pipeline_graphics_render_pass(pipeline, render_pass_state)?;
        self.validate_pipeline_graphics_vertex_buffers(
            pipeline,
            Some((first_vertex, vertex_count)),
            Some((first_instance, instance_count)),
        )?;

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_unchecked(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_draw)(
            self.handle(),
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );

        let command_index = self.next_command_index;
        let command_name = "draw";
        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .unwrap()
            .as_ref();
        record_descriptor_sets_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.descriptor_sets,
            pipeline,
        );
        record_vertex_buffers_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.vertex_buffers,
            pipeline,
        );

        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        self.next_command_index += 1;
        self
    }

    /// Perform multiple draw operations using a graphics pipeline.
    ///
    /// One draw is performed for each [`DrawIndirectCommand`] struct in `indirect_buffer`.
    /// The maximum number of draw commands in the buffer is limited by the
    /// [`max_draw_indirect_count`] limit.
    /// This limit is 1 unless the [`multi_draw_indirect`] feature has been enabled on the device.
    ///
    /// A graphics pipeline must have been bound using [`bind_pipeline_graphics`]. Any resources
    /// used by the graphics pipeline, such as descriptor sets, vertex buffers and dynamic state,
    /// must have been set beforehand. If the bound graphics pipeline uses vertex buffers, then the
    /// vertex and instance ranges of each `DrawIndirectCommand` in the indirect buffer must be in
    /// range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers and images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    ///
    /// [`max_draw_indirect_count`]: crate::device::Properties::max_draw_indirect_count
    /// [`multi_draw_indirect`]: crate::device::Features::multi_draw_indirect
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    #[inline]
    pub unsafe fn draw_indirect(
        &mut self,
        indirect_buffer: Arc<impl TypedBufferAccess<Content = [DrawIndirectCommand]> + 'static>,
    ) -> Result<&mut Self, PipelineExecutionError> {
        let draw_count = indirect_buffer.len() as u32;
        let stride = size_of::<DrawIndirectCommand>() as u32;
        self.validate_draw_indirect(&indirect_buffer, draw_count, stride)?;

        unsafe { Ok(self.draw_indirect_unchecked(indirect_buffer, draw_count, stride)) }
    }

    fn validate_draw_indirect(
        &self,
        indirect_buffer: &dyn BufferAccess,
        draw_count: u32,
        _stride: u32,
    ) -> Result<(), PipelineExecutionError> {
        // VUID-vkCmdDrawIndirect-renderpass
        let render_pass_state = self
            .builder_state
            .render_pass
            .as_ref()
            .ok_or(PipelineExecutionError::ForbiddenOutsideRenderPass)?;

        // VUID-vkCmdDrawIndirect-None-02700
        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or(PipelineExecutionError::PipelineNotBound)?
            .as_ref();

        self.validate_pipeline_descriptor_sets(pipeline)?;
        self.validate_pipeline_push_constants(pipeline.layout())?;
        self.validate_pipeline_graphics_dynamic_state(pipeline)?;
        self.validate_pipeline_graphics_render_pass(pipeline, render_pass_state)?;
        self.validate_pipeline_graphics_vertex_buffers(pipeline, None, None)?;

        self.validate_indirect_buffer(indirect_buffer)?;

        // VUID-vkCmdDrawIndirect-drawCount-02718
        if draw_count > 1 && !self.device().enabled_features().multi_draw_indirect {
            return Err(PipelineExecutionError::RequirementNotMet {
                required_for: "`draw_count` is greater than `1`",
                requires_one_of: RequiresOneOf {
                    features: &["multi_draw_indirect"],
                    ..Default::default()
                },
            });
        }

        let max = self
            .device()
            .physical_device()
            .properties()
            .max_draw_indirect_count;

        // VUID-vkCmdDrawIndirect-drawCount-02719
        if draw_count > max {
            return Err(PipelineExecutionError::MaxDrawIndirectCountExceeded {
                provided: draw_count,
                max,
            });
        }

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indirect_unchecked(
        &mut self,
        indirect_buffer: Arc<dyn BufferAccess>,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let indirect_buffer_inner = indirect_buffer.inner();

        let fns = self.device().fns();
        (fns.v1_0.cmd_draw_indirect)(
            self.handle(),
            indirect_buffer_inner.buffer.handle(),
            indirect_buffer_inner.offset,
            draw_count,
            stride,
        );

        let command_index = self.next_command_index;
        let command_name = "draw_indirect";
        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .unwrap()
            .as_ref();
        record_descriptor_sets_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.descriptor_sets,
            pipeline,
        );
        record_vertex_buffers_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.vertex_buffers,
            pipeline,
        );
        record_indirect_buffer_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &indirect_buffer,
        );

        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        self.resources.push(Box::new(indirect_buffer));

        self.next_command_index += 1;
        self
    }

    /// Perform a single draw operation using a graphics pipeline, using an index buffer.
    ///
    /// The parameters specify the first index and the number of indices in the index buffer that
    /// should be used, and the first instance and number of instances. For non-instanced drawing,
    /// specify `instance_count` as 1 and `first_instance` as 0. The `vertex_offset` is a constant
    /// value that should be added to each index in the index buffer to produce the final vertex
    /// number to be used.
    ///
    /// An index buffer must have been bound using [`bind_index_buffer`], and the provided index
    /// range must be in range of the bound index buffer.
    ///
    /// A graphics pipeline must have been bound using [`bind_pipeline_graphics`]. Any resources
    /// used by the graphics pipeline, such as descriptor sets, vertex buffers and dynamic state,
    /// must have been set beforehand. If the bound graphics pipeline uses vertex buffers, then the
    /// provided instance range must be in range of the bound vertex buffers. The vertex indices in
    /// the index buffer must be in range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers and images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    ///
    /// [`bind_index_buffer`]: Self::bind_index_buffer
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    #[inline]
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<&mut Self, PipelineExecutionError> {
        self.validate_draw_indexed(
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        )?;

        unsafe {
            Ok(self.draw_indexed_unchecked(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            ))
        }
    }

    fn validate_draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        _vertex_offset: i32,
        first_instance: u32,
    ) -> Result<(), PipelineExecutionError> {
        // TODO: how to handle an index out of range of the vertex buffers?

        // VUID-vkCmdDrawIndexed-renderpass
        let render_pass_state = self
            .builder_state
            .render_pass
            .as_ref()
            .ok_or(PipelineExecutionError::ForbiddenOutsideRenderPass)?;

        // VUID-vkCmdDrawIndexed-None-02700
        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or(PipelineExecutionError::PipelineNotBound)?
            .as_ref();

        self.validate_pipeline_descriptor_sets(pipeline)?;
        self.validate_pipeline_push_constants(pipeline.layout())?;
        self.validate_pipeline_graphics_dynamic_state(pipeline)?;
        self.validate_pipeline_graphics_render_pass(pipeline, render_pass_state)?;
        self.validate_pipeline_graphics_vertex_buffers(
            pipeline,
            None,
            Some((first_instance, instance_count)),
        )?;

        self.validate_index_buffer(Some((first_index, index_count)))?;

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_unchecked(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_draw_indexed)(
            self.handle(),
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );

        let command_index = self.next_command_index;
        let command_name = "draw_indexed";
        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .unwrap()
            .as_ref();
        record_descriptor_sets_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.descriptor_sets,
            pipeline,
        );
        record_vertex_buffers_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.vertex_buffers,
            pipeline,
        );
        record_index_buffer_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.index_buffer,
        );

        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        self.next_command_index += 1;
        self
    }

    /// Perform multiple draw operations using a graphics pipeline, using an index buffer.
    ///
    /// One draw is performed for each [`DrawIndexedIndirectCommand`] struct in `indirect_buffer`.
    /// The maximum number of draw commands in the buffer is limited by the
    /// [`max_draw_indirect_count`] limit.
    /// This limit is 1 unless the [`multi_draw_indirect`] feature has been enabled on the device.
    ///
    /// An index buffer must have been bound using [`bind_index_buffer`], and the index ranges of
    /// each `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound
    /// index buffer.
    ///
    /// A graphics pipeline must have been bound using [`bind_pipeline_graphics`]. Any resources
    /// used by the graphics pipeline, such as descriptor sets, vertex buffers and dynamic state,
    /// must have been set beforehand. If the bound graphics pipeline uses vertex buffers, then the
    /// instance ranges of each `DrawIndexedIndirectCommand` in the indirect buffer must be in
    /// range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers and images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    ///
    /// [`max_draw_indirect_count`]: crate::device::Properties::max_draw_indirect_count
    /// [`multi_draw_indirect`]: crate::device::Features::multi_draw_indirect
    /// [`bind_index_buffer`]: Self::bind_index_buffer
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    #[inline]
    pub unsafe fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: Arc<
            impl TypedBufferAccess<Content = [DrawIndexedIndirectCommand]> + 'static,
        >,
    ) -> Result<&mut Self, PipelineExecutionError> {
        let draw_count = indirect_buffer.len() as u32;
        let stride = size_of::<DrawIndexedIndirectCommand>() as u32;
        self.validate_draw_indexed_indirect(&indirect_buffer, draw_count, stride)?;

        unsafe { Ok(self.draw_indexed_indirect_unchecked(indirect_buffer, draw_count, stride)) }
    }

    fn validate_draw_indexed_indirect(
        &self,
        indirect_buffer: &dyn BufferAccess,
        draw_count: u32,
        _stride: u32,
    ) -> Result<(), PipelineExecutionError> {
        // VUID-vkCmdDrawIndexedIndirect-renderpass
        let render_pass_state = self
            .builder_state
            .render_pass
            .as_ref()
            .ok_or(PipelineExecutionError::ForbiddenOutsideRenderPass)?;

        // VUID-vkCmdDrawIndexedIndirect-None-02700
        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or(PipelineExecutionError::PipelineNotBound)?
            .as_ref();

        self.validate_pipeline_descriptor_sets(pipeline)?;
        self.validate_pipeline_push_constants(pipeline.layout())?;
        self.validate_pipeline_graphics_dynamic_state(pipeline)?;
        self.validate_pipeline_graphics_render_pass(pipeline, render_pass_state)?;
        self.validate_pipeline_graphics_vertex_buffers(pipeline, None, None)?;

        self.validate_index_buffer(None)?;
        self.validate_indirect_buffer(indirect_buffer)?;

        // VUID-vkCmdDrawIndexedIndirect-drawCount-02718
        if draw_count > 1 && !self.device().enabled_features().multi_draw_indirect {
            return Err(PipelineExecutionError::RequirementNotMet {
                required_for: "`draw_count` is greater than `1`",
                requires_one_of: RequiresOneOf {
                    features: &["multi_draw_indirect"],
                    ..Default::default()
                },
            });
        }

        let max = self
            .device()
            .physical_device()
            .properties()
            .max_draw_indirect_count;

        // VUID-vkCmdDrawIndexedIndirect-drawCount-02719
        if draw_count > max {
            return Err(PipelineExecutionError::MaxDrawIndirectCountExceeded {
                provided: draw_count,
                max,
            });
        }

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_indirect_unchecked(
        &mut self,
        indirect_buffer: Arc<dyn BufferAccess>,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let indirect_buffer_inner = indirect_buffer.inner();

        let fns = self.device().fns();
        (fns.v1_0.cmd_draw_indexed_indirect)(
            self.handle(),
            indirect_buffer_inner.buffer.handle(),
            indirect_buffer_inner.offset,
            draw_count,
            stride,
        );

        let command_index = self.next_command_index;
        let command_name = "draw_indexed_indirect";
        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .unwrap()
            .as_ref();
        record_descriptor_sets_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.descriptor_sets,
            pipeline,
        );
        record_vertex_buffers_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.vertex_buffers,
            pipeline,
        );
        record_index_buffer_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &self.builder_state.index_buffer,
        );
        record_indirect_buffer_access(
            &mut self.resources_usage_state,
            command_index,
            command_name,
            &indirect_buffer,
        );

        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        self.resources.push(Box::new(indirect_buffer));

        self.next_command_index += 1;
        self
    }

    fn validate_index_buffer(
        &self,
        indices: Option<(u32, u32)>,
    ) -> Result<(), PipelineExecutionError> {
        // VUID?
        let (index_buffer, index_type) = self
            .builder_state
            .index_buffer
            .as_ref()
            .ok_or(PipelineExecutionError::IndexBufferNotBound)?;

        if let Some((first_index, index_count)) = indices {
            let max_index_count = (index_buffer.size() / index_type.size()) as u32;

            // // VUID-vkCmdDrawIndexed-firstIndex-04932
            if first_index + index_count > max_index_count {
                return Err(PipelineExecutionError::IndexBufferRangeOutOfBounds {
                    highest_index: first_index + index_count,
                    max_index_count,
                });
            }
        }

        Ok(())
    }

    fn validate_indirect_buffer(
        &self,
        buffer: &dyn BufferAccess,
    ) -> Result<(), PipelineExecutionError> {
        // VUID-vkCmdDispatchIndirect-commonparent
        assert_eq!(self.device(), buffer.device());

        // VUID-vkCmdDispatchIndirect-buffer-02709
        if !buffer.usage().intersects(BufferUsage::INDIRECT_BUFFER) {
            return Err(PipelineExecutionError::IndirectBufferMissingUsage);
        }

        // VUID-vkCmdDispatchIndirect-offset-02710
        // TODO:

        Ok(())
    }

    fn validate_pipeline_descriptor_sets(
        &self,
        pipeline: &impl Pipeline,
    ) -> Result<(), PipelineExecutionError> {
        fn validate_resources<T>(
            set_num: u32,
            binding_num: u32,
            binding_reqs: &DescriptorBindingRequirements,
            elements: &[Option<T>],
            mut extra_check: impl FnMut(u32, &T) -> Result<(), DescriptorResourceInvalidError>,
        ) -> Result<(), PipelineExecutionError> {
            let elements_to_check = if let Some(descriptor_count) = binding_reqs.descriptor_count {
                // The shader has a fixed-sized array, so it will never access more than
                // the first `descriptor_count` elements.
                elements.get(..descriptor_count as usize).ok_or({
                    // There are less than `descriptor_count` elements in `elements`
                    PipelineExecutionError::DescriptorResourceInvalid {
                        set_num,
                        binding_num,
                        index: elements.len() as u32,
                        error: DescriptorResourceInvalidError::Missing,
                    }
                })?
            } else {
                // The shader has a runtime-sized array, so any element could potentially
                // be accessed. We must check them all.
                elements
            };

            for (index, element) in elements_to_check.iter().enumerate() {
                let index = index as u32;

                // VUID-vkCmdDispatch-None-02699
                let element = match element {
                    Some(x) => x,
                    None => {
                        return Err(PipelineExecutionError::DescriptorResourceInvalid {
                            set_num,
                            binding_num,
                            index,
                            error: DescriptorResourceInvalidError::Missing,
                        })
                    }
                };

                if let Err(error) = extra_check(index, element) {
                    return Err(PipelineExecutionError::DescriptorResourceInvalid {
                        set_num,
                        binding_num,
                        index,
                        error,
                    });
                }
            }

            Ok(())
        }

        if pipeline.num_used_descriptor_sets() == 0 {
            return Ok(());
        }

        // VUID-vkCmdDispatch-None-02697
        let descriptor_set_state = self
            .builder_state
            .descriptor_sets
            .get(&pipeline.bind_point())
            .ok_or(PipelineExecutionError::PipelineLayoutNotCompatible)?;

        // VUID-vkCmdDispatch-None-02697
        if !pipeline.layout().is_compatible_with(
            &descriptor_set_state.pipeline_layout,
            pipeline.num_used_descriptor_sets(),
        ) {
            return Err(PipelineExecutionError::PipelineLayoutNotCompatible);
        }

        for (&(set_num, binding_num), binding_reqs) in pipeline.descriptor_binding_requirements() {
            let layout_binding =
                &pipeline.layout().set_layouts()[set_num as usize].bindings()[&binding_num];

            let check_buffer = |_index: u32, _buffer: &Arc<dyn BufferAccess>| Ok(());

            let check_buffer_view = |index: u32, buffer_view: &Arc<dyn BufferViewAbstract>| {
                for desc_reqs in (binding_reqs.descriptors.get(&Some(index)).into_iter())
                    .chain(binding_reqs.descriptors.get(&None))
                {
                    if layout_binding.descriptor_type == DescriptorType::StorageTexelBuffer {
                        // VUID-vkCmdDispatch-OpTypeImage-06423
                        if binding_reqs.image_format.is_none()
                            && !desc_reqs.memory_write.is_empty()
                            && !buffer_view
                                .format_features()
                                .intersects(FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT)
                        {
                            return Err(DescriptorResourceInvalidError::StorageWriteWithoutFormatNotSupported);
                        }

                        // VUID-vkCmdDispatch-OpTypeImage-06424
                        if binding_reqs.image_format.is_none()
                            && !desc_reqs.memory_read.is_empty()
                            && !buffer_view
                                .format_features()
                                .intersects(FormatFeatures::STORAGE_READ_WITHOUT_FORMAT)
                        {
                            return Err(DescriptorResourceInvalidError::StorageReadWithoutFormatNotSupported);
                        }
                    }
                }

                Ok(())
            };

            let check_image_view_common = |index: u32, image_view: &Arc<dyn ImageViewAbstract>| {
                for desc_reqs in (binding_reqs.descriptors.get(&Some(index)).into_iter())
                    .chain(binding_reqs.descriptors.get(&None))
                {
                    // VUID-vkCmdDispatch-None-02691
                    if desc_reqs.storage_image_atomic
                        && !image_view
                            .format_features()
                            .intersects(FormatFeatures::STORAGE_IMAGE_ATOMIC)
                    {
                        return Err(DescriptorResourceInvalidError::StorageImageAtomicNotSupported);
                    }

                    if layout_binding.descriptor_type == DescriptorType::StorageImage {
                        // VUID-vkCmdDispatch-OpTypeImage-06423
                        if binding_reqs.image_format.is_none()
                            && !desc_reqs.memory_write.is_empty()
                            && !image_view
                                .format_features()
                                .intersects(FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT)
                        {
                            return Err(
                            DescriptorResourceInvalidError::StorageWriteWithoutFormatNotSupported,
                        );
                        }

                        // VUID-vkCmdDispatch-OpTypeImage-06424
                        if binding_reqs.image_format.is_none()
                            && !desc_reqs.memory_read.is_empty()
                            && !image_view
                                .format_features()
                                .intersects(FormatFeatures::STORAGE_READ_WITHOUT_FORMAT)
                        {
                            return Err(
                            DescriptorResourceInvalidError::StorageReadWithoutFormatNotSupported,
                        );
                        }
                    }
                }

                /*
                   Instruction/Sampler/Image View Validation
                   https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap16.html#textures-input-validation
                */

                // The SPIR-V Image Format is not compatible with the image view’s format.
                if let Some(format) = binding_reqs.image_format {
                    if image_view.format() != Some(format) {
                        return Err(DescriptorResourceInvalidError::ImageViewFormatMismatch {
                            required: format,
                            provided: image_view.format(),
                        });
                    }
                }

                // Rules for viewType
                if let Some(image_view_type) = binding_reqs.image_view_type {
                    if image_view.view_type() != image_view_type {
                        return Err(DescriptorResourceInvalidError::ImageViewTypeMismatch {
                            required: image_view_type,
                            provided: image_view.view_type(),
                        });
                    }
                }

                // - If the image was created with VkImageCreateInfo::samples equal to
                //   VK_SAMPLE_COUNT_1_BIT, the instruction must have MS = 0.
                // - If the image was created with VkImageCreateInfo::samples not equal to
                //   VK_SAMPLE_COUNT_1_BIT, the instruction must have MS = 1.
                if binding_reqs.image_multisampled
                    != (image_view.image().samples() != SampleCount::Sample1)
                {
                    return Err(
                        DescriptorResourceInvalidError::ImageViewMultisampledMismatch {
                            required: binding_reqs.image_multisampled,
                            provided: image_view.image().samples() != SampleCount::Sample1,
                        },
                    );
                }

                // - If the Sampled Type of the OpTypeImage does not match the numeric format of the
                //   image, as shown in the SPIR-V Sampled Type column of the
                //   Interpretation of Numeric Format table.
                // - If the signedness of any read or sample operation does not match the signedness of
                //   the image’s format.
                if let Some(scalar_type) = binding_reqs.image_scalar_type {
                    let aspects = image_view.subresource_range().aspects;
                    let view_scalar_type = ShaderScalarType::from(
                        if aspects.intersects(
                            ImageAspects::COLOR
                                | ImageAspects::PLANE_0
                                | ImageAspects::PLANE_1
                                | ImageAspects::PLANE_2,
                        ) {
                            image_view.format().unwrap().type_color().unwrap()
                        } else if aspects.intersects(ImageAspects::DEPTH) {
                            image_view.format().unwrap().type_depth().unwrap()
                        } else if aspects.intersects(ImageAspects::STENCIL) {
                            image_view.format().unwrap().type_stencil().unwrap()
                        } else {
                            // Per `ImageViewBuilder::aspects` and
                            // VUID-VkDescriptorImageInfo-imageView-01976
                            unreachable!()
                        },
                    );

                    if scalar_type != view_scalar_type {
                        return Err(
                            DescriptorResourceInvalidError::ImageViewScalarTypeMismatch {
                                required: scalar_type,
                                provided: view_scalar_type,
                            },
                        );
                    }
                }

                Ok(())
            };

            let check_sampler_common = |index: u32, sampler: &Arc<Sampler>| {
                for desc_reqs in (binding_reqs.descriptors.get(&Some(index)).into_iter())
                    .chain(binding_reqs.descriptors.get(&None))
                {
                    // VUID-vkCmdDispatch-None-02703
                    // VUID-vkCmdDispatch-None-02704
                    if desc_reqs.sampler_no_unnormalized_coordinates
                        && sampler.unnormalized_coordinates()
                    {
                        return Err(
                        DescriptorResourceInvalidError::SamplerUnnormalizedCoordinatesNotAllowed,
                    );
                    }

                    // - OpImageFetch, OpImageSparseFetch, OpImage*Gather, and OpImageSparse*Gather must not
                    //   be used with a sampler that enables sampler Y′CBCR conversion.
                    // - The ConstOffset and Offset operands must not be used with a sampler that enables
                    //   sampler Y′CBCR conversion.
                    if desc_reqs.sampler_no_ycbcr_conversion
                        && sampler.sampler_ycbcr_conversion().is_some()
                    {
                        return Err(
                            DescriptorResourceInvalidError::SamplerYcbcrConversionNotAllowed,
                        );
                    }

                    /*
                        Instruction/Sampler/Image View Validation
                        https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap16.html#textures-input-validation
                    */

                    // - The SPIR-V instruction is one of the OpImage*Dref* instructions and the sampler
                    //   compareEnable is VK_FALSE
                    // - The SPIR-V instruction is not one of the OpImage*Dref* instructions and the sampler
                    //   compareEnable is VK_TRUE
                    if desc_reqs.sampler_compare != sampler.compare().is_some() {
                        return Err(DescriptorResourceInvalidError::SamplerCompareMismatch {
                            required: desc_reqs.sampler_compare,
                            provided: sampler.compare().is_some(),
                        });
                    }
                }

                Ok(())
            };

            let check_image_view = |index: u32, image_view: &Arc<dyn ImageViewAbstract>| {
                check_image_view_common(index, image_view)?;

                if let Some(sampler) = layout_binding.immutable_samplers.get(index as usize) {
                    check_sampler_common(index, sampler)?;
                }

                Ok(())
            };

            let check_image_view_sampler =
                |index: u32, (image_view, sampler): &(Arc<dyn ImageViewAbstract>, Arc<Sampler>)| {
                    check_image_view_common(index, image_view)?;
                    check_sampler_common(index, sampler)?;

                    Ok(())
                };

            let check_sampler = |index: u32, sampler: &Arc<Sampler>| {
                check_sampler_common(index, sampler)?;

                for desc_reqs in (binding_reqs.descriptors.get(&Some(index)).into_iter())
                    .chain(binding_reqs.descriptors.get(&None))
                {
                    // Check sampler-image compatibility. Only done for separate samplers;
                    // combined image samplers are checked when updating the descriptor set.

                    // If the image view isn't actually present in the resources, then just skip it.
                    // It will be caught later by check_resources.
                    let iter = desc_reqs.sampler_with_images.iter().filter_map(|id| {
                        descriptor_set_state
                            .descriptor_sets
                            .get(&id.set)
                            .and_then(|set| set.resources().binding(id.binding))
                            .and_then(|res| match res {
                                DescriptorBindingResources::ImageView(elements) => elements
                                    .get(id.index as usize)
                                    .and_then(|opt| opt.as_ref().map(|opt| (id, opt))),
                                _ => None,
                            })
                    });

                    for (id, image_view) in iter {
                        if let Err(error) = sampler.check_can_sample(image_view.as_ref()) {
                            return Err(
                                DescriptorResourceInvalidError::SamplerImageViewIncompatible {
                                    image_view_set_num: id.set,
                                    image_view_binding_num: id.binding,
                                    image_view_index: id.index,
                                    error,
                                },
                            );
                        }
                    }
                }

                Ok(())
            };

            let check_none = |index: u32, _: &()| {
                if let Some(sampler) = layout_binding.immutable_samplers.get(index as usize) {
                    check_sampler(index, sampler)?;
                }

                Ok(())
            };

            let set_resources = descriptor_set_state
                .descriptor_sets
                .get(&set_num)
                .ok_or(PipelineExecutionError::DescriptorSetNotBound { set_num })?
                .resources();

            let binding_resources = set_resources.binding(binding_num).unwrap();

            match binding_resources {
                DescriptorBindingResources::None(elements) => {
                    validate_resources(set_num, binding_num, binding_reqs, elements, check_none)?;
                }
                DescriptorBindingResources::Buffer(elements) => {
                    validate_resources(set_num, binding_num, binding_reqs, elements, check_buffer)?;
                }
                DescriptorBindingResources::BufferView(elements) => {
                    validate_resources(
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_buffer_view,
                    )?;
                }
                DescriptorBindingResources::ImageView(elements) => {
                    validate_resources(
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_image_view,
                    )?;
                }
                DescriptorBindingResources::ImageViewSampler(elements) => {
                    validate_resources(
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_image_view_sampler,
                    )?;
                }
                DescriptorBindingResources::Sampler(elements) => {
                    validate_resources(
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_sampler,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn validate_pipeline_push_constants(
        &self,
        pipeline_layout: &PipelineLayout,
    ) -> Result<(), PipelineExecutionError> {
        if pipeline_layout.push_constant_ranges().is_empty()
            || self.device().enabled_features().maintenance4
        {
            return Ok(());
        }

        // VUID-vkCmdDispatch-maintenance4-06425
        let constants_pipeline_layout = self
            .builder_state
            .push_constants_pipeline_layout
            .as_ref()
            .ok_or(PipelineExecutionError::PushConstantsMissing)?;

        // VUID-vkCmdDispatch-maintenance4-06425
        if pipeline_layout.handle() != constants_pipeline_layout.handle()
            && pipeline_layout.push_constant_ranges()
                != constants_pipeline_layout.push_constant_ranges()
        {
            return Err(PipelineExecutionError::PushConstantsNotCompatible);
        }

        let set_bytes = &self.builder_state.push_constants;

        // VUID-vkCmdDispatch-maintenance4-06425
        if !pipeline_layout
            .push_constant_ranges()
            .iter()
            .all(|pc_range| set_bytes.contains(pc_range.offset..pc_range.offset + pc_range.size))
        {
            return Err(PipelineExecutionError::PushConstantsMissing);
        }

        Ok(())
    }

    fn validate_pipeline_graphics_dynamic_state(
        &self,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), PipelineExecutionError> {
        let device = pipeline.device();

        // VUID-vkCmdDraw-commandBuffer-02701
        for dynamic_state in pipeline
            .dynamic_states()
            .filter(|(_, d)| *d)
            .map(|(s, _)| s)
        {
            match dynamic_state {
                DynamicState::BlendConstants => {
                    // VUID?
                    if self.builder_state.blend_constants.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::ColorWriteEnable => {
                    // VUID-vkCmdDraw-attachmentCount-06667
                    let enables = self.builder_state.color_write_enable.as_ref().ok_or(PipelineExecutionError::DynamicStateNotSet { dynamic_state })?;

                    // VUID-vkCmdDraw-attachmentCount-06667
                    if enables.len() < pipeline.color_blend_state().unwrap().attachments.len() {
                        return Err(
                            PipelineExecutionError::DynamicColorWriteEnableNotEnoughValues {
                                color_write_enable_count: enables.len() as u32,
                                attachment_count: pipeline
                                    .color_blend_state()
                                    .unwrap()
                                    .attachments
                                    .len() as u32,
                            },
                        );
                    }
                }
                DynamicState::CullMode => {
                    // VUID?
                    if self.builder_state.cull_mode.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthBias => {
                    // VUID?
                    if self.builder_state.depth_bias.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthBiasEnable => {
                    // VUID-vkCmdDraw-None-04877
                    if self.builder_state.depth_bias_enable.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthBounds => {
                    // VUID?
                    if self.builder_state.depth_bounds.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthBoundsTestEnable => {
                    // VUID?
                    if self.builder_state.depth_bounds_test_enable.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthCompareOp => {
                    // VUID?
                    if self.builder_state.depth_compare_op.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthTestEnable => {
                    // VUID?
                    if self.builder_state.depth_test_enable.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthWriteEnable => {
                    // VUID?
                    if self.builder_state.depth_write_enable.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }

                    // TODO: Check if the depth buffer is writable
                }
                DynamicState::DiscardRectangle => {
                    let discard_rectangle_count =
                        match pipeline.discard_rectangle_state().unwrap().rectangles {
                            PartialStateMode::Dynamic(count) => count,
                            _ => unreachable!(),
                        };

                    for num in 0..discard_rectangle_count {
                        // VUID?
                        if !self.builder_state.discard_rectangle.contains_key(&num) {
                            return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                        }
                    }
                }
                DynamicState::ExclusiveScissor => todo!(),
                DynamicState::FragmentShadingRate => todo!(),
                DynamicState::FrontFace => {
                    // VUID?
                    if self.builder_state.front_face.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::LineStipple => {
                    // VUID?
                    if self.builder_state.line_stipple.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::LineWidth => {
                    // VUID?
                    if self.builder_state.line_width.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::LogicOp => {
                    // VUID-vkCmdDraw-logicOp-04878
                    if self.builder_state.logic_op.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::PatchControlPoints => {
                    // VUID-vkCmdDraw-None-04875
                    if self.builder_state.patch_control_points.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::PrimitiveRestartEnable => {
                    // VUID-vkCmdDraw-None-04879
                    let primitive_restart_enable =
                        if let Some(enable) = self.builder_state.primitive_restart_enable {
                            enable
                        } else {
                            return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                        };

                    if primitive_restart_enable {
                        let topology = match pipeline.input_assembly_state().topology {
                            PartialStateMode::Fixed(topology) => topology,
                            PartialStateMode::Dynamic(_) => {
                                if let Some(topology) = self.builder_state.primitive_topology {
                                    topology
                                } else {
                                    return Err(PipelineExecutionError::DynamicStateNotSet {
                                        dynamic_state: DynamicState::PrimitiveTopology,
                                    });
                                }
                            }
                        };

                        match topology {
                            PrimitiveTopology::PointList
                            | PrimitiveTopology::LineList
                            | PrimitiveTopology::TriangleList
                            | PrimitiveTopology::LineListWithAdjacency
                            | PrimitiveTopology::TriangleListWithAdjacency => {
                                // VUID?
                                if !device.enabled_features().primitive_topology_list_restart {
                                    return Err(PipelineExecutionError::RequirementNotMet {
                                        required_for: "The bound pipeline sets \
                                            `DynamicState::PrimitiveRestartEnable` and the \
                                            current primitive topology is \
                                            `PrimitiveTopology::*List`",
                                        requires_one_of: RequiresOneOf {
                                            features: &["primitive_topology_list_restart"],
                                            ..Default::default()
                                        },
                                    });
                                }
                            }
                            PrimitiveTopology::PatchList => {
                                // VUID?
                                if !device
                                    .enabled_features()
                                    .primitive_topology_patch_list_restart
                                {
                                    return Err(PipelineExecutionError::RequirementNotMet {
                                        required_for: "The bound pipeline sets \
                                            `DynamicState::PrimitiveRestartEnable` and the \
                                            current primitive topology is \
                                            `PrimitiveTopology::PatchList`",
                                        requires_one_of: RequiresOneOf {
                                            features: &["primitive_topology_patch_list_restart"],
                                            ..Default::default()
                                        },
                                    });
                                }
                            }
                            _ => (),
                        }
                    }
                }
                DynamicState::PrimitiveTopology => {
                    // VUID-vkCmdDraw-primitiveTopology-03420
                    let topology = if let Some(topology) = self.builder_state.primitive_topology {
                        topology
                    } else {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    };

                    if pipeline.shader(ShaderStage::TessellationControl).is_some() {
                        // VUID?
                        if !matches!(topology, PrimitiveTopology::PatchList) {
                            return Err(PipelineExecutionError::DynamicPrimitiveTopologyInvalid {
                                topology,
                            });
                        }
                    } else {
                        // VUID?
                        if matches!(topology, PrimitiveTopology::PatchList) {
                            return Err(PipelineExecutionError::DynamicPrimitiveTopologyInvalid {
                                topology,
                            });
                        }
                    }

                    let required_topology_class = match pipeline.input_assembly_state().topology {
                        PartialStateMode::Dynamic(topology_class) => topology_class,
                        _ => unreachable!(),
                    };

                    // VUID-vkCmdDraw-primitiveTopology-03420
                    if topology.class() != required_topology_class {
                        return Err(
                            PipelineExecutionError::DynamicPrimitiveTopologyClassMismatch {
                                provided_class: topology.class(),
                                required_class: required_topology_class,
                            },
                        );
                    }

                    // TODO: check that the topology matches the geometry shader
                }
                DynamicState::RasterizerDiscardEnable => {
                    // VUID-vkCmdDraw-None-04876
                    if self.builder_state.rasterizer_discard_enable.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::RayTracingPipelineStackSize => unreachable!(
                    "RayTracingPipelineStackSize dynamic state should not occur on a graphics pipeline"
                ),
                DynamicState::SampleLocations => todo!(),
                DynamicState::Scissor => {
                    for num in 0..pipeline.viewport_state().unwrap().count().unwrap() {
                        // VUID?
                        if !self.builder_state.scissor.contains_key(&num) {
                            return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                        }
                    }
                }
                DynamicState::ScissorWithCount => {
                    // VUID-vkCmdDraw-scissorCount-03418
                    // VUID-vkCmdDraw-viewportCount-03419
                    let scissor_count = self.builder_state.scissor_with_count.as_ref().ok_or(PipelineExecutionError::DynamicStateNotSet { dynamic_state })?.len() as u32;

                    // Check if the counts match, but only if the viewport count is fixed.
                    // If the viewport count is also dynamic, then the
                    // DynamicState::ViewportWithCount match arm will handle it.
                    if let Some(viewport_count) = pipeline.viewport_state().unwrap().count() {
                        // VUID-vkCmdDraw-scissorCount-03418
                        if viewport_count != scissor_count {
                            return Err(
                                PipelineExecutionError::DynamicViewportScissorCountMismatch {
                                    viewport_count,
                                    scissor_count,
                                },
                            );
                        }
                    }
                }
                DynamicState::StencilCompareMask => {
                    let state = self.builder_state.stencil_compare_mask;

                    // VUID?
                    if state.front.is_none() || state.back.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::StencilOp => {
                    let state = self.builder_state.stencil_op;

                    // VUID?
                    if state.front.is_none() || state.back.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::StencilReference => {
                    let state = self.builder_state.stencil_reference;

                    // VUID?
                    if state.front.is_none() || state.back.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::StencilTestEnable => {
                    // VUID?
                    if self.builder_state.stencil_test_enable.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }

                    // TODO: Check if the stencil buffer is writable
                }
                DynamicState::StencilWriteMask => {
                    let state = self.builder_state.stencil_write_mask;

                    // VUID?
                    if state.front.is_none() || state.back.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::VertexInput => todo!(),
                DynamicState::VertexInputBindingStride => todo!(),
                DynamicState::Viewport => {
                    for num in 0..pipeline.viewport_state().unwrap().count().unwrap() {
                        // VUID?
                        if !self.builder_state.viewport.contains_key(&num) {
                            return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                        }
                    }
                }
                DynamicState::ViewportCoarseSampleOrder => todo!(),
                DynamicState::ViewportShadingRatePalette => todo!(),
                DynamicState::ViewportWithCount => {
                    // VUID-vkCmdDraw-viewportCount-03417
                    let viewport_count = self.builder_state.viewport_with_count.as_ref().ok_or(PipelineExecutionError::DynamicStateNotSet { dynamic_state })?.len() as u32;

                    let scissor_count = if let Some(scissor_count) =
                        pipeline.viewport_state().unwrap().count()
                    {
                        // The scissor count is fixed.
                        scissor_count
                    } else {
                        // VUID-vkCmdDraw-viewportCount-03419
                        // The scissor count is also dynamic.
                        self.builder_state.scissor_with_count.as_ref().ok_or(PipelineExecutionError::DynamicStateNotSet { dynamic_state })?.len() as u32
                    };

                    // VUID-vkCmdDraw-viewportCount-03417
                    // VUID-vkCmdDraw-viewportCount-03419
                    if viewport_count != scissor_count {
                        return Err(
                            PipelineExecutionError::DynamicViewportScissorCountMismatch {
                                viewport_count,
                                scissor_count,
                            },
                        );
                    }

                    // TODO: VUID-vkCmdDrawIndexed-primitiveFragmentShadingRateWithMultipleViewports-04552
                    // If the primitiveFragmentShadingRateWithMultipleViewports limit is not supported,
                    // the bound graphics pipeline was created with the
                    // VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT dynamic state enabled, and any of the
                    // shader stages of the bound graphics pipeline write to the PrimitiveShadingRateKHR
                    // built-in, then vkCmdSetViewportWithCountEXT must have been called in the current
                    // command buffer prior to this drawing command, and the viewportCount parameter of
                    // vkCmdSetViewportWithCountEXT must be 1
                }
                DynamicState::ViewportWScaling => todo!(),
                DynamicState::TessellationDomainOrigin => todo!(),
                DynamicState::DepthClampEnable => todo!(),
                DynamicState::PolygonMode => todo!(),
                DynamicState::RasterizationSamples => todo!(),
                DynamicState::SampleMask => todo!(),
                DynamicState::AlphaToCoverageEnable => todo!(),
                DynamicState::AlphaToOneEnable => todo!(),
                DynamicState::LogicOpEnable => todo!(),
                DynamicState::ColorBlendEnable => todo!(),
                DynamicState::ColorBlendEquation => todo!(),
                DynamicState::ColorWriteMask => todo!(),
                DynamicState::RasterizationStream => todo!(),
                DynamicState::ConservativeRasterizationMode => todo!(),
                DynamicState::ExtraPrimitiveOverestimationSize => todo!(),
                DynamicState::DepthClipEnable => todo!(),
                DynamicState::SampleLocationsEnable => todo!(),
                DynamicState::ColorBlendAdvanced => todo!(),
                DynamicState::ProvokingVertexMode => todo!(),
                DynamicState::LineRasterizationMode => todo!(),
                DynamicState::LineStippleEnable => todo!(),
                DynamicState::DepthClipNegativeOneToOne => todo!(),
                DynamicState::ViewportWScalingEnable => todo!(),
                DynamicState::ViewportSwizzle => todo!(),
                DynamicState::CoverageToColorEnable => todo!(),
                DynamicState::CoverageToColorLocation => todo!(),
                DynamicState::CoverageModulationMode => todo!(),
                DynamicState::CoverageModulationTableEnable => todo!(),
                DynamicState::CoverageModulationTable => todo!(),
                DynamicState::ShadingRateImageEnable => todo!(),
                DynamicState::RepresentativeFragmentTestEnable => todo!(),
                DynamicState::CoverageReductionMode => todo!(),
                
            }
        }

        Ok(())
    }

    fn validate_pipeline_graphics_render_pass(
        &self,
        pipeline: &GraphicsPipeline,
        render_pass_state: &RenderPassState,
    ) -> Result<(), PipelineExecutionError> {
        // VUID?
        if render_pass_state.contents != SubpassContents::Inline {
            return Err(PipelineExecutionError::ForbiddenWithSubpassContents {
                subpass_contents: render_pass_state.contents,
            });
        }

        match (&render_pass_state.render_pass, pipeline.render_pass()) {
            (
                RenderPassStateType::BeginRenderPass(state),
                PipelineRenderPassType::BeginRenderPass(pipeline_subpass),
            ) => {
                // VUID-vkCmdDraw-renderPass-02684
                if !pipeline_subpass
                    .render_pass()
                    .is_compatible_with(state.subpass.render_pass())
                {
                    return Err(PipelineExecutionError::PipelineRenderPassNotCompatible);
                }

                // VUID-vkCmdDraw-subpass-02685
                if pipeline_subpass.index() != state.subpass.index() {
                    return Err(PipelineExecutionError::PipelineSubpassMismatch {
                        pipeline: pipeline_subpass.index(),
                        current: state.subpass.index(),
                    });
                }
            }
            (
                RenderPassStateType::BeginRendering(current_rendering_info),
                PipelineRenderPassType::BeginRendering(pipeline_rendering_info),
            ) => {
                // VUID-vkCmdDraw-viewMask-06178
                if pipeline_rendering_info.view_mask != render_pass_state.view_mask {
                    return Err(PipelineExecutionError::PipelineViewMaskMismatch {
                        pipeline_view_mask: pipeline_rendering_info.view_mask,
                        required_view_mask: render_pass_state.view_mask,
                    });
                }

                // VUID-vkCmdDraw-colorAttachmentCount-06179
                if pipeline_rendering_info.color_attachment_formats.len()
                    != current_rendering_info.color_attachment_formats.len()
                {
                    return Err(
                        PipelineExecutionError::PipelineColorAttachmentCountMismatch {
                            pipeline_count: pipeline_rendering_info.color_attachment_formats.len()
                                as u32,
                            required_count: current_rendering_info.color_attachment_formats.len()
                                as u32,
                        },
                    );
                }

                for (color_attachment_index, required_format, pipeline_format) in
                    current_rendering_info
                        .color_attachment_formats
                        .iter()
                        .zip(
                            pipeline_rendering_info
                                .color_attachment_formats
                                .iter()
                                .copied(),
                        )
                        .enumerate()
                        .filter_map(|(i, (r, p))| r.map(|r| (i as u32, r, p)))
                {
                    // VUID-vkCmdDraw-colorAttachmentCount-06180
                    if Some(required_format) != pipeline_format {
                        return Err(
                            PipelineExecutionError::PipelineColorAttachmentFormatMismatch {
                                color_attachment_index,
                                pipeline_format,
                                required_format,
                            },
                        );
                    }
                }

                if let Some((required_format, pipeline_format)) = current_rendering_info
                    .depth_attachment_format
                    .map(|r| (r, pipeline_rendering_info.depth_attachment_format))
                {
                    // VUID-vkCmdDraw-pDepthAttachment-06181
                    if Some(required_format) != pipeline_format {
                        return Err(
                            PipelineExecutionError::PipelineDepthAttachmentFormatMismatch {
                                pipeline_format,
                                required_format,
                            },
                        );
                    }
                }

                if let Some((required_format, pipeline_format)) = current_rendering_info
                    .stencil_attachment_format
                    .map(|r| (r, pipeline_rendering_info.stencil_attachment_format))
                {
                    // VUID-vkCmdDraw-pStencilAttachment-06182
                    if Some(required_format) != pipeline_format {
                        return Err(
                            PipelineExecutionError::PipelineStencilAttachmentFormatMismatch {
                                pipeline_format,
                                required_format,
                            },
                        );
                    }
                }

                // VUID-vkCmdDraw-imageView-06172
                // VUID-vkCmdDraw-imageView-06173
                // VUID-vkCmdDraw-imageView-06174
                // VUID-vkCmdDraw-imageView-06175
                // VUID-vkCmdDraw-imageView-06176
                // VUID-vkCmdDraw-imageView-06177
                // TODO:
            }
            _ => return Err(PipelineExecutionError::PipelineRenderPassTypeMismatch),
        }

        // VUID-vkCmdDraw-None-02686
        // TODO:

        Ok(())
    }

    fn validate_pipeline_graphics_vertex_buffers(
        &self,
        pipeline: &GraphicsPipeline,
        vertices: Option<(u32, u32)>,
        instances: Option<(u32, u32)>,
    ) -> Result<(), PipelineExecutionError> {
        let vertex_input = pipeline.vertex_input_state();
        let mut vertices_in_buffers: Option<u64> = None;
        let mut instances_in_buffers: Option<u64> = None;

        for (&binding_num, binding_desc) in &vertex_input.bindings {
            // VUID-vkCmdDraw-None-04007
            let vertex_buffer = match self.builder_state.vertex_buffers.get(&binding_num) {
                Some(x) => x,
                None => return Err(PipelineExecutionError::VertexBufferNotBound { binding_num }),
            };

            let mut num_elements = vertex_buffer.size() as u64 / binding_desc.stride as u64;

            match binding_desc.input_rate {
                VertexInputRate::Vertex => {
                    vertices_in_buffers = Some(if let Some(x) = vertices_in_buffers {
                        min(x, num_elements)
                    } else {
                        num_elements
                    });
                }
                VertexInputRate::Instance { divisor } => {
                    if divisor == 0 {
                        // A divisor of 0 means the same instance data is used for all instances,
                        // so we can draw any number of instances from a single element.
                        // The buffer must contain at least one element though.
                        if num_elements != 0 {
                            num_elements = u64::MAX;
                        }
                    } else {
                        // If divisor is e.g. 2, we use only half the amount of data from the source
                        // buffer, so the number of instances that can be drawn is twice as large.
                        num_elements = num_elements.saturating_mul(divisor as u64);
                    }

                    instances_in_buffers = Some(if let Some(x) = instances_in_buffers {
                        min(x, num_elements)
                    } else {
                        num_elements
                    });
                }
            };
        }

        if let Some((first_vertex, vertex_count)) = vertices {
            let vertices_needed = first_vertex as u64 + vertex_count as u64;

            if let Some(vertices_in_buffers) = vertices_in_buffers {
                // VUID-vkCmdDraw-None-02721
                if vertices_needed > vertices_in_buffers {
                    return Err(PipelineExecutionError::VertexBufferVertexRangeOutOfBounds {
                        vertices_needed,
                        vertices_in_buffers,
                    });
                }
            }
        }

        if let Some((first_instance, instance_count)) = instances {
            let instances_needed = first_instance as u64 + instance_count as u64;

            if let Some(instances_in_buffers) = instances_in_buffers {
                // VUID-vkCmdDraw-None-02721
                if instances_needed > instances_in_buffers {
                    return Err(
                        PipelineExecutionError::VertexBufferInstanceRangeOutOfBounds {
                            instances_needed,
                            instances_in_buffers,
                        },
                    );
                }
            }

            let view_mask = match pipeline.render_pass() {
                PipelineRenderPassType::BeginRenderPass(subpass) => {
                    subpass.render_pass().views_used()
                }
                PipelineRenderPassType::BeginRendering(rendering_info) => rendering_info.view_mask,
            };

            if view_mask != 0 {
                let max = pipeline
                    .device()
                    .physical_device()
                    .properties()
                    .max_multiview_instance_index
                    .unwrap_or(0);

                let highest_instance = instances_needed.saturating_sub(1);

                // VUID-vkCmdDraw-maxMultiviewInstanceIndex-02688
                if highest_instance > max as u64 {
                    return Err(PipelineExecutionError::MaxMultiviewInstanceIndexExceeded {
                        highest_instance,
                        max,
                    });
                }
            }
        }

        Ok(())
    }
}

fn record_descriptor_sets_access(
    resources_usage_state: &mut ResourcesState,
    command_index: usize,
    command_name: &'static str,
    descriptor_sets_state: &HashMap<PipelineBindPoint, DescriptorSetState>,
    pipeline: &impl Pipeline,
) {
    let descriptor_sets_state = match descriptor_sets_state.get(&pipeline.bind_point()) {
        Some(x) => x,
        None => return,
    };

    for (&(set, binding), binding_reqs) in pipeline.descriptor_binding_requirements() {
        let descriptor_type = descriptor_sets_state.pipeline_layout.set_layouts()[set as usize]
            .bindings()[&binding]
            .descriptor_type;

        // TODO: Should input attachments be handled here or in attachment access?
        if descriptor_type == DescriptorType::InputAttachment {
            continue;
        }

        let use_iter = move |index: u32| {
            let (stages_read, stages_write) = [Some(index), None]
                .into_iter()
                .filter_map(|index| binding_reqs.descriptors.get(&index))
                .fold(
                    (ShaderStages::empty(), ShaderStages::empty()),
                    |(stages_read, stages_write), desc_reqs| {
                        (
                            stages_read | desc_reqs.memory_read,
                            stages_write | desc_reqs.memory_write,
                        )
                    },
                );
            let use_ref = ResourceUseRef {
                command_index,
                command_name,
                resource_in_command: ResourceInCommand::DescriptorSet {
                    set,
                    binding,
                    index,
                },
                secondary_use_ref: None,
            };
            let stage_access_iter = PipelineStageAccess::iter_descriptor_stages(
                descriptor_type,
                stages_read,
                stages_write,
            );
            (use_ref, stage_access_iter)
        };

        match descriptor_sets_state.descriptor_sets[&set]
            .resources()
            .binding(binding)
            .unwrap()
        {
            DescriptorBindingResources::None(_) => continue,
            DescriptorBindingResources::Buffer(elements) => {
                for (index, element) in elements.iter().enumerate() {
                    if let Some(buffer) = element {
                        let buffer_inner = buffer.inner();
                        let (use_ref, stage_access_iter) = use_iter(index as u32);

                        let mut range = 0..buffer.size(); // TODO:
                        range.start += buffer_inner.offset;
                        range.end += buffer_inner.offset;

                        for stage_access in stage_access_iter {
                            resources_usage_state.record_buffer_access(
                                &use_ref,
                                buffer_inner.buffer,
                                range.clone(),
                                stage_access,
                            );
                        }
                    }
                }
            }
            DescriptorBindingResources::BufferView(elements) => {
                for (index, element) in elements.iter().enumerate() {
                    if let Some(buffer_view) = element {
                        let buffer = buffer_view.buffer();
                        let buffer_inner = buffer.inner();
                        let (use_ref, stage_access_iter) = use_iter(index as u32);

                        let mut range = buffer_view.range();
                        range.start += buffer_inner.offset;
                        range.end += buffer_inner.offset;

                        for stage_access in stage_access_iter {
                            resources_usage_state.record_buffer_access(
                                &use_ref,
                                buffer_inner.buffer,
                                range.clone(),
                                stage_access,
                            );
                        }
                    }
                }
            }
            DescriptorBindingResources::ImageView(elements) => {
                for (index, element) in elements.iter().enumerate() {
                    if let Some(image_view) = element {
                        let image = image_view.image();
                        let image_inner = image.inner();
                        let layout = image
                            .descriptor_layouts()
                            .expect(
                                "descriptor_layouts must return Some when used in an image view",
                            )
                            .layout_for(descriptor_type);
                        let (use_ref, stage_access_iter) = use_iter(index as u32);

                        let mut subresource_range = image_view.subresource_range().clone();
                        subresource_range.array_layers.start += image_inner.first_layer;
                        subresource_range.array_layers.end += image_inner.first_layer;
                        subresource_range.mip_levels.start += image_inner.first_mipmap_level;
                        subresource_range.mip_levels.end += image_inner.first_mipmap_level;

                        for stage_access in stage_access_iter {
                            resources_usage_state.record_image_access(
                                &use_ref,
                                image_inner.image,
                                subresource_range.clone(),
                                stage_access,
                                layout,
                            );
                        }
                    }
                }
            }
            DescriptorBindingResources::ImageViewSampler(elements) => {
                for (index, element) in elements.iter().enumerate() {
                    if let Some((image_view, _)) = element {
                        let image = image_view.image();
                        let image_inner = image.inner();
                        let layout = image
                            .descriptor_layouts()
                            .expect(
                                "descriptor_layouts must return Some when used in an image view",
                            )
                            .layout_for(descriptor_type);
                        let (use_ref, stage_access_iter) = use_iter(index as u32);

                        let mut subresource_range = image_view.subresource_range().clone();
                        subresource_range.array_layers.start += image_inner.first_layer;
                        subresource_range.array_layers.end += image_inner.first_layer;
                        subresource_range.mip_levels.start += image_inner.first_mipmap_level;
                        subresource_range.mip_levels.end += image_inner.first_mipmap_level;

                        for stage_access in stage_access_iter {
                            resources_usage_state.record_image_access(
                                &use_ref,
                                image_inner.image,
                                subresource_range.clone(),
                                stage_access,
                                layout,
                            );
                        }
                    }
                }
            }
            DescriptorBindingResources::Sampler(_) => (),
        }
    }
}

fn record_vertex_buffers_access(
    resources_usage_state: &mut ResourcesState,
    command_index: usize,
    command_name: &'static str,
    vertex_buffers_state: &HashMap<u32, Arc<dyn BufferAccess>>,
    pipeline: &GraphicsPipeline,
) {
    for &binding in pipeline.vertex_input_state().bindings.keys() {
        let buffer = &vertex_buffers_state[&binding];
        let buffer_inner = buffer.inner();
        let use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::VertexBuffer { binding },
            secondary_use_ref: None,
        };

        let mut range = 0..buffer.size(); // TODO: take range from draw command
        range.start += buffer_inner.offset;
        range.end += buffer_inner.offset;
        resources_usage_state.record_buffer_access(
            &use_ref,
            buffer_inner.buffer,
            range,
            PipelineStageAccess::VertexAttributeInput_VertexAttributeRead,
        );
    }
}

fn record_index_buffer_access(
    resources_usage_state: &mut ResourcesState,
    command_index: usize,
    command_name: &'static str,
    index_buffer_state: &Option<(Arc<dyn BufferAccess>, IndexType)>,
) {
    let buffer = &index_buffer_state.as_ref().unwrap().0;
    let buffer_inner = buffer.inner();
    let use_ref = ResourceUseRef {
        command_index,
        command_name,
        resource_in_command: ResourceInCommand::IndexBuffer,
        secondary_use_ref: None,
    };

    let mut range = 0..buffer.size(); // TODO: take range from draw command
    range.start += buffer_inner.offset;
    range.end += buffer_inner.offset;
    resources_usage_state.record_buffer_access(
        &use_ref,
        buffer_inner.buffer,
        range,
        PipelineStageAccess::IndexInput_IndexRead,
    );
}

fn record_indirect_buffer_access(
    resources_usage_state: &mut ResourcesState,
    command_index: usize,
    command_name: &'static str,
    buffer: &Arc<dyn BufferAccess>,
) {
    let buffer_inner = buffer.inner();
    let use_ref = ResourceUseRef {
        command_index,
        command_name,
        resource_in_command: ResourceInCommand::IndirectBuffer,
        secondary_use_ref: None,
    };

    let mut range = 0..buffer.size(); // TODO: take range from draw command
    range.start += buffer_inner.offset;
    range.end += buffer_inner.offset;
    resources_usage_state.record_buffer_access(
        &use_ref,
        buffer_inner.buffer,
        range,
        PipelineStageAccess::DrawIndirect_IndirectCommandRead,
    );
}
