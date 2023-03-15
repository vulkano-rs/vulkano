// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{CommandBufferBuilder, RenderPassStateType, SetOrPush};
use crate::{
    buffer::{BufferContents, BufferUsage, Subbuffer},
    command_buffer::{allocator::CommandBufferAllocator, commands::bind_push::BindPushError},
    descriptor_set::{
        check_descriptor_write, layout::DescriptorType, DescriptorBindingResources,
        DescriptorSetResources, DescriptorSetWithOffsets, DescriptorSetsCollection,
        DescriptorWriteInfo, WriteDescriptorSet,
    },
    device::{DeviceOwned, QueueFlags},
    memory::is_aligned,
    pipeline::{
        graphics::{
            input_assembly::{Index, IndexType},
            render_pass::PipelineRenderPassType,
            vertex_input::VertexBuffersCollection,
        },
        ComputePipeline, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    },
    DeviceSize, RequiresOneOf, VulkanObject,
};
use smallvec::SmallVec;
use std::{cmp::min, mem::size_of_val, os::raw::c_void, sync::Arc};

impl<L, A> CommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Binds descriptor sets for future dispatch or draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support `pipeline_bind_point`.
    /// - Panics if the highest descriptor set slot being bound is not less than the number of sets
    ///   in `pipeline_layout`.
    /// - Panics if `self` and any element of `descriptor_sets` do not belong to the same device.
    pub fn bind_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        first_set: u32,
        descriptor_sets: impl DescriptorSetsCollection,
    ) -> &mut Self {
        let descriptor_sets = descriptor_sets.into_vec();
        self.validate_bind_descriptor_sets(
            pipeline_bind_point,
            &pipeline_layout,
            first_set,
            &descriptor_sets,
        )
        .unwrap();

        unsafe {
            self.bind_descriptor_sets_unchecked(
                pipeline_bind_point,
                pipeline_layout,
                first_set,
                descriptor_sets,
            )
        }
    }

    fn validate_bind_descriptor_sets(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[DescriptorSetWithOffsets],
    ) -> Result<(), BindPushError> {
        // VUID-vkCmdBindDescriptorSets-pipelineBindPoint-parameter
        pipeline_bind_point.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBindDescriptorSets-commandBuffer-cmdpool
        // VUID-vkCmdBindDescriptorSets-pipelineBindPoint-00361
        match pipeline_bind_point {
            PipelineBindPoint::Compute => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(BindPushError::NotSupportedByQueueFamily);
                }
            }
            PipelineBindPoint::Graphics => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(BindPushError::NotSupportedByQueueFamily);
                }
            }
        }

        // VUID-vkCmdBindDescriptorSets-firstSet-00360
        if first_set + descriptor_sets.len() as u32 > pipeline_layout.set_layouts().len() as u32 {
            return Err(BindPushError::DescriptorSetOutOfRange {
                set_num: first_set + descriptor_sets.len() as u32,
                pipeline_layout_set_count: pipeline_layout.set_layouts().len() as u32,
            });
        }

        let properties = self.device().physical_device().properties();
        let uniform_alignment = properties.min_uniform_buffer_offset_alignment;
        let storage_alignment = properties.min_storage_buffer_offset_alignment;

        for (i, set) in descriptor_sets.iter().enumerate() {
            let set_num = first_set + i as u32;
            let (set, dynamic_offsets) = set.as_ref();

            // VUID-vkCmdBindDescriptorSets-commonparent
            assert_eq!(self.device(), set.device());

            let set_layout = set.layout();
            let pipeline_set_layout = &pipeline_layout.set_layouts()[set_num as usize];

            // VUID-vkCmdBindDescriptorSets-pDescriptorSets-00358
            if !pipeline_set_layout.is_compatible_with(set_layout) {
                return Err(BindPushError::DescriptorSetNotCompatible { set_num });
            }

            let mut dynamic_offsets_remaining = dynamic_offsets;
            let mut required_dynamic_offset_count = 0;

            for (&binding_num, binding) in set_layout.bindings() {
                let required_alignment = match binding.descriptor_type {
                    DescriptorType::UniformBufferDynamic => uniform_alignment,
                    DescriptorType::StorageBufferDynamic => storage_alignment,
                    _ => continue,
                };

                let count = if binding.variable_descriptor_count {
                    set.variable_descriptor_count()
                } else {
                    binding.descriptor_count
                } as usize;

                required_dynamic_offset_count += count;

                if !dynamic_offsets_remaining.is_empty() {
                    let split_index = min(count, dynamic_offsets_remaining.len());
                    let dynamic_offsets = &dynamic_offsets_remaining[..split_index];
                    dynamic_offsets_remaining = &dynamic_offsets_remaining[split_index..];

                    let elements = match set.resources().binding(binding_num) {
                        Some(DescriptorBindingResources::Buffer(elements)) => elements.as_slice(),
                        _ => unreachable!(),
                    };

                    for (index, (&offset, element)) in
                        dynamic_offsets.iter().zip(elements).enumerate()
                    {
                        // VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01971
                        // VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01972
                        if !is_aligned(offset as DeviceSize, required_alignment) {
                            return Err(BindPushError::DynamicOffsetNotAligned {
                                set_num,
                                binding_num,
                                index: index as u32,
                                offset,
                                required_alignment,
                            });
                        }

                        if let Some((buffer, range)) = element {
                            // VUID-vkCmdBindDescriptorSets-pDescriptorSets-01979
                            if offset as DeviceSize + range.end > buffer.size() {
                                return Err(BindPushError::DynamicOffsetOutOfBufferBounds {
                                    set_num,
                                    binding_num,
                                    index: index as u32,
                                    offset,
                                    range_end: range.end,
                                    buffer_size: buffer.size(),
                                });
                            }
                        }
                    }
                }
            }

            // VUID-vkCmdBindDescriptorSets-dynamicOffsetCount-00359
            if dynamic_offsets.len() != required_dynamic_offset_count {
                return Err(BindPushError::DynamicOffsetCountMismatch {
                    set_num,
                    provided_count: dynamic_offsets.len(),
                    required_count: required_dynamic_offset_count,
                });
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_descriptor_sets_unchecked(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        first_set: u32,
        descriptor_sets: impl DescriptorSetsCollection,
    ) -> &mut Self {
        let descriptor_sets_with_offsets = descriptor_sets.into_vec();

        if descriptor_sets_with_offsets.is_empty() {
            return self;
        }

        let descriptor_sets: SmallVec<[_; 12]> = descriptor_sets_with_offsets
            .iter()
            .map(|x| x.as_ref().0.inner().handle())
            .collect();
        let dynamic_offsets: SmallVec<[_; 32]> = descriptor_sets_with_offsets
            .iter()
            .flat_map(|x| x.as_ref().1.iter().copied())
            .collect();

        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_descriptor_sets)(
            self.handle(),
            pipeline_bind_point.into(),
            pipeline_layout.handle(),
            first_set,
            descriptor_sets.len() as u32,
            descriptor_sets.as_ptr(),
            dynamic_offsets.len() as u32,
            dynamic_offsets.as_ptr(),
        );

        let state = self.builder_state.invalidate_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout.clone(),
            first_set,
            descriptor_sets_with_offsets.len() as u32,
        );

        self.resources
            .reserve(descriptor_sets_with_offsets.len() + 1);

        for (set_num, descriptor_set_with_offsets) in
            descriptor_sets_with_offsets.into_iter().enumerate()
        {
            let descriptor_set = descriptor_set_with_offsets.as_ref().0.clone();
            state.descriptor_sets.insert(
                first_set + set_num as u32,
                SetOrPush::Set(descriptor_set_with_offsets),
            );
            self.resources.push(Box::new(descriptor_set));
        }

        self.resources.push(Box::new(pipeline_layout));

        self.next_command_index += 1;
        self
    }

    /// Binds an index buffer for future indexed draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if `self` and `index_buffer` do not belong to the same device.
    /// - Panics if `index_buffer` does not have the [`BufferUsage::INDEX_BUFFER`] usage enabled.
    /// - If the index buffer contains `u8` indices, panics if the [`index_type_uint8`] feature is
    ///   not enabled on the device.
    ///
    /// [`BufferUsage::INDEX_BUFFER`]: crate::buffer::BufferUsage::INDEX_BUFFER
    /// [`index_type_uint8`]: crate::device::Features::index_type_uint8
    pub fn bind_index_buffer<I: Index>(&mut self, index_buffer: Subbuffer<[I]>) -> &mut Self {
        self.validate_bind_index_buffer(index_buffer.as_bytes(), I::ty())
            .unwrap();

        unsafe { self.bind_index_buffer_unchecked(index_buffer.into_bytes(), I::ty()) }
    }

    fn validate_bind_index_buffer(
        &self,
        index_buffer: &Subbuffer<[u8]>,
        index_type: IndexType,
    ) -> Result<(), BindPushError> {
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBindIndexBuffer-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(BindPushError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdBindIndexBuffer-commonparent
        assert_eq!(self.device(), index_buffer.device());

        // VUID-vkCmdBindIndexBuffer-buffer-00433
        if !index_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::INDEX_BUFFER)
        {
            return Err(BindPushError::IndexBufferMissingUsage);
        }

        // VUID-vkCmdBindIndexBuffer-indexType-02765
        if index_type == IndexType::U8 && !self.device().enabled_features().index_type_uint8 {
            return Err(BindPushError::RequirementNotMet {
                required_for: "`index_type` is `IndexType::U8`",
                requires_one_of: RequiresOneOf {
                    features: &["index_type_uint8"],
                    ..Default::default()
                },
            });
        }

        // TODO:
        // VUID-vkCmdBindIndexBuffer-offset-00432

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_index_buffer_unchecked(
        &mut self,
        buffer: Subbuffer<[u8]>,
        index_type: IndexType,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_index_buffer)(
            self.handle(),
            buffer.buffer().handle(),
            buffer.offset(),
            index_type.into(),
        );

        self.builder_state.index_buffer = Some((buffer.clone(), index_type));
        self.resources.push(Box::new(buffer));

        self.next_command_index += 1;
        self
    }

    /// Binds a compute pipeline for future dispatch calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support compute operations.
    /// - Panics if `self` and `pipeline` do not belong to the same device.
    pub fn bind_pipeline_compute(&mut self, pipeline: Arc<ComputePipeline>) -> &mut Self {
        self.validate_bind_pipeline_compute(&pipeline).unwrap();

        unsafe { self.bind_pipeline_compute_unchecked(pipeline) }
    }

    fn validate_bind_pipeline_compute(
        &self,
        pipeline: &ComputePipeline,
    ) -> Result<(), BindPushError> {
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBindPipeline-pipelineBindPoint-00777
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(BindPushError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdBindPipeline-commonparent
        assert_eq!(self.device(), pipeline.device());

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_pipeline_compute_unchecked(
        &mut self,
        pipeline: Arc<ComputePipeline>,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_pipeline)(
            self.handle(),
            ash::vk::PipelineBindPoint::COMPUTE,
            pipeline.handle(),
        );

        self.builder_state.pipeline_compute = Some(pipeline.clone());
        self.resources.push(Box::new(pipeline));

        self.next_command_index += 1;
        self
    }

    /// Binds a graphics pipeline for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if `self` and `pipeline` do not belong to the same device.
    pub fn bind_pipeline_graphics(&mut self, pipeline: Arc<GraphicsPipeline>) -> &mut Self {
        self.validate_bind_pipeline_graphics(&pipeline).unwrap();

        unsafe { self.bind_pipeline_graphics_unchecked(pipeline) }
    }

    fn validate_bind_pipeline_graphics(
        &self,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), BindPushError> {
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBindPipeline-pipelineBindPoint-00778
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(BindPushError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdBindPipeline-commonparent
        assert_eq!(self.device(), pipeline.device());

        if let Some(last_pipeline) =
            self.builder_state
                .render_pass
                .as_ref()
                .and_then(|render_pass_state| match &render_pass_state.render_pass {
                    RenderPassStateType::BeginRendering(state) if state.pipeline_used => {
                        self.builder_state.pipeline_graphics.as_ref()
                    }
                    _ => None,
                })
        {
            if let (
                PipelineRenderPassType::BeginRendering(pipeline_rendering_info),
                PipelineRenderPassType::BeginRendering(last_pipeline_rendering_info),
            ) = (pipeline.render_pass(), last_pipeline.render_pass())
            {
                // VUID-vkCmdBindPipeline-pipeline-06195
                // VUID-vkCmdBindPipeline-pipeline-06196
                if pipeline_rendering_info.color_attachment_formats
                    != last_pipeline_rendering_info.color_attachment_formats
                {
                    return Err(BindPushError::PreviousPipelineColorAttachmentFormatMismatch);
                }

                // VUID-vkCmdBindPipeline-pipeline-06197
                if pipeline_rendering_info.depth_attachment_format
                    != last_pipeline_rendering_info.depth_attachment_format
                {
                    return Err(BindPushError::PreviousPipelineDepthAttachmentFormatMismatch);
                }

                // VUID-vkCmdBindPipeline-pipeline-06194
                if pipeline_rendering_info.stencil_attachment_format
                    != last_pipeline_rendering_info.stencil_attachment_format
                {
                    return Err(BindPushError::PreviousPipelineStencilAttachmentFormatMismatch);
                }
            }
        }

        // VUID-vkCmdBindPipeline-pipeline-00781
        // TODO:

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_pipeline_graphics_unchecked(
        &mut self,
        pipeline: Arc<GraphicsPipeline>,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_pipeline)(
            self.handle(),
            ash::vk::PipelineBindPoint::GRAPHICS,
            pipeline.handle(),
        );

        // Reset any states that are fixed in the new pipeline. The pipeline bind command will
        // overwrite these states.
        self.builder_state.reset_dynamic_states(
            pipeline
                .dynamic_states()
                .filter(|(_, d)| !d) // not dynamic
                .map(|(s, _)| s),
        );
        self.builder_state.pipeline_graphics = Some(pipeline.clone());
        self.resources.push(Box::new(pipeline));

        self.next_command_index += 1;
        self
    }

    /// Binds vertex buffers for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the highest vertex buffer binding being bound is greater than the
    ///   [`max_vertex_input_bindings`] device property.
    /// - Panics if `self` and any element of `vertex_buffers` do not belong to the same device.
    /// - Panics if any element of `vertex_buffers` does not have the
    ///   [`BufferUsage::VERTEX_BUFFER`] usage enabled.
    ///
    /// [`max_vertex_input_bindings`]: crate::device::Properties::max_vertex_input_bindings
    /// [`BufferUsage::VERTEX_BUFFER`]: crate::buffer::BufferUsage::VERTEX_BUFFER
    pub fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        vertex_buffers: impl VertexBuffersCollection,
    ) -> &mut Self {
        let vertex_buffers = vertex_buffers.into_vec();
        self.validate_bind_vertex_buffers(first_binding, &vertex_buffers)
            .unwrap();

        unsafe { self.bind_vertex_buffers_unchecked(first_binding, vertex_buffers) }
    }

    fn validate_bind_vertex_buffers(
        &self,
        first_binding: u32,
        vertex_buffers: &[Subbuffer<[u8]>],
    ) -> Result<(), BindPushError> {
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBindVertexBuffers-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(BindPushError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdBindVertexBuffers-firstBinding-00624
        // VUID-vkCmdBindVertexBuffers-firstBinding-00625
        if first_binding + vertex_buffers.len() as u32
            > self
                .device()
                .physical_device()
                .properties()
                .max_vertex_input_bindings
        {
            return Err(BindPushError::MaxVertexInputBindingsExceeded {
                _binding_count: first_binding + vertex_buffers.len() as u32,
                _max: self
                    .device()
                    .physical_device()
                    .properties()
                    .max_vertex_input_bindings,
            });
        }

        for buffer in vertex_buffers {
            // VUID-vkCmdBindVertexBuffers-commonparent
            assert_eq!(self.device(), buffer.device());

            // VUID-vkCmdBindVertexBuffers-pBuffers-00627
            if !buffer
                .buffer()
                .usage()
                .intersects(BufferUsage::VERTEX_BUFFER)
            {
                return Err(BindPushError::VertexBufferMissingUsage);
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_vertex_buffers_unchecked(
        &mut self,
        first_binding: u32,
        buffers: impl VertexBuffersCollection,
    ) -> &mut Self {
        let buffers = buffers.into_vec();

        if buffers.is_empty() {
            return self;
        }

        let (buffers_vk, offsets_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) = buffers
            .iter()
            .map(|buffer| (buffer.buffer().handle(), buffer.offset()))
            .unzip();

        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_vertex_buffers)(
            self.handle(),
            first_binding,
            buffers_vk.len() as u32,
            buffers_vk.as_ptr(),
            offsets_vk.as_ptr(),
        );

        self.resources.reserve(buffers.len());

        for (i, buffer) in buffers.into_iter().enumerate() {
            self.builder_state
                .vertex_buffers
                .insert(first_binding + i as u32, buffer.clone());
            self.resources.push(Box::new(buffer));
        }

        self.next_command_index += 1;
        self
    }

    /// Sets push constants for future dispatch or draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if `offset` is not a multiple of 4.
    /// - Panics if the size of `push_constants` is not a multiple of 4.
    /// - Panics if any of the bytes in `push_constants` do not fall within any of the pipeline
    ///   layout's push constant ranges.
    pub fn push_constants(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        push_constants: &(impl BufferContents + ?Sized),
    ) -> &mut Self {
        self.validate_push_constants(&pipeline_layout, offset, size_of_val(push_constants) as u32)
            .unwrap();

        unsafe { self.push_constants_unchecked(pipeline_layout, offset, push_constants) }
    }

    fn validate_push_constants(
        &self,
        pipeline_layout: &PipelineLayout,
        offset: u32,
        data_size: u32,
    ) -> Result<(), BindPushError> {
        if offset % 4 != 0 {
            return Err(BindPushError::PushConstantsOffsetNotAligned);
        }

        if data_size % 4 != 0 {
            return Err(BindPushError::PushConstantsSizeNotAligned);
        }

        let mut current_offset = offset;
        let mut remaining_size = data_size;

        for range in pipeline_layout
            .push_constant_ranges_disjoint()
            .iter()
            .skip_while(|range| range.offset + range.size <= offset)
        {
            // there is a gap between ranges, but the passed push_constants contains
            // some bytes in this gap, exit the loop and report error
            if range.offset > current_offset {
                break;
            }

            // push the minimum of the whole remaining data, and the part until the end of this range
            let push_size = remaining_size.min(range.offset + range.size - current_offset);
            current_offset += push_size;
            remaining_size -= push_size;

            if remaining_size == 0 {
                break;
            }
        }

        if remaining_size != 0 {
            return Err(BindPushError::PushConstantsDataOutOfRange {
                offset: current_offset,
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn push_constants_unchecked(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        push_constants: &(impl BufferContents + ?Sized),
    ) -> &mut Self {
        let mut current_offset = offset;
        let mut remaining_size = size_of_val(push_constants) as u32;

        let fns = self.device().fns();

        for range in pipeline_layout
            .push_constant_ranges_disjoint()
            .iter()
            .skip_while(|range| range.offset + range.size <= offset)
        {
            // there is a gap between ranges, but the passed push_constants contains
            // some bytes in this gap, exit the loop and report error
            if range.offset > current_offset {
                break;
            }

            // push the minimum of the whole remaining data, and the part until the end of this range
            let push_size = min(remaining_size, range.offset + range.size - current_offset);
            let data_offset = (current_offset - offset) as usize;

            (fns.v1_0.cmd_push_constants)(
                self.handle(),
                pipeline_layout.handle(),
                range.stages.into(),
                current_offset,
                push_size,
                (push_constants as *const _ as *const c_void).add(data_offset),
            );

            current_offset += push_size;
            remaining_size -= push_size;

            if remaining_size == 0 {
                break;
            }
        }

        debug_assert!(remaining_size == 0);

        // TODO: Push constant invalidations.
        // The Vulkan spec currently is unclear about this, so Vulkano currently just marks
        // push constants as set, and never unsets them. See:
        // https://github.com/KhronosGroup/Vulkan-Docs/issues/1485
        // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2711
        self.builder_state
            .push_constants
            .insert(offset..offset + size_of_val(push_constants) as u32);
        self.builder_state.push_constants_pipeline_layout = Some(pipeline_layout.clone());
        self.resources.push(Box::new(pipeline_layout));

        self.next_command_index += 1;
        self
    }

    /// Pushes descriptor data directly into the command buffer for future dispatch or draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support `pipeline_bind_point`.
    /// - Panics if the [`khr_push_descriptor`] extension is not enabled on the device.
    /// - Panics if `set_num` is not less than the number of sets in `pipeline_layout`.
    /// - Panics if an element of `descriptor_writes` is not compatible with `pipeline_layout`.
    ///
    /// [`khr_push_descriptor`]: crate::device::DeviceExtensions::khr_push_descriptor
    pub fn push_descriptor_set(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        set_num: u32,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> &mut Self {
        let descriptor_writes: SmallVec<[_; 8]> = descriptor_writes.into_iter().collect();
        self.validate_push_descriptor_set(
            pipeline_bind_point,
            &pipeline_layout,
            set_num,
            &descriptor_writes,
        )
        .unwrap();

        unsafe {
            self.push_descriptor_set_unchecked(
                pipeline_bind_point,
                pipeline_layout,
                set_num,
                descriptor_writes,
            )
        }
    }

    fn validate_push_descriptor_set(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        set_num: u32,
        descriptor_writes: &[WriteDescriptorSet],
    ) -> Result<(), BindPushError> {
        if !self.device().enabled_extensions().khr_push_descriptor {
            return Err(BindPushError::RequirementNotMet {
                required_for: "`CommandBufferBuilder::push_descriptor_set`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_push_descriptor"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdPushDescriptorSetKHR-pipelineBindPoint-parameter
        pipeline_bind_point.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdPushDescriptorSetKHR-commandBuffer-cmdpool
        // VUID-vkCmdPushDescriptorSetKHR-pipelineBindPoint-00363
        match pipeline_bind_point {
            PipelineBindPoint::Compute => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(BindPushError::NotSupportedByQueueFamily);
                }
            }
            PipelineBindPoint::Graphics => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(BindPushError::NotSupportedByQueueFamily);
                }
            }
        }

        // VUID-vkCmdPushDescriptorSetKHR-commonparent
        assert_eq!(self.device(), pipeline_layout.device());

        // VUID-vkCmdPushDescriptorSetKHR-set-00364
        if set_num as usize > pipeline_layout.set_layouts().len() {
            return Err(BindPushError::DescriptorSetOutOfRange {
                set_num,
                pipeline_layout_set_count: pipeline_layout.set_layouts().len() as u32,
            });
        }

        let descriptor_set_layout = &pipeline_layout.set_layouts()[set_num as usize];

        // VUID-vkCmdPushDescriptorSetKHR-set-00365
        if !descriptor_set_layout.push_descriptor() {
            return Err(BindPushError::DescriptorSetNotPush { set_num });
        }

        for write in descriptor_writes {
            check_descriptor_write(write, descriptor_set_layout, 0)?;
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn push_descriptor_set_unchecked(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        set_num: u32,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> &mut Self {
        let descriptor_writes: SmallVec<[WriteDescriptorSet; 8]> =
            descriptor_writes.into_iter().collect();

        debug_assert!(self.device().enabled_extensions().khr_push_descriptor);

        let (infos, mut writes): (SmallVec<[_; 8]>, SmallVec<[_; 8]>) = descriptor_writes
            .iter()
            .map(|write| {
                let binding =
                    &pipeline_layout.set_layouts()[set_num as usize].bindings()[&write.binding()];

                (
                    write.to_vulkan_info(binding.descriptor_type),
                    write.to_vulkan(ash::vk::DescriptorSet::null(), binding.descriptor_type),
                )
            })
            .unzip();

        if writes.is_empty() {
            return self;
        }

        // Set the info pointers separately.
        for (info, write) in infos.iter().zip(writes.iter_mut()) {
            match info {
                DescriptorWriteInfo::Image(info) => {
                    write.descriptor_count = info.len() as u32;
                    write.p_image_info = info.as_ptr();
                }
                DescriptorWriteInfo::Buffer(info) => {
                    write.descriptor_count = info.len() as u32;
                    write.p_buffer_info = info.as_ptr();
                }
                DescriptorWriteInfo::BufferView(info) => {
                    write.descriptor_count = info.len() as u32;
                    write.p_texel_buffer_view = info.as_ptr();
                }
            }

            debug_assert!(write.descriptor_count != 0);
        }

        let fns = self.device().fns();
        (fns.khr_push_descriptor.cmd_push_descriptor_set_khr)(
            self.handle(),
            pipeline_bind_point.into(),
            pipeline_layout.handle(),
            set_num,
            writes.len() as u32,
            writes.as_ptr(),
        );

        let state = self.builder_state.invalidate_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout.clone(),
            set_num,
            1,
        );
        let descriptor_set_layout = state.pipeline_layout.set_layouts()[set_num as usize].as_ref();
        debug_assert!(descriptor_set_layout.push_descriptor());

        let set_resources = match state.descriptor_sets.entry(set_num).or_insert_with(|| {
            SetOrPush::Push(DescriptorSetResources::new(descriptor_set_layout, 0))
        }) {
            SetOrPush::Push(set_resources) => set_resources,
            _ => unreachable!(),
        };

        for write in &descriptor_writes {
            set_resources.update(write);
        }

        self.resources.push(Box::new(pipeline_layout));

        self.next_command_index += 1;
        self
    }
}
