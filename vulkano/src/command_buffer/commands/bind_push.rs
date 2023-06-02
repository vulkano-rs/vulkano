// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::{BufferContents, BufferUsage, IndexBuffer, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator,
        auto::{RenderPassStateType, SetOrPush},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder,
    },
    descriptor_set::{
        layout::DescriptorType, set_descriptor_write_image_layouts, validate_descriptor_write,
        DescriptorBindingResources, DescriptorBufferInfo, DescriptorSetResources,
        DescriptorSetUpdateError, DescriptorSetWithOffsets, DescriptorSetsCollection,
        DescriptorWriteInfo, WriteDescriptorSet,
    },
    device::{DeviceOwned, QueueFlags},
    memory::{is_aligned, DeviceAlignment},
    pipeline::{
        graphics::{subpass::PipelineSubpassType, vertex_input::VertexBuffersCollection},
        ComputePipeline, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    },
    DeviceSize, RequirementNotMet, RequiresOneOf, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    cmp::min,
    error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::size_of,
    slice,
    sync::Arc,
};

/// # Commands to bind or push state for pipeline execution commands.
///
/// These commands require a queue with a pipeline type that uses the given state.
impl<L, A> AutoCommandBufferBuilder<L, A>
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
            );
        }

        self
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

                        if let Some(buffer_info) = element {
                            let DescriptorBufferInfo { buffer, range } = buffer_info;

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
        let descriptor_sets = descriptor_sets.into_vec();

        if descriptor_sets.is_empty() {
            return self;
        }

        let state = self.builder_state.invalidate_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout.clone(),
            first_set,
            descriptor_sets.len() as u32,
        );

        for (set_num, set) in descriptor_sets.iter().enumerate() {
            state
                .descriptor_sets
                .insert(first_set + set_num as u32, SetOrPush::Set(set.clone()));
        }

        self.add_command(
            "bind_descriptor_sets",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.bind_descriptor_sets(
                    pipeline_bind_point,
                    &pipeline_layout,
                    first_set,
                    &descriptor_sets,
                );
            },
        );

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
    pub fn bind_index_buffer(&mut self, index_buffer: impl Into<IndexBuffer>) -> &mut Self {
        let index_buffer = index_buffer.into();
        self.validate_bind_index_buffer(&index_buffer).unwrap();

        unsafe {
            self.bind_index_buffer_unchecked(index_buffer);
        }

        self
    }

    fn validate_bind_index_buffer(&self, index_buffer: &IndexBuffer) -> Result<(), BindPushError> {
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBindIndexBuffer-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(BindPushError::NotSupportedByQueueFamily);
        }

        let index_buffer_bytes = index_buffer.as_bytes();

        // VUID-vkCmdBindIndexBuffer-commonparent
        assert_eq!(self.device(), index_buffer_bytes.device());

        // VUID-vkCmdBindIndexBuffer-buffer-00433
        if !index_buffer_bytes
            .buffer()
            .usage()
            .intersects(BufferUsage::INDEX_BUFFER)
        {
            return Err(BindPushError::IndexBufferMissingUsage);
        }

        // VUID-vkCmdBindIndexBuffer-indexType-02765
        if matches!(index_buffer, IndexBuffer::U8(_))
            && !self.device().enabled_features().index_type_uint8
        {
            return Err(BindPushError::RequirementNotMet {
                required_for: "`index_buffer` is `IndexBuffer::U8`",
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
        index_buffer: impl Into<IndexBuffer>,
    ) -> &mut Self {
        let index_buffer = index_buffer.into();
        self.builder_state.index_buffer = Some(index_buffer.clone());
        self.add_command(
            "bind_index_buffer",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.bind_index_buffer(&index_buffer);
            },
        );

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

        unsafe {
            self.bind_pipeline_compute_unchecked(pipeline);
        }

        self
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
        self.builder_state.pipeline_compute = Some(pipeline.clone());
        self.add_command(
            "bind_pipeline_compute",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.bind_pipeline_compute(&pipeline);
            },
        );

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

        unsafe {
            self.bind_pipeline_graphics_unchecked(pipeline);
        }

        self
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
                PipelineSubpassType::BeginRendering(pipeline_rendering_info),
                PipelineSubpassType::BeginRendering(last_pipeline_rendering_info),
            ) = (pipeline.subpass(), last_pipeline.subpass())
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
        // Reset any states that are fixed in the new pipeline. The pipeline bind command will
        // overwrite these states.
        self.builder_state.reset_dynamic_states(
            pipeline
                .dynamic_states()
                .filter(|(_, d)| !d) // not dynamic
                .map(|(s, _)| s),
        );
        self.builder_state.pipeline_graphics = Some(pipeline.clone());
        self.add_command(
            "bind_pipeline_graphics",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.bind_pipeline_graphics(&pipeline);
            },
        );

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

        unsafe {
            self.bind_vertex_buffers_unchecked(first_binding, vertex_buffers);
        }

        self
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
        vertex_buffers: impl VertexBuffersCollection,
    ) -> &mut Self {
        let vertex_buffers = vertex_buffers.into_vec();

        for (i, buffer) in vertex_buffers.iter().enumerate() {
            self.builder_state
                .vertex_buffers
                .insert(first_binding + i as u32, buffer.clone());
        }

        self.add_command(
            "bind_vertex_buffers",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.bind_vertex_buffers(first_binding, &vertex_buffers);
            },
        );

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
    pub fn push_constants<Pc>(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        push_constants: Pc,
    ) -> &mut Self
    where
        Pc: BufferContents,
    {
        let size = size_of::<Pc>() as u32;

        if size == 0 {
            return self;
        }

        self.validate_push_constants(&pipeline_layout, offset, &push_constants)
            .unwrap();

        unsafe {
            self.push_constants_unchecked(pipeline_layout, offset, push_constants);
        }

        self
    }

    fn validate_push_constants<Pc: BufferContents>(
        &self,
        pipeline_layout: &PipelineLayout,
        offset: u32,
        push_constants: &Pc,
    ) -> Result<(), BindPushError> {
        let size = size_of::<Pc>() as u32;

        // SAFETY: `&push_constants` is a valid pointer, and the size of the struct is `size`,
        //         thus, getting a slice of the whole struct is safe if its not modified.
        let push_constants = unsafe {
            slice::from_raw_parts(push_constants as *const Pc as *const u8, size as usize)
        };

        if offset % 4 != 0 {
            return Err(BindPushError::PushConstantsOffsetNotAligned);
        }

        if push_constants.len() % 4 != 0 {
            return Err(BindPushError::PushConstantsSizeNotAligned);
        }

        let mut current_offset = offset;
        let mut remaining_size = push_constants.len() as u32;
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
    pub unsafe fn push_constants_unchecked<Pc>(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        push_constants: Pc,
    ) -> &mut Self
    where
        Pc: BufferContents,
    {
        // TODO: Push constant invalidations.
        // The Vulkan spec currently is unclear about this, so Vulkano currently just marks
        // push constants as set, and never unsets them. See:
        // https://github.com/KhronosGroup/Vulkan-Docs/issues/1485
        // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2711
        self.builder_state
            .push_constants
            .insert(offset..offset + size_of::<Pc>() as u32);
        self.builder_state.push_constants_pipeline_layout = Some(pipeline_layout.clone());

        self.add_command(
            "push_constants",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.push_constants(&pipeline_layout, offset, &push_constants);
            },
        );

        self
    }

    /// Pushes descriptor data directly into the command buffer for future dispatch or draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support `pipeline_bind_point`.
    /// - Panics if the
    ///   [`khr_push_descriptor`](crate::device::DeviceExtensions::khr_push_descriptor)
    ///   extension is not enabled on the device.
    /// - Panics if `set_num` is not less than the number of sets in `pipeline_layout`.
    /// - Panics if an element of `descriptor_writes` is not compatible with `pipeline_layout`.
    pub fn push_descriptor_set(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        set_num: u32,
        mut descriptor_writes: SmallVec<[WriteDescriptorSet; 8]>,
    ) -> &mut Self {
        // Set the image layouts
        if let Some(set_layout) = pipeline_layout.set_layouts().get(set_num as usize) {
            for write in &mut descriptor_writes {
                set_descriptor_write_image_layouts(write, set_layout);
            }
        }

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
            );
        }

        self
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
                required_for: "`AutoCommandBufferBuilder::push_descriptor_set`",
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
            validate_descriptor_write(write, descriptor_set_layout, 0)?;
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn push_descriptor_set_unchecked(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        set_num: u32,
        mut descriptor_writes: SmallVec<[WriteDescriptorSet; 8]>,
    ) -> &mut Self {
        // Set the image layouts
        if let Some(set_layout) = pipeline_layout.set_layouts().get(set_num as usize) {
            for write in &mut descriptor_writes {
                set_descriptor_write_image_layouts(write, set_layout);
            }
        }

        let state = self.builder_state.invalidate_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout.clone(),
            set_num,
            1,
        );
        let layout = state.pipeline_layout.set_layouts()[set_num as usize].as_ref();
        debug_assert!(layout.push_descriptor());

        let set_resources = match state
            .descriptor_sets
            .entry(set_num)
            .or_insert_with(|| SetOrPush::Push(DescriptorSetResources::new(layout, 0)))
        {
            SetOrPush::Push(set_resources) => set_resources,
            _ => unreachable!(),
        };

        for write in &descriptor_writes {
            set_resources.update(write);
        }

        self.add_command(
            "push_descriptor_set",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.push_descriptor_set(
                    pipeline_bind_point,
                    &pipeline_layout,
                    set_num,
                    &descriptor_writes,
                );
            },
        );

        self
    }
}

impl<A> UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    /// Calls `vkCmdBindDescriptorSets` on the builder.
    ///
    /// Does nothing if the list of descriptor sets is empty, as it would be a no-op and isn't a
    /// valid usage of the command anyway.
    #[inline]
    pub unsafe fn bind_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[DescriptorSetWithOffsets],
    ) -> &mut Self {
        if descriptor_sets.is_empty() {
            return self;
        }

        let descriptor_sets_vk: SmallVec<[_; 12]> = descriptor_sets
            .iter()
            .map(|x| x.as_ref().0.inner().handle())
            .collect();
        let dynamic_offsets_vk: SmallVec<[_; 32]> = descriptor_sets
            .iter()
            .flat_map(|x| x.as_ref().1.iter().copied())
            .collect();

        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_descriptor_sets)(
            self.handle(),
            pipeline_bind_point.into(),
            pipeline_layout.handle(),
            first_set,
            descriptor_sets_vk.len() as u32,
            descriptor_sets_vk.as_ptr(),
            dynamic_offsets_vk.len() as u32,
            dynamic_offsets_vk.as_ptr(),
        );

        self
    }

    /// Calls `vkCmdBindIndexBuffer` on the builder.
    #[inline]
    pub unsafe fn bind_index_buffer(&mut self, index_buffer: &IndexBuffer) -> &mut Self {
        let index_buffer_bytes = index_buffer.as_bytes();

        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_index_buffer)(
            self.handle(),
            index_buffer_bytes.buffer().handle(),
            index_buffer_bytes.offset(),
            index_buffer.index_type().into(),
        );

        self
    }

    /// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_compute(&mut self, pipeline: &ComputePipeline) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_pipeline)(
            self.handle(),
            ash::vk::PipelineBindPoint::COMPUTE,
            pipeline.handle(),
        );

        self
    }

    /// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_graphics(&mut self, pipeline: &GraphicsPipeline) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_pipeline)(
            self.handle(),
            ash::vk::PipelineBindPoint::GRAPHICS,
            pipeline.handle(),
        );

        self
    }

    /// Calls `vkCmdBindVertexBuffers` on the builder.
    ///
    /// Does nothing if the list of buffers is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    // TODO: vkCmdBindVertexBuffers2EXT
    #[inline]
    pub unsafe fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        vertex_buffers: &[Subbuffer<[u8]>],
    ) -> &mut Self {
        if vertex_buffers.is_empty() {
            return self;
        }

        let (buffers_vk, offsets_vk): (SmallVec<[_; 2]>, SmallVec<[_; 2]>) = vertex_buffers
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

        self
    }

    /// Calls `vkCmdPushConstants` on the builder.
    pub unsafe fn push_constants<Pc>(
        &mut self,
        pipeline_layout: &PipelineLayout,
        offset: u32,
        push_constants: &Pc,
    ) -> &mut Self
    where
        Pc: BufferContents,
    {
        let size = size_of::<Pc>() as u32;

        if size == 0 {
            return self;
        }

        // SAFETY: `&push_constants` is a valid pointer, and the size of the struct is `size`,
        //         thus, getting a slice of the whole struct is safe if its not modified.
        let push_constants = unsafe {
            slice::from_raw_parts(push_constants as *const Pc as *const u8, size as usize)
        };

        let fns = self.device().fns();
        let mut current_offset = offset;
        let mut remaining_size = size;
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
            let data_offset = (current_offset - offset) as usize;
            let slice = &push_constants[data_offset..(data_offset + push_size as usize)];

            (fns.v1_0.cmd_push_constants)(
                self.handle(),
                pipeline_layout.handle(),
                range.stages.into(),
                current_offset,
                push_size,
                slice.as_ptr() as _,
            );

            current_offset += push_size;
            remaining_size -= push_size;

            if remaining_size == 0 {
                break;
            }
        }

        debug_assert!(remaining_size == 0);

        self
    }

    /// Calls `vkCmdPushDescriptorSetKHR` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn push_descriptor_set(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        set_num: u32,
        descriptor_writes: &[WriteDescriptorSet],
    ) -> &mut Self {
        if descriptor_writes.is_empty() {
            return self;
        }

        let set_layout = &pipeline_layout.set_layouts()[set_num as usize];

        struct PerDescriptorWrite {
            acceleration_structures: ash::vk::WriteDescriptorSetAccelerationStructureKHR,
        }

        let mut infos_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(descriptor_writes.len());
        let mut writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(descriptor_writes.len());
        let mut per_writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(descriptor_writes.len());

        for write in descriptor_writes {
            let layout_binding = &set_layout.bindings()[&write.binding()];

            infos_vk.push(write.to_vulkan_info(layout_binding.descriptor_type));
            writes_vk.push(write.to_vulkan(
                ash::vk::DescriptorSet::null(),
                layout_binding.descriptor_type,
            ));
            per_writes_vk.push(PerDescriptorWrite {
                acceleration_structures: Default::default(),
            });
        }

        for ((info_vk, write_vk), per_write_vk) in infos_vk
            .iter()
            .zip(writes_vk.iter_mut())
            .zip(per_writes_vk.iter_mut())
        {
            match info_vk {
                DescriptorWriteInfo::Image(info) => {
                    write_vk.descriptor_count = info.len() as u32;
                    write_vk.p_image_info = info.as_ptr();
                }
                DescriptorWriteInfo::Buffer(info) => {
                    write_vk.descriptor_count = info.len() as u32;
                    write_vk.p_buffer_info = info.as_ptr();
                }
                DescriptorWriteInfo::BufferView(info) => {
                    write_vk.descriptor_count = info.len() as u32;
                    write_vk.p_texel_buffer_view = info.as_ptr();
                }
                DescriptorWriteInfo::AccelerationStructure(info) => {
                    write_vk.descriptor_count = info.len() as u32;
                    write_vk.p_next = &per_write_vk.acceleration_structures as *const _ as _;
                    per_write_vk
                        .acceleration_structures
                        .acceleration_structure_count = write_vk.descriptor_count;
                    per_write_vk
                        .acceleration_structures
                        .p_acceleration_structures = info.as_ptr();
                }
            }

            debug_assert!(write_vk.descriptor_count != 0);
        }

        let fns = self.device().fns();
        (fns.khr_push_descriptor.cmd_push_descriptor_set_khr)(
            self.handle(),
            pipeline_bind_point.into(),
            pipeline_layout.handle(),
            set_num,
            writes_vk.len() as u32,
            writes_vk.as_ptr(),
        );

        self
    }
}

#[derive(Clone, Debug)]
pub(in super::super) enum BindPushError {
    DescriptorSetUpdateError(DescriptorSetUpdateError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The element of `descriptor_sets` being bound to a slot is not compatible with the
    /// corresponding slot in `pipeline_layout`.
    DescriptorSetNotCompatible {
        set_num: u32,
    },

    /// The descriptor set number being pushed is not defined for push descriptor sets in the
    /// pipeline layout.
    DescriptorSetNotPush {
        set_num: u32,
    },

    /// The highest descriptor set slot being bound is greater than the number of sets in
    /// `pipeline_layout`.
    DescriptorSetOutOfRange {
        set_num: u32,
        pipeline_layout_set_count: u32,
    },

    /// In an element of `descriptor_sets`, the number of provided dynamic offsets does not match
    /// the number required by the descriptor set.
    DynamicOffsetCountMismatch {
        set_num: u32,
        provided_count: usize,
        required_count: usize,
    },

    /// In an element of `descriptor_sets`, a provided dynamic offset
    /// is not a multiple of the value of the [`min_uniform_buffer_offset_alignment`] or
    /// [`min_storage_buffer_offset_alignment`]  property.
    ///
    /// min_uniform_buffer_offset_alignment: crate::device::Properties::min_uniform_buffer_offset_alignment
    /// min_storage_buffer_offset_alignment: crate::device::Properties::min_storage_buffer_offset_alignment
    DynamicOffsetNotAligned {
        set_num: u32,
        binding_num: u32,
        index: u32,
        offset: u32,
        required_alignment: DeviceAlignment,
    },

    /// In an element of `descriptor_sets`, a provided dynamic offset, when added to the end of the
    /// buffer range bound to the descriptor set, is greater than the size of the buffer.
    DynamicOffsetOutOfBufferBounds {
        set_num: u32,
        binding_num: u32,
        index: u32,
        offset: u32,
        range_end: DeviceSize,
        buffer_size: DeviceSize,
    },

    /// An index buffer is missing the `index_buffer` usage.
    IndexBufferMissingUsage,

    /// The `max_vertex_input_bindings` limit has been exceeded.
    MaxVertexInputBindingsExceeded {
        _binding_count: u32,
        _max: u32,
    },

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,

    /// The newly set pipeline has color attachment formats that do not match the
    /// previously used pipeline.
    PreviousPipelineColorAttachmentFormatMismatch,

    /// The newly set pipeline has a depth attachment format that does not match the
    /// previously used pipeline.
    PreviousPipelineDepthAttachmentFormatMismatch,

    /// The newly set pipeline has a stencil attachment format that does not match the
    /// previously used pipeline.
    PreviousPipelineStencilAttachmentFormatMismatch,

    /// The push constants data to be written at an offset is not included in any push constant
    /// range of the pipeline layout.
    PushConstantsDataOutOfRange {
        offset: u32,
    },

    /// The push constants offset is not a multiple of 4.
    PushConstantsOffsetNotAligned,

    /// The push constants size is not a multiple of 4.
    PushConstantsSizeNotAligned,

    /// A vertex buffer is missing the `vertex_buffer` usage.
    VertexBufferMissingUsage,
}

impl error::Error for BindPushError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            BindPushError::DescriptorSetUpdateError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for BindPushError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::DescriptorSetUpdateError(_) => write!(f, "a DescriptorSetUpdateError"),
            Self::DescriptorSetNotCompatible { set_num } => write!(
                f,
                "the element of `descriptor_sets` being bound to slot {} is not compatible with \
                the corresponding slot in `pipeline_layout`",
                set_num,
            ),
            Self::DescriptorSetNotPush { set_num } => write!(
                f,
                "the descriptor set number being pushed ({}) is not defined for push descriptor \
                sets in the pipeline layout",
                set_num,
            ),
            Self::DescriptorSetOutOfRange {
                set_num,
                pipeline_layout_set_count,
            } => write!(
                f,
                "the highest descriptor set slot being bound ({}) is greater than the number of \
                sets in `pipeline_layout` ({})",
                set_num, pipeline_layout_set_count,
            ),
            Self::DynamicOffsetCountMismatch {
                set_num,
                provided_count,
                required_count,
            } => write!(
                f,
                "in the element of `descriptor_sets` being bound to slot {}, the number of \
                provided dynamic offsets ({}) does not match the number required by the \
                descriptor set ({})",
                set_num, provided_count, required_count,
            ),
            Self::DynamicOffsetNotAligned {
                set_num,
                binding_num,
                index,
                offset,
                required_alignment,
            } => write!(
                f,
                "in the element of `descriptor_sets` being bound to slot {}, the dynamic offset \
                provided for binding {} index {} ({}) is not a multiple of the value of the \
                `min_uniform_buffer_offset_alignment` or `min_storage_buffer_offset_alignment` \
                property ({:?})",
                set_num, binding_num, index, offset, required_alignment,
            ),
            Self::DynamicOffsetOutOfBufferBounds {
                set_num,
                binding_num,
                index,
                offset,
                range_end,
                buffer_size,
            } => write!(
                f,
                "in the element of `descriptor_sets` being bound to slot {}, the dynamic offset \
                provided for binding {} index {} ({}), when added to the end of the buffer range \
                bound to the descriptor set ({}), is greater than the size of the buffer ({})",
                set_num, binding_num, index, offset, range_end, buffer_size,
            ),
            Self::IndexBufferMissingUsage => {
                write!(f, "an index buffer is missing the `index_buffer` usage")
            }
            Self::MaxVertexInputBindingsExceeded { .. } => {
                write!(f, "the `max_vertex_input_bindings` limit has been exceeded")
            }
            Self::NotSupportedByQueueFamily => {
                write!(f, "the queue family doesn't allow this operation")
            }
            Self::PreviousPipelineColorAttachmentFormatMismatch => write!(
                f,
                "the newly set pipeline has color attachment formats that do not match the \
                previously used pipeline",
            ),
            Self::PreviousPipelineDepthAttachmentFormatMismatch => write!(
                f,
                "the newly set pipeline has a depth attachment format that does not match the \
                previously used pipeline",
            ),
            Self::PreviousPipelineStencilAttachmentFormatMismatch => write!(
                f,
                "the newly set pipeline has a stencil attachment format that does not match the \
                previously used pipeline",
            ),
            Self::PushConstantsDataOutOfRange { offset } => write!(
                f,
                "the push constants data to be written at offset {} is not included in any push \
                constant range of the pipeline layout",
                offset,
            ),
            Self::PushConstantsOffsetNotAligned => {
                write!(f, "the push constants offset is not a multiple of 4")
            }
            Self::PushConstantsSizeNotAligned => {
                write!(f, "the push constants size is not a multiple of 4")
            }
            Self::VertexBufferMissingUsage => {
                write!(f, "a vertex buffer is missing the `vertex_buffer` usage")
            }
        }
    }
}

impl From<DescriptorSetUpdateError> for BindPushError {
    fn from(err: DescriptorSetUpdateError) -> Self {
        Self::DescriptorSetUpdateError(err)
    }
}

impl From<RequirementNotMet> for BindPushError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}
