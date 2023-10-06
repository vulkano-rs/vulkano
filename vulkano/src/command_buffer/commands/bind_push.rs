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
        allocator::CommandBufferAllocator, auto::SetOrPush, sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder,
    },
    descriptor_set::{
        layout::{DescriptorBindingFlags, DescriptorSetLayoutCreateFlags, DescriptorType},
        DescriptorBindingResources, DescriptorBufferInfo, DescriptorSetResources,
        DescriptorSetWithOffsets, DescriptorSetsCollection, DescriptorWriteInfo,
        WriteDescriptorSet,
    },
    device::{DeviceOwned, QueueFlags},
    memory::is_aligned,
    pipeline::{
        graphics::vertex_input::VertexBuffersCollection, ComputePipeline, GraphicsPipeline,
        PipelineBindPoint, PipelineLayout,
    },
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};
use smallvec::SmallVec;
use std::{cmp::min, ffi::c_void, mem::size_of, sync::Arc};

/// # Commands to bind or push state for pipeline execution commands.
///
/// These commands require a queue with a pipeline type that uses the given state.
impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Binds descriptor sets for future dispatch or draw calls.
    pub fn bind_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        first_set: u32,
        descriptor_sets: impl DescriptorSetsCollection,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let descriptor_sets = descriptor_sets.into_vec();
        self.validate_bind_descriptor_sets(
            pipeline_bind_point,
            &pipeline_layout,
            first_set,
            &descriptor_sets,
        )?;

        unsafe {
            Ok(self.bind_descriptor_sets_unchecked(
                pipeline_bind_point,
                pipeline_layout,
                first_set,
                descriptor_sets,
            ))
        }
    }

    fn validate_bind_descriptor_sets(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[DescriptorSetWithOffsets],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_bind_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout,
            first_set,
            descriptor_sets,
        )?;

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
                out.bind_descriptor_sets_unchecked(
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
    pub fn bind_index_buffer(
        &mut self,
        index_buffer: impl Into<IndexBuffer>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let index_buffer = index_buffer.into();
        self.validate_bind_index_buffer(&index_buffer)?;

        unsafe { Ok(self.bind_index_buffer_unchecked(index_buffer)) }
    }

    fn validate_bind_index_buffer(
        &self,
        index_buffer: &IndexBuffer,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_bind_index_buffer(index_buffer)?;

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
                out.bind_index_buffer_unchecked(&index_buffer);
            },
        );

        self
    }

    /// Binds a compute pipeline for future dispatch calls.
    pub fn bind_pipeline_compute(
        &mut self,
        pipeline: Arc<ComputePipeline>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_pipeline_compute(&pipeline)?;

        unsafe { Ok(self.bind_pipeline_compute_unchecked(pipeline)) }
    }

    fn validate_bind_pipeline_compute(
        &self,
        pipeline: &ComputePipeline,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_bind_pipeline_compute(pipeline)?;

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
                out.bind_pipeline_compute_unchecked(&pipeline);
            },
        );

        self
    }

    /// Binds a graphics pipeline for future draw calls.
    pub fn bind_pipeline_graphics(
        &mut self,
        pipeline: Arc<GraphicsPipeline>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_pipeline_graphics(&pipeline)?;

        unsafe { Ok(self.bind_pipeline_graphics_unchecked(pipeline)) }
    }

    fn validate_bind_pipeline_graphics(
        &self,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_bind_pipeline_graphics(pipeline)?;

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
        self.builder_state
            .reset_dynamic_states(pipeline.fixed_state().iter().copied());
        self.builder_state.pipeline_graphics = Some(pipeline.clone());
        self.add_command(
            "bind_pipeline_graphics",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.bind_pipeline_graphics_unchecked(&pipeline);
            },
        );

        self
    }

    /// Binds vertex buffers for future draw calls.
    pub fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        vertex_buffers: impl VertexBuffersCollection,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let vertex_buffers = vertex_buffers.into_vec();
        self.validate_bind_vertex_buffers(first_binding, &vertex_buffers)?;

        unsafe { Ok(self.bind_vertex_buffers_unchecked(first_binding, vertex_buffers)) }
    }

    fn validate_bind_vertex_buffers(
        &self,
        first_binding: u32,
        vertex_buffers: &[Subbuffer<[u8]>],
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_bind_vertex_buffers(first_binding, vertex_buffers)?;

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
                out.bind_vertex_buffers_unchecked(first_binding, &vertex_buffers);
            },
        );

        self
    }

    /// Sets push constants for future dispatch or draw calls.
    pub fn push_constants<Pc>(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        push_constants: Pc,
    ) -> Result<&mut Self, Box<ValidationError>>
    where
        Pc: BufferContents,
    {
        let size = size_of::<Pc>() as u32;

        if size == 0 {
            return Ok(self);
        }

        self.validate_push_constants(&pipeline_layout, offset, &push_constants)?;

        unsafe { Ok(self.push_constants_unchecked(pipeline_layout, offset, push_constants)) }
    }

    fn validate_push_constants<Pc: BufferContents>(
        &self,
        pipeline_layout: &PipelineLayout,
        offset: u32,
        push_constants: &Pc,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_push_constants(pipeline_layout, offset, push_constants)?;

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
                out.push_constants_unchecked(&pipeline_layout, offset, &push_constants);
            },
        );

        self
    }

    /// Pushes descriptor data directly into the command buffer for future dispatch or draw calls.
    pub fn push_descriptor_set(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        set_num: u32,
        descriptor_writes: SmallVec<[WriteDescriptorSet; 8]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_push_descriptor_set(
            pipeline_bind_point,
            &pipeline_layout,
            set_num,
            &descriptor_writes,
        )?;

        unsafe {
            Ok(self.push_descriptor_set_unchecked(
                pipeline_bind_point,
                pipeline_layout,
                set_num,
                descriptor_writes,
            ))
        }
    }

    fn validate_push_descriptor_set(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        set_num: u32,
        descriptor_writes: &[WriteDescriptorSet],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_push_descriptor_set(
            pipeline_bind_point,
            pipeline_layout,
            set_num,
            descriptor_writes,
        )?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn push_descriptor_set_unchecked(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        set_num: u32,
        descriptor_writes: SmallVec<[WriteDescriptorSet; 8]>,
    ) -> &mut Self {
        let state = self.builder_state.invalidate_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout.clone(),
            set_num,
            1,
        );
        let layout = state.pipeline_layout.set_layouts()[set_num as usize].as_ref();
        debug_assert!(layout
            .flags()
            .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR));

        let set_resources = match state
            .descriptor_sets
            .entry(set_num)
            .or_insert_with(|| SetOrPush::Push(DescriptorSetResources::new(layout, 0)))
        {
            SetOrPush::Push(set_resources) => set_resources,
            _ => unreachable!(),
        };

        for write in &descriptor_writes {
            set_resources.write(write, layout);
        }

        self.add_command(
            "push_descriptor_set",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.push_descriptor_set_unchecked(
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
    pub unsafe fn bind_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[DescriptorSetWithOffsets],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout,
            first_set,
            descriptor_sets,
        )?;

        Ok(self.bind_descriptor_sets_unchecked(
            pipeline_bind_point,
            pipeline_layout,
            first_set,
            descriptor_sets,
        ))
    }

    fn validate_bind_descriptor_sets(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[DescriptorSetWithOffsets],
    ) -> Result<(), Box<ValidationError>> {
        pipeline_bind_point
            .validate_device(self.device())
            .map_err(|err| {
                err.add_context("pipeline_bind_point")
                    .set_vuids(&["VUID-vkCmdBindDescriptorSets-pipelineBindPoint-parameter"])
            })?;

        let queue_family_properties = self.queue_family_properties();

        match pipeline_bind_point {
            PipelineBindPoint::Compute => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        context: "pipeline_bind_point".into(),
                        problem: "is `PipelineBindPoint::Compute`, but \
                            the queue family of the command buffer does not support \
                            compute operations"
                            .into(),
                        vuids: &[
                            "VUID-vkCmdBindDescriptorSets-pipelineBindPoint-00361",
                            "VUID-vkCmdBindDescriptorSets-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
            PipelineBindPoint::Graphics => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(Box::new(ValidationError {
                        context: "pipeline_bind_point".into(),
                        problem: "is `PipelineBindPoint::Graphics`, but \
                            the queue family of the command buffer does not support \
                            graphics operations"
                            .into(),
                        vuids: &[
                            "VUID-vkCmdBindDescriptorSets-pipelineBindPoint-00361",
                            "VUID-vkCmdBindDescriptorSets-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        if first_set + descriptor_sets.len() as u32 > pipeline_layout.set_layouts().len() as u32 {
            return Err(Box::new(ValidationError {
                problem: "`first_set + descriptor_sets.len()` is greater than \
                    `pipeline_layout.set_layouts().len()`"
                    .into(),
                vuids: &["VUID-vkCmdBindDescriptorSets-firstSet-00360"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        for (descriptor_sets_index, set) in descriptor_sets.iter().enumerate() {
            let set_num = first_set + descriptor_sets_index as u32;
            let (set, dynamic_offsets) = set.as_ref();

            // VUID-vkCmdBindDescriptorSets-commonparent
            assert_eq!(self.device(), set.device());

            let set_layout = set.layout();
            let pipeline_set_layout = &pipeline_layout.set_layouts()[set_num as usize];

            if !pipeline_set_layout.is_compatible_with(set_layout) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`descriptor_sets[{0}]` (for set number {1}) is not compatible with \
                        `pipeline_layout.set_layouts()[{1}]`",
                        descriptor_sets_index, set_num
                    )
                    .into(),
                    vuids: &["VUID-vkCmdBindDescriptorSets-pDescriptorSets-00358"],
                    ..Default::default()
                }));
            }

            let mut dynamic_offsets_remaining = dynamic_offsets;
            let mut required_dynamic_offset_count = 0;

            for (&binding_num, binding) in set_layout.bindings() {
                let required_alignment = match binding.descriptor_type {
                    DescriptorType::UniformBufferDynamic => {
                        properties.min_uniform_buffer_offset_alignment
                    }
                    DescriptorType::StorageBufferDynamic => {
                        properties.min_storage_buffer_offset_alignment
                    }
                    _ => continue,
                };

                let count = if binding
                    .binding_flags
                    .intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
                {
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
                        if !is_aligned(offset as DeviceSize, required_alignment) {
                            match binding.descriptor_type {
                                DescriptorType::UniformBufferDynamic => {
                                    return Err(Box::new(ValidationError {
                                        problem: format!(
                                            "the descriptor type of `descriptor_sets[{}]` \
                                            (for set number {}) is \
                                            `DescriptorType::UniformBufferDynamic`, but the \
                                            dynamic offset provided for binding {} index {} is \
                                            not aligned to the \
                                            `min_uniform_buffer_offset_alignment` device property",
                                            descriptor_sets_index, set_num, binding_num, index,
                                        )
                                        .into(),
                                        vuids: &[
                                            "VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01971",
                                        ],
                                        ..Default::default()
                                    }));
                                }
                                DescriptorType::StorageBufferDynamic => {
                                    return Err(Box::new(ValidationError {
                                        problem: format!(
                                            "the descriptor type of `descriptor_sets[{}]` \
                                            (for set number {}) is \
                                            `DescriptorType::StorageBufferDynamic`, but the \
                                            dynamic offset provided for binding {} index {} is \
                                            not aligned to the \
                                            `min_storage_buffer_offset_alignment` device property",
                                            descriptor_sets_index, set_num, binding_num, index,
                                        )
                                        .into(),
                                        vuids: &[
                                            "VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01972",
                                        ],
                                        ..Default::default()
                                    }));
                                }
                                _ => unreachable!(),
                            }
                        }

                        if let Some(buffer_info) = element {
                            let DescriptorBufferInfo { buffer, range } = buffer_info;

                            if offset as DeviceSize + range.end > buffer.size() {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the dynamic offset of `descriptor_sets[{}]` \
                                        (for set number {}) for binding {} index {}, when \
                                        added to `range.end` of the descriptor write, is \
                                        greater than the size of the bound buffer",
                                        descriptor_sets_index, set_num, binding_num, index,
                                    )
                                    .into(),
                                    vuids: &["VUID-vkCmdBindDescriptorSets-pDescriptorSets-01979"],
                                    ..Default::default()
                                }));
                            }
                        }
                    }
                }
            }

            if dynamic_offsets.len() != required_dynamic_offset_count {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "the number of dynamic offsets provided for `descriptor_sets[{}]` \
                        (for set number {}) does not equal the number required ({})",
                        descriptor_sets_index, set_num, required_dynamic_offset_count,
                    )
                    .into(),
                    vuids: &["VUID-vkCmdBindDescriptorSets-dynamicOffsetCount-00359"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_descriptor_sets_unchecked(
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

    pub unsafe fn bind_index_buffer(
        &mut self,
        index_buffer: &IndexBuffer,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_index_buffer(index_buffer)?;

        Ok(self.bind_index_buffer_unchecked(index_buffer))
    }

    fn validate_bind_index_buffer(
        &self,
        index_buffer: &IndexBuffer,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdBindIndexBuffer-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let index_buffer_bytes = index_buffer.as_bytes();

        // VUID-vkCmdBindIndexBuffer-commonparent
        assert_eq!(self.device(), index_buffer_bytes.device());

        if !index_buffer_bytes
            .buffer()
            .usage()
            .intersects(BufferUsage::INDEX_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "index_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDEX_BUFFER`".into(),
                vuids: &["VUID-vkCmdBindIndexBuffer-buffer-00433"],
                ..Default::default()
            }));
        }

        if matches!(index_buffer, IndexBuffer::U8(_))
            && !self.device().enabled_features().index_type_uint8
        {
            return Err(Box::new(ValidationError {
                context: "index_buffer".into(),
                problem: "is `IndexBuffer::U8`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "index_type_uint8",
                )])]),
                vuids: &["VUID-vkCmdBindIndexBuffer-indexType-02765"],
            }));
        }

        // TODO:
        // VUID-vkCmdBindIndexBuffer-offset-00432

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_index_buffer_unchecked(&mut self, index_buffer: &IndexBuffer) -> &mut Self {
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

    pub unsafe fn bind_pipeline_compute(
        &mut self,
        pipeline: &ComputePipeline,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_pipeline_compute(pipeline)?;

        Ok(self.bind_pipeline_compute_unchecked(pipeline))
    }

    fn validate_bind_pipeline_compute(
        &self,
        pipeline: &ComputePipeline,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    compute operations"
                    .into(),
                vuids: &["VUID-vkCmdBindPipeline-pipelineBindPoint-00777"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBindPipeline-commonparent
        assert_eq!(self.device(), pipeline.device());

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_pipeline_compute_unchecked(
        &mut self,
        pipeline: &ComputePipeline,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_pipeline)(
            self.handle(),
            ash::vk::PipelineBindPoint::COMPUTE,
            pipeline.handle(),
        );

        self
    }

    pub unsafe fn bind_pipeline_graphics(
        &mut self,
        pipeline: &GraphicsPipeline,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_pipeline_graphics(pipeline)?;

        Ok(self.bind_pipeline_graphics_unchecked(pipeline))
    }

    fn validate_bind_pipeline_graphics(
        &self,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdBindPipeline-pipelineBindPoint-00778"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBindPipeline-commonparent
        assert_eq!(self.device(), pipeline.device());

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_pipeline_graphics_unchecked(
        &mut self,
        pipeline: &GraphicsPipeline,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_bind_pipeline)(
            self.handle(),
            ash::vk::PipelineBindPoint::GRAPHICS,
            pipeline.handle(),
        );

        self
    }

    pub unsafe fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        vertex_buffers: &[Subbuffer<[u8]>],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_vertex_buffers(first_binding, vertex_buffers)?;

        Ok(self.bind_vertex_buffers_unchecked(first_binding, vertex_buffers))
    }

    fn validate_bind_vertex_buffers(
        &self,
        first_binding: u32,
        vertex_buffers: &[Subbuffer<[u8]>],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdBindVertexBuffers-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if first_binding + vertex_buffers.len() as u32 > properties.max_vertex_input_bindings {
            return Err(Box::new(ValidationError {
                problem: "`first_binding + vertex_buffers.len()` is greater than the \
                    `max_vertex_input_bindings` limit"
                    .into(),
                vuids: &[
                    "VUID-vkCmdBindVertexBuffers-firstBinding-00624",
                    "VUID-vkCmdBindVertexBuffers-firstBinding-00625",
                ],
                ..Default::default()
            }));
        }

        for (vertex_buffers_index, buffer) in vertex_buffers.iter().enumerate() {
            // VUID-vkCmdBindVertexBuffers-commonparent
            assert_eq!(self.device(), buffer.device());

            if !buffer
                .buffer()
                .usage()
                .intersects(BufferUsage::VERTEX_BUFFER)
            {
                return Err(Box::new(ValidationError {
                    context: format!("vertex_buffers[{}].usage()", vertex_buffers_index).into(),
                    problem: "does not contain `BufferUsage::VERTEX_BUFFER`".into(),
                    vuids: &["VUID-vkCmdBindVertexBuffers-pBuffers-00627"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_vertex_buffers_unchecked(
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

    pub unsafe fn push_constants<Pc>(
        &mut self,
        pipeline_layout: &PipelineLayout,
        offset: u32,
        push_constants: &Pc,
    ) -> Result<&mut Self, Box<ValidationError>>
    where
        Pc: BufferContents,
    {
        self.validate_push_constants(pipeline_layout, offset, push_constants)?;

        Ok(self.push_constants_unchecked(pipeline_layout, offset, push_constants))
    }

    fn validate_push_constants<Pc: BufferContents>(
        &self,
        pipeline_layout: &PipelineLayout,
        offset: u32,
        _push_constants: &Pc,
    ) -> Result<(), Box<ValidationError>> {
        let mut remaining_size = size_of::<Pc>();

        if offset % 4 != 0 {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdPushConstants-offset-00368"],
                ..Default::default()
            }));
        }

        if remaining_size % 4 != 0 {
            return Err(Box::new(ValidationError {
                context: "push_constants".into(),
                problem: "the size is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdPushConstants-size-00369"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if offset >= properties.max_push_constants_size {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not less than the `max_push_constants_size` limit".into(),
                vuids: &["VUID-vkCmdPushConstants-offset-00370"],
                ..Default::default()
            }));
        }

        if offset as usize + remaining_size >= properties.max_push_constants_size as usize {
            return Err(Box::new(ValidationError {
                problem: "`offset` + the size of `push_constants` is not less than the \
                    `max_push_constants_size` limit"
                    .into(),
                vuids: &["VUID-vkCmdPushConstants-size-00371"],
                ..Default::default()
            }));
        }

        let mut current_offset = offset as usize;

        for range in pipeline_layout
            .push_constant_ranges_disjoint()
            .iter()
            .skip_while(|range| range.offset + range.size <= offset)
        {
            // there is a gap between ranges, but the passed push_constants contains
            // some bytes in this gap, exit the loop and report error
            if range.offset as usize > current_offset {
                break;
            }

            // push the minimum of the whole remaining data, and the part until the end of this range
            let push_size =
                remaining_size.min(range.offset as usize + range.size as usize - current_offset);
            current_offset += push_size;
            remaining_size -= push_size;

            if remaining_size == 0 {
                break;
            }
        }

        if remaining_size != 0 {
            return Err(Box::new(ValidationError {
                problem: "one or more bytes of `push_constants` are not within any push constant \
                    range of `pipeline_layout`"
                    .into(),
                vuids: &["VUID-vkCmdPushConstants-offset-01795"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn push_constants_unchecked<Pc>(
        &mut self,
        pipeline_layout: &PipelineLayout,
        offset: u32,
        push_constants: &Pc,
    ) -> &mut Self
    where
        Pc: BufferContents,
    {
        let size = u32::try_from(size_of::<Pc>()).unwrap();

        if size == 0 {
            return self;
        }

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
            debug_assert!(data_offset < size as usize);
            let data = (push_constants as *const Pc as *const c_void).add(data_offset);

            (fns.v1_0.cmd_push_constants)(
                self.handle(),
                pipeline_layout.handle(),
                range.stages.into(),
                current_offset,
                push_size,
                data,
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

    pub unsafe fn push_descriptor_set(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        set_num: u32,
        descriptor_writes: &[WriteDescriptorSet],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_push_descriptor_set(
            pipeline_bind_point,
            pipeline_layout,
            set_num,
            descriptor_writes,
        )?;

        Ok(self.push_descriptor_set_unchecked(
            pipeline_bind_point,
            pipeline_layout,
            set_num,
            descriptor_writes,
        ))
    }

    fn validate_push_descriptor_set(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        set_num: u32,
        descriptor_writes: &[WriteDescriptorSet],
    ) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_extensions().khr_push_descriptor {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_push_descriptor",
                )])]),
                ..Default::default()
            }));
        }

        pipeline_bind_point
            .validate_device(self.device())
            .map_err(|err| {
                err.add_context("pipeline_bind_point")
                    .set_vuids(&["VUID-vkCmdPushDescriptorSetKHR-pipelineBindPoint-parameter"])
            })?;

        let queue_family_properties = self.queue_family_properties();

        match pipeline_bind_point {
            PipelineBindPoint::Compute => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        context: "self".into(),
                        problem: "`pipeline_bind_point` is `PipelineBindPoint::Compute`, and the \
                            queue family does not support compute operations"
                            .into(),
                        vuids: &[
                            "VUID-vkCmdPushDescriptorSetKHR-pipelineBindPoint-00363",
                            "VUID-vkCmdPushDescriptorSetKHR-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
            PipelineBindPoint::Graphics => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(Box::new(ValidationError {
                        context: "self".into(),
                        problem: "`pipeline_bind_point` is `PipelineBindPoint::Graphics`, and the \
                            queue family does not support graphics operations"
                            .into(),
                        vuids: &[
                            "VUID-vkCmdPushDescriptorSetKHR-pipelineBindPoint-00363",
                            "VUID-vkCmdPushDescriptorSetKHR-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        // VUID-vkCmdPushDescriptorSetKHR-commonparent
        assert_eq!(self.device(), pipeline_layout.device());

        if set_num as usize > pipeline_layout.set_layouts().len() {
            return Err(Box::new(ValidationError {
                problem: "`set_num` is greater than the number of descriptor set layouts in \
                    `pipeline_layout`"
                    .into(),
                vuids: &["VUID-vkCmdPushDescriptorSetKHR-set-00364"],
                ..Default::default()
            }));
        }

        let descriptor_set_layout = &pipeline_layout.set_layouts()[set_num as usize];

        if !descriptor_set_layout
            .flags()
            .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
        {
            return Err(Box::new(ValidationError {
                problem: "the descriptor set layout with the number `set_num` in \
                    `pipeline_layout` was not created with the \
                    `DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR` flag"
                    .into(),
                vuids: &["VUID-vkCmdPushDescriptorSetKHR-set-00365"],
                ..Default::default()
            }));
        }

        for (index, write) in descriptor_writes.iter().enumerate() {
            write
                .validate(descriptor_set_layout, 0)
                .map_err(|err| err.add_context(format!("descriptor_writes[{}]", index)))?;
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn push_descriptor_set_unchecked(
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
            write_info: DescriptorWriteInfo,
            acceleration_structures: ash::vk::WriteDescriptorSetAccelerationStructureKHR,
            inline_uniform_block: ash::vk::WriteDescriptorSetInlineUniformBlock,
        }

        let mut writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(descriptor_writes.len());
        let mut per_writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(descriptor_writes.len());

        for write in descriptor_writes {
            let layout_binding = &set_layout.bindings()[&write.binding()];

            writes_vk.push(write.to_vulkan(
                ash::vk::DescriptorSet::null(),
                layout_binding.descriptor_type,
            ));
            per_writes_vk.push(PerDescriptorWrite {
                write_info: write.to_vulkan_info(layout_binding.descriptor_type),
                acceleration_structures: Default::default(),
                inline_uniform_block: Default::default(),
            });
        }

        for (write_vk, per_write_vk) in writes_vk.iter_mut().zip(per_writes_vk.iter_mut()) {
            match &mut per_write_vk.write_info {
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
                DescriptorWriteInfo::InlineUniformBlock(data) => {
                    write_vk.descriptor_count = data.len() as u32;
                    write_vk.p_next = &per_write_vk.inline_uniform_block as *const _ as _;
                    per_write_vk.inline_uniform_block.data_size = write_vk.descriptor_count;
                    per_write_vk.inline_uniform_block.p_data = data.as_ptr() as *const _;
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
