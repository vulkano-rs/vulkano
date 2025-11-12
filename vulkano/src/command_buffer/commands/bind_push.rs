use crate::{
    buffer::{BufferContents, BufferUsage, IndexBuffer, Subbuffer},
    command_buffer::sys::RecordingCommandBuffer,
    descriptor_set::{
        layout::{DescriptorBindingFlags, DescriptorSetLayoutCreateFlags, DescriptorType},
        sys::RawDescriptorSet,
        WriteDescriptorSet,
    },
    device::{DeviceOwned, QueueFlags},
    memory::is_aligned,
    pipeline::{
        ray_tracing::RayTracingPipeline, ComputePipeline, GraphicsPipeline, PipelineBindPoint,
        PipelineLayout,
    },
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use ash::vk;
use smallvec::SmallVec;
use std::{cmp::min, ffi::c_void, ptr};

impl RecordingCommandBuffer {
    #[inline]
    pub unsafe fn bind_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&RawDescriptorSet],
        dynamic_offsets: &[u32],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout,
            first_set,
            descriptor_sets,
            dynamic_offsets,
        )?;

        Ok(unsafe {
            self.bind_descriptor_sets_unchecked(
                pipeline_bind_point,
                pipeline_layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            )
        })
    }

    pub(crate) fn validate_bind_descriptor_sets(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&RawDescriptorSet],
        dynamic_offsets: &[u32],
    ) -> Result<(), Box<ValidationError>> {
        self.validate_bind_descriptor_sets_inner(
            pipeline_bind_point,
            pipeline_layout,
            first_set,
            descriptor_sets.len(),
        )?;

        let properties = self.device().physical_device().properties();
        let mut dynamic_offsets_remaining = dynamic_offsets;
        let mut required_dynamic_offset_count = 0;

        for (descriptor_sets_index, set) in descriptor_sets.iter().enumerate() {
            let set_num = first_set + descriptor_sets_index as u32;

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

            for binding in set_layout.bindings() {
                let binding_num = binding.binding;

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

                    for (index, &offset) in dynamic_offsets.iter().enumerate() {
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
                    }
                }
            }
        }

        if dynamic_offsets.len() != required_dynamic_offset_count {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "the number of dynamic offsets provided does not equal the number required \
                    ({})",
                    required_dynamic_offset_count,
                )
                .into(),
                vuids: &["VUID-vkCmdBindDescriptorSets-dynamicOffsetCount-00359"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn validate_bind_descriptor_sets_inner(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: usize,
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
            PipelineBindPoint::RayTracing => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        context: "pipeline_bind_point".into(),
                        problem: "is `PipelineBindPoint::RayTracing`, but \
                            the queue family of the command buffer does not support \
                            compute operations"
                            .into(),
                        vuids: &[
                            "VUID-vkCmdBindDescriptorSets-pipelineBindPoint-02391",
                            "VUID-vkCmdBindDescriptorSets-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        if first_set + descriptor_sets as u32 > pipeline_layout.set_layouts().len() as u32 {
            return Err(Box::new(ValidationError {
                problem: "`first_set + descriptor_sets.len()` is greater than \
                    `pipeline_layout.set_layouts().len()`"
                    .into(),
                vuids: &["VUID-vkCmdBindDescriptorSets-firstSet-00360"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_descriptor_sets_unchecked(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&RawDescriptorSet],
        dynamic_offsets: &[u32],
    ) -> &mut Self {
        if descriptor_sets.is_empty() {
            return self;
        }

        let descriptor_sets_vk: SmallVec<[_; 12]> =
            descriptor_sets.iter().map(|x| x.handle()).collect();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_bind_descriptor_sets)(
                self.handle(),
                pipeline_bind_point.into(),
                pipeline_layout.handle(),
                first_set,
                descriptor_sets_vk.len() as u32,
                descriptor_sets_vk.as_ptr(),
                dynamic_offsets.len() as u32,
                dynamic_offsets.as_ptr(),
            )
        };

        self
    }

    #[inline]
    pub unsafe fn bind_index_buffer(
        &mut self,
        index_buffer: &IndexBuffer,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_index_buffer(index_buffer)?;

        Ok(unsafe { self.bind_index_buffer_unchecked(index_buffer) })
    }

    pub(crate) fn validate_bind_index_buffer(
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
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
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
        unsafe {
            (fns.v1_0.cmd_bind_index_buffer)(
                self.handle(),
                index_buffer_bytes.buffer().handle(),
                index_buffer_bytes.offset(),
                index_buffer.index_type().into(),
            )
        };

        self
    }

    #[inline]
    pub unsafe fn bind_pipeline_compute(
        &mut self,
        pipeline: &ComputePipeline,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_pipeline_compute(pipeline)?;

        Ok(unsafe { self.bind_pipeline_compute_unchecked(pipeline) })
    }

    pub(crate) fn validate_bind_pipeline_compute(
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
        unsafe {
            (fns.v1_0.cmd_bind_pipeline)(
                self.handle(),
                vk::PipelineBindPoint::COMPUTE,
                pipeline.handle(),
            )
        };

        self
    }

    #[inline]
    pub unsafe fn bind_pipeline_graphics(
        &mut self,
        pipeline: &GraphicsPipeline,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_pipeline_graphics(pipeline)?;

        Ok(unsafe { self.bind_pipeline_graphics_unchecked(pipeline) })
    }

    pub(crate) fn validate_bind_pipeline_graphics(
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
        unsafe {
            (fns.v1_0.cmd_bind_pipeline)(
                self.handle(),
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle(),
            )
        };

        self
    }

    pub unsafe fn bind_pipeline_ray_tracing(
        &mut self,
        pipeline: &RayTracingPipeline,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_pipeline_ray_tracing(pipeline)?;
        Ok(unsafe { self.bind_pipeline_ray_tracing_unchecked(pipeline) })
    }

    pub(crate) fn validate_bind_pipeline_ray_tracing(
        &self,
        pipeline: &RayTracingPipeline,
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
                vuids: &["VUID-vkCmdBindPipeline-pipelineBindPoint-02391"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBindPipeline-commonparent
        assert_eq!(self.device(), pipeline.device());

        // TODO: VUID-vkCmdBindPipeline-pipelineBindPoint-06721

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_pipeline_ray_tracing_unchecked(
        &mut self,
        pipeline: &RayTracingPipeline,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_bind_pipeline)(
                self.handle(),
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.handle(),
            )
        };

        self
    }

    #[inline]
    pub unsafe fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        vertex_buffers: &[Subbuffer<[u8]>],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_vertex_buffers(first_binding, vertex_buffers)?;

        Ok(unsafe { self.bind_vertex_buffers_unchecked(first_binding, vertex_buffers) })
    }

    pub(crate) fn validate_bind_vertex_buffers(
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

        let device = self.device();

        if device.api_version() >= Version::V1_3
            || device.enabled_extensions().ext_extended_dynamic_state
            || device.enabled_extensions().ext_shader_object
        {
            let mut buffers_vk: SmallVec<[_; 2]> = SmallVec::with_capacity(vertex_buffers.len());
            let mut offsets_vk: SmallVec<[_; 2]> = SmallVec::with_capacity(vertex_buffers.len());
            let mut sizes_vk: SmallVec<[_; 2]> = SmallVec::with_capacity(vertex_buffers.len());

            for buffer in vertex_buffers {
                buffers_vk.push(buffer.buffer().handle());
                offsets_vk.push(buffer.offset());
                sizes_vk.push(buffer.size());
            }

            let fns = self.device().fns();
            let cmd_bind_vertex_buffers2 = if device.api_version() >= Version::V1_3 {
                fns.v1_3.cmd_bind_vertex_buffers2
            } else if device.enabled_extensions().ext_extended_dynamic_state {
                fns.ext_extended_dynamic_state.cmd_bind_vertex_buffers2_ext
            } else {
                fns.ext_shader_object.cmd_bind_vertex_buffers2_ext
            };

            unsafe {
                cmd_bind_vertex_buffers2(
                    self.handle(),
                    first_binding,
                    buffers_vk.len() as u32,
                    buffers_vk.as_ptr(),
                    offsets_vk.as_ptr(),
                    sizes_vk.as_ptr(),
                    ptr::null(),
                )
            }
        } else {
            let mut buffers_vk: SmallVec<[_; 2]> = SmallVec::with_capacity(vertex_buffers.len());
            let mut offsets_vk: SmallVec<[_; 2]> = SmallVec::with_capacity(vertex_buffers.len());

            for buffer in vertex_buffers {
                buffers_vk.push(buffer.buffer().handle());
                offsets_vk.push(buffer.offset());
            }

            let fns = self.device().fns();
            unsafe {
                (fns.v1_0.cmd_bind_vertex_buffers)(
                    self.handle(),
                    first_binding,
                    buffers_vk.len() as u32,
                    buffers_vk.as_ptr(),
                    offsets_vk.as_ptr(),
                )
            };
        }

        self
    }

    #[inline]
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

        Ok(unsafe { self.push_constants_unchecked(pipeline_layout, offset, push_constants) })
    }

    pub(crate) fn validate_push_constants<Pc: BufferContents>(
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

        if offset as usize + remaining_size > properties.max_push_constants_size as usize {
            return Err(Box::new(ValidationError {
                problem: "`offset` + the size of `push_constants` is not less than or \
                    equal to the `max_push_constants_size` limit"
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

            // push the minimum of the whole remaining data, and the part until the end of this
            // range
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

            // push the minimum of the whole remaining data, and the part until the end of this
            // range
            let push_size = remaining_size.min(range.offset + range.size - current_offset);
            let data_offset = (current_offset - offset) as usize;
            debug_assert!(data_offset < size as usize);
            let data = unsafe { <*const _>::cast::<c_void>(push_constants).add(data_offset) };

            unsafe {
                (fns.v1_0.cmd_push_constants)(
                    self.handle(),
                    pipeline_layout.handle(),
                    range.stages.into(),
                    current_offset,
                    push_size,
                    data,
                )
            };

            current_offset += push_size;
            remaining_size -= push_size;

            if remaining_size == 0 {
                break;
            }
        }

        debug_assert!(remaining_size == 0);

        self
    }

    #[inline]
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

        Ok(unsafe {
            self.push_descriptor_set_unchecked(
                pipeline_bind_point,
                pipeline_layout,
                set_num,
                descriptor_writes,
            )
        })
    }

    pub(crate) fn validate_push_descriptor_set(
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
            PipelineBindPoint::RayTracing => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        context: "self".into(),
                        problem:
                            "`pipeline_bind_point` is `PipelineBindPoint::RayTracing`, and the \
                            queue family does not support compute operations"
                                .into(),
                        vuids: &[
                            "VUID-vkCmdPushDescriptorSetKHR-pipelineBindPoint-02391",
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
        let writes_fields1_vk: SmallVec<[_; 8]> = descriptor_writes
            .iter()
            .map(|write| {
                let default_image_layout = set_layout
                    .binding(write.binding())
                    .unwrap()
                    .descriptor_type
                    .default_image_layout();
                write.to_vk_fields1(default_image_layout)
            })
            .collect();
        let mut writes_extensions_vk: SmallVec<[_; 8]> = descriptor_writes
            .iter()
            .zip(&writes_fields1_vk)
            .map(|(write, fields1_vk)| write.to_vk_extensions(fields1_vk))
            .collect();
        let writes_vk: SmallVec<[_; 8]> = descriptor_writes
            .iter()
            .zip(&writes_fields1_vk)
            .zip(&mut writes_extensions_vk)
            .map(|((write, write_info_vk), write_extension_vk)| {
                write.to_vk(
                    vk::DescriptorSet::null(),
                    set_layout.binding(write.binding()).unwrap().descriptor_type,
                    write_info_vk,
                    write_extension_vk,
                )
            })
            .collect();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_push_descriptor.cmd_push_descriptor_set_khr)(
                self.handle(),
                pipeline_bind_point.into(),
                pipeline_layout.handle(),
                set_num,
                writes_vk.len() as u32,
                writes_vk.as_ptr(),
            )
        };

        self
    }
}
