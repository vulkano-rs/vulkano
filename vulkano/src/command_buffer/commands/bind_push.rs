// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::{BufferAccess, BufferContents, BufferUsage, TypedBufferAccess},
    command_buffer::{
        allocator::CommandBufferAllocator,
        auto::RenderPassStateType,
        synced::{Command, SetOrPush, SyncCommandBufferBuilder},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder,
    },
    descriptor_set::{
        check_descriptor_write, sys::UnsafeDescriptorSet, DescriptorSetResources,
        DescriptorSetUpdateError, DescriptorSetWithOffsets, DescriptorSetsCollection,
        DescriptorWriteInfo, WriteDescriptorSet,
    },
    device::{DeviceOwned, QueueFlags},
    pipeline::{
        graphics::{
            input_assembly::{Index, IndexType},
            render_pass::PipelineRenderPassType,
            vertex_input::VertexBuffersCollection,
        },
        ComputePipeline, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    },
    shader::ShaderStages,
    DeviceSize, RequirementNotMet, RequiresOneOf, VulkanObject,
};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::{size_of, size_of_val},
    ptr, slice,
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
    pub fn bind_descriptor_sets<S>(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        first_set: u32,
        descriptor_sets: S,
    ) -> &mut Self
    where
        S: DescriptorSetsCollection,
    {
        let descriptor_sets = descriptor_sets.into_vec();
        self.validate_bind_descriptor_sets(
            pipeline_bind_point,
            &pipeline_layout,
            first_set,
            &descriptor_sets,
        )
        .unwrap();

        unsafe {
            let mut sets_binder = self.inner.bind_descriptor_sets();
            for set in descriptor_sets.into_iter() {
                sets_binder.add(set);
            }
            sets_binder.submit(pipeline_bind_point, pipeline_layout, first_set);
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

        for (i, set) in descriptor_sets.iter().enumerate() {
            let set_num = first_set + i as u32;

            // VUID-vkCmdBindDescriptorSets-commonparent
            assert_eq!(self.device(), set.as_ref().0.device());

            let pipeline_layout_set = &pipeline_layout.set_layouts()[set_num as usize];

            // VUID-vkCmdBindDescriptorSets-pDescriptorSets-00358
            if !pipeline_layout_set.is_compatible_with(set.as_ref().0.layout()) {
                return Err(BindPushError::DescriptorSetNotCompatible { set_num });
            }

            // TODO: see https://github.com/vulkano-rs/vulkano/issues/1643
            // VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01971
            // VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01972
            // VUID-vkCmdBindDescriptorSets-pDescriptorSets-01979
            // VUID-vkCmdBindDescriptorSets-pDescriptorSets-06715
        }

        Ok(())
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
    pub fn bind_index_buffer<Ib, I>(&mut self, index_buffer: Arc<Ib>) -> &mut Self
    where
        Ib: TypedBufferAccess<Content = [I]> + 'static,
        I: Index + 'static,
    {
        self.validate_bind_index_buffer(&index_buffer, I::ty())
            .unwrap();

        unsafe {
            self.inner.bind_index_buffer(index_buffer, I::ty());
        }

        self
    }

    fn validate_bind_index_buffer(
        &self,
        index_buffer: &dyn BufferAccess,
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
        if !index_buffer.usage().intersects(BufferUsage::INDEX_BUFFER) {
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

    /// Binds a compute pipeline for future dispatch calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support compute operations.
    /// - Panics if `self` and `pipeline` do not belong to the same device.
    pub fn bind_pipeline_compute(&mut self, pipeline: Arc<ComputePipeline>) -> &mut Self {
        self.validate_bind_pipeline_compute(&pipeline).unwrap();

        unsafe {
            self.inner.bind_pipeline_compute(pipeline);
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

    /// Binds a graphics pipeline for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if `self` and `pipeline` do not belong to the same device.
    pub fn bind_pipeline_graphics(&mut self, pipeline: Arc<GraphicsPipeline>) -> &mut Self {
        self.validate_bind_pipeline_graphics(&pipeline).unwrap();

        unsafe {
            self.inner.bind_pipeline_graphics(pipeline);
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

        if let Some(last_pipeline) = self
            .render_pass_state
            .as_ref()
            .and_then(|render_pass_state| match &render_pass_state.render_pass {
                RenderPassStateType::BeginRendering(state) if state.pipeline_used => {
                    self.state().pipeline_graphics()
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

        Ok(())
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
            let mut binder = self.inner.bind_vertex_buffers();
            for vb in vertex_buffers.into_iter() {
                binder.add(vb);
            }
            binder.submit(first_binding);
        }

        self
    }

    fn validate_bind_vertex_buffers(
        &self,
        first_binding: u32,
        vertex_buffers: &[Arc<dyn BufferAccess>],
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
            if !buffer.usage().intersects(BufferUsage::VERTEX_BUFFER) {
                return Err(BindPushError::VertexBufferMissingUsage);
            }
        }

        Ok(())
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

        // SAFETY: `&push_constants` is a valid pointer, and the size of the struct is `size`,
        //         thus, getting a slice of the whole struct is safe if its not modified.
        let push_constants = unsafe {
            slice::from_raw_parts(&push_constants as *const Pc as *const u8, size as usize)
        };

        self.validate_push_constants(&pipeline_layout, offset, push_constants)
            .unwrap();

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
            unsafe {
                self.inner.push_constants::<[u8]>(
                    pipeline_layout.clone(),
                    range.stages,
                    current_offset,
                    push_size,
                    &push_constants[data_offset..(data_offset + push_size as usize)],
                );
            }
            current_offset += push_size;
            remaining_size -= push_size;

            if remaining_size == 0 {
                break;
            }
        }

        debug_assert!(remaining_size == 0);

        self
    }

    fn validate_push_constants(
        &self,
        pipeline_layout: &PipelineLayout,
        offset: u32,
        push_constants: &[u8],
    ) -> Result<(), BindPushError> {
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
            self.inner.push_descriptor_set(
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
            check_descriptor_write(write, descriptor_set_layout, 0)?;
        }

        Ok(())
    }
}

impl SyncCommandBufferBuilder {
    /// Starts the process of binding descriptor sets. Returns an intermediate struct which can be
    /// used to add the sets.
    #[inline]
    pub fn bind_descriptor_sets(&mut self) -> SyncCommandBufferBuilderBindDescriptorSets<'_> {
        SyncCommandBufferBuilderBindDescriptorSets {
            builder: self,
            descriptor_sets: SmallVec::new(),
        }
    }

    /// Calls `vkCmdBindIndexBuffer` on the builder.
    #[inline]
    pub unsafe fn bind_index_buffer(
        &mut self,
        buffer: Arc<dyn BufferAccess>,
        index_type: IndexType,
    ) {
        struct Cmd {
            buffer: Arc<dyn BufferAccess>,
            index_type: IndexType,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "bind_index_buffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_index_buffer(self.buffer.as_ref(), self.index_type);
            }
        }

        self.current_state.index_buffer = Some((buffer.clone(), index_type));
        self.commands.push(Box::new(Cmd { buffer, index_type }));
    }

    /// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_compute(&mut self, pipeline: Arc<ComputePipeline>) {
        struct Cmd {
            pipeline: Arc<ComputePipeline>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "bind_pipeline_compute"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_pipeline_compute(&self.pipeline);
            }
        }

        self.current_state.pipeline_compute = Some(pipeline.clone());
        self.commands.push(Box::new(Cmd { pipeline }));
    }

    /// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_graphics(&mut self, pipeline: Arc<GraphicsPipeline>) {
        struct Cmd {
            pipeline: Arc<GraphicsPipeline>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "bind_pipeline_graphics"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_pipeline_graphics(&self.pipeline);
            }
        }

        // Reset any states that are fixed in the new pipeline. The pipeline bind command will
        // overwrite these states.
        self.current_state.reset_dynamic_states(
            pipeline
                .dynamic_states()
                .filter(|(_, d)| !d) // not dynamic
                .map(|(s, _)| s),
        );
        self.current_state.pipeline_graphics = Some(pipeline.clone());
        self.commands.push(Box::new(Cmd { pipeline }));
    }

    /// Starts the process of binding vertex buffers. Returns an intermediate struct which can be
    /// used to add the buffers.
    #[inline]
    pub fn bind_vertex_buffers(&mut self) -> SyncCommandBufferBuilderBindVertexBuffer<'_> {
        SyncCommandBufferBuilderBindVertexBuffer {
            builder: self,
            inner: UnsafeCommandBufferBuilderBindVertexBuffer::new(),
            buffers: SmallVec::new(),
        }
    }

    /// Calls `vkCmdPushConstants` on the builder.
    pub unsafe fn push_constants<D>(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        stages: ShaderStages,
        offset: u32,
        size: u32,
        data: &D,
    ) where
        D: ?Sized + Send + Sync + 'static,
    {
        struct Cmd {
            pipeline_layout: Arc<PipelineLayout>,
            stages: ShaderStages,
            offset: u32,
            size: u32,
            data: Box<[u8]>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "push_constants"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.push_constants::<[u8]>(
                    &self.pipeline_layout,
                    self.stages,
                    self.offset,
                    self.size,
                    &self.data,
                );
            }
        }

        debug_assert!(size_of_val(data) >= size as usize);

        let mut out = Vec::with_capacity(size as usize);
        ptr::copy::<u8>(
            data as *const D as *const u8,
            out.as_mut_ptr(),
            size as usize,
        );
        out.set_len(size as usize);

        self.commands.push(Box::new(Cmd {
            pipeline_layout: pipeline_layout.clone(),
            stages,
            offset,
            size,
            data: out.into(),
        }));

        // TODO: Push constant invalidations.
        // The Vulkan spec currently is unclear about this, so Vulkano currently just marks
        // push constants as set, and never unsets them. See:
        // https://github.com/KhronosGroup/Vulkan-Docs/issues/1485
        // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2711
        self.current_state
            .push_constants
            .insert(offset..offset + size);
        self.current_state.push_constants_pipeline_layout = Some(pipeline_layout);
    }

    /// Calls `vkCmdPushDescriptorSetKHR` on the builder.
    pub unsafe fn push_descriptor_set(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        set_num: u32,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) {
        struct Cmd {
            pipeline_bind_point: PipelineBindPoint,
            pipeline_layout: Arc<PipelineLayout>,
            set_num: u32,
            descriptor_writes: SmallVec<[WriteDescriptorSet; 8]>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "push_descriptor_set"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.push_descriptor_set(
                    self.pipeline_bind_point,
                    &self.pipeline_layout,
                    self.set_num,
                    &self.descriptor_writes,
                );
            }
        }

        let descriptor_writes: SmallVec<[WriteDescriptorSet; 8]> =
            descriptor_writes.into_iter().collect();

        let state = self.current_state.invalidate_descriptor_sets(
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

        self.commands.push(Box::new(Cmd {
            pipeline_bind_point,
            pipeline_layout,
            set_num,
            descriptor_writes,
        }));
    }
}

pub struct SyncCommandBufferBuilderBindDescriptorSets<'b> {
    builder: &'b mut SyncCommandBufferBuilder,
    descriptor_sets: SmallVec<[DescriptorSetWithOffsets; 12]>,
}

impl<'b> SyncCommandBufferBuilderBindDescriptorSets<'b> {
    /// Adds a descriptor set to the list.
    pub fn add(&mut self, descriptor_set: impl Into<DescriptorSetWithOffsets>) {
        self.descriptor_sets.push(descriptor_set.into());
    }

    #[inline]
    pub unsafe fn submit(
        self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        first_set: u32,
    ) {
        if self.descriptor_sets.is_empty() {
            return;
        }

        struct Cmd {
            descriptor_sets: SmallVec<[DescriptorSetWithOffsets; 12]>,
            pipeline_bind_point: PipelineBindPoint,
            pipeline_layout: Arc<PipelineLayout>,
            first_set: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "bind_descriptor_sets"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                let descriptor_sets = self.descriptor_sets.iter().map(|x| x.as_ref().0.inner());
                let dynamic_offsets = self
                    .descriptor_sets
                    .iter()
                    .flat_map(|x| x.as_ref().1.iter().copied());

                out.bind_descriptor_sets(
                    self.pipeline_bind_point,
                    &self.pipeline_layout,
                    self.first_set,
                    descriptor_sets,
                    dynamic_offsets,
                );
            }
        }

        let state = self.builder.current_state.invalidate_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout.clone(),
            first_set,
            self.descriptor_sets.len() as u32,
        );

        for (set_num, set) in self.descriptor_sets.iter().enumerate() {
            state
                .descriptor_sets
                .insert(first_set + set_num as u32, SetOrPush::Set(set.clone()));
        }

        self.builder.commands.push(Box::new(Cmd {
            descriptor_sets: self.descriptor_sets,
            pipeline_bind_point,
            pipeline_layout,
            first_set,
        }));
    }
}

/// Prototype for a `vkCmdBindVertexBuffers`.
pub struct SyncCommandBufferBuilderBindVertexBuffer<'a> {
    builder: &'a mut SyncCommandBufferBuilder,
    inner: UnsafeCommandBufferBuilderBindVertexBuffer,
    buffers: SmallVec<[Arc<dyn BufferAccess>; 4]>,
}

impl<'a> SyncCommandBufferBuilderBindVertexBuffer<'a> {
    /// Adds a buffer to the list.
    #[inline]
    pub fn add(&mut self, buffer: Arc<dyn BufferAccess>) {
        self.inner.add(buffer.as_ref());
        self.buffers.push(buffer);
    }

    #[inline]
    pub unsafe fn submit(self, first_set: u32) {
        struct Cmd {
            first_set: u32,
            inner: Mutex<Option<UnsafeCommandBufferBuilderBindVertexBuffer>>,
            _buffers: SmallVec<[Arc<dyn BufferAccess>; 4]>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "bind_vertex_buffers"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_vertex_buffers(self.first_set, self.inner.lock().take().unwrap());
            }
        }

        for (i, buffer) in self.buffers.iter().enumerate() {
            self.builder
                .current_state
                .vertex_buffers
                .insert(first_set + i as u32, buffer.clone());
        }

        self.builder.commands.push(Box::new(Cmd {
            first_set,
            inner: Mutex::new(Some(self.inner)),
            _buffers: self.buffers,
        }));
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdBindDescriptorSets` on the builder.
    ///
    /// Does nothing if the list of descriptor sets is empty, as it would be a no-op and isn't a
    /// valid usage of the command anyway.
    #[inline]
    pub unsafe fn bind_descriptor_sets<'s>(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        sets: impl IntoIterator<Item = &'s UnsafeDescriptorSet>,
        dynamic_offsets: impl IntoIterator<Item = u32>,
    ) {
        let fns = self.device.fns();

        let sets: SmallVec<[_; 12]> = sets.into_iter().map(|s| s.handle()).collect();
        if sets.is_empty() {
            return;
        }
        let dynamic_offsets: SmallVec<[u32; 32]> = dynamic_offsets.into_iter().collect();

        let num_bindings = sets.len() as u32;
        debug_assert!(first_set + num_bindings <= pipeline_layout.set_layouts().len() as u32);

        (fns.v1_0.cmd_bind_descriptor_sets)(
            self.handle,
            pipeline_bind_point.into(),
            pipeline_layout.handle(),
            first_set,
            num_bindings,
            sets.as_ptr(),
            dynamic_offsets.len() as u32,
            dynamic_offsets.as_ptr(),
        );
    }

    /// Calls `vkCmdBindIndexBuffer` on the builder.
    #[inline]
    pub unsafe fn bind_index_buffer(&mut self, buffer: &dyn BufferAccess, index_type: IndexType) {
        let fns = self.device.fns();

        let inner = buffer.inner();
        debug_assert!(inner.offset < inner.buffer.size());
        debug_assert!(inner.buffer.usage().intersects(BufferUsage::INDEX_BUFFER));

        (fns.v1_0.cmd_bind_index_buffer)(
            self.handle,
            inner.buffer.handle(),
            inner.offset,
            index_type.into(),
        );
    }

    /// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_compute(&mut self, pipeline: &ComputePipeline) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_bind_pipeline)(
            self.handle,
            ash::vk::PipelineBindPoint::COMPUTE,
            pipeline.handle(),
        );
    }

    /// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_graphics(&mut self, pipeline: &GraphicsPipeline) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_bind_pipeline)(
            self.handle,
            ash::vk::PipelineBindPoint::GRAPHICS,
            pipeline.handle(),
        );
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
        params: UnsafeCommandBufferBuilderBindVertexBuffer,
    ) {
        debug_assert_eq!(params.raw_buffers.len(), params.offsets.len());

        if params.raw_buffers.is_empty() {
            return;
        }

        let fns = self.device.fns();

        let num_bindings = params.raw_buffers.len() as u32;

        debug_assert!({
            let max_bindings = self
                .device
                .physical_device()
                .properties()
                .max_vertex_input_bindings;
            first_binding + num_bindings <= max_bindings
        });

        (fns.v1_0.cmd_bind_vertex_buffers)(
            self.handle,
            first_binding,
            num_bindings,
            params.raw_buffers.as_ptr(),
            params.offsets.as_ptr(),
        );
    }

    /// Calls `vkCmdPushConstants` on the builder.
    pub unsafe fn push_constants<D>(
        &mut self,
        pipeline_layout: &PipelineLayout,
        stages: ShaderStages,
        offset: u32,
        size: u32,
        data: &D,
    ) where
        D: BufferContents + ?Sized,
    {
        let fns = self.device.fns();

        debug_assert!(!stages.is_empty());
        debug_assert!(size > 0);
        debug_assert_eq!(size % 4, 0);
        debug_assert_eq!(offset % 4, 0);
        debug_assert!(size_of_val(data) >= size as usize);

        (fns.v1_0.cmd_push_constants)(
            self.handle,
            pipeline_layout.handle(),
            stages.into(),
            offset as u32,
            size as u32,
            data.as_bytes().as_ptr() as *const _,
        );
    }

    /// Calls `vkCmdPushDescriptorSetKHR` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn push_descriptor_set<'a>(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        set_num: u32,
        descriptor_writes: impl IntoIterator<Item = &'a WriteDescriptorSet>,
    ) {
        debug_assert!(self.device.enabled_extensions().khr_push_descriptor);

        let (infos, mut writes): (SmallVec<[_; 8]>, SmallVec<[_; 8]>) = descriptor_writes
            .into_iter()
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
            return;
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

        let fns = self.device.fns();

        (fns.khr_push_descriptor.cmd_push_descriptor_set_khr)(
            self.handle,
            pipeline_bind_point.into(),
            pipeline_layout.handle(),
            set_num,
            writes.len() as u32,
            writes.as_ptr(),
        );
    }
}

/// Prototype for a `vkCmdBindVertexBuffers`.
#[derive(Debug)]
pub struct UnsafeCommandBufferBuilderBindVertexBuffer {
    // Raw handles of the buffers to bind.
    pub raw_buffers: SmallVec<[ash::vk::Buffer; 4]>,
    // Raw offsets of the buffers to bind.
    pub offsets: SmallVec<[DeviceSize; 4]>,
}

impl UnsafeCommandBufferBuilderBindVertexBuffer {
    /// Builds a new empty list.
    #[inline]
    pub fn new() -> UnsafeCommandBufferBuilderBindVertexBuffer {
        UnsafeCommandBufferBuilderBindVertexBuffer {
            raw_buffers: SmallVec::new(),
            offsets: SmallVec::new(),
        }
    }

    /// Adds a buffer to the list.
    #[inline]
    pub fn add(&mut self, buffer: &dyn BufferAccess) {
        let inner = buffer.inner();
        debug_assert!(inner.buffer.usage().intersects(BufferUsage::VERTEX_BUFFER));
        self.raw_buffers.push(inner.buffer.handle());
        self.offsets.push(inner.offset);
    }
}

#[derive(Clone, Debug)]
enum BindPushError {
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
