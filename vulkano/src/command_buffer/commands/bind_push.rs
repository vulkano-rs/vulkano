// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::{BufferAccess, BufferContents, TypedBufferAccess},
    command_buffer::{
        synced::{Command, SetOrPush, SyncCommandBufferBuilder},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder,
    },
    descriptor_set::{
        check_descriptor_write, sys::UnsafeDescriptorSet, DescriptorSetResources,
        DescriptorSetWithOffsets, DescriptorSetsCollection, DescriptorWriteInfo,
        WriteDescriptorSet,
    },
    device::DeviceOwned,
    pipeline::{
        graphics::{
            input_assembly::{Index, IndexType},
            vertex_input::VertexBuffersCollection,
        },
        ComputePipeline, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    },
    shader::ShaderStages,
    DeviceSize, VulkanObject,
};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    mem::{size_of, size_of_val},
    ptr, slice,
    sync::Arc,
};

/// # Commands to bind or push state for pipeline execution commands.
///
/// These commands require a queue with a pipeline type that uses the given state.
impl<L, P> AutoCommandBufferBuilder<L, P> {
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
        match pipeline_bind_point {
            PipelineBindPoint::Compute => assert!(
                self.queue_family().supports_compute(),
                "the queue family of the command buffer must support compute operations"
            ),
            PipelineBindPoint::Graphics => assert!(
                self.queue_family().supports_graphics(),
                "the queue family of the command buffer must support graphics operations"
            ),
        }

        let descriptor_sets = descriptor_sets.into_vec();

        assert!(
            first_set as usize + descriptor_sets.len()
                <= pipeline_layout.set_layouts().len(),
            "the highest descriptor set slot being bound must be less than the number of sets in pipeline_layout"
        );

        for (num, set) in descriptor_sets.iter().enumerate() {
            assert_eq!(
                set.as_ref().0.device().internal_object(),
                self.device().internal_object()
            );

            let pipeline_set = &pipeline_layout.set_layouts()[first_set as usize + num];
            assert!(
                pipeline_set.is_compatible_with(set.as_ref().0.layout()),
                "the element of descriptor_sets being bound to slot {} is not compatible with the corresponding slot in pipeline_layout",
                first_set as usize + num,
            );

            // TODO: see https://github.com/vulkano-rs/vulkano/issues/1643
            // For each dynamic uniform or storage buffer binding in pDescriptorSets, the sum of the
            // effective offset, as defined above, and the range of the binding must be less than or
            // equal to the size of the buffer

            // TODO:
            // Each element of pDescriptorSets must not have been allocated from a VkDescriptorPool
            // with the VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_VALVE flag set
        }

        unsafe {
            let mut sets_binder = self.inner.bind_descriptor_sets();
            for set in descriptor_sets.into_iter() {
                sets_binder.add(set);
            }
            sets_binder.submit(pipeline_bind_point, pipeline_layout, first_set);
        }

        self
    }

    /// Binds an index buffer for future indexed draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if `self` and `index_buffer` do not belong to the same device.
    /// - Panics if `index_buffer` does not have the
    ///   [`index_buffer`](crate::buffer::BufferUsage::index_buffer) usage enabled.
    /// - If the index buffer contains `u8` indices, panics if the
    ///   [`index_type_uint8`](crate::device::Features::index_type_uint8) feature is not
    ///   enabled on the device.
    pub fn bind_index_buffer<Ib, I>(&mut self, index_buffer: Arc<Ib>) -> &mut Self
    where
        Ib: TypedBufferAccess<Content = [I]> + 'static,
        I: Index + 'static,
    {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        assert_eq!(
            index_buffer.device().internal_object(),
            self.device().internal_object()
        );

        // TODO:
        // The sum of offset and the address of the range of VkDeviceMemory object that is backing
        // buffer, must be a multiple of the type indicated by indexType

        assert!(
            index_buffer.inner().buffer.usage().index_buffer,
            "index_buffer must have the index_buffer usage enabled"
        );

        // TODO:
        // If buffer is non-sparse then it must be bound completely and contiguously to a single
        // VkDeviceMemory object

        if !self.device().enabled_features().index_type_uint8 {
            assert!(I::ty() != IndexType::U8, "if the index buffer contains u8 indices, the index_type_uint8 feature must be enabled on the device");
        }

        unsafe {
            self.inner.bind_index_buffer(index_buffer, I::ty());
        }

        self
    }

    /// Binds a compute pipeline for future dispatch calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support compute operations.
    /// - Panics if `self` and `pipeline` do not belong to the same device.
    pub fn bind_pipeline_compute(&mut self, pipeline: Arc<ComputePipeline>) -> &mut Self {
        assert!(
            self.queue_family().supports_compute(),
            "the queue family of the command buffer must support compute operations"
        );

        assert_eq!(
            pipeline.device().internal_object(),
            self.device().internal_object()
        );

        // TODO:
        // This command must not be recorded when transform feedback is active

        // TODO:
        // pipeline must not have been created with VK_PIPELINE_CREATE_LIBRARY_BIT_KHR set

        unsafe {
            self.inner.bind_pipeline_compute(pipeline);
        }

        self
    }

    /// Binds a graphics pipeline for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if `self` and `pipeline` do not belong to the same device.
    pub fn bind_pipeline_graphics(&mut self, pipeline: Arc<GraphicsPipeline>) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        assert_eq!(
            pipeline.device().internal_object(),
            self.device().internal_object()
        );

        // TODO:
        // If the variable multisample rate feature is not supported, pipeline is a graphics
        // pipeline, the current subpass uses no attachments, and this is not the first call to
        // this function with a graphics pipeline after transitioning to the current subpass, then
        // the sample count specified by this pipeline must match that set in the previous pipeline

        // TODO:
        // If VkPhysicalDeviceSampleLocationsPropertiesEXT::variableSampleLocations is VK_FALSE, and
        // pipeline is a graphics pipeline created with a
        // VkPipelineSampleLocationsStateCreateInfoEXT structure having its sampleLocationsEnable
        // member set to VK_TRUE but without VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_EXT enabled then the
        // current render pass instance must have been begun by specifying a
        // VkRenderPassSampleLocationsBeginInfoEXT structure whose pPostSubpassSampleLocations
        // member contains an element with a subpassIndex matching the current subpass index and the
        // sampleLocationsInfo member of that element must match the sampleLocationsInfo specified
        // in VkPipelineSampleLocationsStateCreateInfoEXT when the pipeline was created

        // TODO:
        // This command must not be recorded when transform feedback is active

        // TODO:
        // pipeline must not have been created with VK_PIPELINE_CREATE_LIBRARY_BIT_KHR set

        // TODO:
        // If commandBuffer is a secondary command buffer with
        // VkCommandBufferInheritanceViewportScissorInfoNV::viewportScissor2D enabled and
        // pipelineBindPoint is VK_PIPELINE_BIND_POINT_GRAPHICS, then the pipeline must have been
        // created with VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT or VK_DYNAMIC_STATE_VIEWPORT, and
        // VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT_EXT or VK_DYNAMIC_STATE_SCISSOR enabled

        // TODO:
        // If pipelineBindPoint is VK_PIPELINE_BIND_POINT_GRAPHICS and the
        // provokingVertexModePerPipeline limit is VK_FALSE, then pipeline’s
        // VkPipelineRasterizationProvokingVertexStateCreateInfoEXT::provokingVertexMode must be the
        // same as that of any other pipelines previously bound to this bind point within the
        // current renderpass instance, including any pipeline already bound when beginning the
        // renderpass instance

        unsafe {
            self.inner.bind_pipeline_graphics(pipeline);
        }

        self
    }

    /// Binds vertex buffers for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the highest vertex buffer binding being bound is greater than the
    ///   [`max_vertex_input_bindings`](crate::device::Properties::max_vertex_input_bindings)
    //    device property.
    /// - Panics if `self` and any element of `vertex_buffers` do not belong to the same device.
    /// - Panics if any element of `vertex_buffers` does not have the
    ///   [`vertex_buffer`](crate::buffer::BufferUsage::vertex_buffer) usage enabled.
    pub fn bind_vertex_buffers<V>(&mut self, first_binding: u32, vertex_buffers: V) -> &mut Self
    where
        V: VertexBuffersCollection,
    {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        let vertex_buffers = vertex_buffers.into_vec();

        assert!(
            first_binding + vertex_buffers.len() as u32
                <= self
                    .device()
                    .physical_device()
                    .properties()
                    .max_vertex_input_bindings,
            "the highest vertex buffer binding being bound must not be higher than the max_vertex_input_bindings device property"
        );

        for (num, buf) in vertex_buffers.iter().enumerate() {
            assert_eq!(
                buf.device().internal_object(),
                self.device().internal_object()
            );

            assert!(
                buf.inner().buffer.usage().vertex_buffer,
                "vertex_buffers element {} must have the vertex_buffer usage",
                num
            );

            // TODO:
            // Each element of pBuffers that is non-sparse must be bound completely and contiguously
            // to a single VkDeviceMemory object

            // TODO:
            // If the nullDescriptor feature is not enabled, all elements of pBuffers must not be
            // VK_NULL_HANDLE

            // TODO:
            // If an element of pBuffers is VK_NULL_HANDLE, then the corresponding element of
            // pOffsets must be zero
        }

        unsafe {
            let mut binder = self.inner.bind_vertex_buffers();
            for vb in vertex_buffers.into_iter() {
                binder.add(vb);
            }
            binder.submit(first_binding);
        }

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
    ) -> &mut Self {
        let size = size_of::<Pc>() as u32;

        if size == 0 {
            return self;
        }

        assert!(offset % 4 == 0, "the offset must be a multiple of 4");
        assert!(
            size % 4 == 0,
            "the size of push_constants must be a multiple of 4"
        );

        // SAFETY: `&push_constants` is a valid pointer, and the size of the struct is `size`,
        //         thus, getting a slice of the whole struct is safe if its not modified.
        let whole_data = unsafe {
            slice::from_raw_parts(&push_constants as *const Pc as *const u8, size as usize)
        };

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
                    &whole_data[data_offset..(data_offset + push_size as usize)],
                );
            }
            current_offset += push_size;
            remaining_size -= push_size;

            if remaining_size == 0 {
                break;
            }
        }

        assert!(
            remaining_size == 0,
            "There exists data at offset {} that is not included in any range",
            current_offset,
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
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> &mut Self {
        match pipeline_bind_point {
            PipelineBindPoint::Compute => assert!(
                self.queue_family().supports_compute(),
                "the queue family of the command buffer must support compute operations"
            ),
            PipelineBindPoint::Graphics => assert!(
                self.queue_family().supports_graphics(),
                "the queue family of the command buffer must support graphics operations"
            ),
        }

        assert!(
            self.device().enabled_extensions().khr_push_descriptor,
            "the khr_push_descriptor extension must be enabled on the device"
        );
        assert!(
            set_num as usize <= pipeline_layout.set_layouts().len(),
            "the descriptor set slot being bound must be less than the number of sets in pipeline_layout"
        );

        let descriptor_writes: SmallVec<[_; 8]> = descriptor_writes.into_iter().collect();
        let descriptor_set_layout = &pipeline_layout.set_layouts()[set_num as usize];

        for write in &descriptor_writes {
            check_descriptor_write(write, descriptor_set_layout, 0).unwrap();
        }

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
}

impl SyncCommandBufferBuilder {
    /// Starts the process of binding descriptor sets. Returns an intermediate struct which can be
    /// used to add the sets.
    #[inline]
    pub fn bind_descriptor_sets(&mut self) -> SyncCommandBufferBuilderBindDescriptorSets {
        SyncCommandBufferBuilderBindDescriptorSets {
            builder: self,
            descriptor_sets: SmallVec::new(),
        }
    }

    /// Calls `vkCmdBindIndexBuffer` on the builder.
    #[inline]
    pub unsafe fn bind_index_buffer(&mut self, buffer: Arc<dyn BufferAccess>, index_ty: IndexType) {
        struct Cmd {
            buffer: Arc<dyn BufferAccess>,
            index_ty: IndexType,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdBindIndexBuffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_index_buffer(self.buffer.as_ref(), self.index_ty);
            }
        }

        self.current_state.index_buffer = Some((buffer.clone(), index_ty));
        self.append_command(Cmd { buffer, index_ty }, []).unwrap();
    }

    /// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_compute(&mut self, pipeline: Arc<ComputePipeline>) {
        struct Cmd {
            pipeline: Arc<ComputePipeline>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdBindPipeline"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_pipeline_compute(&self.pipeline);
            }
        }

        self.current_state.pipeline_compute = Some(pipeline.clone());
        self.append_command(Cmd { pipeline }, []).unwrap();
    }

    /// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_graphics(&mut self, pipeline: Arc<GraphicsPipeline>) {
        struct Cmd {
            pipeline: Arc<GraphicsPipeline>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdBindPipeline"
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
        self.append_command(Cmd { pipeline }, []).unwrap();
    }

    /// Starts the process of binding vertex buffers. Returns an intermediate struct which can be
    /// used to add the buffers.
    #[inline]
    pub fn bind_vertex_buffers(&mut self) -> SyncCommandBufferBuilderBindVertexBuffer {
        SyncCommandBufferBuilderBindVertexBuffer {
            builder: self,
            inner: UnsafeCommandBufferBuilderBindVertexBuffer::new(),
            buffers: SmallVec::new(),
        }
    }

    /// Calls `vkCmdPushConstants` on the builder.
    #[inline]
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
                "vkCmdPushConstants"
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

        self.append_command(
            Cmd {
                pipeline_layout: pipeline_layout.clone(),
                stages,
                offset,
                size,
                data: out.into(),
            },
            [],
        )
        .unwrap();

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
                "vkCmdPushDescriptorSetKHR"
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
            .or_insert(SetOrPush::Push(DescriptorSetResources::new(layout, 0)))
        {
            SetOrPush::Push(set_resources) => set_resources,
            _ => unreachable!(),
        };

        for write in &descriptor_writes {
            set_resources.update(write);
        }

        self.append_command(
            Cmd {
                pipeline_bind_point,
                pipeline_layout,
                set_num,
                descriptor_writes,
            },
            [],
        )
        .unwrap();
    }
}

pub struct SyncCommandBufferBuilderBindDescriptorSets<'b> {
    builder: &'b mut SyncCommandBufferBuilder,
    descriptor_sets: SmallVec<[DescriptorSetWithOffsets; 12]>,
}

impl<'b> SyncCommandBufferBuilderBindDescriptorSets<'b> {
    /// Adds a descriptor set to the list.
    #[inline]
    pub fn add<S>(&mut self, descriptor_set: S)
    where
        S: Into<DescriptorSetWithOffsets>,
    {
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
                "vkCmdBindDescriptorSets"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                let descriptor_sets = self.descriptor_sets.iter().map(|x| x.as_ref().0.inner());
                let dynamic_offsets = self
                    .descriptor_sets
                    .iter()
                    .map(|x| x.as_ref().1.iter().copied())
                    .flatten();

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

        self.builder
            .append_command(
                Cmd {
                    descriptor_sets: self.descriptor_sets,
                    pipeline_bind_point,
                    pipeline_layout,
                    first_set,
                },
                [],
            )
            .unwrap();
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
            buffers: SmallVec<[Arc<dyn BufferAccess>; 4]>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdBindVertexBuffers"
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

        self.builder
            .append_command(
                Cmd {
                    first_set,
                    inner: Mutex::new(Some(self.inner)),
                    buffers: self.buffers,
                },
                [],
            )
            .unwrap();
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

        let sets: SmallVec<[_; 12]> = sets.into_iter().map(|s| s.internal_object()).collect();
        if sets.is_empty() {
            return;
        }
        let dynamic_offsets: SmallVec<[u32; 32]> = dynamic_offsets.into_iter().collect();

        let num_bindings = sets.len() as u32;
        debug_assert!(first_set + num_bindings <= pipeline_layout.set_layouts().len() as u32);

        fns.v1_0.cmd_bind_descriptor_sets(
            self.handle,
            pipeline_bind_point.into(),
            pipeline_layout.internal_object(),
            first_set,
            num_bindings,
            sets.as_ptr(),
            dynamic_offsets.len() as u32,
            dynamic_offsets.as_ptr(),
        );
    }

    /// Calls `vkCmdBindIndexBuffer` on the builder.
    #[inline]
    pub unsafe fn bind_index_buffer(&mut self, buffer: &dyn BufferAccess, index_ty: IndexType) {
        let fns = self.device.fns();

        let inner = buffer.inner();
        debug_assert!(inner.offset < inner.buffer.size());
        debug_assert!(inner.buffer.usage().index_buffer);

        fns.v1_0.cmd_bind_index_buffer(
            self.handle,
            inner.buffer.internal_object(),
            inner.offset,
            index_ty.into(),
        );
    }

    /// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_compute(&mut self, pipeline: &ComputePipeline) {
        let fns = self.device.fns();
        fns.v1_0.cmd_bind_pipeline(
            self.handle,
            ash::vk::PipelineBindPoint::COMPUTE,
            pipeline.internal_object(),
        );
    }

    /// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_graphics(&mut self, pipeline: &GraphicsPipeline) {
        let fns = self.device.fns();
        fns.v1_0.cmd_bind_pipeline(
            self.handle,
            ash::vk::PipelineBindPoint::GRAPHICS,
            pipeline.internal_object(),
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

        fns.v1_0.cmd_bind_vertex_buffers(
            self.handle,
            first_binding,
            num_bindings,
            params.raw_buffers.as_ptr(),
            params.offsets.as_ptr(),
        );
    }

    /// Calls `vkCmdPushConstants` on the builder.
    #[inline]
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

        debug_assert!(stages != ShaderStages::none());
        debug_assert!(size > 0);
        debug_assert_eq!(size % 4, 0);
        debug_assert_eq!(offset % 4, 0);
        debug_assert!(size_of_val(data) >= size as usize);

        fns.v1_0.cmd_push_constants(
            self.handle,
            pipeline_layout.internal_object(),
            stages.into(),
            offset as u32,
            size as u32,
            data.as_bytes().as_ptr() as *const _,
        );
    }

    /// Calls `vkCmdPushDescriptorSetKHR` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
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

        fns.khr_push_descriptor.cmd_push_descriptor_set_khr(
            self.handle,
            pipeline_bind_point.into(),
            pipeline_layout.internal_object(),
            set_num,
            writes.len() as u32,
            writes.as_ptr(),
        );
    }
}

/// Prototype for a `vkCmdBindVertexBuffers`.
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
        debug_assert!(inner.buffer.usage().vertex_buffer);
        self.raw_buffers.push(inner.buffer.internal_object());
        self.offsets.push(inner.offset);
    }
}
