use crate::{
    command_buffer::{RecordingCommandBuffer, Result},
    Id,
};
use ash::vk;
use smallvec::SmallVec;
use std::{ffi::c_void, ptr, sync::Arc};
use vulkano::{
    self,
    buffer::{Buffer, BufferContents, IndexType},
    device::DeviceOwned,
    pipeline::{
        ray_tracing::RayTracingPipeline, ComputePipeline, GraphicsPipeline, PipelineLayout,
    },
    DeviceSize, Version, VulkanObject,
};

/// # Commands to bind or push state for pipeline execution commands
///
/// These commands require a queue with a pipeline type that uses the given state.
impl RecordingCommandBuffer<'_> {
    /// Binds an index buffer for future indexed draw calls.
    pub unsafe fn bind_index_buffer(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        size: DeviceSize,
        index_type: IndexType,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.bind_index_buffer_unchecked(buffer, offset, size, index_type) })
    }

    pub unsafe fn bind_index_buffer_unchecked(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        size: DeviceSize,
        index_type: IndexType,
    ) -> &mut Self {
        let buffer = unsafe { self.accesses.buffer_unchecked(buffer) };

        let fns = self.device().fns();

        if self.device().enabled_extensions().khr_maintenance5 {
            unsafe {
                (fns.khr_maintenance5.cmd_bind_index_buffer2_khr)(
                    self.handle(),
                    buffer.handle(),
                    offset,
                    size,
                    index_type.into(),
                )
            };
        } else {
            unsafe {
                (fns.v1_0.cmd_bind_index_buffer)(
                    self.handle(),
                    buffer.handle(),
                    offset,
                    index_type.into(),
                )
            };
        }

        self
    }

    /// Binds a compute pipeline for future dispatch calls.
    pub unsafe fn bind_pipeline_compute(
        &mut self,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.bind_pipeline_compute_unchecked(pipeline) })
    }

    pub unsafe fn bind_pipeline_compute_unchecked(
        &mut self,
        pipeline: &Arc<ComputePipeline>,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_bind_pipeline)(
                self.handle(),
                vk::PipelineBindPoint::COMPUTE,
                pipeline.handle(),
            )
        };

        self.death_row.push(pipeline.clone());

        self
    }

    /// Binds a graphics pipeline for future draw calls.
    pub unsafe fn bind_pipeline_graphics(
        &mut self,
        pipeline: &Arc<GraphicsPipeline>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.bind_pipeline_graphics_unchecked(pipeline) })
    }

    pub unsafe fn bind_pipeline_graphics_unchecked(
        &mut self,
        pipeline: &Arc<GraphicsPipeline>,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_bind_pipeline)(
                self.handle(),
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle(),
            )
        };

        self.death_row.push(pipeline.clone());

        self
    }

    /// Binds a ray tracing pipeline for future ray tracing calls.
    pub unsafe fn bind_pipeline_ray_tracing(
        &mut self,
        pipeline: &Arc<RayTracingPipeline>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.bind_pipeline_ray_tracing_unchecked(pipeline) })
    }

    pub unsafe fn bind_pipeline_ray_tracing_unchecked(
        &mut self,
        pipeline: &Arc<RayTracingPipeline>,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_bind_pipeline)(
                self.handle(),
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.handle(),
            )
        };

        self.death_row.push(pipeline.clone());

        self
    }

    /// Binds vertex buffers for future draw calls.
    pub unsafe fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[Id<Buffer>],
        offsets: &[DeviceSize],
        sizes: &[DeviceSize],
        strides: &[DeviceSize],
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.bind_vertex_buffers_unchecked(first_binding, buffers, offsets, sizes, strides)
        })
    }

    pub unsafe fn bind_vertex_buffers_unchecked(
        &mut self,
        first_binding: u32,
        buffers: &[Id<Buffer>],
        offsets: &[DeviceSize],
        sizes: &[DeviceSize],
        strides: &[DeviceSize],
    ) -> &mut Self {
        if buffers.is_empty() {
            return self;
        }

        let buffers_vk = buffers
            .iter()
            .map(|&buffer| unsafe { self.accesses.buffer_unchecked(buffer) }.handle())
            .collect::<SmallVec<[_; 2]>>();

        let device = self.device();
        let fns = self.device().fns();

        if device.api_version() >= Version::V1_3
            || device.enabled_extensions().ext_extended_dynamic_state
            || device.enabled_extensions().ext_shader_object
        {
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
                    offsets.as_ptr(),
                    if sizes.is_empty() {
                        ptr::null()
                    } else {
                        sizes.as_ptr()
                    },
                    if strides.is_empty() {
                        ptr::null()
                    } else {
                        strides.as_ptr()
                    },
                )
            };
        } else {
            unsafe {
                (fns.v1_0.cmd_bind_vertex_buffers)(
                    self.handle(),
                    first_binding,
                    buffers_vk.len() as u32,
                    buffers_vk.as_ptr(),
                    offsets.as_ptr(),
                )
            };
        }

        self
    }

    /// Sets push constants for future dispatch or draw calls.
    pub unsafe fn push_constants(
        &mut self,
        layout: &Arc<PipelineLayout>,
        offset: u32,
        values: &(impl BufferContents + ?Sized),
    ) -> Result<&mut Self> {
        Ok(unsafe { self.push_constants_unchecked(layout, offset, values) })
    }

    pub unsafe fn push_constants_unchecked(
        &mut self,
        layout: &Arc<PipelineLayout>,
        offset: u32,
        values: &(impl BufferContents + ?Sized),
    ) -> &mut Self {
        unsafe {
            self.push_constants_unchecked_inner(
                layout,
                offset,
                <*const _>::cast(values),
                size_of_val(values) as u32,
            )
        }
    }

    unsafe fn push_constants_unchecked_inner(
        &mut self,
        layout: &Arc<PipelineLayout>,
        offset: u32,
        values: *const c_void,
        size: u32,
    ) -> &mut Self {
        if size == 0 {
            return self;
        }

        let fns = self.device().fns();
        let mut current_offset = offset;
        let mut remaining_size = size;

        for range in layout
            .push_constant_ranges_disjoint()
            .iter()
            .skip_while(|range| range.offset + range.size <= offset)
        {
            // There is a gap between ranges, but the passed `values` contain some bytes in this
            // gap.
            if range.offset > current_offset {
                std::process::abort();
            }

            // Push the minimum of the whole remaining data and the part until the end of this
            // range.
            let push_size = remaining_size.min(range.offset + range.size - current_offset);
            let push_offset = (current_offset - offset) as usize;
            debug_assert!(push_offset < size as usize);
            let push_values = unsafe { values.add(push_offset) };

            unsafe {
                (fns.v1_0.cmd_push_constants)(
                    self.handle(),
                    layout.handle(),
                    range.stages.into(),
                    current_offset,
                    push_size,
                    push_values,
                )
            };

            current_offset += push_size;
            remaining_size -= push_size;

            if remaining_size == 0 {
                break;
            }
        }

        self
    }
}
