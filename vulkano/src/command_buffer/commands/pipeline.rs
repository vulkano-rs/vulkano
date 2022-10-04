// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::{view::BufferViewAbstract, BufferAccess, TypedBufferAccess},
    command_buffer::{
        allocator::CommandBufferAllocator,
        auto::{RenderPassState, RenderPassStateType},
        synced::{Command, Resource, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, DispatchIndirectCommand, DrawIndexedIndirectCommand,
        DrawIndirectCommand, SubpassContents,
    },
    descriptor_set::{layout::DescriptorType, DescriptorBindingResources},
    device::DeviceOwned,
    format::Format,
    image::{
        view::ImageViewType, ImageAccess, ImageSubresourceRange, ImageViewAbstract, SampleCount,
    },
    pipeline::{
        graphics::{
            input_assembly::{PrimitiveTopology, PrimitiveTopologyClass},
            render_pass::PipelineRenderPassType,
            vertex_input::{VertexInputRate, VertexInputState},
        },
        DynamicState, GraphicsPipeline, PartialStateMode, Pipeline, PipelineBindPoint,
        PipelineLayout,
    },
    sampler::{Sampler, SamplerImageViewIncompatibleError},
    shader::{DescriptorRequirements, ShaderScalarType, ShaderStage},
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStages},
    DeviceSize, RequiresOneOf, VulkanObject,
};
use std::{
    borrow::Cow,
    cmp::min,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::size_of,
    ops::Range,
    sync::Arc,
};

/// # Commands to execute a bound pipeline.
///
/// Dispatch commands require a compute queue, draw commands require a graphics queue.
impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Perform a single compute operation using a compute pipeline.
    ///
    /// A compute pipeline must have been bound using
    /// [`bind_pipeline_compute`](Self::bind_pipeline_compute). Any resources used by the compute
    /// pipeline, such as descriptor sets, must have been set beforehand.
    #[inline]
    pub fn dispatch(
        &mut self,
        group_counts: [u32; 3],
    ) -> Result<&mut Self, PipelineExecutionError> {
        self.validate_dispatch(group_counts)?;

        unsafe {
            self.inner.dispatch(group_counts)?;
        }

        Ok(self)
    }

    fn validate_dispatch(&self, group_counts: [u32; 3]) -> Result<(), PipelineExecutionError> {
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdDispatch-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.compute {
            return Err(PipelineExecutionError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdDispatch-renderpass
        if self.render_pass_state.is_some() {
            return Err(PipelineExecutionError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdDispatch-None-02700
        let pipeline = match self.state().pipeline_compute() {
            Some(x) => x.as_ref(),
            None => return Err(PipelineExecutionError::PipelineNotBound),
        };

        self.validate_pipeline_descriptor_sets(pipeline, pipeline.descriptor_requirements())?;
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

        Ok(())
    }

    /// Perform multiple compute operations using a compute pipeline. One dispatch is performed for
    /// each [`DispatchIndirectCommand`] struct in `indirect_buffer`.
    ///
    /// A compute pipeline must have been bound using
    /// [`bind_pipeline_compute`](Self::bind_pipeline_compute). Any resources used by the compute
    /// pipeline, such as descriptor sets, must have been set beforehand.
    #[inline]
    pub fn dispatch_indirect<Inb>(
        &mut self,
        indirect_buffer: Arc<Inb>,
    ) -> Result<&mut Self, PipelineExecutionError>
    where
        Inb: TypedBufferAccess<Content = [DispatchIndirectCommand]> + 'static,
    {
        self.validate_dispatch_indirect(&indirect_buffer)?;

        unsafe {
            self.inner.dispatch_indirect(indirect_buffer)?;
        }

        Ok(self)
    }

    fn validate_dispatch_indirect(
        &self,
        indirect_buffer: &dyn BufferAccess,
    ) -> Result<(), PipelineExecutionError> {
        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdDispatchIndirect-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.compute {
            return Err(PipelineExecutionError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdDispatchIndirect-renderpass
        if self.render_pass_state.is_some() {
            return Err(PipelineExecutionError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdDispatchIndirect-None-02700
        let pipeline = match self.state().pipeline_compute() {
            Some(x) => x.as_ref(),
            None => return Err(PipelineExecutionError::PipelineNotBound),
        };

        self.validate_pipeline_descriptor_sets(pipeline, pipeline.descriptor_requirements())?;
        self.validate_pipeline_push_constants(pipeline.layout())?;
        self.validate_indirect_buffer(indirect_buffer)?;

        Ok(())
    }

    /// Perform a single draw operation using a graphics pipeline.
    ///
    /// The parameters specify the first vertex and the number of vertices to draw, and the first
    /// instance and number of instances. For non-instanced drawing, specify `instance_count` as 1
    /// and `first_instance` as 0.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the graphics
    /// pipeline, such as descriptor sets, vertex buffers and dynamic state, must have been set
    /// beforehand. If the bound graphics pipeline uses vertex buffers, then the provided vertex and
    /// instance ranges must be in range of the bound vertex buffers.
    #[inline]
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<&mut Self, PipelineExecutionError> {
        self.validate_draw(vertex_count, instance_count, first_vertex, first_instance)?;

        unsafe {
            self.inner
                .draw(vertex_count, instance_count, first_vertex, first_instance)?;
        }

        if let RenderPassStateType::BeginRendering(state) =
            &mut self.render_pass_state.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        Ok(self)
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
            .render_pass_state
            .as_ref()
            .ok_or(PipelineExecutionError::ForbiddenOutsideRenderPass)?;

        // VUID-vkCmdDraw-None-02700
        let pipeline = match self.state().pipeline_graphics() {
            Some(x) => x.as_ref(),
            None => return Err(PipelineExecutionError::PipelineNotBound),
        };

        self.validate_pipeline_descriptor_sets(pipeline, pipeline.descriptor_requirements())?;
        self.validate_pipeline_push_constants(pipeline.layout())?;
        self.validate_pipeline_graphics_dynamic_state(pipeline)?;
        self.validate_pipeline_graphics_render_pass(pipeline, render_pass_state)?;
        self.validate_pipeline_graphics_vertex_buffers(
            pipeline,
            Some((first_vertex, vertex_count)),
            Some((first_instance, instance_count)),
        )?;

        Ok(())
    }

    /// Perform multiple draw operations using a graphics pipeline.
    ///
    /// One draw is performed for each [`DrawIndirectCommand`] struct in `indirect_buffer`.
    /// The maximum number of draw commands in the buffer is limited by the
    /// [`max_draw_indirect_count`](crate::device::Properties::max_draw_indirect_count) limit.
    /// This limit is 1 unless the
    /// [`multi_draw_indirect`](crate::device::Features::multi_draw_indirect) feature has been
    /// enabled.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the graphics
    /// pipeline, such as descriptor sets, vertex buffers and dynamic state, must have been set
    /// beforehand. If the bound graphics pipeline uses vertex buffers, then the vertex and instance
    /// ranges of each `DrawIndirectCommand` in the indirect buffer must be in range of the bound
    /// vertex buffers.
    #[inline]
    pub fn draw_indirect<Inb>(
        &mut self,
        indirect_buffer: Arc<Inb>,
    ) -> Result<&mut Self, PipelineExecutionError>
    where
        Inb: TypedBufferAccess<Content = [DrawIndirectCommand]> + Send + Sync + 'static,
    {
        let draw_count = indirect_buffer.len() as u32;
        let stride = size_of::<DrawIndirectCommand>() as u32;
        self.validate_draw_indirect(&indirect_buffer, draw_count, stride)?;

        unsafe {
            self.inner
                .draw_indirect(indirect_buffer, draw_count, stride)?;
        }

        if let RenderPassStateType::BeginRendering(state) =
            &mut self.render_pass_state.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        Ok(self)
    }

    fn validate_draw_indirect(
        &self,
        indirect_buffer: &dyn BufferAccess,
        draw_count: u32,
        _stride: u32,
    ) -> Result<(), PipelineExecutionError> {
        // VUID-vkCmdDrawIndirect-renderpass
        let render_pass_state = self
            .render_pass_state
            .as_ref()
            .ok_or(PipelineExecutionError::ForbiddenOutsideRenderPass)?;

        // VUID-vkCmdDrawIndirect-None-02700
        let pipeline = match self.state().pipeline_graphics() {
            Some(x) => x.as_ref(),
            None => return Err(PipelineExecutionError::PipelineNotBound),
        };

        self.validate_pipeline_descriptor_sets(pipeline, pipeline.descriptor_requirements())?;
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

        Ok(())
    }

    /// Perform a single draw operation using a graphics pipeline, using an index buffer.
    ///
    /// The parameters specify the first index and the number of indices in the index buffer that
    /// should be used, and the first instance and number of instances. For non-instanced drawing,
    /// specify `instance_count` as 1 and `first_instance` as 0. The `vertex_offset` is a constant
    /// value that should be added to each index in the index buffer to produce the final vertex
    /// number to be used.
    ///
    /// An index buffer must have been bound using
    /// [`bind_index_buffer`](Self::bind_index_buffer), and the provided index range must be in
    /// range of the bound index buffer.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the graphics
    /// pipeline, such as descriptor sets, vertex buffers and dynamic state, must have been set
    /// beforehand. If the bound graphics pipeline uses vertex buffers, then the provided instance
    /// range must be in range of the bound vertex buffers. The vertex indices in the index buffer
    /// must be in range of the bound vertex buffers.
    #[inline]
    pub fn draw_indexed(
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
            self.inner.draw_indexed(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )?;
        }

        if let RenderPassStateType::BeginRendering(state) =
            &mut self.render_pass_state.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        Ok(self)
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
            .render_pass_state
            .as_ref()
            .ok_or(PipelineExecutionError::ForbiddenOutsideRenderPass)?;

        // VUID-vkCmdDrawIndexed-None-02700
        let pipeline = match self.state().pipeline_graphics() {
            Some(x) => x.as_ref(),
            None => return Err(PipelineExecutionError::PipelineNotBound),
        };

        self.validate_pipeline_descriptor_sets(pipeline, pipeline.descriptor_requirements())?;
        self.validate_pipeline_push_constants(pipeline.layout())?;
        self.validate_pipeline_graphics_dynamic_state(pipeline)?;
        self.validate_pipeline_graphics_render_pass(pipeline, render_pass_state)?;
        self.validate_pipeline_graphics_vertex_buffers(
            pipeline,
            None,
            Some((first_instance, instance_count)),
        )?;

        self.validate_index_buffer(Some((first_index, index_count)))?;

        Ok(())
    }

    /// Perform multiple draw operations using a graphics pipeline, using an index buffer.
    ///
    /// One draw is performed for each [`DrawIndexedIndirectCommand`] struct in `indirect_buffer`.
    /// The maximum number of draw commands in the buffer is limited by the
    /// [`max_draw_indirect_count`](crate::device::Properties::max_draw_indirect_count) limit.
    /// This limit is 1 unless the
    /// [`multi_draw_indirect`](crate::device::Features::multi_draw_indirect) feature has been
    /// enabled.
    ///
    /// An index buffer must have been bound using
    /// [`bind_index_buffer`](Self::bind_index_buffer), and the index ranges of each
    /// `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound index
    /// buffer.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the graphics
    /// pipeline, such as descriptor sets, vertex buffers and dynamic state, must have been set
    /// beforehand. If the bound graphics pipeline uses vertex buffers, then the instance ranges of
    /// each `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound
    /// vertex buffers.
    #[inline]
    pub fn draw_indexed_indirect<Inb>(
        &mut self,
        indirect_buffer: Arc<Inb>,
    ) -> Result<&mut Self, PipelineExecutionError>
    where
        Inb: TypedBufferAccess<Content = [DrawIndexedIndirectCommand]> + 'static,
    {
        let draw_count = indirect_buffer.len() as u32;
        let stride = size_of::<DrawIndexedIndirectCommand>() as u32;
        self.validate_draw_indexed_indirect(&indirect_buffer, draw_count, stride)?;

        unsafe {
            self.inner
                .draw_indexed_indirect(indirect_buffer, draw_count, stride)?;
        }

        if let RenderPassStateType::BeginRendering(state) =
            &mut self.render_pass_state.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        Ok(self)
    }

    fn validate_draw_indexed_indirect(
        &self,
        indirect_buffer: &dyn BufferAccess,
        draw_count: u32,
        _stride: u32,
    ) -> Result<(), PipelineExecutionError> {
        // VUID-vkCmdDrawIndexedIndirect-renderpass
        let render_pass_state = self
            .render_pass_state
            .as_ref()
            .ok_or(PipelineExecutionError::ForbiddenOutsideRenderPass)?;

        // VUID-vkCmdDrawIndexedIndirect-None-02700
        let pipeline = match self.state().pipeline_graphics() {
            Some(x) => x.as_ref(),
            None => return Err(PipelineExecutionError::PipelineNotBound),
        };

        self.validate_pipeline_descriptor_sets(pipeline, pipeline.descriptor_requirements())?;
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

        Ok(())
    }

    fn validate_index_buffer(
        &self,
        indices: Option<(u32, u32)>,
    ) -> Result<(), PipelineExecutionError> {
        let current_state = self.state();

        // VUID?
        let (index_buffer, index_type) = match current_state.index_buffer() {
            Some(x) => x,
            None => return Err(PipelineExecutionError::IndexBufferNotBound),
        };

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
        if !buffer.inner().buffer.usage().indirect_buffer {
            return Err(PipelineExecutionError::IndirectBufferMissingUsage);
        }

        // VUID-vkCmdDispatchIndirect-offset-02710
        // TODO:

        Ok(())
    }

    fn validate_pipeline_descriptor_sets<'a, Pl: Pipeline>(
        &self,
        pipeline: &Pl,
        descriptor_requirements: impl IntoIterator<Item = ((u32, u32), &'a DescriptorRequirements)>,
    ) -> Result<(), PipelineExecutionError> {
        fn validate_resources<T>(
            set_num: u32,
            binding_num: u32,
            reqs: &DescriptorRequirements,
            elements: &[Option<T>],
            mut extra_check: impl FnMut(u32, &T) -> Result<(), DescriptorResourceInvalidError>,
        ) -> Result<(), PipelineExecutionError> {
            let elements_to_check = if let Some(descriptor_count) = reqs.descriptor_count {
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

        let current_state = self.state();

        // VUID-vkCmdDispatch-None-02697
        let bindings_pipeline_layout =
            match current_state.descriptor_sets_pipeline_layout(pipeline.bind_point()) {
                Some(x) => x,
                None => return Err(PipelineExecutionError::PipelineLayoutNotCompatible),
            };

        // VUID-vkCmdDispatch-None-02697
        if !pipeline.layout().is_compatible_with(
            bindings_pipeline_layout,
            pipeline.num_used_descriptor_sets(),
        ) {
            return Err(PipelineExecutionError::PipelineLayoutNotCompatible);
        }

        for ((set_num, binding_num), reqs) in descriptor_requirements {
            let layout_binding =
                &pipeline.layout().set_layouts()[set_num as usize].bindings()[&binding_num];

            let check_buffer = |_index: u32, _buffer: &Arc<dyn BufferAccess>| Ok(());

            let check_buffer_view = |index: u32, buffer_view: &Arc<dyn BufferViewAbstract>| {
                if layout_binding.descriptor_type == DescriptorType::StorageTexelBuffer {
                    // VUID-vkCmdDispatch-OpTypeImage-06423
                    if reqs.image_format.is_none()
                        && reqs.storage_write.contains(&index)
                        && !buffer_view.format_features().storage_write_without_format
                    {
                        return Err(
                            DescriptorResourceInvalidError::StorageWriteWithoutFormatNotSupported,
                        );
                    }

                    // VUID-vkCmdDispatch-OpTypeImage-06424
                    if reqs.image_format.is_none()
                        && reqs.storage_read.contains(&index)
                        && !buffer_view.format_features().storage_read_without_format
                    {
                        return Err(
                            DescriptorResourceInvalidError::StorageReadWithoutFormatNotSupported,
                        );
                    }
                }

                Ok(())
            };

            let check_image_view_common = |index: u32, image_view: &Arc<dyn ImageViewAbstract>| {
                // VUID-vkCmdDispatch-None-02691
                if reqs.storage_image_atomic.contains(&index)
                    && !image_view.format_features().storage_image_atomic
                {
                    return Err(DescriptorResourceInvalidError::StorageImageAtomicNotSupported);
                }

                if layout_binding.descriptor_type == DescriptorType::StorageImage {
                    // VUID-vkCmdDispatch-OpTypeImage-06423
                    if reqs.image_format.is_none()
                        && reqs.storage_write.contains(&index)
                        && !image_view.format_features().storage_write_without_format
                    {
                        return Err(
                            DescriptorResourceInvalidError::StorageWriteWithoutFormatNotSupported,
                        );
                    }

                    // VUID-vkCmdDispatch-OpTypeImage-06424
                    if reqs.image_format.is_none()
                        && reqs.storage_read.contains(&index)
                        && !image_view.format_features().storage_read_without_format
                    {
                        return Err(
                            DescriptorResourceInvalidError::StorageReadWithoutFormatNotSupported,
                        );
                    }
                }

                /*
                   Instruction/Sampler/Image View Validation
                   https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap16.html#textures-input-validation
                */

                // The SPIR-V Image Format is not compatible with the image view’s format.
                if let Some(format) = reqs.image_format {
                    if image_view.format() != Some(format) {
                        return Err(DescriptorResourceInvalidError::ImageViewFormatMismatch {
                            required: format,
                            provided: image_view.format(),
                        });
                    }
                }

                // Rules for viewType
                if let Some(image_view_type) = reqs.image_view_type {
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
                if reqs.image_multisampled != (image_view.image().samples() != SampleCount::Sample1)
                {
                    return Err(
                        DescriptorResourceInvalidError::ImageViewMultisampledMismatch {
                            required: reqs.image_multisampled,
                            provided: image_view.image().samples() != SampleCount::Sample1,
                        },
                    );
                }

                // - If the Sampled Type of the OpTypeImage does not match the numeric format of the
                //   image, as shown in the SPIR-V Sampled Type column of the
                //   Interpretation of Numeric Format table.
                // - If the signedness of any read or sample operation does not match the signedness of
                //   the image’s format.
                if let Some(scalar_type) = reqs.image_scalar_type {
                    let aspects = image_view.subresource_range().aspects;
                    let view_scalar_type = ShaderScalarType::from(
                        if aspects.color || aspects.plane0 || aspects.plane1 || aspects.plane2 {
                            image_view.format().unwrap().type_color().unwrap()
                        } else if aspects.depth {
                            image_view.format().unwrap().type_depth().unwrap()
                        } else if aspects.stencil {
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
                // VUID-vkCmdDispatch-None-02703
                // VUID-vkCmdDispatch-None-02704
                if reqs.sampler_no_unnormalized_coordinates.contains(&index)
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
                if reqs.sampler_no_ycbcr_conversion.contains(&index)
                    && sampler.sampler_ycbcr_conversion().is_some()
                {
                    return Err(DescriptorResourceInvalidError::SamplerYcbcrConversionNotAllowed);
                }

                /*
                    Instruction/Sampler/Image View Validation
                    https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap16.html#textures-input-validation
                */

                // - The SPIR-V instruction is one of the OpImage*Dref* instructions and the sampler
                //   compareEnable is VK_FALSE
                // - The SPIR-V instruction is not one of the OpImage*Dref* instructions and the sampler
                //   compareEnable is VK_TRUE
                if reqs.sampler_compare.contains(&index) != sampler.compare().is_some() {
                    return Err(DescriptorResourceInvalidError::SamplerCompareMismatch {
                        required: reqs.sampler_compare.contains(&index),
                        provided: sampler.compare().is_some(),
                    });
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

                // Check sampler-image compatibility. Only done for separate samplers; combined image
                // samplers are checked when updating the descriptor set.
                if let Some(with_images) = reqs.sampler_with_images.get(&index) {
                    // If the image view isn't actually present in the resources, then just skip it.
                    // It will be caught later by check_resources.
                    let iter = with_images.iter().filter_map(|id| {
                        current_state
                            .descriptor_set(pipeline.bind_point(), id.set)
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

            let set_resources = match current_state.descriptor_set(pipeline.bind_point(), set_num) {
                Some(x) => x.resources(),
                None => return Err(PipelineExecutionError::DescriptorSetNotBound { set_num }),
            };

            let binding_resources = set_resources.binding(binding_num).unwrap();

            match binding_resources {
                DescriptorBindingResources::None(elements) => {
                    validate_resources(set_num, binding_num, reqs, elements, check_none)?;
                }
                DescriptorBindingResources::Buffer(elements) => {
                    validate_resources(set_num, binding_num, reqs, elements, check_buffer)?;
                }
                DescriptorBindingResources::BufferView(elements) => {
                    validate_resources(set_num, binding_num, reqs, elements, check_buffer_view)?;
                }
                DescriptorBindingResources::ImageView(elements) => {
                    validate_resources(set_num, binding_num, reqs, elements, check_image_view)?;
                }
                DescriptorBindingResources::ImageViewSampler(elements) => {
                    validate_resources(
                        set_num,
                        binding_num,
                        reqs,
                        elements,
                        check_image_view_sampler,
                    )?;
                }
                DescriptorBindingResources::Sampler(elements) => {
                    validate_resources(set_num, binding_num, reqs, elements, check_sampler)?;
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

        let current_state = self.state();

        // VUID-vkCmdDispatch-maintenance4-06425
        let constants_pipeline_layout = match current_state.push_constants_pipeline_layout() {
            Some(x) => x,
            None => return Err(PipelineExecutionError::PushConstantsMissing),
        };

        // VUID-vkCmdDispatch-maintenance4-06425
        if pipeline_layout.internal_object() != constants_pipeline_layout.internal_object()
            && pipeline_layout.push_constant_ranges()
                != constants_pipeline_layout.push_constant_ranges()
        {
            return Err(PipelineExecutionError::PushConstantsNotCompatible);
        }

        let set_bytes = current_state.push_constants();

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
        let current_state = self.state();

        // VUID-vkCmdDraw-commandBuffer-02701
        for dynamic_state in pipeline
            .dynamic_states()
            .filter(|(_, d)| *d)
            .map(|(s, _)| s)
        {
            match dynamic_state {
                DynamicState::BlendConstants => {
                    // VUID?
                    if current_state.blend_constants().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::ColorWriteEnable => {
                    // VUID-vkCmdDraw-attachmentCount-06667
                    let enables = if let Some(enables) = current_state.color_write_enable() {
                        enables
                    } else {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    };

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
                    if current_state.cull_mode().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthBias => {
                    // VUID?
                    if current_state.depth_bias().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthBiasEnable => {
                    // VUID-vkCmdDraw-None-04877
                    if current_state.depth_bias_enable().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthBounds => {
                    // VUID?
                    if current_state.depth_bounds().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthBoundsTestEnable => {
                    // VUID?
                    if current_state.depth_bounds_test_enable().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthCompareOp => {
                    // VUID?
                    if current_state.depth_compare_op().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthTestEnable => {
                    // VUID?
                    if current_state.depth_test_enable().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::DepthWriteEnable => {
                    // VUID?
                    if current_state.depth_write_enable().is_none() {
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
                        if current_state.discard_rectangle(num).is_none() {
                            return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                        }
                    }
                }
                DynamicState::ExclusiveScissor => todo!(),
                DynamicState::FragmentShadingRate => todo!(),
                DynamicState::FrontFace => {
                    // VUID?
                    if current_state.front_face().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::LineStipple => {
                    // VUID?
                    if current_state.line_stipple().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::LineWidth => {
                    // VUID?
                    if current_state.line_width().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::LogicOp => {
                    // VUID-vkCmdDraw-logicOp-04878
                    if current_state.logic_op().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::PatchControlPoints => {
                    // VUID-vkCmdDraw-None-04875
                    if current_state.patch_control_points().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::PrimitiveRestartEnable => {
                    // VUID-vkCmdDraw-None-04879
                    let primitive_restart_enable =
                        if let Some(enable) = current_state.primitive_restart_enable() {
                            enable
                        } else {
                            return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                        };

                    if primitive_restart_enable {
                        let topology = match pipeline.input_assembly_state().topology {
                            PartialStateMode::Fixed(topology) => topology,
                            PartialStateMode::Dynamic(_) => {
                                if let Some(topology) = current_state.primitive_topology() {
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
                                        required_for: "The bound pipeline sets `DynamicState::PrimitiveRestartEnable` and the current primitive topology is `PrimitiveTopology::*List`",
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
                                        required_for: "The bound pipeline sets `DynamicState::PrimitiveRestartEnable` and the current primitive topology is `PrimitiveTopology::PatchList`",
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
                    let topology = if let Some(topology) = current_state.primitive_topology() {
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
                    if current_state.rasterizer_discard_enable().is_none() {
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
                        if current_state.scissor(num).is_none() {
                            return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                        }
                    }
                }
                DynamicState::ScissorWithCount => {
                    // VUID-vkCmdDraw-scissorCount-03418
                    // VUID-vkCmdDraw-viewportCount-03419
                    let scissor_count = if let Some(scissors) = current_state.scissor_with_count() {
                        scissors.len() as u32
                    } else {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    };

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
                    let state = current_state.stencil_compare_mask();

                    // VUID?
                    if state.front.is_none() || state.back.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::StencilOp => {
                    let state = current_state.stencil_op();

                    // VUID?
                    if state.front.is_none() || state.back.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::StencilReference => {
                    let state = current_state.stencil_reference();

                    // VUID?
                    if state.front.is_none() || state.back.is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }
                }
                DynamicState::StencilTestEnable => {
                    // VUID?
                    if current_state.stencil_test_enable().is_none() {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    }

                    // TODO: Check if the stencil buffer is writable
                }
                DynamicState::StencilWriteMask => {
                    let state = current_state.stencil_write_mask();

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
                        if current_state.viewport(num).is_none() {
                            return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                        }
                    }
                }
                DynamicState::ViewportCoarseSampleOrder => todo!(),
                DynamicState::ViewportShadingRatePalette => todo!(),
                DynamicState::ViewportWithCount => {
                    // VUID-vkCmdDraw-viewportCount-03417
                    let viewport_count = if let Some(viewports) = current_state.viewport_with_count() {
                        viewports.len() as u32
                    } else {
                        return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                    };

                    let scissor_count = if let Some(scissor_count) =
                        pipeline.viewport_state().unwrap().count()
                    {
                        // The scissor count is fixed.
                        scissor_count
                    } else {
                        // VUID-vkCmdDraw-viewportCount-03419
                        // The scissor count is also dynamic.
                        if let Some(scissors) = current_state.scissor_with_count() {
                            scissors.len() as u32
                        } else {
                            return Err(PipelineExecutionError::DynamicStateNotSet { dynamic_state });
                        }
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
        let current_state = self.state();

        for (&binding_num, binding_desc) in &vertex_input.bindings {
            // VUID-vkCmdDraw-None-04007
            let vertex_buffer = match current_state.vertex_buffer(binding_num) {
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

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdDispatch` on the builder.
    #[inline]
    pub unsafe fn dispatch(
        &mut self,
        group_counts: [u32; 3],
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            group_counts: [u32; 3],
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "dispatch"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.dispatch(self.group_counts);
            }
        }

        let pipeline = self.current_state.pipeline_compute.as_ref().unwrap();

        let mut resources = Vec::new();
        self.add_descriptor_set_resources(
            &mut resources,
            PipelineBindPoint::Compute,
            pipeline.descriptor_requirements(),
        );

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd { group_counts }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdDispatchIndirect` on the builder.
    #[inline]
    pub unsafe fn dispatch_indirect(
        &mut self,
        indirect_buffer: Arc<dyn BufferAccess>,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            indirect_buffer: Arc<dyn BufferAccess>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "dispatch_indirect"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.dispatch_indirect(self.indirect_buffer.as_ref());
            }
        }

        let pipeline = self.current_state.pipeline_compute.as_ref().unwrap();

        let mut resources = Vec::new();
        self.add_descriptor_set_resources(
            &mut resources,
            PipelineBindPoint::Compute,
            pipeline.descriptor_requirements(),
        );
        self.add_indirect_buffer_resources(&mut resources, &indirect_buffer);

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd { indirect_buffer }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdDraw` on the builder.
    #[inline]
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "draw"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.draw(
                    self.vertex_count,
                    self.instance_count,
                    self.first_vertex,
                    self.first_instance,
                );
            }
        }

        let pipeline = self.current_state.pipeline_graphics.as_ref().unwrap();

        let mut resources = Vec::new();
        self.add_descriptor_set_resources(
            &mut resources,
            PipelineBindPoint::Graphics,
            pipeline.descriptor_requirements(),
        );
        self.add_vertex_buffer_resources(&mut resources, pipeline.vertex_input_state());

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdDrawIndexed` on the builder.
    #[inline]
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            index_count: u32,
            instance_count: u32,
            first_index: u32,
            vertex_offset: i32,
            first_instance: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "draw_indexed"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.draw_indexed(
                    self.index_count,
                    self.instance_count,
                    self.first_index,
                    self.vertex_offset,
                    self.first_instance,
                );
            }
        }

        let pipeline = self.current_state.pipeline_graphics.as_ref().unwrap();

        let mut resources = Vec::new();
        self.add_descriptor_set_resources(
            &mut resources,
            PipelineBindPoint::Graphics,
            pipeline.descriptor_requirements(),
        );
        self.add_vertex_buffer_resources(&mut resources, pipeline.vertex_input_state());
        self.add_index_buffer_resources(&mut resources);

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdDrawIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indirect(
        &mut self,
        indirect_buffer: Arc<dyn BufferAccess>,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            indirect_buffer: Arc<dyn BufferAccess>,
            draw_count: u32,
            stride: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "draw_indirect"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.draw_indirect(self.indirect_buffer.as_ref(), self.draw_count, self.stride);
            }
        }

        let pipeline = self.current_state.pipeline_graphics.as_ref().unwrap();

        let mut resources = Vec::new();
        self.add_descriptor_set_resources(
            &mut resources,
            PipelineBindPoint::Graphics,
            pipeline.descriptor_requirements(),
        );
        self.add_vertex_buffer_resources(&mut resources, pipeline.vertex_input_state());
        self.add_indirect_buffer_resources(&mut resources, &indirect_buffer);

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            indirect_buffer,
            draw_count,
            stride,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdDrawIndexedIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: Arc<dyn BufferAccess>,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            indirect_buffer: Arc<dyn BufferAccess>,
            draw_count: u32,
            stride: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "draw_indexed_indirect"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.draw_indexed_indirect(
                    self.indirect_buffer.as_ref(),
                    self.draw_count,
                    self.stride,
                );
            }
        }

        let pipeline = self.current_state.pipeline_graphics.as_ref().unwrap();

        let mut resources = Vec::new();
        self.add_descriptor_set_resources(
            &mut resources,
            PipelineBindPoint::Graphics,
            pipeline.descriptor_requirements(),
        );
        self.add_vertex_buffer_resources(&mut resources, pipeline.vertex_input_state());
        self.add_index_buffer_resources(&mut resources);
        self.add_indirect_buffer_resources(&mut resources, &indirect_buffer);

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            indirect_buffer,
            draw_count,
            stride,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    fn add_descriptor_set_resources<'a>(
        &self,
        resources: &mut Vec<(Cow<'static, str>, Resource)>,
        pipeline_bind_point: PipelineBindPoint,
        descriptor_requirements: impl IntoIterator<Item = ((u32, u32), &'a DescriptorRequirements)>,
    ) {
        let state = match self.current_state.descriptor_sets.get(&pipeline_bind_point) {
            Some(x) => x,
            None => return,
        };

        for ((set, binding), reqs) in descriptor_requirements {
            // TODO: Can things be refactored so that the pipeline layout isn't needed at all?
            let descriptor_type = state.pipeline_layout.set_layouts()[set as usize].bindings()
                [&binding]
                .descriptor_type;

            // FIXME: This is tricky. Since we read from the input attachment
            // and this input attachment is being written in an earlier pass,
            // vulkano will think that it needs to put a pipeline barrier and will
            // return a `Conflict` error. For now as a work-around we simply ignore
            // input attachments.
            if descriptor_type == DescriptorType::InputAttachment {
                continue;
            }

            // TODO: Maybe include this on DescriptorRequirements?
            let access = PipelineMemoryAccess {
                stages: reqs.stages.into(),
                access: match descriptor_type {
                    DescriptorType::Sampler => continue,
                    DescriptorType::CombinedImageSampler
                    | DescriptorType::SampledImage
                    | DescriptorType::StorageImage
                    | DescriptorType::UniformTexelBuffer
                    | DescriptorType::StorageTexelBuffer
                    | DescriptorType::StorageBuffer
                    | DescriptorType::StorageBufferDynamic => AccessFlags {
                        shader_read: true,
                        shader_write: false,
                        ..AccessFlags::empty()
                    },
                    DescriptorType::InputAttachment => AccessFlags {
                        input_attachment_read: true,
                        ..AccessFlags::empty()
                    },
                    DescriptorType::UniformBuffer | DescriptorType::UniformBufferDynamic => {
                        AccessFlags {
                            uniform_read: true,
                            ..AccessFlags::empty()
                        }
                    }
                },
                exclusive: false,
            };

            let access = (0..).map(|index| {
                let mut access = access;
                let mutable = reqs.storage_write.contains(&index);
                access.access.shader_write = mutable;
                access.exclusive = mutable;
                access
            });

            let buffer_resource = move |(buffer, range, memory): (
                Arc<dyn BufferAccess>,
                Range<DeviceSize>,
                PipelineMemoryAccess,
            )| {
                (
                    format!("Buffer bound to set {} descriptor {}", set, binding).into(),
                    Resource::Buffer {
                        buffer,
                        range,
                        memory,
                    },
                )
            };
            let image_resource = move |(image, subresource_range, memory): (
                Arc<dyn ImageAccess>,
                ImageSubresourceRange,
                PipelineMemoryAccess,
            )| {
                let layout = image
                    .descriptor_layouts()
                    .expect("descriptor_layouts must return Some when used in an image view")
                    .layout_for(descriptor_type);
                (
                    format!("Image bound to set {} descriptor {}", set, binding).into(),
                    Resource::Image {
                        image,
                        subresource_range,
                        memory,
                        start_layout: layout,
                        end_layout: layout,
                    },
                )
            };

            match state.descriptor_sets[&set]
                .resources()
                .binding(binding)
                .unwrap()
            {
                DescriptorBindingResources::None(_) => continue,
                DescriptorBindingResources::Buffer(elements) => {
                    resources.extend(
                        access
                            .zip(elements)
                            .filter_map(|(access, element)| {
                                element.as_ref().map(|buffer| {
                                    (
                                        buffer.clone(),
                                        0..buffer.size(), // TODO:
                                        access,
                                    )
                                })
                            })
                            .map(buffer_resource),
                    );
                }
                DescriptorBindingResources::BufferView(elements) => {
                    resources.extend(
                        access
                            .zip(elements)
                            .filter_map(|(access, element)| {
                                element.as_ref().map(|buffer_view| {
                                    (buffer_view.buffer(), buffer_view.range(), access)
                                })
                            })
                            .map(buffer_resource),
                    );
                }
                DescriptorBindingResources::ImageView(elements) => {
                    resources.extend(
                        access
                            .zip(elements)
                            .filter_map(|(access, element)| {
                                element.as_ref().map(|image_view| {
                                    (
                                        image_view.image(),
                                        image_view.subresource_range().clone(),
                                        access,
                                    )
                                })
                            })
                            .map(image_resource),
                    );
                }
                DescriptorBindingResources::ImageViewSampler(elements) => {
                    resources.extend(
                        access
                            .zip(elements)
                            .filter_map(|(access, element)| {
                                element.as_ref().map(|(image_view, _)| {
                                    (
                                        image_view.image(),
                                        image_view.subresource_range().clone(),
                                        access,
                                    )
                                })
                            })
                            .map(image_resource),
                    );
                }
                DescriptorBindingResources::Sampler(_) => (),
            }
        }
    }

    fn add_vertex_buffer_resources(
        &self,
        resources: &mut Vec<(Cow<'static, str>, Resource)>,
        vertex_input: &VertexInputState,
    ) {
        resources.extend(vertex_input.bindings.iter().map(|(&binding_num, _)| {
            let vertex_buffer = &self.current_state.vertex_buffers[&binding_num];
            (
                format!("Vertex buffer binding {}", binding_num).into(),
                Resource::Buffer {
                    buffer: vertex_buffer.clone(),
                    range: 0..vertex_buffer.size(), // TODO:
                    memory: PipelineMemoryAccess {
                        stages: PipelineStages {
                            vertex_input: true,
                            ..PipelineStages::empty()
                        },
                        access: AccessFlags {
                            vertex_attribute_read: true,
                            ..AccessFlags::empty()
                        },
                        exclusive: false,
                    },
                },
            )
        }));
    }

    fn add_index_buffer_resources(&self, resources: &mut Vec<(Cow<'static, str>, Resource)>) {
        let index_buffer = &self.current_state.index_buffer.as_ref().unwrap().0;
        resources.push((
            "index buffer".into(),
            Resource::Buffer {
                buffer: index_buffer.clone(),
                range: 0..index_buffer.size(), // TODO:
                memory: PipelineMemoryAccess {
                    stages: PipelineStages {
                        vertex_input: true,
                        ..PipelineStages::empty()
                    },
                    access: AccessFlags {
                        index_read: true,
                        ..AccessFlags::empty()
                    },
                    exclusive: false,
                },
            },
        ));
    }

    fn add_indirect_buffer_resources(
        &self,
        resources: &mut Vec<(Cow<'static, str>, Resource)>,
        indirect_buffer: &Arc<dyn BufferAccess>,
    ) {
        resources.push((
            "indirect buffer".into(),
            Resource::Buffer {
                buffer: indirect_buffer.clone(),
                range: 0..indirect_buffer.size(), // TODO:
                memory: PipelineMemoryAccess {
                    stages: PipelineStages {
                        draw_indirect: true,
                        ..PipelineStages::empty()
                    }, // TODO: is draw_indirect correct for dispatch too?
                    access: AccessFlags {
                        indirect_command_read: true,
                        ..AccessFlags::empty()
                    },
                    exclusive: false,
                },
            },
        ));
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdDispatch` on the builder.
    #[inline]
    pub unsafe fn dispatch(&mut self, group_counts: [u32; 3]) {
        debug_assert!({
            let max_group_counts = self
                .device
                .physical_device()
                .properties()
                .max_compute_work_group_count;
            group_counts[0] <= max_group_counts[0]
                && group_counts[1] <= max_group_counts[1]
                && group_counts[2] <= max_group_counts[2]
        });

        let fns = self.device.fns();
        (fns.v1_0.cmd_dispatch)(
            self.handle,
            group_counts[0],
            group_counts[1],
            group_counts[2],
        );
    }

    /// Calls `vkCmdDispatchIndirect` on the builder.
    #[inline]
    pub unsafe fn dispatch_indirect(&mut self, buffer: &dyn BufferAccess) {
        let fns = self.device.fns();

        let inner = buffer.inner();
        debug_assert!(inner.offset < inner.buffer.size());
        debug_assert!(inner.buffer.usage().indirect_buffer);
        debug_assert_eq!(inner.offset % 4, 0);

        (fns.v1_0.cmd_dispatch_indirect)(self.handle, inner.buffer.internal_object(), inner.offset);
    }

    /// Calls `vkCmdDraw` on the builder.
    #[inline]
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_draw)(
            self.handle,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );
    }

    /// Calls `vkCmdDrawIndexed` on the builder.
    #[inline]
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_draw_indexed)(
            self.handle,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
    }

    /// Calls `vkCmdDrawIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indirect(
        &mut self,
        buffer: &dyn BufferAccess,
        draw_count: u32,
        stride: u32,
    ) {
        let fns = self.device.fns();

        debug_assert!(
            draw_count == 0
                || ((stride % 4) == 0)
                    && stride as usize >= size_of::<ash::vk::DrawIndirectCommand>()
        );

        let inner = buffer.inner();
        debug_assert!(inner.offset < inner.buffer.size());
        debug_assert!(inner.buffer.usage().indirect_buffer);

        (fns.v1_0.cmd_draw_indirect)(
            self.handle,
            inner.buffer.internal_object(),
            inner.offset,
            draw_count,
            stride,
        );
    }

    /// Calls `vkCmdDrawIndexedIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &dyn BufferAccess,
        draw_count: u32,
        stride: u32,
    ) {
        let fns = self.device.fns();

        let inner = buffer.inner();
        debug_assert!(inner.offset < inner.buffer.size());
        debug_assert!(inner.buffer.usage().indirect_buffer);

        (fns.v1_0.cmd_draw_indexed_indirect)(
            self.handle,
            inner.buffer.internal_object(),
            inner.offset,
            draw_count,
            stride,
        );
    }
}

/// Error that can happen when recording a bound pipeline execution command.
#[derive(Debug, Clone)]
pub enum PipelineExecutionError {
    SyncCommandBufferBuilderError(SyncCommandBufferBuilderError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The resource bound to a descriptor set binding at a particular index is not compatible
    /// with the requirements of the pipeline and shaders.
    DescriptorResourceInvalid {
        set_num: u32,
        binding_num: u32,
        index: u32,
        error: DescriptorResourceInvalidError,
    },

    /// The pipeline layout requires a descriptor set bound to a set number, but none was bound.
    DescriptorSetNotBound {
        set_num: u32,
    },

    /// The bound pipeline uses a dynamic color write enable setting, but the number of provided
    /// enable values is less than the number of attachments in the current render subpass.
    DynamicColorWriteEnableNotEnoughValues {
        color_write_enable_count: u32,
        attachment_count: u32,
    },

    /// The bound pipeline uses a dynamic primitive topology, but the provided topology is of a
    /// different topology class than what the pipeline requires.
    DynamicPrimitiveTopologyClassMismatch {
        provided_class: PrimitiveTopologyClass,
        required_class: PrimitiveTopologyClass,
    },

    /// The bound pipeline uses a dynamic primitive topology, but the provided topology is not
    /// compatible with the shader stages in the pipeline.
    DynamicPrimitiveTopologyInvalid {
        topology: PrimitiveTopology,
    },

    /// The pipeline requires a particular dynamic state, but this state was not set.
    DynamicStateNotSet {
        dynamic_state: DynamicState,
    },

    /// The bound pipeline uses a dynamic scissor and/or viewport count, but the scissor count
    /// does not match the viewport count.
    DynamicViewportScissorCountMismatch {
        viewport_count: u32,
        scissor_count: u32,
    },

    /// Operation forbidden inside a render pass.
    ForbiddenInsideRenderPass,

    /// Operation forbidden outside a render pass.
    ForbiddenOutsideRenderPass,

    /// Operation forbidden inside a render subpass with the specified contents.
    ForbiddenWithSubpassContents {
        subpass_contents: SubpassContents,
    },

    /// An indexed draw command was recorded, but no index buffer was bound.
    IndexBufferNotBound,

    /// The highest index to be drawn exceeds the available number of indices in the bound index buffer.
    IndexBufferRangeOutOfBounds {
        highest_index: u32,
        max_index_count: u32,
    },

    /// The `indirect_buffer` usage was not enabled on the indirect buffer.
    IndirectBufferMissingUsage,

    /// The `max_compute_work_group_count` limit has been exceeded.
    MaxComputeWorkGroupCountExceeded {
        requested: [u32; 3],
        max: [u32; 3],
    },

    /// The `max_draw_indirect_count` limit has been exceeded.
    MaxDrawIndirectCountExceeded {
        provided: u32,
        max: u32,
    },

    /// The `max_multiview_instance_index` limit has been exceeded.
    MaxMultiviewInstanceIndexExceeded {
        highest_instance: u64,
        max: u32,
    },

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,

    /// The color attachment count in the bound pipeline does not match the count of the current
    /// render pass.
    PipelineColorAttachmentCountMismatch {
        pipeline_count: u32,
        required_count: u32,
    },

    /// The format of a color attachment in the bound pipeline does not match the format of the
    /// corresponding color attachment in the current render pass.
    PipelineColorAttachmentFormatMismatch {
        color_attachment_index: u32,
        pipeline_format: Option<Format>,
        required_format: Format,
    },

    /// The format of the depth attachment in the bound pipeline does not match the format of the
    /// depth attachment in the current render pass.
    PipelineDepthAttachmentFormatMismatch {
        pipeline_format: Option<Format>,
        required_format: Format,
    },

    /// The bound pipeline is not compatible with the layout used to bind the descriptor sets.
    PipelineLayoutNotCompatible,

    /// No pipeline was bound to the bind point used by the operation.
    PipelineNotBound,

    /// The bound graphics pipeline uses a render pass that is not compatible with the currently
    /// active render pass.
    PipelineRenderPassNotCompatible,

    /// The bound graphics pipeline uses a render pass of a different type than the currently
    /// active render pass.
    PipelineRenderPassTypeMismatch,

    /// The bound graphics pipeline uses a render subpass index that doesn't match the currently
    /// active subpass index.
    PipelineSubpassMismatch {
        pipeline: u32,
        current: u32,
    },

    /// The format of the stencil attachment in the bound pipeline does not match the format of the
    /// stencil attachment in the current render pass.
    PipelineStencilAttachmentFormatMismatch {
        pipeline_format: Option<Format>,
        required_format: Format,
    },

    /// The view mask of the bound pipeline does not match the view mask of the current render pass.
    PipelineViewMaskMismatch {
        pipeline_view_mask: u32,
        required_view_mask: u32,
    },

    /// The push constants are not compatible with the pipeline layout.
    PushConstantsNotCompatible,

    /// Not all push constants used by the pipeline have been set.
    PushConstantsMissing,

    /// The bound graphics pipeline requires a vertex buffer bound to a binding number, but none
    /// was bound.
    VertexBufferNotBound {
        binding_num: u32,
    },

    /// The number of instances to be drawn exceeds the available number of indices in the
    /// bound vertex buffers used by the pipeline.
    VertexBufferInstanceRangeOutOfBounds {
        instances_needed: u64,
        instances_in_buffers: u64,
    },

    /// The number of vertices to be drawn exceeds the lowest available number of vertices in the
    /// bound vertex buffers used by the pipeline.
    VertexBufferVertexRangeOutOfBounds {
        vertices_needed: u64,
        vertices_in_buffers: u64,
    },
}

impl Error for PipelineExecutionError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SyncCommandBufferBuilderError(err) => Some(err),
            Self::DescriptorResourceInvalid { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl Display for PipelineExecutionError {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::SyncCommandBufferBuilderError(_) => write!(f, "a SyncCommandBufferBuilderError"),

            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),

            Self::DescriptorResourceInvalid { set_num, binding_num, index, .. } => write!(
                f,
                "the resource bound to descriptor set {} binding {} at index {} is not compatible with the requirements of the pipeline and shaders",
                set_num, binding_num, index,
            ),
            Self::DescriptorSetNotBound {
                set_num,
            } => write!(
                f,
                "the pipeline layout requires a descriptor set bound to set number {}, but none was bound",
                set_num,
            ),
            Self::DynamicColorWriteEnableNotEnoughValues {
                color_write_enable_count,
                attachment_count,
            } => write!(
                f,
                "the bound pipeline uses a dynamic color write enable setting, but the number of provided enable values ({}) is less than the number of attachments in the current render subpass ({})",
                color_write_enable_count,
                attachment_count,
            ),
            Self::DynamicPrimitiveTopologyClassMismatch {
                provided_class,
                required_class,
            } => write!(
                f,
                "The bound pipeline uses a dynamic primitive topology, but the provided topology is of a different topology class ({:?}) than what the pipeline requires ({:?})",
                provided_class, required_class,
            ),
            Self::DynamicPrimitiveTopologyInvalid {
                topology,
            } => write!(
                f,
                "the bound pipeline uses a dynamic primitive topology, but the provided topology ({:?}) is not compatible with the shader stages in the pipeline",
                topology,
            ),
            Self::DynamicStateNotSet { dynamic_state } => write!(
                f,
                "the pipeline requires the dynamic state {:?}, but this state was not set",
                dynamic_state,
            ),
            Self::DynamicViewportScissorCountMismatch {
                viewport_count,
                scissor_count,
            } => write!(
                f,
                "the bound pipeline uses a dynamic scissor and/or viewport count, but the scissor count ({}) does not match the viewport count ({})",
                scissor_count,
                viewport_count,
            ),
            Self::ForbiddenInsideRenderPass => write!(
                f,
                "operation forbidden inside a render pass",
            ),
            Self::ForbiddenOutsideRenderPass => write!(
                f,
                "operation forbidden outside a render pass",
            ),
            Self::ForbiddenWithSubpassContents { subpass_contents } => write!(
                f,
                "operation forbidden inside a render subpass with contents {:?}",
                subpass_contents,
            ),
            Self::IndexBufferNotBound => write!(
                f,
                "an indexed draw command was recorded, but no index buffer was bound",
            ),
            Self::IndexBufferRangeOutOfBounds {
                highest_index,
                max_index_count,
            } => write!(
                f,
                "the highest index to be drawn ({}) exceeds the available number of indices in the bound index buffer ({})",
                highest_index,
                max_index_count,
            ),
            Self::IndirectBufferMissingUsage => write!(
                f,
                "the `indirect_buffer` usage was not enabled on the indirect buffer",
            ),
            Self::MaxComputeWorkGroupCountExceeded { .. } => write!(
                f,
                "the `max_compute_work_group_count` limit has been exceeded",
            ),
            Self::MaxDrawIndirectCountExceeded { .. } => write!(
                f,
                "the `max_draw_indirect_count` limit has been exceeded",
            ),
            Self::MaxMultiviewInstanceIndexExceeded { .. } => write!(
                f,
                "the `max_multiview_instance_index` limit has been exceeded",
            ),
            Self::NotSupportedByQueueFamily => write!(
                f,
                "the queue family doesn't allow this operation",
            ),
            Self::PipelineColorAttachmentCountMismatch {
                pipeline_count,
                required_count,
            } => write!(
                f,
                "the color attachment count in the bound pipeline ({}) does not match the count of the current render pass ({})",
                pipeline_count, required_count,
            ),
            Self::PipelineColorAttachmentFormatMismatch {
                color_attachment_index,
                pipeline_format,
                required_format,
            } => write!(
                f,
                "the format of color attachment {} in the bound pipeline ({:?}) does not match the format of the corresponding color attachment in the current render pass ({:?})",
                color_attachment_index, pipeline_format, required_format,
            ),
            Self::PipelineDepthAttachmentFormatMismatch {
                pipeline_format,
                required_format,
            } => write!(
                f,
                "the format of the depth attachment in the bound pipeline ({:?}) does not match the format of the depth attachment in the current render pass ({:?})",
                pipeline_format, required_format,
            ),
            Self::PipelineLayoutNotCompatible => write!(
                f,
                "the bound pipeline is not compatible with the layout used to bind the descriptor sets",
            ),
            Self::PipelineNotBound => write!(
                f,
                "no pipeline was bound to the bind point used by the operation",
            ),
            Self::PipelineRenderPassNotCompatible => write!(
                f,
                "the bound graphics pipeline uses a render pass that is not compatible with the currently active render pass",
            ),
            Self::PipelineRenderPassTypeMismatch => write!(
                f,
                "the bound graphics pipeline uses a render pass of a different type than the currently active render pass",
            ),
            Self::PipelineSubpassMismatch {
                pipeline,
                current,
            } => write!(
                f,
                "the bound graphics pipeline uses a render subpass index ({}) that doesn't match the currently active subpass index ({})",
                pipeline,
                current,
            ),
            Self::PipelineStencilAttachmentFormatMismatch {
                pipeline_format,
                required_format,
            } => write!(
                f,
                "the format of the stencil attachment in the bound pipeline ({:?}) does not match the format of the stencil attachment in the current render pass ({:?})",
                pipeline_format, required_format,
            ),
            Self::PipelineViewMaskMismatch {
                pipeline_view_mask,
                required_view_mask,
            } => write!(
                f,
                "the view mask of the bound pipeline ({}) does not match the view mask of the current render pass ({})",
                pipeline_view_mask, required_view_mask,
            ),
            Self::PushConstantsNotCompatible => write!(
                f,
                "the push constants are not compatible with the pipeline layout",
            ),
            Self::PushConstantsMissing => write!(
                f,
                "not all push constants used by the pipeline have been set",
            ),
            Self::VertexBufferNotBound {
                binding_num,
            } => write!(
                f,
                "the bound graphics pipeline requires a vertex buffer bound to binding number {}, but none was bound",
                binding_num,
            ),
            Self::VertexBufferInstanceRangeOutOfBounds {
                instances_needed,
                instances_in_buffers,
            } => write!(
                f,
                "the number of instances to be drawn ({}) exceeds the available number of instances in the bound vertex buffers ({}) used by the pipeline",
                instances_needed, instances_in_buffers,
            ),
            Self::VertexBufferVertexRangeOutOfBounds {
                vertices_needed,
                vertices_in_buffers,
            } => write!(
                f,
                "the number of vertices to be drawn ({}) exceeds the available number of vertices in the bound vertex buffers ({}) used by the pipeline",
                vertices_needed, vertices_in_buffers,
            ),
        }
    }
}

impl From<SyncCommandBufferBuilderError> for PipelineExecutionError {
    #[inline]
    fn from(err: SyncCommandBufferBuilderError) -> Self {
        Self::SyncCommandBufferBuilderError(err)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DescriptorResourceInvalidError {
    ImageViewFormatMismatch {
        required: Format,
        provided: Option<Format>,
    },
    ImageViewMultisampledMismatch {
        required: bool,
        provided: bool,
    },
    ImageViewScalarTypeMismatch {
        required: ShaderScalarType,
        provided: ShaderScalarType,
    },
    ImageViewTypeMismatch {
        required: ImageViewType,
        provided: ImageViewType,
    },
    Missing,
    SamplerCompareMismatch {
        required: bool,
        provided: bool,
    },
    SamplerImageViewIncompatible {
        image_view_set_num: u32,
        image_view_binding_num: u32,
        image_view_index: u32,
        error: SamplerImageViewIncompatibleError,
    },
    SamplerUnnormalizedCoordinatesNotAllowed,
    SamplerYcbcrConversionNotAllowed,
    StorageImageAtomicNotSupported,
    StorageReadWithoutFormatNotSupported,
    StorageWriteWithoutFormatNotSupported,
}

impl Error for DescriptorResourceInvalidError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SamplerImageViewIncompatible { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl Display for DescriptorResourceInvalidError {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::ImageViewFormatMismatch { provided, required } => write!(
                f,
                "the format of the bound image view ({:?}) does not match what the pipeline requires ({:?})",
                provided, required
            ),
            Self::ImageViewMultisampledMismatch { provided, required } => write!(
                f,
                "the multisampling of the bound image ({}) does not match what the pipeline requires ({})",
                provided, required,
            ),
            Self::ImageViewScalarTypeMismatch { provided, required } => write!(
                f,
                "the scalar type of the format and aspect of the bound image view ({:?}) does not match what the pipeline requires ({:?})",
                provided, required,
            ),
            Self::ImageViewTypeMismatch { provided, required } => write!(
                f,
                "the image view type of the bound image view ({:?}) does not match what the pipeline requires ({:?})",
                provided, required,
            ),
            Self::Missing => write!(
                f,
                "no resource was bound",
            ),
            Self::SamplerImageViewIncompatible { .. } => write!(
                f,
                "the bound sampler samples an image view that is not compatible with that sampler",
            ),
            Self::SamplerCompareMismatch { provided, required } => write!(
                f,
                "the depth comparison state of the bound sampler ({}) does not match what the pipeline requires ({})",
                provided, required,
            ),
            Self::SamplerUnnormalizedCoordinatesNotAllowed => write!(
                f,
                "the bound sampler is required to have unnormalized coordinates disabled",
            ),
            Self::SamplerYcbcrConversionNotAllowed => write!(
                f,
                "the bound sampler is required to have no attached sampler YCbCr conversion",
            ),
            Self::StorageImageAtomicNotSupported => write!(
                f,
                "the bound image view does not support the `storage_image_atomic` format feature",
            ),
            Self::StorageReadWithoutFormatNotSupported => write!(
                f,
                "the bound image view or buffer view does not support the `storage_read_without_format` format feature",
            ),
            Self::StorageWriteWithoutFormatNotSupported => write!(
                f,
                "the bound image view or buffer view does not support the `storage_write_without_format` format feature",
            ),
        }
    }
}
