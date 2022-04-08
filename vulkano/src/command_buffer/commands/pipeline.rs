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
        synced::{
            Command, CommandBufferState, Resource, SyncCommandBufferBuilder,
            SyncCommandBufferBuilderError,
        },
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, DispatchError,
        DispatchIndirectCommand, DispatchIndirectError, DrawError, DrawIndexedError,
        DrawIndexedIndirectCommand, DrawIndexedIndirectError, DrawIndirectCommand,
        DrawIndirectError,
    },
    descriptor_set::{layout::DescriptorType, DescriptorBindingResources},
    device::{Device, DeviceOwned},
    format::Format,
    image::{
        view::ImageViewType, ImageAccess, ImageSubresourceRange, ImageViewAbstract, SampleCount,
    },
    pipeline::{
        graphics::{
            input_assembly::PrimitiveTopology,
            vertex_input::{VertexInputRate, VertexInputState},
        },
        ComputePipeline, DynamicState, GraphicsPipeline, PartialStateMode, Pipeline,
        PipelineBindPoint, PipelineLayout,
    },
    sampler::{Sampler, SamplerImageViewIncompatibleError},
    shader::{DescriptorRequirements, ShaderScalarType, ShaderStage},
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStages},
    DeviceSize, VulkanObject,
};
use std::{borrow::Cow, error, fmt, mem::size_of, sync::Arc};

/// # Commands to execute a bound pipeline.
///
/// Dispatch commands require a compute queue, draw commands require a graphics queue.
impl<L, P> AutoCommandBufferBuilder<L, P> {
    /// Perform a single compute operation using a compute pipeline.
    ///
    /// A compute pipeline must have been bound using
    /// [`bind_pipeline_compute`](Self::bind_pipeline_compute). Any resources used by the compute
    /// pipeline, such as descriptor sets, must have been set beforehand.
    #[inline]
    pub fn dispatch(&mut self, group_counts: [u32; 3]) -> Result<&mut Self, DispatchError> {
        if !self.queue_family().supports_compute() {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        let pipeline = check_pipeline_compute(self.state())?;
        self.ensure_outside_render_pass()?;
        check_descriptor_sets_validity(self.state(), pipeline, pipeline.descriptor_requirements())?;
        check_push_constants_validity(self.state(), pipeline.layout())?;
        check_dispatch(self.device(), group_counts)?;

        unsafe {
            self.inner.dispatch(group_counts)?;
        }

        Ok(self)
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
    ) -> Result<&mut Self, DispatchIndirectError>
    where
        Inb: TypedBufferAccess<Content = [DispatchIndirectCommand]> + 'static,
    {
        if !self.queue_family().supports_compute() {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        let pipeline = check_pipeline_compute(self.state())?;
        self.ensure_outside_render_pass()?;
        check_descriptor_sets_validity(self.state(), pipeline, pipeline.descriptor_requirements())?;
        check_push_constants_validity(self.state(), pipeline.layout())?;
        check_indirect_buffer(self.device(), indirect_buffer.as_ref())?;

        unsafe {
            self.inner.dispatch_indirect(indirect_buffer)?;
        }

        Ok(self)
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
    ) -> Result<&mut Self, DrawError> {
        let pipeline = check_pipeline_graphics(self.state())?;
        self.ensure_inside_render_pass_inline(pipeline)?;
        check_dynamic_state_validity(self.state(), pipeline)?;
        check_descriptor_sets_validity(self.state(), pipeline, pipeline.descriptor_requirements())?;
        check_push_constants_validity(self.state(), pipeline.layout())?;
        check_vertex_buffers(
            self.state(),
            pipeline,
            Some((first_vertex, vertex_count)),
            Some((first_instance, instance_count)),
        )?;

        unsafe {
            self.inner
                .draw(vertex_count, instance_count, first_vertex, first_instance)?;
        }

        Ok(self)
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
    ) -> Result<&mut Self, DrawIndirectError>
    where
        Inb: TypedBufferAccess<Content = [DrawIndirectCommand]> + Send + Sync + 'static,
    {
        let pipeline = check_pipeline_graphics(self.state())?;
        self.ensure_inside_render_pass_inline(pipeline)?;
        check_dynamic_state_validity(self.state(), pipeline)?;
        check_descriptor_sets_validity(self.state(), pipeline, pipeline.descriptor_requirements())?;
        check_push_constants_validity(self.state(), pipeline.layout())?;
        check_vertex_buffers(self.state(), pipeline, None, None)?;
        check_indirect_buffer(self.device(), indirect_buffer.as_ref())?;

        let draw_count = indirect_buffer.len() as u32;
        let limit = self
            .device()
            .physical_device()
            .properties()
            .max_draw_indirect_count;

        if draw_count > limit {
            return Err(
                CheckIndirectBufferError::MaxDrawIndirectCountLimitExceeded {
                    limit,
                    requested: draw_count,
                }
                .into(),
            );
        }

        unsafe {
            self.inner.draw_indirect(
                indirect_buffer,
                draw_count,
                size_of::<DrawIndirectCommand>() as u32,
            )?;
        }

        Ok(self)
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
    ) -> Result<&mut Self, DrawIndexedError> {
        // TODO: how to handle an index out of range of the vertex buffers?
        let pipeline = check_pipeline_graphics(self.state())?;
        self.ensure_inside_render_pass_inline(pipeline)?;
        check_dynamic_state_validity(self.state(), pipeline)?;
        check_descriptor_sets_validity(self.state(), pipeline, pipeline.descriptor_requirements())?;
        check_push_constants_validity(self.state(), pipeline.layout())?;
        check_vertex_buffers(
            self.state(),
            pipeline,
            None,
            Some((first_instance, instance_count)),
        )?;
        check_index_buffer(self.state(), Some((first_index, index_count)))?;

        unsafe {
            self.inner.draw_indexed(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )?;
        }

        Ok(self)
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
    ) -> Result<&mut Self, DrawIndexedIndirectError>
    where
        Inb: TypedBufferAccess<Content = [DrawIndexedIndirectCommand]> + 'static,
    {
        let pipeline = check_pipeline_graphics(self.state())?;
        self.ensure_inside_render_pass_inline(pipeline)?;
        check_dynamic_state_validity(self.state(), pipeline)?;
        check_descriptor_sets_validity(self.state(), pipeline, pipeline.descriptor_requirements())?;
        check_push_constants_validity(self.state(), pipeline.layout())?;
        check_vertex_buffers(self.state(), pipeline, None, None)?;
        check_index_buffer(self.state(), None)?;
        check_indirect_buffer(self.device(), indirect_buffer.as_ref())?;

        let draw_count = indirect_buffer.len() as u32;
        let limit = self
            .device()
            .physical_device()
            .properties()
            .max_draw_indirect_count;

        if draw_count > limit {
            return Err(
                CheckIndirectBufferError::MaxDrawIndirectCountLimitExceeded {
                    limit,
                    requested: draw_count,
                }
                .into(),
            );
        }

        unsafe {
            self.inner.draw_indexed_indirect(
                indirect_buffer,
                draw_count,
                size_of::<DrawIndexedIndirectCommand>() as u32,
            )?;
        }

        Ok(self)
    }
}

fn check_pipeline_compute(
    current_state: CommandBufferState,
) -> Result<&ComputePipeline, CheckPipelineError> {
    let pipeline = match current_state.pipeline_compute() {
        Some(x) => x,
        None => return Err(CheckPipelineError::PipelineNotBound),
    };

    Ok(pipeline)
}

fn check_pipeline_graphics(
    current_state: CommandBufferState,
) -> Result<&GraphicsPipeline, CheckPipelineError> {
    let pipeline = match current_state.pipeline_graphics() {
        Some(x) => x,
        None => return Err(CheckPipelineError::PipelineNotBound),
    };

    Ok(pipeline)
}

/// Error that can happen when checking whether the pipeline is valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckPipelineError {
    /// No pipeline was bound to the bind point used by the operation.
    PipelineNotBound,
}

impl error::Error for CheckPipelineError {}

impl fmt::Display for CheckPipelineError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            CheckPipelineError::PipelineNotBound => write!(
                fmt,
                "no pipeline was bound to the bind point used by the operation",
            ),
        }
    }
}

/// Checks whether descriptor sets are compatible with the pipeline.
fn check_descriptor_sets_validity<'a, P: Pipeline>(
    current_state: CommandBufferState,
    pipeline: &P,
    descriptor_requirements: impl IntoIterator<Item = ((u32, u32), &'a DescriptorRequirements)>,
) -> Result<(), CheckDescriptorSetsValidityError> {
    if pipeline.num_used_descriptor_sets() == 0 {
        return Ok(());
    }

    // VUID-vkCmdDispatch-None-02697
    let bindings_pipeline_layout =
        match current_state.descriptor_sets_pipeline_layout(pipeline.bind_point()) {
            Some(x) => x,
            None => return Err(CheckDescriptorSetsValidityError::IncompatiblePipelineLayout),
        };

    // VUID-vkCmdDispatch-None-02697
    if !pipeline.layout().is_compatible_with(
        bindings_pipeline_layout,
        pipeline.num_used_descriptor_sets(),
    ) {
        return Err(CheckDescriptorSetsValidityError::IncompatiblePipelineLayout);
    }

    for ((set_num, binding_num), reqs) in descriptor_requirements {
        let layout_binding =
            &pipeline.layout().set_layouts()[set_num as usize].bindings()[&binding_num];

        let check_buffer = |index: u32, buffer: &Arc<dyn BufferAccess>| Ok(());

        let check_buffer_view = |index: u32, buffer_view: &Arc<dyn BufferViewAbstract>| {
            if layout_binding.descriptor_type == DescriptorType::StorageTexelBuffer {
                // VUID-vkCmdDispatch-OpTypeImage-06423
                if reqs.image_format.is_none()
                    && reqs.storage_write.contains(&index)
                    && !buffer_view.format_features().storage_write_without_format
                {
                    return Err(InvalidDescriptorResource::StorageWriteWithoutFormatNotSupported);
                }

                // VUID-vkCmdDispatch-OpTypeImage-06424
                if reqs.image_format.is_none()
                    && reqs.storage_read.contains(&index)
                    && !buffer_view.format_features().storage_read_without_format
                {
                    return Err(InvalidDescriptorResource::StorageReadWithoutFormatNotSupported);
                }
            }

            Ok(())
        };

        let check_image_view_common = |index: u32, image_view: &Arc<dyn ImageViewAbstract>| {
            // VUID-vkCmdDispatch-None-02691
            if reqs.storage_image_atomic.contains(&index)
                && !image_view.format_features().storage_image_atomic
            {
                return Err(InvalidDescriptorResource::StorageImageAtomicNotSupported);
            }

            if layout_binding.descriptor_type == DescriptorType::StorageImage {
                // VUID-vkCmdDispatch-OpTypeImage-06423
                if reqs.image_format.is_none()
                    && reqs.storage_write.contains(&index)
                    && !image_view.format_features().storage_write_without_format
                {
                    return Err(InvalidDescriptorResource::StorageWriteWithoutFormatNotSupported);
                }

                // VUID-vkCmdDispatch-OpTypeImage-06424
                if reqs.image_format.is_none()
                    && reqs.storage_read.contains(&index)
                    && !image_view.format_features().storage_read_without_format
                {
                    return Err(InvalidDescriptorResource::StorageReadWithoutFormatNotSupported);
                }
            }

            /*
               Instruction/Sampler/Image View Validation
               https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap16.html#textures-input-validation
            */

            // The SPIR-V Image Format is not compatible with the image view’s format.
            if let Some(format) = reqs.image_format {
                if image_view.format() != Some(format) {
                    return Err(InvalidDescriptorResource::ImageViewFormatMismatch {
                        required: format,
                        obtained: image_view.format(),
                    });
                }
            }

            // Rules for viewType
            if let Some(image_view_type) = reqs.image_view_type {
                if image_view.view_type() != image_view_type {
                    return Err(InvalidDescriptorResource::ImageViewTypeMismatch {
                        required: image_view_type,
                        obtained: image_view.view_type(),
                    });
                }
            }

            // - If the image was created with VkImageCreateInfo::samples equal to
            //   VK_SAMPLE_COUNT_1_BIT, the instruction must have MS = 0.
            // - If the image was created with VkImageCreateInfo::samples not equal to
            //   VK_SAMPLE_COUNT_1_BIT, the instruction must have MS = 1.
            if reqs.image_multisampled != (image_view.image().samples() != SampleCount::Sample1) {
                return Err(InvalidDescriptorResource::ImageViewMultisampledMismatch {
                    required: reqs.image_multisampled,
                    obtained: image_view.image().samples() != SampleCount::Sample1,
                });
            }

            // - If the Sampled Type of the OpTypeImage does not match the numeric format of the
            //   image, as shown in the SPIR-V Sampled Type column of the
            //   Interpretation of Numeric Format table.
            // - If the signedness of any read or sample operation does not match the signedness of
            //   the image’s format.
            if let Some(scalar_type) = reqs.image_scalar_type {
                let aspects = image_view.aspects();
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
                    return Err(InvalidDescriptorResource::ImageViewScalarTypeMismatch {
                        required: scalar_type,
                        obtained: view_scalar_type,
                    });
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
                return Err(InvalidDescriptorResource::SamplerUnnormalizedCoordinatesNotAllowed);
            }

            // - OpImageFetch, OpImageSparseFetch, OpImage*Gather, and OpImageSparse*Gather must not
            //   be used with a sampler that enables sampler Y′CBCR conversion.
            // - The ConstOffset and Offset operands must not be used with a sampler that enables
            //   sampler Y′CBCR conversion.
            if reqs.sampler_no_ycbcr_conversion.contains(&index)
                && sampler.sampler_ycbcr_conversion().is_some()
            {
                return Err(InvalidDescriptorResource::SamplerYcbcrConversionNotAllowed);
            }

            /*
                Instruction/Sampler/Image View Validation
                https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap16.html#textures-input-validation
            */

            // - The SPIR-V instruction is one of the OpImage*Dref* instructions and the sampler
            //   compareEnable is VK_FALSE
            // - The SPIR-V instruction is not one of the OpImage*Dref* instructions and the sampler
            //   compareEnable is VK_TRUE
            if reqs.sampler_compare.contains(&index) != sampler.compare().is_some() {
                return Err(InvalidDescriptorResource::SamplerCompareMismatch {
                    required: reqs.sampler_compare.contains(&index),
                    obtained: sampler.compare().is_some(),
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
                        return Err(InvalidDescriptorResource::SamplerImageViewIncompatible {
                            image_view_set_num: id.set,
                            image_view_binding_num: id.binding,
                            image_view_index: id.index,
                            error,
                        });
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
            None => return Err(CheckDescriptorSetsValidityError::MissingDescriptorSet { set_num }),
        };

        let binding_resources = set_resources.binding(binding_num).unwrap();

        match binding_resources {
            DescriptorBindingResources::None(elements) => {
                check_resources(set_num, binding_num, reqs, elements, check_none)?;
            }
            DescriptorBindingResources::Buffer(elements) => {
                check_resources(set_num, binding_num, reqs, elements, check_buffer)?;
            }
            DescriptorBindingResources::BufferView(elements) => {
                check_resources(set_num, binding_num, reqs, elements, check_buffer_view)?;
            }
            DescriptorBindingResources::ImageView(elements) => {
                check_resources(set_num, binding_num, reqs, elements, check_image_view)?;
            }
            DescriptorBindingResources::ImageViewSampler(elements) => {
                check_resources(
                    set_num,
                    binding_num,
                    reqs,
                    elements,
                    check_image_view_sampler,
                )?;
            }
            DescriptorBindingResources::Sampler(elements) => {
                check_resources(set_num, binding_num, reqs, elements, check_sampler)?;
            }
        }
    }

    Ok(())
}

/// Error that can happen when checking descriptor sets validity.
#[derive(Clone, Debug)]
pub enum CheckDescriptorSetsValidityError {
    IncompatiblePipelineLayout,
    InvalidDescriptorResource {
        set_num: u32,
        binding_num: u32,
        index: u32,
        error: InvalidDescriptorResource,
    },
    MissingDescriptorSet {
        set_num: u32,
    },
}

impl error::Error for CheckDescriptorSetsValidityError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::InvalidDescriptorResource { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for CheckDescriptorSetsValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::IncompatiblePipelineLayout => {
                write!(fmt, "the bound pipeline is not compatible with the layout used to bind the descriptor sets")
            }
            Self::InvalidDescriptorResource {
                set_num,
                binding_num,
                index,
                ..
            } => {
                write!(
                    fmt,
                    "the resource bound to descriptor set {} binding {} index {} was not valid",
                    set_num, binding_num, index,
                )
            }
            Self::MissingDescriptorSet { set_num } => {
                write!(fmt, "descriptor set {} has not been not bound, but is required by the pipeline layout", set_num)
            }
        }
    }
}

fn check_resources<T>(
    set_num: u32,
    binding_num: u32,
    reqs: &DescriptorRequirements,
    elements: &[Option<T>],
    mut extra_check: impl FnMut(u32, &T) -> Result<(), InvalidDescriptorResource>,
) -> Result<(), CheckDescriptorSetsValidityError> {
    for (index, element) in elements[0..reqs.descriptor_count as usize]
        .iter()
        .enumerate()
    {
        let index = index as u32;

        // VUID-vkCmdDispatch-None-02699
        let element = match element {
            Some(x) => x,
            None => {
                return Err(
                    CheckDescriptorSetsValidityError::InvalidDescriptorResource {
                        set_num,
                        binding_num,
                        index,
                        error: InvalidDescriptorResource::Missing,
                    },
                )
            }
        };

        if let Err(error) = extra_check(index, element) {
            return Err(
                CheckDescriptorSetsValidityError::InvalidDescriptorResource {
                    set_num,
                    binding_num,
                    index,
                    error,
                },
            );
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub enum InvalidDescriptorResource {
    ImageViewFormatMismatch {
        required: Format,
        obtained: Option<Format>,
    },
    ImageViewMultisampledMismatch {
        required: bool,
        obtained: bool,
    },
    ImageViewScalarTypeMismatch {
        required: ShaderScalarType,
        obtained: ShaderScalarType,
    },
    ImageViewTypeMismatch {
        required: ImageViewType,
        obtained: ImageViewType,
    },
    Missing,
    SamplerCompareMismatch {
        required: bool,
        obtained: bool,
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

impl error::Error for InvalidDescriptorResource {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::SamplerImageViewIncompatible { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for InvalidDescriptorResource {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::ImageViewFormatMismatch { required, obtained } => {
                write!(fmt, "the bound image view did not have the required format; required {:?}, obtained {:?}", required, obtained)
            }
            Self::ImageViewMultisampledMismatch { required, obtained } => {
                write!(fmt, "the bound image did not have the required multisampling; required {}, obtained {}", required, obtained)
            }
            Self::ImageViewScalarTypeMismatch { required, obtained } => {
                write!(fmt, "the bound image view did not have a format and aspect with the required scalar type; required {:?}, obtained {:?}", required, obtained)
            }
            Self::ImageViewTypeMismatch { required, obtained } => {
                write!(fmt, "the bound image view did not have the required type; required {:?}, obtained {:?}", required, obtained)
            }
            Self::Missing => {
                write!(fmt, "no resource was bound")
            }
            Self::SamplerImageViewIncompatible {
                image_view_set_num,
                image_view_binding_num,
                image_view_index,
                ..
            } => {
                write!(
                    fmt,
                    "the bound sampler samples an image view that is not compatible with it"
                )
            }
            Self::SamplerCompareMismatch { required, obtained } => {
                write!(
                    fmt,
                    "the bound sampler did not have the required depth comparison state; required {}, obtained {}", required, obtained
                )
            }
            Self::SamplerUnnormalizedCoordinatesNotAllowed => {
                write!(
                    fmt,
                    "the bound sampler is required to have unnormalized coordinates disabled"
                )
            }
            Self::SamplerYcbcrConversionNotAllowed => {
                write!(
                    fmt,
                    "the bound sampler is required to have no attached sampler YCbCr conversion"
                )
            }
            Self::StorageImageAtomicNotSupported => {
                write!(fmt, "the bound image view did not support the `storage_image_atomic` format feature")
            }
            Self::StorageReadWithoutFormatNotSupported => {
                write!(fmt, "the bound image view or buffer view did not support the `storage_read_without_format` format feature")
            }
            Self::StorageWriteWithoutFormatNotSupported => {
                write!(fmt, "the bound image view or buffer view did not support the `storage_write_without_format` format feature")
            }
        }
    }
}

/// Checks whether push constants are compatible with the pipeline.
fn check_push_constants_validity(
    current_state: CommandBufferState,
    pipeline_layout: &PipelineLayout,
) -> Result<(), CheckPushConstantsValidityError> {
    if pipeline_layout.push_constant_ranges().is_empty() {
        return Ok(());
    }

    let constants_pipeline_layout = match current_state.push_constants_pipeline_layout() {
        Some(x) => x,
        None => return Err(CheckPushConstantsValidityError::MissingPushConstants),
    };

    if pipeline_layout.internal_object() != constants_pipeline_layout.internal_object()
        && pipeline_layout.push_constant_ranges()
            != constants_pipeline_layout.push_constant_ranges()
    {
        return Err(CheckPushConstantsValidityError::IncompatiblePushConstants);
    }

    let set_bytes = current_state.push_constants();

    if !pipeline_layout
        .push_constant_ranges()
        .iter()
        .all(|pc_range| set_bytes.contains(pc_range.offset..pc_range.offset + pc_range.size))
    {
        return Err(CheckPushConstantsValidityError::MissingPushConstants);
    }

    Ok(())
}

/// Error that can happen when checking push constants validity.
#[derive(Debug, Copy, Clone)]
pub enum CheckPushConstantsValidityError {
    /// The push constants are incompatible with the pipeline layout.
    IncompatiblePushConstants,
    /// Not all push constants used by the pipeline have been set.
    MissingPushConstants,
}

impl error::Error for CheckPushConstantsValidityError {}

impl fmt::Display for CheckPushConstantsValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            CheckPushConstantsValidityError::IncompatiblePushConstants => {
                write!(
                    fmt,
                    "the push constants are incompatible with the pipeline layout"
                )
            }
            CheckPushConstantsValidityError::MissingPushConstants => {
                write!(
                    fmt,
                    "not all push constants used by the pipeline have been set"
                )
            }
        }
    }
}

/// Checks whether states that are about to be set are correct.
fn check_dynamic_state_validity(
    current_state: CommandBufferState,
    pipeline: &GraphicsPipeline,
) -> Result<(), CheckDynamicStateValidityError> {
    let device = pipeline.device();

    for dynamic_state in pipeline
        .dynamic_states()
        .filter(|(_, d)| *d)
        .map(|(s, _)| s)
    {
        match dynamic_state {
            DynamicState::BlendConstants => {
                if current_state.blend_constants().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::ColorWriteEnable => {
                let enables = if let Some(enables) = current_state.color_write_enable() {
                    enables
                } else {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                };

                if enables.len() < pipeline.color_blend_state().unwrap().attachments.len() {
                    return Err(CheckDynamicStateValidityError::NotEnoughColorWriteEnable {
                        color_write_enable_count: enables.len() as u32,
                        attachment_count: pipeline.color_blend_state().unwrap().attachments.len()
                            as u32,
                    });
                }
            }
            DynamicState::CullMode => {
                if current_state.cull_mode().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::DepthBias => {
                if current_state.depth_bias().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::DepthBiasEnable => {
                if current_state.depth_bias_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::DepthBounds => {
                if current_state.depth_bounds().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::DepthBoundsTestEnable => {
                if current_state.depth_bounds_test_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::DepthCompareOp => {
                if current_state.depth_compare_op().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::DepthTestEnable => {
                if current_state.depth_test_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::DepthWriteEnable => {
                if current_state.depth_write_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
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
                    if current_state.discard_rectangle(num).is_none() {
                        return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                    }
                }
            }
            DynamicState::ExclusiveScissor => todo!(),
            DynamicState::FragmentShadingRate => todo!(),
            DynamicState::FrontFace => {
                if current_state.front_face().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::LineStipple => {
                if current_state.line_stipple().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::LineWidth => {
                if current_state.line_width().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::LogicOp => {
                if current_state.logic_op().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::PatchControlPoints => {
                if current_state.patch_control_points().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::PrimitiveRestartEnable => {
                let primitive_restart_enable =
                    if let Some(enable) = current_state.primitive_restart_enable() {
                        enable
                    } else {
                        return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                    };

                if primitive_restart_enable {
                    let topology = match pipeline.input_assembly_state().topology {
                        PartialStateMode::Fixed(topology) => topology,
                        PartialStateMode::Dynamic(_) => {
                            if let Some(topology) = current_state.primitive_topology() {
                                topology
                            } else {
                                return Err(CheckDynamicStateValidityError::NotSet {
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
                            if !device.enabled_features().primitive_topology_list_restart {
                                return Err(CheckDynamicStateValidityError::FeatureNotEnabled {
                                    feature: "primitive_topology_list_restart",
                                    reason: "the PrimitiveRestartEnable dynamic state was true in combination with a List PrimitiveTopology",
                                });
                            }
                        }
                        PrimitiveTopology::PatchList => {
                            if !device
                                .enabled_features()
                                .primitive_topology_patch_list_restart
                            {
                                return Err(CheckDynamicStateValidityError::FeatureNotEnabled {
                                    feature: "primitive_topology_patch_list_restart",
                                    reason: "the PrimitiveRestartEnable dynamic state was true in combination with PrimitiveTopology::PatchList",
                                });
                            }
                        }
                        _ => (),
                    }
                }
            }
            DynamicState::PrimitiveTopology => {
                let topology = if let Some(topology) = current_state.primitive_topology() {
                    topology
                } else {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                };

                if pipeline.shader(ShaderStage::TessellationControl).is_some() {
                    if !matches!(topology, PrimitiveTopology::PatchList) {
                        return Err(CheckDynamicStateValidityError::InvalidPrimitiveTopology {
                            topology,
                            reason: "the graphics pipeline includes tessellation shaders, so the topology must be PatchList",
                        });
                    }
                } else {
                    if matches!(topology, PrimitiveTopology::PatchList) {
                        return Err(CheckDynamicStateValidityError::InvalidPrimitiveTopology {
                            topology,
                            reason: "the graphics pipeline doesn't include tessellation shaders",
                        });
                    }
                }

                let topology_class = match pipeline.input_assembly_state().topology {
                    PartialStateMode::Dynamic(topology_class) => topology_class,
                    _ => unreachable!(),
                };

                if topology.class() != topology_class {
                    return Err(CheckDynamicStateValidityError::InvalidPrimitiveTopology {
                        topology,
                        reason: "the topology class does not match the class the pipeline was created for",
                    });
                }

                // TODO: check that the topology matches the geometry shader
            }
            DynamicState::RasterizerDiscardEnable => {
                if current_state.rasterizer_discard_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::RayTracingPipelineStackSize => unreachable!(
                "RayTracingPipelineStackSize dynamic state should not occur on a graphics pipeline"
            ),
            DynamicState::SampleLocations => todo!(),
            DynamicState::Scissor => {
                for num in 0..pipeline.viewport_state().unwrap().count().unwrap() {
                    if current_state.scissor(num).is_none() {
                        return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                    }
                }
            }
            DynamicState::ScissorWithCount => {
                let scissor_count = if let Some(scissors) = current_state.scissor_with_count() {
                    scissors.len() as u32
                } else {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                };

                // Check if the counts match, but only if the viewport count is fixed.
                // If the viewport count is also dynamic, then the DynamicState::ViewportWithCount
                // match arm will handle it.
                if let Some(viewport_count) = pipeline.viewport_state().unwrap().count() {
                    if viewport_count != scissor_count {
                        return Err(
                            CheckDynamicStateValidityError::ViewportScissorCountMismatch {
                                viewport_count,
                                scissor_count,
                            },
                        );
                    }
                }
            }
            DynamicState::StencilCompareMask => {
                let state = current_state.stencil_compare_mask();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::StencilOp => {
                let state = current_state.stencil_op();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::StencilReference => {
                let state = current_state.stencil_reference();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::StencilTestEnable => {
                if current_state.stencil_test_enable().is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }

                // TODO: Check if the stencil buffer is writable
            }
            DynamicState::StencilWriteMask => {
                let state = current_state.stencil_write_mask();

                if state.front.is_none() || state.back.is_none() {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                }
            }
            DynamicState::VertexInput => todo!(),
            DynamicState::VertexInputBindingStride => todo!(),
            DynamicState::Viewport => {
                for num in 0..pipeline.viewport_state().unwrap().count().unwrap() {
                    if current_state.viewport(num).is_none() {
                        return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                    }
                }
            }
            DynamicState::ViewportCoarseSampleOrder => todo!(),
            DynamicState::ViewportShadingRatePalette => todo!(),
            DynamicState::ViewportWithCount => {
                let viewport_count = if let Some(viewports) = current_state.viewport_with_count() {
                    viewports.len() as u32
                } else {
                    return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                };

                let scissor_count =
                    if let Some(scissor_count) = pipeline.viewport_state().unwrap().count() {
                        // The scissor count is fixed.
                        scissor_count
                    } else {
                        // The scissor count is also dynamic.
                        if let Some(scissors) = current_state.scissor_with_count() {
                            scissors.len() as u32
                        } else {
                            return Err(CheckDynamicStateValidityError::NotSet { dynamic_state });
                        }
                    };

                if viewport_count != scissor_count {
                    return Err(
                        CheckDynamicStateValidityError::ViewportScissorCountMismatch {
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

/// Error that can happen when validating dynamic states.
#[derive(Debug, Copy, Clone)]
pub enum CheckDynamicStateValidityError {
    /// A device feature that was required for a particular dynamic state value was not enabled.
    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
    },

    /// The provided dynamic primitive topology is not compatible with the pipeline.
    InvalidPrimitiveTopology {
        topology: PrimitiveTopology,
        reason: &'static str,
    },

    /// The number of ColorWriteEnable values was less than the number of attachments in the
    /// color blend state of the pipeline.
    NotEnoughColorWriteEnable {
        color_write_enable_count: u32,
        attachment_count: u32,
    },

    /// The pipeline requires a particular state to be set dynamically, but the value was not or
    /// only partially set.
    NotSet { dynamic_state: DynamicState },

    /// The viewport count and scissor count do not match.
    ViewportScissorCountMismatch {
        viewport_count: u32,
        scissor_count: u32,
    },
}

impl error::Error for CheckDynamicStateValidityError {}

impl fmt::Display for CheckDynamicStateValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::FeatureNotEnabled { feature, reason } => {
                write!(fmt, "the feature {} must be enabled: {}", feature, reason)
            }
            Self::InvalidPrimitiveTopology { topology, reason } => {
                write!(
                    fmt,
                    "invalid dynamic PrimitiveTypology::{:?}: {}",
                    topology, reason
                )
            }
            Self::NotEnoughColorWriteEnable {
                color_write_enable_count,
                attachment_count,
            } => {
                write!(fmt, "the number of ColorWriteEnable values ({}) was less than the number of attachments ({}) in the color blend state of the pipeline", color_write_enable_count, attachment_count)
            }
            Self::NotSet { dynamic_state } => {
                write!(fmt, "the pipeline requires the dynamic state {:?} to be set, but the value was not or only partially set", dynamic_state)
            }
            Self::ViewportScissorCountMismatch {
                viewport_count,
                scissor_count,
            } => {
                write!(fmt, "the viewport count and scissor count do not match; viewport count is {}, scissor count is {}", viewport_count, scissor_count)
            }
        }
    }
}

/// Checks whether an index buffer can be bound.
///
/// # Panic
///
/// - Panics if the buffer was not created with `device`.
///
fn check_index_buffer(
    current_state: CommandBufferState,
    indices: Option<(u32, u32)>,
) -> Result<(), CheckIndexBufferError> {
    let (index_buffer, index_type) = match current_state.index_buffer() {
        Some(x) => x,
        None => return Err(CheckIndexBufferError::BufferNotBound),
    };

    if let Some((first_index, index_count)) = indices {
        let max_index_count = (index_buffer.size() / index_type.size()) as u32;

        if first_index + index_count > max_index_count {
            return Err(CheckIndexBufferError::TooManyIndices {
                index_count,
                max_index_count,
            }
            .into());
        }
    }

    Ok(())
}

/// Error that can happen when checking whether binding an index buffer is valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckIndexBufferError {
    /// No index buffer was bound.
    BufferNotBound,
    /// A draw command requested too many indices.
    TooManyIndices {
        /// The used amount of indices.
        index_count: u32,
        /// The allowed amount of indices.
        max_index_count: u32,
    },
    /// The "index buffer" usage must be enabled on the index buffer.
    BufferMissingUsage,
    /// The data or size must be 4-bytes aligned.
    WrongAlignment,
    /// The type of the indices is not supported by the device.
    UnsupportIndexType,
}

impl error::Error for CheckIndexBufferError {}

impl fmt::Display for CheckIndexBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckIndexBufferError::BufferNotBound => {
                    "no index buffer was bound"
                }
                CheckIndexBufferError::TooManyIndices { .. } => {
                    "the draw command requested too many indices"
                }
                CheckIndexBufferError::BufferMissingUsage => {
                    "the index buffer usage must be enabled on the index buffer"
                }
                CheckIndexBufferError::WrongAlignment => {
                    "the sum of offset and the address of the range of VkDeviceMemory object that is \
                 backing buffer, must be a multiple of the type indicated by indexType"
                }
                CheckIndexBufferError::UnsupportIndexType => {
                    "the type of the indices is not supported by the device"
                }
            }
        )
    }
}

/// Checks whether an indirect buffer can be bound.
fn check_indirect_buffer(
    device: &Device,
    buffer: &dyn BufferAccess,
) -> Result<(), CheckIndirectBufferError> {
    assert_eq!(
        buffer.inner().buffer.device().internal_object(),
        device.internal_object()
    );

    if !buffer.inner().buffer.usage().indirect_buffer {
        return Err(CheckIndirectBufferError::BufferMissingUsage);
    }

    Ok(())
}

/// Error that can happen when checking whether binding an indirect buffer is valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckIndirectBufferError {
    /// The "indirect buffer" usage must be enabled on the indirect buffer.
    BufferMissingUsage,
    /// The maximum number of indirect draws has been exceeded.
    MaxDrawIndirectCountLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },
}

impl error::Error for CheckIndirectBufferError {}

impl fmt::Display for CheckIndirectBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckIndirectBufferError::BufferMissingUsage => {
                    "the indirect buffer usage must be enabled on the indirect buffer"
                }
                CheckIndirectBufferError::MaxDrawIndirectCountLimitExceeded {
                    limit,
                    requested,
                } => {
                    "the maximum number of indirect draws has been exceeded"
                }
            }
        )
    }
}

fn check_vertex_buffers(
    current_state: CommandBufferState,
    pipeline: &GraphicsPipeline,
    vertices: Option<(u32, u32)>,
    instances: Option<(u32, u32)>,
) -> Result<(), CheckVertexBufferError> {
    let vertex_input = pipeline.vertex_input_state();
    let mut max_vertex_count: Option<u32> = None;
    let mut max_instance_count: Option<u32> = None;

    for (&binding_num, binding_desc) in &vertex_input.bindings {
        let vertex_buffer = match current_state.vertex_buffer(binding_num) {
            Some(x) => x,
            None => return Err(CheckVertexBufferError::BufferNotBound { binding_num }),
        };

        let mut num_elements = (vertex_buffer.size() / binding_desc.stride as DeviceSize)
            .try_into()
            .unwrap_or(u32::MAX);

        match binding_desc.input_rate {
            VertexInputRate::Vertex => {
                max_vertex_count = Some(if let Some(x) = max_vertex_count {
                    x.min(num_elements)
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
                        num_elements = u32::MAX;
                    }
                } else {
                    // If divisor is 2, we use only half the amount of data from the source buffer,
                    // so the number of instances that can be drawn is twice as large.
                    num_elements = num_elements.saturating_mul(divisor);
                }

                max_instance_count = Some(if let Some(x) = max_instance_count {
                    x.min(num_elements)
                } else {
                    num_elements
                });
            }
        };
    }

    if let Some((first_vertex, vertex_count)) = vertices {
        if let Some(max_vertex_count) = max_vertex_count {
            if first_vertex + vertex_count > max_vertex_count {
                return Err(CheckVertexBufferError::TooManyVertices {
                    vertex_count,
                    max_vertex_count,
                });
            }
        }
    }

    if let Some((first_instance, instance_count)) = instances {
        if let Some(max_instance_count) = max_instance_count {
            if first_instance + instance_count > max_instance_count {
                return Err(CheckVertexBufferError::TooManyInstances {
                    instance_count,
                    max_instance_count,
                }
                .into());
            }
        }

        if pipeline.subpass().render_pass().views_used() != 0 {
            let max_instance_index = pipeline
                .device()
                .physical_device()
                .properties()
                .max_multiview_instance_index
                .unwrap_or(0);

            // The condition is somewhat convoluted to avoid integer overflows.
            let out_of_range = first_instance > max_instance_index
                || (instance_count > 0 && instance_count - 1 > max_instance_index - first_instance);
            if out_of_range {
                return Err(CheckVertexBufferError::TooManyInstances {
                    instance_count,
                    max_instance_count: max_instance_index + 1, // TODO: this can overflow
                }
                .into());
            }
        }
    }

    Ok(())
}

/// Error that can happen when checking whether the vertex buffers are valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckVertexBufferError {
    /// No buffer was bound to a binding slot needed by the pipeline.
    BufferNotBound { binding_num: u32 },

    /// The "vertex buffer" usage must be enabled on the buffer.
    BufferMissingUsage {
        /// Index of the buffer that is missing usage.
        binding_num: u32,
    },

    /// A draw command requested too many vertices.
    TooManyVertices {
        /// The used amount of vertices.
        vertex_count: u32,
        /// The allowed amount of vertices.
        max_vertex_count: u32,
    },

    /// A draw command requested too many instances.
    ///
    /// When the `multiview` feature is used the maximum amount of instances may be reduced
    /// because the implementation may use instancing internally to implement `multiview`.
    TooManyInstances {
        /// The used amount of instances.
        instance_count: u32,
        /// The allowed amount of instances.
        max_instance_count: u32,
    },
}

impl error::Error for CheckVertexBufferError {}

impl fmt::Display for CheckVertexBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckVertexBufferError::BufferNotBound { .. } => {
                    "no buffer was bound to a binding slot needed by the pipeline"
                }
                CheckVertexBufferError::BufferMissingUsage { .. } => {
                    "the vertex buffer usage is missing on a vertex buffer"
                }
                CheckVertexBufferError::TooManyVertices { .. } => {
                    "the draw command requested too many vertices"
                }
                CheckVertexBufferError::TooManyInstances { .. } => {
                    "the draw command requested too many instances"
                }
            }
        )
    }
}

/// Checks whether the dispatch dimensions are supported by the device.
fn check_dispatch(device: &Device, dimensions: [u32; 3]) -> Result<(), CheckDispatchError> {
    let max = device
        .physical_device()
        .properties()
        .max_compute_work_group_count;

    if dimensions[0] > max[0] || dimensions[1] > max[1] || dimensions[2] > max[2] {
        return Err(CheckDispatchError::UnsupportedDimensions {
            requested: dimensions,
            max_supported: max,
        });
    }

    Ok(())
}

/// Error that can happen when checking dispatch command validity.
#[derive(Debug, Copy, Clone)]
pub enum CheckDispatchError {
    /// The dimensions are too large for the device's limits.
    UnsupportedDimensions {
        /// The requested dimensions.
        requested: [u32; 3],
        /// The actual supported dimensions.
        max_supported: [u32; 3],
    },
}

impl error::Error for CheckDispatchError {}

impl fmt::Display for CheckDispatchError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckDispatchError::UnsupportedDimensions { .. } => {
                    "the dimensions are too large for the device's limits"
                }
            }
        )
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
                        ..AccessFlags::none()
                    },
                    DescriptorType::InputAttachment => AccessFlags {
                        input_attachment_read: true,
                        ..AccessFlags::none()
                    },
                    DescriptorType::UniformBuffer | DescriptorType::UniformBufferDynamic => {
                        AccessFlags {
                            uniform_read: true,
                            ..AccessFlags::none()
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

            let buffer_resource =
                move |(memory, buffer): (PipelineMemoryAccess, Arc<dyn BufferAccess>)| {
                    let range = 0..buffer.size(); // TODO:
                    (
                        format!("Buffer bound to set {} descriptor {}", set, binding).into(),
                        Resource::Buffer {
                            buffer,
                            range,
                            memory,
                        },
                    )
                };
            let image_resource =
                move |(memory, image): (PipelineMemoryAccess, Arc<dyn ImageAccess>)| {
                    let subresource_range = ImageSubresourceRange {
                        // TODO:
                        aspects: image.format().aspects(),
                        mip_levels: image.current_mip_levels_access(),
                        array_layers: image.current_array_layers_access(),
                    };
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
                                element.as_ref().map(|buffer| (access, buffer.clone()))
                            })
                            .map(buffer_resource),
                    );
                }
                DescriptorBindingResources::BufferView(elements) => {
                    resources.extend(
                        access
                            .zip(elements)
                            .filter_map(|(access, element)| {
                                element
                                    .as_ref()
                                    .map(|buffer_view| (access, buffer_view.buffer()))
                            })
                            .map(buffer_resource),
                    );
                }
                DescriptorBindingResources::ImageView(elements) => {
                    resources.extend(
                        access
                            .zip(elements)
                            .filter_map(|(access, element)| {
                                element
                                    .as_ref()
                                    .map(|image_view| (access, image_view.image()))
                            })
                            .map(image_resource),
                    );
                }
                DescriptorBindingResources::ImageViewSampler(elements) => {
                    resources.extend(
                        access
                            .zip(elements)
                            .filter_map(|(access, element)| {
                                element
                                    .as_ref()
                                    .map(|(image_view, _)| (access, image_view.image()))
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
                            ..PipelineStages::none()
                        },
                        access: AccessFlags {
                            vertex_attribute_read: true,
                            ..AccessFlags::none()
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
                        ..PipelineStages::none()
                    },
                    access: AccessFlags {
                        index_read: true,
                        ..AccessFlags::none()
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
                        ..PipelineStages::none()
                    }, // TODO: is draw_indirect correct for dispatch too?
                    access: AccessFlags {
                        indirect_command_read: true,
                        ..AccessFlags::none()
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
        fns.v1_0.cmd_dispatch(
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

        fns.v1_0
            .cmd_dispatch_indirect(self.handle, inner.buffer.internal_object(), inner.offset);
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
        fns.v1_0.cmd_draw(
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
        fns.v1_0.cmd_draw_indexed(
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

        fns.v1_0.cmd_draw_indirect(
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

        fns.v1_0.cmd_draw_indexed_indirect(
            self.handle,
            inner.buffer.internal_object(),
            inner.offset,
            draw_count,
            stride,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_checked() {
        let (device, _) = gfx_dev_and_queue!();

        let attempted = [u32::MAX, u32::MAX, u32::MAX];

        // Just in case the device is some kind of software implementation.
        if device
            .physical_device()
            .properties()
            .max_compute_work_group_count
            == attempted
        {
            return;
        }

        match check_dispatch(&device, attempted) {
            Err(CheckDispatchError::UnsupportedDimensions { requested, .. }) => {
                assert_eq!(requested, attempted);
            }
            _ => panic!(),
        }
    }
}
