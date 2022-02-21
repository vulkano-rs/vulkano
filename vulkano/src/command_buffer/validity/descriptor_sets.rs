// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::view::BufferViewAbstract;
use crate::buffer::BufferAccess;
use crate::command_buffer::synced::CommandBufferState;
use crate::descriptor_set::layout::DescriptorType;
use crate::descriptor_set::DescriptorBindingResources;
use crate::format::Format;
use crate::image::view::{ImageViewAbstract, ImageViewType};
use crate::image::SampleCount;
use crate::pipeline::Pipeline;
use crate::sampler::Sampler;
use crate::sampler::SamplerImageViewIncompatibleError;
use crate::shader::DescriptorRequirements;
use crate::shader::ShaderScalarType;
use std::error;
use std::fmt;
use std::sync::Arc;

/// Checks whether descriptor sets are compatible with the pipeline.
pub(in super::super) fn check_descriptor_sets_validity<'a, P: Pipeline>(
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
        let layout_binding = pipeline.layout().descriptor_set_layouts()[set_num as usize]
            .desc()
            .descriptor(binding_num)
            .unwrap();

        let check_buffer = |index: u32, buffer: &Arc<dyn BufferAccess>| Ok(());

        let check_buffer_view = |index: u32, buffer_view: &Arc<dyn BufferViewAbstract>| {
            if layout_binding.ty == DescriptorType::StorageTexelBuffer {
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

            if layout_binding.ty == DescriptorType::StorageImage {
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
                if image_view.format() != format {
                    return Err(InvalidDescriptorResource::ImageViewFormatMismatch {
                        required: format,
                        obtained: image_view.format(),
                    });
                }
            }

            // Rules for viewType
            if let Some(image_view_type) = reqs.image_view_type {
                if image_view.ty() != image_view_type {
                    return Err(InvalidDescriptorResource::ImageViewTypeMismatch {
                        required: image_view_type,
                        obtained: image_view.ty(),
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
                        image_view.format().type_color().unwrap()
                    } else if aspects.depth {
                        image_view.format().type_depth().unwrap()
                    } else if aspects.stencil {
                        image_view.format().type_stencil().unwrap()
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
        obtained: Format,
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
