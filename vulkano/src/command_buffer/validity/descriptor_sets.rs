// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::command_buffer::synced::CommandBufferState;
use crate::descriptor_set::DescriptorBindingResources;
use crate::format::Format;
use crate::image::view::ImageViewType;
use crate::image::ImageViewAbstract;
use crate::image::SampleCount;
use crate::pipeline::Pipeline;
use crate::shader::DescriptorRequirements;
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

    let bindings_pipeline_layout =
        match current_state.descriptor_sets_pipeline_layout(pipeline.bind_point()) {
            Some(x) => x,
            None => return Err(CheckDescriptorSetsValidityError::IncompatiblePipelineLayout),
        };

    if !pipeline.layout().is_compatible_with(
        bindings_pipeline_layout,
        pipeline.num_used_descriptor_sets(),
    ) {
        return Err(CheckDescriptorSetsValidityError::IncompatiblePipelineLayout);
    }

    for ((set_num, binding_num), reqs) in descriptor_requirements {
        let check_image_view = |image_view: &Arc<dyn ImageViewAbstract>| {
            if let Some(image_view_type) = reqs.image_view_type {
                if image_view.ty() != image_view_type {
                    return Err(InvalidDescriptorResource::ImageViewTypeMismatch {
                        required: reqs.image_view_type.unwrap(),
                        obtained: image_view.ty(),
                    });
                }
            }

            if let Some(format) = reqs.format {
                if image_view.format() != format {
                    return Err(InvalidDescriptorResource::ImageViewFormatMismatch {
                        required: format,
                        obtained: image_view.format(),
                    });
                }
            }

            if reqs.multisampled != (image_view.image().samples() != SampleCount::Sample1) {
                return Err(InvalidDescriptorResource::ImageMultisampledMismatch {
                    required: reqs.multisampled,
                    obtained: image_view.image().samples() != SampleCount::Sample1,
                });
            }

            Ok(())
        };

        let set_resources = match current_state.descriptor_set(pipeline.bind_point(), set_num) {
            Some(x) => x.resources(),
            None => return Err(CheckDescriptorSetsValidityError::MissingDescriptorSet { set_num }),
        };

        let binding_resources = set_resources.binding(binding_num).unwrap();

        match binding_resources {
            DescriptorBindingResources::None => (),
            DescriptorBindingResources::Buffer(elements) => {
                check_resources(set_num, binding_num, reqs, elements, |_| Ok(()))?;
            }
            DescriptorBindingResources::BufferView(elements) => {
                check_resources(set_num, binding_num, reqs, elements, |_| Ok(()))?;
            }
            DescriptorBindingResources::ImageView(elements) => {
                check_resources(set_num, binding_num, reqs, elements, |i| {
                    check_image_view(i)
                })?;
            }
            DescriptorBindingResources::ImageViewSampler(elements) => {
                check_resources(set_num, binding_num, reqs, elements, |(i, s)| {
                    check_image_view(i)
                })?;
            }
            DescriptorBindingResources::Sampler(elements) => {
                check_resources(set_num, binding_num, reqs, elements, |_| Ok(()))?;
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
    mut extra_check: impl FnMut(&T) -> Result<(), InvalidDescriptorResource>,
) -> Result<(), CheckDescriptorSetsValidityError> {
    for (index, element) in elements[0..reqs.descriptor_count as usize]
        .iter()
        .enumerate()
    {
        let element = match element {
            Some(x) => x,
            None => {
                return Err(
                    CheckDescriptorSetsValidityError::InvalidDescriptorResource {
                        set_num,
                        binding_num,
                        index: index as u32,
                        error: InvalidDescriptorResource::Missing,
                    },
                )
            }
        };

        if let Err(error) = extra_check(element) {
            return Err(
                CheckDescriptorSetsValidityError::InvalidDescriptorResource {
                    set_num,
                    binding_num,
                    index: index as u32,
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
    ImageMultisampledMismatch {
        required: bool,
        obtained: bool,
    },
    ImageViewTypeMismatch {
        required: ImageViewType,
        obtained: ImageViewType,
    },
    Missing,
}

impl error::Error for InvalidDescriptorResource {}

impl fmt::Display for InvalidDescriptorResource {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::Missing => {
                write!(fmt, "no resource was bound")
            }
            Self::ImageViewFormatMismatch { required, obtained } => {
                write!(fmt, "the bound image view did not have the required format; required {:?}, obtained {:?}", required, obtained)
            }
            Self::ImageMultisampledMismatch { required, obtained } => {
                write!(fmt, "the bound image did not have the required multisampling; required {}, obtained {}", required, obtained)
            }
            Self::ImageViewTypeMismatch { required, obtained } => {
                write!(fmt, "the bound image view did not have the required type; required {:?}, obtained {:?}", required, obtained)
            }
        }
    }
}
