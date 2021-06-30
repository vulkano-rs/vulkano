// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Contains the `check_desc_against_limits` function and the `PipelineLayoutLimitsError` error.

use crate::descriptor_set::layout::DescriptorType;
use crate::device::Properties;
use crate::pipeline::layout::PipelineLayoutDesc;
use crate::pipeline::layout::PipelineLayoutDescPcRange;
use crate::pipeline::shader::ShaderStages;
use std::error;
use std::fmt;

/// Checks whether the pipeline layout description fulfills the device limits requirements.
pub fn check_desc_against_limits(
    desc: &PipelineLayoutDesc,
    properties: &Properties,
) -> Result<(), PipelineLayoutLimitsError> {
    let mut num_resources = Counter::default();
    let mut num_samplers = Counter::default();
    let mut num_uniform_buffers = Counter::default();
    let mut num_uniform_buffers_dynamic = 0;
    let mut num_storage_buffers = Counter::default();
    let mut num_storage_buffers_dynamic = 0;
    let mut num_sampled_images = Counter::default();
    let mut num_storage_images = Counter::default();
    let mut num_input_attachments = Counter::default();

    for set in desc.descriptor_sets() {
        for (_, descriptor) in set
            .bindings()
            .iter()
            .enumerate()
            .filter_map(|(binding, desc)| desc.as_ref().map(|desc| (binding, desc)))
        {
            num_resources.increment(descriptor.array_count, &descriptor.stages);

            match descriptor.ty.ty() {
                // TODO:
                DescriptorType::Sampler => {
                    num_samplers.increment(descriptor.array_count, &descriptor.stages);
                }
                DescriptorType::CombinedImageSampler => {
                    num_samplers.increment(descriptor.array_count, &descriptor.stages);
                    num_sampled_images.increment(descriptor.array_count, &descriptor.stages);
                }
                DescriptorType::SampledImage | DescriptorType::UniformTexelBuffer => {
                    num_sampled_images.increment(descriptor.array_count, &descriptor.stages);
                }
                DescriptorType::StorageImage | DescriptorType::StorageTexelBuffer => {
                    num_storage_images.increment(descriptor.array_count, &descriptor.stages);
                }
                DescriptorType::UniformBuffer => {
                    num_uniform_buffers.increment(descriptor.array_count, &descriptor.stages);
                }
                DescriptorType::UniformBufferDynamic => {
                    num_uniform_buffers.increment(descriptor.array_count, &descriptor.stages);
                    num_uniform_buffers_dynamic += 1;
                }
                DescriptorType::StorageBuffer => {
                    num_storage_buffers.increment(descriptor.array_count, &descriptor.stages);
                }
                DescriptorType::StorageBufferDynamic => {
                    num_storage_buffers.increment(descriptor.array_count, &descriptor.stages);
                    num_storage_buffers_dynamic += 1;
                }
                DescriptorType::InputAttachment => {
                    num_input_attachments.increment(descriptor.array_count, &descriptor.stages);
                }
            }
        }
    }

    if desc.descriptor_sets().len() > properties.max_bound_descriptor_sets.unwrap() as usize {
        return Err(PipelineLayoutLimitsError::MaxDescriptorSetsLimitExceeded {
            limit: properties.max_bound_descriptor_sets.unwrap() as usize,
            requested: desc.descriptor_sets().len(),
        });
    }

    if num_resources.max_per_stage() > properties.max_per_stage_resources.unwrap() {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageResourcesLimitExceeded {
                limit: properties.max_per_stage_resources.unwrap(),
                requested: num_resources.max_per_stage(),
            },
        );
    }

    if num_samplers.max_per_stage() > properties.max_per_stage_descriptor_samplers.unwrap() {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorSamplersLimitExceeded {
                limit: properties.max_per_stage_descriptor_samplers.unwrap(),
                requested: num_samplers.max_per_stage(),
            },
        );
    }
    if num_uniform_buffers.max_per_stage()
        > properties.max_per_stage_descriptor_uniform_buffers.unwrap()
    {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorUniformBuffersLimitExceeded {
                limit: properties.max_per_stage_descriptor_uniform_buffers.unwrap(),
                requested: num_uniform_buffers.max_per_stage(),
            },
        );
    }
    if num_storage_buffers.max_per_stage()
        > properties.max_per_stage_descriptor_storage_buffers.unwrap()
    {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorStorageBuffersLimitExceeded {
                limit: properties.max_per_stage_descriptor_storage_buffers.unwrap(),
                requested: num_storage_buffers.max_per_stage(),
            },
        );
    }
    if num_sampled_images.max_per_stage()
        > properties.max_per_stage_descriptor_sampled_images.unwrap()
    {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorSampledImagesLimitExceeded {
                limit: properties.max_per_stage_descriptor_sampled_images.unwrap(),
                requested: num_sampled_images.max_per_stage(),
            },
        );
    }
    if num_storage_images.max_per_stage()
        > properties.max_per_stage_descriptor_storage_images.unwrap()
    {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorStorageImagesLimitExceeded {
                limit: properties.max_per_stage_descriptor_storage_images.unwrap(),
                requested: num_storage_images.max_per_stage(),
            },
        );
    }
    if num_input_attachments.max_per_stage()
        > properties
            .max_per_stage_descriptor_input_attachments
            .unwrap()
    {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorInputAttachmentsLimitExceeded {
                limit: properties
                    .max_per_stage_descriptor_input_attachments
                    .unwrap(),
                requested: num_input_attachments.max_per_stage(),
            },
        );
    }

    if num_samplers.total > properties.max_descriptor_set_samplers.unwrap() {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetSamplersLimitExceeded {
                limit: properties.max_descriptor_set_samplers.unwrap(),
                requested: num_samplers.total,
            },
        );
    }
    if num_uniform_buffers.total > properties.max_descriptor_set_uniform_buffers.unwrap() {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetUniformBuffersLimitExceeded {
                limit: properties.max_descriptor_set_uniform_buffers.unwrap(),
                requested: num_uniform_buffers.total,
            },
        );
    }
    if num_uniform_buffers_dynamic
        > properties
            .max_descriptor_set_uniform_buffers_dynamic
            .unwrap()
    {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetUniformBuffersDynamicLimitExceeded {
                limit: properties
                    .max_descriptor_set_uniform_buffers_dynamic
                    .unwrap(),
                requested: num_uniform_buffers_dynamic,
            },
        );
    }
    if num_storage_buffers.total > properties.max_descriptor_set_storage_buffers.unwrap() {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetStorageBuffersLimitExceeded {
                limit: properties.max_descriptor_set_storage_buffers.unwrap(),
                requested: num_storage_buffers.total,
            },
        );
    }
    if num_storage_buffers_dynamic
        > properties
            .max_descriptor_set_storage_buffers_dynamic
            .unwrap()
    {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetStorageBuffersDynamicLimitExceeded {
                limit: properties
                    .max_descriptor_set_storage_buffers_dynamic
                    .unwrap(),
                requested: num_storage_buffers_dynamic,
            },
        );
    }
    if num_sampled_images.total > properties.max_descriptor_set_sampled_images.unwrap() {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetSampledImagesLimitExceeded {
                limit: properties.max_descriptor_set_sampled_images.unwrap(),
                requested: num_sampled_images.total,
            },
        );
    }
    if num_storage_images.total > properties.max_descriptor_set_storage_images.unwrap() {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetStorageImagesLimitExceeded {
                limit: properties.max_descriptor_set_storage_images.unwrap(),
                requested: num_storage_images.total,
            },
        );
    }
    if num_input_attachments.total > properties.max_descriptor_set_input_attachments.unwrap() {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetInputAttachmentsLimitExceeded {
                limit: properties.max_descriptor_set_input_attachments.unwrap(),
                requested: num_input_attachments.total,
            },
        );
    }

    for &PipelineLayoutDescPcRange { offset, size, .. } in desc.push_constants() {
        if offset + size > properties.max_push_constants_size.unwrap() as usize {
            return Err(PipelineLayoutLimitsError::MaxPushConstantsSizeExceeded {
                limit: properties.max_push_constants_size.unwrap() as usize,
                requested: offset + size,
            });
        }
    }

    Ok(())
}

/// The pipeline layout description isn't compatible with the hardware limits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineLayoutLimitsError {
    /// The maximum number of descriptor sets has been exceeded.
    MaxDescriptorSetsLimitExceeded {
        /// The limit that must be fulfilled.
        limit: usize,
        /// What was requested.
        requested: usize,
    },

    /// The maximum size of push constants has been exceeded.
    MaxPushConstantsSizeExceeded {
        /// The limit that must be fulfilled.
        limit: usize,
        /// What was requested.
        requested: usize,
    },

    /// The `max_per_stage_resources()` limit has been exceeded.
    MaxPerStageResourcesLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_per_stage_descriptor_samplers()` limit has been exceeded.
    MaxPerStageDescriptorSamplersLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_per_stage_descriptor_uniform_buffers()` limit has been exceeded.
    MaxPerStageDescriptorUniformBuffersLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_per_stage_descriptor_storage_buffers()` limit has been exceeded.
    MaxPerStageDescriptorStorageBuffersLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_per_stage_descriptor_sampled_images()` limit has been exceeded.
    MaxPerStageDescriptorSampledImagesLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_per_stage_descriptor_storage_images()` limit has been exceeded.
    MaxPerStageDescriptorStorageImagesLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_per_stage_descriptor_input_attachments()` limit has been exceeded.
    MaxPerStageDescriptorInputAttachmentsLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_descriptor_set_samplers()` limit has been exceeded.
    MaxDescriptorSetSamplersLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_descriptor_set_uniform_buffers()` limit has been exceeded.
    MaxDescriptorSetUniformBuffersLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_descriptor_set_uniform_buffers_dynamic()` limit has been exceeded.
    MaxDescriptorSetUniformBuffersDynamicLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_descriptor_set_storage_buffers()` limit has been exceeded.
    MaxDescriptorSetStorageBuffersLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_descriptor_set_storage_buffers_dynamic()` limit has been exceeded.
    MaxDescriptorSetStorageBuffersDynamicLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_descriptor_set_sampled_images()` limit has been exceeded.
    MaxDescriptorSetSampledImagesLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_descriptor_set_storage_images()` limit has been exceeded.
    MaxDescriptorSetStorageImagesLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },

    /// The `max_descriptor_set_input_attachments()` limit has been exceeded.
    MaxDescriptorSetInputAttachmentsLimitExceeded {
        /// The limit that must be fulfilled.
        limit: u32,
        /// What was requested.
        requested: u32,
    },
}

impl error::Error for PipelineLayoutLimitsError {}

impl fmt::Display for PipelineLayoutLimitsError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                PipelineLayoutLimitsError::MaxDescriptorSetsLimitExceeded { .. } => {
                    "the maximum number of descriptor sets has been exceeded"
                }
                PipelineLayoutLimitsError::MaxPushConstantsSizeExceeded { .. } => {
                    "the maximum size of push constants has been exceeded"
                }
                PipelineLayoutLimitsError::MaxPerStageResourcesLimitExceeded { .. } => {
                    "the `max_per_stage_resources()` limit has been exceeded"
                }
                PipelineLayoutLimitsError::MaxPerStageDescriptorSamplersLimitExceeded {
                    ..
                } => {
                    "the `max_per_stage_descriptor_samplers()` limit has been exceeded"
                }
                PipelineLayoutLimitsError::MaxPerStageDescriptorUniformBuffersLimitExceeded {
                    ..
                } => "the `max_per_stage_descriptor_uniform_buffers()` limit has been exceeded",
                PipelineLayoutLimitsError::MaxPerStageDescriptorStorageBuffersLimitExceeded {
                    ..
                } => "the `max_per_stage_descriptor_storage_buffers()` limit has been exceeded",
                PipelineLayoutLimitsError::MaxPerStageDescriptorSampledImagesLimitExceeded {
                    ..
                } => "the `max_per_stage_descriptor_sampled_images()` limit has been exceeded",
                PipelineLayoutLimitsError::MaxPerStageDescriptorStorageImagesLimitExceeded {
                    ..
                } => "the `max_per_stage_descriptor_storage_images()` limit has been exceeded",
                PipelineLayoutLimitsError::MaxPerStageDescriptorInputAttachmentsLimitExceeded {
                    ..
                } => "the `max_per_stage_descriptor_input_attachments()` limit has been exceeded",
                PipelineLayoutLimitsError::MaxDescriptorSetSamplersLimitExceeded { .. } => {
                    "the `max_descriptor_set_samplers()` limit has been exceeded"
                }
                PipelineLayoutLimitsError::MaxDescriptorSetUniformBuffersLimitExceeded {
                    ..
                } => {
                    "the `max_descriptor_set_uniform_buffers()` limit has been exceeded"
                }
                PipelineLayoutLimitsError::MaxDescriptorSetUniformBuffersDynamicLimitExceeded {
                    ..
                } => "the `max_descriptor_set_uniform_buffers_dynamic()` limit has been exceeded",
                PipelineLayoutLimitsError::MaxDescriptorSetStorageBuffersLimitExceeded {
                    ..
                } => {
                    "the `max_descriptor_set_storage_buffers()` limit has been exceeded"
                }
                PipelineLayoutLimitsError::MaxDescriptorSetStorageBuffersDynamicLimitExceeded {
                    ..
                } => "the `max_descriptor_set_storage_buffers_dynamic()` limit has been exceeded",
                PipelineLayoutLimitsError::MaxDescriptorSetSampledImagesLimitExceeded {
                    ..
                } => {
                    "the `max_descriptor_set_sampled_images()` limit has been exceeded"
                }
                PipelineLayoutLimitsError::MaxDescriptorSetStorageImagesLimitExceeded {
                    ..
                } => {
                    "the `max_descriptor_set_storage_images()` limit has been exceeded"
                }
                PipelineLayoutLimitsError::MaxDescriptorSetInputAttachmentsLimitExceeded {
                    ..
                } => {
                    "the `max_descriptor_set_input_attachments()` limit has been exceeded"
                }
            }
        )
    }
}

// Helper struct for the main function.
#[derive(Default)]
struct Counter {
    total: u32,
    compute: u32,
    vertex: u32,
    geometry: u32,
    tess_ctl: u32,
    tess_eval: u32,
    frag: u32,
}

impl Counter {
    fn increment(&mut self, num: u32, stages: &ShaderStages) {
        self.total += num;
        if stages.compute {
            self.compute += num;
        }
        if stages.vertex {
            self.vertex += num;
        }
        if stages.tessellation_control {
            self.tess_ctl += num;
        }
        if stages.tessellation_evaluation {
            self.tess_eval += num;
        }
        if stages.geometry {
            self.geometry += num;
        }
        if stages.fragment {
            self.frag += num;
        }
    }

    fn max_per_stage(&self) -> u32 {
        let mut max = 0;
        if self.compute > max {
            max = self.compute;
        }
        if self.vertex > max {
            max = self.vertex;
        }
        if self.geometry > max {
            max = self.geometry;
        }
        if self.tess_ctl > max {
            max = self.tess_ctl;
        }
        if self.tess_eval > max {
            max = self.tess_eval;
        }
        if self.frag > max {
            max = self.frag;
        }
        max
    }
}
