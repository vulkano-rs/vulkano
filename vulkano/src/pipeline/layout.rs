// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! The layout of descriptor sets and push constants used by a pipeline.
//!
//! # Overview
//!
//! The layout itself only *describes* the descriptors and push constants, and does not contain
//! their content itself. Instead, you can think of it as a `struct` definition that states which
//! members there are, what types they have, and in what order.
//! One could imagine a Rust definition somewhat like this:
//!
//! ```text
//! #[repr(C)]
//! struct MyPipelineLayout {
//!     push_constants: Pc,
//!     descriptor_set0: Ds0,
//!     descriptor_set1: Ds1,
//!     descriptor_set2: Ds2,
//!     descriptor_set3: Ds3,
//! }
//! ```
//!
//! Of course, a pipeline layout is created at runtime, unlike a Rust type.
//!
//! # Layout compatibility
//!
//! When binding descriptor sets or setting push constants, you must provide a pipeline layout.
//! This layout is used to decide where in memory Vulkan should write the new data. The
//! descriptor sets and push constants can later be read by dispatch or draw calls, but only if
//! the bound pipeline being used for the command has a layout that is *compatible* with the layout
//! that was used to bind the resources.
//!
//! *Compatible* means that the pipeline layout must be the same object, or a different layout in
//! which the push constant ranges and descriptor set layouts were be identically defined.
//! However, Vulkan allows for partial compatibility as well. In the `struct` analogy used above,
//! one could imagine that using a different definition would leave some members with the same
//! offset and size within the struct as in the old definition, while others are no longer
//! positioned correctly. For example, if a new, incompatible type were used for `Ds1`, then the
//! `descriptor_set1`, `descriptor_set2` and `descriptor_set3` members would no longer be correct,
//! but `descriptor_set0` and `push_constants` would remain accessible in the new layout.
//! Because of this behaviour, the following rules apply to compatibility between the layouts used
//! in subsequent descriptor set binding calls:
//!
//! - An incompatible definition of `Pc` invalidates all bound descriptor sets.
//! - An incompatible definition of `DsN` invalidates all bound descriptor sets *N* and higher.
//! - If *N* is the highest set being assigned in a bind command, and it and all lower sets
//!   have compatible definitions, including the push constants, then descriptor sets above *N*
//!   remain valid.
//!
//! [`SyncCommandBufferBuilder`](crate::command_buffer::synced::SyncCommandBufferBuilder) keeps
//! track of this state and will automatically remove descriptor sets that have been invalidated
//! by incompatible layouts in subsequent binding commands.
//!
//! # Creating pipeline layouts
//!
//! A pipeline layout is a Vulkan object type, represented in Vulkano with the `PipelineLayout`
//! type. Each pipeline that you create holds a pipeline layout object.

use crate::{
    check_errors,
    descriptor_set::layout::{DescriptorRequirementsNotMet, DescriptorSetLayout, DescriptorType},
    device::{Device, DeviceOwned},
    shader::{DescriptorRequirements, ShaderStages},
    Error, OomError, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    error, fmt,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ptr,
    sync::Arc,
};

/// Describes the layout of descriptor sets and push constants that are made available to shaders.
#[derive(Debug)]
pub struct PipelineLayout {
    handle: ash::vk::PipelineLayout,
    device: Arc<Device>,

    set_layouts: Vec<Arc<DescriptorSetLayout>>,
    push_constant_ranges: Vec<PushConstantRange>,

    push_constant_ranges_disjoint: Vec<PushConstantRange>,
}

impl PipelineLayout {
    /// Creates a new `PipelineLayout`.
    ///
    /// # Panics
    ///
    /// - Panics if an element of `create_info.push_constant_ranges` has an empty `stages` value.
    /// - Panics if an element of `create_info.push_constant_ranges` has an `offset` or `size`
    ///   that's not divisible by 4.
    /// - Panics if an element of `create_info.push_constant_ranges` has an `size` of zero.
    pub fn new(
        device: Arc<Device>,
        mut create_info: PipelineLayoutCreateInfo,
    ) -> Result<Arc<PipelineLayout>, PipelineLayoutCreationError> {
        Self::validate(&device, &mut create_info)?;
        let handle = unsafe { Self::create(&device, &create_info)? };

        let PipelineLayoutCreateInfo {
            set_layouts,
            mut push_constant_ranges,
            _ne: _,
        } = create_info;

        // Sort the ranges for the purpose of comparing for equality.
        // The stage mask is guaranteed to be unique, so it's a suitable sorting key.
        push_constant_ranges.sort_unstable_by_key(|range| {
            (
                range.offset,
                range.size,
                ash::vk::ShaderStageFlags::from(range.stages),
            )
        });

        // Create a list of disjoint ranges.
        let mut push_constant_ranges_disjoint: Vec<PushConstantRange> =
            Vec::with_capacity(push_constant_ranges.len());

        if !push_constant_ranges.is_empty() {
            let mut min_offset = push_constant_ranges[0].offset;
            loop {
                let mut max_offset = u32::MAX;
                let mut stages = ShaderStages::none();

                for range in push_constant_ranges.iter() {
                    // new start (begin next time from it)
                    if range.offset > min_offset {
                        max_offset = max_offset.min(range.offset);
                        break;
                    } else if range.offset + range.size > min_offset {
                        // inside the range, include the stage
                        // use the minimum of the end of all ranges that are overlapping
                        max_offset = max_offset.min(range.offset + range.size);
                        stages = stages | range.stages;
                    }
                }
                // finished all stages
                if stages == ShaderStages::none() {
                    break;
                }

                push_constant_ranges_disjoint.push(PushConstantRange {
                    stages,
                    offset: min_offset,
                    size: max_offset - min_offset,
                });
                // prepare for next range
                min_offset = max_offset;
            }
        }

        Ok(Arc::new(PipelineLayout {
            handle,
            device,
            set_layouts,
            push_constant_ranges,
            push_constant_ranges_disjoint,
        }))
    }

    fn validate(
        device: &Device,
        create_info: &mut PipelineLayoutCreateInfo,
    ) -> Result<(), PipelineLayoutCreationError> {
        let &mut PipelineLayoutCreateInfo {
            ref set_layouts,
            ref push_constant_ranges,
            _ne: _,
        } = create_info;

        let properties = device.physical_device().properties();

        /* Check descriptor set layouts */

        // VUID-VkPipelineLayoutCreateInfo-setLayoutCount-00286
        if set_layouts.len() > properties.max_bound_descriptor_sets as usize {
            return Err(
                PipelineLayoutCreationError::MaxBoundDescriptorSetsExceeded {
                    provided: set_layouts.len() as u32,
                    max_supported: properties.max_bound_descriptor_sets,
                },
            );
        }

        {
            let mut num_resources = Counter::default();
            let mut num_samplers = Counter::default();
            let mut num_uniform_buffers = Counter::default();
            let mut num_uniform_buffers_dynamic = 0;
            let mut num_storage_buffers = Counter::default();
            let mut num_storage_buffers_dynamic = 0;
            let mut num_sampled_images = Counter::default();
            let mut num_storage_images = Counter::default();
            let mut num_input_attachments = Counter::default();
            let mut push_descriptor_set = None;

            for (set_num, set_layout) in set_layouts.iter().enumerate() {
                let set_num = set_num as u32;

                if set_layout.push_descriptor() {
                    // VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293
                    if let Some(num) = push_descriptor_set {
                        return Err(PipelineLayoutCreationError::SetLayoutsPushDescriptorMultiple);
                    } else {
                        push_descriptor_set = Some(set_num);
                    }
                }

                for layout_binding in set_layout.bindings().values() {
                    num_resources
                        .increment(layout_binding.descriptor_count, &layout_binding.stages);

                    match layout_binding.descriptor_type {
                        DescriptorType::Sampler => {
                            num_samplers
                                .increment(layout_binding.descriptor_count, &layout_binding.stages);
                        }
                        DescriptorType::CombinedImageSampler => {
                            num_samplers
                                .increment(layout_binding.descriptor_count, &layout_binding.stages);
                            num_sampled_images
                                .increment(layout_binding.descriptor_count, &layout_binding.stages);
                        }
                        DescriptorType::SampledImage | DescriptorType::UniformTexelBuffer => {
                            num_sampled_images
                                .increment(layout_binding.descriptor_count, &layout_binding.stages);
                        }
                        DescriptorType::StorageImage | DescriptorType::StorageTexelBuffer => {
                            num_storage_images
                                .increment(layout_binding.descriptor_count, &layout_binding.stages);
                        }
                        DescriptorType::UniformBuffer => {
                            num_uniform_buffers
                                .increment(layout_binding.descriptor_count, &layout_binding.stages);
                        }
                        DescriptorType::UniformBufferDynamic => {
                            num_uniform_buffers
                                .increment(layout_binding.descriptor_count, &layout_binding.stages);
                            num_uniform_buffers_dynamic += 1;
                        }
                        DescriptorType::StorageBuffer => {
                            num_storage_buffers
                                .increment(layout_binding.descriptor_count, &layout_binding.stages);
                        }
                        DescriptorType::StorageBufferDynamic => {
                            num_storage_buffers
                                .increment(layout_binding.descriptor_count, &layout_binding.stages);
                            num_storage_buffers_dynamic += 1;
                        }
                        DescriptorType::InputAttachment => {
                            num_input_attachments
                                .increment(layout_binding.descriptor_count, &layout_binding.stages);
                        }
                    }
                }
            }

            if num_resources.max_per_stage() > properties.max_per_stage_resources {
                return Err(PipelineLayoutCreationError::MaxPerStageResourcesExceeded {
                    provided: num_resources.max_per_stage(),
                    max_supported: properties.max_per_stage_resources,
                });
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03016
            if num_samplers.max_per_stage() > properties.max_per_stage_descriptor_samplers {
                return Err(
                    PipelineLayoutCreationError::MaxPerStageDescriptorSamplersExceeded {
                        provided: num_samplers.max_per_stage(),
                        max_supported: properties.max_per_stage_descriptor_samplers,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03017
            if num_uniform_buffers.max_per_stage()
                > properties.max_per_stage_descriptor_uniform_buffers
            {
                return Err(
                    PipelineLayoutCreationError::MaxPerStageDescriptorUniformBuffersExceeded {
                        provided: num_uniform_buffers.max_per_stage(),
                        max_supported: properties.max_per_stage_descriptor_uniform_buffers,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03018
            if num_storage_buffers.max_per_stage()
                > properties.max_per_stage_descriptor_storage_buffers
            {
                return Err(
                    PipelineLayoutCreationError::MaxPerStageDescriptorStorageBuffersExceeded {
                        provided: num_storage_buffers.max_per_stage(),
                        max_supported: properties.max_per_stage_descriptor_storage_buffers,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03019
            if num_sampled_images.max_per_stage()
                > properties.max_per_stage_descriptor_sampled_images
            {
                return Err(
                    PipelineLayoutCreationError::MaxPerStageDescriptorSampledImagesExceeded {
                        provided: num_sampled_images.max_per_stage(),
                        max_supported: properties.max_per_stage_descriptor_sampled_images,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03020
            if num_storage_images.max_per_stage()
                > properties.max_per_stage_descriptor_storage_images
            {
                return Err(
                    PipelineLayoutCreationError::MaxPerStageDescriptorStorageImagesExceeded {
                        provided: num_storage_images.max_per_stage(),
                        max_supported: properties.max_per_stage_descriptor_storage_images,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03021
            if num_input_attachments.max_per_stage()
                > properties.max_per_stage_descriptor_input_attachments
            {
                return Err(
                    PipelineLayoutCreationError::MaxPerStageDescriptorInputAttachmentsExceeded {
                        provided: num_input_attachments.max_per_stage(),
                        max_supported: properties.max_per_stage_descriptor_input_attachments,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03028
            if num_samplers.total > properties.max_descriptor_set_samplers {
                return Err(
                    PipelineLayoutCreationError::MaxDescriptorSetSamplersExceeded {
                        provided: num_samplers.total,
                        max_supported: properties.max_descriptor_set_samplers,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03029
            if num_uniform_buffers.total > properties.max_descriptor_set_uniform_buffers {
                return Err(
                    PipelineLayoutCreationError::MaxDescriptorSetUniformBuffersExceeded {
                        provided: num_uniform_buffers.total,
                        max_supported: properties.max_descriptor_set_uniform_buffers,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03030
            if num_uniform_buffers_dynamic > properties.max_descriptor_set_uniform_buffers_dynamic {
                return Err(
                    PipelineLayoutCreationError::MaxDescriptorSetUniformBuffersDynamicExceeded {
                        provided: num_uniform_buffers_dynamic,
                        max_supported: properties.max_descriptor_set_uniform_buffers_dynamic,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03031
            if num_storage_buffers.total > properties.max_descriptor_set_storage_buffers {
                return Err(
                    PipelineLayoutCreationError::MaxDescriptorSetStorageBuffersExceeded {
                        provided: num_storage_buffers.total,
                        max_supported: properties.max_descriptor_set_storage_buffers,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03032
            if num_storage_buffers_dynamic > properties.max_descriptor_set_storage_buffers_dynamic {
                return Err(
                    PipelineLayoutCreationError::MaxDescriptorSetStorageBuffersDynamicExceeded {
                        provided: num_storage_buffers_dynamic,
                        max_supported: properties.max_descriptor_set_storage_buffers_dynamic,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03033
            if num_sampled_images.total > properties.max_descriptor_set_sampled_images {
                return Err(
                    PipelineLayoutCreationError::MaxDescriptorSetSampledImagesExceeded {
                        provided: num_sampled_images.total,
                        max_supported: properties.max_descriptor_set_sampled_images,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03034
            if num_storage_images.total > properties.max_descriptor_set_storage_images {
                return Err(
                    PipelineLayoutCreationError::MaxDescriptorSetStorageImagesExceeded {
                        provided: num_storage_images.total,
                        max_supported: properties.max_descriptor_set_storage_images,
                    },
                );
            }

            // VUID-VkPipelineLayoutCreateInfo-descriptorType-03035
            if num_input_attachments.total > properties.max_descriptor_set_input_attachments {
                return Err(
                    PipelineLayoutCreationError::MaxDescriptorSetInputAttachmentsExceeded {
                        provided: num_input_attachments.total,
                        max_supported: properties.max_descriptor_set_input_attachments,
                    },
                );
            }
        }

        /* Check push constant ranges */

        push_constant_ranges.iter().try_fold(
            ash::vk::ShaderStageFlags::empty(),
            |total, range| {
                let stages = ash::vk::ShaderStageFlags::from(range.stages);

                // VUID-VkPushConstantRange-offset-00295
                // VUID-VkPushConstantRange-size-00296
                // VUID-VkPushConstantRange-size-00297
                // VUID-VkPushConstantRange-stageFlags-requiredbitmask
                assert!(
                    !stages.is_empty()
                        && (range.size % 4) == 0
                        && range.size != 0
                        && (range.size % 4) == 0
                );

                // VUID-VkPushConstantRange-offset-00294
                // VUID-VkPushConstantRange-size-00298
                if range.offset + range.size > properties.max_push_constants_size {
                    return Err(PipelineLayoutCreationError::MaxPushConstantsSizeExceeded {
                        provided: range.offset + range.size,
                        max_supported: properties.max_push_constants_size,
                    });
                }

                // VUID-VkPipelineLayoutCreateInfo-pPushConstantRanges-00292
                if !(total & stages).is_empty() {
                    return Err(PipelineLayoutCreationError::PushConstantRangesStageMultiple);
                }

                Ok(total | stages)
            },
        )?;

        Ok(())
    }

    unsafe fn create(
        device: &Device,
        create_info: &PipelineLayoutCreateInfo,
    ) -> Result<ash::vk::PipelineLayout, PipelineLayoutCreationError> {
        let PipelineLayoutCreateInfo {
            set_layouts,
            push_constant_ranges,
            _ne: _,
        } = create_info;

        let set_layouts: SmallVec<[_; 4]> =
            set_layouts.iter().map(|l| l.internal_object()).collect();

        let push_constant_ranges: SmallVec<[_; 4]> = push_constant_ranges
            .iter()
            .map(|range| ash::vk::PushConstantRange {
                stage_flags: range.stages.into(),
                offset: range.offset,
                size: range.size,
            })
            .collect();

        let create_info = ash::vk::PipelineLayoutCreateInfo {
            flags: ash::vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: push_constant_ranges.len() as u32,
            p_push_constant_ranges: push_constant_ranges.as_ptr(),
            ..Default::default()
        };

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_pipeline_layout(
                device.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(handle)
    }

    /// Returns the descriptor set layouts this pipeline layout was created from.
    #[inline]
    pub fn set_layouts(&self) -> &[Arc<DescriptorSetLayout>] {
        &self.set_layouts
    }

    /// Returns a slice containing the push constant ranges this pipeline layout was created from.
    ///
    /// The ranges are guaranteed to be sorted deterministically by offset, size, then stages.
    /// This means that two slices containing the same elements will always have the same order.
    #[inline]
    pub fn push_constant_ranges(&self) -> &[PushConstantRange] {
        &self.push_constant_ranges
    }

    /// Returns a slice containing the push constant ranges in with all disjoint stages.
    ///
    /// For example, if we have these `push_constant_ranges`:
    /// - `offset=0, size=4, stages=vertex`
    /// - `offset=0, size=12, stages=fragment`
    ///
    /// The returned value will be:
    /// - `offset=0, size=4, stages=vertex|fragment`
    /// - `offset=4, size=8, stages=fragment`
    ///
    /// The ranges are guaranteed to be sorted deterministically by offset, and
    /// guaranteed to be disjoint, meaning that there is no overlap between the ranges.
    #[inline]
    pub(crate) fn push_constant_ranges_disjoint(&self) -> &[PushConstantRange] {
        &self.push_constant_ranges_disjoint
    }

    /// Returns whether `self` is compatible with `other` for the given number of sets.
    pub fn is_compatible_with(&self, other: &PipelineLayout, num_sets: u32) -> bool {
        let num_sets = num_sets as usize;
        assert!(num_sets >= self.set_layouts.len());

        if self == other {
            return true;
        }

        if self.push_constant_ranges != other.push_constant_ranges {
            return false;
        }

        let other_sets = match other.set_layouts.get(0..num_sets) {
            Some(x) => x,
            None => return false,
        };

        self.set_layouts
            .iter()
            .zip(other_sets)
            .all(|(self_set_layout, other_set_layout)| {
                self_set_layout.is_compatible_with(other_set_layout)
            })
    }

    /// Makes sure that `self` is a superset of the provided descriptor set layouts and push
    /// constant ranges. Returns an `Err` if this is not the case.
    pub fn ensure_compatible_with_shader<'a>(
        &self,
        descriptor_requirements: impl IntoIterator<Item = ((u32, u32), &'a DescriptorRequirements)>,
        push_constant_range: Option<&PushConstantRange>,
    ) -> Result<(), PipelineLayoutSupersetError> {
        for ((set_num, binding_num), reqs) in descriptor_requirements.into_iter() {
            let layout_binding = self
                .set_layouts
                .get(set_num as usize)
                .and_then(|set_layout| set_layout.bindings().get(&binding_num));

            let layout_binding = match layout_binding {
                Some(x) => x,
                None => {
                    return Err(PipelineLayoutSupersetError::DescriptorMissing {
                        set_num,
                        binding_num,
                    })
                }
            };

            if let Err(error) = layout_binding.ensure_compatible_with_shader(reqs) {
                return Err(PipelineLayoutSupersetError::DescriptorRequirementsNotMet {
                    set_num,
                    binding_num,
                    error,
                });
            }
        }

        // FIXME: check push constants
        if let Some(range) = push_constant_range {
            for own_range in self.push_constant_ranges.iter() {
                if range.stages.intersects(&own_range.stages) &&       // check if it shares any stages
                    (range.offset < own_range.offset || // our range must start before and end after the given range
                        own_range.offset + own_range.size < range.offset + range.size)
                {
                    return Err(PipelineLayoutSupersetError::PushConstantRange {
                        first_range: *own_range,
                        second_range: *range,
                    });
                }
            }
        }

        Ok(())
    }
}

impl Drop for PipelineLayout {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0.destroy_pipeline_layout(
                self.device.internal_object(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

unsafe impl VulkanObject for PipelineLayout {
    type Object = ash::vk::PipelineLayout;

    fn internal_object(&self) -> Self::Object {
        self.handle
    }
}

unsafe impl DeviceOwned for PipelineLayout {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for PipelineLayout {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for PipelineLayout {}

impl Hash for PipelineLayout {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Error that can happen when creating a pipeline layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineLayoutCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// The number of elements in `set_layouts` is greater than the
    /// [`max_bound_descriptor_sets`](crate::device::Properties::max_bound_descriptor_sets) limit.
    MaxBoundDescriptorSetsExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::Sampler`],
    /// [`DescriptorType::CombinedImageSampler`] and [`DescriptorType::UniformTexelBuffer`]
    /// descriptors than the
    /// [`max_descriptor_set_samplers`](crate::device::Properties::max_descriptor_set_samplers)
    /// limit.
    MaxDescriptorSetSamplersExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::UniformBuffer`] descriptors than the
    /// [`max_descriptor_set_uniform_buffers`](crate::device::Properties::max_descriptor_set_uniform_buffers)
    /// limit.
    MaxDescriptorSetUniformBuffersExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::UniformBufferDynamic`] descriptors than the
    /// [`max_descriptor_set_uniform_buffers_dynamic`](crate::device::Properties::max_descriptor_set_uniform_buffers_dynamic)
    /// limit.
    MaxDescriptorSetUniformBuffersDynamicExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::StorageBuffer`] descriptors than the
    /// [`max_descriptor_set_storage_buffers`](crate::device::Properties::max_descriptor_set_storage_buffers)
    /// limit.
    MaxDescriptorSetStorageBuffersExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::StorageBufferDynamic`] descriptors than the
    /// [`max_descriptor_set_storage_buffers_dynamic`](crate::device::Properties::max_descriptor_set_storage_buffers_dynamic)
    /// limit.
    MaxDescriptorSetStorageBuffersDynamicExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::SampledImage`] and
    /// [`DescriptorType::CombinedImageSampler`] descriptors than the
    /// [`max_descriptor_set_sampled_images`](crate::device::Properties::max_descriptor_set_sampled_images)
    /// limit.
    MaxDescriptorSetSampledImagesExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::StorageImage`] and
    /// [`DescriptorType::StorageTexelBuffer`] descriptors than the
    /// [`max_descriptor_set_storage_images`](crate::device::Properties::max_descriptor_set_storage_images)
    /// limit.
    MaxDescriptorSetStorageImagesExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::InputAttachment`] descriptors than the
    /// [`max_descriptor_set_input_attachments`](crate::device::Properties::max_descriptor_set_input_attachments)
    /// limit.
    MaxDescriptorSetInputAttachmentsExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more bound resources in a single stage than the
    /// [`max_per_stage_resources`](crate::device::Properties::max_per_stage_resources)
    /// limit.
    MaxPerStageResourcesExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::Sampler`] and
    /// [`DescriptorType::CombinedImageSampler`] descriptors in a single stage than the
    /// [`max_per_stage_descriptor_samplers`](crate::device::Properties::max_per_stage_descriptor_samplers)
    /// limit.
    MaxPerStageDescriptorSamplersExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::UniformBuffer`] and
    /// [`DescriptorType::UniformBufferDynamic`] descriptors in a single stage than the
    /// [`max_per_stage_descriptor_uniform_buffers`](crate::device::Properties::max_per_stage_descriptor_uniform_buffers)
    /// limit.
    MaxPerStageDescriptorUniformBuffersExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::StorageBuffer`] and
    /// [`DescriptorType::StorageBufferDynamic`] descriptors in a single stage than the
    /// [`max_per_stage_descriptor_storage_buffers`](crate::device::Properties::max_per_stage_descriptor_storage_buffers)
    /// limit.
    MaxPerStageDescriptorStorageBuffersExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::SampledImage`],
    /// [`DescriptorType::CombinedImageSampler`] and [`DescriptorType::UniformTexelBuffer`]
    /// descriptors in a single stage than the
    /// [`max_per_stage_descriptor_sampled_images`](crate::device::Properties::max_per_stage_descriptor_sampled_images)
    /// limit.
    MaxPerStageDescriptorSampledImagesExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::StorageImage`] and
    /// [`DescriptorType::StorageTexelBuffer`] descriptors in a single stage than the
    /// [`max_per_stage_descriptor_storage_images`](crate::device::Properties::max_per_stage_descriptor_storage_images)
    /// limit.
    MaxPerStageDescriptorStorageImagesExceeded { provided: u32, max_supported: u32 },

    /// The `set_layouts` contain more [`DescriptorType::InputAttachment`] descriptors in a single
    /// stage than the
    /// [`max_per_stage_descriptor_input_attachments`](crate::device::Properties::max_per_stage_descriptor_input_attachments)
    /// limit.
    MaxPerStageDescriptorInputAttachmentsExceeded { provided: u32, max_supported: u32 },

    /// An element in `push_constant_ranges` has an `offset + size` greater than the
    /// [`max_push_constants_size`](crate::device::Properties::max_push_constants_size) limit.
    MaxPushConstantsSizeExceeded { provided: u32, max_supported: u32 },

    /// A shader stage appears in multiple elements of `push_constant_ranges`.
    PushConstantRangesStageMultiple,

    /// Multiple elements of `set_layouts` have `push_descriptor` enabled.
    SetLayoutsPushDescriptorMultiple,
}

impl error::Error for PipelineLayoutCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for PipelineLayoutCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::OomError(_) => write!(fmt, "not enough memory available"),
            Self::MaxBoundDescriptorSetsExceeded { provided, max_supported } => write!(
                fmt,
                "the number of elements in `set_layouts` ({}) is greater than the `max_bound_descriptor_sets` limit ({})",
                provided, max_supported,
            ),
            Self::MaxDescriptorSetSamplersExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::Sampler` and `DescriptorType::CombinedImageSampler` descriptors ({}) than the `max_descriptor_set_samplers` limit ({})",
                provided, max_supported,
            ),
            Self::MaxDescriptorSetUniformBuffersExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::UniformBuffer` descriptors ({}) than the `max_descriptor_set_uniform_buffers` limit ({})",
                provided, max_supported,
            ),
            Self::MaxDescriptorSetUniformBuffersDynamicExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::UniformBufferDynamic` descriptors ({}) than the `max_descriptor_set_uniform_buffers_dynamic` limit ({})",
                provided, max_supported,
            ),
            Self::MaxDescriptorSetStorageBuffersExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::StorageBuffer` descriptors ({}) than the `max_descriptor_set_storage_buffers` limit ({})",
                provided, max_supported,
            ),
            Self::MaxDescriptorSetStorageBuffersDynamicExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::StorageBufferDynamic` descriptors ({}) than the `max_descriptor_set_storage_buffers_dynamic` limit ({})",
                provided, max_supported,
            ),
            Self::MaxDescriptorSetSampledImagesExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::SampledImage`, `DescriptorType::CombinedImageSampler` and `DescriptorType::UniformTexelBuffer` descriptors ({}) than the `max_descriptor_set_sampled_images` limit ({})",
                provided, max_supported,
            ),
            Self::MaxDescriptorSetStorageImagesExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::StorageImage` and `DescriptorType::StorageTexelBuffer` descriptors ({}) than the `max_descriptor_set_storage_images` limit ({})",
                provided, max_supported,
            ),
            Self::MaxDescriptorSetInputAttachmentsExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::InputAttachment` descriptors ({}) than the `max_descriptor_set_input_attachments` limit ({})",
                provided, max_supported,
            ),
            Self::MaxPerStageResourcesExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more bound resources ({}) in a single stage than the `max_per_stage_resources` limit ({})",
                provided, max_supported,
            ),
            Self::MaxPerStageDescriptorSamplersExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::Sampler` and `DescriptorType::CombinedImageSampler` descriptors ({}) in a single stage than the `max_per_stage_descriptor_set_samplers` limit ({})",
                provided, max_supported,
            ),
            Self::MaxPerStageDescriptorUniformBuffersExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::UniformBuffer` and `DescriptorType::UniformBufferDynamic` descriptors ({}) in a single stage than the `max_per_stage_descriptor_set_uniform_buffers` limit ({})",
                provided, max_supported,
            ),
            Self::MaxPerStageDescriptorStorageBuffersExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::StorageBuffer` and `DescriptorType::StorageBufferDynamic` descriptors ({}) in a single stage than the `max_per_stage_descriptor_set_storage_buffers` limit ({})",
                provided, max_supported,
            ),
            Self::MaxPerStageDescriptorSampledImagesExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::SampledImage`, `DescriptorType::CombinedImageSampler` and `DescriptorType::UniformTexelBuffer` descriptors ({}) in a single stage than the `max_per_stage_descriptor_set_sampled_images` limit ({})",
                provided, max_supported,
            ),
            Self::MaxPerStageDescriptorStorageImagesExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::StorageImage` and `DescriptorType::StorageTexelBuffer` descriptors ({}) in a single stage than the `max_per_stage_descriptor_set_storage_images` limit ({})",
                provided, max_supported,
            ),
            Self::MaxPerStageDescriptorInputAttachmentsExceeded { provided, max_supported } => write!(
                fmt,
                "the `set_layouts` contain more `DescriptorType::InputAttachment` descriptors ({}) in a single stage than the `max_per_stage_descriptor_set_input_attachments` limit ({})",
                provided, max_supported,
            ),
            Self::MaxPushConstantsSizeExceeded { provided, max_supported } => write!(
                fmt,
                "an element in `push_constant_ranges` has an `offset + size` ({}) greater than the `max_push_constants_size` limit ({})",
                provided, max_supported,
            ),
            Self::PushConstantRangesStageMultiple => write!(
                fmt,
                "a shader stage appears in multiple elements of `push_constant_ranges`",
            ),
            Self::SetLayoutsPushDescriptorMultiple => write!(
                fmt,
                "multiple elements of `set_layouts` have `push_descriptor` enabled"
            ),
        }
    }
}

impl From<OomError> for PipelineLayoutCreationError {
    #[inline]
    fn from(err: OomError) -> PipelineLayoutCreationError {
        PipelineLayoutCreationError::OomError(err)
    }
}

impl From<Error> for PipelineLayoutCreationError {
    #[inline]
    fn from(err: Error) -> PipelineLayoutCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                PipelineLayoutCreationError::OomError(OomError::from(err))
            }
            err @ Error::OutOfDeviceMemory => {
                PipelineLayoutCreationError::OomError(OomError::from(err))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// Error when checking whether a pipeline layout is a superset of another one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineLayoutSupersetError {
    DescriptorMissing {
        set_num: u32,
        binding_num: u32,
    },
    DescriptorRequirementsNotMet {
        set_num: u32,
        binding_num: u32,
        error: DescriptorRequirementsNotMet,
    },
    PushConstantRange {
        first_range: PushConstantRange,
        second_range: PushConstantRange,
    },
}

impl error::Error for PipelineLayoutSupersetError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            PipelineLayoutSupersetError::DescriptorRequirementsNotMet { ref error, .. } => {
                Some(error)
            }
            _ => None,
        }
    }
}

impl fmt::Display for PipelineLayoutSupersetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            PipelineLayoutSupersetError::DescriptorRequirementsNotMet { set_num, binding_num, .. } => write!(
                fmt,
                "the descriptor at set {} binding {} does not meet the requirements",
                set_num, binding_num
            ),
            PipelineLayoutSupersetError::DescriptorMissing {
                set_num,
                binding_num,
            } => write!(
                fmt,
                "a descriptor at set {} binding {} is required by the shaders, but is missing from the pipeline layout",
                set_num, binding_num
            ),
            PipelineLayoutSupersetError::PushConstantRange {
                first_range,
                second_range,
            } => {
                writeln!(
                    fmt,
                    "our range did not completely encompass the other range"
                )?;
                writeln!(fmt, "    our stages: {:?}", first_range.stages)?;
                writeln!(
                    fmt,
                    "    our range: {} - {}",
                    first_range.offset,
                    first_range.offset + first_range.size
                )?;
                writeln!(fmt, "    other stages: {:?}", second_range.stages)?;
                write!(
                    fmt,
                    "    other range: {} - {}",
                    second_range.offset,
                    second_range.offset + second_range.size
                )
            }
        }
    }
}

/// Parameters to create a new `PipelineLayout`.
#[derive(Clone, Debug)]
pub struct PipelineLayoutCreateInfo {
    /// The descriptor set layouts that should be part of the pipeline layout.
    ///
    /// They are provided in order of set number.
    ///
    /// The default value is empty.
    pub set_layouts: Vec<Arc<DescriptorSetLayout>>,

    /// The ranges of push constants that the pipeline will access.
    ///
    /// A shader stage can only appear in one element of the list, but it is possible to combine
    /// ranges for multiple shader stages if they are the same.
    ///
    /// The default value is empty.
    pub push_constant_ranges: Vec<PushConstantRange>,

    pub _ne: crate::NonExhaustive,
}

impl Default for PipelineLayoutCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            set_layouts: Vec::new(),
            push_constant_ranges: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Description of a range of the push constants of a pipeline layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PushConstantRange {
    /// The stages which can access this range. A stage can access at most one push constant range.
    ///
    /// The default value is [`ShaderStages::none()`], which must be overridden.
    pub stages: ShaderStages,

    /// Offset in bytes from the start of the push constants to this range.
    ///
    /// The value must be a multiple of 4.
    ///
    /// The default value is `0`.
    pub offset: u32,

    /// Size in bytes of the range.
    ///
    /// The value must be a multiple of 4, and not 0.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: u32,
}

impl Default for PushConstantRange {
    #[inline]
    fn default() -> Self {
        Self {
            stages: ShaderStages::none(),
            offset: 0,
            size: 0,
        }
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
        [
            self.compute,
            self.vertex,
            self.tess_ctl,
            self.tess_eval,
            self.geometry,
            self.frag,
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange},
        shader::ShaderStages,
    };

    use super::PipelineLayout;

    #[test]
    fn push_constant_ranges_disjoint() {
        let test_cases = [
            // input:
            // - `0..12`, stage=fragment
            // - `0..40`, stage=vertex
            //
            // output:
            // - `0..12`, stage=fragment|vertex
            // - `12..40`, stage=vertex
            (
                &[
                    PushConstantRange {
                        stages: ShaderStages {
                            fragment: true,
                            ..Default::default()
                        },
                        offset: 0,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            vertex: true,
                            ..Default::default()
                        },
                        offset: 0,
                        size: 40,
                    },
                ][..],
                &[
                    PushConstantRange {
                        stages: ShaderStages {
                            vertex: true,
                            fragment: true,
                            ..Default::default()
                        },
                        offset: 0,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            vertex: true,
                            ..Default::default()
                        },
                        offset: 12,
                        size: 28,
                    },
                ][..],
            ),
            // input:
            // - `0..12`, stage=fragment
            // - `4..40`, stage=vertex
            //
            // output:
            // - `0..4`, stage=fragment
            // - `4..12`, stage=fragment|vertex
            // - `12..40`, stage=vertex
            (
                &[
                    PushConstantRange {
                        stages: ShaderStages {
                            fragment: true,
                            ..Default::default()
                        },
                        offset: 0,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            vertex: true,
                            ..Default::default()
                        },
                        offset: 4,
                        size: 36,
                    },
                ][..],
                &[
                    PushConstantRange {
                        stages: ShaderStages {
                            fragment: true,
                            ..Default::default()
                        },
                        offset: 0,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            fragment: true,
                            vertex: true,
                            ..Default::default()
                        },
                        offset: 4,
                        size: 8,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            vertex: true,
                            ..Default::default()
                        },
                        offset: 12,
                        size: 28,
                    },
                ][..],
            ),
            // input:
            // - `0..12`, stage=fragment
            // - `8..20`, stage=compute
            // - `4..16`, stage=vertex
            // - `8..32`, stage=tess_ctl
            //
            // output:
            // - `0..4`, stage=fragment
            // - `4..8`, stage=fragment|vertex
            // - `8..16`, stage=fragment|vertex|compute|tess_ctl
            // - `16..20`, stage=compute|tess_ctl
            // - `20..32` stage=tess_ctl
            (
                &[
                    PushConstantRange {
                        stages: ShaderStages {
                            fragment: true,
                            ..Default::default()
                        },
                        offset: 0,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            compute: true,
                            ..Default::default()
                        },
                        offset: 8,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            vertex: true,
                            ..Default::default()
                        },
                        offset: 4,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            tessellation_control: true,
                            ..Default::default()
                        },
                        offset: 8,
                        size: 24,
                    },
                ][..],
                &[
                    PushConstantRange {
                        stages: ShaderStages {
                            fragment: true,
                            ..Default::default()
                        },
                        offset: 0,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            fragment: true,
                            vertex: true,
                            ..Default::default()
                        },
                        offset: 4,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            vertex: true,
                            fragment: true,
                            compute: true,
                            tessellation_control: true,
                            ..Default::default()
                        },
                        offset: 8,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            vertex: true,
                            compute: true,
                            tessellation_control: true,
                            ..Default::default()
                        },
                        offset: 12,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            compute: true,
                            tessellation_control: true,
                            ..Default::default()
                        },
                        offset: 16,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages {
                            tessellation_control: true,
                            ..Default::default()
                        },
                        offset: 20,
                        size: 12,
                    },
                ][..],
            ),
        ];

        let (device, _) = gfx_dev_and_queue!();

        for (input, expected) in test_cases {
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    push_constant_ranges: input.into(),
                    ..Default::default()
                },
            )
            .unwrap();

            assert_eq!(layout.push_constant_ranges_disjoint.as_slice(), expected);
        }
    }
}

/* TODO: restore
#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::Arc;
    use descriptor::descriptor::ShaderStages;
    use descriptor::descriptor_set::DescriptorSetLayout;
    use descriptor::pipeline_layout::sys::PipelineLayout;
    use descriptor::pipeline_layout::sys::PipelineLayoutCreationError;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = PipelineLayout::new(&device, iter::empty(), iter::empty()).unwrap();
    }

    #[test]
    fn wrong_device_panic() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let set = match DescriptorSetLayout::raw(device1, iter::empty()) {
            Ok(s) => Arc::new(s),
            Err(_) => return
        };

        assert_should_panic!({
            let _ = PipelineLayout::new(&device2, Some(&set), iter::empty());
        });
    }

    #[test]
    fn invalid_push_constant_stages() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 8, ShaderStages::none());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }

    #[test]
    fn invalid_push_constant_size1() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 0, ShaderStages::all_graphics());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }

    #[test]
    fn invalid_push_constant_size2() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 11, ShaderStages::all_graphics());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }
}
*/
