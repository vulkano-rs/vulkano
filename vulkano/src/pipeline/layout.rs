// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A pipeline layout describes the layout of descriptor sets and push constants used by a pipeline.
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
//! This pipeline is used to decide where in memory Vulkan should write the new data. The
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
//!
//! By default, creating a pipeline automatically builds a new pipeline layout object describing the
//! union of all the descriptors and push constants of all the shaders used by the pipeline.
//! However, it is also possible to create a pipeline layout separately, and provide that to the
//! pipeline constructor. This can in some cases be more efficient than using the auto-generated
//! pipeline layouts.

use crate::check_errors;
use crate::descriptor_set::layout::DescriptorRequirementsNotMet;
use crate::descriptor_set::layout::DescriptorSetLayout;
use crate::descriptor_set::layout::DescriptorSetLayoutError;
use crate::descriptor_set::layout::DescriptorType;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::device::Properties;
use crate::shader::DescriptorRequirements;
use crate::shader::ShaderStages;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// Describes the layout of descriptor sets and push constants that are made available to shaders.
pub struct PipelineLayout {
    handle: ash::vk::PipelineLayout,
    device: Arc<Device>,
    descriptor_set_layouts: SmallVec<[Arc<DescriptorSetLayout>; 4]>,
    push_constant_ranges: SmallVec<[PipelineLayoutPcRange; 5]>,
    overlapping_push_constant_ranges: SmallVec<[PipelineLayoutPcRange; 5]>,
}

impl PipelineLayout {
    /// Creates a new `PipelineLayout`.
    #[inline]
    pub fn new<D, P>(
        device: Arc<Device>,
        descriptor_set_layouts: D,
        push_constant_ranges: P,
    ) -> Result<Arc<PipelineLayout>, PipelineLayoutCreationError>
    where
        D: IntoIterator<Item = Arc<DescriptorSetLayout>>,
        P: IntoIterator<Item = PipelineLayoutPcRange>,
    {
        let fns = device.fns();
        let descriptor_set_layouts: SmallVec<[Arc<DescriptorSetLayout>; 4]> =
            descriptor_set_layouts.into_iter().collect();

        if descriptor_set_layouts
            .iter()
            .filter(|layout| layout.desc().is_push_descriptor())
            .count()
            > 1
        {
            return Err(PipelineLayoutCreationError::MultiplePushDescriptor);
        }

        let mut push_constant_ranges: SmallVec<[PipelineLayoutPcRange; 5]> =
            push_constant_ranges.into_iter().collect();

        // Check for overlapping stages
        for (a_id, a) in push_constant_ranges.iter().enumerate() {
            for b in push_constant_ranges.iter().skip(a_id + 1) {
                if a.stages.intersects(&b.stages) {
                    return Err(PipelineLayoutCreationError::PushConstantsConflict {
                        first_range: *a,
                        second_range: *b,
                    });
                }
            }
        }

        // Sort the ranges for the purpose of comparing for equality.
        // The stage mask is guaranteed to be unique by the above check, so it's a suitable
        // sorting key.
        push_constant_ranges.sort_unstable_by_key(|range| {
            (
                range.offset,
                range.size,
                ash::vk::ShaderStageFlags::from(range.stages),
            )
        });

        let mut overlapping_push_constant_ranges: SmallVec<[PipelineLayoutPcRange; 5]> =
            SmallVec::new();

        if !push_constant_ranges.is_empty() {
            let mut min_offset = push_constant_ranges[0].offset;
            loop {
                let mut max_offset = u32::MAX;
                let mut stages = ShaderStages::none();

                for range in &push_constant_ranges {
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

                overlapping_push_constant_ranges.push(PipelineLayoutPcRange {
                    offset: min_offset,
                    size: max_offset - min_offset,
                    stages,
                });
                // prepare for next range
                min_offset = max_offset;
            }
        }

        // Check against device limits
        check_desc_against_limits(
            device.physical_device().properties(),
            &descriptor_set_layouts,
            &push_constant_ranges,
        )?;

        // Grab the list of `vkDescriptorSetLayout` objects from `layouts`.
        let layouts_ids = descriptor_set_layouts
            .iter()
            .map(|l| l.internal_object())
            .collect::<SmallVec<[_; 4]>>();

        // Builds a list of `vkPushConstantRange` that describe the push constants.
        let push_constants = {
            let mut out: SmallVec<[_; 4]> = SmallVec::new();

            for &PipelineLayoutPcRange {
                offset,
                size,
                stages,
            } in &push_constant_ranges
            {
                if stages == ShaderStages::none() || size == 0 || (size % 4) != 0 {
                    return Err(PipelineLayoutCreationError::InvalidPushConstant);
                }

                out.push(ash::vk::PushConstantRange {
                    stage_flags: stages.into(),
                    offset,
                    size,
                });
            }

            out
        };

        // Each bit of `stageFlags` must only be present in a single push constants range.
        // We check that with a debug_assert because it's supposed to be enforced by the
        // `PipelineLayoutDesc`.
        debug_assert!({
            let mut stages = ash::vk::ShaderStageFlags::empty();
            let mut outcome = true;
            for pc in push_constants.iter() {
                if !(stages & pc.stage_flags).is_empty() {
                    outcome = false;
                    break;
                }
                stages &= pc.stage_flags;
            }
            outcome
        });

        // FIXME: it is not legal to pass eg. the TESSELLATION_SHADER bit when the device doesn't
        //        have tess shaders enabled

        // Build the final object.
        let handle = unsafe {
            let infos = ash::vk::PipelineLayoutCreateInfo {
                flags: ash::vk::PipelineLayoutCreateFlags::empty(),
                set_layout_count: layouts_ids.len() as u32,
                p_set_layouts: layouts_ids.as_ptr(),
                push_constant_range_count: push_constants.len() as u32,
                p_push_constant_ranges: push_constants.as_ptr(),
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_pipeline_layout(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(PipelineLayout {
            handle,
            device: device.clone(),
            descriptor_set_layouts,
            push_constant_ranges,
            overlapping_push_constant_ranges,
        }))
    }

    /// Returns the descriptor set layouts this pipeline layout was created from.
    #[inline]
    pub fn descriptor_set_layouts(&self) -> &[Arc<DescriptorSetLayout>] {
        &self.descriptor_set_layouts
    }

    /// Returns a slice containing the push constant ranges this pipeline layout was created from.
    ///
    /// The ranges are guaranteed to be sorted deterministically by offset, size, then stages.
    /// This means that two slices containing the same elements will always have the same order.
    #[inline]
    pub fn push_constant_ranges(&self) -> &[PipelineLayoutPcRange] {
        &self.push_constant_ranges
    }

    /// Returns a slice containing the push constant ranges in with all overlapping stages.
    ///
    /// For example, if we have these `push_constant_ranges`:
    /// - `offset=0, size=4, stages=vertex`
    /// - `offset=0, size=12, stages=fragment`
    ///
    /// The returned value will be:
    /// - `offset=0, size=4, stages=vertex|fragment`
    /// - `offset=4, size=8, stages=fragment`
    #[inline]
    pub(crate) fn overlapping_push_constant_ranges(&self) -> &[PipelineLayoutPcRange] {
        &self.overlapping_push_constant_ranges
    }

    /// Returns whether `self` is compatible with `other` for the given number of sets.
    pub fn is_compatible_with(&self, other: &PipelineLayout, num_sets: u32) -> bool {
        let num_sets = num_sets as usize;
        assert!(num_sets >= self.descriptor_set_layouts.len());

        if self.handle == other.handle {
            return true;
        }

        if self.push_constant_ranges != other.push_constant_ranges {
            return false;
        }

        let other_sets = match other.descriptor_set_layouts.get(0..num_sets) {
            Some(x) => x,
            None => return false,
        };

        self.descriptor_set_layouts.iter().zip(other_sets).all(
            |(self_set_layout, other_set_layout)| {
                self_set_layout.is_compatible_with(other_set_layout)
            },
        )
    }

    /// Makes sure that `self` is a superset of the provided descriptor set layouts and push
    /// constant ranges. Returns an `Err` if this is not the case.
    pub fn ensure_compatible_with_shader<'a>(
        &self,
        descriptor_requirements: impl IntoIterator<Item = ((u32, u32), &'a DescriptorRequirements)>,
        push_constant_range: Option<&PipelineLayoutPcRange>,
    ) -> Result<(), PipelineLayoutSupersetError> {
        for ((set_num, binding_num), reqs) in descriptor_requirements.into_iter() {
            let descriptor_desc = self
                .descriptor_set_layouts
                .get(set_num as usize)
                .and_then(|set_desc| set_desc.descriptor(binding_num));

            let descriptor_desc = match descriptor_desc {
                Some(x) => x,
                None => {
                    return Err(PipelineLayoutSupersetError::DescriptorMissing {
                        set_num,
                        binding_num,
                    })
                }
            };

            if let Err(error) = descriptor_desc.ensure_compatible_with_shader(reqs) {
                return Err(PipelineLayoutSupersetError::DescriptorRequirementsNotMet {
                    set_num,
                    binding_num,
                    error,
                });
            }
        }

        // FIXME: check push constants
        if let Some(range) = push_constant_range {
            for own_range in self.push_constant_ranges.as_ref().into_iter() {
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

unsafe impl DeviceOwned for PipelineLayout {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for PipelineLayout {
    type Object = ash::vk::PipelineLayout;

    fn internal_object(&self) -> Self::Object {
        self.handle
    }
}

impl fmt::Debug for PipelineLayout {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_struct("PipelineLayout")
            .field("raw", &self.handle)
            .field("device", &self.device)
            .field("descriptor_set_layouts", &self.descriptor_set_layouts)
            .field("push_constant_ranges", &self.push_constant_ranges)
            .finish()
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

/// Error that can happen when creating a pipeline layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineLayoutCreationError {
    /// Not enough memory.
    OomError(OomError),
    /// The pipeline layout description doesn't fulfill the limit requirements.
    LimitsError(PipelineLayoutLimitsError),
    /// One of the push constants range didn't obey the rules. The list of stages must not be
    /// empty, the size must not be 0, and the size must be a multiple or 4.
    InvalidPushConstant,
    /// More than one descriptor set layout was set for push descriptors.
    MultiplePushDescriptor,
    /// Conflict between different push constants ranges.
    PushConstantsConflict {
        first_range: PipelineLayoutPcRange,
        second_range: PipelineLayoutPcRange,
    },
    /// One of the set layouts has an error.
    SetLayoutError(DescriptorSetLayoutError),
}

impl error::Error for PipelineLayoutCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            Self::LimitsError(ref err) => Some(err),
            Self::SetLayoutError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for PipelineLayoutCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::OomError(_) => write!(fmt, "not enough memory available"),
            Self::LimitsError(_) => {
                write!(
                    fmt,
                    "the pipeline layout description doesn't fulfill the limit requirements"
                )
            }
            Self::InvalidPushConstant => {
                write!(fmt, "one of the push constants range didn't obey the rules")
            }
            Self::MultiplePushDescriptor => {
                write!(
                    fmt,
                    "more than one descriptor set layout was set for push descriptors"
                )
            }
            Self::PushConstantsConflict { .. } => {
                write!(fmt, "conflict between different push constants ranges")
            }
            Self::SetLayoutError(_) => write!(fmt, "one of the sets has an error"),
        }
    }
}

impl From<OomError> for PipelineLayoutCreationError {
    #[inline]
    fn from(err: OomError) -> PipelineLayoutCreationError {
        PipelineLayoutCreationError::OomError(err)
    }
}

impl From<PipelineLayoutLimitsError> for PipelineLayoutCreationError {
    #[inline]
    fn from(err: PipelineLayoutLimitsError) -> PipelineLayoutCreationError {
        PipelineLayoutCreationError::LimitsError(err)
    }
}

impl From<DescriptorSetLayoutError> for PipelineLayoutCreationError {
    #[inline]
    fn from(err: DescriptorSetLayoutError) -> PipelineLayoutCreationError {
        PipelineLayoutCreationError::SetLayoutError(err)
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
        first_range: PipelineLayoutPcRange,
        second_range: PipelineLayoutPcRange,
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

/// Description of a range of the push constants of a pipeline layout.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PipelineLayoutPcRange {
    /// Offset in bytes from the start of the push constants to this range.
    pub offset: u32,
    /// Size in bytes of the range.
    pub size: u32,
    /// The stages which can access this range.
    /// A stage can access at most one push constant range.
    pub stages: ShaderStages,
}

/// Checks whether the pipeline layout description fulfills the device limits requirements.
fn check_desc_against_limits(
    properties: &Properties,
    descriptor_set_layouts: &[Arc<DescriptorSetLayout>],
    push_constants_ranges: &[PipelineLayoutPcRange],
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

    for set in descriptor_set_layouts {
        for descriptor in (0..set.num_bindings()).filter_map(|i| set.descriptor(i).map(|d| d)) {
            num_resources.increment(descriptor.descriptor_count, &descriptor.stages);

            match descriptor.ty {
                // TODO:
                DescriptorType::Sampler => {
                    num_samplers.increment(descriptor.descriptor_count, &descriptor.stages);
                }
                DescriptorType::CombinedImageSampler => {
                    num_samplers.increment(descriptor.descriptor_count, &descriptor.stages);
                    num_sampled_images.increment(descriptor.descriptor_count, &descriptor.stages);
                }
                DescriptorType::SampledImage | DescriptorType::UniformTexelBuffer => {
                    num_sampled_images.increment(descriptor.descriptor_count, &descriptor.stages);
                }
                DescriptorType::StorageImage | DescriptorType::StorageTexelBuffer => {
                    num_storage_images.increment(descriptor.descriptor_count, &descriptor.stages);
                }
                DescriptorType::UniformBuffer => {
                    num_uniform_buffers.increment(descriptor.descriptor_count, &descriptor.stages);
                }
                DescriptorType::UniformBufferDynamic => {
                    num_uniform_buffers.increment(descriptor.descriptor_count, &descriptor.stages);
                    num_uniform_buffers_dynamic += 1;
                }
                DescriptorType::StorageBuffer => {
                    num_storage_buffers.increment(descriptor.descriptor_count, &descriptor.stages);
                }
                DescriptorType::StorageBufferDynamic => {
                    num_storage_buffers.increment(descriptor.descriptor_count, &descriptor.stages);
                    num_storage_buffers_dynamic += 1;
                }
                DescriptorType::InputAttachment => {
                    num_input_attachments
                        .increment(descriptor.descriptor_count, &descriptor.stages);
                }
            }
        }
    }

    if descriptor_set_layouts.len() > properties.max_bound_descriptor_sets as usize {
        return Err(PipelineLayoutLimitsError::MaxDescriptorSetsLimitExceeded {
            limit: properties.max_bound_descriptor_sets as usize,
            requested: descriptor_set_layouts.len(),
        });
    }

    if num_resources.max_per_stage() > properties.max_per_stage_resources {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageResourcesLimitExceeded {
                limit: properties.max_per_stage_resources,
                requested: num_resources.max_per_stage(),
            },
        );
    }

    if num_samplers.max_per_stage() > properties.max_per_stage_descriptor_samplers {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorSamplersLimitExceeded {
                limit: properties.max_per_stage_descriptor_samplers,
                requested: num_samplers.max_per_stage(),
            },
        );
    }
    if num_uniform_buffers.max_per_stage() > properties.max_per_stage_descriptor_uniform_buffers {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorUniformBuffersLimitExceeded {
                limit: properties.max_per_stage_descriptor_uniform_buffers,
                requested: num_uniform_buffers.max_per_stage(),
            },
        );
    }
    if num_storage_buffers.max_per_stage() > properties.max_per_stage_descriptor_storage_buffers {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorStorageBuffersLimitExceeded {
                limit: properties.max_per_stage_descriptor_storage_buffers,
                requested: num_storage_buffers.max_per_stage(),
            },
        );
    }
    if num_sampled_images.max_per_stage() > properties.max_per_stage_descriptor_sampled_images {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorSampledImagesLimitExceeded {
                limit: properties.max_per_stage_descriptor_sampled_images,
                requested: num_sampled_images.max_per_stage(),
            },
        );
    }
    if num_storage_images.max_per_stage() > properties.max_per_stage_descriptor_storage_images {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorStorageImagesLimitExceeded {
                limit: properties.max_per_stage_descriptor_storage_images,
                requested: num_storage_images.max_per_stage(),
            },
        );
    }
    if num_input_attachments.max_per_stage() > properties.max_per_stage_descriptor_input_attachments
    {
        return Err(
            PipelineLayoutLimitsError::MaxPerStageDescriptorInputAttachmentsLimitExceeded {
                limit: properties.max_per_stage_descriptor_input_attachments,
                requested: num_input_attachments.max_per_stage(),
            },
        );
    }

    if num_samplers.total > properties.max_descriptor_set_samplers {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetSamplersLimitExceeded {
                limit: properties.max_descriptor_set_samplers,
                requested: num_samplers.total,
            },
        );
    }
    if num_uniform_buffers.total > properties.max_descriptor_set_uniform_buffers {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetUniformBuffersLimitExceeded {
                limit: properties.max_descriptor_set_uniform_buffers,
                requested: num_uniform_buffers.total,
            },
        );
    }
    if num_uniform_buffers_dynamic > properties.max_descriptor_set_uniform_buffers_dynamic {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetUniformBuffersDynamicLimitExceeded {
                limit: properties.max_descriptor_set_uniform_buffers_dynamic,
                requested: num_uniform_buffers_dynamic,
            },
        );
    }
    if num_storage_buffers.total > properties.max_descriptor_set_storage_buffers {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetStorageBuffersLimitExceeded {
                limit: properties.max_descriptor_set_storage_buffers,
                requested: num_storage_buffers.total,
            },
        );
    }
    if num_storage_buffers_dynamic > properties.max_descriptor_set_storage_buffers_dynamic {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetStorageBuffersDynamicLimitExceeded {
                limit: properties.max_descriptor_set_storage_buffers_dynamic,
                requested: num_storage_buffers_dynamic,
            },
        );
    }
    if num_sampled_images.total > properties.max_descriptor_set_sampled_images {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetSampledImagesLimitExceeded {
                limit: properties.max_descriptor_set_sampled_images,
                requested: num_sampled_images.total,
            },
        );
    }
    if num_storage_images.total > properties.max_descriptor_set_storage_images {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetStorageImagesLimitExceeded {
                limit: properties.max_descriptor_set_storage_images,
                requested: num_storage_images.total,
            },
        );
    }
    if num_input_attachments.total > properties.max_descriptor_set_input_attachments {
        return Err(
            PipelineLayoutLimitsError::MaxDescriptorSetInputAttachmentsLimitExceeded {
                limit: properties.max_descriptor_set_input_attachments,
                requested: num_input_attachments.total,
            },
        );
    }

    for &PipelineLayoutPcRange { offset, size, .. } in push_constants_ranges {
        if offset + size > properties.max_push_constants_size {
            return Err(PipelineLayoutLimitsError::MaxPushConstantsSizeExceeded {
                limit: properties.max_push_constants_size,
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
        limit: u32,
        /// What was requested.
        requested: u32,
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

#[cfg(test)]
mod tests {

    use crate::{pipeline::layout::PipelineLayoutPcRange, shader::ShaderStages};

    use super::PipelineLayout;

    #[test]
    fn overlapping_push_constant_ranges() {
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
                    PipelineLayoutPcRange {
                        offset: 0,
                        size: 12,
                        stages: ShaderStages {
                            fragment: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 0,
                        size: 40,
                        stages: ShaderStages {
                            vertex: true,
                            ..Default::default()
                        },
                    },
                ][..],
                &[
                    PipelineLayoutPcRange {
                        offset: 0,
                        size: 12,
                        stages: ShaderStages {
                            vertex: true,
                            fragment: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 12,
                        size: 28,
                        stages: ShaderStages {
                            vertex: true,
                            ..Default::default()
                        },
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
                    PipelineLayoutPcRange {
                        offset: 0,
                        size: 12,
                        stages: ShaderStages {
                            fragment: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 4,
                        size: 36,
                        stages: ShaderStages {
                            vertex: true,
                            ..Default::default()
                        },
                    },
                ][..],
                &[
                    PipelineLayoutPcRange {
                        offset: 0,
                        size: 4,
                        stages: ShaderStages {
                            fragment: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 4,
                        size: 8,
                        stages: ShaderStages {
                            fragment: true,
                            vertex: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 12,
                        size: 28,
                        stages: ShaderStages {
                            vertex: true,
                            ..Default::default()
                        },
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
                    PipelineLayoutPcRange {
                        offset: 0,
                        size: 12,
                        stages: ShaderStages {
                            fragment: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 8,
                        size: 12,
                        stages: ShaderStages {
                            compute: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 4,
                        size: 12,
                        stages: ShaderStages {
                            vertex: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 8,
                        size: 24,
                        stages: ShaderStages {
                            tessellation_control: true,
                            ..Default::default()
                        },
                    },
                ][..],
                &[
                    PipelineLayoutPcRange {
                        offset: 0,
                        size: 4,
                        stages: ShaderStages {
                            fragment: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 4,
                        size: 4,
                        stages: ShaderStages {
                            fragment: true,
                            vertex: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 8,
                        size: 4,
                        stages: ShaderStages {
                            vertex: true,
                            fragment: true,
                            compute: true,
                            tessellation_control: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 12,
                        size: 4,
                        stages: ShaderStages {
                            vertex: true,
                            compute: true,
                            tessellation_control: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 16,
                        size: 4,
                        stages: ShaderStages {
                            compute: true,
                            tessellation_control: true,
                            ..Default::default()
                        },
                    },
                    PipelineLayoutPcRange {
                        offset: 20,
                        size: 12,
                        stages: ShaderStages {
                            tessellation_control: true,
                            ..Default::default()
                        },
                    },
                ][..],
            ),
        ];

        let (device, _) = gfx_dev_and_queue!();

        for (input, expected) in test_cases {
            let layout = PipelineLayout::new(device.clone(), [], input.iter().cloned()).unwrap();

            assert_eq!(layout.overlapping_push_constant_ranges.as_slice(), expected);
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
