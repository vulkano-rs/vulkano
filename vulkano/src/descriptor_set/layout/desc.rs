// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Description of a single descriptor.
//!
//! This module contains traits and structs related to describing a single descriptor. A descriptor
//! is a slot where you can bind a buffer or an image so that it can be accessed from your shaders.
//! In order to specify which buffer or image to bind to a descriptor, see the `descriptor_set`
//! module.
//!
//! There are four different kinds of descriptors that give access to buffers:
//!
//! - Uniform texel buffers. Gives read-only access to the content of a buffer. Only supports
//!   certain buffer formats.
//! - Storage texel buffers. Gives read and/or write access to the content of a buffer. Only
//!   supports certain buffer formats. Less restrictive but sometimes slower than uniform texel
//!   buffers.
//! - Uniform buffers. Gives read-only access to the content of a buffer. Less restrictive but
//!   sometimes slower than uniform texel buffers.
//! - Storage buffers. Gives read and/or write access to the content of a buffer. Less restrictive
//!   but sometimes slower than uniform buffers and storage texel buffers.
//!
//! There are five different kinds of descriptors related to images:
//!
//! - Storage images. Gives read and/or write access to individual pixels in an image. The image
//!   cannot be sampled. In other words, you have exactly specify which pixel to read or write.
//! - Sampled images. Gives read-only access to an image. Before you can use a sampled image in a
//!   a shader, you have to combine it with a sampler (see below). The sampler describes how
//!   reading the image will behave.
//! - Samplers. Doesn't contain an image but a sampler object that describes how an image will be
//!   accessed. This is meant to be combined with a sampled image (see above).
//! - Combined image and sampler. Similar to a sampled image, but also directly includes the
//!   sampler which indicates how the sampling is done.
//! - Input attachments. The fastest but also most restrictive access to images. Must be integrated
//!   in a render pass. Can only give access to the same pixel as the one you're processing.
//!

use crate::pipeline::shader::DescriptorRequirements;
use crate::pipeline::shader::ShaderStages;
use crate::sampler::Sampler;
use smallvec::SmallVec;
use std::cmp;
use std::error;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct DescriptorSetDesc {
    descriptors: SmallVec<[Option<DescriptorDesc>; 4]>,
    push_descriptor: bool,
}

impl DescriptorSetDesc {
    /// Builds a new `DescriptorSetDesc` with the given descriptors.
    ///
    /// The descriptors must be passed in the order of the bindings. In order words, descriptor
    /// at bind point 0 first, then descriptor at bind point 1, and so on. If a binding must remain
    /// empty, you can make the iterator yield `None` for an element.
    #[inline]
    pub fn new<I>(descriptors: I) -> Self
    where
        I: IntoIterator<Item = Option<DescriptorDesc>>,
    {
        Self {
            descriptors: descriptors.into_iter().collect(),
            push_descriptor: false,
        }
    }

    /// Builds a list of `DescriptorSetDesc` from an iterator of `DescriptorRequirement` originating
    /// from a shader.
    #[inline]
    pub fn from_requirements<'a>(
        descriptor_requirements: impl IntoIterator<Item = ((u32, u32), &'a DescriptorRequirements)>,
    ) -> Vec<Self> {
        let mut descriptor_sets: Vec<Self> = Vec::new();

        for ((set_num, binding_num), reqs) in descriptor_requirements {
            let set_num = set_num as usize;
            let binding_num = binding_num as usize;

            if set_num >= descriptor_sets.len() {
                descriptor_sets.resize(set_num + 1, Self::default());
            }

            let descriptors = &mut descriptor_sets[set_num].descriptors;

            if binding_num >= descriptors.len() {
                descriptors.resize(binding_num + 1, None);
            }

            descriptors[binding_num] = Some(reqs.into());
        }

        descriptor_sets
    }

    /// Builds a new empty `DescriptorSetDesc`.
    #[inline]
    pub fn empty() -> DescriptorSetDesc {
        DescriptorSetDesc {
            descriptors: SmallVec::new(),
            push_descriptor: false,
        }
    }

    /// Returns the descriptors in the set.
    #[inline]
    pub fn bindings(&self) -> &[Option<DescriptorDesc>] {
        &self.descriptors
    }

    /// Returns the descriptor with the given binding number, or `None` if the binding is empty.
    #[inline]
    pub fn descriptor(&self, num: u32) -> Option<&DescriptorDesc> {
        self.descriptors.get(num as usize).and_then(|b| b.as_ref())
    }

    /// Returns whether the description is set to be a push descriptor.
    #[inline]
    pub fn is_push_descriptor(&self) -> bool {
        self.push_descriptor
    }

    /// Changes a buffer descriptor's type to dynamic.
    ///
    /// # Panics
    ///
    /// - Panics if the description is set to be a push descriptor.
    /// - Panics if `binding_num` does not refer to a `StorageBuffer` or `UniformBuffer` descriptor.
    pub fn set_buffer_dynamic(&mut self, binding_num: u32) {
        assert!(
            !self.push_descriptor,
            "push descriptor is enabled, which does not allow dynamic buffer descriptors"
        );
        assert!(
            self.descriptor(binding_num).map_or(false, |desc| matches!(
                desc.ty,
                DescriptorType::StorageBuffer | DescriptorType::UniformBuffer
            )),
            "tried to make the non-buffer descriptor at binding {} a dynamic buffer",
            binding_num
        );

        let binding = self
            .descriptors
            .get_mut(binding_num as usize)
            .and_then(|b| b.as_mut());

        if let Some(desc) = binding {
            match &desc.ty {
                DescriptorType::StorageBuffer => {
                    desc.ty = DescriptorType::StorageBufferDynamic;
                }
                DescriptorType::UniformBuffer => {
                    desc.ty = DescriptorType::UniformBufferDynamic;
                }
                _ => (),
            };
        }
    }

    /// Sets the immutable samplers for a sampler or combined image sampler descriptor.
    ///
    /// # Panics
    ///
    /// - Panics if the binding number does not refer to a sampler or combined image sampler
    ///   descriptor.
    pub fn set_immutable_samplers(
        &mut self,
        binding_num: u32,
        samplers: impl IntoIterator<Item = Arc<Sampler>>,
    ) {
        let immutable_samplers = self
            .descriptors
            .get_mut(binding_num as usize)
            .and_then(|b| b.as_mut())
            .and_then(|desc| match desc.ty {
                DescriptorType::Sampler | DescriptorType::CombinedImageSampler => {
                    Some(&mut desc.immutable_samplers)
                }
                _ => None,
            })
            .expect("binding_num does not refer to a sampler or combined image sampler descriptor");

        immutable_samplers.clear();
        immutable_samplers.extend(samplers.into_iter());
    }

    /// Sets the descriptor set layout to use push descriptors instead of descriptor sets.
    ///
    /// If set to enabled, the
    /// [`khr_push_descriptor`](crate::device::DeviceExtensions::khr_push_descriptor) extension must
    /// be enabled on the device.
    ///
    /// # Panics
    ///
    /// - If enabled, panics if the description contains a dynamic buffer descriptor.
    pub fn set_push_descriptor(&mut self, enabled: bool) {
        if enabled {
            assert!(
                !self.descriptors.iter().flatten().any(|desc| {
                    matches!(
                        desc.ty,
                        DescriptorType::UniformBufferDynamic | DescriptorType::StorageBufferDynamic
                    )
                }),
                "descriptor set contains a dynamic buffer descriptor"
            );
        }
        self.push_descriptor = enabled;
    }

    /// Sets the descriptor count for a descriptor that has a variable count.
    pub fn set_variable_descriptor_count(&mut self, binding_num: u32, descriptor_count: u32) {
        // TODO: Errors instead of panic

        match self
            .descriptors
            .get_mut(binding_num as usize)
            .and_then(|b| b.as_mut())
        {
            Some(desc) => {
                desc.variable_count = true;
                desc.descriptor_count = descriptor_count;
            }
            None => panic!("descriptor is empty"),
        }
    }

    /// Returns whether `self` is compatible with `other`.
    ///
    /// "Compatible" in this sense is defined by the Vulkan specification under the section
    /// "Pipeline layout compatibility": the two must be identically defined to the Vulkan API,
    /// meaning that all descriptors are compatible and flags are identical.
    #[inline]
    pub fn is_compatible_with(&self, other: &DescriptorSetDesc) -> bool {
        if self.push_descriptor != other.push_descriptor {
            return false;
        }

        let num_bindings = cmp::max(self.descriptors.len(), other.descriptors.len()) as u32;
        (0..num_bindings).all(|binding_num| {
            match (self.descriptor(binding_num), other.descriptor(binding_num)) {
                (None, None) => true,
                (Some(first), Some(second)) => first == second,
                _ => false,
            }
        })
    }
}

impl<I> From<I> for DescriptorSetDesc
where
    I: IntoIterator<Item = Option<DescriptorDesc>>,
{
    #[inline]
    fn from(val: I) -> Self {
        DescriptorSetDesc {
            descriptors: val.into_iter().collect(),
            push_descriptor: false,
        }
    }
}

/// Contains the exact description of a single descriptor.
///
/// > **Note**: You are free to fill a `DescriptorDesc` struct the way you want, but its validity
/// > will be checked when you create a pipeline layout, a descriptor set, or when you try to bind
/// > a descriptor set.
// TODO: add example
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescriptorDesc {
    /// Describes the content and layout of each array element of a descriptor.
    pub ty: DescriptorType,

    /// How many array elements this descriptor is made of. The value 0 is invalid and may trigger
    /// a panic depending on the situation.
    pub descriptor_count: u32,

    /// True if the descriptor has a variable descriptor count. The value of `descriptor_count`
    /// is taken as the maximum number of descriptors allowed. There may only be one binding with a
    /// variable count in a descriptor set, and it must be the last binding.
    pub variable_count: bool,

    /// Which shader stages are going to access this descriptor.
    pub stages: ShaderStages,

    /// Samplers that are included as a fixed part of the descriptor set layout. Once bound, they
    /// do not need to be provided when creating a descriptor set.
    ///
    /// The list must be either empty, or contain exactly `descriptor_count` samplers. It must be
    /// empty if `ty` is something other than `Sampler` or `CombinedImageSampler`.
    pub immutable_samplers: Vec<Arc<Sampler>>,
}

impl DescriptorDesc {
    /// Checks whether the descriptor of a pipeline layout `self` is compatible with the descriptor
    /// of a shader `other`.
    #[inline]
    pub fn ensure_compatible_with_shader(
        &self,
        descriptor_requirements: &DescriptorRequirements,
    ) -> Result<(), DescriptorRequirementsNotMet> {
        let DescriptorRequirements {
            descriptor_types,
            descriptor_count,
            format,
            image_view_type,
            multisampled,
            mutable,
            stages,
        } = descriptor_requirements;

        if !descriptor_types.contains(&self.ty) {
            return Err(DescriptorRequirementsNotMet::DescriptorType {
                required: descriptor_types.clone(),
                obtained: self.ty,
            });
        }

        if self.descriptor_count < *descriptor_count {
            return Err(DescriptorRequirementsNotMet::DescriptorCount {
                required: *descriptor_count,
                obtained: self.descriptor_count,
            });
        }

        if !self.stages.is_superset_of(stages) {
            return Err(DescriptorRequirementsNotMet::ShaderStages {
                required: *stages,
                obtained: self.stages,
            });
        }

        Ok(())
    }
}

impl From<&DescriptorRequirements> for DescriptorDesc {
    fn from(reqs: &DescriptorRequirements) -> Self {
        let ty = match reqs.descriptor_types[0] {
            DescriptorType::Sampler => DescriptorType::Sampler,
            DescriptorType::CombinedImageSampler => DescriptorType::CombinedImageSampler,
            DescriptorType::SampledImage => DescriptorType::SampledImage,
            DescriptorType::StorageImage => DescriptorType::StorageImage,
            DescriptorType::UniformTexelBuffer => DescriptorType::UniformTexelBuffer,
            DescriptorType::StorageTexelBuffer => DescriptorType::StorageTexelBuffer,
            DescriptorType::UniformBuffer => DescriptorType::UniformBuffer,
            DescriptorType::StorageBuffer => DescriptorType::StorageBuffer,
            DescriptorType::UniformBufferDynamic => DescriptorType::UniformBufferDynamic,
            DescriptorType::StorageBufferDynamic => DescriptorType::StorageBufferDynamic,
            DescriptorType::InputAttachment => DescriptorType::InputAttachment,
        };

        Self {
            ty,
            descriptor_count: reqs.descriptor_count,
            variable_count: false,
            stages: reqs.stages,
            immutable_samplers: Vec::new(),
        }
    }
}

/// Describes what kind of resource may later be bound to a descriptor.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum DescriptorType {
    Sampler = ash::vk::DescriptorType::SAMPLER.as_raw(),
    CombinedImageSampler = ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER.as_raw(),
    SampledImage = ash::vk::DescriptorType::SAMPLED_IMAGE.as_raw(),
    StorageImage = ash::vk::DescriptorType::STORAGE_IMAGE.as_raw(),
    UniformTexelBuffer = ash::vk::DescriptorType::UNIFORM_TEXEL_BUFFER.as_raw(),
    StorageTexelBuffer = ash::vk::DescriptorType::STORAGE_TEXEL_BUFFER.as_raw(),
    UniformBuffer = ash::vk::DescriptorType::UNIFORM_BUFFER.as_raw(),
    StorageBuffer = ash::vk::DescriptorType::STORAGE_BUFFER.as_raw(),
    UniformBufferDynamic = ash::vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC.as_raw(),
    StorageBufferDynamic = ash::vk::DescriptorType::STORAGE_BUFFER_DYNAMIC.as_raw(),
    InputAttachment = ash::vk::DescriptorType::INPUT_ATTACHMENT.as_raw(),
}

impl From<DescriptorType> for ash::vk::DescriptorType {
    #[inline]
    fn from(val: DescriptorType) -> Self {
        Self::from_raw(val as i32)
    }
}

/// Error when checking whether the requirements for a descriptor have been met.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorRequirementsNotMet {
    /// The descriptor's type is not one of those required.
    DescriptorType {
        required: Vec<DescriptorType>,
        obtained: DescriptorType,
    },

    /// The descriptor count is less than what is required.
    DescriptorCount { required: u32, obtained: u32 },

    /// The descriptor's shader stages do not contain the stages that are required.
    ShaderStages {
        required: ShaderStages,
        obtained: ShaderStages,
    },
}

impl error::Error for DescriptorRequirementsNotMet {}

impl fmt::Display for DescriptorRequirementsNotMet {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::DescriptorType { required, obtained } => write!(
                fmt,
                "the descriptor's type ({:?}) is not one of those required ({:?})",
                obtained, required
            ),
            Self::DescriptorCount { required, obtained } => write!(
                fmt,
                "the descriptor count ({}) is less than what is required ({})",
                obtained, required
            ),
            Self::ShaderStages { required, obtained } => write!(
                fmt,
                "the descriptor's shader stages do not contain the stages that are required",
            ),
        }
    }
}
