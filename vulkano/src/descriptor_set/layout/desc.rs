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

use crate::format::Format;
use crate::image::view::ImageViewType;
use crate::pipeline::shader::DescriptorRequirements;
use crate::pipeline::shader::ShaderStages;
use crate::sampler::Sampler;
use crate::sync::AccessFlags;
use crate::sync::PipelineStages;
use smallvec::SmallVec;
use std::cmp;
use std::error;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct DescriptorSetDesc {
    descriptors: SmallVec<[Option<DescriptorDesc>; 32]>,
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
        }
    }

    /// Returns the descriptors in the set.
    pub fn bindings(&self) -> &[Option<DescriptorDesc>] {
        &self.descriptors
    }

    /// Returns the descriptor with the given binding number, or `None` if the binding is empty.
    #[inline]
    pub fn descriptor(&self, num: u32) -> Option<&DescriptorDesc> {
        self.descriptors.get(num as usize).and_then(|b| b.as_ref())
    }

    /// Changes a buffer descriptor's type to dynamic.
    pub fn set_buffer_dynamic(&mut self, binding_num: u32) {
        assert!(
            self.descriptor(binding_num).map_or(false, |desc| matches!(
                desc.ty,
                DescriptorDescTy::StorageBuffer | DescriptorDescTy::UniformBuffer
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
                DescriptorDescTy::StorageBuffer => {
                    desc.ty = DescriptorDescTy::StorageBufferDynamic;
                }
                DescriptorDescTy::UniformBuffer => {
                    desc.ty = DescriptorDescTy::UniformBufferDynamic;
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
            .and_then(|desc| match &mut desc.ty {
                DescriptorDescTy::Sampler {
                    immutable_samplers, ..
                }
                | DescriptorDescTy::CombinedImageSampler {
                    immutable_samplers, ..
                } => Some(immutable_samplers),
                _ => None,
            })
            .expect("binding_num does not refer to a sampler or combined image sampler descriptor");

        immutable_samplers.clear();
        immutable_samplers.extend(samplers.into_iter());
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
    /// meaning that all descriptors are compatible.
    #[inline]
    pub fn is_compatible_with(&self, other: &DescriptorSetDesc) -> bool {
        let num_bindings = cmp::max(self.descriptors.len(), other.descriptors.len()) as u32;
        (0..num_bindings).all(|binding_num| {
            match (self.descriptor(binding_num), other.descriptor(binding_num)) {
                (None, None) => true,
                (Some(first), Some(second)) => first.is_compatible_with(second),
                _ => false,
            }
        })
    }

    /// Checks whether the descriptor of a pipeline layout `self` is compatible with the descriptor
    /// of a descriptor set being bound `other`.
    pub fn ensure_compatible_with_bind(
        &self,
        other: &DescriptorSetDesc,
    ) -> Result<(), DescriptorSetCompatibilityError> {
        if self.descriptors.len() != other.descriptors.len() {
            return Err(DescriptorSetCompatibilityError::DescriptorsCountMismatch {
                self_num: self.descriptors.len() as u32,
                other_num: other.descriptors.len() as u32,
            });
        }

        for binding_num in 0..other.descriptors.len() as u32 {
            let self_desc = self.descriptor(binding_num);
            let other_desc = self.descriptor(binding_num);

            match (self_desc, other_desc) {
                (Some(mine), Some(other)) => {
                    if let Err(err) = mine.ensure_compatible_with_bind(&other) {
                        return Err(DescriptorSetCompatibilityError::IncompatibleDescriptors {
                            error: err,
                            binding_num: binding_num as u32,
                        });
                    }
                }
                (None, None) => (),
                (a, b) => {
                    return Err(DescriptorSetCompatibilityError::IncompatibleDescriptors {
                        error: DescriptorCompatibilityError::Empty {
                            first: a.is_none(),
                            second: b.is_none(),
                        },
                        binding_num: binding_num as u32,
                    })
                }
            }
        }

        Ok(())
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
        }
    }
}

/// Contains the exact description of a single descriptor.
///
/// > **Note**: You are free to fill a `DescriptorDesc` struct the way you want, but its validity
/// > will be checked when you create a pipeline layout, a descriptor set, or when you try to bind
/// > a descriptor set.
// TODO: add example
#[derive(Clone, Debug, PartialEq)]
pub struct DescriptorDesc {
    /// Describes the content and layout of each array element of a descriptor.
    pub ty: DescriptorDescTy,

    /// How many array elements this descriptor is made of. The value 0 is invalid and may trigger
    /// a panic depending on the situation.
    pub descriptor_count: u32,

    /// Which shader stages are going to access this descriptor.
    pub stages: ShaderStages,

    /// True if the descriptor has a variable descriptor count.
    pub variable_count: bool,

    /// True if the attachment can be written by the shader.
    pub mutable: bool,
}

impl DescriptorDesc {
    /// Returns whether `self` is compatible with `other`.
    ///
    /// "Compatible" in this sense is defined by the Vulkan specification under the section
    /// "Pipeline layout compatibility": the two must be identically defined to the Vulkan API,
    /// meaning they have identical `VkDescriptorSetLayoutBinding` values.
    #[inline]
    pub fn is_compatible_with(&self, other: &DescriptorDesc) -> bool {
        self.ty.ty() == other.ty.ty()
            && self.ty.immutable_samplers() == other.ty.immutable_samplers()
            && self.stages == other.stages
            && self.descriptor_count == other.descriptor_count
            && self.variable_count == other.variable_count
    }

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

        if !descriptor_types.contains(&self.ty.ty()) {
            return Err(DescriptorRequirementsNotMet::DescriptorType {
                required: descriptor_types.clone(),
                obtained: self.ty.ty(),
            });
        }

        if self.descriptor_count < *descriptor_count {
            return Err(DescriptorRequirementsNotMet::DescriptorCount {
                required: *descriptor_count,
                obtained: self.descriptor_count,
            });
        }

        if let Some(format) = *format {
            if self.ty.format() != Some(format) {
                return Err(DescriptorRequirementsNotMet::Format {
                    required: format,
                    obtained: self.ty.format(),
                });
            }
        }

        if let Some(image_view_type) = *image_view_type {
            if self.ty.image_view_type() != Some(image_view_type) {
                return Err(DescriptorRequirementsNotMet::ImageViewType {
                    required: image_view_type,
                    obtained: self.ty.image_view_type(),
                });
            }
        }

        if *multisampled != self.ty.multisampled() {
            return Err(DescriptorRequirementsNotMet::Multisampling {
                required: *multisampled,
                obtained: self.ty.multisampled(),
            });
        }

        if *mutable && !self.mutable {
            return Err(DescriptorRequirementsNotMet::Mutability);
        }

        if !self.stages.is_superset_of(stages) {
            return Err(DescriptorRequirementsNotMet::ShaderStages {
                required: *stages,
                obtained: self.stages,
            });
        }

        Ok(())
    }

    /// Checks whether the descriptor of a pipeline layout `self` is compatible with the descriptor
    /// of a descriptor set being bound `other`.
    #[inline]
    pub fn ensure_compatible_with_bind(
        &self,
        other: &DescriptorDesc,
    ) -> Result<(), DescriptorCompatibilityError> {
        other.ty.ensure_superset_of(&self.ty)?;

        if self.stages != other.stages {
            return Err(DescriptorCompatibilityError::ShaderStages {
                first: self.stages,
                second: other.stages,
            });
        }

        if self.descriptor_count != other.descriptor_count {
            return Err(DescriptorCompatibilityError::DescriptorCount {
                first: self.descriptor_count,
                second: other.descriptor_count,
            });
        }

        if self.variable_count != other.variable_count {
            return Err(DescriptorCompatibilityError::VariableCount {
                first: self.variable_count,
                second: other.variable_count,
            });
        }

        if self.mutable && !other.mutable {
            return Err(DescriptorCompatibilityError::Mutability {
                first: self.mutable,
                second: other.mutable,
            });
        }

        Ok(())
    }

    /// Returns the pipeline stages and access flags corresponding to the usage of this descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the type is `Sampler`.
    ///
    pub fn pipeline_stages_and_access(&self) -> (PipelineStages, AccessFlags) {
        let stages: PipelineStages = self.stages.into();

        let access = match self.ty.ty() {
            DescriptorType::Sampler => panic!(),
            DescriptorType::CombinedImageSampler
            | DescriptorType::SampledImage
            | DescriptorType::StorageImage => AccessFlags {
                shader_read: true,
                shader_write: self.mutable,
                ..AccessFlags::none()
            },
            DescriptorType::InputAttachment => AccessFlags {
                input_attachment_read: true,
                ..AccessFlags::none()
            },
            DescriptorType::UniformTexelBuffer | DescriptorType::StorageTexelBuffer => {
                AccessFlags {
                    shader_read: true,
                    shader_write: self.mutable,
                    ..AccessFlags::none()
                }
            }
            DescriptorType::UniformBuffer | DescriptorType::UniformBufferDynamic => AccessFlags {
                uniform_read: true,
                ..AccessFlags::none()
            },
            DescriptorType::StorageBuffer | DescriptorType::StorageBufferDynamic => AccessFlags {
                shader_read: true,
                shader_write: self.mutable,
                ..AccessFlags::none()
            },
        };

        (stages, access)
    }
}

impl From<&DescriptorRequirements> for DescriptorDesc {
    fn from(reqs: &DescriptorRequirements) -> Self {
        let ty = match reqs.descriptor_types[0] {
            DescriptorType::Sampler => DescriptorDescTy::Sampler {
                immutable_samplers: Vec::new(),
            },
            DescriptorType::CombinedImageSampler => DescriptorDescTy::CombinedImageSampler {
                image_desc: DescriptorDescImage {
                    format: reqs.format,
                    multisampled: reqs.multisampled,
                    view_type: reqs.image_view_type.unwrap(),
                },
                immutable_samplers: Vec::new(),
            },
            DescriptorType::SampledImage => DescriptorDescTy::SampledImage {
                image_desc: DescriptorDescImage {
                    format: reqs.format,
                    multisampled: reqs.multisampled,
                    view_type: reqs.image_view_type.unwrap(),
                },
            },
            DescriptorType::StorageImage => DescriptorDescTy::StorageImage {
                image_desc: DescriptorDescImage {
                    format: reqs.format,
                    multisampled: reqs.multisampled,
                    view_type: reqs.image_view_type.unwrap(),
                },
            },
            DescriptorType::UniformTexelBuffer => DescriptorDescTy::UniformTexelBuffer {
                format: reqs.format,
            },
            DescriptorType::StorageTexelBuffer => DescriptorDescTy::StorageTexelBuffer {
                format: reqs.format,
            },
            DescriptorType::UniformBuffer => DescriptorDescTy::UniformBuffer,
            DescriptorType::StorageBuffer => DescriptorDescTy::StorageBuffer,
            DescriptorType::UniformBufferDynamic => DescriptorDescTy::UniformBufferDynamic,
            DescriptorType::StorageBufferDynamic => DescriptorDescTy::StorageBufferDynamic,
            DescriptorType::InputAttachment => DescriptorDescTy::InputAttachment {
                multisampled: reqs.multisampled,
            },
        };

        Self {
            ty,
            descriptor_count: reqs.descriptor_count,
            stages: reqs.stages,
            variable_count: false,
            mutable: reqs.mutable,
        }
    }
}

/// Describes what kind of resource may later be bound to a descriptor.
///
/// This is mostly the same as a `DescriptorDescTy` but with less precise information.
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

    /// The descriptor's format does not match what is required.
    Format {
        required: Format,
        obtained: Option<Format>,
    },

    /// The descriptor's image view type does not match what is required.
    ImageViewType {
        required: ImageViewType,
        obtained: Option<ImageViewType>,
    },

    /// The descriptor's multisampling does not match what is required.
    Multisampling { required: bool, obtained: bool },

    /// The descriptor is marked as read-only, but mutability is required.
    Mutability,

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
            Self::Format { required, obtained } => write!(
                fmt,
                "the descriptor's format ({:?}) does not match what is required ({:?})",
                obtained, required
            ),
            Self::ImageViewType { required, obtained } => write!(
                fmt,
                "the descriptor's image view type ({:?}) does not match what is required ({:?})",
                obtained, required
            ),
            Self::Multisampling { required, obtained } => write!(
                fmt,
                "the descriptor's multisampling ({}) does not match what is required ({})",
                obtained, required
            ),
            Self::Mutability => write!(
                fmt,
                "the descriptor is marked as read-only, but mutability is required",
            ),
            Self::ShaderStages { required, obtained } => write!(
                fmt,
                "the descriptor's shader stages do not contain the stages that are required",
            ),
        }
    }
}

/// Describes the content and layout of each array element of a descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorDescTy {
    Sampler {
        /// Samplers that are included as a fixed part of the descriptor set layout. Once bound, they
        /// do not need to be provided when creating a descriptor set.
        ///
        /// The list must be either empty, or contain exactly `descriptor_count` samplers.
        immutable_samplers: Vec<Arc<Sampler>>,
    },
    CombinedImageSampler {
        image_desc: DescriptorDescImage,

        /// Samplers that are included as a fixed part of the descriptor set layout. Once bound, they
        /// do not need to be provided when creating a descriptor set.
        ///
        /// The list must be either empty, or contain exactly `descriptor_count` samplers.
        immutable_samplers: Vec<Arc<Sampler>>,
    },
    SampledImage {
        image_desc: DescriptorDescImage,
    },
    StorageImage {
        image_desc: DescriptorDescImage,
    },
    UniformTexelBuffer {
        /// The format of the content, or `None` if the format is unknown. Depending on the
        /// context, it may be invalid to have a `None` value here. If the format is `Some`, only
        /// buffer views that have this exact format can be attached to this descriptor.
        format: Option<Format>,
    },
    StorageTexelBuffer {
        /// The format of the content, or `None` if the format is unknown. Depending on the
        /// context, it may be invalid to have a `None` value here. If the format is `Some`, only
        /// buffer views that have this exact format can be attached to this descriptor.
        format: Option<Format>,
    },
    UniformBuffer,
    StorageBuffer,
    UniformBufferDynamic,
    StorageBufferDynamic,
    InputAttachment {
        /// If `true`, the input attachment is multisampled. Only multisampled images can be
        /// attached to this descriptor. If `false`, only single-sampled images can be attached.
        multisampled: bool,
    },
}

impl DescriptorDescTy {
    /// Returns the type of descriptor.
    // TODO: add example
    #[inline]
    pub fn ty(&self) -> DescriptorType {
        match *self {
            Self::Sampler { .. } => DescriptorType::Sampler,
            Self::CombinedImageSampler { .. } => DescriptorType::CombinedImageSampler,
            Self::SampledImage { .. } => DescriptorType::SampledImage,
            Self::StorageImage { .. } => DescriptorType::StorageImage,
            Self::UniformTexelBuffer { .. } => DescriptorType::UniformTexelBuffer,
            Self::StorageTexelBuffer { .. } => DescriptorType::StorageTexelBuffer,
            Self::UniformBuffer => DescriptorType::UniformBuffer,
            Self::StorageBuffer => DescriptorType::StorageBuffer,
            Self::UniformBufferDynamic => DescriptorType::UniformBufferDynamic,
            Self::StorageBufferDynamic => DescriptorType::StorageBufferDynamic,
            Self::InputAttachment { .. } => DescriptorType::InputAttachment,
        }
    }

    #[inline]
    fn format(&self) -> Option<Format> {
        match self {
            Self::CombinedImageSampler { image_desc, .. }
            | Self::SampledImage { image_desc, .. }
            | Self::StorageImage { image_desc, .. } => image_desc.format,
            Self::UniformTexelBuffer { format } | Self::StorageTexelBuffer { format } => *format,
            _ => None,
        }
    }

    #[inline]
    fn image_desc(&self) -> Option<&DescriptorDescImage> {
        match self {
            Self::CombinedImageSampler { image_desc, .. }
            | Self::SampledImage { image_desc, .. }
            | Self::StorageImage { image_desc, .. } => Some(image_desc),
            _ => None,
        }
    }

    #[inline]
    fn image_view_type(&self) -> Option<ImageViewType> {
        match self {
            Self::CombinedImageSampler { image_desc, .. }
            | Self::SampledImage { image_desc, .. }
            | Self::StorageImage { image_desc, .. } => Some(image_desc.view_type),
            _ => None,
        }
    }

    #[inline]
    pub(super) fn immutable_samplers(&self) -> &[Arc<Sampler>] {
        match self {
            Self::Sampler {
                immutable_samplers, ..
            } => immutable_samplers,
            Self::CombinedImageSampler {
                immutable_samplers, ..
            } => immutable_samplers,
            _ => &[],
        }
    }

    #[inline]
    fn multisampled(&self) -> bool {
        match self {
            Self::CombinedImageSampler { image_desc, .. }
            | Self::SampledImage { image_desc, .. }
            | Self::StorageImage { image_desc, .. } => image_desc.multisampled,
            DescriptorDescTy::InputAttachment { multisampled } => *multisampled,
            _ => false,
        }
    }

    /// Checks whether we are a superset of another descriptor type.
    // TODO: add example
    #[inline]
    pub fn ensure_superset_of(&self, other: &Self) -> Result<(), DescriptorCompatibilityError> {
        if self.ty() != other.ty() {
            return Err(DescriptorCompatibilityError::Type {
                first: self.ty(),
                second: other.ty(),
            });
        }

        if self.immutable_samplers() != other.immutable_samplers() {
            return Err(DescriptorCompatibilityError::ImmutableSamplers);
        }

        if let (Some(me), Some(other)) = (self.image_desc(), other.image_desc()) {
            me.ensure_superset_of(other)?;
        }

        if let (me, other @ Some(_)) = (self.format(), other.format()) {
            if me != other {
                return Err(DescriptorCompatibilityError::Format {
                    first: me,
                    second: other.unwrap(),
                });
            }
        }

        if self.multisampled() != other.multisampled() {
            return Err(DescriptorCompatibilityError::Multisampling {
                first: self.multisampled(),
                second: other.multisampled(),
            });
        }

        Ok(())
    }
}

/// Additional description for descriptors that contain images.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DescriptorDescImage {
    /// The image format that is required for an attached image, or `None` no particular format is required.
    pub format: Option<Format>,
    /// True if the image is multisampled.
    pub multisampled: bool,
    /// The type of image view that must be attached to this descriptor.
    pub view_type: ImageViewType,
}

impl DescriptorDescImage {
    /// Checks whether we are a superset of another image.
    // TODO: add example
    #[inline]
    pub fn ensure_superset_of(
        &self,
        other: &DescriptorDescImage,
    ) -> Result<(), DescriptorCompatibilityError> {
        if other.format.is_some() && self.format != other.format {
            return Err(DescriptorCompatibilityError::Format {
                first: self.format,
                second: other.format.unwrap(),
            });
        }

        if self.multisampled != other.multisampled {
            return Err(DescriptorCompatibilityError::Multisampling {
                first: self.multisampled,
                second: other.multisampled,
            });
        }

        if self.view_type != other.view_type {
            return Err(DescriptorCompatibilityError::ImageViewType {
                first: self.view_type,
                second: other.view_type,
            });
        }

        Ok(())
    }
}

/// Error when checking whether a descriptor set is compatible with another one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorSetCompatibilityError {
    /// The number of descriptors in the two sets is not compatible.
    DescriptorsCountMismatch { self_num: u32, other_num: u32 },

    /// Two descriptors are incompatible.
    IncompatibleDescriptors {
        error: DescriptorCompatibilityError,
        binding_num: u32,
    },
}

impl error::Error for DescriptorSetCompatibilityError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            DescriptorSetCompatibilityError::IncompatibleDescriptors { ref error, .. } => {
                Some(error)
            }
            _ => None,
        }
    }
}

impl fmt::Display for DescriptorSetCompatibilityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                DescriptorSetCompatibilityError::DescriptorsCountMismatch { .. } => {
                    "the number of descriptors in the two sets is not compatible."
                }
                DescriptorSetCompatibilityError::IncompatibleDescriptors { .. } => {
                    "two descriptors are incompatible"
                }
            }
        )
    }
}

/// Error when checking whether a descriptor compatible with another one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorCompatibilityError {
    /// The number of descriptors is not compatible.
    DescriptorCount { first: u32, second: u32 },

    /// The variable counts of the descriptors is not compatible.
    VariableCount { first: bool, second: bool },

    /// The presence or absence of a descriptor in a binding is not compatible.
    Empty { first: bool, second: bool },

    /// The formats of an image descriptor are not compatible.
    Format {
        first: Option<Format>,
        second: Format,
    },

    /// The image view types of an image descriptor are not compatible.
    ImageViewType {
        first: ImageViewType,
        second: ImageViewType,
    },

    /// The immutable samplers of the descriptors are not compatible.
    ImmutableSamplers,

    /// The multisampling of an image descriptor is not compatible.
    Multisampling { first: bool, second: bool },

    /// The mutability of the descriptors is not compatible.
    Mutability { first: bool, second: bool },

    /// The shader stages of the descriptors are not compatible.
    ShaderStages {
        first: ShaderStages,
        second: ShaderStages,
    },

    /// The types of the two descriptors are not compatible.
    Type {
        first: DescriptorType,
        second: DescriptorType,
    },
}

impl error::Error for DescriptorCompatibilityError {}

impl fmt::Display for DescriptorCompatibilityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                DescriptorCompatibilityError::DescriptorCount { .. } => {
                    "the number of descriptors is not compatible"
                }
                DescriptorCompatibilityError::VariableCount { .. } => {
                    "the variable counts of the descriptors is not compatible"
                }
                DescriptorCompatibilityError::Empty { .. } => {
                    "the presence or absence of a descriptor in a binding is not compatible"
                }
                DescriptorCompatibilityError::Format { .. } => {
                    "the formats of an image descriptor are not compatible"
                }
                DescriptorCompatibilityError::ImageViewType { .. } => {
                    "the image view types of an image descriptor are not compatible"
                }
                DescriptorCompatibilityError::ImmutableSamplers { .. } => {
                    "the immutable samplers of the descriptors are not compatible"
                }
                DescriptorCompatibilityError::Multisampling { .. } => {
                    "the multisampling of an image descriptor is not compatible"
                }
                DescriptorCompatibilityError::Mutability { .. } => {
                    "the mutability of the descriptors is not compatible"
                }
                DescriptorCompatibilityError::ShaderStages { .. } => {
                    "the shader stages of the descriptors are not compatible"
                }
                DescriptorCompatibilityError::Type { .. } => {
                    "the types of the two descriptors are not compatible"
                }
            }
        )
    }
}
