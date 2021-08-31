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
use crate::pipeline::shader::ShaderStages;
use crate::sync::AccessFlags;
use crate::sync::PipelineStages;
use smallvec::SmallVec;
use std::cmp;
use std::error;
use std::fmt;

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
    pub fn new<I>(descriptors: I) -> DescriptorSetDesc
    where
        I: IntoIterator<Item = Option<DescriptorDesc>>,
    {
        DescriptorSetDesc {
            descriptors: descriptors.into_iter().collect(),
        }
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
    pub fn descriptor(&self, num: usize) -> Option<&DescriptorDesc> {
        self.descriptors.get(num).and_then(|b| b.as_ref())
    }

    /// Builds the union of this layout description and another.
    #[inline]
    pub fn union(
        first: &DescriptorSetDesc,
        second: &DescriptorSetDesc,
    ) -> Result<DescriptorSetDesc, ()> {
        let num_bindings = cmp::max(first.descriptors.len(), second.descriptors.len());
        let descriptors = (0..num_bindings)
            .map(|binding_num| {
                DescriptorDesc::union(
                    first
                        .descriptors
                        .get(binding_num)
                        .map(|desc| desc.as_ref())
                        .flatten(),
                    second
                        .descriptors
                        .get(binding_num)
                        .map(|desc| desc.as_ref())
                        .flatten(),
                )
            })
            .collect::<Result<_, ()>>()?;
        Ok(DescriptorSetDesc { descriptors })
    }

    /// Builds the union of multiple descriptor sets.
    pub fn union_multiple(
        first: &[DescriptorSetDesc],
        second: &[DescriptorSetDesc],
    ) -> Result<Vec<DescriptorSetDesc>, ()> {
        // Ewwwwwww
        let empty = DescriptorSetDesc::empty();
        let num_sets = cmp::max(first.len(), second.len());

        (0..num_sets)
            .map(|set_num| {
                Ok(DescriptorSetDesc::union(
                    first.get(set_num).unwrap_or_else(|| &empty),
                    second.get(set_num).unwrap_or_else(|| &empty),
                )?)
            })
            .collect()
    }

    /// Transforms a `DescriptorSetDesc`.
    ///
    /// Used to adjust automatically inferred `DescriptorSetDesc`s with information that cannot be inferred.
    pub fn tweak<I>(&mut self, dynamic_buffers: I)
    where
        I: IntoIterator<Item = usize>,
    {
        for binding_num in dynamic_buffers {
            debug_assert!(
                self.descriptor(binding_num).map_or(false, |desc| matches!(
                    desc.ty,
                    DescriptorDescTy::StorageBuffer | DescriptorDescTy::UniformBuffer
                )),
                "tried to make the non-buffer descriptor at binding {} a dynamic buffer",
                binding_num
            );

            let binding = self
                .descriptors
                .get_mut(binding_num)
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
    }

    pub fn tweak_multiple<I>(sets: &mut [DescriptorSetDesc], dynamic_buffers: I)
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        for (set_num, binding_num) in dynamic_buffers {
            debug_assert!(
                set_num < sets.len(),
                "tried to make a dynamic buffer in the nonexistent set {}",
                set_num,
            );

            sets.get_mut(set_num).map(|set| set.tweak([binding_num]));
        }
    }

    pub fn set_variable_descriptor_count(&mut self, binding_num: u32, descriptor_count: u32) {
        // TODO: Errors instead of panic

        match self.descriptors.get_mut(binding_num as usize) {
            Some(desc_op) => match desc_op.as_mut() {
                Some(desc) => {
                    if desc.variable_count {
                        desc.descriptor_count = descriptor_count;
                    } else {
                        panic!("descriptor isn't variable count")
                    }
                }
                None => panic!("descriptor is empty"),
            },
            None => panic!("descriptor doesn't exist"),
        }
    }

    /// Returns whether `self` is compatible with `other`.
    ///
    /// "Compatible" in this sense is defined by the Vulkan specification under the section
    /// "Pipeline layout compatibility": the two must be identically defined to the Vulkan API,
    /// meaning that all descriptors are compatible.
    #[inline]
    pub fn is_compatible_with(&self, other: &DescriptorSetDesc) -> bool {
        let num_bindings = cmp::max(self.descriptors.len(), other.descriptors.len());
        (0..num_bindings).all(|binding_num| {
            match (self.descriptor(binding_num), other.descriptor(binding_num)) {
                (None, None) => true,
                (Some(first), Some(second)) => first.is_compatible_with(second),
                _ => false,
            }
        })
    }

    /// Checks whether the descriptor of a pipeline layout `self` is compatible with the descriptor
    /// of a shader `other`.
    pub fn ensure_compatible_with_shader(
        &self,
        other: &DescriptorSetDesc,
    ) -> Result<(), DescriptorSetCompatibilityError> {
        if self.descriptors.len() < other.descriptors.len() {
            return Err(DescriptorSetCompatibilityError::DescriptorsCountMismatch {
                self_num: self.descriptors.len() as u32,
                other_num: other.descriptors.len() as u32,
            });
        }

        for binding_num in 0..other.descriptors.len() {
            let self_desc = self.descriptor(binding_num);
            let other_desc = self.descriptor(binding_num);

            match (self_desc, other_desc) {
                (Some(mine), Some(other)) => {
                    if let Err(err) = mine.ensure_compatible_with_shader(&other) {
                        return Err(DescriptorSetCompatibilityError::IncompatibleDescriptors {
                            error: err,
                            binding_num: binding_num as u32,
                        });
                    }
                }
                (None, Some(_)) => {
                    return Err(DescriptorSetCompatibilityError::IncompatibleDescriptors {
                        error: DescriptorCompatibilityError::Empty {
                            first: true,
                            second: false,
                        },
                        binding_num: binding_num as u32,
                    })
                }
                _ => (),
            }
        }

        Ok(())
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

        for binding_num in 0..other.descriptors.len() {
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
            && self.stages == other.stages
            && self.descriptor_count == other.descriptor_count
    }

    /// Checks whether the descriptor of a pipeline layout `self` is compatible with the descriptor
    /// of a shader `other`.
    #[inline]
    pub fn ensure_compatible_with_shader(
        &self,
        other: &DescriptorDesc,
    ) -> Result<(), DescriptorCompatibilityError> {
        match (self.ty.ty(), other.ty.ty()) {
            (DescriptorType::UniformBufferDynamic, DescriptorType::UniformBuffer) => (),
            (DescriptorType::StorageBufferDynamic, DescriptorType::StorageBuffer) => (),
            _ => self.ty.ensure_superset_of(&other.ty)?,
        }

        if !self.stages.is_superset_of(&other.stages) {
            return Err(DescriptorCompatibilityError::ShaderStages {
                first: self.stages,
                second: other.stages,
            });
        }

        if self.descriptor_count < other.descriptor_count {
            return Err(DescriptorCompatibilityError::DescriptorCount {
                first: self.descriptor_count,
                second: other.descriptor_count,
            });
        }

        if !self.mutable && other.mutable {
            return Err(DescriptorCompatibilityError::Mutability {
                first: self.mutable,
                second: other.mutable,
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

        if self.mutable && !other.mutable {
            return Err(DescriptorCompatibilityError::Mutability {
                first: self.mutable,
                second: other.mutable,
            });
        }

        Ok(())
    }

    /// Builds a `DescriptorDesc` that is the union of `self` and `other`, if possible.
    ///
    /// The returned value will be a superset of both `self` and `other`, or `None` if both were
    /// `None`.
    ///
    /// `Err` is returned if the descriptors are not compatible.
    ///
    ///# Example
    ///```
    ///use vulkano::descriptor_set::layout::DescriptorDesc;
    ///use vulkano::descriptor_set::layout::DescriptorDescTy::*;
    ///use vulkano::pipeline::shader::ShaderStages;
    ///
    ///let desc_part1 = DescriptorDesc{ ty: Sampler, descriptor_count: 2, stages: ShaderStages{
    ///  vertex: true,
    ///  tessellation_control: true,
    ///  tessellation_evaluation: false,
    ///  geometry: true,
    ///  fragment: false,
    ///  compute: true
    ///}, mutable: true, variable_count: false };
    ///
    ///let desc_part2 = DescriptorDesc{ ty: Sampler, descriptor_count: 1, stages: ShaderStages{
    ///  vertex: true,
    ///  tessellation_control: false,
    ///  tessellation_evaluation: true,
    ///  geometry: false,
    ///  fragment: true,
    ///  compute: true
    ///}, mutable: false, variable_count: false };
    ///
    ///let desc_union = DescriptorDesc{ ty: Sampler, descriptor_count: 2, stages: ShaderStages{
    ///  vertex: true,
    ///  tessellation_control: true,
    ///  tessellation_evaluation: true,
    ///  geometry: true,
    ///  fragment: true,
    ///  compute: true
    ///}, mutable: true, variable_count: false };
    ///
    ///assert_eq!(DescriptorDesc::union(Some(&desc_part1), Some(&desc_part2)), Ok(Some(desc_union)));
    ///```
    #[inline]
    pub fn union(
        first: Option<&DescriptorDesc>,
        second: Option<&DescriptorDesc>,
    ) -> Result<Option<DescriptorDesc>, ()> {
        if let (Some(first), Some(second)) = (first, second) {
            if first.ty != second.ty {
                return Err(());
            }

            Ok(Some(DescriptorDesc {
                ty: first.ty.clone(),
                descriptor_count: cmp::max(first.descriptor_count, second.descriptor_count),
                stages: first.stages | second.stages,
                mutable: first.mutable || second.mutable,
                variable_count: first.variable_count && second.variable_count, // TODO: What is the correct behavior here?
            }))
        } else {
            Ok(first.or(second).cloned())
        }
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

/// Describes the content and layout of each array element of a descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorDescTy {
    Sampler,                                   // TODO: the sampler has some restrictions as well
    CombinedImageSampler(DescriptorDescImage), // TODO: the sampler has some restrictions as well
    SampledImage(DescriptorDescImage),
    StorageImage(DescriptorDescImage),
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
            DescriptorDescTy::Sampler => DescriptorType::Sampler,
            DescriptorDescTy::CombinedImageSampler(_) => DescriptorType::CombinedImageSampler,
            DescriptorDescTy::SampledImage(_) => DescriptorType::SampledImage,
            DescriptorDescTy::StorageImage(_) => DescriptorType::StorageImage,
            DescriptorDescTy::UniformTexelBuffer { .. } => DescriptorType::UniformTexelBuffer,
            DescriptorDescTy::StorageTexelBuffer { .. } => DescriptorType::StorageTexelBuffer,
            DescriptorDescTy::UniformBuffer => DescriptorType::UniformBuffer,
            DescriptorDescTy::StorageBuffer => DescriptorType::StorageBuffer,
            DescriptorDescTy::UniformBufferDynamic => DescriptorType::UniformBufferDynamic,
            DescriptorDescTy::StorageBufferDynamic => DescriptorType::StorageBufferDynamic,
            DescriptorDescTy::InputAttachment { .. } => DescriptorType::InputAttachment,
        }
    }

    /// Checks whether we are a superset of another descriptor type.
    // TODO: add example
    #[inline]
    pub fn ensure_superset_of(
        &self,
        other: &DescriptorDescTy,
    ) -> Result<(), DescriptorCompatibilityError> {
        if self.ty() != other.ty() {
            return Err(DescriptorCompatibilityError::Type {
                first: self.ty(),
                second: other.ty(),
            });
        }

        match (self, other) {
            (
                DescriptorDescTy::CombinedImageSampler(ref me)
                | DescriptorDescTy::SampledImage(ref me)
                | DescriptorDescTy::StorageImage(ref me),
                DescriptorDescTy::CombinedImageSampler(ref other)
                | DescriptorDescTy::SampledImage(ref other)
                | DescriptorDescTy::StorageImage(ref other),
            ) => me.ensure_superset_of(other)?,

            (
                DescriptorDescTy::UniformTexelBuffer { format: me_format }
                | DescriptorDescTy::StorageTexelBuffer { format: me_format },
                DescriptorDescTy::UniformTexelBuffer {
                    format: other_format,
                }
                | DescriptorDescTy::StorageTexelBuffer {
                    format: other_format,
                },
            ) => {
                if other_format.is_some() && me_format != other_format {
                    return Err(DescriptorCompatibilityError::Format {
                        first: *me_format,
                        second: other_format.unwrap(),
                    });
                }
            }

            (
                DescriptorDescTy::InputAttachment {
                    multisampled: me_multisampled,
                },
                DescriptorDescTy::InputAttachment {
                    multisampled: other_multisampled,
                },
            ) => {
                if me_multisampled != other_multisampled {
                    return Err(DescriptorCompatibilityError::Multisampling {
                        first: *me_multisampled,
                        second: *other_multisampled,
                    });
                }
            }

            _ => (),
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
    DescriptorCount {
        first: u32,
        second: u32,
    },

    /// The presence or absence of a descriptor in a binding is not compatible.
    Empty {
        first: bool,
        second: bool,
    },

    // The formats of an image descriptor are not compatible.
    Format {
        first: Option<Format>,
        second: Format,
    },

    // The image view types of an image descriptor are not compatible.
    ImageViewType {
        first: ImageViewType,
        second: ImageViewType,
    },

    // The multisampling of an image descriptor is not compatible.
    Multisampling {
        first: bool,
        second: bool,
    },

    /// The mutability of the descriptors is not compatible.
    Mutability {
        first: bool,
        second: bool,
    },

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
                DescriptorCompatibilityError::Empty { .. } => {
                    "the presence or absence of a descriptor in a binding is not compatible"
                }
                DescriptorCompatibilityError::Format { .. } => {
                    "the formats of an image descriptor are not compatible"
                }
                DescriptorCompatibilityError::ImageViewType { .. } => {
                    "the image view types of an image descriptor are not compatible"
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
