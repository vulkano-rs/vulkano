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
use crate::pipeline::shader::ShaderStagesSupersetError;
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
                self.descriptor(binding_num)
                    .map_or(false, |desc| match desc.ty {
                        DescriptorDescTy::Buffer(_) => true,
                        _ => false,
                    }),
                "tried to make the non-buffer descriptor at binding {} a dynamic buffer",
                binding_num
            );

            let binding = self
                .descriptors
                .get_mut(binding_num)
                .and_then(|b| b.as_mut());

            if let Some(desc) = binding {
                if let DescriptorDescTy::Buffer(ref buffer_desc) = desc.ty {
                    desc.ty = DescriptorDescTy::Buffer(DescriptorBufferDesc {
                        dynamic: Some(true),
                        ..*buffer_desc
                    });
                }
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

    /// Returns whether `self` is a superset of `other`.
    pub fn ensure_superset_of(
        &self,
        other: &DescriptorSetDesc,
    ) -> Result<(), DescriptorSetDescSupersetError> {
        if self.descriptors.len() < other.descriptors.len() {
            return Err(DescriptorSetDescSupersetError::DescriptorsCountMismatch {
                self_num: self.descriptors.len() as u32,
                other_num: other.descriptors.len() as u32,
            });
        }

        for binding_num in 0..other.descriptors.len() {
            let self_desc = self.descriptor(binding_num);
            let other_desc = self.descriptor(binding_num);

            match (self_desc, other_desc) {
                (Some(mine), Some(other)) => {
                    if let Err(err) = mine.ensure_superset_of(&other) {
                        return Err(DescriptorSetDescSupersetError::IncompatibleDescriptors {
                            error: err,
                            binding_num: binding_num as u32,
                        });
                    }
                }
                (None, Some(_)) => {
                    return Err(DescriptorSetDescSupersetError::ExpectedEmptyDescriptor {
                        binding_num: binding_num as u32,
                    })
                }
                _ => (),
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
    pub array_count: u32,

    /// Which shader stages are going to access this descriptor.
    pub stages: ShaderStages,

    /// True if the attachment is only ever read by the shader. False if it is also written.
    pub readonly: bool,
}

impl DescriptorDesc {
    /// Checks whether we are a superset of another descriptor.
    ///
    /// Returns true if `self` is the same descriptor as `other`, or if `self` is the same as
    /// `other` but with a larger array elements count and/or more shader stages.
    ///
    ///# Example
    ///```
    ///use vulkano::descriptor_set::layout::DescriptorDesc;
    ///use vulkano::descriptor_set::layout::DescriptorDescTy::*;
    ///use vulkano::pipeline::shader::ShaderStages;
    ///
    ///let desc_super = DescriptorDesc{ ty: Sampler, array_count: 2, stages: ShaderStages{
    ///  vertex: true,
    ///  tessellation_control: true,
    ///  tessellation_evaluation: true,
    ///  geometry: true,
    ///  fragment: true,
    ///  compute: true
    ///}, readonly: false };
    ///let desc_sub = DescriptorDesc{ ty: Sampler, array_count: 1, stages: ShaderStages{
    ///  vertex: true,
    ///  tessellation_control: false,
    ///  tessellation_evaluation: false,
    ///  geometry: false,
    ///  fragment: true,
    ///  compute: false
    ///}, readonly: true };
    ///
    ///assert_eq!(desc_super.ensure_superset_of(&desc_sub).unwrap(), ());
    ///
    ///```
    #[inline]
    pub fn ensure_superset_of(
        &self,
        other: &DescriptorDesc,
    ) -> Result<(), DescriptorDescSupersetError> {
        self.ty.ensure_superset_of(&other.ty)?;
        self.stages.ensure_superset_of(&other.stages)?;

        if self.array_count < other.array_count {
            return Err(DescriptorDescSupersetError::ArrayTooSmall {
                len: self.array_count,
                required: other.array_count,
            });
        }

        if self.readonly && !other.readonly {
            return Err(DescriptorDescSupersetError::MutabilityRequired);
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
    ///let desc_part1 = DescriptorDesc{ ty: Sampler, array_count: 2, stages: ShaderStages{
    ///  vertex: true,
    ///  tessellation_control: true,
    ///  tessellation_evaluation: false,
    ///  geometry: true,
    ///  fragment: false,
    ///  compute: true
    ///}, readonly: false };
    ///
    ///let desc_part2 = DescriptorDesc{ ty: Sampler, array_count: 1, stages: ShaderStages{
    ///  vertex: true,
    ///  tessellation_control: false,
    ///  tessellation_evaluation: true,
    ///  geometry: false,
    ///  fragment: true,
    ///  compute: true
    ///}, readonly: true };
    ///
    ///let desc_union = DescriptorDesc{ ty: Sampler, array_count: 2, stages: ShaderStages{
    ///  vertex: true,
    ///  tessellation_control: true,
    ///  tessellation_evaluation: true,
    ///  geometry: true,
    ///  fragment: true,
    ///  compute: true
    ///}, readonly: false };
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
                array_count: cmp::max(first.array_count, second.array_count),
                stages: first.stages | second.stages,
                readonly: first.readonly && second.readonly,
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

        let access = match self.ty {
            DescriptorDescTy::Sampler => panic!(),
            DescriptorDescTy::CombinedImageSampler(_) | DescriptorDescTy::Image(_) => AccessFlags {
                shader_read: true,
                shader_write: !self.readonly,
                ..AccessFlags::none()
            },
            DescriptorDescTy::TexelBuffer { .. } => AccessFlags {
                shader_read: true,
                shader_write: !self.readonly,
                ..AccessFlags::none()
            },
            DescriptorDescTy::InputAttachment { .. } => AccessFlags {
                input_attachment_read: true,
                ..AccessFlags::none()
            },
            DescriptorDescTy::Buffer(ref buf) => {
                if buf.storage {
                    AccessFlags {
                        shader_read: true,
                        shader_write: !self.readonly,
                        ..AccessFlags::none()
                    }
                } else {
                    AccessFlags {
                        uniform_read: true,
                        ..AccessFlags::none()
                    }
                }
            }
        };

        (stages, access)
    }
}

/// Describes the content and layout of each array element of a descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorDescTy {
    Sampler,                                   // TODO: the sampler has some restrictions as well
    CombinedImageSampler(DescriptorImageDesc), // TODO: the sampler has some restrictions as well
    Image(DescriptorImageDesc),
    TexelBuffer {
        /// If `true`, this describes a storage texel buffer.
        storage: bool,

        /// The format of the content, or `None` if the format is unknown. Depending on the
        /// context, it may be invalid to have a `None` value here. If the format is `Some`, only
        /// buffer views that have this exact format can be attached to this descriptor.
        format: Option<Format>,
    },
    InputAttachment {
        /// If `true`, the input attachment is multisampled. Only multisampled images can be
        /// attached to this descriptor. If `false`, only single-sampled images can be attached.
        multisampled: bool,
        array_layers: DescriptorImageDescArray,
    },
    Buffer(DescriptorBufferDesc),
}

impl DescriptorDescTy {
    /// Returns the type of descriptor.
    // TODO: add example
    pub fn ty(&self) -> DescriptorType {
        match *self {
            DescriptorDescTy::Sampler => DescriptorType::Sampler,
            DescriptorDescTy::CombinedImageSampler(_) => DescriptorType::CombinedImageSampler,
            DescriptorDescTy::Image(ref desc) => {
                if desc.sampled {
                    DescriptorType::SampledImage
                } else {
                    DescriptorType::StorageImage
                }
            }
            DescriptorDescTy::InputAttachment { .. } => DescriptorType::InputAttachment,
            DescriptorDescTy::Buffer(ref desc) => {
                let dynamic = desc.dynamic.unwrap_or(false);
                match (desc.storage, dynamic) {
                    (false, false) => DescriptorType::UniformBuffer,
                    (true, false) => DescriptorType::StorageBuffer,
                    (false, true) => DescriptorType::UniformBufferDynamic,
                    (true, true) => DescriptorType::StorageBufferDynamic,
                }
            }
            DescriptorDescTy::TexelBuffer { storage, .. } => {
                if storage {
                    DescriptorType::StorageTexelBuffer
                } else {
                    DescriptorType::UniformTexelBuffer
                }
            }
        }
    }

    /// Checks whether we are a superset of another descriptor type.
    // TODO: add example
    #[inline]
    pub fn ensure_superset_of(
        &self,
        other: &DescriptorDescTy,
    ) -> Result<(), DescriptorDescSupersetError> {
        match (self, other) {
            (&DescriptorDescTy::Sampler, &DescriptorDescTy::Sampler) => Ok(()),

            (
                &DescriptorDescTy::CombinedImageSampler(ref me),
                &DescriptorDescTy::CombinedImageSampler(ref other),
            ) => me.ensure_superset_of(other),

            (&DescriptorDescTy::Image(ref me), &DescriptorDescTy::Image(ref other)) => {
                me.ensure_superset_of(other)
            }

            (
                &DescriptorDescTy::InputAttachment {
                    multisampled: me_multisampled,
                    array_layers: me_array_layers,
                },
                &DescriptorDescTy::InputAttachment {
                    multisampled: other_multisampled,
                    array_layers: other_array_layers,
                },
            ) => {
                if me_multisampled != other_multisampled {
                    return Err(DescriptorDescSupersetError::MultisampledMismatch {
                        provided: me_multisampled,
                        expected: other_multisampled,
                    });
                }

                if me_array_layers != other_array_layers {
                    return Err(DescriptorDescSupersetError::IncompatibleArrayLayers {
                        provided: me_array_layers,
                        required: other_array_layers,
                    });
                }

                Ok(())
            }

            (&DescriptorDescTy::Buffer(ref me), &DescriptorDescTy::Buffer(ref other)) => {
                if me.storage != other.storage {
                    return Err(DescriptorDescSupersetError::TypeMismatch);
                }

                match (me.dynamic, other.dynamic) {
                    (Some(_), None) => Ok(()),
                    (Some(m), Some(o)) => {
                        if m == o {
                            Ok(())
                        } else {
                            Err(DescriptorDescSupersetError::TypeMismatch)
                        }
                    }
                    (None, None) => Ok(()),
                    (None, Some(_)) => Err(DescriptorDescSupersetError::TypeMismatch),
                }
            }

            (
                &DescriptorDescTy::TexelBuffer {
                    storage: me_storage,
                    format: me_format,
                },
                &DescriptorDescTy::TexelBuffer {
                    storage: other_storage,
                    format: other_format,
                },
            ) => {
                if me_storage != other_storage {
                    return Err(DescriptorDescSupersetError::TypeMismatch);
                }

                match (me_format, other_format) {
                    (Some(_), None) => Ok(()),
                    (Some(m), Some(o)) => {
                        if m == o {
                            Ok(())
                        } else {
                            Err(DescriptorDescSupersetError::FormatMismatch {
                                provided: Some(m),
                                expected: o,
                            })
                        }
                    }
                    (None, None) => Ok(()),
                    (None, Some(a)) => Err(DescriptorDescSupersetError::FormatMismatch {
                        provided: Some(a),
                        expected: a,
                    }),
                }
            }

            // Any other combination is invalid.
            _ => Err(DescriptorDescSupersetError::TypeMismatch),
        }
    }
}

/// Additional description for descriptors that contain images.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DescriptorImageDesc {
    /// If `true`, the image can be sampled by the shader. Only images that were created with the
    /// `sampled` usage can be attached to the descriptor.
    pub sampled: bool,
    /// The kind of image: one-dimensional, two-dimensional, three-dimensional, or cube.
    pub dimensions: DescriptorImageDescDimensions,
    /// The format of the image, or `None` if the format is unknown. If `Some`, only images with
    /// exactly that format can be attached.
    pub format: Option<Format>,
    /// True if the image is multisampled.
    pub multisampled: bool,
    /// Whether the descriptor contains one or more array layers of an image.
    pub array_layers: DescriptorImageDescArray,
}

impl DescriptorImageDesc {
    /// Checks whether we are a superset of another image.
    // TODO: add example
    #[inline]
    pub fn ensure_superset_of(
        &self,
        other: &DescriptorImageDesc,
    ) -> Result<(), DescriptorDescSupersetError> {
        if self.dimensions != other.dimensions {
            return Err(DescriptorDescSupersetError::DimensionsMismatch {
                provided: self.dimensions,
                expected: other.dimensions,
            });
        }

        if self.multisampled != other.multisampled {
            return Err(DescriptorDescSupersetError::MultisampledMismatch {
                provided: self.multisampled,
                expected: other.multisampled,
            });
        }

        match (self.format, other.format) {
            (Some(a), Some(b)) => {
                if a != b {
                    return Err(DescriptorDescSupersetError::FormatMismatch {
                        provided: Some(a),
                        expected: b,
                    });
                }
            }
            (Some(_), None) => (),
            (None, None) => (),
            (None, Some(a)) => {
                return Err(DescriptorDescSupersetError::FormatMismatch {
                    provided: None,
                    expected: a,
                });
            }
        };

        match (self.array_layers, other.array_layers) {
            (DescriptorImageDescArray::NonArrayed, DescriptorImageDescArray::NonArrayed) => (),
            (
                DescriptorImageDescArray::Arrayed { max_layers: my_max },
                DescriptorImageDescArray::Arrayed {
                    max_layers: other_max,
                },
            ) => {
                match (my_max, other_max) {
                    (Some(m), Some(o)) => {
                        if m < o {
                            return Err(DescriptorDescSupersetError::IncompatibleArrayLayers {
                                provided: DescriptorImageDescArray::Arrayed { max_layers: my_max },
                                required: DescriptorImageDescArray::Arrayed {
                                    max_layers: other_max,
                                },
                            });
                        }
                    }
                    (Some(_), None) => (),
                    (None, Some(m)) => {
                        return Err(DescriptorDescSupersetError::IncompatibleArrayLayers {
                            provided: DescriptorImageDescArray::Arrayed { max_layers: my_max },
                            required: DescriptorImageDescArray::Arrayed {
                                max_layers: other_max,
                            },
                        });
                    }
                    (None, None) => (), // TODO: is this correct?
                };
            }
            (a, b) => {
                return Err(DescriptorDescSupersetError::IncompatibleArrayLayers {
                    provided: a,
                    required: b,
                })
            }
        };

        Ok(())
    }
}

// TODO: documentation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DescriptorImageDescArray {
    NonArrayed,
    Arrayed { max_layers: Option<u32> },
}

// TODO: documentation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DescriptorImageDescDimensions {
    OneDimensional,
    TwoDimensional,
    ThreeDimensional,
    Cube,
}

impl DescriptorImageDescDimensions {
    /// Builds the `DescriptorImageDescDimensions` that corresponds to the view type.
    #[inline]
    pub fn from_image_view_type(ty: ImageViewType) -> DescriptorImageDescDimensions {
        match ty {
            ImageViewType::Dim1d => DescriptorImageDescDimensions::OneDimensional,
            ImageViewType::Dim1dArray => DescriptorImageDescDimensions::OneDimensional,
            ImageViewType::Dim2d => DescriptorImageDescDimensions::TwoDimensional,
            ImageViewType::Dim2dArray => DescriptorImageDescDimensions::TwoDimensional,
            ImageViewType::Dim3d => DescriptorImageDescDimensions::ThreeDimensional,
            ImageViewType::Cubemap => DescriptorImageDescDimensions::Cube,
            ImageViewType::CubemapArray => DescriptorImageDescDimensions::Cube,
        }
    }
}

/// Additional description for descriptors that contain buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorBufferDesc {
    /// If `true`, this buffer is a dynamic buffer. Assumes false if `None`.
    pub dynamic: Option<bool>,
    /// If `true`, this buffer is a storage buffer.
    pub storage: bool,
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

/// Error when checking whether a descriptor set is a superset of another one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorSetDescSupersetError {
    /// There are more descriptors in the child than in the parent layout.
    DescriptorsCountMismatch { self_num: u32, other_num: u32 },

    /// Expected an empty descriptor, but got something instead.
    ExpectedEmptyDescriptor { binding_num: u32 },

    /// Two descriptors are incompatible.
    IncompatibleDescriptors {
        error: DescriptorDescSupersetError,
        binding_num: u32,
    },
}

impl error::Error for DescriptorSetDescSupersetError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            DescriptorSetDescSupersetError::IncompatibleDescriptors { ref error, .. } => {
                Some(error)
            }
            _ => None,
        }
    }
}

impl fmt::Display for DescriptorSetDescSupersetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                DescriptorSetDescSupersetError::DescriptorsCountMismatch { .. } => {
                    "there are more descriptors in the child than in the parent layout"
                }
                DescriptorSetDescSupersetError::ExpectedEmptyDescriptor { .. } => {
                    "expected an empty descriptor, but got something instead"
                }
                DescriptorSetDescSupersetError::IncompatibleDescriptors { .. } => {
                    "two descriptors are incompatible"
                }
            }
        )
    }
}

/// Error when checking whether a descriptor is a superset of another one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorDescSupersetError {
    /// The number of array elements of the descriptor is smaller than expected.
    ArrayTooSmall {
        len: u32,
        required: u32,
    },

    DimensionsMismatch {
        provided: DescriptorImageDescDimensions,
        expected: DescriptorImageDescDimensions,
    },

    FormatMismatch {
        provided: Option<Format>,
        expected: Format,
    },

    IncompatibleArrayLayers {
        provided: DescriptorImageDescArray,
        required: DescriptorImageDescArray,
    },

    MultisampledMismatch {
        provided: bool,
        expected: bool,
    },

    /// The descriptor is marked as read-only, but the other is not.
    MutabilityRequired,

    /// The shader stages are not a superset of one another.
    ShaderStagesNotSuperset,

    /// The descriptor type doesn't match the type of the other descriptor.
    TypeMismatch,
}

impl error::Error for DescriptorDescSupersetError {}

impl fmt::Display for DescriptorDescSupersetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                DescriptorDescSupersetError::ArrayTooSmall { .. } => {
                    "the number of array elements of the descriptor is smaller than expected"
                }
                DescriptorDescSupersetError::TypeMismatch => {
                    "the descriptor type doesn't match the type of the other descriptor"
                }
                DescriptorDescSupersetError::MutabilityRequired => {
                    "the descriptor is marked as read-only, but the other is not"
                }
                DescriptorDescSupersetError::ShaderStagesNotSuperset => {
                    "the shader stages are not a superset of one another"
                }
                DescriptorDescSupersetError::DimensionsMismatch { .. } => {
                    "mismatch between the dimensions of the two descriptors"
                }
                DescriptorDescSupersetError::FormatMismatch { .. } => {
                    "mismatch between the format of the two descriptors"
                }
                DescriptorDescSupersetError::MultisampledMismatch { .. } => {
                    "mismatch between whether the descriptors are multisampled"
                }
                DescriptorDescSupersetError::IncompatibleArrayLayers { .. } => {
                    "the array layers of the descriptors aren't compatible"
                }
            }
        )
    }
}

impl From<ShaderStagesSupersetError> for DescriptorDescSupersetError {
    #[inline]
    fn from(err: ShaderStagesSupersetError) -> DescriptorDescSupersetError {
        match err {
            ShaderStagesSupersetError::NotSuperset => {
                DescriptorDescSupersetError::ShaderStagesNotSuperset
            }
        }
    }
}
