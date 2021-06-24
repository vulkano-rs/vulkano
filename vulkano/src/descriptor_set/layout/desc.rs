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
use crate::sync::AccessFlags;
use crate::sync::PipelineStages;
use crate::SafeDeref;
use std::cmp;
use std::error;
use std::fmt;
use std::ops::BitOr;

/// Trait for objects that describe the layout of the descriptors of a set.
pub unsafe trait DescriptorSetDesc {
    /// Returns the number of binding slots in the set.
    fn num_bindings(&self) -> usize;

    /// Returns a description of a descriptor, or `None` if out of range.
    fn descriptor(&self, binding: usize) -> Option<DescriptorDesc>;
}

unsafe impl<T> DescriptorSetDesc for T
where
    T: SafeDeref,
    T::Target: DescriptorSetDesc,
{
    #[inline]
    fn num_bindings(&self) -> usize {
        (**self).num_bindings()
    }

    #[inline]
    fn descriptor(&self, binding: usize) -> Option<DescriptorDesc> {
        (**self).descriptor(binding)
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
    ///use vulkano::descriptor_set::layout::ShaderStages;
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
    ///assert_eq!(desc_super.is_superset_of(&desc_sub).unwrap(), ());
    ///
    ///```
    #[inline]
    pub fn is_superset_of(
        &self,
        other: &DescriptorDesc,
    ) -> Result<(), DescriptorDescSupersetError> {
        self.ty.is_superset_of(&other.ty)?;
        self.stages.is_superset_of(&other.stages)?;

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
    /// The returned value will be a superset of both `self` and `other`.
    ///
    ///# Example
    ///```
    ///use vulkano::descriptor_set::layout::DescriptorDesc;
    ///use vulkano::descriptor_set::layout::DescriptorDescTy::*;
    ///use vulkano::descriptor_set::layout::ShaderStages;
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
    ///assert_eq!(desc_part1.union(&desc_part2), Some(desc_union));
    ///```
    // TODO: Result instead of Option
    #[inline]
    pub fn union(&self, other: &DescriptorDesc) -> Option<DescriptorDesc> {
        if self.ty != other.ty {
            return None;
        }

        Some(DescriptorDesc {
            ty: self.ty.clone(),
            array_count: cmp::max(self.array_count, other.array_count),
            stages: self.stages | other.stages,
            readonly: self.readonly && other.readonly,
        })
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
    pub fn is_superset_of(
        &self,
        other: &DescriptorDescTy,
    ) -> Result<(), DescriptorDescSupersetError> {
        match (self, other) {
            (&DescriptorDescTy::Sampler, &DescriptorDescTy::Sampler) => Ok(()),

            (
                &DescriptorDescTy::CombinedImageSampler(ref me),
                &DescriptorDescTy::CombinedImageSampler(ref other),
            ) => me.is_superset_of(other),

            (&DescriptorDescTy::Image(ref me), &DescriptorDescTy::Image(ref other)) => {
                me.is_superset_of(other)
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
    pub fn is_superset_of(
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

/// Error when checking whether a descriptor is a superset of another one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorDescSupersetError {
    /// The number of array elements of the descriptor is smaller than expected.
    ArrayTooSmall {
        len: u32,
        required: u32,
    },

    /// The descriptor type doesn't match the type of the other descriptor.
    TypeMismatch,

    /// The descriptor is marked as read-only, but the other is not.
    MutabilityRequired,

    /// The shader stages are not a superset of one another.
    ShaderStagesNotSuperset,

    DimensionsMismatch {
        provided: DescriptorImageDescDimensions,
        expected: DescriptorImageDescDimensions,
    },

    FormatMismatch {
        provided: Option<Format>,
        expected: Format,
    },

    MultisampledMismatch {
        provided: bool,
        expected: bool,
    },

    IncompatibleArrayLayers {
        provided: DescriptorImageDescArray,
        required: DescriptorImageDescArray,
    },
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

/// Describes which shader stages have access to a descriptor.
// TODO: add example with BitOr
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ShaderStages {
    /// `True` means that the descriptor will be used by the vertex shader.
    pub vertex: bool,
    /// `True` means that the descriptor will be used by the tessellation control shader.
    pub tessellation_control: bool,
    /// `True` means that the descriptor will be used by the tessellation evaluation shader.
    pub tessellation_evaluation: bool,
    /// `True` means that the descriptor will be used by the geometry shader.
    pub geometry: bool,
    /// `True` means that the descriptor will be used by the fragment shader.
    pub fragment: bool,
    /// `True` means that the descriptor will be used by the compute shader.
    pub compute: bool,
}

impl ShaderStages {
    /// Creates a `ShaderStages` struct will all stages set to `true`.
    // TODO: add example
    #[inline]
    pub const fn all() -> ShaderStages {
        ShaderStages {
            vertex: true,
            tessellation_control: true,
            tessellation_evaluation: true,
            geometry: true,
            fragment: true,
            compute: true,
        }
    }

    /// Creates a `ShaderStages` struct will all stages set to `false`.
    // TODO: add example
    #[inline]
    pub const fn none() -> ShaderStages {
        ShaderStages {
            vertex: false,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: false,
            compute: false,
        }
    }

    /// Creates a `ShaderStages` struct with all graphics stages set to `true`.
    // TODO: add example
    #[inline]
    pub const fn all_graphics() -> ShaderStages {
        ShaderStages {
            vertex: true,
            tessellation_control: true,
            tessellation_evaluation: true,
            geometry: true,
            fragment: true,
            compute: false,
        }
    }

    /// Creates a `ShaderStages` struct with the compute stage set to `true`.
    // TODO: add example
    #[inline]
    pub const fn compute() -> ShaderStages {
        ShaderStages {
            vertex: false,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: false,
            compute: true,
        }
    }

    /// Checks whether we have more stages enabled than `other`.
    // TODO: add example
    #[inline]
    pub const fn is_superset_of(
        &self,
        other: &ShaderStages,
    ) -> Result<(), ShaderStagesSupersetError> {
        if (self.vertex || !other.vertex)
            && (self.tessellation_control || !other.tessellation_control)
            && (self.tessellation_evaluation || !other.tessellation_evaluation)
            && (self.geometry || !other.geometry)
            && (self.fragment || !other.fragment)
            && (self.compute || !other.compute)
        {
            Ok(())
        } else {
            Err(ShaderStagesSupersetError::NotSuperset)
        }
    }

    /// Checks whether any of the stages in `self` are also present in `other`.
    // TODO: add example
    #[inline]
    pub const fn intersects(&self, other: &ShaderStages) -> bool {
        (self.vertex && other.vertex)
            || (self.tessellation_control && other.tessellation_control)
            || (self.tessellation_evaluation && other.tessellation_evaluation)
            || (self.geometry && other.geometry)
            || (self.fragment && other.fragment)
            || (self.compute && other.compute)
    }
}

impl From<ShaderStages> for ash::vk::ShaderStageFlags {
    #[inline]
    fn from(val: ShaderStages) -> Self {
        let mut result = ash::vk::ShaderStageFlags::empty();
        if val.vertex {
            result |= ash::vk::ShaderStageFlags::VERTEX;
        }
        if val.tessellation_control {
            result |= ash::vk::ShaderStageFlags::TESSELLATION_CONTROL;
        }
        if val.tessellation_evaluation {
            result |= ash::vk::ShaderStageFlags::TESSELLATION_EVALUATION;
        }
        if val.geometry {
            result |= ash::vk::ShaderStageFlags::GEOMETRY;
        }
        if val.fragment {
            result |= ash::vk::ShaderStageFlags::FRAGMENT;
        }
        if val.compute {
            result |= ash::vk::ShaderStageFlags::COMPUTE;
        }
        result
    }
}

impl From<ash::vk::ShaderStageFlags> for ShaderStages {
    #[inline]
    fn from(val: ash::vk::ShaderStageFlags) -> Self {
        Self {
            vertex: val.intersects(ash::vk::ShaderStageFlags::VERTEX),
            tessellation_control: val.intersects(ash::vk::ShaderStageFlags::TESSELLATION_CONTROL),
            tessellation_evaluation: val
                .intersects(ash::vk::ShaderStageFlags::TESSELLATION_EVALUATION),
            geometry: val.intersects(ash::vk::ShaderStageFlags::GEOMETRY),
            fragment: val.intersects(ash::vk::ShaderStageFlags::FRAGMENT),
            compute: val.intersects(ash::vk::ShaderStageFlags::COMPUTE),
        }
    }
}

impl BitOr for ShaderStages {
    type Output = ShaderStages;

    #[inline]
    fn bitor(self, other: ShaderStages) -> ShaderStages {
        ShaderStages {
            vertex: self.vertex || other.vertex,
            tessellation_control: self.tessellation_control || other.tessellation_control,
            tessellation_evaluation: self.tessellation_evaluation || other.tessellation_evaluation,
            geometry: self.geometry || other.geometry,
            fragment: self.fragment || other.fragment,
            compute: self.compute || other.compute,
        }
    }
}

impl From<ShaderStages> for PipelineStages {
    #[inline]
    fn from(stages: ShaderStages) -> PipelineStages {
        PipelineStages {
            vertex_shader: stages.vertex,
            tessellation_control_shader: stages.tessellation_control,
            tessellation_evaluation_shader: stages.tessellation_evaluation,
            geometry_shader: stages.geometry,
            fragment_shader: stages.fragment,
            compute_shader: stages.compute,
            ..PipelineStages::none()
        }
    }
}

/// Error when checking whether some shader stages are superset of others.
#[derive(Debug, Clone)]
pub enum ShaderStagesSupersetError {
    NotSuperset,
}

impl error::Error for ShaderStagesSupersetError {}

impl fmt::Display for ShaderStagesSupersetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                ShaderStagesSupersetError::NotSuperset => "shader stages not a superset",
            }
        )
    }
}
