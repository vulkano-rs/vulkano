// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
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

use std::cmp;
use std::ops::BitOr;
use format::Format;
use vk;

/// Contains the exact description of a single descriptor.
///
/// > **Note**: You are free to fill a `DescriptorDesc` struct the way you want, but its validity
/// > will be checked when you create a pipeline layout, a descriptor set, or when you try to bind
/// > a descriptor set.
// TODO: add example
#[derive(Debug, Clone)]
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
    // TODO: add example
    // TODO: return Result instead of bool
    #[inline]
    pub fn is_superset_of(&self, other: &DescriptorDesc) -> bool {
        self.ty.is_superset_of(&other.ty) &&
        self.array_count >= other.array_count && self.stages.is_superset_of(&other.stages) &&
        (!self.readonly || other.readonly)
    }

    /// Builds a `DescriptorDesc` that is the union of `self` and `other`, if possible.
    ///
    /// The returned value will be a superset of both `self` and `other`.
    // TODO: Result instead of Option
    // TODO: add example
    #[inline]
    pub fn union(&self, other: &DescriptorDesc) -> Option<DescriptorDesc> {
        if self.ty != other.ty { return None; }

        Some(DescriptorDesc {
            ty: self.ty.clone(),
            array_count: cmp::max(self.array_count, other.array_count),
            stages: self.stages | other.stages,
            readonly: self.readonly && other.readonly,
        })
    }
}

/// Describes the content and layout of each array element of a descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorDescTy {
    Sampler,                // TODO: the sampler has some restrictions as well
    CombinedImageSampler(DescriptorImageDesc),               // TODO: the sampler has some restrictions as well
    ImageAccess(DescriptorImageDesc),
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
        /// attached to this descriptor.
        multisampled: bool,
        array_layers: DescriptorImageDescArray,
    },
    BufferAccess(DescriptorBufferDesc),
}

impl DescriptorDescTy {
    /// Returns the type of descriptor.
    ///
    /// Returns `None` if there's not enough info to determine the type.
    // TODO: add example
    pub fn ty(&self) -> Option<DescriptorType> {
        Some(match *self {
            DescriptorDescTy::Sampler => DescriptorType::Sampler,
            DescriptorDescTy::CombinedImageSampler(_) => DescriptorType::CombinedImageSampler,
            DescriptorDescTy::ImageAccess(ref desc) => {
                if desc.sampled { DescriptorType::SampledImage }
                else { DescriptorType::StorageImage }
            },
            DescriptorDescTy::InputAttachment { .. } => DescriptorType::InputAttachment,
            DescriptorDescTy::BufferAccess(ref desc) => {
                let dynamic = match desc.dynamic { Some(d) => d, None => return None };
                match (desc.storage, dynamic) {
                    (false, false) => DescriptorType::UniformBuffer,
                    (true, false) => DescriptorType::StorageBuffer,
                    (false, true) => DescriptorType::UniformBufferDynamic,
                    (true, true) => DescriptorType::StorageBufferDynamic,
                }
            },
            DescriptorDescTy::TexelBuffer { storage, .. } => {
                if storage { DescriptorType::StorageTexelBuffer }
                else { DescriptorType::UniformTexelBuffer }
            },
        })
    }

    /// Checks whether we are a superset of another descriptor type.
    // TODO: add example
    #[inline]
    pub fn is_superset_of(&self, other: &DescriptorDescTy) -> bool {
        match (self, other) {
            (&DescriptorDescTy::Sampler, &DescriptorDescTy::Sampler) => true,

            (&DescriptorDescTy::CombinedImageSampler(ref me),
             &DescriptorDescTy::CombinedImageSampler(ref other)) => me.is_superset_of(other),

            (&DescriptorDescTy::ImageAccess(ref me),
             &DescriptorDescTy::ImageAccess(ref other)) => me.is_superset_of(other),

            (&DescriptorDescTy::InputAttachment { multisampled: me_multisampled,
                                                  array_layers: me_array_layers },
             &DescriptorDescTy::InputAttachment { multisampled: other_multisampled,
                                                  array_layers: other_array_layers }) =>
            {
                me_multisampled == other_multisampled && me_array_layers == other_array_layers
            },

            (&DescriptorDescTy::BufferAccess(ref me), &DescriptorDescTy::BufferAccess(ref other)) => {
                if me.storage != other.storage {
                    return false;
                }

                match (me.dynamic, other.dynamic) {
                    (Some(_), None) => true,
                    (Some(m), Some(o)) => m == o,
                    (None, None) => true,
                    (None, Some(_)) => false,
                }
            },

            (&DescriptorDescTy::TexelBuffer { storage: me_storage, format: me_format },
             &DescriptorDescTy::TexelBuffer { storage: other_storage, format: other_format }) =>
            {
                if me_storage != other_storage {
                    return false;
                }

                match (me_format, other_format) {
                    (Some(_), None) => true,
                    (Some(m), Some(o)) => m == o,
                    (None, None) => true,
                    (None, Some(_)) => false,
                }
            },

            // Any other combination is invalid.
            _ => false
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
    pub fn is_superset_of(&self, other: &DescriptorImageDesc) -> bool {
        if self.dimensions != other.dimensions {
            return false;
        }

        if self.multisampled != other.multisampled {
            return false;
        }

        match (self.format, other.format) {
            (Some(a), Some(b)) => if a != b { return false; },
            (Some(_), None) => (),
            (None, None) => (),
            (None, Some(_)) => return false,
        };

        match (self.array_layers, other.array_layers) {
            (DescriptorImageDescArray::NonArrayed, DescriptorImageDescArray::NonArrayed) => (),
            (DescriptorImageDescArray::Arrayed { max_layers: my_max },
             DescriptorImageDescArray::Arrayed { max_layers: other_max }) =>
            {
                match (my_max, other_max) {
                    (Some(m), Some(o)) => if m < o { return false; },
                    (Some(_), None) => (),
                    (None, _) => return false,
                };
            },
            _ => return false
        };

        true
    }
}

// TODO: documentation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DescriptorImageDescArray {
    NonArrayed,
    Arrayed { max_layers: Option<u32> }
}

// TODO: documentation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DescriptorImageDescDimensions {
    OneDimensional,
    TwoDimensional,
    ThreeDimensional,
    Cube,
}

// TODO: documentation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorBufferDesc {
    pub dynamic: Option<bool>,
    pub storage: bool,
    pub content: DescriptorBufferContentDesc,
}

// TODO: documentation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorBufferContentDesc {
    F32,
    F64,
    Struct {

    },
    Array {
        len: Box<DescriptorBufferContentDesc>, num_array: usize
    },
}

/// Describes what kind of resource may later be bound to a descriptor.
///
/// This is mostly the same as a `DescriptorDescTy` but with less precise information.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum DescriptorType {
    Sampler = vk::DESCRIPTOR_TYPE_SAMPLER,
    CombinedImageSampler = vk::DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    SampledImage = vk::DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    StorageImage = vk::DESCRIPTOR_TYPE_STORAGE_IMAGE,
    UniformTexelBuffer = vk::DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
    StorageTexelBuffer = vk::DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
    UniformBuffer = vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    StorageBuffer = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER,
    UniformBufferDynamic = vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
    StorageBufferDynamic = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
    InputAttachment = vk::DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
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
    pub fn all() -> ShaderStages {
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
    pub fn none() -> ShaderStages {
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
    pub fn all_graphics() -> ShaderStages {
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
    pub fn compute() -> ShaderStages {
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
    pub fn is_superset_of(&self, other: &ShaderStages) -> bool {
        (self.vertex || !other.vertex) &&
        (self.tessellation_control || !other.tessellation_control) &&
        (self.tessellation_evaluation || !other.tessellation_evaluation) &&
        (self.geometry || !other.geometry) &&
        (self.fragment || !other.fragment) &&
        (self.compute || !other.compute)
    }

    /// Checks whether any of the stages in `self` are also present in `other`.
    // TODO: add example
    #[inline]
    pub fn intersects(&self, other: &ShaderStages) -> bool {
        (self.vertex && other.vertex) ||
        (self.tessellation_control && other.tessellation_control) ||
        (self.tessellation_evaluation && other.tessellation_evaluation) ||
        (self.geometry && other.geometry) ||
        (self.fragment && other.fragment) ||
        (self.compute && other.compute)
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

#[doc(hidden)]
impl Into<vk::ShaderStageFlags> for ShaderStages {
    #[inline]
    fn into(self) -> vk::ShaderStageFlags {
        let mut result = 0;
        if self.vertex { result |= vk::SHADER_STAGE_VERTEX_BIT; }
        if self.tessellation_control { result |= vk::SHADER_STAGE_TESSELLATION_CONTROL_BIT; }
        if self.tessellation_evaluation { result |= vk::SHADER_STAGE_TESSELLATION_EVALUATION_BIT; }
        if self.geometry { result |= vk::SHADER_STAGE_GEOMETRY_BIT; }
        if self.fragment { result |= vk::SHADER_STAGE_FRAGMENT_BIT; }
        if self.compute { result |= vk::SHADER_STAGE_COMPUTE_BIT; }
        result
    }
}
