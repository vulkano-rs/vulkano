// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::ops::BitOr;
use format::Format;
use vk;

/// Describes a single descriptor.
#[derive(Debug, Copy, Clone)]
pub struct DescriptorDesc {
    /// Describes the content and layout of each array element of a descriptor.
    pub ty: DescriptorDescTy,

    /// How many array elements this descriptor is made of.
    pub array_count: u32,

    /// Which shader stages are going to access this descriptor.
    pub stages: ShaderStages,

    /// True if the attachment is only ever read by the shader. False if it is also written.
    pub readonly: bool,
}

impl DescriptorDesc {
    /// Checks whether we are a superset of another descriptor.
    ///
    /// This means that either the descriptor is the same, or it is the same but with a larger
    /// array elements count, or it is the same with more shader stages.
    #[inline]
    pub fn is_superset_of(&self, other: &DescriptorDesc) -> bool {
        self.ty.is_superset_of(&other.ty) &&
        self.array_count >= other.array_count && self.stages.is_superset_of(&other.stages) &&
        (!self.readonly || other.readonly)
    }

    /// Builds a `DescriptorDesc` that is the union of `self` and `other`.
    ///
    /// The returned value will be a superset of both `self` and `other`.
    // TODO: Result instead of Option
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DescriptorDescTy {
    Sampler,                // TODO: the sampler has some restrictions as well
    CombinedImageSampler(DescriptorImageDesc),               // TODO: the sampler has some restrictions as well
    Image(DescriptorImageDesc),
    TexelBuffer {
        /// If `true`, this describes a storage texel buffer.
        storage: bool,
        /// The format of the content, or `None` if the format is unknown. Depending on the
        /// context, it may be invalid to have a `None` value here.
        format: Option<Format>,
    },
    InputAttachment { multisampled: bool, array_layers: DescriptorImageDescArray },
    Buffer(DescriptorBufferDesc),
}

impl DescriptorDescTy {
    /// Returns the type of descriptor.
    ///
    /// Returns `None` if there's not enough info to determine the type.
    pub fn ty(&self) -> Option<DescriptorType> {
        Some(match *self {
            DescriptorDescTy::Sampler => DescriptorType::Sampler,
            DescriptorDescTy::CombinedImageSampler(_) => DescriptorType::CombinedImageSampler,
            DescriptorDescTy::Image(desc) => {
                if desc.sampled { DescriptorType::SampledImage }
                else { DescriptorType::StorageImage }
            },
            DescriptorDescTy::InputAttachment { .. } => DescriptorType::InputAttachment,
            DescriptorDescTy::Buffer(desc) => {
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
    #[inline]
    pub fn is_superset_of(&self, other: &DescriptorDescTy) -> bool {
        match (*self, *other) {
            (DescriptorDescTy::Sampler, DescriptorDescTy::Sampler) => true,

            (DescriptorDescTy::CombinedImageSampler(ref me),
             DescriptorDescTy::CombinedImageSampler(ref other)) => me.is_superset_of(other),

            (DescriptorDescTy::Image(ref me),
             DescriptorDescTy::Image(ref other)) => me.is_superset_of(other),

            (DescriptorDescTy::InputAttachment { multisampled: me_multisampled,
                                                 array_layers: me_array_layers },
             DescriptorDescTy::InputAttachment { multisampled: other_multisampled,
                                                 array_layers: other_array_layers }) =>
            {
                me_multisampled == other_multisampled && me_array_layers == other_array_layers
            },

            (DescriptorDescTy::Buffer(ref me), DescriptorDescTy::Buffer(ref other)) => {
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

            (DescriptorDescTy::TexelBuffer { storage: me_storage, format: me_format },
             DescriptorDescTy::TexelBuffer { storage: other_storage, format: other_format }) =>
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DescriptorImageDesc {
    pub sampled: bool,
    pub dimensions: DescriptorImageDescDimensions,
    /// The format of the image, or `None` if the format is unknown.
    pub format: Option<Format>,
    /// True if the image is multisampled.
    pub multisampled: bool,
    pub array_layers: DescriptorImageDescArray,
}

impl DescriptorImageDesc {
    /// Checks whether we are a superset of another image.
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DescriptorImageDescArray {
    NonArrayed,
    Arrayed { max_layers: Option<u32> }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DescriptorImageDescDimensions {
    OneDimensional,
    TwoDimensional,
    ThreeDimensional,
    Cube,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DescriptorBufferDesc {
    pub dynamic: Option<bool>,
    pub storage: bool,
    // FIXME: store content
}

/// Describes what kind of resource may later be bound to a descriptor.
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

    /// Creates a `ShaderStages` struct will all graphics stages set to `true`.
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

    /// Creates a `ShaderStages` struct will the compute stage set to `true`.
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
    #[inline]
    pub fn is_superset_of(&self, other: &ShaderStages) -> bool {
        (self.vertex || !other.vertex) &&
        (self.tessellation_control || !other.tessellation_control) &&
        (self.tessellation_evaluation || !other.tessellation_evaluation) &&
        (self.geometry || !other.geometry) &&
        (self.fragment || !other.fragment) &&
        (self.compute || !other.compute)
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
