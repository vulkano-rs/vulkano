// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::Range;
use std::sync::Arc;

use vk;

use buffer::BufferSlice;
use buffer::traits::Buffer;
use image::traits::ImageView;
use image::traits::Image;
use sampler::Sampler;

/// Represents a single write entry to a descriptor set.
pub struct DescriptorWrite {
    binding: u32,
    first_array_element: u32,
    inner: DescriptorWriteInner,
}

// FIXME: incomplete
// TODO: hacky visibility
#[derive(Clone)]        // TODO: Debug
#[doc(hidden)]
pub enum DescriptorWriteInner {
    StorageImage(Arc<ImageView>, Arc<Image>, Vec<(u32, u32)>),
    Sampler(Arc<Sampler>),
    SampledImage(Arc<ImageView>, Arc<Image>, Vec<(u32, u32)>),
    CombinedImageSampler(Arc<Sampler>, Arc<ImageView>, Arc<Image>, Vec<(u32, u32)>),
    //UniformTexelBuffer(Arc<Buffer>),      // FIXME: requires buffer views
    //StorageTexelBuffer(Arc<Buffer>),      // FIXME: requires buffer views
    UniformBuffer { buffer: Arc<Buffer>, offset: usize, size: usize },
    StorageBuffer { buffer: Arc<Buffer>, offset: usize, size: usize },
    DynamicUniformBuffer { buffer: Arc<Buffer>, offset: usize, size: usize },
    DynamicStorageBuffer { buffer: Arc<Buffer>, offset: usize, size: usize },
    InputAttachment(Arc<ImageView>, Arc<Image>, Vec<(u32, u32)>),
}

impl DescriptorWrite {
    #[inline]
    pub fn storage_image<I>(binding: u32, image: &Arc<I>) -> DescriptorWrite
        where I: ImageView + 'static
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::StorageImage(image.clone(), ImageView::parent_arc(image), image.blocks())
        }
    }

    #[inline]
    pub fn sampler(binding: u32, sampler: &Arc<Sampler>) -> DescriptorWrite {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::Sampler(sampler.clone())
        }
    }

    #[inline]
    pub fn sampled_image<I>(binding: u32, image: &Arc<I>) -> DescriptorWrite
        where I: ImageView + 'static
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::SampledImage(image.clone(), ImageView::parent_arc(image), image.blocks())
        }
    }

    #[inline]
    pub fn combined_image_sampler<I>(binding: u32, sampler: &Arc<Sampler>, image: &Arc<I>) -> DescriptorWrite
        where I: ImageView + 'static
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::CombinedImageSampler(sampler.clone(), image.clone(), ImageView::parent_arc(image), image.blocks())
        }
    }

    #[inline]
    pub fn uniform_buffer<'a, S, T: ?Sized, B>(binding: u32, buffer: S) -> DescriptorWrite
        where S: Into<BufferSlice<'a, T, B>>, B: Buffer + 'static
    {
        let buffer = buffer.into();

        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::UniformBuffer {
                buffer: buffer.buffer().clone(),
                offset: buffer.offset(),
                size: buffer.size(),
            }
        }
    }

    #[inline]
    pub unsafe fn unchecked_uniform_buffer<B>(binding: u32, buffer: &Arc<B>, range: Range<usize>)
                                              -> DescriptorWrite
        where B: Buffer + 'static
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::UniformBuffer {
                buffer: buffer.clone(),
                offset: range.start,
                size: range.end - range.start,
            }
        }
    }

    #[inline]
    pub fn storage_buffer<'a, S, T: ?Sized, B>(binding: u32, buffer: S) -> DescriptorWrite
        where S: Into<BufferSlice<'a, T, B>>, B: Buffer + 'static
    {
        let buffer = buffer.into();

        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::StorageBuffer {
                buffer: buffer.buffer().clone(),
                offset: buffer.offset(),
                size: buffer.size(),
            }
        }
    }

    #[inline]
    pub unsafe fn unchecked_storage_buffer<B>(binding: u32, buffer: &Arc<B>, range: Range<usize>)
                                              -> DescriptorWrite
        where B: Buffer + 'static
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::StorageBuffer {
                buffer: buffer.clone(),
                offset: range.start,
                size: range.end - range.start,
            }
        }
    }

    #[inline]
    pub fn dynamic_uniform_buffer<'a, S, T: ?Sized, B>(binding: u32, buffer: S) -> DescriptorWrite
        where S: Into<BufferSlice<'a, T, B>>, B: Buffer + 'static
    {
        let buffer = buffer.into();

        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::DynamicUniformBuffer {
                buffer: buffer.buffer().clone(),
                offset: buffer.offset(),
                size: buffer.size(),
            }
        }
    }

    #[inline]
    pub unsafe fn unchecked_dynamic_uniform_buffer<B>(binding: u32, buffer: &Arc<B>, range: Range<usize>)
                                                      -> DescriptorWrite
        where B: Buffer + 'static
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::DynamicUniformBuffer {
                buffer: buffer.clone(),
                offset: range.start,
                size: range.end - range.start,
            }
        }
    }

    #[inline]
    pub fn dynamic_storage_buffer<'a, S, T: ?Sized, B>(binding: u32, buffer: S) -> DescriptorWrite
        where S: Into<BufferSlice<'a, T, B>>, B: Buffer + 'static
    {
        let buffer = buffer.into();

        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::DynamicStorageBuffer {
                buffer: buffer.buffer().clone(),
                offset: buffer.offset(),
                size: buffer.size(),
            }
        }
    }

    #[inline]
    pub unsafe fn unchecked_dynamic_storage_buffer<B>(binding: u32, buffer: &Arc<B>, range: Range<usize>)
                                                      -> DescriptorWrite
        where B: Buffer + 'static
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::DynamicStorageBuffer {
                buffer: buffer.clone(),
                offset: range.start,
                size: range.end - range.start,
            }
        }
    }

    #[inline]
    pub fn input_attachment<I>(binding: u32, image: &Arc<I>) -> DescriptorWrite
        where I: ImageView + 'static
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: DescriptorWriteInner::InputAttachment(image.clone(), ImageView::parent_arc(image), image.blocks())
        }
    }

    /// Returns the type corresponding to this write.
    #[inline]
    pub fn ty(&self) -> DescriptorType {
        match self.inner {
            DescriptorWriteInner::Sampler(_) => DescriptorType::Sampler,
            DescriptorWriteInner::CombinedImageSampler(_, _, _, _) => DescriptorType::CombinedImageSampler,
            DescriptorWriteInner::SampledImage(_, _, _) => DescriptorType::SampledImage,
            DescriptorWriteInner::StorageImage(_, _, _) => DescriptorType::StorageImage,
            //DescriptorWriteInner::UniformTexelBuffer(_) => DescriptorType::UniformTexelBuffer,
            //DescriptorWriteInner::StorageTexelBuffer(_) => DescriptorType::StorageTexelBuffer,
            DescriptorWriteInner::UniformBuffer { .. } => DescriptorType::UniformBuffer,
            DescriptorWriteInner::StorageBuffer { .. } => DescriptorType::StorageBuffer,
            DescriptorWriteInner::DynamicUniformBuffer { .. } => DescriptorType::UniformBufferDynamic,
            DescriptorWriteInner::DynamicStorageBuffer { .. } => DescriptorType::StorageBufferDynamic,
            DescriptorWriteInner::InputAttachment(_, _, _) => DescriptorType::InputAttachment,
        }
    }

    // TODO: hacky
    #[doc(hidden)]
    pub fn inner(&self) -> (u32, u32, &DescriptorWriteInner) { (self.binding, self.first_array_element, &self.inner) }
}

/// Describes a single descriptor.
#[derive(Debug, Copy, Clone)]
pub struct DescriptorDesc {
    /// Offset of the binding within the descriptor.
    pub binding: u32,

    /// What kind of resource can later be bound to this descriptor.
    pub ty: DescriptorType,

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
        self.binding == other.binding && self.ty == other.ty &&
        self.array_count >= other.array_count && self.stages.is_superset_of(&other.stages) &&
        (!self.readonly || other.readonly)
    }
}

/// Describes what kind of resource may later be bound to a descriptor.
// FIXME: add immutable sampler when relevant
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

impl DescriptorType {
    /// Turns the `DescriptorType` into the corresponding Vulkan constant.
    // this function exists because when immutable samplers are added, it will no longer be possible to do `as u32`
    // TODO: hacky
    #[inline]
    #[doc(hidden)]
    pub fn vk_enum(&self) -> u32 {
        *self as u32
    }
}

/// Describes which shader stages have access to a descriptor.
#[derive(Debug, Copy, Clone)]
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
