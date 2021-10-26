// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level descriptor set.

use crate::buffer::{BufferAccess, BufferInner, BufferViewAbstract};
use crate::descriptor_set::layout::{DescriptorSetLayout, DescriptorType};
use crate::device::DeviceOwned;
use crate::image::view::ImageViewAbstract;
use crate::sampler::Sampler;
use crate::DeviceSize;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::fmt;
use std::ptr;
use std::sync::Arc;

/// Low-level descriptor set.
///
/// Contrary to most other objects in this library, this one doesn't free itself automatically and
/// doesn't hold the pool or the device it is associated to.
/// Instead it is an object meant to be used with the `UnsafeDescriptorPool`.
pub struct UnsafeDescriptorSet {
    handle: ash::vk::DescriptorSet,
}

impl UnsafeDescriptorSet {
    pub(crate) fn new(handle: ash::vk::DescriptorSet) -> Self {
        Self { handle }
    }

    /// Modifies a descriptor set. Doesn't check that the writes or copies are correct, and
    /// doesn't check whether the descriptor set is in use.
    ///
    /// # Safety
    ///
    /// - The `Device` must be the device the pool of this set was created with.
    /// - Doesn't verify that the things you write in the descriptor set match its layout.
    /// - Doesn't keep the resources alive. You have to do that yourself.
    /// - Updating a descriptor set obeys synchronization rules that aren't checked here. Once a
    ///   command buffer contains a pointer/reference to a descriptor set, it is illegal to write
    ///   to it.
    ///
    pub unsafe fn write<'a>(
        &mut self,
        layout: &DescriptorSetLayout,
        writes: impl IntoIterator<Item = &'a DescriptorWrite>,
    ) {
        let (infos, mut writes): (SmallVec<[_; 8]>, SmallVec<[_; 8]>) = writes
            .into_iter()
            .map(|write| {
                let descriptor_type = layout.descriptor(write.binding_num).unwrap().ty.ty();

                (
                    write.to_vulkan_info(descriptor_type),
                    write.to_vulkan(self.handle, descriptor_type),
                )
            })
            .unzip();

        // It is forbidden to call `vkUpdateDescriptorSets` with 0 writes, so we need to perform
        // this emptiness check.
        if writes.is_empty() {
            return;
        }

        // Set the info pointers separately.
        for (info, write) in infos.iter().zip(writes.iter_mut()) {
            match info {
                DescriptorWriteInfo::Image(info) => {
                    write.descriptor_count = info.len() as u32;
                    write.p_image_info = info.as_ptr();
                }
                DescriptorWriteInfo::Buffer(info) => {
                    write.descriptor_count = info.len() as u32;
                    write.p_buffer_info = info.as_ptr();
                }
                DescriptorWriteInfo::BufferView(info) => {
                    write.descriptor_count = info.len() as u32;
                    write.p_texel_buffer_view = info.as_ptr();
                }
            }

            debug_assert!(write.descriptor_count != 0);
        }

        let fns = layout.device().fns();

        fns.v1_0.update_descriptor_sets(
            layout.device().internal_object(),
            writes.len() as u32,
            writes.as_ptr(),
            0,
            ptr::null(),
        );
    }

    // TODO: add copying from other descriptor sets
    //       add a `copy` method that just takes a copy, and an `update` method that takes both
    //       writes and copies and that actually performs the operation
}

unsafe impl VulkanObject for UnsafeDescriptorSet {
    type Object = ash::vk::DescriptorSet;

    #[inline]
    fn internal_object(&self) -> ash::vk::DescriptorSet {
        self.handle
    }
}

impl fmt::Debug for UnsafeDescriptorSet {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan descriptor set {:?}>", self.handle)
    }
}

/// Represents a single write entry to a descriptor set.
///
/// Use the various constructors to build a `DescriptorWrite`. While it is safe to build a
/// `DescriptorWrite`, it is unsafe to actually use it to write to a descriptor set.
pub struct DescriptorWrite {
    pub(crate) binding_num: u32,
    first_array_element: u32,
    elements: DescriptorWriteElements,
}

impl DescriptorWrite {
    #[inline]
    pub unsafe fn buffer(
        binding_num: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Arc<dyn BufferAccess>>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());
        Self {
            binding_num,
            first_array_element,
            elements: DescriptorWriteElements::Buffer(elements),
        }
    }

    #[inline]
    pub unsafe fn buffer_view(
        binding_num: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Arc<dyn BufferViewAbstract>>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());
        Self {
            binding_num,
            first_array_element,
            elements: DescriptorWriteElements::BufferView(elements),
        }
    }

    #[inline]
    pub unsafe fn image_view(
        binding_num: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Arc<dyn ImageViewAbstract>>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());
        Self {
            binding_num,
            first_array_element,
            elements: DescriptorWriteElements::ImageView(elements),
        }
    }

    #[inline]
    pub unsafe fn image_view_sampler(
        binding_num: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = (Arc<dyn ImageViewAbstract>, Arc<Sampler>)>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());
        Self {
            binding_num,
            first_array_element,
            elements: DescriptorWriteElements::ImageViewSampler(elements),
        }
    }

    #[inline]
    pub unsafe fn sampler(
        binding_num: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Arc<Sampler>>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());
        Self {
            binding_num,
            first_array_element,
            elements: DescriptorWriteElements::Sampler(elements),
        }
    }

    /// Returns the binding number that is updated by this descriptor write.
    #[inline]
    pub fn binding_num(&self) -> u32 {
        self.binding_num
    }

    /// Returns the first array element in the binding that is updated by this descriptor write.
    #[inline]
    pub fn first_array_element(&self) -> u32 {
        self.first_array_element
    }

    /// Returns a reference to the elements held by this descriptor write.
    #[inline]
    pub fn elements(&self) -> &DescriptorWriteElements {
        &self.elements
    }

    pub(crate) fn to_vulkan_info(&self, descriptor_type: DescriptorType) -> DescriptorWriteInfo {
        match &self.elements {
            DescriptorWriteElements::Buffer(elements) => {
                debug_assert!(matches!(
                    descriptor_type,
                    DescriptorType::UniformBuffer
                        | DescriptorType::StorageBuffer
                        | DescriptorType::UniformBufferDynamic
                        | DescriptorType::StorageBufferDynamic
                ));
                DescriptorWriteInfo::Buffer(
                    elements
                        .iter()
                        .map(|buffer| {
                            let size = buffer.size();
                            let BufferInner { buffer, offset } = buffer.inner();

                            debug_assert_eq!(
                                offset
                                    % buffer
                                        .device()
                                        .physical_device()
                                        .properties()
                                        .min_storage_buffer_offset_alignment,
                                0
                            );
                            debug_assert!(
                                size <= buffer
                                    .device()
                                    .physical_device()
                                    .properties()
                                    .max_storage_buffer_range
                                    as DeviceSize
                            );
                            ash::vk::DescriptorBufferInfo {
                                buffer: buffer.internal_object(),
                                offset,
                                range: size,
                            }
                        })
                        .collect(),
                )
            }
            DescriptorWriteElements::BufferView(elements) => {
                debug_assert!(matches!(
                    descriptor_type,
                    DescriptorType::UniformTexelBuffer | DescriptorType::StorageTexelBuffer
                ));
                DescriptorWriteInfo::BufferView(
                    elements
                        .iter()
                        .map(|buffer_view| buffer_view.inner())
                        .collect(),
                )
            }
            DescriptorWriteElements::ImageView(elements) => {
                // Note: combined image sampler can occur with immutable samplers
                debug_assert!(matches!(
                    descriptor_type,
                    DescriptorType::CombinedImageSampler
                        | DescriptorType::SampledImage
                        | DescriptorType::StorageImage
                        | DescriptorType::InputAttachment
                ));
                DescriptorWriteInfo::Image(
                    elements
                        .iter()
                        .map(|image_view| {
                            let layouts = image_view.image().descriptor_layouts().expect(
                                "descriptor_layouts must return Some when used in an image view",
                            );
                            ash::vk::DescriptorImageInfo {
                                sampler: ash::vk::Sampler::null(),
                                image_view: image_view.inner().internal_object(),
                                image_layout: layouts.layout_for(descriptor_type).into(),
                            }
                        })
                        .collect(),
                )
            }
            DescriptorWriteElements::ImageViewSampler(elements) => {
                debug_assert!(matches!(
                    descriptor_type,
                    DescriptorType::CombinedImageSampler
                ));
                DescriptorWriteInfo::Image(
                    elements
                        .iter()
                        .map(|(image_view, sampler)| {
                            let layouts = image_view.image().descriptor_layouts().expect(
                                "descriptor_layouts must return Some when used in an image view",
                            );
                            ash::vk::DescriptorImageInfo {
                                sampler: sampler.internal_object(),
                                image_view: image_view.inner().internal_object(),
                                image_layout: layouts.layout_for(descriptor_type).into(),
                            }
                        })
                        .collect(),
                )
            }
            DescriptorWriteElements::Sampler(elements) => {
                debug_assert!(matches!(descriptor_type, DescriptorType::Sampler));
                DescriptorWriteInfo::Image(
                    elements
                        .iter()
                        .map(|sampler| ash::vk::DescriptorImageInfo {
                            sampler: sampler.internal_object(),
                            image_view: ash::vk::ImageView::null(),
                            image_layout: ash::vk::ImageLayout::UNDEFINED,
                        })
                        .collect(),
                )
            }
        }
    }

    pub(crate) fn to_vulkan(
        &self,
        dst_set: ash::vk::DescriptorSet,
        descriptor_type: DescriptorType,
    ) -> ash::vk::WriteDescriptorSet {
        ash::vk::WriteDescriptorSet {
            dst_set,
            dst_binding: self.binding_num,
            dst_array_element: self.first_array_element,
            descriptor_count: 0,
            descriptor_type: descriptor_type.into(),
            p_image_info: ptr::null(),
            p_buffer_info: ptr::null(),
            p_texel_buffer_view: ptr::null(),
            ..Default::default()
        }
    }
}

/// The elements held by a descriptor write.
pub enum DescriptorWriteElements {
    Buffer(SmallVec<[Arc<dyn BufferAccess>; 1]>),
    BufferView(SmallVec<[Arc<dyn BufferViewAbstract>; 1]>),
    ImageView(SmallVec<[Arc<dyn ImageViewAbstract>; 1]>),
    ImageViewSampler(SmallVec<[(Arc<dyn ImageViewAbstract>, Arc<Sampler>); 1]>),
    Sampler(SmallVec<[Arc<Sampler>; 1]>),
}

impl DescriptorWriteElements {
    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> u32 {
        match self {
            DescriptorWriteElements::Buffer(elements) => elements.len() as u32,
            DescriptorWriteElements::BufferView(elements) => elements.len() as u32,
            DescriptorWriteElements::ImageView(elements) => elements.len() as u32,
            DescriptorWriteElements::ImageViewSampler(elements) => elements.len() as u32,
            DescriptorWriteElements::Sampler(elements) => elements.len() as u32,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum DescriptorWriteInfo {
    Image(SmallVec<[ash::vk::DescriptorImageInfo; 1]>),
    Buffer(SmallVec<[ash::vk::DescriptorBufferInfo; 1]>),
    BufferView(SmallVec<[ash::vk::BufferView; 1]>),
}

impl DescriptorWriteInfo {
    fn set_info(&self, write: &mut ash::vk::WriteDescriptorSet) {
        match self {
            DescriptorWriteInfo::Image(info) => {
                write.descriptor_count = info.len() as u32;
                write.p_image_info = info.as_ptr();
            }
            DescriptorWriteInfo::Buffer(info) => {
                write.descriptor_count = info.len() as u32;
                write.p_buffer_info = info.as_ptr();
            }
            DescriptorWriteInfo::BufferView(info) => {
                write.descriptor_count = info.len() as u32;
                write.p_texel_buffer_view = info.as_ptr();
            }
        }

        debug_assert!(write.descriptor_count != 0);
    }
}
