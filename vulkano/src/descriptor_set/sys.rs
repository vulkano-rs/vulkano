// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level descriptor set.

use crate::buffer::BufferAccess;
use crate::buffer::BufferInner;
use crate::buffer::BufferView;
use crate::descriptor_set::layout::DescriptorType;
use crate::device::Device;
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
    pub(super) set: ash::vk::DescriptorSet,
}

impl UnsafeDescriptorSet {
    // TODO: add copying from other descriptor sets
    //       add a `copy` method that just takes a copy, and an `update` method that takes both
    //       writes and copies and that actually performs the operation

    /// Modifies a descriptor set. Doesn't check that the writes or copies are correct, and
    /// doesn't check whether the descriptor set is in use.
    ///
    /// **Important**: You must ensure that the `DescriptorSetLayout` object is alive before
    /// updating a descriptor set.
    ///
    /// # Safety
    ///
    /// - The `Device` must be the device the pool of this set was created with.
    /// - The `DescriptorSetLayout` object this set was created with must be alive.
    /// - Doesn't verify that the things you write in the descriptor set match its layout.
    /// - Doesn't keep the resources alive. You have to do that yourself.
    /// - Updating a descriptor set obeys synchronization rules that aren't checked here. Once a
    ///   command buffer contains a pointer/reference to a descriptor set, it is illegal to write
    ///   to it.
    ///
    pub unsafe fn write(&mut self, device: &Device, writes: &[DescriptorWrite]) {
        let fns = device.fns();

        let raw_writes: SmallVec<[_; 8]> = writes
            .iter()
            .map(|write| write.to_vulkan(self.set))
            .collect();

        // It is forbidden to call `vkUpdateDescriptorSets` with 0 writes, so we need to perform
        // this emptiness check.
        if raw_writes.is_empty() {
            return;
        }

        fns.v1_0.update_descriptor_sets(
            device.internal_object(),
            raw_writes.len() as u32,
            raw_writes.as_ptr(),
            0,
            ptr::null(),
        );
    }
}

unsafe impl VulkanObject for UnsafeDescriptorSet {
    type Object = ash::vk::DescriptorSet;

    #[inline]
    fn internal_object(&self) -> ash::vk::DescriptorSet {
        self.set
    }
}

impl fmt::Debug for UnsafeDescriptorSet {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan descriptor set {:?}>", self.set)
    }
}

/// Represents a single write entry to a descriptor set.
///
/// Use the various constructors to build a `DescriptorWrite`. While it is safe to build a
/// `DescriptorWrite`, it is unsafe to actually use it to write to a descriptor set.
pub struct DescriptorWrite {
    binding: u32,
    first_array_element: u32,
    descriptor_type: DescriptorType,
    info: DescriptorWriteInfo,
}

#[derive(Clone, Debug)]
enum DescriptorWriteInfo {
    Image(SmallVec<[ash::vk::DescriptorImageInfo; 1]>),
    Buffer(SmallVec<[ash::vk::DescriptorBufferInfo; 1]>),
    BufferView(SmallVec<[ash::vk::BufferView; 1]>),
}

impl DescriptorWrite {
    #[inline]
    pub fn storage_image<'a, I>(
        binding: u32,
        first_array_element: u32,
        image_views: impl IntoIterator<Item = &'a I>,
    ) -> DescriptorWrite
    where
        I: ImageViewAbstract + 'a,
    {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::StorageImage,
            info: DescriptorWriteInfo::Image(
                image_views
                    .into_iter()
                    .map(|image_view| {
                        let layouts = image_view.image().descriptor_layouts().expect(
                            "descriptor_layouts must return Some when used in an image view",
                        );
                        ash::vk::DescriptorImageInfo {
                            sampler: ash::vk::Sampler::null(),
                            image_view: image_view.inner().internal_object(),
                            image_layout: layouts.storage_image.into(),
                        }
                    })
                    .collect(),
            ),
        }
    }

    #[inline]
    pub fn sampler<'a>(
        binding: u32,
        first_array_element: u32,
        samplers: impl IntoIterator<Item = &'a Arc<Sampler>>,
    ) -> DescriptorWrite {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::Sampler,
            info: DescriptorWriteInfo::Image(
                samplers
                    .into_iter()
                    .map(|sampler| ash::vk::DescriptorImageInfo {
                        sampler: sampler.internal_object(),
                        image_view: ash::vk::ImageView::null(),
                        image_layout: ash::vk::ImageLayout::UNDEFINED,
                    })
                    .collect(),
            ),
        }
    }

    #[inline]
    pub fn sampled_image<'a, I>(
        binding: u32,
        first_array_element: u32,
        image_views: impl IntoIterator<Item = &'a I>,
    ) -> DescriptorWrite
    where
        I: ImageViewAbstract + 'a,
    {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::SampledImage,
            info: DescriptorWriteInfo::Image(
                image_views
                    .into_iter()
                    .map(|image_view| {
                        let layouts = image_view.image().descriptor_layouts().expect(
                            "descriptor_layouts must return Some when used in an image view",
                        );
                        ash::vk::DescriptorImageInfo {
                            sampler: ash::vk::Sampler::null(),
                            image_view: image_view.inner().internal_object(),
                            image_layout: layouts.sampled_image.into(),
                        }
                    })
                    .collect(),
            ),
        }
    }

    #[inline]
    pub fn combined_image_sampler<'a, I>(
        binding: u32,
        first_array_element: u32,
        image_views_samplers: impl IntoIterator<Item = (Option<&'a Arc<Sampler>>, &'a I)>, // Some for dynamic sampler, None for immutable
    ) -> DescriptorWrite
    where
        I: ImageViewAbstract + 'a,
    {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::CombinedImageSampler,
            info: DescriptorWriteInfo::Image(
                image_views_samplers
                    .into_iter()
                    .map(|(sampler, image_view)| {
                        let layouts = image_view.image().descriptor_layouts().expect(
                            "descriptor_layouts must return Some when used in an image view",
                        );
                        ash::vk::DescriptorImageInfo {
                            sampler: sampler.map(|s| s.internal_object()).unwrap_or_default(),
                            image_view: image_view.inner().internal_object(),
                            image_layout: layouts.combined_image_sampler.into(),
                        }
                    })
                    .collect(),
            ),
        }
    }

    #[inline]
    pub fn uniform_texel_buffer<'a, B>(
        binding: u32,
        first_array_element: u32,
        buffer_views: impl IntoIterator<Item = &'a BufferView<B>>,
    ) -> DescriptorWrite
    where
        B: BufferAccess + 'a,
    {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::UniformTexelBuffer,
            info: DescriptorWriteInfo::BufferView(
                buffer_views
                    .into_iter()
                    .map(|buffer_view| {
                        assert!(buffer_view.uniform_texel_buffer());
                        buffer_view.internal_object()
                    })
                    .collect(),
            ),
        }
    }

    #[inline]
    pub fn storage_texel_buffer<'a, B>(
        binding: u32,
        first_array_element: u32,
        buffer_view: impl IntoIterator<Item = &'a BufferView<B>>,
    ) -> DescriptorWrite
    where
        B: BufferAccess + 'a,
    {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::StorageTexelBuffer,
            info: DescriptorWriteInfo::BufferView(
                buffer_view
                    .into_iter()
                    .map(|buffer_view| {
                        assert!(buffer_view.storage_texel_buffer());
                        buffer_view.internal_object()
                    })
                    .collect(),
            ),
        }
    }

    #[inline]
    pub unsafe fn uniform_buffer<'a, B>(
        binding: u32,
        first_array_element: u32,
        buffers: impl IntoIterator<Item = &'a B>,
    ) -> DescriptorWrite
    where
        B: BufferAccess + 'a,
    {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::UniformBuffer,
            info: DescriptorWriteInfo::Buffer(
                buffers
                    .into_iter()
                    .map(|buffer| {
                        let size = buffer.size();
                        let BufferInner { buffer, offset } = buffer.inner();

                        debug_assert_eq!(
                            offset
                                % buffer
                                    .device()
                                    .physical_device()
                                    .properties()
                                    .min_uniform_buffer_offset_alignment,
                            0
                        );
                        debug_assert!(
                            size <= buffer
                                .device()
                                .physical_device()
                                .properties()
                                .max_uniform_buffer_range
                                as DeviceSize
                        );
                        ash::vk::DescriptorBufferInfo {
                            buffer: buffer.internal_object(),
                            offset,
                            range: size,
                        }
                    })
                    .collect(),
            ),
        }
    }

    #[inline]
    pub unsafe fn storage_buffer<'a, B>(
        binding: u32,
        first_array_element: u32,
        buffers: impl IntoIterator<Item = &'a B>,
    ) -> DescriptorWrite
    where
        B: BufferAccess + 'a,
    {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::StorageBuffer,
            info: DescriptorWriteInfo::Buffer(
                buffers
                    .into_iter()
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
            ),
        }
    }

    #[inline]
    pub unsafe fn dynamic_uniform_buffer<'a, B>(
        binding: u32,
        first_array_element: u32,
        buffers: impl IntoIterator<Item = &'a B>,
    ) -> DescriptorWrite
    where
        B: BufferAccess + 'a,
    {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::UniformBufferDynamic,
            info: DescriptorWriteInfo::Buffer(
                buffers
                    .into_iter()
                    .map(|buffer| {
                        let size = buffer.size();
                        let BufferInner { buffer, offset } = buffer.inner();

                        debug_assert_eq!(
                            offset
                                % buffer
                                    .device()
                                    .physical_device()
                                    .properties()
                                    .min_uniform_buffer_offset_alignment,
                            0
                        );
                        debug_assert!(
                            size <= buffer
                                .device()
                                .physical_device()
                                .properties()
                                .max_uniform_buffer_range
                                as DeviceSize
                        );
                        ash::vk::DescriptorBufferInfo {
                            buffer: buffer.internal_object(),
                            offset,
                            range: size,
                        }
                    })
                    .collect(),
            ),
        }
    }

    #[inline]
    pub unsafe fn dynamic_storage_buffer<'a, B>(
        binding: u32,
        first_array_element: u32,
        buffers: impl IntoIterator<Item = &'a B>,
    ) -> DescriptorWrite
    where
        B: BufferAccess + 'a,
    {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::StorageBufferDynamic,
            info: DescriptorWriteInfo::Buffer(
                buffers
                    .into_iter()
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
            ),
        }
    }

    #[inline]
    pub fn input_attachment<'a, I>(
        binding: u32,
        first_array_element: u32,
        image_views: impl IntoIterator<Item = &'a I>,
    ) -> DescriptorWrite
    where
        I: ImageViewAbstract + 'a,
    {
        DescriptorWrite {
            binding,
            first_array_element,
            descriptor_type: DescriptorType::InputAttachment,
            info: DescriptorWriteInfo::Image(
                image_views
                    .into_iter()
                    .map(|image_view| {
                        let layouts = image_view.image().descriptor_layouts().expect(
                            "descriptor_layouts must return Some when used in an image view",
                        );
                        ash::vk::DescriptorImageInfo {
                            sampler: ash::vk::Sampler::null(),
                            image_view: image_view.inner().internal_object(),
                            image_layout: layouts.input_attachment.into(),
                        }
                    })
                    .collect(),
            ),
        }
    }

    pub(crate) fn to_vulkan(&self, dst_set: ash::vk::DescriptorSet) -> ash::vk::WriteDescriptorSet {
        let mut result = ash::vk::WriteDescriptorSet {
            dst_set,
            dst_binding: self.binding,
            dst_array_element: self.first_array_element,
            descriptor_count: 0,
            descriptor_type: self.descriptor_type.into(),
            p_image_info: ptr::null(),
            p_buffer_info: ptr::null(),
            p_texel_buffer_view: ptr::null(),
            ..Default::default()
        };

        // Set the pointers separately.
        // You must keep `*self` alive and unmoved until the function call is done.
        match &self.info {
            DescriptorWriteInfo::Image(info) => {
                result.descriptor_count = info.len() as u32;
                result.p_image_info = info.as_ptr();
            }
            DescriptorWriteInfo::Buffer(info) => {
                result.descriptor_count = info.len() as u32;
                result.p_buffer_info = info.as_ptr();
            }
            DescriptorWriteInfo::BufferView(info) => {
                result.descriptor_count = info.len() as u32;
                result.p_texel_buffer_view = info.as_ptr();
            }
        }

        // Since the `DescriptorWrite` objects are built only through functions, we know for
        // sure that it's impossible to have an empty descriptor write.
        debug_assert!(result.descriptor_count != 0);
        result
    }
}
