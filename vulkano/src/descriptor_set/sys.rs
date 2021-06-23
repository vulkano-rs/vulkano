// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::buffer::BufferInner;
use crate::buffer::BufferView;
use crate::descriptor_set::descriptor::DescriptorType;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::image::view::ImageViewAbstract;
use crate::sampler::Sampler;
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
    pub unsafe fn write<I>(&mut self, device: &Device, writes: I)
    where
        I: Iterator<Item = DescriptorWrite>,
    {
        let fns = device.fns();

        // In this function, we build 4 arrays: one array of image descriptors (image_descriptors),
        // one for buffer descriptors (buffer_descriptors), one for buffer view descriptors
        // (buffer_views_descriptors), and one for the final list of writes (raw_writes).
        // Only the final list is passed to Vulkan, but it will contain pointers to the first three
        // lists in `pImageInfo`, `pBufferInfo` and `pTexelBufferView`.
        //
        // In order to handle that, we start by writing null pointers as placeholders in the final
        // writes, and we store in `raw_writes_img_infos`, `raw_writes_buf_infos` and
        // `raw_writes_buf_view_infos` the offsets of the pointers compared to the start of the
        // list.
        // Once we have finished iterating all the writes requested by the user, we modify
        // `raw_writes` to point to the correct locations.

        let mut buffer_descriptors: SmallVec<[_; 64]> = SmallVec::new();
        let mut image_descriptors: SmallVec<[_; 64]> = SmallVec::new();
        let mut buffer_views_descriptors: SmallVec<[_; 64]> = SmallVec::new();

        let mut raw_writes: SmallVec<[_; 64]> = SmallVec::new();
        let mut raw_writes_img_infos: SmallVec<[_; 64]> = SmallVec::new();
        let mut raw_writes_buf_infos: SmallVec<[_; 64]> = SmallVec::new();
        let mut raw_writes_buf_view_infos: SmallVec<[_; 64]> = SmallVec::new();

        for indiv_write in writes {
            // Since the `DescriptorWrite` objects are built only through functions, we know for
            // sure that it's impossible to have an empty descriptor write.
            debug_assert!(!indiv_write.inner.is_empty());

            // The whole struct thats written here is valid, except for pImageInfo, pBufferInfo
            // and pTexelBufferView which are placeholder values.
            raw_writes.push(ash::vk::WriteDescriptorSet {
                dst_set: self.set,
                dst_binding: indiv_write.binding,
                dst_array_element: indiv_write.first_array_element,
                descriptor_count: indiv_write.inner.len() as u32,
                descriptor_type: indiv_write.ty().into(),
                p_image_info: ptr::null(),
                p_buffer_info: ptr::null(),
                p_texel_buffer_view: ptr::null(),
                ..Default::default()
            });

            match indiv_write.inner[0] {
                DescriptorWriteInner::Sampler(_)
                | DescriptorWriteInner::CombinedImageSampler(_, _, _)
                | DescriptorWriteInner::SampledImage(_, _)
                | DescriptorWriteInner::StorageImage(_, _)
                | DescriptorWriteInner::InputAttachment(_, _) => {
                    raw_writes_img_infos.push(Some(image_descriptors.len()));
                    raw_writes_buf_infos.push(None);
                    raw_writes_buf_view_infos.push(None);
                }
                DescriptorWriteInner::UniformBuffer(_, _, _)
                | DescriptorWriteInner::StorageBuffer(_, _, _)
                | DescriptorWriteInner::DynamicUniformBuffer(_, _, _)
                | DescriptorWriteInner::DynamicStorageBuffer(_, _, _) => {
                    raw_writes_img_infos.push(None);
                    raw_writes_buf_infos.push(Some(buffer_descriptors.len()));
                    raw_writes_buf_view_infos.push(None);
                }
                DescriptorWriteInner::UniformTexelBuffer(_)
                | DescriptorWriteInner::StorageTexelBuffer(_) => {
                    raw_writes_img_infos.push(None);
                    raw_writes_buf_infos.push(None);
                    raw_writes_buf_view_infos.push(Some(buffer_views_descriptors.len()));
                }
            }

            for elem in indiv_write.inner.iter() {
                match *elem {
                    DescriptorWriteInner::UniformBuffer(buffer, offset, size)
                    | DescriptorWriteInner::DynamicUniformBuffer(buffer, offset, size) => {
                        buffer_descriptors.push(ash::vk::DescriptorBufferInfo {
                            buffer,
                            offset: offset as u64,
                            range: size as u64,
                        });
                    }
                    DescriptorWriteInner::StorageBuffer(buffer, offset, size)
                    | DescriptorWriteInner::DynamicStorageBuffer(buffer, offset, size) => {
                        buffer_descriptors.push(ash::vk::DescriptorBufferInfo {
                            buffer,
                            offset: offset as u64,
                            range: size as u64,
                        });
                    }
                    DescriptorWriteInner::Sampler(sampler) => {
                        image_descriptors.push(ash::vk::DescriptorImageInfo {
                            sampler,
                            image_view: ash::vk::ImageView::null(),
                            image_layout: ash::vk::ImageLayout::UNDEFINED,
                        });
                    }
                    DescriptorWriteInner::CombinedImageSampler(sampler, view, layout) => {
                        image_descriptors.push(ash::vk::DescriptorImageInfo {
                            sampler,
                            image_view: view,
                            image_layout: layout,
                        });
                    }
                    DescriptorWriteInner::StorageImage(view, layout) => {
                        image_descriptors.push(ash::vk::DescriptorImageInfo {
                            sampler: ash::vk::Sampler::null(),
                            image_view: view,
                            image_layout: layout,
                        });
                    }
                    DescriptorWriteInner::SampledImage(view, layout) => {
                        image_descriptors.push(ash::vk::DescriptorImageInfo {
                            sampler: ash::vk::Sampler::null(),
                            image_view: view,
                            image_layout: layout,
                        });
                    }
                    DescriptorWriteInner::InputAttachment(view, layout) => {
                        image_descriptors.push(ash::vk::DescriptorImageInfo {
                            sampler: ash::vk::Sampler::null(),
                            image_view: view,
                            image_layout: layout,
                        });
                    }
                    DescriptorWriteInner::UniformTexelBuffer(view)
                    | DescriptorWriteInner::StorageTexelBuffer(view) => {
                        buffer_views_descriptors.push(view);
                    }
                }
            }
        }

        // Now that `image_descriptors`, `buffer_descriptors` and `buffer_views_descriptors` are
        // entirely filled and will never move again, we can fill the pointers in `raw_writes`.
        for (i, write) in raw_writes.iter_mut().enumerate() {
            write.p_image_info = match raw_writes_img_infos[i] {
                Some(off) => image_descriptors.as_ptr().offset(off as isize),
                None => ptr::null(),
            };

            write.p_buffer_info = match raw_writes_buf_infos[i] {
                Some(off) => buffer_descriptors.as_ptr().offset(off as isize),
                None => ptr::null(),
            };

            write.p_texel_buffer_view = match raw_writes_buf_view_infos[i] {
                Some(off) => buffer_views_descriptors.as_ptr().offset(off as isize),
                None => ptr::null(),
            };
        }

        // It is forbidden to call `vkUpdateDescriptorSets` with 0 writes, so we need to perform
        // this emptiness check.
        if !raw_writes.is_empty() {
            fns.v1_0.update_descriptor_sets(
                device.internal_object(),
                raw_writes.len() as u32,
                raw_writes.as_ptr(),
                0,
                ptr::null(),
            );
        }
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
// TODO: allow binding whole arrays at once
pub struct DescriptorWrite {
    binding: u32,
    first_array_element: u32,
    inner: SmallVec<[DescriptorWriteInner; 1]>,
}

#[derive(Debug, Clone)]
enum DescriptorWriteInner {
    Sampler(ash::vk::Sampler),
    StorageImage(ash::vk::ImageView, ash::vk::ImageLayout),
    SampledImage(ash::vk::ImageView, ash::vk::ImageLayout),
    CombinedImageSampler(ash::vk::Sampler, ash::vk::ImageView, ash::vk::ImageLayout),
    UniformTexelBuffer(ash::vk::BufferView),
    StorageTexelBuffer(ash::vk::BufferView),
    UniformBuffer(ash::vk::Buffer, usize, usize),
    StorageBuffer(ash::vk::Buffer, usize, usize),
    DynamicUniformBuffer(ash::vk::Buffer, usize, usize),
    DynamicStorageBuffer(ash::vk::Buffer, usize, usize),
    InputAttachment(ash::vk::ImageView, ash::vk::ImageLayout),
}

macro_rules! smallvec {
    ($elem:expr) => {{
        let mut s = SmallVec::new();
        s.push($elem);
        s
    }};
}

impl DescriptorWrite {
    #[inline]
    pub fn storage_image<I>(binding: u32, array_element: u32, image_view: &I) -> DescriptorWrite
    where
        I: ImageViewAbstract,
    {
        let layouts = image_view
            .image()
            .descriptor_layouts()
            .expect("descriptor_layouts must return Some when used in an image view");

        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!({
                DescriptorWriteInner::StorageImage(
                    image_view.inner().internal_object(),
                    layouts.storage_image.into(),
                )
            }),
        }
    }

    #[inline]
    pub fn sampler(binding: u32, array_element: u32, sampler: &Arc<Sampler>) -> DescriptorWrite {
        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!(DescriptorWriteInner::Sampler(sampler.internal_object())),
        }
    }

    #[inline]
    pub fn sampled_image<I>(binding: u32, array_element: u32, image_view: &I) -> DescriptorWrite
    where
        I: ImageViewAbstract,
    {
        let layouts = image_view
            .image()
            .descriptor_layouts()
            .expect("descriptor_layouts must return Some when used in an image view");

        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!({
                DescriptorWriteInner::SampledImage(
                    image_view.inner().internal_object(),
                    layouts.sampled_image.into(),
                )
            }),
        }
    }

    #[inline]
    pub fn combined_image_sampler<I>(
        binding: u32,
        array_element: u32,
        sampler: &Arc<Sampler>,
        image_view: &I,
    ) -> DescriptorWrite
    where
        I: ImageViewAbstract,
    {
        let layouts = image_view
            .image()
            .descriptor_layouts()
            .expect("descriptor_layouts must return Some when used in an image view");

        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!({
                DescriptorWriteInner::CombinedImageSampler(
                    sampler.internal_object(),
                    image_view.inner().internal_object(),
                    layouts.combined_image_sampler.into(),
                )
            }),
        }
    }

    #[inline]
    pub fn uniform_texel_buffer<'a, B>(
        binding: u32,
        array_element: u32,
        view: &BufferView<B>,
    ) -> DescriptorWrite
    where
        B: BufferAccess,
    {
        assert!(view.uniform_texel_buffer());

        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!(DescriptorWriteInner::UniformTexelBuffer(
                view.internal_object()
            )),
        }
    }

    #[inline]
    pub fn storage_texel_buffer<'a, B>(
        binding: u32,
        array_element: u32,
        view: &BufferView<B>,
    ) -> DescriptorWrite
    where
        B: BufferAccess,
    {
        assert!(view.storage_texel_buffer());

        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!(DescriptorWriteInner::StorageTexelBuffer(
                view.internal_object()
            )),
        }
    }

    #[inline]
    pub unsafe fn uniform_buffer<B>(binding: u32, array_element: u32, buffer: &B) -> DescriptorWrite
    where
        B: BufferAccess,
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        debug_assert_eq!(
            offset
                % buffer
                    .device()
                    .physical_device()
                    .properties()
                    .min_uniform_buffer_offset_alignment
                    .unwrap() as usize,
            0
        );
        debug_assert!(
            size <= buffer
                .device()
                .physical_device()
                .properties()
                .max_uniform_buffer_range
                .unwrap() as usize
        );

        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!({
                DescriptorWriteInner::UniformBuffer(buffer.internal_object(), offset, size)
            }),
        }
    }

    #[inline]
    pub unsafe fn storage_buffer<B>(binding: u32, array_element: u32, buffer: &B) -> DescriptorWrite
    where
        B: BufferAccess,
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        debug_assert_eq!(
            offset
                % buffer
                    .device()
                    .physical_device()
                    .properties()
                    .min_storage_buffer_offset_alignment
                    .unwrap() as usize,
            0
        );
        debug_assert!(
            size <= buffer
                .device()
                .physical_device()
                .properties()
                .max_storage_buffer_range
                .unwrap() as usize
        );

        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!({
                DescriptorWriteInner::StorageBuffer(buffer.internal_object(), offset, size)
            }),
        }
    }

    #[inline]
    pub unsafe fn dynamic_uniform_buffer<B>(
        binding: u32,
        array_element: u32,
        buffer: &B,
    ) -> DescriptorWrite
    where
        B: BufferAccess,
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        debug_assert_eq!(
            offset
                % buffer
                    .device()
                    .physical_device()
                    .properties()
                    .min_uniform_buffer_offset_alignment
                    .unwrap() as usize,
            0
        );
        debug_assert!(
            size <= buffer
                .device()
                .physical_device()
                .properties()
                .max_uniform_buffer_range
                .unwrap() as usize
        );

        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!(DescriptorWriteInner::DynamicUniformBuffer(
                buffer.internal_object(),
                offset,
                size
            )),
        }
    }

    #[inline]
    pub unsafe fn dynamic_storage_buffer<B>(
        binding: u32,
        array_element: u32,
        buffer: &B,
    ) -> DescriptorWrite
    where
        B: BufferAccess,
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        debug_assert_eq!(
            offset
                % buffer
                    .device()
                    .physical_device()
                    .properties()
                    .min_storage_buffer_offset_alignment
                    .unwrap() as usize,
            0
        );
        debug_assert!(
            size <= buffer
                .device()
                .physical_device()
                .properties()
                .max_storage_buffer_range
                .unwrap() as usize
        );

        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!(DescriptorWriteInner::DynamicStorageBuffer(
                buffer.internal_object(),
                offset,
                size
            )),
        }
    }

    #[inline]
    pub fn input_attachment<I>(binding: u32, array_element: u32, image_view: &I) -> DescriptorWrite
    where
        I: ImageViewAbstract,
    {
        let layouts = image_view
            .image()
            .descriptor_layouts()
            .expect("descriptor_layouts must return Some when used in an image view");

        DescriptorWrite {
            binding,
            first_array_element: array_element,
            inner: smallvec!({
                DescriptorWriteInner::InputAttachment(
                    image_view.inner().internal_object(),
                    layouts.input_attachment.into(),
                )
            }),
        }
    }

    /// Returns the type corresponding to this write.
    #[inline]
    pub fn ty(&self) -> DescriptorType {
        match self.inner[0] {
            DescriptorWriteInner::Sampler(_) => DescriptorType::Sampler,
            DescriptorWriteInner::CombinedImageSampler(_, _, _) => {
                DescriptorType::CombinedImageSampler
            }
            DescriptorWriteInner::SampledImage(_, _) => DescriptorType::SampledImage,
            DescriptorWriteInner::StorageImage(_, _) => DescriptorType::StorageImage,
            DescriptorWriteInner::UniformTexelBuffer(_) => DescriptorType::UniformTexelBuffer,
            DescriptorWriteInner::StorageTexelBuffer(_) => DescriptorType::StorageTexelBuffer,
            DescriptorWriteInner::UniformBuffer(_, _, _) => DescriptorType::UniformBuffer,
            DescriptorWriteInner::StorageBuffer(_, _, _) => DescriptorType::StorageBuffer,
            DescriptorWriteInner::DynamicUniformBuffer(_, _, _) => {
                DescriptorType::UniformBufferDynamic
            }
            DescriptorWriteInner::DynamicStorageBuffer(_, _, _) => {
                DescriptorType::StorageBufferDynamic
            }
            DescriptorWriteInner::InputAttachment(_, _) => DescriptorType::InputAttachment,
        }
    }
}
