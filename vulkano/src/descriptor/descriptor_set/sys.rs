// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use check_errors;
use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use vk;

use buffer::Buffer;
use descriptor::descriptor::DescriptorWrite;
use descriptor::descriptor::DescriptorWriteInner;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::DescriptorPool;
use device::Device;
use image::Image;
use image::ImageView;
use image::Layout as ImageLayout;
use sampler::Sampler;

/// Low-level descriptor set.
pub struct UnsafeDescriptorSet {
    set: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
    layout: Arc<UnsafeDescriptorSetLayout>,

    // Here we store the resources used by the descriptor set.
    // TODO: for the moment even when a resource is overwritten it stays in these lists
    resources_samplers: Vec<Arc<Sampler>>,
    resources_images: Vec<(Arc<Image>, (u32, u32), ImageLayout)>,
    resources_image_views: Vec<Arc<ImageView>>,
    resources_buffers: Vec<Arc<Buffer>>,
}

impl UnsafeDescriptorSet {
    /// Builds a new descriptor set.
    ///
    /// # Panic
    ///
    /// - Panicks if the pool and the layout were not created from the same `Device`.
    ///
    // FIXME: this has to check whether there's still enough room in the pool
    pub unsafe fn uninitialized(pool: &Arc<DescriptorPool>,
                                layout: &Arc<UnsafeDescriptorSetLayout>)
                                -> Result<UnsafeDescriptorSet, OomError>
    {
        assert_eq!(&**pool.device() as *const Device, &**layout.device() as *const Device);

        let vk = pool.device().pointers();

        let set = {
            let pool_obj = pool.internal_object_guard();

            let infos = vk::DescriptorSetAllocateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                pNext: ptr::null(),
                descriptorPool: *pool_obj,
                descriptorSetCount: 1,
                pSetLayouts: &layout.internal_object(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.AllocateDescriptorSets(pool.device().internal_object(), &infos,
                                                        &mut output)));
            output
        };

        Ok(UnsafeDescriptorSet {
            set: set,
            pool: pool.clone(),
            layout: layout.clone(),

            resources_samplers: Vec::new(),
            resources_images: Vec::new(),
            resources_image_views: Vec::new(),
            resources_buffers: Vec::new(),
        })
    }

    /// Modifies a descriptor set without checking that the writes are correct.
    ///
    /// # Safety
    ///
    /// - Doesn't verify that the things you write in the descriptor set match its layout.
    ///
    pub unsafe fn write(&mut self, write: Vec<DescriptorWrite>) {
        let vk = self.pool.device().pointers();

        // TODO: how do we remove the existing resources that are overwritten?

        // This function uses multiple closures which all borrow `self`. In order to satisfy the
        // borrow checker, we extract references to the members here.
        let ref mut self_resources_buffers = self.resources_buffers;
        let ref mut self_resources_samplers = self.resources_samplers;
        let ref mut self_resources_images = self.resources_images;
        let ref mut self_resources_image_views = self.resources_image_views;
        let self_set = self.set;

        let buffer_descriptors = write.iter().filter_map(|write| {
            match *write.inner().2 {
                DescriptorWriteInner::UniformBuffer { ref buffer, offset, size } |
                DescriptorWriteInner::DynamicUniformBuffer { ref buffer, offset, size } => {
                    assert!(buffer.inner_buffer().usage_uniform_buffer());
                    self_resources_buffers.push(buffer.clone());
                    Some(vk::DescriptorBufferInfo {
                        buffer: buffer.inner_buffer().internal_object(),
                        offset: offset as u64,
                        range: size as u64,
                    })
                },
                DescriptorWriteInner::StorageBuffer { ref buffer, offset, size } |
                DescriptorWriteInner::DynamicStorageBuffer { ref buffer, offset, size } => {
                    assert!(buffer.inner_buffer().usage_storage_buffer());
                    self_resources_buffers.push(buffer.clone());
                    Some(vk::DescriptorBufferInfo {
                        buffer: buffer.inner_buffer().internal_object(),
                        offset: offset as u64,
                        range: size as u64,
                    })
                },
                _ => None
            }
        }).collect::<SmallVec<[_; 64]>>();

        let image_descriptors = write.iter().filter_map(|write| {
            match *write.inner().2 {
                DescriptorWriteInner::Sampler(ref sampler) => {
                    self_resources_samplers.push(sampler.clone());
                    Some(vk::DescriptorImageInfo {
                        sampler: sampler.internal_object(),
                        imageView: 0,
                        imageLayout: 0,
                    })
                },
                DescriptorWriteInner::CombinedImageSampler(ref sampler, ref view, ref image, ref blocks) => {
                    assert!(view.inner_view().usage_sampled());
                    let layout = view.descriptor_set_combined_image_sampler_layout();
                    self_resources_samplers.push(sampler.clone());
                    self_resources_image_views.push(view.clone());
                    for &block in blocks.iter() {
                        self_resources_images.push((image.clone(), block, layout));       // TODO: check for collisions
                    }
                    Some(vk::DescriptorImageInfo {
                        sampler: sampler.internal_object(),
                        imageView: view.inner_view().internal_object(),
                        imageLayout: layout as u32,
                    })
                },
                DescriptorWriteInner::StorageImage(ref view, ref image, ref blocks) => {
                    assert!(view.inner_view().usage_storage());
                    assert!(view.identity_swizzle());
                    let layout = view.descriptor_set_storage_image_layout();
                    self_resources_image_views.push(view.clone());
                    for &block in blocks.iter() {
                        self_resources_images.push((image.clone(), block, layout));       // TODO: check for collisions
                    }
                    Some(vk::DescriptorImageInfo {
                        sampler: 0,
                        imageView: view.inner_view().internal_object(),
                        imageLayout: layout as u32,
                    })
                },
                DescriptorWriteInner::SampledImage(ref view, ref image, ref blocks) => {
                    assert!(view.inner_view().usage_sampled());
                    let layout = view.descriptor_set_sampled_image_layout();
                    self_resources_image_views.push(view.clone());
                    for &block in blocks.iter() {
                        self_resources_images.push((image.clone(), block, layout));       // TODO: check for collisions
                    }
                    Some(vk::DescriptorImageInfo {
                        sampler: 0,
                        imageView: view.inner_view().internal_object(),
                        imageLayout: layout as u32,
                    })
                },
                DescriptorWriteInner::InputAttachment(ref view, ref image, ref blocks) => {
                    assert!(view.inner_view().usage_input_attachment());
                    assert!(view.identity_swizzle());
                    let layout = view.descriptor_set_input_attachment_layout();
                    self_resources_image_views.push(view.clone());
                    for &block in blocks.iter() {
                        self_resources_images.push((image.clone(), block, layout));       // TODO: check for collisions
                    }
                    Some(vk::DescriptorImageInfo {
                        sampler: 0,
                        imageView: view.inner_view().internal_object(),
                        imageLayout: layout as u32,
                    })
                },
                _ => None
            }
        }).collect::<SmallVec<[_; 64]>>();


        let mut next_buffer_desc = 0;
        let mut next_image_desc = 0;

        let vk_writes = write.iter().map(|write| {
            let (buffer_info, image_info) = match *write.inner().2 {
                DescriptorWriteInner::Sampler(_) | DescriptorWriteInner::CombinedImageSampler(_, _, _, _) |
                DescriptorWriteInner::SampledImage(_, _, _) | DescriptorWriteInner::StorageImage(_, _, _) |
                DescriptorWriteInner::InputAttachment(_, _, _) => {
                    let img = image_descriptors.as_ptr().offset(next_image_desc as isize);
                    next_image_desc += 1;
                    (ptr::null(), img)
                },
                //DescriptorWriteInner::UniformTexelBuffer(_) | DescriptorWriteInner::StorageTexelBuffer(_) =>
                DescriptorWriteInner::UniformBuffer { .. } | DescriptorWriteInner::StorageBuffer { .. } |
                DescriptorWriteInner::DynamicUniformBuffer { .. } |
                DescriptorWriteInner::DynamicStorageBuffer { .. } => {
                    let buf = buffer_descriptors.as_ptr().offset(next_buffer_desc as isize);
                    next_buffer_desc += 1;
                    (buf, ptr::null())
                },
            };

            // FIXME: the descriptor set must be synchronized
            vk::WriteDescriptorSet {
                sType: vk::STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: ptr::null(),
                dstSet: self_set,
                dstBinding: write.inner().0,
                dstArrayElement: write.inner().1,
                descriptorCount: 1,
                descriptorType: write.ty() as u32,
                pImageInfo: image_info,
                pBufferInfo: buffer_info,
                pTexelBufferView: ptr::null(),      // TODO:
            }
        }).collect::<SmallVec<[_; 64]>>();

        debug_assert_eq!(next_buffer_desc, buffer_descriptors.len());
        debug_assert_eq!(next_image_desc, image_descriptors.len());

        if !vk_writes.is_empty() {
            vk.UpdateDescriptorSets(self.pool.device().internal_object(),
                                    vk_writes.len() as u32, vk_writes.as_ptr(), 0, ptr::null());
        }
    }

    /// Returns the layout used to create this descriptor set.
    #[inline]
    pub fn layout(&self) -> &Arc<UnsafeDescriptorSetLayout> {
        &self.layout
    }


    // TODO: hacky
    #[doc(hidden)]
    #[inline]
    pub fn images_list(&self) -> &[(Arc<Image>, (u32, u32), ImageLayout)] {
        &self.resources_images
    }

    // TODO: hacky
    #[doc(hidden)]
    #[inline]
    pub fn buffers_list(&self) -> &[Arc<Buffer>] {
        &self.resources_buffers
    }
}

unsafe impl VulkanObject for UnsafeDescriptorSet {
    type Object = vk::DescriptorSet;

    #[inline]
    fn internal_object(&self) -> vk::DescriptorSet {
        self.set
    }
}

impl Drop for UnsafeDescriptorSet {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.pool.device().pointers();
            vk.FreeDescriptorSets(self.pool.device().internal_object(),
                                  *self.pool.internal_object_guard(), 1, &self.set);
        }
    }
}
