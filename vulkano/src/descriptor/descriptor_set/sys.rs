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
use buffer::BufferInner;
use buffer::BufferView;
use descriptor::descriptor::DescriptorType;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::DescriptorPool;
use device::Device;
use image::ImageView;
use sampler::Sampler;

/// Low-level descriptor set.
pub struct UnsafeDescriptorSet {
    set: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
    layout: Arc<UnsafeDescriptorSetLayout>,
}

impl UnsafeDescriptorSet {
    /// See the docs of uninitialized().
    // FIXME: this has to check whether there's still enough room in the pool
    pub unsafe fn uninitialized_raw(pool: &Arc<DescriptorPool>,
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
        })
    }
    
    /// Builds a new descriptor set.
    ///
    /// # Panic
    ///
    /// - Panics if the pool and the layout were not created from the same `Device`.
    /// - Panics if the device or host ran out of memory.
    ///
    // FIXME: this has to check whether there's still enough room in the pool
    #[inline]
    pub unsafe fn uninitialized(pool: &Arc<DescriptorPool>,
                                layout: &Arc<UnsafeDescriptorSetLayout>)
                                -> UnsafeDescriptorSet
    {
        UnsafeDescriptorSet::uninitialized_raw(pool, layout).unwrap()
    }

    // TODO: add copying from other descriptor sets
    //       add a `copy` method that just takes a copy, and an `update` method that takes both
    //       writes and copies and that actually performs the operation

    /// Modifies a descriptor set. Doesn't check that the writes or copies are correct.
    ///
    /// # Safety
    ///
    /// - Doesn't verify that the things you write in the descriptor set match its layout.
    /// - Doesn't keep the resources alive. You have to do that yourself.
    /// - Updating a descriptor set obeys synchronization rules that aren't checked here.
    ///
    pub unsafe fn write<I>(&mut self, writes: I)
        where I: Iterator<Item = DescriptorWrite>
    {
        let vk = self.pool.device().pointers();

        let mut buffer_descriptors: SmallVec<[_; 64]> = SmallVec::new();
        let mut image_descriptors: SmallVec<[_; 64]> = SmallVec::new();
        let mut buffer_views_descriptors: SmallVec<[_; 64]> = SmallVec::new();

        let mut raw_writes: SmallVec<[_; 64]> = SmallVec::new();
        let mut raw_writes_img_infos: SmallVec<[_; 64]> = SmallVec::new();
        let mut raw_writes_buf_infos: SmallVec<[_; 64]> = SmallVec::new();
        let mut raw_writes_buf_view_infos: SmallVec<[_; 64]> = SmallVec::new();

        for indiv_write in writes {
            if indiv_write.inner.is_empty() {
                continue;
            }

            raw_writes.push(vk::WriteDescriptorSet {
                sType: vk::STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: ptr::null(),
                dstSet: self.set,
                dstBinding: indiv_write.binding,
                dstArrayElement: indiv_write.first_array_element,
                descriptorCount: indiv_write.inner.len() as u32,
                descriptorType: indiv_write.ty() as u32,
                pImageInfo: ptr::null(),
                pBufferInfo: ptr::null(),
                pTexelBufferView: ptr::null(),
            });

            match indiv_write.inner[0] {
                DescriptorWriteInner::Sampler(_) |
                DescriptorWriteInner::CombinedImageSampler(_, _, _) |
                DescriptorWriteInner::SampledImage(_, _) |
                DescriptorWriteInner::StorageImage(_, _) |
                DescriptorWriteInner::InputAttachment(_, _) => {
                    raw_writes_img_infos.push(Some(image_descriptors.len()));
                    raw_writes_buf_infos.push(None);
                    raw_writes_buf_view_infos.push(None);
                },
                DescriptorWriteInner::UniformBuffer(_, _, _) |
                DescriptorWriteInner::StorageBuffer(_, _, _) |
                DescriptorWriteInner::DynamicUniformBuffer(_, _, _) |
                DescriptorWriteInner::DynamicStorageBuffer(_, _, _) => {
                    raw_writes_img_infos.push(None);
                    raw_writes_buf_infos.push(Some(buffer_descriptors.len()));
                    raw_writes_buf_view_infos.push(None);
                },
                DescriptorWriteInner::UniformTexelBuffer(_) |
                DescriptorWriteInner::StorageTexelBuffer(_) => {
                    raw_writes_img_infos.push(None);
                    raw_writes_buf_infos.push(None);
                    raw_writes_buf_view_infos.push(Some(buffer_views_descriptors.len()));
                },
            }

            for elem in indiv_write.inner.iter() {
                match *elem {
                    DescriptorWriteInner::UniformBuffer(buffer, offset, size) |
                    DescriptorWriteInner::DynamicUniformBuffer(buffer, offset, size) => {
                        buffer_descriptors.push(vk::DescriptorBufferInfo {
                            buffer: buffer,
                            offset: offset as u64,
                            range: size as u64,
                        });
                    },
                    DescriptorWriteInner::StorageBuffer(buffer, offset, size) |
                    DescriptorWriteInner::DynamicStorageBuffer(buffer, offset, size) => {
                        buffer_descriptors.push(vk::DescriptorBufferInfo {
                            buffer: buffer,
                            offset: offset as u64,
                            range: size as u64,
                        });
                    },
                    DescriptorWriteInner::Sampler(sampler) => {
                        image_descriptors.push(vk::DescriptorImageInfo {
                            sampler: sampler,
                            imageView: 0,
                            imageLayout: 0,
                        });
                    },
                    DescriptorWriteInner::CombinedImageSampler(sampler, view, layout) => {
                        image_descriptors.push(vk::DescriptorImageInfo {
                            sampler: sampler,
                            imageView: view,
                            imageLayout: layout,
                        });
                    },
                    DescriptorWriteInner::StorageImage(view, layout) => {
                        image_descriptors.push(vk::DescriptorImageInfo {
                            sampler: 0,
                            imageView: view,
                            imageLayout: layout,
                        });
                    },
                    DescriptorWriteInner::SampledImage(view, layout) => {
                        image_descriptors.push(vk::DescriptorImageInfo {
                            sampler: 0,
                            imageView: view,
                            imageLayout: layout,
                        });
                    },
                    DescriptorWriteInner::InputAttachment(view, layout) => {
                        image_descriptors.push(vk::DescriptorImageInfo {
                            sampler: 0,
                            imageView: view,
                            imageLayout: layout,
                        });
                    },
                    DescriptorWriteInner::UniformTexelBuffer(view) |
                    DescriptorWriteInner::StorageTexelBuffer(view) => {
                        buffer_views_descriptors.push(view);
                    },
                }
            }
        }

        for (i, write) in raw_writes.iter_mut().enumerate() {
            write.pImageInfo = match raw_writes_img_infos[i] {
                Some(off) => image_descriptors.as_ptr().offset(off as isize),
                None => ptr::null()
            };
            
            write.pBufferInfo = match raw_writes_buf_infos[i] {
                Some(off) => buffer_descriptors.as_ptr().offset(off as isize),
                None => ptr::null()
            };

            write.pTexelBufferView = match raw_writes_buf_view_infos[i] {
                Some(off) => buffer_views_descriptors.as_ptr().offset(off as isize),
                None => ptr::null()
            };
        }

        if !raw_writes.is_empty() {
            vk.UpdateDescriptorSets(self.pool.device().internal_object(),
                                    raw_writes.len() as u32, raw_writes.as_ptr(), 0, ptr::null());
        }
    }

    /// Returns the layout used to create this descriptor set.
    #[inline]
    pub fn layout(&self) -> &Arc<UnsafeDescriptorSetLayout> {
        &self.layout
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

/// Represents a single write entry to a descriptor set.
///
/// Use the various constructors to build a `DescriptorWrite`. While it is not unsafe to build a
/// `DescriptorWrite`, it is unsafe to actually use it to write to a descriptor set. 
pub struct DescriptorWrite {
    binding: u32,
    first_array_element: u32,
    inner: SmallVec<[DescriptorWriteInner; 1]>,
}

#[derive(Debug, Clone)]
enum DescriptorWriteInner {
    Sampler(vk::Sampler),
    StorageImage(vk::ImageView, vk::ImageLayout),
    SampledImage(vk::ImageView, vk::ImageLayout),
    CombinedImageSampler(vk::Sampler, vk::ImageView, vk::ImageLayout),
    UniformTexelBuffer(vk::BufferView),
    StorageTexelBuffer(vk::BufferView),
    UniformBuffer(vk::Buffer, usize, usize),
    StorageBuffer(vk::Buffer, usize, usize),
    DynamicUniformBuffer(vk::Buffer, usize, usize),
    DynamicStorageBuffer(vk::Buffer, usize, usize),
    InputAttachment(vk::ImageView, vk::ImageLayout),
}

macro_rules! smallvec {
    ($elem:expr) => ({ let mut s = SmallVec::new(); s.push($elem); s });
}

impl DescriptorWrite {
    #[inline]
    pub fn storage_image<I>(binding: u32, image: &I) -> DescriptorWrite
        where I: ImageView
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!({
                let layout = image.descriptor_set_storage_image_layout() as u32;
                DescriptorWriteInner::StorageImage(image.inner().internal_object(), layout)
            }),
        }
    }

    #[inline]
    pub fn sampler(binding: u32, sampler: &Arc<Sampler>) -> DescriptorWrite {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!(DescriptorWriteInner::Sampler(sampler.internal_object()))
        }
    }

    #[inline]
    pub fn sampled_image<I>(binding: u32, image: &I) -> DescriptorWrite
        where I: ImageView
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!({
                let layout = image.descriptor_set_sampled_image_layout() as u32;
                DescriptorWriteInner::SampledImage(image.inner().internal_object(), layout)
            }),
        }
    }

    #[inline]
    pub fn combined_image_sampler<I>(binding: u32, sampler: &Arc<Sampler>, image: &I) -> DescriptorWrite
        where I: ImageView
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!({
                let layout = image.descriptor_set_combined_image_sampler_layout() as u32;
                DescriptorWriteInner::CombinedImageSampler(sampler.internal_object(), image.inner().internal_object(), layout)
            }),
        }
    }

    #[inline]
    pub fn uniform_texel_buffer<'a, F, B>(binding: u32, view: &Arc<BufferView<F, B>>) -> DescriptorWrite
        where B: Buffer,
              F: 'static + Send + Sync,
    {
        assert!(view.uniform_texel_buffer());

        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!(DescriptorWriteInner::UniformTexelBuffer(view.internal_object())),
        }
    }

    #[inline]
    pub fn storage_texel_buffer<'a, F, B>(binding: u32, view: &Arc<BufferView<F, B>>) -> DescriptorWrite
        where B: Buffer + 'static,
              F: 'static + Send + Sync,
    {
        assert!(view.storage_texel_buffer());

        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!(DescriptorWriteInner::StorageTexelBuffer(view.internal_object())),
        }
    }

    #[inline]
    pub unsafe fn uniform_buffer<B>(binding: u32, buffer: &B) -> DescriptorWrite
        where B: Buffer
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!({
                DescriptorWriteInner::UniformBuffer(buffer.internal_object(), offset, size)
            }),
        }
    }

    #[inline]
    pub unsafe fn storage_buffer<B>(binding: u32, buffer: &B) -> DescriptorWrite
        where B: Buffer
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!({
                DescriptorWriteInner::StorageBuffer(buffer.internal_object(), offset, size)
            }),
        }
    }

    #[inline]
    pub unsafe fn dynamic_uniform_buffer<B>(binding: u32, buffer: &B) -> DescriptorWrite
        where B: Buffer
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!(DescriptorWriteInner::DynamicUniformBuffer(buffer.internal_object(),
                                                                        offset, size)),
        }
    }

    #[inline]
    pub unsafe fn dynamic_storage_buffer<B>(binding: u32, buffer: &B) -> DescriptorWrite
        where B: Buffer
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!(DescriptorWriteInner::DynamicStorageBuffer(buffer.internal_object(),
                                                                        offset, size)),
        }
    }

    #[inline]
    pub fn input_attachment<I>(binding: u32, image: &I) -> DescriptorWrite
        where I: ImageView
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: 0,
            inner: smallvec!({
                let layout = image.descriptor_set_input_attachment_layout() as u32;
                DescriptorWriteInner::InputAttachment(image.inner().internal_object(), layout)
            }),
        }
    }

    /// Returns the type corresponding to this write.
    #[inline]
    pub fn ty(&self) -> DescriptorType {
        match self.inner[0] {
            DescriptorWriteInner::Sampler(_) => DescriptorType::Sampler,
            DescriptorWriteInner::CombinedImageSampler(_, _, _) => DescriptorType::CombinedImageSampler,
            DescriptorWriteInner::SampledImage(_, _) => DescriptorType::SampledImage,
            DescriptorWriteInner::StorageImage(_, _) => DescriptorType::StorageImage,
            DescriptorWriteInner::UniformTexelBuffer(_) => DescriptorType::UniformTexelBuffer,
            DescriptorWriteInner::StorageTexelBuffer(_) => DescriptorType::StorageTexelBuffer,
            DescriptorWriteInner::UniformBuffer(_, _, _) => DescriptorType::UniformBuffer,
            DescriptorWriteInner::StorageBuffer(_, _, _) => DescriptorType::StorageBuffer,
            DescriptorWriteInner::DynamicUniformBuffer(_, _, _) => DescriptorType::UniformBufferDynamic,
            DescriptorWriteInner::DynamicStorageBuffer(_, _, _) => DescriptorType::StorageBufferDynamic,
            DescriptorWriteInner::InputAttachment(_, _) => DescriptorType::InputAttachment,
        }
    }
}
