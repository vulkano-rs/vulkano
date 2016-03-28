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

use buffer::Buffer;
use descriptor_set::layout_def::Layout;
use descriptor_set::layout_def::SetLayout;
use descriptor_set::layout_def::SetLayoutWrite;
use descriptor_set::layout_def::SetLayoutInit;
use descriptor_set::layout_def::DescriptorWrite;
use descriptor_set::layout_def::DescriptorBindInner;
use descriptor_set::pool::DescriptorPool;
use device::Device;
use image::sys::Layout as ImageLayout;
use image::traits::Image;
use image::traits::ImageView;
use sampler::Sampler;

use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// An actual descriptor set with the resources that are binded to it.
pub struct DescriptorSet<S> {
    set: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
    layout: Arc<DescriptorSetLayout<S>>,

    // Here we store the resources used by the descriptor set.
    // TODO: for the moment even when a resource is overwritten it stays in these lists
    resources_samplers: Vec<Arc<Sampler>>,
    resources_images: Vec<(Arc<Image>, (u32, u32), ImageLayout)>,
    resources_image_views: Vec<Arc<ImageView>>,
    resources_buffers: Vec<Arc<Buffer>>,
}

impl<S> DescriptorSet<S> where S: SetLayout {
    ///
    /// # Panic
    ///
    /// - Panicks if the pool and the layout were not created from the same `Device`.
    ///
    pub fn new<I>(pool: &Arc<DescriptorPool>, layout: &Arc<DescriptorSetLayout<S>>, init: I)
                  -> Result<Arc<DescriptorSet<S>>, OomError>
        where S: SetLayoutInit<I>
    {
        unsafe {
            let mut set = try!(DescriptorSet::uninitialized(pool, layout));
            Arc::get_mut(&mut set).unwrap().unchecked_write(layout.layout().decode(init));
            Ok(set)
        }
    }

    ///
    /// # Panic
    ///
    /// - Panicks if the pool and the layout were not created from the same `Device`.
    ///
    // FIXME: this has to check whether there's still enough room in the pool
    pub unsafe fn uninitialized(pool: &Arc<DescriptorPool>, layout: &Arc<DescriptorSetLayout<S>>)
                                -> Result<Arc<DescriptorSet<S>>, OomError>
    {
        assert_eq!(&**pool.device() as *const Device, &*layout.device as *const Device);

        let vk = pool.device().pointers();

        let set = {
            let pool_obj = pool.internal_object_guard();

            let infos = vk::DescriptorSetAllocateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                pNext: ptr::null(),
                descriptorPool: *pool_obj,
                descriptorSetCount: 1,
                pSetLayouts: &layout.layout,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.AllocateDescriptorSets(pool.device().internal_object(), &infos,
                                                        &mut output)));
            output
        };

        Ok(Arc::new(DescriptorSet {
            set: set,
            pool: pool.clone(),
            layout: layout.clone(),

            resources_samplers: Vec::new(),
            resources_images: Vec::new(),
            resources_image_views: Vec::new(),
            resources_buffers: Vec::new(),
        }))
    }

    /// Modifies a descriptor set.
    ///
    /// The parameter depends on your implementation of `SetLayout`.
    ///
    /// This function trusts the implementation of `SetLayout` when it comes to making sure
    /// that the correct resource type is written to the correct descriptor.
    pub fn write<W>(&mut self, write: W)
        where S: SetLayoutWrite<W>
    {
        let write = self.layout.layout().decode(write);
        unsafe { self.unchecked_write(write); }
    }

    /// Modifies a descriptor set without checking that the writes are correct.
    pub unsafe fn unchecked_write(&mut self, write: Vec<DescriptorWrite>) {
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
            match *write.content.inner() {
                DescriptorBindInner::UniformBuffer { ref buffer, offset, size } |
                DescriptorBindInner::DynamicUniformBuffer { ref buffer, offset, size } => {
                    assert!(buffer.inner_buffer().usage_uniform_buffer());
                    self_resources_buffers.push(buffer.clone());
                    Some(vk::DescriptorBufferInfo {
                        buffer: buffer.inner_buffer().internal_object(),
                        offset: offset as u64,
                        range: size as u64,
                    })
                },
                DescriptorBindInner::StorageBuffer { ref buffer, offset, size } |
                DescriptorBindInner::DynamicStorageBuffer { ref buffer, offset, size } => {
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
            match *write.content.inner() {
                DescriptorBindInner::Sampler(ref sampler) => {
                    self_resources_samplers.push(sampler.clone());
                    Some(vk::DescriptorImageInfo {
                        sampler: sampler.internal_object(),
                        imageView: 0,
                        imageLayout: 0,
                    })
                },
                DescriptorBindInner::CombinedImageSampler(ref sampler, ref view, ref image, ref blocks) => {
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
                DescriptorBindInner::StorageImage(ref view, ref image, ref blocks) => {
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
                DescriptorBindInner::SampledImage(ref view, ref image, ref blocks) => {
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
                DescriptorBindInner::InputAttachment(ref view, ref image, ref blocks) => {
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
            let (buffer_info, image_info) = match *write.content.inner() {
                DescriptorBindInner::Sampler(_) | DescriptorBindInner::CombinedImageSampler(_, _, _, _) |
                DescriptorBindInner::SampledImage(_, _, _) | DescriptorBindInner::StorageImage(_, _, _) |
                DescriptorBindInner::InputAttachment(_, _, _) => {
                    let img = image_descriptors.as_ptr().offset(next_image_desc as isize);
                    next_image_desc += 1;
                    (ptr::null(), img)
                },
                //DescriptorBindInner::UniformTexelBuffer(_) | DescriptorBindInner::StorageTexelBuffer(_) =>
                DescriptorBindInner::UniformBuffer { .. } | DescriptorBindInner::StorageBuffer { .. } |
                DescriptorBindInner::DynamicUniformBuffer { .. } |
                DescriptorBindInner::DynamicStorageBuffer { .. } => {
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
                dstBinding: write.binding,
                dstArrayElement: write.array_element,
                descriptorCount: 1,
                descriptorType: write.content.ty() as u32,
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
}

unsafe impl<S> VulkanObject for DescriptorSet<S> {
    type Object = vk::DescriptorSet;

    #[inline]
    fn internal_object(&self) -> vk::DescriptorSet {
        self.set
    }
}

impl<S> Drop for DescriptorSet<S> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.pool.device().pointers();
            vk.FreeDescriptorSets(self.pool.device().internal_object(),
                                  *self.pool.internal_object_guard(), 1, &self.set);
        }
    }
}


/// Implemented on all `DescriptorSet` objects. Hides the template parameters.
pub unsafe trait AbstractDescriptorSet: ::VulkanObjectU64 {}
unsafe impl<S> AbstractDescriptorSet for DescriptorSet<S> {}

/// Describes the layout of all descriptors within a descriptor set.
pub struct DescriptorSetLayout<S> {
    layout: vk::DescriptorSetLayout,
    device: Arc<Device>,
    description: S,
}

impl<S> DescriptorSetLayout<S> where S: SetLayout {
    pub fn new(device: &Arc<Device>, description: S)
               -> Result<Arc<DescriptorSetLayout<S>>, OomError>
    {
        let vk = device.pointers();

        let bindings = description.descriptors().into_iter().map(|desc| {
            vk::DescriptorSetLayoutBinding {
                binding: desc.binding,
                descriptorType: desc.ty.vk_enum(),
                descriptorCount: desc.array_count,
                stageFlags: desc.stages.into(),
                pImmutableSamplers: ptr::null(),        // FIXME: not yet implemented
            }
        }).collect::<SmallVec<[_; 64]>>();

        let layout = unsafe {
            let infos = vk::DescriptorSetLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                bindingCount: bindings.len() as u32,
                pBindings: bindings.as_ptr(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateDescriptorSetLayout(device.internal_object(), &infos,
                                                           ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(DescriptorSetLayout {
            layout: layout,
            device: device.clone(),
            description: description,
        }))
    }

    #[inline]
    pub fn layout(&self) -> &S {
        &self.description
    }
}

unsafe impl<S> VulkanObject for DescriptorSetLayout<S> {
    type Object = vk::DescriptorSetLayout;

    #[inline]
    fn internal_object(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl<S> Drop for DescriptorSetLayout<S> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyDescriptorSetLayout(self.device.internal_object(), self.layout, ptr::null());
        }
    }
}

/// Implemented on all `DescriptorSetLayout` objects. Hides the template parameters.
pub unsafe trait AbstractDescriptorSetLayout: ::VulkanObjectU64 {}
unsafe impl<S> AbstractDescriptorSetLayout for DescriptorSetLayout<S> {}

/// A collection of `DescriptorSetLayout` structs.
// TODO: push constants.
pub struct PipelineLayout<P> {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
    description: P,
    layouts: Vec<Arc<AbstractDescriptorSetLayout>>,     // TODO: is it necessary to keep the layouts alive? check the specs
}

impl<P> PipelineLayout<P> where P: Layout {
    /// Creates a new `PipelineLayout`.
    pub fn new(device: &Arc<Device>, description: P, layouts: P::DescriptorSetLayouts)
               -> Result<Arc<PipelineLayout<P>>, OomError>
    {
        let vk = device.pointers();

        let layouts = description.decode_descriptor_set_layouts(layouts);
        let layouts_ids = layouts.iter().map(|l| {
            // FIXME: check that they belong to the same device
            ::VulkanObjectU64::internal_object(&**l)
        }).collect::<SmallVec<[_; 32]>>();

        let layout = unsafe {
            let infos = vk::PipelineLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                setLayoutCount: layouts_ids.len() as u32,
                pSetLayouts: layouts_ids.as_ptr(),
                pushConstantRangeCount: 0,      // TODO: unimplemented
                pPushConstantRanges: ptr::null(),    // TODO: unimplemented
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreatePipelineLayout(device.internal_object(), &infos,
                                                      ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(PipelineLayout {
            device: device.clone(),
            layout: layout,
            description: description,
            layouts: layouts,
        }))
    }

    #[inline]
    pub fn layout(&self) -> &P {
        &self.description
    }
}

unsafe impl<P> VulkanObject for PipelineLayout<P> {
    type Object = vk::PipelineLayout;

    #[inline]
    fn internal_object(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl<P> Drop for PipelineLayout<P> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipelineLayout(self.device.internal_object(), self.layout, ptr::null());
        }
    }
}
