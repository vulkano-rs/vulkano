// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferInner;
use crate::buffer::BufferView;
use crate::descriptor_set::BufferAccess;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::device::Queue;
use crate::image::ImageViewAbstract;
use crate::sampler::Sampler;
use crate::sync::AccessError;
use crate::DeviceSize;
use std::sync::Arc;

pub struct DescriptorSetResources {
    buffers: Vec<(Arc<dyn BufferAccess + 'static>, u32)>,
    images: Vec<(Arc<dyn ImageViewAbstract + Send + Sync + 'static>, u32)>,
    samplers: Vec<(Arc<Sampler>, u32)>,
}

struct BufferViewResource<B>(Arc<BufferView<B>>)
where
    B: BufferAccess;

unsafe impl<B> DeviceOwned for BufferViewResource<B>
where
    B: BufferAccess,
{
    fn device(&self) -> &Arc<Device> {
        self.0.device()
    }
}

unsafe impl<B> BufferAccess for BufferViewResource<B>
where
    B: BufferAccess,
{
    fn inner(&self) -> BufferInner<'_> {
        self.0.buffer().inner()
    }

    fn size(&self) -> DeviceSize {
        self.0.buffer().size()
    }

    fn conflict_key(&self) -> (u64, u64) {
        self.0.buffer().conflict_key()
    }

    fn try_gpu_lock(&self, exclusive_access: bool, queue: &Queue) -> Result<(), AccessError> {
        self.0.buffer().try_gpu_lock(exclusive_access, queue)
    }

    unsafe fn increase_gpu_lock(&self) {
        self.0.buffer().increase_gpu_lock()
    }

    unsafe fn unlock(&self) {
        self.0.buffer().unlock()
    }
}

impl DescriptorSetResources {
    pub fn new(buffer_capacity: usize, image_capacity: usize, sampler_capacity: usize) -> Self {
        Self {
            buffers: Vec::with_capacity(buffer_capacity),
            images: Vec::with_capacity(image_capacity),
            samplers: Vec::with_capacity(sampler_capacity),
        }
    }

    pub fn num_buffers(&self) -> usize {
        self.buffers.len()
    }

    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    pub fn num_samplers(&self) -> usize {
        self.samplers.len()
    }

    pub fn add_buffer(
        &mut self,
        desc_index: u32,
        buffer: Arc<dyn BufferAccess + 'static>,
    ) {
        self.buffers.push((buffer, desc_index));
    }

    pub fn add_buffer_view<B>(&mut self, desc_index: u32, view: Arc<BufferView<B>>)
    where
        B: BufferAccess + 'static,
    {
        self.buffers
            .push((Arc::new(BufferViewResource(view)), desc_index));
    }

    pub fn add_image(
        &mut self,
        desc_index: u32,
        image: Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
    ) {
        self.images.push((image, desc_index));
    }

    pub fn add_sampler(&mut self, desc_index: u32, sampler: Arc<Sampler>) {
        self.samplers.push((sampler, desc_index))
    }

    pub fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)> {
        self.buffers
            .get(index)
            .map(|(buf, bind)| (&**buf as _, *bind))
    }

    pub fn image(&self, index: usize) -> Option<(&dyn ImageViewAbstract, u32)> {
        self.images
            .get(index)
            .map(|(img, bind)| (&**img as _, *bind))
    }
}
