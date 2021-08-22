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

pub struct BoundResources {
    resources: Vec<BoundResource>,
}

struct BoundResource {
    desc_index: u32,
    ty: BoundResourceTy,
    ty_index: usize,
    data: BoundResourceData,
}

#[derive(PartialEq)]
enum BoundResourceTy {
    Buffer,
    Image,
    Sampler,
}

enum BoundResourceData {
    Buffer(Arc<dyn BufferAccess + Send + Sync + 'static>),
    Image(Arc<dyn ImageViewAbstract + Send + Sync + 'static>),
    Sampler(Arc<Sampler>),
}

impl BoundResourceData {
    fn buffer_ref(&self) -> &(dyn BufferAccess + Send + Sync + 'static) {
        match self {
            Self::Buffer(buf) => &*buf,
            _ => panic!("resource is not a buffer"),
        }
    }

    fn image_ref(&self) -> &(dyn ImageViewAbstract + Send + Sync + 'static) {
        match self {
            Self::Image(img) => &*img,
            _ => panic!("resource is not an image"),
        }
    }
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

impl BoundResources {
    pub fn new(capacity: usize) -> Self {
        Self {
            resources: Vec::with_capacity(capacity),
        }
    }

    pub fn num_buffers(&self) -> usize {
        self.resources
            .iter()
            .filter(|r| r.ty == BoundResourceTy::Buffer)
            .count()
    }

    pub fn num_images(&self) -> usize {
        self.resources
            .iter()
            .filter(|r| r.ty == BoundResourceTy::Image)
            .count()
    }

    pub fn num_samplers(&self) -> usize {
        self.resources
            .iter()
            .filter(|r| r.ty == BoundResourceTy::Sampler)
            .count()
    }

    pub fn add_buffer(
        &mut self,
        desc_index: u32,
        buffer: Arc<dyn BufferAccess + Send + Sync + 'static>,
    ) {
        let ty_index = self.num_buffers();

        self.resources.push(BoundResource {
            desc_index,
            ty: BoundResourceTy::Buffer,
            ty_index,
            data: BoundResourceData::Buffer(buffer),
        });
    }

    pub fn add_buffer_view<B>(&mut self, desc_index: u32, view: Arc<BufferView<B>>)
    where
        B: BufferAccess + 'static,
    {
        let ty_index = self.num_buffers();

        self.resources.push(BoundResource {
            desc_index,
            ty: BoundResourceTy::Buffer,
            ty_index,
            data: BoundResourceData::Buffer(Arc::new(BufferViewResource(view))),
        });
    }

    pub fn add_image(
        &mut self,
        desc_index: u32,
        image: Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
    ) {
        let ty_index = self.num_images();

        self.resources.push(BoundResource {
            desc_index,
            ty: BoundResourceTy::Image,
            ty_index,
            data: BoundResourceData::Image(image),
        })
    }

    pub fn add_sampler(&mut self, desc_index: u32, sampler: Arc<Sampler>) {
        let ty_index = self.num_samplers();

        self.resources.push(BoundResource {
            desc_index,
            ty: BoundResourceTy::Sampler,
            ty_index,
            data: BoundResourceData::Sampler(sampler),
        });
    }

    pub fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)> {
        for resource in &self.resources {
            if resource.ty == BoundResourceTy::Buffer {
                if resource.ty_index == index {
                    return Some((resource.data.buffer_ref(), resource.desc_index));
                }
            }
        }

        None
    }

    pub fn image(&self, index: usize) -> Option<(&dyn ImageViewAbstract, u32)> {
        for resource in &self.resources {
            if resource.ty == BoundResourceTy::Image {
                if resource.ty_index == index {
                    return Some((resource.data.image_ref(), resource.desc_index));
                }
            }
        }

        None
    }
}
