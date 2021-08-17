use crate::descriptor_set::BufferAccess;
use crate::image::ImageViewAbstract;
use crate::sampler::Sampler;
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

impl BoundResources {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
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
        unimplemented!()
    }

    pub fn image(&self, index: usize) -> Option<(&dyn ImageViewAbstract, u32)> {
        unimplemented!()
    }
}
