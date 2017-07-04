// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::sync::Arc;

use device::Device;
use device::Queue;
use format::FormatDesc;
use image::Dimensions;
use image::ImageInner;
use image::ImageLayout;
use image::ImageUsage;
use image::MipmapsCount;
use image::sys::ImageCreationError;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use image::traits::ImageAccess;
use image::traits::ImageContent;
use image::traits::ImageViewAccess;
use instance::QueueFamily;
use memory::pool::AllocLayout;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdMemoryPool;
use sync::AccessError;
use sync::Sharing;

/// Image whose purpose is to be used for read-only purposes. You can write to the image once,
/// but then you must only ever read from it. TODO: clarify because of blit operations
// TODO: type (2D, 3D, array, etc.) as template parameter
#[derive(Debug)]
pub struct ImmutableImage<F, A = Arc<StdMemoryPool>>
    where A: MemoryPool
{
    image: UnsafeImage,
    view: UnsafeImageView,
    dimensions: Dimensions,
    memory: A::Alloc,
    format: F,
}

impl<F> ImmutableImage<F> {
    /// Builds a new immutable image.
    // TODO: one mipmap is probably not a great default
    #[inline]
    pub fn new<'a, I>(device: Arc<Device>, dimensions: Dimensions, format: F, queue_families: I)
                      -> Result<Arc<ImmutableImage<F>>, ImageCreationError>
        where F: FormatDesc,
              I: IntoIterator<Item = QueueFamily<'a>>
    {
        ImmutableImage::with_mipmaps(device,
                                     dimensions,
                                     format,
                                     MipmapsCount::One,
                                     queue_families)
    }

    /// Builds a new immutable image with the given number of mipmaps.
    pub fn with_mipmaps<'a, I, M>(device: Arc<Device>, dimensions: Dimensions, format: F,
                                  mipmaps: M, queue_families: I)
                                  -> Result<Arc<ImmutableImage<F>>, ImageCreationError>
        where F: FormatDesc,
              I: IntoIterator<Item = QueueFamily<'a>>,
              M: Into<MipmapsCount>
    {
        let usage = ImageUsage {
            transfer_source: true, // for blits
            transfer_destination: true,
            sampled: true,
            ..ImageUsage::none()
        };

        let queue_families = queue_families
            .into_iter()
            .map(|f| f.id())
            .collect::<SmallVec<[u32; 4]>>();

        let (image, mem_reqs) = unsafe {
            let sharing = if queue_families.len() >= 2 {
                Sharing::Concurrent(queue_families.iter().cloned())
            } else {
                Sharing::Exclusive
            };

            UnsafeImage::new(device.clone(),
                             usage,
                             format.format(),
                             dimensions.to_image_dimensions(),
                             1,
                             mipmaps,
                             sharing,
                             false,
                             false)?
        };

        let mem_ty = {
            let device_local = device
                .physical_device()
                .memory_types()
                .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                .filter(|t| t.is_device_local());
            let any = device
                .physical_device()
                .memory_types()
                .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0);
            device_local.chain(any).next().unwrap()
        };

        let mem = MemoryPool::alloc(&Device::standard_pool(&device),
                                    mem_ty,
                                    mem_reqs.size,
                                    mem_reqs.alignment,
                                    AllocLayout::Optimal)?;
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        unsafe {
            image.bind_memory(mem.memory(), mem.offset())?;
        }

        let view = unsafe {
            UnsafeImageView::raw(&image,
                                 dimensions.to_view_type(),
                                 0 .. image.mipmap_levels(),
                                 0 .. image.dimensions().array_layers())?
        };

        Ok(Arc::new(ImmutableImage {
                        image: image,
                        view: view,
                        memory: mem,
                        dimensions: dimensions,
                        format: format,
                    }))
    }
}

impl<F, A> ImmutableImage<F, A>
    where A: MemoryPool
{
    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    /// Returns the number of mipmap levels of the image.
    #[inline]
    pub fn mipmap_levels(&self) -> u32 {
        self.image.mipmap_levels()
    }
}

unsafe impl<F, A> ImageAccess for ImmutableImage<F, A>
    where F: 'static + Send + Sync,
          A: MemoryPool
{
    #[inline]
    fn inner(&self) -> ImageInner {
        ImageInner {
            image: &self.image,
            first_layer: 0,
            num_layers: self.image.dimensions().array_layers() as usize,
            first_mipmap_level: 0,
            num_mipmap_levels: self.image.mipmap_levels() as usize,
        }
    }

    #[inline]
    fn initial_layout_requirement(&self) -> ImageLayout {
        ImageLayout::ShaderReadOnlyOptimal // TODO: ?
    }

    #[inline]
    fn final_layout_requirement(&self) -> ImageLayout {
        ImageLayout::ShaderReadOnlyOptimal // TODO: ?
    }

    #[inline]
    fn conflict_key(&self, _: u32, _: u32, _: u32, _: u32) -> u64 {
        self.image.key()
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive_access: bool, queue: &Queue) -> Result<(), AccessError> {
        Ok(()) // FIXME:
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        // FIXME:
    }

    #[inline]
    unsafe fn unlock(&self) {
        // TODO:
    }
}

unsafe impl<P, F, A> ImageContent<P> for ImmutableImage<F, A>
    where F: 'static + Send + Sync,
          A: MemoryPool
{
    #[inline]
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

unsafe impl<F: 'static, A> ImageViewAccess for ImmutableImage<F, A>
    where F: 'static + Send + Sync,
          A: MemoryPool
{
    #[inline]
    fn parent(&self) -> &ImageAccess {
        self
    }

    #[inline]
    fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    #[inline]
    fn inner(&self) -> &UnsafeImageView {
        &self.view
    }

    #[inline]
    fn descriptor_set_storage_image_layout(&self) -> ImageLayout {
        ImageLayout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout {
        ImageLayout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn descriptor_set_sampled_image_layout(&self) -> ImageLayout {
        ImageLayout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn descriptor_set_input_attachment_layout(&self) -> ImageLayout {
        ImageLayout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn identity_swizzle(&self) -> bool {
        true
    }
}
