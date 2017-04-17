// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter::Empty;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use smallvec::SmallVec;

use device::Device;
use device::Queue;
use format::ClearValue;
use format::FormatDesc;
use format::FormatTy;
use image::Dimensions;
use image::sys::ImageCreationError;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use image::sys::Usage;
use image::traits::ImageAccess;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::traits::ImageViewAccess;
use image::traits::Image;
use image::traits::ImageView;
use instance::QueueFamily;
use memory::pool::AllocLayout;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdMemoryPool;
use sync::Sharing;

/// General-purpose image in device memory. Can be used for any usage, but will be slower than a
/// specialized image.
#[derive(Debug)]
pub struct StorageImage<F, A = Arc<StdMemoryPool>> where A: MemoryPool {
    // Inner implementation.
    image: UnsafeImage,

    // We maintain a view of the whole image.
    view: UnsafeImageView,

    // Memory used to back the image.
    memory: A::Alloc,

    // Dimensions of the image view.
    dimensions: Dimensions,

    // Format.
    format: F,

    // Queue families allowed to access this image.
    queue_families: SmallVec<[u32; 4]>,

    // Number of times this image is locked on the GPU side.
    gpu_lock: AtomicUsize,
}

impl<F> StorageImage<F> {
    /// Creates a new image with the given dimensions and format.
    pub fn new<'a, I>(device: &Arc<Device>, dimensions: Dimensions, format: F, queue_families: I)
                      -> Result<Arc<StorageImage<F>>, ImageCreationError>
        where F: FormatDesc,
                 I: IntoIterator<Item = QueueFamily<'a>>
    {
        let is_depth = match format.format().ty() {
            FormatTy::Depth => true,
            FormatTy::DepthStencil => true,
            FormatTy::Stencil => true,
            FormatTy::Compressed => panic!(),
            _ => false
        };

        let usage = Usage {
            transfer_source: true,
            transfer_dest: true,
            sampled: true,
            storage: true,
            color_attachment: !is_depth,
            depth_stencil_attachment: is_depth,
            input_attachment: true,
            transient_attachment: false,
        };

        let queue_families = queue_families.into_iter().map(|f| f.id())
                                           .collect::<SmallVec<[u32; 4]>>();

        let (image, mem_reqs) = unsafe {
            let sharing = if queue_families.len() >= 2 {
                Sharing::Concurrent(queue_families.iter().cloned())
            } else {
                Sharing::Exclusive
            };

            try!(UnsafeImage::new(device, &usage, format.format(), dimensions.to_image_dimensions(),
                                  1, 1, Sharing::Exclusive::<Empty<u32>>, false, false))
        };

        let mem_ty = {
            let device_local = device.physical_device().memory_types()
                                     .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                                     .filter(|t| t.is_device_local());
            let any = device.physical_device().memory_types()
                            .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0);
            device_local.chain(any).next().unwrap()
        };

        let mem = try!(MemoryPool::alloc(&Device::standard_pool(device), mem_ty,
                                         mem_reqs.size, mem_reqs.alignment, AllocLayout::Optimal));
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        unsafe { try!(image.bind_memory(mem.memory(), mem.offset())); }

        let view = unsafe {
            try!(UnsafeImageView::raw(&image, dimensions.to_view_type(), 0 .. image.mipmap_levels(),
                                      0 .. image.dimensions().array_layers()))
        };

        Ok(Arc::new(StorageImage {
            image: image,
            view: view,
            memory: mem,
            dimensions: dimensions,
            format: format,
            queue_families: queue_families,
            gpu_lock: AtomicUsize::new(0),
        }))
    }
}

impl<F, A> StorageImage<F, A> where A: MemoryPool {
    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }
}

// FIXME: wrong
unsafe impl<F, A> Image for Arc<StorageImage<F, A>>
    where F: 'static + Send + Sync, A: MemoryPool
{
    type Access = Self;

    #[inline]
    fn access(self) -> Self {
        self
    }
}

// FIXME: wrong
unsafe impl<F, A> ImageView for Arc<StorageImage<F, A>>
    where F: 'static + Send + Sync, A: MemoryPool
{
    type Target = Self;

    #[inline]
    fn into_image_view(self) -> Self {
        self
    }
}

unsafe impl<F, A> ImageAccess for StorageImage<F, A> where F: 'static + Send + Sync, A: MemoryPool {
    #[inline]
    fn inner(&self) -> &UnsafeImage {
        &self.image
    }

    #[inline]
    fn default_layout(&self) -> Layout {
        Layout::General
    }

    #[inline]
    fn conflict_key(&self, _: u32, _: u32, _: u32, _: u32) -> u64 {
        self.image.key()
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, _: &Queue) -> bool {
        let val = self.gpu_lock.fetch_add(1, Ordering::SeqCst);
        if val == 1 {
            true
        } else {
            self.gpu_lock.fetch_sub(1, Ordering::SeqCst);
            false
        }
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        let val = self.gpu_lock.fetch_add(1, Ordering::SeqCst);
        debug_assert!(val >= 1);
    }
}

unsafe impl<F, A> ImageClearValue<F::ClearValue> for StorageImage<F, A>
    where F: FormatDesc + 'static + Send + Sync, A: MemoryPool
{
    #[inline]
    fn decode(&self, value: F::ClearValue) -> Option<ClearValue> {
        Some(self.format.decode_clear_value(value))
    }
}

unsafe impl<P, F, A> ImageContent<P> for StorageImage<F, A>
    where F: 'static + Send + Sync, A: MemoryPool
{
    #[inline]
    fn matches_format(&self) -> bool {
        true        // FIXME:
    }
}

unsafe impl<F, A> ImageViewAccess for StorageImage<F, A>
    where F: 'static + Send + Sync, A: MemoryPool
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
    fn descriptor_set_storage_image_layout(&self) -> Layout {
        Layout::General
    }

    #[inline]
    fn descriptor_set_combined_image_sampler_layout(&self) -> Layout {
        Layout::General
    }

    #[inline]
    fn descriptor_set_sampled_image_layout(&self) -> Layout {
        Layout::General
    }

    #[inline]
    fn descriptor_set_input_attachment_layout(&self) -> Layout {
        Layout::General
    }

    #[inline]
    fn identity_swizzle(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::StorageImage;
    use format::Format;
    use image::Dimensions;

    #[test]
    fn create() {
        let (device, queue) = gfx_dev_and_queue!();
        let _img = StorageImage::new(&device, Dimensions::Dim2d { width: 32, height: 32 },
                                     Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
    }
}
