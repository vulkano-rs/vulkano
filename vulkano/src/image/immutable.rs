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
use std::sync::Mutex;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use smallvec::SmallVec;

use command_buffer::Submission;
use device::Device;
use format::FormatDesc;
use image::Dimensions;
use image::sys::ImageCreationError;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use image::sys::Usage;
use image::traits::Image;
use image::traits::ImageContent;
use image::traits::ImageView;
use instance::QueueFamily;
use memory::pool::AllocLayout;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdMemoryPool;
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
    per_layer: SmallVec<[PerLayer; 1]>,
}

#[derive(Debug)]
struct PerLayer {
    latest_write_submission: Mutex<Option<Weak<Submission>>>, // TODO: can use `Weak::new()` once it's stabilized
    started_reading: AtomicBool,
}

impl<F> ImmutableImage<F> {
    /// Builds a new immutable image.
    pub fn new<'a, I>(device: &Arc<Device>, dimensions: Dimensions, format: F, queue_families: I) -> Result<Arc<ImmutableImage<F>>, ImageCreationError>
        where F: FormatDesc,
              I: IntoIterator<Item = QueueFamily<'a>>
    {
        let usage = Usage {
            transfer_source: true, // for blits
            transfer_dest: true,
            sampled: true,
            ..Usage::none()
        };

        let queue_families = queue_families.into_iter()
            .map(|f| f.id())
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
            let device_local = device.physical_device()
                .memory_types()
                .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                .filter(|t| t.is_device_local());
            let any = device.physical_device()
                .memory_types()
                .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0);
            device_local.chain(any).next().unwrap()
        };

        let mem = try!(MemoryPool::alloc(&Device::standard_pool(device), mem_ty,
                                         mem_reqs.size, mem_reqs.alignment, AllocLayout::Optimal));
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        unsafe {
            try!(image.bind_memory(mem.memory(), mem.offset()));
        }

        let view = unsafe {
            try!(UnsafeImageView::raw(&image, dimensions.to_view_type(), 0 .. image.mipmap_levels(),
                                      0 .. image.dimensions().array_layers()))
        };

        Ok(Arc::new(ImmutableImage {
            image: image,
            view: view,
            memory: mem,
            dimensions: dimensions,
            format: format,
            per_layer: {
                let mut v = SmallVec::new();
                for _ in 0..dimensions.array_layers_with_cube() {
                    v.push(PerLayer {
                        latest_write_submission: Mutex::new(None),
                        started_reading: AtomicBool::new(false),
                    });
                }
                v
            },
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
}

unsafe impl<F, A> Image for ImmutableImage<F, A>
    where F: 'static + Send + Sync,
          A: MemoryPool
{
    #[inline]
    fn inner(&self) -> &UnsafeImage {
        &self.image
    }

    #[inline]
    fn conflict_key(&self, _: u32, _: u32, _: u32, _: u32) -> u64 {
        self.image.key()
    }
}

unsafe impl<P, F, A> ImageContent<P> for ImmutableImage<F, A>
    where F: 'static + Send + Sync,
          A: MemoryPool
{
    #[inline]
    fn matches_format(&self) -> bool {
        true        // FIXME:
    }
}

unsafe impl<F: 'static, A> ImageView for ImmutableImage<F, A>
    where F: 'static + Send + Sync,
          A: MemoryPool
{
    #[inline]
    fn parent(&self) -> &Image {
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
        Layout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn descriptor_set_combined_image_sampler_layout(&self) -> Layout {
        Layout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn descriptor_set_sampled_image_layout(&self) -> Layout {
        Layout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn descriptor_set_input_attachment_layout(&self) -> Layout {
        Layout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn identity_swizzle(&self) -> bool {
        true
    }
}
