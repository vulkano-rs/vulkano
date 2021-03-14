// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use buffer::BufferAccess;
use device::Device;
use format::ClearValue;
use format::FormatDesc;
use format::FormatTy;
use image::sys::ImageCreationError;
use image::sys::UnsafeImage;
use image::traits::ImageAccess;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::ImageCreateFlags;
use image::ImageDescriptorLayouts;
use image::ImageDimensions;
use image::ImageInner;
use image::ImageLayout;
use image::ImageUsage;
use instance::QueueFamily;
use memory::pool::AllocFromRequirementsFilter;
use memory::pool::AllocLayout;
use memory::pool::MappingRequirement;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::PotentialDedicatedAllocation;
use memory::pool::StdMemoryPool;
use memory::DedicatedAlloc;
use sync::AccessError;
use sync::Sharing;

/// General-purpose image in device memory. Can be used for any usage, but will be slower than a
/// specialized image.
#[derive(Debug)]
pub struct StorageImage<F, A = Arc<StdMemoryPool>>
where
    A: MemoryPool,
{
    // Inner implementation.
    image: UnsafeImage,

    // Memory used to back the image.
    memory: PotentialDedicatedAllocation<A::Alloc>,

    // Dimensions of the image.
    dimensions: ImageDimensions,

    // Format.
    format: F,

    // Queue families allowed to access this image.
    queue_families: SmallVec<[u32; 4]>,

    // Number of times this image is locked on the GPU side.
    gpu_lock: AtomicUsize,
}

impl<F> StorageImage<F> {
    /// Creates a new image with the given dimensions and format.
    #[inline]
    pub fn new<'a, I>(
        device: Arc<Device>,
        dimensions: ImageDimensions,
        format: F,
        queue_families: I,
    ) -> Result<Arc<StorageImage<F>>, ImageCreationError>
    where
        F: FormatDesc,
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        let is_depth = match format.format().ty() {
            FormatTy::Depth => true,
            FormatTy::DepthStencil => true,
            FormatTy::Stencil => true,
            FormatTy::Compressed => panic!(),
            _ => false,
        };

        let usage = ImageUsage {
            transfer_source: true,
            transfer_destination: true,
            sampled: true,
            storage: true,
            color_attachment: !is_depth,
            depth_stencil_attachment: is_depth,
            input_attachment: true,
            transient_attachment: false,
        };
        let flags = ImageCreateFlags::none();

        StorageImage::with_usage(device, dimensions, format, usage, flags, queue_families)
    }

    /// Same as `new`, but allows specifying the usage.
    pub fn with_usage<'a, I>(
        device: Arc<Device>,
        dimensions: ImageDimensions,
        format: F,
        usage: ImageUsage,
        flags: ImageCreateFlags,
        queue_families: I,
    ) -> Result<Arc<StorageImage<F>>, ImageCreationError>
    where
        F: FormatDesc,
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
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

            UnsafeImage::new(
                device.clone(),
                usage,
                format.format(),
                flags,
                dimensions,
                1,
                1,
                sharing,
                false,
                false,
            )?
        };

        let memory = MemoryPool::alloc_from_requirements(
            &Device::standard_pool(&device),
            &mem_reqs,
            AllocLayout::Optimal,
            MappingRequirement::DoNotMap,
            DedicatedAlloc::Image(&image),
            |t| {
                if t.is_device_local() {
                    AllocFromRequirementsFilter::Preferred
                } else {
                    AllocFromRequirementsFilter::Allowed
                }
            },
        )?;
        debug_assert!((memory.offset() % mem_reqs.alignment) == 0);
        unsafe {
            image.bind_memory(memory.memory(), memory.offset())?;
        }

        Ok(Arc::new(StorageImage {
            image,
            memory,
            dimensions,
            format,
            queue_families,
            gpu_lock: AtomicUsize::new(0),
        }))
    }
}

impl<F, A> StorageImage<F, A>
where
    A: MemoryPool,
{
    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> ImageDimensions {
        self.dimensions
    }
}

unsafe impl<F, A> ImageAccess for StorageImage<F, A>
where
    F: 'static + Send + Sync,
    A: MemoryPool,
{
    #[inline]
    fn inner(&self) -> ImageInner {
        ImageInner {
            image: &self.image,
            first_layer: 0,
            num_layers: self.dimensions.array_layers() as usize,
            first_mipmap_level: 0,
            num_mipmap_levels: 1,
        }
    }

    #[inline]
    fn initial_layout_requirement(&self) -> ImageLayout {
        ImageLayout::General
    }

    #[inline]
    fn final_layout_requirement(&self) -> ImageLayout {
        ImageLayout::General
    }

    #[inline]
    fn descriptor_layouts(&self) -> Option<ImageDescriptorLayouts> {
        Some(ImageDescriptorLayouts {
            storage_image: ImageLayout::General,
            combined_image_sampler: ImageLayout::General,
            sampled_image: ImageLayout::General,
            input_attachment: ImageLayout::General,
        })
    }

    #[inline]
    fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool {
        false
    }

    #[inline]
    fn conflicts_image(&self, other: &dyn ImageAccess) -> bool {
        self.conflict_key() == other.conflict_key() // TODO:
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.image.key()
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, expected_layout: ImageLayout) -> Result<(), AccessError> {
        // TODO: handle initial layout transition
        if expected_layout != ImageLayout::General && expected_layout != ImageLayout::Undefined {
            return Err(AccessError::UnexpectedImageLayout {
                requested: expected_layout,
                allowed: ImageLayout::General,
            });
        }

        let val = self
            .gpu_lock
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap_or_else(|e| e);
        if val == 0 {
            Ok(())
        } else {
            Err(AccessError::AlreadyInUse)
        }
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        let val = self.gpu_lock.fetch_add(1, Ordering::SeqCst);
        debug_assert!(val >= 1);
    }

    #[inline]
    unsafe fn unlock(&self, new_layout: Option<ImageLayout>) {
        assert!(new_layout.is_none() || new_layout == Some(ImageLayout::General));
        self.gpu_lock.fetch_sub(1, Ordering::SeqCst);
    }

    #[inline]
    fn current_miplevels_access(&self) -> std::ops::Range<u32> {
        0..self.mipmap_levels()
    }

    #[inline]
    fn current_layer_levels_access(&self) -> std::ops::Range<u32> {
        0..self.dimensions().array_layers()
    }
}

unsafe impl<F, A> ImageClearValue<F::ClearValue> for StorageImage<F, A>
where
    F: FormatDesc + 'static + Send + Sync,
    A: MemoryPool,
{
    #[inline]
    fn decode(&self, value: F::ClearValue) -> Option<ClearValue> {
        Some(self.format.decode_clear_value(value))
    }
}

unsafe impl<P, F, A> ImageContent<P> for StorageImage<F, A>
where
    F: 'static + Send + Sync,
    A: MemoryPool,
{
    #[inline]
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

impl<F, A> PartialEq for StorageImage<F, A>
where
    F: 'static + Send + Sync,
    A: MemoryPool,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ImageAccess::inner(self) == ImageAccess::inner(other)
    }
}

impl<F, A> Eq for StorageImage<F, A>
where
    F: 'static + Send + Sync,
    A: MemoryPool,
{
}

impl<F, A> Hash for StorageImage<F, A>
where
    F: 'static + Send + Sync,
    A: MemoryPool,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        ImageAccess::inner(self).hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::StorageImage;
    use format::Format;
    use image::ImageDimensions;

    #[test]
    fn create() {
        let (device, queue) = gfx_dev_and_queue!();
        let _img = StorageImage::new(
            device,
            ImageDimensions::Dim2d {
                width: 32,
                height: 32,
                array_layers: 1,
            },
            Format::R8G8B8A8Unorm,
            Some(queue.family()),
        )
        .unwrap();
    }
}
