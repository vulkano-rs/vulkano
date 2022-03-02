// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::device::physical::QueueFamily;
use crate::device::Device;
use crate::format::ClearValue;
use crate::format::Format;
use crate::image::sys::ImageCreationError;
use crate::image::sys::UnsafeImage;
use crate::image::sys::UnsafeImageCreateInfo;
use crate::image::traits::ImageAccess;
use crate::image::traits::ImageClearValue;
use crate::image::traits::ImageContent;
use crate::image::ImageCreateFlags;
use crate::image::ImageDescriptorLayouts;
use crate::image::ImageDimensions;
use crate::image::ImageInner;
use crate::image::ImageLayout;
use crate::image::ImageUsage;
use crate::memory::pool::alloc_dedicated_with_exportable_fd;
use crate::memory::pool::AllocFromRequirementsFilter;
use crate::memory::pool::AllocLayout;
use crate::memory::pool::MappingRequirement;
use crate::memory::pool::MemoryPool;
use crate::memory::pool::MemoryPoolAlloc;
use crate::memory::pool::PotentialDedicatedAllocation;
use crate::memory::pool::StdMemoryPool;
use crate::memory::DedicatedAllocation;
use crate::memory::ExternalMemoryHandleType;
use crate::memory::{DeviceMemoryExportError, ExternalMemoryHandleTypes};
use crate::sync::AccessError;
use crate::sync::Sharing;
use crate::DeviceSize;
use smallvec::SmallVec;
use std::fs::File;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// General-purpose image in device memory. Can be used for any usage, but will be slower than a
/// specialized image.
#[derive(Debug)]
pub struct StorageImage<A = Arc<StdMemoryPool>>
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
    format: Format,

    // Queue families allowed to access this image.
    queue_families: SmallVec<[u32; 4]>,

    // Number of times this image is locked on the GPU side.
    gpu_lock: AtomicUsize,
}

impl StorageImage {
    /// Creates a new image with the given dimensions and format.
    #[inline]
    pub fn new<'a, I>(
        device: Arc<Device>,
        dimensions: ImageDimensions,
        format: Format,
        queue_families: I,
    ) -> Result<Arc<StorageImage>, ImageCreationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        let aspects = format.aspects();
        let is_depth = aspects.depth || aspects.stencil;

        if format.compression().is_some() {
            panic!() // TODO: message?
        }

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
        format: Format,
        usage: ImageUsage,
        flags: ImageCreateFlags,
        queue_families: I,
    ) -> Result<Arc<StorageImage>, ImageCreationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        let queue_families = queue_families
            .into_iter()
            .map(|f| f.id())
            .collect::<SmallVec<[u32; 4]>>();

        let image = UnsafeImage::new(
            device.clone(),
            UnsafeImageCreateInfo {
                dimensions,
                format: Some(format),
                usage,
                sharing: if queue_families.len() >= 2 {
                    Sharing::Concurrent(queue_families.iter().cloned().collect())
                } else {
                    Sharing::Exclusive
                },
                mutable_format: flags.mutable_format,
                cube_compatible: flags.cube_compatible,
                array_2d_compatible: flags.array_2d_compatible,
                block_texel_view_compatible: flags.block_texel_view_compatible,
                ..Default::default()
            },
        )?;

        let mem_reqs = image.memory_requirements();
        let memory = MemoryPool::alloc_from_requirements(
            &Device::standard_pool(&device),
            &mem_reqs,
            AllocLayout::Optimal,
            MappingRequirement::DoNotMap,
            Some(DedicatedAllocation::Image(&image)),
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

    pub fn new_with_exportable_fd<'a, I>(
        device: Arc<Device>,
        dimensions: ImageDimensions,
        format: Format,
        usage: ImageUsage,
        flags: ImageCreateFlags,
        queue_families: I,
    ) -> Result<Arc<StorageImage>, ImageCreationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        let queue_families = queue_families
            .into_iter()
            .map(|f| f.id())
            .collect::<SmallVec<[u32; 4]>>();

        let image = UnsafeImage::new(
            device.clone(),
            UnsafeImageCreateInfo {
                dimensions,
                format: Some(format),
                usage,
                sharing: if queue_families.len() >= 2 {
                    Sharing::Concurrent(queue_families.iter().cloned().collect())
                } else {
                    Sharing::Exclusive
                },
                external_memory_handle_types: ExternalMemoryHandleTypes {
                    opaque_fd: true,
                    ..ExternalMemoryHandleTypes::none()
                },
                mutable_format: flags.mutable_format,
                cube_compatible: flags.cube_compatible,
                array_2d_compatible: flags.array_2d_compatible,
                block_texel_view_compatible: flags.block_texel_view_compatible,
                ..Default::default()
            },
        )?;

        let mem_reqs = image.memory_requirements();
        let memory = alloc_dedicated_with_exportable_fd(
            device.clone(),
            &mem_reqs,
            AllocLayout::Optimal,
            MappingRequirement::DoNotMap,
            DedicatedAllocation::Image(&image),
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

    /// Exports posix file descriptor for the allocated memory
    /// requires `khr_external_memory_fd` and `khr_external_memory` extensions to be loaded.
    pub fn export_posix_fd(&self) -> Result<File, DeviceMemoryExportError> {
        self.memory
            .memory()
            .export_fd(ExternalMemoryHandleType::OpaqueFd)
    }

    /// Return the size of the allocated memory (used for e.g. with cuda)
    pub fn mem_size(&self) -> DeviceSize {
        self.memory.memory().allocation_size()
    }
}

unsafe impl<A> ImageAccess for StorageImage<A>
where
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
    fn conflict_key(&self) -> u64 {
        self.image.key()
    }

    #[inline]
    fn try_gpu_lock(
        &self,
        _: bool,
        uninitialized_safe: bool,
        expected_layout: ImageLayout,
    ) -> Result<(), AccessError> {
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
    fn current_mip_levels_access(&self) -> std::ops::Range<u32> {
        0..self.mip_levels()
    }

    #[inline]
    fn current_array_layers_access(&self) -> std::ops::Range<u32> {
        0..self.dimensions().array_layers()
    }
}

unsafe impl<A> ImageClearValue<ClearValue> for StorageImage<A>
where
    A: MemoryPool,
{
    #[inline]
    fn decode(&self, value: ClearValue) -> Option<ClearValue> {
        Some(self.format.decode_clear_value(value))
    }
}

unsafe impl<P, A> ImageContent<P> for StorageImage<A>
where
    A: MemoryPool,
{
    #[inline]
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

impl<A> PartialEq for StorageImage<A>
where
    A: MemoryPool,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl<A> Eq for StorageImage<A> where A: MemoryPool {}

impl<A> Hash for StorageImage<A>
where
    A: MemoryPool,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::StorageImage;
    use crate::format::Format;
    use crate::image::ImageDimensions;

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
            Format::R8G8B8A8_UNORM,
            Some(queue.family()),
        )
        .unwrap();
    }
}
