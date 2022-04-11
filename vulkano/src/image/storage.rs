// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    sys::UnsafeImage,
    traits::{ImageClearValue, ImageContent},
    ImageAccess, ImageCreateFlags, ImageCreationError, ImageDescriptorLayouts, ImageDimensions,
    ImageInner, ImageLayout, ImageUsage,
};
use crate::{
    device::{physical::QueueFamily, Device, DeviceOwned},
    format::{ClearValue, Format},
    image::sys::UnsafeImageCreateInfo,
    memory::{
        pool::{
            alloc_dedicated_with_exportable_fd, AllocFromRequirementsFilter, AllocLayout,
            MappingRequirement, MemoryPoolAlloc, PotentialDedicatedAllocation, StdMemoryPool,
        },
        DedicatedAllocation, DeviceMemoryExportError, ExternalMemoryHandleType,
        ExternalMemoryHandleTypes, MemoryPool,
    },
    sync::Sharing,
    DeviceSize,
};
use smallvec::SmallVec;
use std::{
    fs::File,
    hash::{Hash, Hasher},
    ops::Range,
    sync::Arc,
};

/// General-purpose image in device memory. Can be used for any usage, but will be slower than a
/// specialized image.
#[derive(Debug)]
pub struct StorageImage<A = Arc<StdMemoryPool>>
where
    A: MemoryPool,
{
    // Inner implementation.
    image: Arc<UnsafeImage>,

    // Memory used to back the image.
    memory: PotentialDedicatedAllocation<A::Alloc>,

    // Dimensions of the image.
    dimensions: ImageDimensions,

    // Format.
    format: Format,

    // Queue families allowed to access this image.
    queue_families: SmallVec<[u32; 4]>,
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
            transfer_src: true,
            transfer_dst: true,
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

unsafe impl<A> DeviceOwned for StorageImage<A>
where
    A: MemoryPool,
{
    fn device(&self) -> &Arc<Device> {
        self.image.device()
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
            num_layers: self.dimensions.array_layers(),
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
    fn current_mip_levels_access(&self) -> Range<u32> {
        0..self.mip_levels()
    }

    #[inline]
    fn current_array_layers_access(&self) -> Range<u32> {
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
