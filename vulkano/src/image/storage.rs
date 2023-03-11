// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    sys::{Image, ImageMemory, RawImage},
    traits::ImageContent,
    ImageAccess, ImageAspects, ImageCreateFlags, ImageDescriptorLayouts, ImageDimensions,
    ImageError, ImageInner, ImageLayout, ImageUsage,
};
use crate::{
    device::{Device, DeviceOwned, Queue},
    format::Format,
    image::{sys::ImageCreateInfo, view::ImageView, ImageFormatInfo},
    memory::{
        allocator::{
            AllocationCreateInfo, AllocationType, MemoryAllocatePreference,
            MemoryAllocator, MemoryUsage,
        },
        DedicatedAllocation, DeviceMemoryError, ExternalMemoryHandleType,
        ExternalMemoryHandleTypes,
    },
    sync::Sharing,
    DeviceSize,
};
use smallvec::SmallVec;

#[cfg(target_os = "linux")]
use std::os::unix::prelude::{FromRawFd, IntoRawFd, RawFd};
#[cfg(target_os = "linux")]
use ash::vk::{ImageDrmFormatModifierExplicitCreateInfoEXT, SubresourceLayout};
#[cfg(target_os = "linux")]
use crate::{
    image::ImageTiling,
    memory::{
        allocator::MemoryAlloc,
        DeviceMemory,
        MemoryAllocateFlags,
        MemoryAllocateInfo
    }
};

use std::{
    fs::File,
    hash::{Hash, Hasher},
    sync::Arc,
};

/// General-purpose image in device memory. Can be used for any usage, but will be slower than a
/// specialized image.
#[derive(Debug)]
pub struct StorageImage {
    inner: Arc<Image>,
}

impl StorageImage {
    /// Creates a new image with the given dimensions and format.
    pub fn new(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: ImageDimensions,
        format: Format,
        queue_family_indices: impl IntoIterator<Item = u32>,
    ) -> Result<Arc<StorageImage>, ImageError> {
        let aspects = format.aspects();
        let is_depth_stencil = aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL);

        if format.compression().is_some() {
            panic!() // TODO: message?
        }

        let usage = ImageUsage::TRANSFER_SRC
            | ImageUsage::TRANSFER_DST
            | ImageUsage::SAMPLED
            | ImageUsage::STORAGE
            | ImageUsage::INPUT_ATTACHMENT
            | if is_depth_stencil {
                ImageUsage::DEPTH_STENCIL_ATTACHMENT
            } else {
                ImageUsage::COLOR_ATTACHMENT
            };
        let flags = ImageCreateFlags::empty();

        StorageImage::with_usage(
            allocator,
            dimensions,
            format,
            usage,
            flags,
            queue_family_indices,
        )
    }

    /// Same as `new`, but allows specifying the usage.
    pub fn with_usage(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: ImageDimensions,
        format: Format,
        usage: ImageUsage,
        flags: ImageCreateFlags,
        queue_family_indices: impl IntoIterator<Item = u32>,
    ) -> Result<Arc<StorageImage>, ImageError> {
        let queue_family_indices: SmallVec<[_; 4]> = queue_family_indices.into_iter().collect();
        assert!(!flags.intersects(ImageCreateFlags::DISJOINT)); // TODO: adjust the code below to make this safe

        let raw_image = RawImage::new(
            allocator.device().clone(),
            ImageCreateInfo {
                flags,
                dimensions,
                format: Some(format),
                usage,
                sharing: if queue_family_indices.len() >= 2 {
                    Sharing::Concurrent(queue_family_indices)
                } else {
                    Sharing::Exclusive
                },
                ..Default::default()
            },
        )?;
        let requirements = raw_image.memory_requirements()[0];
        let create_info = AllocationCreateInfo {
            requirements,
            allocation_type: AllocationType::NonLinear,
            usage: MemoryUsage::GpuOnly,
            allocate_preference: MemoryAllocatePreference::Unknown,
            dedicated_allocation: Some(DedicatedAllocation::Image(&raw_image)),
            ..Default::default()
        };

        match unsafe { allocator.allocate_unchecked(create_info) } {
            Ok(alloc) => {
                debug_assert!(alloc.offset() % requirements.layout.alignment().as_nonzero() == 0);
                debug_assert!(alloc.size() == requirements.layout.size());

                let inner = Arc::new(unsafe {
                    raw_image
                        .bind_memory_unchecked([alloc])
                        .map_err(|(err, _, _)| err)?
                });

                Ok(Arc::new(StorageImage { inner }))
            }
            Err(err) => Err(err.into()),
        }
    }

    pub fn new_with_exportable_fd(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: ImageDimensions,
        format: Format,
        usage: ImageUsage,
        flags: ImageCreateFlags,
        queue_family_indices: impl IntoIterator<Item = u32>,
    ) -> Result<Arc<StorageImage>, ImageError> {
        let queue_family_indices: SmallVec<[_; 4]> = queue_family_indices.into_iter().collect();
        assert!(!flags.intersects(ImageCreateFlags::DISJOINT)); // TODO: adjust the code below to make this safe

        let external_memory_properties = allocator
            .device()
            .physical_device()
            .image_format_properties(ImageFormatInfo {
                flags,
                format: Some(format),
                image_type: dimensions.image_type(),
                usage,
                external_memory_handle_type: Some(ExternalMemoryHandleType::OpaqueFd),
                ..Default::default()
            })
            .unwrap()
            .unwrap()
            .external_memory_properties;
        // VUID-VkExportMemoryAllocateInfo-handleTypes-00656
        assert!(external_memory_properties.exportable);

        // VUID-VkMemoryAllocateInfo-pNext-00639
        // Guaranteed because we always create a dedicated allocation

        let external_memory_handle_types = ExternalMemoryHandleTypes::OPAQUE_FD;
        let raw_image = RawImage::new(
            allocator.device().clone(),
            ImageCreateInfo {
                flags,
                dimensions,
                format: Some(format),
                usage,
                sharing: if queue_family_indices.len() >= 2 {
                    Sharing::Concurrent(queue_family_indices)
                } else {
                    Sharing::Exclusive
                },
                external_memory_handle_types,
                ..Default::default()
            },
        )?;
        let requirements = raw_image.memory_requirements()[0];
        let memory_type_index = allocator
            .find_memory_type_index(requirements.memory_type_bits, MemoryUsage::GpuOnly.into())
            .expect("failed to find a suitable memory type");

        match unsafe {
            allocator.allocate_dedicated_unchecked(
                memory_type_index,
                requirements.layout.size(),
                Some(DedicatedAllocation::Image(&raw_image)),
                external_memory_handle_types,
            )
        } {
            Ok(alloc) => {
                debug_assert!(alloc.offset() % requirements.layout.alignment().as_nonzero() == 0);
                debug_assert!(alloc.size() == requirements.layout.size());

                let inner = Arc::new(unsafe {
                    raw_image
                        .bind_memory_unchecked([alloc])
                        .map_err(|(err, _, _)| err)?
                });

                Ok(Arc::new(StorageImage { inner }))
            }
            Err(err) => Err(err.into()),
        }
    }

    #[cfg(target_os = "linux")]
    /// Creates a new image from a set of Linux dma_buf file descriptors. The memory will be imported from the file desciptors, and will be bound to the image.
    /// # Arguments
    /// * `fds` - The list of file descriptors to import from. Single planar images should only use one, and multiplanar images can use multiple, for example, for each color.
    /// * `offset` - The byte offset from the start of the image of the plane where the image subresource begins.
    /// * `pitch` - Describes the number of bytes between each row of texels in an image.
    pub fn new_from_dma_buf_fd(
        allocator: &(impl MemoryAllocator + ?Sized),
        device: Arc<Device>,
        dimensions: ImageDimensions,
        format: Format,
        usage: ImageUsage,
        flags: ImageCreateFlags,
        queue_family_indices: impl IntoIterator<Item = u32>,
        mut subresource_data: Vec<SubresourceData>,
        drm_format_modifier: u64,
    ) -> Result<Arc<StorageImage>, ImageError> {
        let queue_family_indices: SmallVec<[_; 4]> = queue_family_indices.into_iter().collect();

        // Create a vector of the layout of each image plane.
        let layout: Vec<SubresourceLayout> = subresource_data
            .iter_mut()
            .map(
                |SubresourceData {
                     fd: _,
                     offset,
                     row_pitch,
                 }| {
                    SubresourceLayout {
                        offset: *offset,
                        size: 0,
                        row_pitch: *row_pitch,
                        array_pitch: 0_u64,
                        depth_pitch: 0_u64,
                    }
                },
            )
            .collect();

        let fds: Vec<RawFd> = subresource_data
            .iter_mut()
            .map(
                |SubresourceData {
                     fd,
                     offset: _,
                     row_pitch: _,
                 }| { *fd },
            )
            .collect();

        let drm_mod = ImageDrmFormatModifierExplicitCreateInfoEXT::builder()
            .drm_format_modifier(drm_format_modifier)
            .plane_layouts(layout.as_ref())
            .build();

        let external_memory_handle_types = ExternalMemoryHandleTypes::DMA_BUF;

        let image = RawImage::new(
            device.clone(),
            ImageCreateInfo {
                flags,
                dimensions,
                format: Some(format),
                usage,
                sharing: if queue_family_indices.len() >= 2 {
                    Sharing::Concurrent(queue_family_indices)
                } else {
                    Sharing::Exclusive
                },
                external_memory_handle_types,
                tiling: ImageTiling::DrmFormatModifier,
                image_drm_format_modifier_create_info: Some(drm_mod),
                ..Default::default()
            },
        )?;

        let requirements = image.memory_requirements()[0];
        let memory_type_index = allocator
            .find_memory_type_index(requirements.memory_type_bits, MemoryUsage::GpuOnly.into())
            .expect("failed to find a suitable memory type");

        assert!(device.enabled_extensions().khr_external_memory_fd);
        assert!(device.enabled_extensions().khr_external_memory);
        assert!(device.enabled_extensions().ext_external_memory_dma_buf);

        let memory = unsafe {
            // TODO: For completeness, importing memory from muliple file descriptors should be added (In order to support importing multiplanar images). As of now, only single planar image importing will work.
            if fds.len() != 1 {
                panic!("Only single-planar image importing is currently supported.")
            }

            // Try cloning underlying fd
            let file = File::from_raw_fd(*fds.first().expect("File descriptor Vec is empty"));
            let new_file = file.try_clone().expect("Error cloning file descriptor");

            // Turn the original file descriptor back into a raw fd to avoid ownership problems
            file.into_raw_fd();
            DeviceMemory::import(
                device,
                MemoryAllocateInfo {
                    allocation_size: requirements.layout.size(),
                    memory_type_index,
                    dedicated_allocation: Some(DedicatedAllocation::Image(&image)),
                    export_handle_types: ExternalMemoryHandleTypes::empty(),
                    flags: MemoryAllocateFlags::empty(),
                    ..Default::default()
                },
                crate::memory::MemoryImportInfo::Fd {
                    handle_type: crate::memory::ExternalMemoryHandleType::DmaBuf,
                    file: new_file,
                },
            )
            .unwrap() // TODO: Handle
        };

        let x = MemoryAlloc::new(memory).unwrap();

        debug_assert!(x.offset() % requirements.layout.alignment().as_nonzero() == 0);
        debug_assert!(x.size() == requirements.layout.size());

        let inner = Arc::new(unsafe {
            image
                .bind_memory_unchecked([x])
                .map_err(|(err, _, _)| err)?
        });
        Ok(Arc::new(StorageImage { inner }))
    }
    /// Allows the creation of a simple 2D general purpose image view from `StorageImage`.
    #[inline]
    pub fn general_purpose_image_view(
        allocator: &(impl MemoryAllocator + ?Sized),
        queue: Arc<Queue>,
        size: [u32; 2],
        format: Format,
        usage: ImageUsage,
    ) -> Result<Arc<ImageView<StorageImage>>, ImageError> {
        let dims = ImageDimensions::Dim2d {
            width: size[0],
            height: size[1],
            array_layers: 1,
        };
        let flags = ImageCreateFlags::empty();
        let image_result = StorageImage::with_usage(
            allocator,
            dims,
            format,
            usage,
            flags,
            Some(queue.queue_family_index()),
        );

        match image_result {
            Ok(image) => {
                let image_view = ImageView::new_default(image);
                match image_view {
                    Ok(view) => Ok(view),
                    Err(e) => Err(ImageError::DirectImageViewCreationFailed(e)),
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Exports posix file descriptor for the allocated memory.
    /// Requires `khr_external_memory_fd` and `khr_external_memory` extensions to be loaded.
    #[inline]
    pub fn export_posix_fd(&self) -> Result<File, DeviceMemoryError> {
        let allocation = match self.inner.memory() {
            ImageMemory::Normal(a) => &a[0],
            _ => unreachable!(),
        };

        allocation
            .device_memory()
            .export_fd(ExternalMemoryHandleType::OpaqueFd)
    }

    /// Return the size of the allocated memory (used e.g. with cuda).
    #[inline]
    pub fn mem_size(&self) -> DeviceSize {
        let allocation = match self.inner.memory() {
            ImageMemory::Normal(a) => &a[0],
            _ => unreachable!(),
        };

        allocation.device_memory().allocation_size()
    }
}

#[cfg(target_os = "linux")]
/// Struct that contains a Linux file descriptor for importing, when creating an image. Since a file descriptor is used for each
/// plane in the case of multiplanar images, each fd needs to have an offset and a row pitch in order to interpret the imported data.
pub struct SubresourceData {
    // The file descriptor hanfle of a layer of an image.
    pub fd: RawFd,

    // The byte offset from the start of the plane where the image subresource begins.
    pub offset: u64,

    //  Describes the number of bytes between each row of texels in an image plane.
    pub row_pitch: u64,
}

unsafe impl DeviceOwned for StorageImage {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl ImageAccess for StorageImage {
    #[inline]
    fn inner(&self) -> ImageInner<'_> {
        ImageInner {
            image: &self.inner,
            first_layer: 0,
            num_layers: self.inner.dimensions().array_layers(),
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
}

unsafe impl<P> ImageContent<P> for StorageImage {
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

impl PartialEq for StorageImage {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for StorageImage {}

impl Hash for StorageImage {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{image::view::ImageViewCreationError, memory::allocator::StandardMemoryAllocator};

    #[test]
    fn create() {
        let (device, queue) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let _img = StorageImage::new(
            &memory_allocator,
            ImageDimensions::Dim2d {
                width: 32,
                height: 32,
                array_layers: 1,
            },
            Format::R8G8B8A8_UNORM,
            Some(queue.queue_family_index()),
        )
        .unwrap();
    }

    #[test]
    fn create_general_purpose_image_view() {
        let (device, queue) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let usage =
            ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::COLOR_ATTACHMENT;
        let img_view = StorageImage::general_purpose_image_view(
            &memory_allocator,
            queue,
            [32, 32],
            Format::R8G8B8A8_UNORM,
            usage,
        )
        .unwrap();
        assert_eq!(img_view.image().usage(), usage);
    }

    #[test]
    fn create_general_purpose_image_view_failed() {
        let (device, queue) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);
        // Not valid for image view...
        let usage = ImageUsage::TRANSFER_SRC;
        let img_result = StorageImage::general_purpose_image_view(
            &memory_allocator,
            queue,
            [32, 32],
            Format::R8G8B8A8_UNORM,
            usage,
        );
        assert_eq!(
            img_result,
            Err(ImageError::DirectImageViewCreationFailed(
                ImageViewCreationError::ImageMissingUsage
            ))
        );
    }
}
