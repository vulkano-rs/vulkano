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
    ImageAccess, ImageAspects, ImageDescriptorLayouts, ImageError, ImageLayout, ImageUsage,
    SampleCount,
};
use crate::{
    device::{Device, DeviceOwned},
    format::Format,
    image::{sys::ImageCreateInfo, ImageCreateFlags, ImageDimensions, ImageFormatInfo},
    memory::{
        allocator::{
            AllocationCreateInfo, AllocationType, MemoryAllocatePreference, MemoryAllocator,
            MemoryUsage,
        },
        is_aligned, DedicatedAllocation, DeviceMemoryError, ExternalMemoryHandleType,
        ExternalMemoryHandleTypes,
    },
    DeviceSize,
};
use std::{
    fs::File,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// ImageAccess whose purpose is to be used as a framebuffer attachment.
///
/// The image is always two-dimensional and has only one mipmap, but it can have any kind of
/// format. Trying to use a format that the backend doesn't support for rendering will result in
/// an error being returned when creating the image. Once you have an `AttachmentImage`, you are
/// guaranteed that you will be able to draw on it.
///
/// The template parameter of `AttachmentImage` is a type that describes the format of the image.
///
/// # Regular vs transient
///
/// Calling `AttachmentImage::new` will create a regular image, while calling
/// `AttachmentImage::transient` will create a *transient* image. Transient image are only
/// relevant for images that serve as attachments, so `AttachmentImage` is the only type of
/// image in vulkano that provides a shortcut for this.
///
/// A transient image is a special kind of image whose content is undefined outside of render
/// passes. Once you finish drawing, reading from it will returned undefined data (which can be
/// either valid or garbage, depending on the implementation).
///
/// This gives a hint to the Vulkan implementation that it is possible for the image's content to
/// live exclusively in some cache memory, and that no real memory has to be allocated for it.
///
/// In other words, if you are going to read from the image after drawing to it, use a regular
/// image. If you don't need to read from it (for example if it's some kind of intermediary color,
/// or a depth buffer that is only used once) then use a transient image as it may improve
/// performance.
///
// TODO: forbid reading transient images outside render passes?
#[derive(Debug)]
pub struct AttachmentImage {
    inner: Arc<Image>,

    // Layout to use when the image is used as a framebuffer attachment.
    // Must be either "depth-stencil optimal" or "color optimal".
    attachment_layout: ImageLayout,

    // If true, then the image is in the layout of `attachment_layout` (above). If false, then it
    // is still `Undefined`.
    layout_initialized: AtomicBool,
}

impl AttachmentImage {
    /// Creates a new image with the given dimensions and format.
    ///
    /// Returns an error if the dimensions are too large or if the backend doesn't support this
    /// format as a framebuffer attachment.
    #[inline]
    pub fn new(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        AttachmentImage::new_impl(
            allocator,
            dimensions,
            1,
            format,
            ImageUsage::empty(),
            SampleCount::Sample1,
        )
    }

    /// Same as `new`, but creates an image that can be used as an input attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn input_attachment(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let base_usage = ImageUsage::INPUT_ATTACHMENT;

        AttachmentImage::new_impl(
            allocator,
            dimensions,
            1,
            format,
            base_usage,
            SampleCount::Sample1,
        )
    }

    /// Same as `new`, but creates a multisampled image.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of samples if you
    /// > want a regular image.
    #[inline]
    pub fn multisampled(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        AttachmentImage::new_impl(
            allocator,
            dimensions,
            1,
            format,
            ImageUsage::empty(),
            samples,
        )
    }

    /// Same as `multisampled`, but creates an image that can be used as an input attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn multisampled_input_attachment(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let base_usage = ImageUsage::INPUT_ATTACHMENT;

        AttachmentImage::new_impl(allocator, dimensions, 1, format, base_usage, samples)
    }

    /// Same as `new`, but lets you specify additional usages.
    ///
    /// The `color_attachment` or `depth_stencil_attachment` usages are automatically added based
    /// on the format of the usage. Therefore the `usage` parameter allows you specify usages in
    /// addition to these two.
    #[inline]
    pub fn with_usage(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        format: Format,
        usage: ImageUsage,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        AttachmentImage::new_impl(
            allocator,
            dimensions,
            1,
            format,
            usage,
            SampleCount::Sample1,
        )
    }

    /// Same as `with_usage`, but creates a multisampled image.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of samples if you
    /// > want a regular image.
    #[inline]
    pub fn multisampled_with_usage(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
        usage: ImageUsage,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        AttachmentImage::new_impl(allocator, dimensions, 1, format, usage, samples)
    }

    /// Same as `multisampled_with_usage`, but creates an image with multiple layers.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of layers if you
    /// > want a regular image.
    #[inline]
    pub fn multisampled_with_usage_with_layers(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        array_layers: u32,
        samples: SampleCount,
        format: Format,
        usage: ImageUsage,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        AttachmentImage::new_impl(allocator, dimensions, array_layers, format, usage, samples)
    }

    /// Same as `new`, except that the image can later be sampled.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn sampled(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let base_usage = ImageUsage::SAMPLED;

        AttachmentImage::new_impl(
            allocator,
            dimensions,
            1,
            format,
            base_usage,
            SampleCount::Sample1,
        )
    }

    /// Same as `sampled`, except that the image can be used as an input attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn sampled_input_attachment(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let base_usage = ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT;

        AttachmentImage::new_impl(
            allocator,
            dimensions,
            1,
            format,
            base_usage,
            SampleCount::Sample1,
        )
    }

    /// Same as `sampled`, but creates a multisampled image.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of samples if you
    /// > want a regular image.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn sampled_multisampled(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let base_usage = ImageUsage::SAMPLED;

        AttachmentImage::new_impl(allocator, dimensions, 1, format, base_usage, samples)
    }

    /// Same as `sampled_multisampled`, but creates an image that can be used as an input
    /// attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn sampled_multisampled_input_attachment(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let base_usage = ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT;

        AttachmentImage::new_impl(allocator, dimensions, 1, format, base_usage, samples)
    }

    /// Same as `new`, except that the image will be transient.
    ///
    /// A transient image is special because its content is undefined outside of a render pass.
    /// This means that the implementation has the possibility to not allocate any memory for it.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn transient(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let base_usage = ImageUsage::TRANSIENT_ATTACHMENT;

        AttachmentImage::new_impl(
            allocator,
            dimensions,
            1,
            format,
            base_usage,
            SampleCount::Sample1,
        )
    }

    /// Same as `transient`, except that the image can be used as an input attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn transient_input_attachment(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let base_usage = ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT;

        AttachmentImage::new_impl(
            allocator,
            dimensions,
            1,
            format,
            base_usage,
            SampleCount::Sample1,
        )
    }

    /// Same as `transient`, but creates a multisampled image.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of samples if you
    /// > want a regular image.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn transient_multisampled(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let base_usage = ImageUsage::TRANSIENT_ATTACHMENT;

        AttachmentImage::new_impl(allocator, dimensions, 1, format, base_usage, samples)
    }

    /// Same as `transient_multisampled`, but creates an image that can be used as an input
    /// attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn transient_multisampled_input_attachment(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let base_usage = ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT;

        AttachmentImage::new_impl(allocator, dimensions, 1, format, base_usage, samples)
    }

    // All constructors dispatch to this one.
    fn new_impl(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        array_layers: u32,
        format: Format,
        mut usage: ImageUsage,
        samples: SampleCount,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let physical_device = allocator.device().physical_device();
        let device_properties = physical_device.properties();

        if dimensions[0] > device_properties.max_framebuffer_height {
            panic!("AttachmentImage height exceeds physical device's max_framebuffer_height");
        }
        if dimensions[1] > device_properties.max_framebuffer_width {
            panic!("AttachmentImage width exceeds physical device's max_framebuffer_width");
        }
        if array_layers > device_properties.max_framebuffer_layers {
            panic!("AttachmentImage layer count exceeds physical device's max_framebuffer_layers");
        }

        let aspects = format.aspects();
        let is_depth_stencil = aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL);

        if is_depth_stencil {
            usage -= ImageUsage::COLOR_ATTACHMENT;
            usage |= ImageUsage::DEPTH_STENCIL_ATTACHMENT;
        } else {
            usage |= ImageUsage::COLOR_ATTACHMENT;
            usage -= ImageUsage::DEPTH_STENCIL_ATTACHMENT;
        }

        if format.compression().is_some() {
            panic!() // TODO: message?
        }

        let raw_image = RawImage::new(
            allocator.device().clone(),
            ImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: dimensions[0],
                    height: dimensions[1],
                    array_layers,
                },
                format: Some(format),
                samples,
                usage,
                ..Default::default()
            },
        )?;
        let requirements = raw_image.memory_requirements()[0];
        let res = unsafe {
            allocator.allocate_unchecked(
                requirements,
                AllocationType::NonLinear,
                AllocationCreateInfo {
                    usage: MemoryUsage::DeviceOnly,
                    allocate_preference: MemoryAllocatePreference::Unknown,
                    _ne: crate::NonExhaustive(()),
                },
                Some(DedicatedAllocation::Image(&raw_image)),
            )
        };

        match res {
            Ok(alloc) => {
                debug_assert!(is_aligned(alloc.offset(), requirements.layout.alignment()));
                debug_assert!(alloc.size() == requirements.layout.size());

                let inner = Arc::new(
                    unsafe { raw_image.bind_memory_unchecked([alloc]) }
                        .map_err(|(err, _, _)| err)?,
                );

                Ok(Arc::new(AttachmentImage {
                    inner,
                    attachment_layout: if is_depth_stencil {
                        ImageLayout::DepthStencilAttachmentOptimal
                    } else {
                        ImageLayout::ColorAttachmentOptimal
                    },
                    layout_initialized: AtomicBool::new(false),
                }))
            }
            Err(err) => Err(err.into()),
        }
    }

    pub fn new_with_exportable_fd(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        array_layers: u32,
        format: Format,
        mut usage: ImageUsage,
        samples: SampleCount,
    ) -> Result<Arc<AttachmentImage>, ImageError> {
        let physical_device = allocator.device().physical_device();
        let device_properties = physical_device.properties();

        if dimensions[0] > device_properties.max_framebuffer_height {
            panic!("AttachmentImage height exceeds physical device's max_framebuffer_height");
        }
        if dimensions[1] > device_properties.max_framebuffer_width {
            panic!("AttachmentImage width exceeds physical device's max_framebuffer_width");
        }
        if array_layers > device_properties.max_framebuffer_layers {
            panic!("AttachmentImage layer count exceeds physical device's max_framebuffer_layers");
        }

        let aspects = format.aspects();
        let is_depth_stencil = aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL);

        if is_depth_stencil {
            usage -= ImageUsage::COLOR_ATTACHMENT;
            usage |= ImageUsage::DEPTH_STENCIL_ATTACHMENT;
        } else {
            usage |= ImageUsage::COLOR_ATTACHMENT;
            usage -= ImageUsage::DEPTH_STENCIL_ATTACHMENT;
        }

        let external_memory_properties = allocator
            .device()
            .physical_device()
            .image_format_properties(ImageFormatInfo {
                flags: ImageCreateFlags::MUTABLE_FORMAT,
                format: Some(format),
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
                flags: ImageCreateFlags::MUTABLE_FORMAT,
                dimensions: ImageDimensions::Dim2d {
                    width: dimensions[0],
                    height: dimensions[1],
                    array_layers,
                },
                format: Some(format),
                samples,
                usage,
                external_memory_handle_types,
                ..Default::default()
            },
        )?;
        let requirements = raw_image.memory_requirements()[0];
        let memory_type_index = allocator
            .find_memory_type_index(
                requirements.memory_type_bits,
                MemoryUsage::DeviceOnly.into(),
            )
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
                debug_assert!(is_aligned(alloc.offset(), requirements.layout.alignment()));
                debug_assert!(alloc.size() == requirements.layout.size());

                let inner = Arc::new(unsafe {
                    raw_image
                        .bind_memory_unchecked([alloc])
                        .map_err(|(err, _, _)| err)?
                });

                Ok(Arc::new(AttachmentImage {
                    inner,
                    attachment_layout: if is_depth_stencil {
                        ImageLayout::DepthStencilAttachmentOptimal
                    } else {
                        ImageLayout::ColorAttachmentOptimal
                    },
                    layout_initialized: AtomicBool::new(false),
                }))
            }
            Err(err) => Err(err.into()),
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

unsafe impl ImageAccess for AttachmentImage {
    #[inline]
    fn inner(&self) -> &Arc<Image> {
        &self.inner
    }

    #[inline]
    fn initial_layout_requirement(&self) -> ImageLayout {
        self.attachment_layout
    }

    #[inline]
    fn final_layout_requirement(&self) -> ImageLayout {
        self.attachment_layout
    }

    #[inline]
    fn descriptor_layouts(&self) -> Option<ImageDescriptorLayouts> {
        Some(ImageDescriptorLayouts {
            storage_image: ImageLayout::General,
            combined_image_sampler: ImageLayout::ShaderReadOnlyOptimal,
            sampled_image: ImageLayout::ShaderReadOnlyOptimal,
            input_attachment: ImageLayout::ShaderReadOnlyOptimal,
        })
    }

    #[inline]
    unsafe fn layout_initialized(&self) {
        self.layout_initialized.store(true, Ordering::SeqCst);
    }

    #[inline]
    fn is_layout_initialized(&self) -> bool {
        self.layout_initialized.load(Ordering::SeqCst)
    }
}

unsafe impl DeviceOwned for AttachmentImage {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<P> ImageContent<P> for AttachmentImage {
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

impl PartialEq for AttachmentImage {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for AttachmentImage {}

impl Hash for AttachmentImage {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::StandardMemoryAllocator;

    #[test]
    fn create_regular() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let _img =
            AttachmentImage::new(&memory_allocator, [32, 32], Format::R8G8B8A8_UNORM).unwrap();
    }

    #[test]
    fn create_transient() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let _img = AttachmentImage::transient(&memory_allocator, [32, 32], Format::R8G8B8A8_UNORM)
            .unwrap();
    }

    #[test]
    fn d16_unorm_always_supported() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let _img = AttachmentImage::new(&memory_allocator, [32, 32], Format::D16_UNORM).unwrap();
    }
}
