// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::device::Device;
use crate::format::ClearValue;
use crate::format::Format;
use crate::format::FormatTy;
use crate::image::sys::ImageCreationError;
use crate::image::sys::UnsafeImage;
use crate::image::traits::ImageAccess;
use crate::image::traits::ImageClearValue;
use crate::image::traits::ImageContent;
use crate::image::ImageCreateFlags;
use crate::image::ImageDescriptorLayouts;
use crate::image::ImageDimensions;
use crate::image::ImageInner;
use crate::image::ImageLayout;
use crate::image::ImageUsage;
use crate::image::SampleCount;
use crate::memory::pool::AllocFromRequirementsFilter;
use crate::memory::pool::AllocLayout;
use crate::memory::pool::MappingRequirement;
use crate::memory::pool::MemoryPool;
use crate::memory::pool::MemoryPoolAlloc;
use crate::memory::pool::PotentialDedicatedAllocation;
use crate::memory::pool::StdMemoryPoolAlloc;
use crate::memory::DedicatedAlloc;
use crate::sync::AccessError;
use crate::sync::Sharing;
use std::hash::Hash;
use std::hash::Hasher;
use std::iter::Empty;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

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
pub struct AttachmentImage<A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>> {
    // Inner implementation.
    image: UnsafeImage,

    // Memory used to back the image.
    memory: A,

    // Format.
    format: Format,

    // Layout to use when the image is used as a framebuffer attachment.
    // Must be either "depth-stencil optimal" or "color optimal".
    attachment_layout: ImageLayout,

    // If true, then the image is in the layout of `attachment_layout` (above). If false, then it
    // is still `Undefined`.
    initialized: AtomicBool,

    // Number of times this image is locked on the GPU side.
    gpu_lock: AtomicUsize,
}

impl AttachmentImage {
    /// Creates a new image with the given dimensions and format.
    ///
    /// Returns an error if the dimensions are too large or if the backend doesn't support this
    /// format as a framebuffer attachment.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        AttachmentImage::new_impl(
            device,
            dimensions,
            1,
            format,
            ImageUsage::none(),
            SampleCount::Sample1,
        )
    }

    /// Same as `new`, but creates an image that can be used as an input attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn input_attachment(
        device: Arc<Device>,
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        let base_usage = ImageUsage {
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(
            device,
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
        device: Arc<Device>,
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        AttachmentImage::new_impl(device, dimensions, 1, format, ImageUsage::none(), samples)
    }

    /// Same as `multisampled`, but creates an image that can be used as an input attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn multisampled_input_attachment(
        device: Arc<Device>,
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        let base_usage = ImageUsage {
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, 1, format, base_usage, samples)
    }

    /// Same as `new`, but lets you specify additional usages.
    ///
    /// The `color_attachment` or `depth_stencil_attachment` usages are automatically added based
    /// on the format of the usage. Therefore the `usage` parameter allows you specify usages in
    /// addition to these two.
    #[inline]
    pub fn with_usage(
        device: Arc<Device>,
        dimensions: [u32; 2],
        format: Format,
        usage: ImageUsage,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        AttachmentImage::new_impl(device, dimensions, 1, format, usage, SampleCount::Sample1)
    }

    /// Same as `with_usage`, but creates a multisampled image.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of samples if you
    /// > want a regular image.
    #[inline]
    pub fn multisampled_with_usage(
        device: Arc<Device>,
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
        usage: ImageUsage,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        AttachmentImage::new_impl(device, dimensions, 1, format, usage, samples)
    }

    /// Same as `multisampled_with_usage`, but creates an image with multiple layers.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of layers if you
    /// > want a regular image.
    #[inline]
    pub fn multisampled_with_usage_with_layers(
        device: Arc<Device>,
        dimensions: [u32; 2],
        array_layers: u32,
        samples: SampleCount,
        format: Format,
        usage: ImageUsage,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        AttachmentImage::new_impl(device, dimensions, array_layers, format, usage, samples)
    }

    /// Same as `new`, except that the image can later be sampled.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn sampled(
        device: Arc<Device>,
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        let base_usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(
            device,
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
        device: Arc<Device>,
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        let base_usage = ImageUsage {
            sampled: true,
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(
            device,
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
        device: Arc<Device>,
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        let base_usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, 1, format, base_usage, samples)
    }

    /// Same as `sampled_multisampled`, but creates an image that can be used as an input
    /// attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn sampled_multisampled_input_attachment(
        device: Arc<Device>,
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        let base_usage = ImageUsage {
            sampled: true,
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, 1, format, base_usage, samples)
    }

    /// Same as `new`, except that the image will be transient.
    ///
    /// A transient image is special because its content is undefined outside of a render pass.
    /// This means that the implementation has the possibility to not allocate any memory for it.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn transient(
        device: Arc<Device>,
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        let base_usage = ImageUsage {
            transient_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(
            device,
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
        device: Arc<Device>,
        dimensions: [u32; 2],
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        let base_usage = ImageUsage {
            transient_attachment: true,
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(
            device,
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
        device: Arc<Device>,
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        let base_usage = ImageUsage {
            transient_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, 1, format, base_usage, samples)
    }

    /// Same as `transient_multisampled`, but creates an image that can be used as an input
    /// attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn transient_multisampled_input_attachment(
        device: Arc<Device>,
        dimensions: [u32; 2],
        samples: SampleCount,
        format: Format,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        let base_usage = ImageUsage {
            transient_attachment: true,
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, 1, format, base_usage, samples)
    }

    // All constructors dispatch to this one.
    fn new_impl(
        device: Arc<Device>,
        dimensions: [u32; 2],
        array_layers: u32,
        format: Format,
        base_usage: ImageUsage,
        samples: SampleCount,
    ) -> Result<Arc<AttachmentImage>, ImageCreationError> {
        // TODO: check dimensions against the max_framebuffer_width/height/layers limits

        let is_depth = match format.ty() {
            FormatTy::Depth => true,
            FormatTy::DepthStencil => true,
            FormatTy::Stencil => true,
            FormatTy::Compressed => panic!(),
            _ => false,
        };

        let usage = ImageUsage {
            color_attachment: !is_depth,
            depth_stencil_attachment: is_depth,
            ..base_usage
        };

        let (image, mem_reqs) = unsafe {
            let dims = ImageDimensions::Dim2d {
                width: dimensions[0],
                height: dimensions[1],
                array_layers,
            };

            UnsafeImage::new(
                device.clone(),
                usage,
                format,
                ImageCreateFlags::none(),
                dims,
                samples,
                1,
                Sharing::Exclusive::<Empty<u32>>,
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

        Ok(Arc::new(AttachmentImage {
            image,
            memory,
            format,
            attachment_layout: if is_depth {
                ImageLayout::DepthStencilAttachmentOptimal
            } else {
                ImageLayout::ColorAttachmentOptimal
            },
            initialized: AtomicBool::new(false),
            gpu_lock: AtomicUsize::new(0),
        }))
    }
}

impl<A> AttachmentImage<A> {
    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> [u32; 2] {
        let dims = self.image.dimensions();
        [dims.width(), dims.height()]
    }
}

unsafe impl<A> ImageAccess for AttachmentImage<A> {
    #[inline]
    fn inner(&self) -> ImageInner {
        ImageInner {
            image: &self.image,
            first_layer: 0,
            num_layers: self.image.dimensions().array_layers() as usize,
            first_mipmap_level: 0,
            num_mipmap_levels: 1,
        }
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
            storage_image: ImageLayout::ShaderReadOnlyOptimal,
            combined_image_sampler: ImageLayout::ShaderReadOnlyOptimal,
            sampled_image: ImageLayout::ShaderReadOnlyOptimal,
            input_attachment: ImageLayout::ShaderReadOnlyOptimal,
        })
    }

    #[inline]
    fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool {
        false
    }

    #[inline]
    fn conflicts_image(&self, other: &dyn ImageAccess) -> bool {
        self.conflict_key() == other.conflict_key()
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.image.key()
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, expected_layout: ImageLayout) -> Result<(), AccessError> {
        if expected_layout != self.attachment_layout && expected_layout != ImageLayout::Undefined {
            if self.initialized.load(Ordering::SeqCst) {
                return Err(AccessError::UnexpectedImageLayout {
                    requested: expected_layout,
                    allowed: self.attachment_layout,
                });
            } else {
                return Err(AccessError::UnexpectedImageLayout {
                    requested: expected_layout,
                    allowed: ImageLayout::Undefined,
                });
            }
        }

        if expected_layout != ImageLayout::Undefined {
            if !self.initialized.load(Ordering::SeqCst) {
                return Err(AccessError::ImageNotInitialized {
                    requested: expected_layout,
                });
            }
        }

        if self
            .gpu_lock
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap_or_else(|e| e)
            == 0
        {
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
        if let Some(new_layout) = new_layout {
            debug_assert_eq!(new_layout, self.attachment_layout);
            self.initialized.store(true, Ordering::SeqCst);
        }

        let prev_val = self.gpu_lock.fetch_sub(1, Ordering::SeqCst);
        debug_assert!(prev_val >= 1);
    }

    #[inline]
    unsafe fn layout_initialized(&self) {
        self.initialized.store(true, Ordering::SeqCst);
    }

    #[inline]
    fn is_layout_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    #[inline]
    fn current_miplevels_access(&self) -> std::ops::Range<u32> {
        0..self.mipmap_levels()
    }

    #[inline]
    fn current_layer_levels_access(&self) -> std::ops::Range<u32> {
        0..1
    }
}

unsafe impl<A> ImageClearValue<ClearValue> for Arc<AttachmentImage<A>> {
    #[inline]
    fn decode(&self, value: ClearValue) -> Option<ClearValue> {
        Some(self.format.decode_clear_value(value))
    }
}

unsafe impl<P, A> ImageContent<P> for Arc<AttachmentImage<A>> {
    #[inline]
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

impl<A> PartialEq for AttachmentImage<A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ImageAccess::inner(self) == ImageAccess::inner(other)
    }
}

impl<A> Eq for AttachmentImage<A> {}

impl<A> Hash for AttachmentImage<A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        ImageAccess::inner(self).hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::AttachmentImage;
    use crate::format::Format;

    #[test]
    fn create_regular() {
        let (device, _) = gfx_dev_and_queue!();
        let _img = AttachmentImage::new(device, [32, 32], Format::R8G8B8A8Unorm).unwrap();
    }

    #[test]
    fn create_transient() {
        let (device, _) = gfx_dev_and_queue!();
        let _img = AttachmentImage::transient(device, [32, 32], Format::R8G8B8A8Unorm).unwrap();
    }

    #[test]
    fn d16_unorm_always_supported() {
        let (device, _) = gfx_dev_and_queue!();
        let _img = AttachmentImage::new(device, [32, 32], Format::D16Unorm).unwrap();
    }
}
