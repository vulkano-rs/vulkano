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
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use buffer::BufferAccess;
use device::Device;
use format::ClearValue;
use format::Format;
use format::FormatDesc;
use format::FormatTy;
use image::Dimensions;
use image::ImageDimensions;
use image::ImageInner;
use image::ImageLayout;
use image::ImageUsage;
use image::ViewType;
use image::sys::ImageCreationError;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use image::traits::ImageAccess;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::traits::ImageViewAccess;
use memory::DedicatedAlloc;
use memory::pool::AllocFromRequirementsFilter;
use memory::pool::AllocLayout;
use memory::pool::MappingRequirement;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::PotentialDedicatedAllocation;
use memory::pool::StdMemoryPoolAlloc;
use sync::AccessError;
use sync::Sharing;

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
/// performances.
///
// TODO: forbid reading transient images outside render passes?
#[derive(Debug)]
pub struct AttachmentImage<F = Format, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>> {
    // Inner implementation.
    image: UnsafeImage,

    // We maintain a view of the whole image since we will need it when rendering.
    view: UnsafeImageView,

    // Memory used to back the image.
    memory: A,

    // Format.
    format: F,

    // Layout to use when the image is used as a framebuffer attachment.
    // Must be either "depth-stencil optimal" or "color optimal".
    attachment_layout: ImageLayout,

    // If true, then the image is in the layout of `attachment_layout` (above). If false, then it
    // is still `Undefined`.
    initialized: AtomicBool,

    // Number of times this image is locked on the GPU side.
    gpu_lock: AtomicUsize,
}

impl<F> AttachmentImage<F> {
    /// Creates a new image with the given dimensions and format.
    ///
    /// Returns an error if the dimensions are too large or if the backend doesn't support this
    /// format as a framebuffer attachment.
    #[inline]
    pub fn new(device: Arc<Device>, dimensions: [u32; 2], format: F)
               -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        AttachmentImage::new_impl(device, dimensions, format, ImageUsage::none(), 1)
    }

    /// Same as `new`, but creates an image that can be used as an input attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn input_attachment(device: Arc<Device>, dimensions: [u32; 2], format: F)
                            -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = ImageUsage {
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage, 1)
    }

    /// Same as `new`, but creates a multisampled image.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of samples if you
    /// > want a regular image.
    #[inline]
    pub fn multisampled(device: Arc<Device>, dimensions: [u32; 2], samples: u32, format: F)
                        -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        AttachmentImage::new_impl(device, dimensions, format, ImageUsage::none(), samples)
    }

    /// Same as `multisampled`, but creates an image that can be used as an input attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn multisampled_input_attachment(device: Arc<Device>, dimensions: [u32; 2], samples: u32,
                                         format: F)
                                         -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = ImageUsage {
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage, samples)
    }

    /// Same as `new`, but lets you specify additional usages.
    ///
    /// The `color_attachment` or `depth_stencil_attachment` usages are automatically added based
    /// on the format of the usage. Therefore the `usage` parameter allows you specify usages in
    /// addition to these two.
    #[inline]
    pub fn with_usage(device: Arc<Device>, dimensions: [u32; 2], format: F, usage: ImageUsage)
                      -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        AttachmentImage::new_impl(device, dimensions, format, usage, 1)
    }

    /// Same as `with_usage`, but creates a multisampled image.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of samples if you
    /// > want a regular image.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn multisampled_with_usage(device: Arc<Device>, dimensions: [u32; 2], samples: u32,
                                   format: F, usage: ImageUsage)
                                   -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        AttachmentImage::new_impl(device, dimensions, format, usage, samples)
    }

    /// Same as `new`, except that the image can later be sampled.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn sampled(device: Arc<Device>, dimensions: [u32; 2], format: F)
                   -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage, 1)
    }

    /// Same as `sampled`, except that the image can be used as an input attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn sampled_input_attachment(device: Arc<Device>, dimensions: [u32; 2], format: F)
                                    -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = ImageUsage {
            sampled: true,
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage, 1)
    }

    /// Same as `sampled`, but creates a multisampled image.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of samples if you
    /// > want a regular image.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn sampled_multisampled(device: Arc<Device>, dimensions: [u32; 2], samples: u32, format: F)
                                -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage, samples)
    }

    /// Same as `sampled_multisampled`, but creates an image that can be used as an input
    /// attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn sampled_multisampled_input_attachment(
        device: Arc<Device>, dimensions: [u32; 2], samples: u32, format: F)
        -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = ImageUsage {
            sampled: true,
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage, samples)
    }

    /// Same as `new`, except that the image will be transient.
    ///
    /// A transient image is special because its content is undefined outside of a render pass.
    /// This means that the implementation has the possibility to not allocate any memory for it.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn transient(device: Arc<Device>, dimensions: [u32; 2], format: F)
                     -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = ImageUsage {
            transient_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage, 1)
    }

    /// Same as `transient`, except that the image can be used as an input attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `with_usage`.
    #[inline]
    pub fn transient_input_attachment(device: Arc<Device>, dimensions: [u32; 2], format: F)
                                      -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = ImageUsage {
            transient_attachment: true,
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage, 1)
    }

    /// Same as `transient`, but creates a multisampled image.
    ///
    /// > **Note**: You can also use this function and pass `1` for the number of samples if you
    /// > want a regular image.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn transient_multisampled(device: Arc<Device>, dimensions: [u32; 2], samples: u32,
                                  format: F)
                                  -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = ImageUsage {
            transient_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage, samples)
    }

    /// Same as `transient_multisampled`, but creates an image that can be used as an input
    /// attachment.
    ///
    /// > **Note**: This function is just a convenient shortcut for `multisampled_with_usage`.
    #[inline]
    pub fn transient_multisampled_input_attachment(
        device: Arc<Device>, dimensions: [u32; 2], samples: u32, format: F)
        -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = ImageUsage {
            transient_attachment: true,
            input_attachment: true,
            ..ImageUsage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage, samples)
    }

    // All constructors dispatch to this one.
    fn new_impl(device: Arc<Device>, dimensions: [u32; 2], format: F, base_usage: ImageUsage,
                samples: u32)
                -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        // TODO: check dimensions against the max_framebuffer_width/height/layers limits

        let is_depth = match format.format().ty() {
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
                array_layers: 1,
                cubemap_compatible: false,
            };

            UnsafeImage::new(device.clone(),
                             usage,
                             format.format(),
                             dims,
                             samples,
                             1,
                             Sharing::Exclusive::<Empty<u32>>,
                             false,
                             false)?
        };

        let mem = MemoryPool::alloc_from_requirements(&Device::standard_pool(&device),
                                    &mem_reqs,
                                    AllocLayout::Optimal,
                                    MappingRequirement::DoNotMap,
                                    DedicatedAlloc::Image(&image),
                                    |t| if t.is_device_local() {
                                        AllocFromRequirementsFilter::Preferred
                                    } else {
                                        AllocFromRequirementsFilter::Allowed
                                    })?;
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        unsafe {
            image.bind_memory(mem.memory(), mem.offset())?;
        }

        let view = unsafe { UnsafeImageView::raw(&image, ViewType::Dim2d, 0 .. 1, 0 .. 1)? };

        Ok(Arc::new(AttachmentImage {
                        image: image,
                        view: view,
                        memory: mem,
                        format: format,
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

impl<F, A> AttachmentImage<F, A> {
    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> [u32; 2] {
        let dims = self.image.dimensions();
        [dims.width(), dims.height()]
    }
}

unsafe impl<F, A> ImageAccess for AttachmentImage<F, A>
    where F: 'static + Send + Sync
{
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
    fn conflicts_buffer(&self, other: &BufferAccess) -> bool {
        false
    }

    #[inline]
    fn conflicts_image(&self, other: &ImageAccess) -> bool {
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
                return Err(AccessError::ImageNotInitialized { requested: expected_layout });
            }
        }

        if self.gpu_lock.compare_and_swap(0, 1, Ordering::SeqCst) == 0 {
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
}

unsafe impl<F, A> ImageClearValue<F::ClearValue> for Arc<AttachmentImage<F, A>>
    where F: FormatDesc + 'static + Send + Sync
{
    #[inline]
    fn decode(&self, value: F::ClearValue) -> Option<ClearValue> {
        Some(self.format.decode_clear_value(value))
    }
}

unsafe impl<P, F, A> ImageContent<P> for Arc<AttachmentImage<F, A>>
    where F: 'static + Send + Sync
{
    #[inline]
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

unsafe impl<F, A> ImageViewAccess for AttachmentImage<F, A>
    where F: 'static + Send + Sync
{
    #[inline]
    fn parent(&self) -> &ImageAccess {
        self
    }

    #[inline]
    fn dimensions(&self) -> Dimensions {
        let dims = self.image.dimensions();
        Dimensions::Dim2d {
            width: dims.width(),
            height: dims.height(),
        }
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

#[cfg(test)]
mod tests {
    use super::AttachmentImage;
    use format::Format;

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
