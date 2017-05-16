// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use device::Queue;
use format::ClearValue;
use format::Format;
use format::FormatDesc;
use image::ImageDimensions;
use image::Dimensions;
use image::ViewType;
use image::traits::ImageAccess;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::traits::ImageViewAccess;
use image::traits::Image;
use image::traits::ImageView;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use swapchain::Swapchain;

use OomError;

/// An image that is part of a swapchain.
///
/// Creating a `SwapchainImage` is automatically done when creating a swapchain.
///
/// A swapchain image is special in the sense that it can only be used after being acquired by
/// calling the `acquire` method on the swapchain. You have no way to know in advance which
/// swapchain image is going to be acquired, so you should keep all of them alive.
///
/// After a swapchain image has been acquired, you are free to perform all the usual operations
/// on it. When you are done you can then *present* the image (by calling the corresponding
/// method on the swapchain), which will have the effect of showing the content of the image to
/// the screen. Once an image has been presented, it can no longer be used unless it is acquired
/// again.
// TODO: #[derive(Debug)] (needs https://github.com/aturon/crossbeam/issues/62)
pub struct SwapchainImage {
    image: UnsafeImage,
    view: UnsafeImageView,
    format: Format,
    swapchain: Arc<Swapchain>,
    id: u32,
}

impl SwapchainImage {
    /// Builds a `SwapchainImage` from raw components.
    ///
    /// This is an internal method that you shouldn't call.
    pub unsafe fn from_raw(image: UnsafeImage, format: Format, swapchain: &Arc<Swapchain>, id: u32)
                           -> Result<Arc<SwapchainImage>, OomError>
    {
        let view = try!(UnsafeImageView::raw(&image, ViewType::Dim2d, 0 .. 1, 0 .. 1));

        Ok(Arc::new(SwapchainImage {
            image: image,
            view: view,
            format: format,
            swapchain: swapchain.clone(),
            id: id,
        }))
    }

    /// Returns the dimensions of the image.
    ///
    /// A `SwapchainImage` is always two-dimensional.
    #[inline]
    pub fn dimensions(&self) -> [u32; 2] {
        let dims = self.image.dimensions();
        [dims.width(), dims.height()]
    }

    /// Returns the format of the image.
    // TODO: return `ColorFormat` or something like this instead, for stronger typing
    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns the swapchain this image belongs to.
    #[inline]
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
    }
}

unsafe impl ImageAccess for SwapchainImage {
    #[inline]
    fn inner(&self) -> &UnsafeImage {
        &self.image
    }

    #[inline]
    fn initial_layout_requirement(&self) -> Layout {
        Layout::PresentSrc
    }

    #[inline]
    fn final_layout_requirement(&self) -> Layout {
        Layout::PresentSrc
    }

    #[inline]
    fn conflict_key(&self, _: u32, _: u32, _: u32, _: u32) -> u64 {
        self.image.key()
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, _: &Queue) -> bool {
        // Swapchain image are only accessible after being acquired.
        false
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
    }
}

unsafe impl ImageClearValue<<Format as FormatDesc>::ClearValue> for SwapchainImage
{
    #[inline]
    fn decode(&self, value: <Format as FormatDesc>::ClearValue) -> Option<ClearValue> {
        Some(self.format.decode_clear_value(value))
    }
}

unsafe impl<P> ImageContent<P> for SwapchainImage {
    #[inline]
    fn matches_format(&self) -> bool {
        true        // FIXME:
    }
}

unsafe impl ImageViewAccess for SwapchainImage {
    #[inline]
    fn parent(&self) -> &ImageAccess {
        self
    }

    #[inline]
    fn dimensions(&self) -> Dimensions {
        let dims = self.image.dimensions();
        Dimensions::Dim2d { width: dims.width(), height: dims.height() }
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

unsafe impl Image for SwapchainImage {
    type Access = SwapchainImage;

    #[inline]
    fn access(self) -> Self::Access {
        self
    }

    #[inline]
    fn format(&self) -> Format {
        self.image.format()
    }

    #[inline]
    fn samples(&self) -> u32 {
        self.image.samples()
    }

    #[inline]
    fn dimensions(&self) -> ImageDimensions {
        self.image.dimensions()
    }
}

unsafe impl ImageView for SwapchainImage {
    type Access = SwapchainImage;

    fn access(self) -> Self::Access {
        self
    }
}

unsafe impl Image for Arc<SwapchainImage> {
    type Access = Arc<SwapchainImage>;

    #[inline]
    fn access(self) -> Self::Access {
        self
    }

    #[inline]
    fn format(&self) -> Format {
        self.image.format()
    }

    #[inline]
    fn samples(&self) -> u32 {
        self.image.samples()
    }

    #[inline]
    fn dimensions(&self) -> ImageDimensions {
        self.image.dimensions()
    }
}

unsafe impl ImageView for Arc<SwapchainImage> {
    type Access = Arc<SwapchainImage>;

    fn access(self) -> Self::Access {
        self
    }
}
