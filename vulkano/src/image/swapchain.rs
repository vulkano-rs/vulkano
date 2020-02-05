// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use buffer::BufferAccess;
use format::ClearValue;
use format::Format;
use format::FormatDesc;
use image::Dimensions;
use image::ImageInner;
use image::ImageLayout;
use image::ViewType;
use image::sys::UnsafeImageView;
use image::traits::ImageAccess;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::traits::ImageViewAccess;
use swapchain::Swapchain;
use sync::AccessError;

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
// TODO: #[derive(Debug)]
pub struct SwapchainImage<W> {
    swapchain: Arc<Swapchain<W>>,
    image_offset: usize,
    view: UnsafeImageView,
}

impl<W> SwapchainImage<W> {
    /// Builds a `SwapchainImage` from raw components.
    ///
    /// This is an internal method that you shouldn't call.
    pub unsafe fn from_raw(swapchain: Arc<Swapchain<W>>, id: usize)
                           -> Result<Arc<SwapchainImage<W>>, OomError> {
        let image = swapchain.raw_image(id).unwrap();
        let view = UnsafeImageView::raw(&image.image, ViewType::Dim2d, 0 .. 1, 0 .. 1)?;

        Ok(Arc::new(SwapchainImage {
                        swapchain: swapchain.clone(),
                        image_offset: id,
                        view: view,
                    }))
    }

    /// Returns the dimensions of the image.
    ///
    /// A `SwapchainImage` is always two-dimensional.
    #[inline]
    pub fn dimensions(&self) -> [u32; 2] {
        let dims = self.my_image().image.dimensions();
        [dims.width(), dims.height()]
    }

    /// Returns the swapchain this image belongs to.
    #[inline]
    pub fn swapchain(&self) -> &Arc<Swapchain<W>> {
        &self.swapchain
    }

    #[inline]
    fn my_image(&self) -> ImageInner {
        self.swapchain.raw_image(self.image_offset).unwrap()
    }

    #[inline]
    fn layout_initialized(&self) {
        self.swapchain.image_layout_initialized(self.image_offset);
    }

    #[inline]
    fn is_layout_initialized(&self) -> bool {
       self.swapchain.is_image_layout_initialized(self.image_offset)
    }
}

unsafe impl<W> ImageAccess for SwapchainImage<W> {
    #[inline]
    fn inner(&self) -> ImageInner {
        self.my_image()
    }

    #[inline]
    fn initial_layout_requirement(&self) -> ImageLayout {
        ImageLayout::PresentSrc
    }

    #[inline]
    fn final_layout_requirement(&self) -> ImageLayout {
        ImageLayout::PresentSrc
    }

    #[inline]
    fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool {
        false
    }

    #[inline]
    fn conflicts_image(&self, other: &dyn ImageAccess) -> bool {
        self.my_image().image.key() == other.conflict_key() // TODO:
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.my_image().image.key()
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, _: ImageLayout) -> Result<(), AccessError> {
        if self.swapchain.is_fullscreen_exclusive() {
            Ok(())
        } else {
            // Swapchain image are only accessible after being acquired.
            Err(AccessError::SwapchainImageAcquireOnly)
        }
    }

    #[inline]
    unsafe fn layout_initialized(&self) {
        self.layout_initialized();
    }

    #[inline]
    fn is_layout_initialized(&self) -> bool{
        self.is_layout_initialized()
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
    }

    #[inline]
    unsafe fn unlock(&self, _: Option<ImageLayout>) {
        // TODO: store that the image was initialized
    }
}

unsafe impl<W> ImageClearValue<<Format as FormatDesc>::ClearValue> for SwapchainImage<W> {
    #[inline]
    fn decode(&self, value: <Format as FormatDesc>::ClearValue) -> Option<ClearValue> {
        Some(self.swapchain.format().decode_clear_value(value))
    }
}

unsafe impl<P,W> ImageContent<P> for SwapchainImage<W> {
    #[inline]
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

unsafe impl<W> ImageViewAccess for SwapchainImage<W> {
    #[inline]
    fn parent(&self) -> &dyn ImageAccess {
        self
    }

    #[inline]
    fn dimensions(&self) -> Dimensions {
        let dims = self.swapchain.dimensions();
        Dimensions::Dim2d {
            width: dims[0],
            height: dims[1],
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

impl<W> PartialEq for SwapchainImage<W> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ImageAccess::inner(self) == ImageAccess::inner(other)
    }
}

impl<W> Eq for SwapchainImage<W> {}

impl<W> Hash for SwapchainImage<W> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        ImageAccess::inner(self).hash(state);
    }
}
