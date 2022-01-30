// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::format::ClearValue;
use crate::image::traits::ImageAccess;
use crate::image::traits::ImageClearValue;
use crate::image::traits::ImageContent;
use crate::image::ImageDescriptorLayouts;
use crate::image::ImageInner;
use crate::image::ImageLayout;
use crate::swapchain::Swapchain;
use crate::sync::AccessError;
use crate::OomError;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

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
}

impl<W> SwapchainImage<W> {
    /// Builds a `SwapchainImage` from raw components.
    ///
    /// This is an internal method that you shouldn't call.
    pub unsafe fn from_raw(
        swapchain: Arc<Swapchain<W>>,
        id: usize,
    ) -> Result<Arc<SwapchainImage<W>>, OomError> {
        let image = swapchain.raw_image(id).unwrap();

        Ok(Arc::new(SwapchainImage {
            swapchain: swapchain.clone(),
            image_offset: id,
        }))
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
        self.swapchain
            .is_image_layout_initialized(self.image_offset)
    }
}

unsafe impl<W> ImageAccess for SwapchainImage<W>
where
    W: Send + Sync,
{
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
    fn descriptor_layouts(&self) -> Option<ImageDescriptorLayouts> {
        Some(ImageDescriptorLayouts {
            storage_image: ImageLayout::General,
            combined_image_sampler: ImageLayout::ShaderReadOnlyOptimal,
            sampled_image: ImageLayout::ShaderReadOnlyOptimal,
            input_attachment: ImageLayout::ShaderReadOnlyOptimal,
        })
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.my_image().image.key()
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, _: bool, _: ImageLayout) -> Result<(), AccessError> {
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
    fn is_layout_initialized(&self) -> bool {
        self.is_layout_initialized()
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {}

    #[inline]
    unsafe fn unlock(&self, _: Option<ImageLayout>) {
        // TODO: store that the image was initialized
    }

    #[inline]
    fn current_mip_levels_access(&self) -> std::ops::Range<u32> {
        0..self.mip_levels()
    }

    #[inline]
    fn current_array_layers_access(&self) -> std::ops::Range<u32> {
        0..1
    }
}

unsafe impl<W> ImageClearValue<ClearValue> for SwapchainImage<W>
where
    W: Send + Sync,
{
    #[inline]
    fn decode(&self, value: ClearValue) -> Option<ClearValue> {
        Some(self.swapchain.format().decode_clear_value(value))
    }
}

unsafe impl<P, W> ImageContent<P> for SwapchainImage<W>
where
    W: Send + Sync,
{
    #[inline]
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

impl<W> PartialEq for SwapchainImage<W>
where
    W: Send + Sync,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl<W> Eq for SwapchainImage<W> where W: Send + Sync {}

impl<W> Hash for SwapchainImage<W>
where
    W: Send + Sync,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}
