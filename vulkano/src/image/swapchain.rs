// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    sys::{Image, ImageMemory},
    traits::ImageContent,
    ImageAccess, ImageLayout,
};
use crate::{
    device::{Device, DeviceOwned},
    swapchain::Swapchain,
    OomError,
};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

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
#[derive(Debug)]
pub struct SwapchainImage {
    inner: Arc<Image>,
}

impl SwapchainImage {
    pub(crate) unsafe fn from_handle(
        handle: ash::vk::Image,
        swapchain: Arc<Swapchain>,
        image_index: u32,
    ) -> Result<Arc<SwapchainImage>, OomError> {
        Ok(Arc::new(SwapchainImage {
            inner: Arc::new(Image::from_swapchain(handle, swapchain, image_index)),
        }))
    }

    /// Returns the swapchain this image belongs to.
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        match self.inner.memory() {
            ImageMemory::Swapchain {
                swapchain,
                image_index: _,
            } => swapchain,
            _ => unreachable!(),
        }
    }
}

unsafe impl DeviceOwned for SwapchainImage {
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl ImageAccess for SwapchainImage {
    fn inner(&self) -> &Arc<Image> {
        &self.inner
    }

    fn initial_layout_requirement(&self) -> ImageLayout {
        ImageLayout::PresentSrc
    }

    fn final_layout_requirement(&self) -> ImageLayout {
        ImageLayout::PresentSrc
    }

    unsafe fn layout_initialized(&self) {
        match self.inner.memory() {
            &ImageMemory::Swapchain {
                ref swapchain,
                image_index,
            } => swapchain.image_layout_initialized(image_index),
            _ => unreachable!(),
        }
    }

    fn is_layout_initialized(&self) -> bool {
        match self.inner.memory() {
            &ImageMemory::Swapchain {
                ref swapchain,
                image_index,
            } => swapchain.is_image_layout_initialized(image_index),
            _ => unreachable!(),
        }
    }
}

unsafe impl<P> ImageContent<P> for SwapchainImage {
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

impl PartialEq for SwapchainImage {
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for SwapchainImage {}

impl Hash for SwapchainImage {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}
