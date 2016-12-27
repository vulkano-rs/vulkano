// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;

use command_buffer::Submission;
use format::ClearValue;
use format::Format;
use format::FormatDesc;
use image::Dimensions;
use image::ViewType;
use image::traits::Image;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::traits::ImageView;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use swapchain::Swapchain;
use sync::AccessFlagBits;
use sync::PipelineStages;

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
    guarded: Mutex<Guarded>,
}

#[derive(Debug)]
struct Guarded {
    present_layout: bool,
    latest_submission: Option<Weak<Submission>>, // TODO: can use `Weak::new()` once it's stabilized
}

impl SwapchainImage {
    /// Builds a `SwapchainImage` from raw components.
    ///
    /// This is an internal method that you shouldn't call.
    pub unsafe fn from_raw(image: UnsafeImage, format: Format, swapchain: &Arc<Swapchain>, id: u32) -> Result<Arc<SwapchainImage>, OomError> {
        let view = try!(UnsafeImageView::raw(&image, ViewType::Dim2d, 0..1, 0..1));

        Ok(Arc::new(SwapchainImage {
            image: image,
            view: view,
            format: format,
            swapchain: swapchain.clone(),
            id: id,
            guarded: Mutex::new(Guarded {
                present_layout: false,
                latest_submission: None,
            }),
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

unsafe impl Image for SwapchainImage {
    #[inline]
    fn inner(&self) -> &UnsafeImage {
        &self.image
    }

    #[inline]
    fn conflict_key(&self, _: u32, _: u32, _: u32, _: u32) -> u64 {
        self.image.key()
    }
}

unsafe impl ImageClearValue<<Format as FormatDesc>::ClearValue> for SwapchainImage {
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

unsafe impl ImageView for SwapchainImage {
    #[inline]
    fn parent(&self) -> &Image {
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

pub struct SwapchainImageCbState {
    stages: PipelineStages,
    access: AccessFlagBits,
    command_num: usize,
    layout: Layout,
}
