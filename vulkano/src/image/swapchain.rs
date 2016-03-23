use std::sync::Arc;

use command_buffer::Submission;
use format::Format;
use image::traits::AccessRange;
use image::traits::Image;
use image::traits::ImageView;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use image::sys::Usage;
use swapchain::Swapchain;

use OomError;

pub struct SwapchainImage {
    image: UnsafeImage,
    view: UnsafeImageView,
    format: Format,
    usage: Usage,
    swapchain: Arc<Swapchain>,
}

impl SwapchainImage {
    pub unsafe fn from_raw(image: UnsafeImage, format: Format, swapchain: &Arc<Swapchain>,
                           usage: &Usage) -> Result<Arc<SwapchainImage>, OomError>
    {
        let view = try!(UnsafeImageView::new(&image));

        Ok(Arc::new(SwapchainImage {
            image: image,
            view: view,
            format: format,
            usage: usage.clone(),
            swapchain: swapchain.clone(),
        }))
    }
}

unsafe impl Image for SwapchainImage {
    #[inline]
    fn inner_image(&self) -> &UnsafeImage {
        &self.image
    }

    fn needs_fence(&self, access: &mut Iterator<Item = AccessRange>) -> Option<bool> {
        Some(false)
    }

    unsafe fn gpu_access(&self, access: &mut Iterator<Item = AccessRange>,
                         submission: &Arc<Submission>) -> Vec<Arc<Submission>>
    {
        vec![]
    }
}

unsafe impl ImageView for SwapchainImage {
    #[inline]
    fn parent(&self) -> &Image {
        self
    }

    #[inline]
    fn inner_view(&self) -> &UnsafeImageView {
        &self.view
    }

    #[inline]
    fn descriptor_set_storage_image_layout(&self, _: AccessRange) -> Layout {
        Layout::ColorAttachmentOptimal
    }

    #[inline]
    fn descriptor_set_combined_image_sampler_layout(&self, _: AccessRange) -> Layout {
        Layout::ColorAttachmentOptimal
    }

    #[inline]
    fn descriptor_set_sampled_image_layout(&self, _: AccessRange) -> Layout {
        Layout::ColorAttachmentOptimal
    }

    #[inline]
    fn descriptor_set_input_attachment_layout(&self, _: AccessRange) -> Layout {
        Layout::ColorAttachmentOptimal
    }

    #[inline]
    fn identity_swizzle(&self) -> bool {
        true
    }
}
