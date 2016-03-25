use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;

use command_buffer::Submission;
use format::Format;
use image::traits::AccessRange;
use image::traits::Image;
use image::traits::ImageContent;
use image::traits::ImageView;
use image::sys::Dimensions;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use swapchain::Swapchain;

use OomError;

pub struct SwapchainImage {
    image: UnsafeImage,
    view: UnsafeImageView,
    format: Format,
    swapchain: Arc<Swapchain>,
    id: u32,

    /// True if already in the `PresentSrc` layout.
    // TODO: use AtomicBool ; however it's not that easy because we need to single-thread the part
    //       where we transition the layout
    present_layout: Mutex<bool>,
}

impl SwapchainImage {
    pub unsafe fn from_raw(image: UnsafeImage, format: Format, swapchain: &Arc<Swapchain>, id: u32)
                           -> Result<Arc<SwapchainImage>, OomError>
    {
        let view = try!(UnsafeImageView::new(&image));

        Ok(Arc::new(SwapchainImage {
            image: image,
            view: view,
            format: format,
            swapchain: swapchain.clone(),
            id: id,
            present_layout: Mutex::new(false),
        }))
    }

    #[inline]
    pub fn dimensions(&self) -> Dimensions {
        self.image.dimensions()
    }
}

unsafe impl Image for SwapchainImage {
    #[inline]
    fn inner_image(&self) -> &UnsafeImage {
        &self.image
    }

    #[inline]
    fn blocks(&self, _: Range<u32>, _: Range<u32>) -> Vec<(u32, u32)> {
        vec![(0, 0)]
    }

    #[inline]
    fn initial_layout(&self, _: (u32, u32), _: Layout) -> Layout {
        Layout::PresentSrc
    }

    #[inline]
    fn final_layout(&self, _: (u32, u32), _: Layout) -> Layout {
        Layout::PresentSrc
    }

    fn needs_fence(&self, access: &mut Iterator<Item = AccessRange>) -> Option<bool> {
        Some(false)
    }

    unsafe fn gpu_access(&self, access: &mut Iterator<Item = AccessRange>,
                         submission: &Arc<Submission>) -> Vec<Arc<Submission>>
    {
        let mut present_layout = self.present_layout.lock().unwrap();

        if *present_layout {
            return vec![];
        }

        // FIXME: submit a command buffer to transition the layout of the image

        *present_layout = true;
        vec![]
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
    fn parent_arc(me: &Arc<Self>) -> Arc<Image> where Self: Sized {
        me.clone() as Arc<_>
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
