use std::mem;
use std::iter::Empty;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;

use command_buffer::Submission;
use device::Device;
use format::FormatDesc;
use format::FormatTy;
use image::sys::Dimensions;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use image::sys::Usage;
use image::traits::AccessRange;
use image::traits::Image;
use image::traits::ImageContent;
use image::traits::ImageView;
use memory::DeviceMemory;
use sync::Sharing;

use OomError;

pub struct AttachmentImage<F> {
    image: UnsafeImage,
    view: UnsafeImageView,
    memory: DeviceMemory,
    format: F,

    // Layout to use when the image is used as a framebuffer attachment.
    // Should be either "depth-stencil optimal" or "color optimal".
    attachment_layout: Layout,

    latest_submission: Mutex<Option<Arc<Submission>>>,
}

impl<F> AttachmentImage<F> {
    pub fn new(device: &Arc<Device>, dimensions: [u32; 2], format: F)
               -> Result<Arc<AttachmentImage<F>>, OomError>
        where F: FormatDesc
    {
        let is_depth = match format.format().ty() {
            FormatTy::Depth => true,
            FormatTy::DepthStencil => true,
            FormatTy::Compressed => panic!(),
            _ => false
        };

        let usage = Usage {
            transfer_source: true,
            sampled: true,
            color_attachment: !is_depth,
            depth_stencil_attachment: is_depth,
            input_attachment: true,
            .. Usage::none()
        };

        let (image, mem_reqs) = unsafe {
            try!(UnsafeImage::new(device, &usage, format.format(),
                                  Dimensions::Dim2d { width: dimensions[0], height: dimensions[1] },
                                  1, 1, Sharing::Exclusive::<Empty<u32>>, false, false))
        };

        let mem_ty = {
            let device_local = device.physical_device().memory_types()
                                     .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                                     .filter(|t| t.is_device_local());
            let any = device.physical_device().memory_types()
                            .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0);
            device_local.chain(any).next().unwrap()
        };

        // note: alignment doesn't need to be checked because allocating memory is guaranteed to
        //       fulfill any alignment requirement

        let mem = try!(DeviceMemory::alloc(device, &mem_ty, mem_reqs.size));
        unsafe { try!(image.bind_memory(&mem, 0 .. mem_reqs.size)); }

        let view = unsafe {
            try!(UnsafeImageView::new(&image))
        };

        Ok(Arc::new(AttachmentImage {
            image: image,
            view: view,
            memory: mem,
            format: format,
            attachment_layout: if is_depth { Layout::DepthStencilAttachmentOptimal }
                               else { Layout::ColorAttachmentOptimal },
            latest_submission: Mutex::new(None),
        }))
    }

    #[inline]
    pub fn dimensions(&self) -> Dimensions {
        self.image.dimensions()
    }
}

unsafe impl<F> Image for AttachmentImage<F> {
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
        self.attachment_layout
    }

    #[inline]
    fn final_layout(&self, _: (u32, u32), _: Layout) -> Layout {
        self.attachment_layout
    }

    fn needs_fence(&self, access: &mut Iterator<Item = AccessRange>) -> Option<bool> {
        Some(false)
    }

    unsafe fn gpu_access(&self, _: &mut Iterator<Item = AccessRange>,
                         submission: &Arc<Submission>) -> Vec<Arc<Submission>>
    {
        let mut latest_submission = self.latest_submission.lock().unwrap();

        let dependency = mem::replace(&mut *latest_submission, Some(submission.clone()));
        if let Some(dependency) = dependency {
            vec![dependency]
        } else {
            vec![]
        }
    }
}

unsafe impl<P, F> ImageContent<P> for AttachmentImage<F> {
    #[inline]
    fn matches_format(&self) -> bool {
        true        // FIXME:
    }
}

unsafe impl<F: 'static> ImageView for AttachmentImage<F> {
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
