use std::iter::Empty;
use std::ops::Range;
use std::sync::Arc;

use command_buffer::Submission;
use device::Device;
use format::FormatDesc;
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
}

impl<F> AttachmentImage<F> {
    pub fn new(device: &Arc<Device>, dimensions: [u32; 2], format: F)
               -> Result<Arc<AttachmentImage<F>>, OomError>
        where F: FormatDesc
    {
        let usage = Usage {
            transfer_source: true,
            sampled: true,
            color_attachment: true,
            depth_stencil_attachment: true,
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

    fn needs_fence(&self, access: &mut Iterator<Item = AccessRange>) -> Option<bool> {
        Some(false)
    }

    unsafe fn gpu_access(&self, access: &mut Iterator<Item = AccessRange>,
                         submission: &Arc<Submission>) -> Vec<Arc<Submission>>
    {
        vec![]
    }
}

unsafe impl<P, F> ImageContent<P> for AttachmentImage<F> {
    #[inline]
    fn matches_format(&self) -> bool {
        true        // FIXME:
    }
}

unsafe impl<F> ImageView for AttachmentImage<F> {
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
