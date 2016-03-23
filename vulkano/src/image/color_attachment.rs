use std::iter::Empty;
use std::sync::Arc;

use device::Device;
use format::FormatDesc;
use image::sys::Dimensions;
use image::sys::UnsafeImage;
use image::sys::Usage;
use memory::DeviceMemory;
use sync::Sharing;

use OomError;

pub struct ColorAttachmentImage<F> {
    image: UnsafeImage,
    memory: DeviceMemory,
    format: F,
}

impl<F> ColorAttachmentImage<F> {
    pub fn new(device: &Arc<Device>, dimensions: [u32; 2], format: F)
               -> Result<Arc<ColorAttachmentImage<F>>, OomError>
        where F: FormatDesc
    {
        let usage = Usage {
            transfer_source: true,
            sampled: true,
            color_attachment: true,
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

        Ok(Arc::new(ColorAttachmentImage {
            image: image,
            memory: mem,
            format: format,
        }))
    }
}
