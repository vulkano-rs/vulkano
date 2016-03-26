// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

use device::Device;
use device::Queue;
use format::FormatDesc;
use image::sys::Dimensions;
use image::sys::UnsafeImage;
use image::sys::Usage as ImageUsage;
use image::swapchain::SwapchainImage;
use swapchain::CompositeAlpha;
use swapchain::PresentMode;
use swapchain::Surface;
use swapchain::SurfaceTransform;
use sync::Semaphore;
use sync::SharingMode;

use check_errors;
use Error;
use OomError;
use Success;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use vk;

/// Contains the swapping system and the images that can be shown on a surface.
pub struct Swapchain {
    device: Arc<Device>,
    surface: Arc<Surface>,
    swapchain: vk::SwapchainKHR,

    images_semaphores: Mutex<Vec<Option<Arc<Semaphore>>>>,
}

impl Swapchain {
    /// Builds a new swapchain. Allocates images who content can be made visible on a surface.
    ///
    /// See also the `Surface::get_capabilities` function which returns the values that are
    /// supported by the implementation. All the parameters that you pass to `Swapchain::new`
    /// must be supported. 
    ///
    /// The `clipped` parameter indicates whether the implementation is allowed to discard 
    /// rendering operations that affect regions of the surface which aren't visible. This is
    /// important to take into account if your fragment shader has side-effects or if you want to
    /// read back the content of the image afterwards.
    ///
    /// This function returns the swapchain plus a list of the images that belong to the
    /// swapchain. The order in which the images are returned is important for the
    /// `acquire_next_image` and `present` functions.
    ///
    /// # Panic
    ///
    /// - Panicks if the device and the surface don't belong to the same instance.
    /// - Panicks if `color_attachment` is false in `usage`.
    ///
    pub fn new<F, S>(device: &Arc<Device>, surface: &Arc<Surface>, num_images: u32, format: F,
                     dimensions: [u32; 2], layers: u32, usage: &ImageUsage, sharing: S,
                     transform: SurfaceTransform, alpha: CompositeAlpha, mode: PresentMode,
                     clipped: bool) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), OomError>
        where F: FormatDesc + Clone, S: Into<SharingMode>
    {
        Swapchain::new_inner(device, surface, num_images, format, dimensions, layers, usage,
                             sharing, transform, alpha, mode, clipped)
    }

    // TODO:
    //pub fn recreate() { ... }

    // TODO: images layouts should always be set to "PRESENT", since we have no way to switch the
    //       layout at present time
    fn new_inner<F, S>(device: &Arc<Device>, surface: &Arc<Surface>, num_images: u32, format: F,
                       dimensions: [u32; 2], layers: u32, usage: &ImageUsage, sharing: S,
                       transform: SurfaceTransform, alpha: CompositeAlpha, mode: PresentMode,
                       clipped: bool) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), OomError>
        where F: FormatDesc + Clone, S: Into<SharingMode>
    {
        // FIXME: check that the parameters are supported

        // FIXME: check that the device and the surface belong to the same instance
        let vk = device.pointers();
        assert!(device.loaded_extensions().khr_swapchain);     // TODO: return error instead

        assert!(usage.color_attachment);
        let usage = usage.to_usage_bits();

        let sharing = sharing.into();

        let swapchain = unsafe {
            let (sh_mode, sh_count, sh_indices) = match sharing {
                SharingMode::Exclusive(id) => (vk::SHARING_MODE_EXCLUSIVE, 0, ptr::null()),
                SharingMode::Concurrent(ref ids) => (vk::SHARING_MODE_CONCURRENT, ids.len() as u32,
                                                     ids.as_ptr()),
            };

            let infos = vk::SwapchainCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0,   // reserved
                surface: surface.internal_object(),
                minImageCount: num_images,
                imageFormat: format.format() as u32,
                imageColorSpace: vk::COLORSPACE_SRGB_NONLINEAR_KHR,     // only available value
                imageExtent: vk::Extent2D { width: dimensions[0], height: dimensions[1] },
                imageArrayLayers: layers,
                imageUsage: usage,
                imageSharingMode: sh_mode,
                queueFamilyIndexCount: sh_count,
                pQueueFamilyIndices: sh_indices,
                preTransform: transform as u32,
                compositeAlpha: alpha as u32,
                presentMode: mode as u32,
                clipped: if clipped { vk::TRUE } else { vk::FALSE },
                oldSwapchain: 0,      // TODO:
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateSwapchainKHR(device.internal_object(), &infos,
                                                    ptr::null(), &mut output)));
            output
        };

        let swapchain = Arc::new(Swapchain {
            device: device.clone(),
            surface: surface.clone(),
            swapchain: swapchain,
            images_semaphores: Mutex::new(Vec::new()),
        });

        let images = unsafe {
            let mut num = 0;
            try!(check_errors(vk.GetSwapchainImagesKHR(device.internal_object(),
                                                       swapchain.swapchain, &mut num,
                                                       ptr::null_mut())));

            let mut images = Vec::with_capacity(num as usize);
            try!(check_errors(vk.GetSwapchainImagesKHR(device.internal_object(),
                                                       swapchain.swapchain, &mut num,
                                                       images.as_mut_ptr())));
            images.set_len(num as usize);
            images
        };

        let images = images.into_iter().enumerate().map(|(id, image)| unsafe {
            let unsafe_image = UnsafeImage::from_raw(device, image, usage, format.format(),
                                                     Dimensions::Dim2d { width: dimensions[0], height: dimensions[1] }, 1, 1);
            SwapchainImage::from_raw(unsafe_image, format.format(), &swapchain, id as u32).unwrap()     // TODO: propagate error
        }).collect::<Vec<_>>();

        {
            let mut semaphores = swapchain.images_semaphores.lock().unwrap();
            for _ in 0 .. images.len() {
                semaphores.push(None);
            }
        }

        Ok((swapchain, images))
    }

    /// Tries to take ownership of an image in order to draw on it.
    ///
    /// The function returns the index of the image in the array of images that was returned
    /// when creating the swapchain.
    ///
    /// If you try to draw on an image without acquiring it first, the execution will block. (TODO
    /// behavior may change).
    pub fn acquire_next_image(&self, timeout_ns: u64) -> Result<usize, AcquireError> {
        let vk = self.device.pointers();

        unsafe {
            let semaphore = Semaphore::new(&self.device).unwrap();      // TODO: error

            let mut out = mem::uninitialized();
            let r = try!(check_errors(vk.AcquireNextImageKHR(self.device.internal_object(),
                                                             self.swapchain, timeout_ns,
                                                             semaphore.internal_object(), 0,     // TODO: timeout
                                                             &mut out)));

            let id = match r {
                Success::Success => out as usize,
                Success::Suboptimal => out as usize,        // TODO: give that info to the user
                Success::NotReady => return Err(AcquireError::Timeout),
                Success::Timeout => return Err(AcquireError::Timeout),
                s => panic!("unexpected success value: {:?}", s)
            };

            let mut images_semaphores = self.images_semaphores.lock().unwrap();
            images_semaphores[id] = Some(semaphore);

            Ok(id)
        }
    }

    /// Presents an image on the screen.
    ///
    /// The parameter is the same index as what `acquire_next_image` returned. The image must
    /// have been acquired first.
    ///
    /// The actual behavior depends on the present mode that you passed when creating the
    /// swapchain.
    pub fn present(&self, queue: &Arc<Queue>, index: usize) -> Result<(), OomError> {     // FIXME: wrong error
        let vk = self.device.pointers();

        let wait_semaphore = {
            let mut images_semaphores = self.images_semaphores.lock().unwrap();
            images_semaphores[index].take()
        };

        // FIXME: the semaphore will be destroyed ; need to return it

        unsafe {
            let mut result = mem::uninitialized();

            let queue = queue.internal_object_guard();
            let semaphore = if let Some(ref sem) = wait_semaphore { sem.internal_object() } else { 0 };

            let index = index as u32;
            let infos = vk::PresentInfoKHR {
                sType: vk::STRUCTURE_TYPE_PRESENT_INFO_KHR,
                pNext: ptr::null(),
                waitSemaphoreCount: if let Some(_) = wait_semaphore { 1 } else { 0 },
                pWaitSemaphores: &semaphore,
                swapchainCount: 1,
                pSwapchains: &self.swapchain,
                pImageIndices: &index,
                pResults: &mut result,
            };

            try!(check_errors(vk.QueuePresentKHR(*queue, &infos)));
            //try!(check_errors(result));       // TODO: AMD driver doesn't seem to write the result
            Ok(())
        }
    }
}

impl Drop for Swapchain {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroySwapchainKHR(self.device.internal_object(), self.swapchain, ptr::null());
        }
    }
}

/// Error that can happen when calling `acquire_next_image`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum AcquireError {
    Timeout,
    SurfaceLost,
    OutOfDate,
}

impl error::Error for AcquireError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            AcquireError::Timeout => "no image is available for acquiring yet",
            AcquireError::SurfaceLost => "the surface of this swapchain is no longer valid",
            AcquireError::OutOfDate => "the swapchain needs to be recreated",
        }
    }
}

impl fmt::Display for AcquireError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for AcquireError {
    #[inline]
    fn from(err: Error) -> AcquireError {
        match err {
            Error::SurfaceLost => AcquireError::SurfaceLost,
            Error::OutOfDate => AcquireError::OutOfDate,
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}
