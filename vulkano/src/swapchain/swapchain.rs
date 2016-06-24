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
use std::sync::atomic::Ordering;
use std::time::Duration;
use crossbeam::sync::MsQueue;

use device::Device;
use device::Queue;
use format::Format;
use format::FormatDesc;
use image::sys::Dimensions;
use image::sys::UnsafeImage;
use image::sys::Usage as ImageUsage;
use image::swapchain::SwapchainImage;
use swapchain::CompositeAlpha;
use swapchain::PresentMode;
use swapchain::Surface;
use swapchain::SurfaceTransform;
use swapchain::SurfaceSwapchainLock;
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
// TODO: #[derive(Debug)] (waiting on https://github.com/aturon/crossbeam/issues/62)
pub struct Swapchain {
    device: Arc<Device>,
    surface: Arc<Surface>,
    swapchain: vk::SwapchainKHR,

    /// Pool of semaphores from which a semaphore is retrieved when acquiring an image.
    ///
    /// We need to use a queue so that we don't use the same semaphore twice in a row. The length
    /// of the queue is strictly superior to the number of images, in case the driver lets us
    /// acquire an image before it is presented.
    semaphores_pool: MsQueue<Arc<Semaphore>>,

    images_semaphores: Mutex<Vec<Option<Arc<Semaphore>>>>,

    // If true, that means we have used this swapchain to recreate a new swapchain. The current
    // swapchain can no longer be used for anything except presenting already-acquired images.
    //
    // We use a `Mutex` instead of an `AtomicBool` because we want to keep that locked while
    // we acquire the image.
    stale: Mutex<bool>,
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
    #[inline]
    pub fn new<F, S>(device: &Arc<Device>, surface: &Arc<Surface>, num_images: u32, format: F,
                     dimensions: [u32; 2], layers: u32, usage: &ImageUsage, sharing: S,
                     transform: SurfaceTransform, alpha: CompositeAlpha, mode: PresentMode,
                     clipped: bool, old_swapchain: Option<&Arc<Swapchain>>)
                     -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), OomError>
        where F: FormatDesc, S: Into<SharingMode>
    {
        Swapchain::new_inner(device, surface, num_images, format.format(), dimensions, layers,
                             usage, sharing.into(), transform, alpha, mode, clipped, old_swapchain)
    }

    // TODO:
    //pub fn recreate() { ... }

    // TODO: images layouts should always be set to "PRESENT", since we have no way to switch the
    //       layout at present time
    fn new_inner(device: &Arc<Device>, surface: &Arc<Surface>, num_images: u32, format: Format,
                 dimensions: [u32; 2], layers: u32, usage: &ImageUsage, sharing: SharingMode,
                 transform: SurfaceTransform, alpha: CompositeAlpha, mode: PresentMode,
                 clipped: bool, old_swapchain: Option<&Arc<Swapchain>>)
                 -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), OomError>
    {
        // Checking that the requested parameters match the capabilities.
        let capabilities = try!(surface.get_capabilities(&device.physical_device()));
        // TODO: return errors instead
        assert!(num_images >= capabilities.min_image_count);
        if let Some(c) = capabilities.max_image_count { assert!(num_images <= c) };
        assert!(capabilities.supported_formats.iter().find(|&&(f, _)| f == format).is_some());
        assert!(dimensions[0] >= capabilities.min_image_extent[0]);
        assert!(dimensions[1] >= capabilities.min_image_extent[1]);
        assert!(dimensions[0] <= capabilities.max_image_extent[0]);
        assert!(dimensions[1] <= capabilities.max_image_extent[1]);
        assert!(layers >= 1 && layers <= capabilities.max_image_array_layers);
        assert!((usage.to_usage_bits() & capabilities.supported_usage_flags.to_usage_bits()) == usage.to_usage_bits());
        assert!(capabilities.supported_transforms.supports(transform));
        assert!(capabilities.supported_composite_alpha.supports(alpha));
        assert!(capabilities.present_modes.supports(mode));

        // If we recreate a swapchain, make sure that the surface is the same.
        if let Some(sc) = old_swapchain {
            // TODO: return proper error instead of panicking?
            assert_eq!(surface.internal_object(), sc.surface.internal_object());
        }

        // Checking that the surface doesn't already have a swapchain.
        if old_swapchain.is_none() {
            // TODO: return proper error instead of panicking?
            let has_already = surface.flag().swap(true, Ordering::AcqRel);
            if has_already { panic!("The surface already has a swapchain alive"); }
        }

        // FIXME: check that the device and the surface belong to the same instance
        let vk = device.pointers();
        assert!(device.loaded_extensions().khr_swapchain);     // TODO: return error instead

        assert!(usage.color_attachment);
        let usage = usage.to_usage_bits();

        let sharing = sharing.into();

        if let Some(ref old_swapchain) = old_swapchain {
            *old_swapchain.stale.lock().unwrap() = false;
        }

        let swapchain = unsafe {
            let (sh_mode, sh_count, sh_indices) = match sharing {
                SharingMode::Exclusive(_) => (vk::SHARING_MODE_EXCLUSIVE, 0, ptr::null()),
                SharingMode::Concurrent(ref ids) => (vk::SHARING_MODE_CONCURRENT, ids.len() as u32,
                                                     ids.as_ptr()),
            };

            let infos = vk::SwapchainCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0,   // reserved
                surface: surface.internal_object(),
                minImageCount: num_images,
                imageFormat: format as u32,
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
                oldSwapchain: if let Some(ref old_swapchain) = old_swapchain {
                    old_swapchain.swapchain
                } else {
                    0
                },
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
            semaphores_pool: MsQueue::new(),
            images_semaphores: Mutex::new(Vec::new()),
            stale: Mutex::new(false),
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
            let unsafe_image = UnsafeImage::from_raw(device, image, usage, format,
                                                     Dimensions::Dim2d { width: dimensions[0], height: dimensions[1] }, 1, 1);
            SwapchainImage::from_raw(unsafe_image, format, &swapchain, id as u32).unwrap()     // TODO: propagate error
        }).collect::<Vec<_>>();

        {
            let mut semaphores = swapchain.images_semaphores.lock().unwrap();
            for _ in 0 .. images.len() {
                semaphores.push(None);
            }
        }

        for _ in 0 .. images.len() + 1 {
            // TODO: check if this change is okay (maybe the Arc can be omitted?) - Mixthos
            //swapchain.semaphores_pool.push(try!(Semaphore::new(device)));
            swapchain.semaphores_pool.push(Arc::new(try!(Semaphore::raw(device))));
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
    pub fn acquire_next_image(&self, timeout: Duration) -> Result<usize, AcquireError> {
        unsafe {
            let stale = self.stale.lock().unwrap();
            if *stale {
                return Err(AcquireError::OutOfDate);
            }

            let vk = self.device.pointers();

            let semaphore = self.semaphores_pool.try_pop().expect("Failed to obtain a semaphore \
                                                                   from the swapchain semaphores \
                                                                   pool");

            let timeout_ns = timeout.as_secs().saturating_mul(1_000_000_000)
                                              .saturating_add(timeout.subsec_nanos() as u64);

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
    pub fn present(&self, queue: &Arc<Queue>, index: usize) -> Result<(), PresentError> {
        let vk = self.device.pointers();

        let wait_semaphore = {
            let mut images_semaphores = self.images_semaphores.lock().unwrap();
            images_semaphores[index].take().expect("Trying to present an image that was \
                                                    not acquired")
        };

        // FIXME: the semaphore may be destroyed ; need to return it

        unsafe {
            let mut result = mem::uninitialized();

            let queue = queue.internal_object_guard();
            let index = index as u32;

            let infos = vk::PresentInfoKHR {
                sType: vk::STRUCTURE_TYPE_PRESENT_INFO_KHR,
                pNext: ptr::null(),
                waitSemaphoreCount: 1,
                pWaitSemaphores: &wait_semaphore.internal_object(),
                swapchainCount: 1,
                pSwapchains: &self.swapchain,
                pImageIndices: &index,
                pResults: &mut result,
            };

            try!(check_errors(vk.QueuePresentKHR(*queue, &infos)));
            //try!(check_errors(result));       // TODO: AMD driver doesn't seem to write the result
        }

        self.semaphores_pool.push(wait_semaphore);
        Ok(())
    }

    /*/// Returns the semaphore that is going to be signalled when the image is going to be ready
    /// to be drawn upon.
    ///
    /// Returns `None` if the image was not acquired first, or was already presented.
    // TODO: racy, as someone could present the image before using the semaphore
    #[inline]
    pub fn image_semaphore(&self, id: u32) -> Option<Arc<Semaphore>> {
        let semaphores = self.images_semaphores.lock().unwrap();
        semaphores[id as usize].as_ref().map(|s| s.clone())
    }*/
    // TODO: the design of this functions depends on https://github.com/KhronosGroup/Vulkan-Docs/issues/155
    #[inline]
    #[doc(hidden)]
    pub fn image_semaphore(&self, id: u32, semaphore: Arc<Semaphore>) -> Option<Arc<Semaphore>> {
        let mut semaphores = self.images_semaphores.lock().unwrap();
        mem::replace(&mut semaphores[id as usize], Some(semaphore))
    }
}

impl Drop for Swapchain {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroySwapchainKHR(self.device.internal_object(), self.swapchain, ptr::null());
            self.surface.flag().store(false, Ordering::Release);
        }
    }
}

/// Error that can happen when calling `acquire_next_image`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum AcquireError {
    /// Not enough memory.
    OomError(OomError),

    /// The connection to the device has been lost.
    DeviceLost,

    /// The timeout of the function has been reached before an image was available.
    Timeout,

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,

    /// The surface has changed in a way that makes the swapchain unusable. You must query the
    /// surface's new properties and recreate a new swapchain if you want to continue drawing.
    OutOfDate,
}

impl error::Error for AcquireError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            AcquireError::OomError(_) => "not enough memory",
            AcquireError::DeviceLost => "the connection to the device has been lost",
            AcquireError::Timeout => "no image is available for acquiring yet",
            AcquireError::SurfaceLost => "the surface of this swapchain is no longer valid",
            AcquireError::OutOfDate => "the swapchain needs to be recreated",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            AcquireError::OomError(ref err) => Some(err),
            _ => None
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
            err @ Error::OutOfHostMemory => AcquireError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => AcquireError::OomError(OomError::from(err)),
            Error::DeviceLost => AcquireError::DeviceLost,
            Error::SurfaceLost => AcquireError::SurfaceLost,
            Error::OutOfDate => AcquireError::OutOfDate,
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}

/// Error that can happen when calling `acquire_next_image`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum PresentError {
    /// Not enough memory.
    OomError(OomError),

    /// The connection to the device has been lost.
    DeviceLost,

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,

    /// The surface has changed in a way that makes the swapchain unusable. You must query the
    /// surface's new properties and recreate a new swapchain if you want to continue drawing.
    OutOfDate,
}

impl error::Error for PresentError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            PresentError::OomError(_) => "not enough memory",
            PresentError::DeviceLost => "the connection to the device has been lost",
            PresentError::SurfaceLost => "the surface of this swapchain is no longer valid",
            PresentError::OutOfDate => "the swapchain needs to be recreated",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            PresentError::OomError(ref err) => Some(err),
            _ => None
        }
    }
}

impl fmt::Display for PresentError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for PresentError {
    #[inline]
    fn from(err: Error) -> PresentError {
        match err {
            err @ Error::OutOfHostMemory => PresentError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => PresentError::OomError(OomError::from(err)),
            Error::DeviceLost => PresentError::DeviceLost,
            Error::SurfaceLost => PresentError::SurfaceLost,
            Error::OutOfDate => PresentError::OutOfDate,
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}
