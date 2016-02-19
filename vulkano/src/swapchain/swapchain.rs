use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

use device::Device;
use device::Queue;
use formats::FormatMarker;
use image::Image;
use image::ImagePrototype;
use image::Type2d;
use image::Usage as ImageUsage;
use memory::ChunkProperties;
use memory::ChunkRange;
use memory::MemorySourceChunk;
use swapchain::CompositeAlpha;
use swapchain::PresentMode;
use swapchain::Surface;
use swapchain::SurfaceTransform;
use sync::Fence;
use sync::Semaphore;
use sync::SharingMode;

use check_errors;
use Error;
use OomError;
use Success;
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
                     clipped: bool) -> Result<(Arc<Swapchain>, Vec<ImagePrototype<Type2d, F, SwapchainAllocatedChunk>>), OomError>
        where F: FormatMarker, S: Into<SharingMode>
    {
        // FIXME: check that the parameters are supported

        // FIXME: check that the device and the surface belong to the same instance
        let vk = device.pointers();

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
                imageFormat: F::format() as u32,
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
            let mut num = mem::uninitialized();
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
            let mem = SwapchainAllocatedChunk { swapchain: swapchain.clone(), id: id };
            Image::from_raw_unowned(&device, image, mem, sharing.clone(), usage, dimensions, (), 1)
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
    pub fn present(&self, queue: &mut Queue, index: usize) -> Result<(), OomError> {     // FIXME: wrong error
        let vk = self.device.pointers();

        let wait_semaphore = {
            let mut images_semaphores = self.images_semaphores.lock().unwrap();
            images_semaphores[index].take()
        };

        // FIXME: the semaphore will be destroyed ; need to return it

        unsafe {
            let mut result = mem::uninitialized();

            let index = index as u32;
            let infos = vk::PresentInfoKHR {
                sType: vk::STRUCTURE_TYPE_PRESENT_INFO_KHR,
                pNext: ptr::null(),
                waitSemaphoreCount: if let Some(_) = wait_semaphore { 1 } else { 0 },
                pWaitSemaphores: if let Some(ref sem) = wait_semaphore { &sem.internal_object() } else { ptr::null() },
                swapchainCount: 1,
                pSwapchains: &self.swapchain,
                pImageIndices: &index,
                pResults: &mut result,
            };

            try!(check_errors(vk.QueuePresentKHR(queue.internal_object(), &infos)));
            try!(check_errors(result));
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

/// "Dummy" object used for images that indicates that they were allocated as part of a swapchain.
pub struct SwapchainAllocatedChunk {
    swapchain: Arc<Swapchain>,
    id: usize,
}

// FIXME: needs correct synchronization as well
unsafe impl MemorySourceChunk for SwapchainAllocatedChunk {
    #[inline]
    fn properties(&self) -> ChunkProperties {
        unreachable!()
    }

    #[inline]
    fn requires_fence(&self) -> bool { false }
    #[inline]
    fn requires_semaphore(&self) -> bool { true }
    #[inline]
    fn may_alias(&self) -> bool { false }

    #[inline]
    fn gpu_access(&self, _: bool, _: ChunkRange, _: &mut Queue, _: Option<Arc<Fence>>,
                  post_semaphore: Option<Arc<Semaphore>>) -> Option<Arc<Semaphore>>
    {
        assert!(post_semaphore.is_some());
        // FIXME: must also check that image has been acquired
        let mut semaphores = self.swapchain.images_semaphores.lock().unwrap();
        let pre_semaphore = mem::replace(&mut semaphores[self.id], post_semaphore);
        pre_semaphore
    }
}
