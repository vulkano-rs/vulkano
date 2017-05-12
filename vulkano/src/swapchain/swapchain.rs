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
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use buffer::BufferAccess;
use command_buffer::submit::SubmitAnyBuilder;
use command_buffer::submit::SubmitPresentBuilder;
use command_buffer::submit::SubmitSemaphoresWaitBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use format::Format;
use format::FormatDesc;
use image::ImageAccess;
use image::ImageDimensions;
use image::Layout;
use image::sys::UnsafeImage;
use image::sys::Usage as ImageUsage;
use image::swapchain::SwapchainImage;
use swapchain::ColorSpace;
use swapchain::CompositeAlpha;
use swapchain::PresentMode;
use swapchain::Surface;
use swapchain::SurfaceTransform;
use swapchain::SurfaceSwapchainLock;
use sync::AccessCheckError;
use sync::AccessError;
use sync::AccessFlagBits;
use sync::FlushError;
use sync::GpuFuture;
use sync::PipelineStages;
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
// TODO: #[derive(Debug)] (waiting on https://github.com/aturon/crossbeam/issues/62)
pub struct Swapchain {
    device: Arc<Device>,
    surface: Arc<Surface>,
    swapchain: vk::SwapchainKHR,

    // If true, that means we have used this swapchain to recreate a new swapchain. The current
    // swapchain can no longer be used for anything except presenting already-acquired images.
    //
    // We use a `Mutex` instead of an `AtomicBool` because we want to keep that locked while
    // we acquire the image.
    stale: Mutex<bool>,

    // Parameters passed to the constructor.
    num_images: u32,
    format: Format,
    color_space: ColorSpace,
    dimensions: [u32; 2],
    layers: u32,
    usage: ImageUsage,
    sharing: SharingMode,
    transform: SurfaceTransform,
    alpha: CompositeAlpha,
    mode: PresentMode,
    clipped: bool,

    // TODO: meh for Mutex
    images: Mutex<Vec<(Weak<SwapchainImage>, bool)>>,
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
    /// - Panics if the device and the surface don't belong to the same instance.
    /// - Panics if `color_attachment` is false in `usage`.
    ///
    // TODO: remove `old_swapchain` parameter and add another function `with_old_swapchain`.
    // TODO: add `ColorSpace` parameter
    #[inline]
    pub fn new<F, S>(device: &Arc<Device>, surface: &Arc<Surface>, num_images: u32, format: F,
                     dimensions: [u32; 2], layers: u32, usage: &ImageUsage, sharing: S,
                     transform: SurfaceTransform, alpha: CompositeAlpha, mode: PresentMode,
                     clipped: bool, old_swapchain: Option<&Arc<Swapchain>>)
                     -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), OomError>
        where F: FormatDesc, S: Into<SharingMode>
    {
        Swapchain::new_inner(device, surface, num_images, format.format(),
                             ColorSpace::SrgbNonLinear, dimensions, layers, usage, sharing.into(),
                             transform, alpha, mode, clipped, old_swapchain.map(|s| &**s))
    }

     /// Recreates the swapchain with new dimensions.
    pub fn recreate_with_dimension(&self, dimensions: [u32; 2])
                                   -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), OomError>
    {
        Swapchain::new_inner(&self.device, &self.surface, self.num_images, self.format,
                             self.color_space, dimensions, self.layers, &self.usage,
                             self.sharing.clone(), self.transform, self.alpha, self.mode,
                             self.clipped, Some(self))
    }

    // TODO: images layouts should always be set to "PRESENT", since we have no way to switch the
    //       layout at present time
    fn new_inner(device: &Arc<Device>, surface: &Arc<Surface>, num_images: u32, format: Format,
                 color_space: ColorSpace, dimensions: [u32; 2], layers: u32, usage: &ImageUsage,
                 sharing: SharingMode, transform: SurfaceTransform, alpha: CompositeAlpha,
                 mode: PresentMode, clipped: bool, old_swapchain: Option<&Swapchain>)
                 -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), OomError>
    {
        // Checking that the requested parameters match the capabilities.
        let capabilities = try!(surface.get_capabilities(&device.physical_device()));
        // TODO: return errors instead
        assert!(num_images >= capabilities.min_image_count);
        if let Some(c) = capabilities.max_image_count { assert!(num_images <= c) };
        assert!(capabilities.supported_formats.iter().any(|&(f, c)| f == format && c == color_space));
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
            // TODO: return proper error instead of panicing?
            assert_eq!(surface.internal_object(), sc.surface.internal_object());
        }

        // Checking that the surface doesn't already have a swapchain.
        if old_swapchain.is_none() {
            // TODO: return proper error instead of panicing?
            let has_already = surface.flag().swap(true, Ordering::AcqRel);
            if has_already { panic!("The surface already has a swapchain alive"); }
        }

        // FIXME: check that the device and the surface belong to the same instance
        let vk = device.pointers();
        assert!(device.loaded_extensions().khr_swapchain);     // TODO: return error instead

        assert!(usage.color_attachment);

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
                imageColorSpace: color_space as u32,
                imageExtent: vk::Extent2D { width: dimensions[0], height: dimensions[1] },
                imageArrayLayers: layers,
                imageUsage: usage.to_usage_bits(),
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
            stale: Mutex::new(false),
            num_images: num_images,
            format: format,
            color_space: color_space,
            dimensions: dimensions,
            layers: layers,
            usage: usage.clone(),
            sharing: sharing,
            transform: transform,
            alpha: alpha,
            mode: mode,
            clipped: clipped,
            images: Mutex::new(Vec::new()),     // Filled below.
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
            let unsafe_image = UnsafeImage::from_raw(device, image, usage.to_usage_bits(), format,
                                                     ImageDimensions::Dim2d { width: dimensions[0], height: dimensions[1], array_layers: 1, cubemap_compatible: false }, 1, 1);
            SwapchainImage::from_raw(unsafe_image, format, &swapchain, id as u32).unwrap()     // TODO: propagate error
        }).collect::<Vec<_>>();

        *swapchain.images.lock().unwrap() = images.iter().map(|i| (Arc::downgrade(i), true)).collect();
        Ok((swapchain, images))
    }

    /// Tries to take ownership of an image in order to draw on it.
    ///
    /// The function returns the index of the image in the array of images that was returned
    /// when creating the swapchain, plus a future that represents the moment when the image will
    /// become available from the GPU (which may not be *immediately*).
    ///
    /// If you try to draw on an image without acquiring it first, the execution will block. (TODO
    /// behavior may change).
    // TODO: has to make sure vkQueuePresent is called, because calling acquire_next_image many
    // times in a row is an error
    // TODO: swapchain must not have been replaced by being passed as the VkSwapchainCreateInfoKHR::oldSwapchain value to vkCreateSwapchainKHR
    pub fn acquire_next_image(&self, timeout: Duration) -> Result<(usize, SwapchainAcquireFuture), AcquireError> {
        unsafe {
            let stale = self.stale.lock().unwrap();
            if *stale {
                return Err(AcquireError::OutOfDate);
            }

            let vk = self.device.pointers();

            let semaphore = try!(Semaphore::new(self.device.clone()));

            let timeout_ns = timeout.as_secs().saturating_mul(1_000_000_000)
                                              .saturating_add(timeout.subsec_nanos() as u64);

            let mut out = mem::uninitialized();
            let r = try!(check_errors(vk.AcquireNextImageKHR(self.device.internal_object(),
                                                             self.swapchain, timeout_ns,
                                                             semaphore.internal_object(), 0,
                                                             &mut out)));

            let id = match r {
                Success::Success => out as usize,
                Success::Suboptimal => out as usize,        // TODO: give that info to the user
                Success::NotReady => return Err(AcquireError::Timeout),
                Success::Timeout => return Err(AcquireError::Timeout),
                s => panic!("unexpected success value: {:?}", s)
            };

            let mut images = self.images.lock().unwrap();
            let undefined_layout = mem::replace(&mut images.get_mut(id).unwrap().1, false);

            Ok((id, SwapchainAcquireFuture {
                semaphore: semaphore,
                id: id,
                image: images.get(id).unwrap().0.clone(),
                finished: AtomicBool::new(false),
                undefined_layout: undefined_layout,
            }))
        }
    }

    /// Presents an image on the screen.
    ///
    /// The parameter is the same index as what `acquire_next_image` returned. The image must
    /// have been acquired first.
    ///
    /// The actual behavior depends on the present mode that you passed when creating the
    /// swapchain.
    // TODO: use another API, since taking by Arc is meh
    pub fn present<F>(me: Arc<Self>, before: F, queue: Arc<Queue>, index: usize)
                      -> PresentFuture<F>
        where F: GpuFuture
    {
        assert!(index < me.num_images as usize);

        let swapchain_image = me.images.lock().unwrap().get(index).unwrap().0.upgrade().unwrap();       // TODO: return error instead
        // Normally if `check_image_access` returns false we're supposed to call the `gpu_access`
        // function on the image instead. But since we know that this method on `SwapchainImage`
        // always returns false anyway (by design), we don't need to do it.
        assert!(before.check_image_access(&swapchain_image, Layout::PresentSrc, true, &queue).is_ok());         // TODO: return error instead

        PresentFuture {
            previous: before,
            queue: queue,
            swapchain: me,
            image_id: index as u32,
            finished: AtomicBool::new(false),
        }
    }

    /// Returns the number of images of the swapchain.
    ///
    /// See the documentation of `Swapchain::new`. 
    #[inline]
    pub fn num_images(&self) -> u32 {
        self.num_images
    }

    /// Returns the format of the images of the swapchain.
    ///
    /// See the documentation of `Swapchain::new`. 
    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns the dimensions of the images of the swapchain.
    ///
    /// See the documentation of `Swapchain::new`. 
    #[inline]
    pub fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }

    /// Returns the number of layers of the images of the swapchain.
    ///
    /// See the documentation of `Swapchain::new`. 
    #[inline]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Returns the transform that was passed when creating the swapchain.
    ///
    /// See the documentation of `Swapchain::new`. 
    #[inline]
    pub fn transform(&self) -> SurfaceTransform {
        self.transform
    }

    /// Returns the alpha mode that was passed when creating the swapchain.
    ///
    /// See the documentation of `Swapchain::new`. 
    #[inline]
    pub fn composite_alpha(&self) -> CompositeAlpha {
        self.alpha
    }

    /// Returns the present mode that was passed when creating the swapchain.
    ///
    /// See the documentation of `Swapchain::new`. 
    #[inline]
    pub fn present_mode(&self) -> PresentMode {
        self.mode
    }

    /// Returns the value of `clipped` that was passed when creating the swapchain.
    ///
    /// See the documentation of `Swapchain::new`. 
    #[inline]
    pub fn clipped(&self) -> bool {
        self.clipped
    }
}

unsafe impl VulkanObject for Swapchain {
    type Object = vk::SwapchainKHR;

    #[inline]
    fn internal_object(&self) -> vk::SwapchainKHR {
        self.swapchain
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

/// Represents the moment when the GPU will have access to a swapchain image.
#[must_use]
pub struct SwapchainAcquireFuture {
    semaphore: Semaphore,
    id: usize,
    image: Weak<SwapchainImage>,
    finished: AtomicBool,
    // If true, then the acquired image is still in the undefined layout and must be transitionned.
    undefined_layout: bool,
}

impl SwapchainAcquireFuture {
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    #[inline]
    pub fn image_id(&self) -> usize {
        self.id
    }

    /// Returns the acquired image.
    #[inline]
    pub fn image(&self) -> Option<Arc<SwapchainImage>> {
        self.image.upgrade()
    }
}

unsafe impl GpuFuture for SwapchainAcquireFuture {
    #[inline]
    fn cleanup_finished(&mut self) {
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        let mut sem = SubmitSemaphoresWaitBuilder::new();
        sem.add_wait_semaphore(&self.semaphore);
        Ok(SubmitAnyBuilder::SemaphoresWait(sem))
    }

    #[inline]
    fn flush(&self) -> Result<(), FlushError> {
        Ok(())
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        self.finished.store(true, Ordering::SeqCst);
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        true
    }

    #[inline]
    fn queue(&self) -> Option<Arc<Queue>> {
        None
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        Err(AccessCheckError::Unknown)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: Layout, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        if let Some(sc_img) = self.image.upgrade() {
            if sc_img.inner().internal_object() != image.inner().internal_object() {
                return Err(AccessCheckError::Unknown);
            }

            if self.undefined_layout && layout != Layout::Undefined {
                return Err(AccessCheckError::Denied(AccessError::ImageNotInitialized {
                    requested: layout
                }));
            }

            if layout != Layout::Undefined && layout != Layout::PresentSrc {
                return Err(AccessCheckError::Denied(AccessError::UnexpectedImageLayout {
                    allowed: Layout::PresentSrc,
                    requested: layout,
                }));
            }

            Ok(None)

        } else {
            // The swapchain image no longer exists, therefore the `image` parameter received by
            // this function cannot possibly be the swapchain image.
            Err(AccessCheckError::Unknown)
        }
    }
}

unsafe impl DeviceOwned for SwapchainAcquireFuture {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.semaphore.device()
    }
}

impl Drop for SwapchainAcquireFuture {
    fn drop(&mut self) {
        if !*self.finished.get_mut() {
            panic!()        // FIXME: what to do?
            /*// TODO: handle errors?
            let fence = Fence::new(self.device().clone()).unwrap();
            let mut builder = SubmitCommandBufferBuilder::new();
            builder.add_wait_semaphore(&self.semaphore);
            builder.set_signal_fence(&fence);
            builder.submit(... which queue ? ...).unwrap();
            fence.wait(Duration::from_secs(600)).unwrap();*/
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

impl From<OomError> for AcquireError {
    #[inline]
    fn from(err: OomError) -> AcquireError {
        AcquireError::OomError(err)
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

/// Represents a swapchain image being presented on the screen.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct PresentFuture<P> where P: GpuFuture {
    previous: P,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    image_id: u32,
    finished: AtomicBool,
}

unsafe impl<P> GpuFuture for PresentFuture<P> where P: GpuFuture {
    #[inline]
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        let queue = self.previous.queue().map(|q| q.clone());

        // TODO: if the swapchain image layout is not PRESENT, should add a transition command
        // buffer

        Ok(match try!(self.previous.build_submission()) {
            SubmitAnyBuilder::Empty => {
                let mut builder = SubmitPresentBuilder::new();
                builder.add_swapchain(&self.swapchain, self.image_id);
                SubmitAnyBuilder::QueuePresent(builder)
            },
            SubmitAnyBuilder::SemaphoresWait(sem) => {
                let mut builder: SubmitPresentBuilder = sem.into();
                builder.add_swapchain(&self.swapchain, self.image_id);
                SubmitAnyBuilder::QueuePresent(builder)
            },
            SubmitAnyBuilder::CommandBuffer(cb) => {
                try!(cb.submit(&queue.unwrap()));        // FIXME: wrong because build_submission can be called multiple times
                let mut builder = SubmitPresentBuilder::new();
                builder.add_swapchain(&self.swapchain, self.image_id);
                SubmitAnyBuilder::QueuePresent(builder)
            },
            SubmitAnyBuilder::QueuePresent(present) => {
                unimplemented!()        // TODO:
                /*present.submit();
                let mut builder = SubmitPresentBuilder::new();
                builder.add_swapchain(self.command_buffer.inner(), self.image_id);
                SubmitAnyBuilder::CommandBuffer(builder)*/
            },
        })
    }

    #[inline]
    fn flush(&self) -> Result<(), FlushError> {
        unimplemented!()
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        self.finished.store(true, Ordering::SeqCst);
        self.previous.signal_finished();
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        false
    }

    #[inline]
    fn queue(&self) -> Option<Arc<Queue>> {
        debug_assert!(match self.previous.queue() {
            None => true,
            Some(q) => q.is_same(&self.queue)
        });

        Some(self.queue.clone())
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        unimplemented!()        // TODO: VK specs don't say whether it is legal to do that
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: Layout, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>
    {
        unimplemented!()        // TODO: VK specs don't say whether it is legal to do that
    }
}

unsafe impl<P> DeviceOwned for PresentFuture<P> where P: GpuFuture {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.queue.device()
    }
}

impl<P> Drop for PresentFuture<P> where P: GpuFuture {
    fn drop(&mut self) {
        unsafe {
            if !*self.finished.get_mut() {
                // TODO: handle errors?
                self.flush().unwrap();
                // Block until the queue finished.
                self.queue().unwrap().wait().unwrap();
                self.previous.signal_finished();
            }
        }
    }
}
