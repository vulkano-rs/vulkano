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
use image::ImageLayout;
use image::ImageUsage;
use image::swapchain::SwapchainImage;
use image::sys::UnsafeImage;
use swapchain::CapabilitiesError;
use swapchain::ColorSpace;
use swapchain::CompositeAlpha;
use swapchain::PresentMode;
use swapchain::Surface;
use swapchain::SurfaceSwapchainLock;
use swapchain::SurfaceTransform;
use sync::AccessCheckError;
use sync::AccessError;
use sync::AccessFlagBits;
use sync::FlushError;
use sync::GpuFuture;
use sync::PipelineStages;
use sync::Semaphore;
use sync::SharingMode;

use Error;
use OomError;
use Success;
use VulkanObject;
use check_errors;
use vk;

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
pub fn acquire_next_image(swapchain: Arc<Swapchain>, timeout: Option<Duration>)
                          -> Result<(usize, SwapchainAcquireFuture), AcquireError> {
    unsafe {
        // Check that this is not an old swapchain. From specs:
        // > swapchain must not have been replaced by being passed as the
        // > VkSwapchainCreateInfoKHR::oldSwapchain value to vkCreateSwapchainKHR
        let stale = swapchain.stale.lock().unwrap();
        if *stale {
            return Err(AcquireError::OutOfDate);
        }

        let vk = swapchain.device.pointers();

        let semaphore = Semaphore::new(swapchain.device.clone())?;

        let timeout_ns = if let Some(timeout) = timeout {
            timeout
                .as_secs()
                .saturating_mul(1_000_000_000)
                .saturating_add(timeout.subsec_nanos() as u64)
        } else {
            u64::max_value()
        };

        let mut out = mem::uninitialized();
        let r = check_errors(vk.AcquireNextImageKHR(swapchain.device.internal_object(),
                                                    swapchain.swapchain,
                                                    timeout_ns,
                                                    semaphore.internal_object(),
                                                    0,
                                                    &mut out))?;

        let id = match r {
            Success::Success => out as usize,
            Success::Suboptimal => out as usize,        // TODO: give that info to the user
            Success::NotReady => return Err(AcquireError::Timeout),
            Success::Timeout => return Err(AcquireError::Timeout),
            s => panic!("unexpected success value: {:?}", s),
        };

        Ok((id,
            SwapchainAcquireFuture {
                swapchain: swapchain.clone(), // TODO: don't clone
                semaphore: semaphore,
                image_id: id,
                finished: AtomicBool::new(false),
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
pub fn present<F>(swapchain: Arc<Swapchain>, before: F, queue: Arc<Queue>, index: usize)
                  -> PresentFuture<F>
    where F: GpuFuture
{
    assert!(index < swapchain.images.len());

    // TODO: restore this check with a dummy ImageAccess implementation
    /*let swapchain_image = me.images.lock().unwrap().get(index).unwrap().0.upgrade().unwrap();       // TODO: return error instead
    // Normally if `check_image_access` returns false we're supposed to call the `gpu_access`
    // function on the image instead. But since we know that this method on `SwapchainImage`
    // always returns false anyway (by design), we don't need to do it.
    assert!(before.check_image_access(&swapchain_image, ImageLayout::PresentSrc, true, &queue).is_ok());         // TODO: return error instead*/

    PresentFuture {
        previous: before,
        queue: queue,
        swapchain: swapchain,
        image_id: index,
        flushed: AtomicBool::new(false),
        finished: AtomicBool::new(false),
    }
}

/// Contains the swapping system and the images that can be shown on a surface.
pub struct Swapchain {
    // The Vulkan device this swapchain was created with.
    device: Arc<Device>,
    // The surface, which we need to keep alive.
    surface: Arc<Surface>,
    // The swapchain object.
    swapchain: vk::SwapchainKHR,

    // The images of this swapchain.
    images: Vec<ImageEntry>,

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
}

struct ImageEntry {
    image: UnsafeImage,
    // If true, then the image is still in the undefined layout and must be transitionned.
    undefined_layout: AtomicBool,
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
    /// - Panics if `usage` is empty.
    ///
    // TODO: remove `old_swapchain` parameter and add another function `with_old_swapchain`.
    // TODO: add `ColorSpace` parameter
    // TODO: isn't it unsafe to take the surface through an Arc when it comes to vulkano-win?
    #[inline]
    pub fn new<F, S>(
        device: Arc<Device>, surface: Arc<Surface>, num_images: u32, format: F,
        dimensions: [u32; 2], layers: u32, usage: ImageUsage, sharing: S,
        transform: SurfaceTransform, alpha: CompositeAlpha, mode: PresentMode, clipped: bool,
        old_swapchain: Option<&Arc<Swapchain>>)
        -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), SwapchainCreationError>
        where F: FormatDesc,
              S: Into<SharingMode>
    {
        Swapchain::new_inner(device,
                             surface,
                             num_images,
                             format.format(),
                             ColorSpace::SrgbNonLinear,
                             dimensions,
                             layers,
                             usage,
                             sharing.into(),
                             transform,
                             alpha,
                             mode,
                             clipped,
                             old_swapchain.map(|s| &**s))
    }

    /// Recreates the swapchain with new dimensions.
    pub fn recreate_with_dimension(
        &self, dimensions: [u32; 2])
        -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), SwapchainCreationError> {
        Swapchain::new_inner(self.device.clone(),
                             self.surface.clone(),
                             self.num_images,
                             self.format,
                             self.color_space,
                             dimensions,
                             self.layers,
                             self.usage,
                             self.sharing.clone(),
                             self.transform,
                             self.alpha,
                             self.mode,
                             self.clipped,
                             Some(self))
    }

    fn new_inner(device: Arc<Device>, surface: Arc<Surface>, num_images: u32, format: Format,
                 color_space: ColorSpace, dimensions: [u32; 2], layers: u32, usage: ImageUsage,
                 sharing: SharingMode, transform: SurfaceTransform, alpha: CompositeAlpha,
                 mode: PresentMode, clipped: bool, old_swapchain: Option<&Swapchain>)
                 -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), SwapchainCreationError> {
        assert_eq!(device.instance().internal_object(),
                   surface.instance().internal_object());

        // Checking that the requested parameters match the capabilities.
        let capabilities = surface.capabilities(device.physical_device())?;
        if num_images < capabilities.min_image_count {
            return Err(SwapchainCreationError::UnsupportedMinImagesCount);
        }
        if let Some(c) = capabilities.max_image_count {
            if num_images > c {
                return Err(SwapchainCreationError::UnsupportedMaxImagesCount);
            }
        }
        if !capabilities
            .supported_formats
            .iter()
            .any(|&(f, c)| f == format && c == color_space)
        {
            return Err(SwapchainCreationError::UnsupportedFormat);
        }
        if dimensions[0] < capabilities.min_image_extent[0] {
            return Err(SwapchainCreationError::UnsupportedDimensions);
        }
        if dimensions[1] < capabilities.min_image_extent[1] {
            return Err(SwapchainCreationError::UnsupportedDimensions);
        }
        if dimensions[0] > capabilities.max_image_extent[0] {
            return Err(SwapchainCreationError::UnsupportedDimensions);
        }
        if dimensions[1] > capabilities.max_image_extent[1] {
            return Err(SwapchainCreationError::UnsupportedDimensions);
        }
        if layers < 1 && layers > capabilities.max_image_array_layers {
            return Err(SwapchainCreationError::UnsupportedArrayLayers);
        }
        if (usage.to_usage_bits() & capabilities.supported_usage_flags.to_usage_bits()) !=
            usage.to_usage_bits()
        {
            return Err(SwapchainCreationError::UnsupportedUsageFlags);
        }
        if !capabilities.supported_transforms.supports(transform) {
            return Err(SwapchainCreationError::UnsupportedSurfaceTransform);
        }
        if !capabilities.supported_composite_alpha.supports(alpha) {
            return Err(SwapchainCreationError::UnsupportedCompositeAlpha);
        }
        if !capabilities.present_modes.supports(mode) {
            return Err(SwapchainCreationError::UnsupportedPresentMode);
        }

        // If we recreate a swapchain, make sure that the surface is the same.
        if let Some(sc) = old_swapchain {
            if surface.internal_object() != sc.surface.internal_object() {
                return Err(SwapchainCreationError::OldSwapchainSurfaceMismatch);
            }
        }

        // Checking that the surface doesn't already have a swapchain.
        if old_swapchain.is_none() {
            let has_already = surface.flag().swap(true, Ordering::AcqRel);
            if has_already {
                return Err(SwapchainCreationError::SurfaceInUse);
            }
        }

        if !device.loaded_extensions().khr_swapchain {
            return Err(SwapchainCreationError::MissingExtension);
        }

        // Required by the specs.
        assert_ne!(usage, ImageUsage::none());

        if let Some(ref old_swapchain) = old_swapchain {
            *old_swapchain.stale.lock().unwrap() = false;
        }

        let vk = device.pointers();

        let swapchain = unsafe {
            let (sh_mode, sh_count, sh_indices) = match sharing {
                SharingMode::Exclusive(_) => (vk::SHARING_MODE_EXCLUSIVE, 0, ptr::null()),
                SharingMode::Concurrent(ref ids) => (vk::SHARING_MODE_CONCURRENT,
                                                     ids.len() as u32,
                                                     ids.as_ptr()),
            };

            let infos = vk::SwapchainCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0, // reserved
                surface: surface.internal_object(),
                minImageCount: num_images,
                imageFormat: format as u32,
                imageColorSpace: color_space as u32,
                imageExtent: vk::Extent2D {
                    width: dimensions[0],
                    height: dimensions[1],
                },
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
            check_errors(vk.CreateSwapchainKHR(device.internal_object(),
                                               &infos,
                                               ptr::null(),
                                               &mut output))?;
            output
        };

        let image_handles = unsafe {
            let mut num = 0;
            check_errors(vk.GetSwapchainImagesKHR(device.internal_object(),
                                                  swapchain,
                                                  &mut num,
                                                  ptr::null_mut()))?;

            let mut images = Vec::with_capacity(num as usize);
            check_errors(vk.GetSwapchainImagesKHR(device.internal_object(),
                                                  swapchain,
                                                  &mut num,
                                                  images.as_mut_ptr()))?;
            images.set_len(num as usize);
            images
        };

        let images = image_handles
            .into_iter()
            .enumerate()
            .map(|(id, image)| unsafe {
                let dims = ImageDimensions::Dim2d {
                    width: dimensions[0],
                    height: dimensions[1],
                    array_layers: layers,
                    cubemap_compatible: false,
                };

                let img = UnsafeImage::from_raw(device.clone(),
                                                image,
                                                usage.to_usage_bits(),
                                                format,
                                                dims,
                                                1,
                                                1);

                ImageEntry {
                    image: img,
                    undefined_layout: AtomicBool::new(true),
                }
            })
            .collect::<Vec<_>>();

        let swapchain = Arc::new(Swapchain {
                                     device: device.clone(),
                                     surface: surface.clone(),
                                     swapchain: swapchain,
                                     images: images,
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
                                 });

        let swapchain_images = unsafe {
            let mut swapchain_images = Vec::with_capacity(swapchain.images.len());
            for n in 0 .. swapchain.images.len() {
                swapchain_images.push(SwapchainImage::from_raw(swapchain.clone(), n)?);
            }
            swapchain_images
        };

        Ok((swapchain, swapchain_images))
    }

    /// Returns of the images that belong to this swapchain.
    #[inline]
    pub fn raw_image(&self, offset: usize) -> Option<&UnsafeImage> {
        self.images.get(offset).map(|i| &i.image)
    }

    /// Returns the number of images of the swapchain.
    ///
    /// See the documentation of `Swapchain::new`.
    #[inline]
    pub fn num_images(&self) -> u32 {
        self.images.len() as u32
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

impl fmt::Debug for Swapchain {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan swapchain {:?}>", self.swapchain)
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

/// Error that can happen when creation a swapchain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwapchainCreationError {
    /// Not enough memory.
    OomError(OomError),
    /// The device was lost.
    DeviceLost,
    /// The surface was lost.
    SurfaceLost,
    /// The surface is already used by another swapchain.
    SurfaceInUse,
    /// The window is already in use by another API.
    NativeWindowInUse,
    /// The `VK_KHR_swapchain` extension was not enabled.
    MissingExtension,
    /// Surface mismatch between old and new swapchain.
    OldSwapchainSurfaceMismatch,
    /// The requested number of swapchain images is not supported by the surface.
    UnsupportedMinImagesCount,
    /// The requested number of swapchain images is not supported by the surface.
    UnsupportedMaxImagesCount,
    /// The requested image format is not supported by the surface.
    UnsupportedFormat,
    /// The requested dimensions are not supported by the surface.
    UnsupportedDimensions,
    /// The requested array layers count is not supported by the surface.
    UnsupportedArrayLayers,
    /// The requested image usage is not supported by the surface.
    UnsupportedUsageFlags,
    /// The requested surface transform is not supported by the surface.
    UnsupportedSurfaceTransform,
    /// The requested composite alpha is not supported by the surface.
    UnsupportedCompositeAlpha,
    /// The requested present mode is not supported by the surface.
    UnsupportedPresentMode,
}

impl error::Error for SwapchainCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            SwapchainCreationError::OomError(_) => {
                "not enough memory available"
            },
            SwapchainCreationError::DeviceLost => {
                "the device was lost"
            },
            SwapchainCreationError::SurfaceLost => {
                "the surface was lost"
            },
            SwapchainCreationError::SurfaceInUse => {
                "the surface is already used by another swapchain"
            },
            SwapchainCreationError::NativeWindowInUse => {
                "the window is already in use by another API"
            },
            SwapchainCreationError::MissingExtension => {
                "the `VK_KHR_swapchain` extension was not enabled"
            },
            SwapchainCreationError::OldSwapchainSurfaceMismatch => {
                "surface mismatch between old and new swapchain"
            },
            SwapchainCreationError::UnsupportedMinImagesCount => {
                "the requested number of swapchain images is not supported by the surface"
            },
            SwapchainCreationError::UnsupportedMaxImagesCount => {
                "the requested number of swapchain images is not supported by the surface"
            },
            SwapchainCreationError::UnsupportedFormat => {
                "the requested image format is not supported by the surface"
            },
            SwapchainCreationError::UnsupportedDimensions => {
                "the requested dimensions are not supported by the surface"
            },
            SwapchainCreationError::UnsupportedArrayLayers => {
                "the requested array layers count is not supported by the surface"
            },
            SwapchainCreationError::UnsupportedUsageFlags => {
                "the requested image usage is not supported by the surface"
            },
            SwapchainCreationError::UnsupportedSurfaceTransform => {
                "the requested surface transform is not supported by the surface"
            },
            SwapchainCreationError::UnsupportedCompositeAlpha => {
                "the requested composite alpha is not supported by the surface"
            },
            SwapchainCreationError::UnsupportedPresentMode => {
                "the requested present mode is not supported by the surface"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            SwapchainCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SwapchainCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for SwapchainCreationError {
    #[inline]
    fn from(err: Error) -> SwapchainCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                SwapchainCreationError::OomError(OomError::from(err))
            },
            err @ Error::OutOfDeviceMemory => {
                SwapchainCreationError::OomError(OomError::from(err))
            },
            err @ Error::DeviceLost => {
                SwapchainCreationError::DeviceLost
            },
            err @ Error::SurfaceLost => {
                SwapchainCreationError::SurfaceLost
            },
            err @ Error::NativeWindowInUse => {
                SwapchainCreationError::NativeWindowInUse
            },
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for SwapchainCreationError {
    #[inline]
    fn from(err: OomError) -> SwapchainCreationError {
        SwapchainCreationError::OomError(err)
    }
}

impl From<CapabilitiesError> for SwapchainCreationError {
    #[inline]
    fn from(err: CapabilitiesError) -> SwapchainCreationError {
        match err {
            CapabilitiesError::OomError(err) => SwapchainCreationError::OomError(err),
            CapabilitiesError::SurfaceLost => SwapchainCreationError::SurfaceLost,
        }
    }
}

/// Represents the moment when the GPU will have access to a swapchain image.
#[must_use]
pub struct SwapchainAcquireFuture {
    swapchain: Arc<Swapchain>,
    image_id: usize,
    semaphore: Semaphore,
    finished: AtomicBool,
}

impl SwapchainAcquireFuture {
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    #[inline]
    pub fn image_id(&self) -> usize {
        self.image_id
    }

    /// Returns the corresponding swapchain.
    #[inline]
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
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
    fn check_buffer_access(
        &self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
        -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        Err(AccessCheckError::Unknown)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool,
                          queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        let swapchain_image = self.swapchain.raw_image(self.image_id).unwrap();
        if swapchain_image.internal_object() != image.inner().internal_object() {
            return Err(AccessCheckError::Unknown);
        }

        if self.swapchain.images[self.image_id]
            .undefined_layout
            .load(Ordering::Relaxed) && layout != ImageLayout::Undefined
        {
            return Err(AccessCheckError::Denied(AccessError::ImageNotInitialized {
                                                    requested: layout,
                                                }));
        }

        if layout != ImageLayout::Undefined && layout != ImageLayout::PresentSrc {
            return Err(AccessCheckError::Denied(AccessError::UnexpectedImageLayout {
                                                    allowed: ImageLayout::PresentSrc,
                                                    requested: layout,
                                                }));
        }

        Ok(None)
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
            panic!() // FIXME: what to do?
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
            _ => None,
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
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// Represents a swapchain image being presented on the screen.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct PresentFuture<P>
    where P: GpuFuture
{
    previous: P,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    image_id: usize,
    // True if `flush()` has been called on the future, which means that the present command has
    // been submitted.
    flushed: AtomicBool,
    // True if `signal_finished()` has been called on the future, which means that the future has
    // been submitted and has already been processed by the GPU.
    finished: AtomicBool,
}

impl<P> PresentFuture<P>
    where P: GpuFuture
{
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    #[inline]
    pub fn image_id(&self) -> usize {
        self.image_id
    }

    /// Returns the corresponding swapchain.
    #[inline]
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
    }
}

unsafe impl<P> GpuFuture for PresentFuture<P>
    where P: GpuFuture
{
    #[inline]
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        if self.flushed.load(Ordering::SeqCst) {
            return Ok(SubmitAnyBuilder::Empty);
        }

        let queue = self.previous.queue().map(|q| q.clone());

        // TODO: if the swapchain image layout is not PRESENT, should add a transition command
        // buffer

        Ok(match self.previous.build_submission()? {
               SubmitAnyBuilder::Empty => {
                   let mut builder = SubmitPresentBuilder::new();
                   builder.add_swapchain(&self.swapchain, self.image_id as u32);
                   SubmitAnyBuilder::QueuePresent(builder)
               },
               SubmitAnyBuilder::SemaphoresWait(sem) => {
                   let mut builder: SubmitPresentBuilder = sem.into();
                   builder.add_swapchain(&self.swapchain, self.image_id as u32);
                   SubmitAnyBuilder::QueuePresent(builder)
               },
               SubmitAnyBuilder::CommandBuffer(cb) => {
                   cb.submit(&queue.unwrap())?; // FIXME: wrong because build_submission can be called multiple times
                   let mut builder = SubmitPresentBuilder::new();
                   builder.add_swapchain(&self.swapchain, self.image_id as u32);
                   SubmitAnyBuilder::QueuePresent(builder)
               },
               SubmitAnyBuilder::BindSparse(cb) => {
                   cb.submit(&queue.unwrap())?; // FIXME: wrong because build_submission can be called multiple times
                   let mut builder = SubmitPresentBuilder::new();
                   builder.add_swapchain(&self.swapchain, self.image_id as u32);
                   SubmitAnyBuilder::QueuePresent(builder)
               },
               SubmitAnyBuilder::QueuePresent(present) => {
                   unimplemented!() // TODO:
                /*present.submit();
                let mut builder = SubmitPresentBuilder::new();
                builder.add_swapchain(self.command_buffer.inner(), self.image_id);
                SubmitAnyBuilder::CommandBuffer(builder)*/
               },
           })
    }

    #[inline]
    fn flush(&self) -> Result<(), FlushError> {
        unsafe {
            // If `flushed` already contains `true`, then `build_submission` will return `Empty`.

            match self.build_submission()? {
                SubmitAnyBuilder::Empty => {},
                SubmitAnyBuilder::QueuePresent(present) => {
                    present.submit(&self.queue)?;
                },
                _ => unreachable!(),
            }

            self.flushed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        self.flushed.store(true, Ordering::SeqCst);
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
                          Some(q) => q.is_same(&self.queue),
                      });

        Some(self.queue.clone())
    }

    #[inline]
    fn check_buffer_access(
        &self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
        -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        self.previous.check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool,
                          queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        let swapchain_image = self.swapchain.raw_image(self.image_id).unwrap();
        if swapchain_image.internal_object() == image.inner().internal_object() {
            // This future presents the swapchain image, which "unlocks" it. Therefore any attempt
            // to use this swapchain image afterwards shouldn't get granted automatic access.
            // Instead any attempt to access the image afterwards should get an authorization from
            // a later swapchain acquire future. Hence why we return `Unknown` here.
            Err(AccessCheckError::Unknown)
        } else {
            self.previous
                .check_image_access(image, layout, exclusive, queue)
        }
    }
}

unsafe impl<P> DeviceOwned for PresentFuture<P>
    where P: GpuFuture
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.queue.device()
    }
}

impl<P> Drop for PresentFuture<P>
    where P: GpuFuture
{
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
