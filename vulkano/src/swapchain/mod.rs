// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Link between Vulkan and a window and/or the screen.
//!
//! Before you can draw on the screen or a window, you have to create two objects:
//!
//! - Create a `Surface` object that represents the location where the image will show up (either
//!   a window or a monitor).
//! - Create a `Swapchain` that uses that `Surface`.
//!
//! Creating a surface can be done with only an `Instance` object. However creating a swapchain
//! requires a `Device` object.
//!
//! Once you have a swapchain, you can retrieve `Image` objects from it and draw to them just like
//! you would draw on any other image.
//!
//! # Surfaces
//!
//! A surface is an object that represents a location where to render. It can be created from an
//! instance and either a window handle (in a platform-specific way) or a monitor.
//!
//! In order to use surfaces, you will have to enable the `VK_KHR_surface` extension on the
//! instance. See the `instance` module for more information about how to enable extensions.
//!
//! ## Creating a surface from a window
//!
//! There are 5 extensions that each allow you to create a surface from a type of window:
//!
//! - `VK_KHR_xlib_surface`
//! - `VK_KHR_xcb_surface`
//! - `VK_KHR_wayland_surface`
//! - `VK_KHR_android_surface`
//! - `VK_KHR_win32_surface`
//!
//! For example if you want to create a surface from an Android surface, you will have to enable
//! the `VK_KHR_android_surface` extension and use `Surface::from_android`.
//! See the documentation of `Surface` for all the possible constructors.
//!
//! Trying to use one of these functions without enabling the proper extension will result in an
//! error.
//!
//! **Note that the `Surface` object is potentially unsafe**. It is your responsibility to
//! keep the window alive for at least as long as the surface exists. In many cases Surface
//! may be able to do this for you, if you pass it ownership of your Window (or a
//! reference-counting container for it).
//!
//! ### Examples
//!
//! ```no_run
//! use std::ptr;
//! use vulkano::{
//!     instance::{Instance, InstanceCreateInfo, InstanceExtensions},
//!     swapchain::Surface,
//!     Version, VulkanLibrary,
//! };
//!
//! let instance = {
//!     let library = VulkanLibrary::new()
//!         .unwrap_or_else(|err| panic!("Couldn't load Vulkan library: {:?}", err));
//!
//!     let extensions = InstanceExtensions {
//!         khr_surface: true,
//!         khr_win32_surface: true,        // If you don't enable this, `from_hwnd` will fail.
//!         .. InstanceExtensions::empty()
//!     };
//!
//!     Instance::new(
//!         library,
//!         InstanceCreateInfo {
//!             enabled_extensions: extensions,
//!             ..Default::default()
//!         },
//!     )
//!     .unwrap_or_else(|err| panic!("Couldn't create instance: {:?}", err))
//! };
//!
//! # use std::sync::Arc;
//! # struct Window(*const u32);
//! # impl Window { fn hwnd(&self) -> *const u32 { self.0 } }
//! # unsafe impl Send for Window {}
//! # unsafe impl Sync for Window {}
//! # fn build_window() -> Arc<Window> { Arc::new(Window(ptr::null())) }
//! let window = build_window(); // Third-party function, not provided by vulkano
//! let _surface = unsafe {
//!     let hinstance: *const () = ptr::null(); // Windows-specific object
//!     Surface::from_win32(
//!         instance.clone(),
//!         hinstance, window.hwnd(),
//!         Some(window),
//!     ).unwrap()
//! };
//! ```
//!
//! ## Creating a surface from a monitor
//!
//! Currently no system provides the `VK_KHR_display` extension that contains this feature.
//! This feature is still a work-in-progress in vulkano and will reside in the `display` module.
//!
//! # Swapchains
//!
//! A surface represents a location on the screen and can be created from an instance. Once you
//! have a surface, the next step is to create a swapchain. Creating a swapchain requires a device,
//! and allocates the resources that will be used to display images on the screen.
//!
//! A swapchain is composed of one or multiple images. Each image of the swapchain is presented in
//! turn on the screen, one after another. More information below.
//!
//! Swapchains have several properties:
//!
//!  - The number of images that will cycle on the screen.
//!  - The format of the images.
//!  - The 2D dimensions of the images, plus a number of layers, for a total of three dimensions.
//!  - The usage of the images, similar to creating other images.
//!  - The queue families that are going to use the images, similar to creating other images.
//!  - An additional transformation (rotation or mirroring) to perform on the final output.
//!  - How the alpha of the final output will be interpreted.
//!  - How to perform the cycling between images in regard to vsync.
//!
//! You can query the supported values of all these properties from the physical device.
//!
//! ## Creating a swapchain
//!
//! In order to create a swapchain, you will first have to enable the `VK_KHR_swapchain` extension
//! on the device (and not on the instance like `VK_KHR_surface`):
//!
//! ```no_run
//! # use vulkano::device::DeviceExtensions;
//! let ext = DeviceExtensions {
//!     khr_swapchain: true,
//!     .. DeviceExtensions::empty()
//! };
//! ```
//!
//! Then, query the capabilities of the surface with
//! [`PhysicalDevice::surface_capabilities`](crate::device::physical::PhysicalDevice::surface_capabilities)
//! and
//! [`PhysicalDevice::surface_formats`](crate::device::physical::PhysicalDevice::surface_formats)
//! and choose which values you are going to use.
//!
//! ```no_run
//! # use std::{error::Error, sync::Arc};
//! # use vulkano::device::Device;
//! # use vulkano::swapchain::Surface;
//! # use std::cmp::{max, min};
//! # fn choose_caps(device: Arc<Device>, surface: Arc<Surface>) -> Result<(), Box<dyn Error>> {
//! let surface_capabilities = device
//!     .physical_device()
//!     .surface_capabilities(&surface, Default::default())?;
//!
//! // Use the current window size or some fixed resolution.
//! let image_extent = surface_capabilities.current_extent.unwrap_or([640, 480]);
//!
//! // Try to use double-buffering.
//! let min_image_count = match surface_capabilities.max_image_count {
//!     None => max(2, surface_capabilities.min_image_count),
//!     Some(limit) => min(max(2, surface_capabilities.min_image_count), limit)
//! };
//!
//! // Preserve the current surface transform.
//! let pre_transform = surface_capabilities.current_transform;
//!
//! // Use the first available format.
//! let (image_format, color_space) = device
//!     .physical_device()
//!     .surface_formats(&surface, Default::default())?[0];
//! # Ok(())
//! # }
//! ```
//!
//! Then, call [`Swapchain::new`](crate::swapchain::Swapchain::new).
//!
//! ```no_run
//! # use std::{error::Error, sync::Arc};
//! # use vulkano::device::{Device, Queue};
//! # use vulkano::image::ImageUsage;
//! # use vulkano::sync::SharingMode;
//! # use vulkano::format::Format;
//! # use vulkano::swapchain::{Surface, Swapchain, SurfaceTransform, PresentMode, CompositeAlpha, ColorSpace, FullScreenExclusive, SwapchainCreateInfo};
//! # fn create_swapchain(
//! #     device: Arc<Device>, surface: Arc<Surface>,
//! #     min_image_count: u32, image_format: Format, image_extent: [u32; 2],
//! #     pre_transform: SurfaceTransform, composite_alpha: CompositeAlpha,
//! #     present_mode: PresentMode, full_screen_exclusive: FullScreenExclusive
//! # ) -> Result<(), Box<dyn Error>> {
//! // The created swapchain will be used as a color attachment for rendering.
//! let image_usage = ImageUsage {
//!     color_attachment: true,
//!     .. ImageUsage::empty()
//! };
//!
//! // Create the swapchain and its images.
//! let (swapchain, images) = Swapchain::new(
//!     // Create the swapchain in this `device`'s memory.
//!     device,
//!     // The surface where the images will be presented.
//!     surface,
//!     // The creation parameters.
//!     SwapchainCreateInfo {
//!         // How many images to use in the swapchain.
//!         min_image_count,
//!         // The format of the images.
//!         image_format: Some(image_format),
//!         // The size of each image.
//!         image_extent,
//!         // What the images are going to be used for.
//!         image_usage,
//!         // What transformation to use with the surface.
//!         pre_transform,
//!         // How to handle the alpha channel.
//!         composite_alpha,
//!         // How to present images.
//!         present_mode,
//!         // How to handle full-screen exclusivity
//!         full_screen_exclusive,
//!         ..Default::default()
//!     }
//! )?;
//!
//! # Ok(())
//! # }
//! ```
//!
//! Creating a swapchain not only returns the swapchain object, but also all the images that belong
//! to it.
//!
//! ## Acquiring and presenting images
//!
//! Once you created a swapchain and retrieved all the images that belong to it (see previous
//! section), you can draw on it. This is done in three steps:
//!
//!  - Call `swapchain::acquire_next_image`. This function will return the index of the image
//!    (within the list returned by `Swapchain::new`) that is available to draw, plus a future
//!    representing the moment when the GPU will gain access to that image.
//!  - Draw on that image just like you would draw to any other image (see the documentation of
//!    the `pipeline` module). You need to chain the draw after the future that was returned by
//!    `acquire_next_image`.
//!  - Call `Swapchain::present` with the same index and by chaining the futures, in order to tell
//!    the implementation that you are finished drawing to the image and that it can queue a
//!    command to present the image on the screen after the draw operations are finished.
//!
//! ```
//! use vulkano::swapchain::{self, SwapchainPresentInfo};
//! use vulkano::sync::GpuFuture;
//! # let queue: ::std::sync::Arc<::vulkano::device::Queue> = return;
//! # let mut swapchain: ::std::sync::Arc<swapchain::Swapchain> = return;
//! // let mut (swapchain, images) = Swapchain::new(...);
//! loop {
//!     # let mut command_buffer: ::vulkano::command_buffer::PrimaryAutoCommandBuffer = return;
//!     let (image_index, suboptimal, acquire_future)
//!         = swapchain::acquire_next_image(swapchain.clone(), None).unwrap();
//!
//!     // The command_buffer contains the draw commands that modify the framebuffer
//!     // constructed from images[image_index]
//!     acquire_future
//!         .then_execute(queue.clone(), command_buffer)
//!         .unwrap()
//!         .then_swapchain_present(
//!             queue.clone(),
//!             SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
//!         )
//!         .then_signal_fence_and_flush()
//!         .unwrap();
//! }
//! ```
//!
//! ## Recreating a swapchain
//!
//! In some situations, the swapchain will become invalid by itself. This includes for example when
//! the window is resized (as the images of the swapchain will no longer match the window's) or,
//! on Android, when the application went to the background and goes back to the foreground.
//!
//! In this situation, acquiring a swapchain image or presenting it will return an error. Rendering
//! to an image of that swapchain will not produce any error, but may or may not work. To continue
//! rendering, you will need to *recreate* the swapchain by creating a new swapchain and passing
//! as last parameter the old swapchain.
//!
//! ```
//! use vulkano::swapchain;
//! use vulkano::swapchain::{AcquireError, SwapchainCreateInfo, SwapchainPresentInfo};
//! use vulkano::sync::GpuFuture;
//!
//! // let (swapchain, images) = Swapchain::new(...);
//! # let mut swapchain: ::std::sync::Arc<::vulkano::swapchain::Swapchain> = return;
//! # let mut images: Vec<::std::sync::Arc<::vulkano::image::SwapchainImage>> = return;
//! # let queue: ::std::sync::Arc<::vulkano::device::Queue> = return;
//! let mut recreate_swapchain = false;
//!
//! loop {
//!     if recreate_swapchain {
//!         let (new_swapchain, new_images) = swapchain.recreate(SwapchainCreateInfo {
//!             image_extent: [1024, 768],
//!             ..swapchain.create_info()
//!         })
//!         .unwrap();
//!         swapchain = new_swapchain;
//!         images = new_images;
//!         recreate_swapchain = false;
//!     }
//!
//!     let (image_index, suboptimal, acq_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
//!         Ok(r) => r,
//!         Err(AcquireError::OutOfDate) => { recreate_swapchain = true; continue; },
//!         Err(err) => panic!("{:?}", err),
//!     };
//!
//!     // ...
//!
//!     let final_future = acq_future
//!         // .then_execute(...)
//!         .then_swapchain_present(
//!             queue.clone(),
//!             SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
//!         )
//!         .then_signal_fence_and_flush()
//!         .unwrap(); // TODO: PresentError?
//!
//!     if suboptimal {
//!         recreate_swapchain = true;
//!     }
//! }
//! ```

pub use self::{
    surface::{
        ColorSpace, CompositeAlpha, PresentMode, SupportedCompositeAlpha,
        SupportedSurfaceTransforms, Surface, SurfaceApi, SurfaceCapabilities, SurfaceCreationError,
        SurfaceInfo, SurfaceTransform,
    },
    swapchain::{
        acquire_next_image, acquire_next_image_raw, present, wait_for_present, AcquireError,
        AcquiredImage, FullScreenExclusive, FullScreenExclusiveError, PresentFuture,
        PresentWaitError, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
        SwapchainCreationError, Win32Monitor,
    },
};
#[cfg(target_os = "ios")]
pub use surface::IOSMetalLayer;

use crate::sync::Semaphore;
use std::{
    num::NonZeroU64,
    sync::{atomic::AtomicBool, Arc},
};

pub mod display;
mod surface;
mod swapchain;

/// Parameters to execute present operations on a queue.
#[derive(Clone, Debug)]
pub struct PresentInfo {
    /// The semaphores to wait for before beginning the execution of the present operations.
    ///
    /// The default value is empty.
    pub wait_semaphores: Vec<Arc<Semaphore>>,

    /// The present operations to perform.
    ///
    /// The default value is empty.
    pub swapchain_infos: Vec<SwapchainPresentInfo>,

    pub _ne: crate::NonExhaustive,
}

impl Default for PresentInfo {
    #[inline]
    fn default() -> Self {
        Self {
            wait_semaphores: Vec::new(),
            swapchain_infos: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters for a single present operation on a swapchain.
#[derive(Clone, Debug)]
pub struct SwapchainPresentInfo {
    /// The swapchain to present to.
    ///
    /// There is no default value.
    pub swapchain: Arc<Swapchain>,

    /// The index of the swapchain image to present to.
    ///
    /// The image must have been acquired first; this is the index that `acquire_next_image`
    /// returns.
    ///
    /// There is no default value.
    pub image_index: u32,

    /// An id used to identify this present operation.
    ///
    /// If `present_id` is `Some`, the [`present_id`](crate::device::Features::present_id) feature
    /// must be enabled on the device. The id must be greater than any id previously used for
    /// `swapchain`. If a swapchain is recreated, this resets.
    ///
    /// The default value is `None`.
    pub present_id: Option<NonZeroU64>,

    /// Am optimization hint to the implementation, that only some parts of the swapchain image are
    /// going to be updated by the present operation.
    ///
    /// If `present_regions` is not empty, the
    /// [`khr_incremental_present`](crate::device::DeviceExtensions::khr_incremental_present)
    /// extension must be enabled on the device. The implementation will update the provided
    /// regions of the swapchain image, and _may_ ignore the other areas. However, as this is just
    /// a hint, the Vulkan implementation is free to ignore the regions altogether and update
    /// everything.
    ///
    /// If `present_regions` is empty, that means that all of the swapchain image must be updated.
    ///
    /// The default value is empty.
    pub present_regions: Vec<RectangleLayer>,

    pub _ne: crate::NonExhaustive,
}

impl SwapchainPresentInfo {
    /// Returns a `SwapchainPresentInfo` with the specified `swapchain` and `image_index`.
    #[inline]
    pub fn swapchain_image_index(swapchain: Arc<Swapchain>, image_index: u32) -> Self {
        Self {
            swapchain,
            image_index,
            present_id: None,
            present_regions: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Represents a rectangular region on an image layer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RectangleLayer {
    /// Coordinates in pixels of the top-left hand corner of the rectangle.
    pub offset: [u32; 2],

    /// Dimensions in pixels of the rectangle.
    pub extent: [u32; 2],

    /// The layer of the image. For images with only one layer, the value of layer must be 0.
    pub layer: u32,
}

impl RectangleLayer {
    /// Returns true if this rectangle layer is compatible with swapchain.
    #[inline]
    pub fn is_compatible_with(&self, swapchain: &Swapchain) -> bool {
        self.offset[0] + self.extent[0] <= swapchain.image_extent()[0]
            && self.offset[1] + self.extent[1] <= swapchain.image_extent()[1]
            && self.layer < swapchain.image_array_layers()
    }
}

impl From<&RectangleLayer> for ash::vk::RectLayerKHR {
    #[inline]
    fn from(val: &RectangleLayer) -> Self {
        ash::vk::RectLayerKHR {
            offset: ash::vk::Offset2D {
                x: val.offset[0] as i32,
                y: val.offset[1] as i32,
            },
            extent: ash::vk::Extent2D {
                width: val.extent[0],
                height: val.extent[1],
            },
            layer: val.layer,
        }
    }
}

/// Internal trait so that creating/destroying a swapchain can access the surface's "has_swapchain"
/// flag.
// TODO: use pub(crate) maybe?
unsafe trait SurfaceSwapchainLock {
    fn flag(&self) -> &AtomicBool;
}
