//! Link between Vulkan and a window and/or the screen.
//!
//! Before you can draw on the screen or a window, you have to create two objects:
//!
//! - Create a `Surface` object that represents the location where the image will show up (either a
//!   window or a monitor).
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
//!         khr_win32_surface: true, // If you don't enable this, `from_hwnd` will fail.
//!         ..InstanceExtensions::empty()
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
//!     let hinstance: ash::vk::HINSTANCE = 0; // Windows-specific object
//!     Surface::from_win32(
//!         instance.clone(),
//!         hinstance,
//!         window.hwnd() as ash::vk::HWND,
//!         Some(window),
//!     )
//!     .unwrap()
//! };
//! ```
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
//!     ..DeviceExtensions::empty()
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
//! Then, call [`Swapchain::new`].
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
//!         image_format,
//!         // The size of each image.
//!         image_extent,
//!         // The created swapchain images will be used as a color attachment for rendering.
//!         image_usage: ImageUsage::COLOR_ATTACHMENT,
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
//!  - Draw on that image just like you would draw to any other image (see the documentation of the
//!    `pipeline` module). You need to chain the draw after the future that was returned by
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
//!     # let mut command_buffer: ::std::sync::Arc<::vulkano::command_buffer::CommandBuffer> = return;
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
//! use vulkano::{
//!     swapchain::{self, SwapchainCreateInfo, SwapchainPresentInfo},
//!     sync::GpuFuture,
//!     Validated, VulkanError,
//! };
//!
//! // let (swapchain, images) = Swapchain::new(...);
//! # let mut swapchain: ::std::sync::Arc<::vulkano::swapchain::Swapchain> = return;
//! # let mut images: Vec<::std::sync::Arc<::vulkano::image::Image>> = return;
//! # let queue: ::std::sync::Arc<::vulkano::device::Queue> = return;
//! let mut recreate_swapchain = false;
//!
//! loop {
//!     if recreate_swapchain {
//!         let (new_swapchain, new_images) = swapchain
//!             .recreate(SwapchainCreateInfo {
//!                 image_extent: [1024, 768],
//!                 ..swapchain.create_info()
//!             })
//!             .unwrap();
//!         swapchain = new_swapchain;
//!         images = new_images;
//!         recreate_swapchain = false;
//!     }
//!
//!     let (image_index, suboptimal, acq_future) =
//!         match swapchain::acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap)
//!         {
//!             Ok(r) => r,
//!             Err(VulkanError::OutOfDate) => {
//!                 recreate_swapchain = true;
//!                 continue;
//!             }
//!             Err(err) => panic!("{:?}", err),
//!         };
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

pub use self::{acquire_present::*, surface::*};
#[cfg(target_os = "ios")]
pub use surface::IOSMetalLayer;

mod acquire_present;
mod surface;

use crate::{
    device::{Device, DeviceOwned},
    format::Format,
    image::{Image, ImageCreateFlags, ImageFormatInfo, ImageTiling, ImageType, ImageUsage},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_bitflags_enum, vulkan_enum},
    sync::Sharing,
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    fmt::Debug,
    mem::MaybeUninit,
    num::NonZeroU64,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

/// Contains the swapping system and the images that can be shown on a surface.
#[derive(Debug)]
pub struct Swapchain {
    handle: ash::vk::SwapchainKHR,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    surface: InstanceOwnedDebugWrapper<Arc<Surface>>,
    id: NonZeroU64,

    flags: SwapchainCreateFlags,
    min_image_count: u32,
    image_format: Format,
    image_view_formats: Vec<Format>,
    image_color_space: ColorSpace,
    image_extent: [u32; 2],
    image_array_layers: u32,
    image_usage: ImageUsage,
    image_sharing: Sharing<SmallVec<[u32; 4]>>,
    pre_transform: SurfaceTransform,
    composite_alpha: CompositeAlpha,
    present_mode: PresentMode,
    present_modes: SmallVec<[PresentMode; PresentMode::COUNT]>,
    clipped: bool,
    scaling_behavior: Option<PresentScaling>,
    present_gravity: Option<[PresentGravity; 2]>,
    full_screen_exclusive: FullScreenExclusive,
    win32_monitor: Option<Win32Monitor>,

    prev_present_id: AtomicU64,

    // Whether full-screen exclusive is currently held.
    full_screen_exclusive_held: AtomicBool,

    // The images of this swapchain.
    images: Vec<ImageEntry>,

    // If true, that means we have tried to use this swapchain to recreate a new swapchain. The
    // current swapchain can no longer be used for anything except presenting already-acquired
    // images.
    //
    // We use a `Mutex` instead of an `AtomicBool` because we want to keep that locked while
    // we acquire the image.
    is_retired: Mutex<bool>,
}

#[derive(Debug)]
struct ImageEntry {
    handle: ash::vk::Image,
    layout_initialized: AtomicBool,
}

impl Swapchain {
    /// Creates a new `Swapchain`.
    ///
    /// This function returns the swapchain plus a list of the images that belong to the
    /// swapchain. The order in which the images are returned is important for the
    /// `acquire_next_image` and `present` functions.
    ///
    /// # Panics
    ///
    /// - Panics if the device and the surface don't belong to the same instance.
    /// - Panics if `create_info.usage` is empty.
    // TODO: isn't it unsafe to take the surface through an Arc when it comes to vulkano-win?
    #[inline]
    pub fn new(
        device: Arc<Device>,
        surface: Arc<Surface>,
        create_info: SwapchainCreateInfo,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), Validated<VulkanError>> {
        Self::validate_new_inner(&device, &surface, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, surface, create_info)?) }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        surface: Arc<Surface>,
        create_info: SwapchainCreateInfo,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), VulkanError> {
        let (handle, image_handles) =
            Self::new_inner_unchecked(&device, &surface, &create_info, None)?;

        Self::from_handle(device, handle, image_handles, surface, create_info)
    }

    /// Creates a new swapchain from this one.
    ///
    /// Use this when a swapchain has become invalidated, such as due to window resizes.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.usage` is empty.
    pub fn recreate(
        self: &Arc<Self>,
        create_info: SwapchainCreateInfo,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), Validated<VulkanError>> {
        Self::validate_new_inner(&self.device, &self.surface, &create_info)?;

        {
            let mut is_retired = self.is_retired.lock();

            // The swapchain has already been used to create a new one.
            if *is_retired {
                return Err(Box::new(ValidationError {
                    context: "self".into(),
                    problem: "has already been used to recreate a swapchain".into(),
                    vuids: &["VUID-VkSwapchainCreateInfoKHR-oldSwapchain-01933"],
                    ..Default::default()
                })
                .into());
            } else {
                // According to the documentation of VkSwapchainCreateInfoKHR:
                //
                // > Upon calling vkCreateSwapchainKHR with a oldSwapchain that is not VK_NULL_HANDLE,
                // > any images not acquired by the application may be freed by the implementation,
                // > which may occur even if creation of the new swapchain fails.
                //
                // Therefore, we set retired to true and keep it to true even if the call to `vkCreateSwapchainKHR` below fails.
                *is_retired = true;
            }
        }

        unsafe { Ok(self.recreate_unchecked(create_info)?) }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn recreate_unchecked(
        self: &Arc<Self>,
        create_info: SwapchainCreateInfo,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), VulkanError> {
        // According to the documentation of VkSwapchainCreateInfoKHR:
        //
        // > Upon calling vkCreateSwapchainKHR with a oldSwapchain that is not VK_NULL_HANDLE,
        // > any images not acquired by the application may be freed by the implementation,
        // > which may occur even if creation of the new swapchain fails.
        //
        // Therefore, we set retired to true and keep it to true,
        // even if the call to `vkCreateSwapchainKHR` below fails.
        *self.is_retired.lock() = true;

        let (handle, image_handles) = unsafe {
            Self::new_inner_unchecked(&self.device, &self.surface, &create_info, Some(self))?
        };

        let (mut swapchain, swapchain_images) = Self::from_handle(
            self.device.clone(),
            handle,
            image_handles,
            self.surface.clone(),
            create_info,
        )?;

        if self.full_screen_exclusive == FullScreenExclusive::ApplicationControlled {
            Arc::get_mut(&mut swapchain)
                .unwrap()
                .full_screen_exclusive_held =
                AtomicBool::new(self.full_screen_exclusive_held.load(Ordering::Relaxed))
        };

        Ok((swapchain, swapchain_images))
    }

    fn validate_new_inner(
        device: &Device,
        surface: &Surface,
        create_info: &SwapchainCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !device.enabled_extensions().khr_swapchain {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_swapchain",
                )])]),
                ..Default::default()
            }));
        }

        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        let &SwapchainCreateInfo {
            flags: _,
            min_image_count,
            image_format,
            image_view_formats: _,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing: _,
            pre_transform,
            composite_alpha,
            present_mode,
            ref present_modes,
            clipped: _,
            scaling_behavior,
            present_gravity,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = create_info;

        assert_eq!(device.instance(), surface.instance());

        // VUID-VkSwapchainCreateInfoKHR-surface-01270
        if !device
            .active_queue_family_indices()
            .iter()
            .any(|&index| unsafe {
                device
                    .physical_device()
                    .surface_support_unchecked(index, surface)
                    .unwrap_or_default()
            })
        {
            return Err(Box::new(ValidationError {
                context: "surface".into(),
                problem: "is not supported by the physical device".into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-surface-01270"],
                ..Default::default()
            }));
        }

        let surface_capabilities = unsafe {
            device
                .physical_device()
                .surface_capabilities_unchecked(
                    surface,
                    SurfaceInfo {
                        present_mode: device
                            .enabled_extensions()
                            .ext_swapchain_maintenance1
                            .then_some(present_mode),
                        full_screen_exclusive,
                        win32_monitor: win32_monitor
                            .filter(|_| full_screen_exclusive != FullScreenExclusive::Default),
                        ..Default::default()
                    },
                )
                .map_err(|_err| {
                    Box::new(ValidationError {
                        problem: "`PhysicalDevice::surface_capabilities` \
                            returned an error"
                            .into(),
                        ..Default::default()
                    })
                })?
        };
        let surface_formats = unsafe {
            device
                .physical_device()
                .surface_formats_unchecked(
                    surface,
                    SurfaceInfo {
                        present_mode: device
                            .enabled_extensions()
                            .ext_swapchain_maintenance1
                            .then_some(present_mode),
                        full_screen_exclusive,
                        win32_monitor: win32_monitor.filter(|_| {
                            surface.api() == SurfaceApi::Win32
                                && full_screen_exclusive
                                    == FullScreenExclusive::ApplicationControlled
                        }),
                        ..Default::default()
                    },
                )
                .map_err(|_err| {
                    Box::new(ValidationError {
                        problem: "`PhysicalDevice::surface_formats` \
                            returned an error"
                            .into(),
                        ..Default::default()
                    })
                })?
        };
        let surface_present_modes = unsafe {
            device
                .physical_device()
                .surface_present_modes_unchecked(
                    surface,
                    SurfaceInfo {
                        full_screen_exclusive,
                        win32_monitor: win32_monitor.filter(|_| {
                            surface.api() == SurfaceApi::Win32
                                && full_screen_exclusive
                                    == FullScreenExclusive::ApplicationControlled
                        }),
                        ..Default::default()
                    },
                )
                .map_err(|_err| {
                    Box::new(ValidationError {
                        problem: "`PhysicalDevice::surface_present_modes` \
                            returned an error"
                            .into(),
                        ..Default::default()
                    })
                })?
        };

        if surface_capabilities
            .max_image_count
            .map_or(false, |c| min_image_count > c)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.min_image_count` is greater than the `max_image_count` \
                    value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-minImageCount-01272"],
                ..Default::default()
            }));
        }

        if min_image_count < surface_capabilities.min_image_count {
            return Err(Box::new(ValidationError {
                problem: "`create_info.min_image_count` is less than the `min_image_count` \
                    value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-presentMode-02839"],
                ..Default::default()
            }));
        }

        if !surface_formats
            .iter()
            .any(|&fc| fc == (image_format, image_color_space))
        {
            return Err(Box::new(ValidationError {
                problem: "the combination of `create_info.image_format` and \
                    `create_info.image_color_space` is not supported for `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageFormat-01273"],
                ..Default::default()
            }));
        }

        if image_array_layers > surface_capabilities.max_image_array_layers {
            return Err(Box::new(ValidationError {
                problem: "`create_info.image_array_layers` is greater than the \
                    `max_image_array_layers` value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageArrayLayers-01275"],
                ..Default::default()
            }));
        }

        if matches!(
            present_mode,
            PresentMode::Immediate
                | PresentMode::Mailbox
                | PresentMode::Fifo
                | PresentMode::FifoRelaxed
        ) && !surface_capabilities
            .supported_usage_flags
            .contains(image_usage)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.present_mode` is `PresentMode::Immediate`, \
                    `PresentMode::Mailbox`, `PresentMode::Fifo` or `PresentMode::FifoRelaxed`, \
                    and `create_info.image_usage` contains flags that are not set in \
                    the `supported_usage_flags` value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-presentMode-01427"],
                ..Default::default()
            }));
        }

        if !surface_capabilities
            .supported_transforms
            .contains_enum(pre_transform)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.pre_transform` is not present in the \
                    `supported_transforms` value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-preTransform-01279"],
                ..Default::default()
            }));
        }

        if !surface_capabilities
            .supported_composite_alpha
            .contains_enum(composite_alpha)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.composite_alpha` is not present in the \
                    `supported_composite_alpha` value of the capabilities of `surface`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-compositeAlpha-01280"],
                ..Default::default()
            }));
        }

        if !surface_present_modes.contains(&present_mode) {
            return Err(Box::new(ValidationError {
                problem: "`create_info.present_mode` is not supported for `surface`".into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-presentMode-01281"],
                ..Default::default()
            }));
        }

        if present_modes.is_empty() {
            if let Some(scaling_behavior) = scaling_behavior {
                if !surface_capabilities
                    .supported_present_scaling
                    .contains_enum(scaling_behavior)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`create_info.scaling_behavior` is not present in the \
                            `supported_present_scaling` value of the \
                            capabilities of `surface`"
                            .into(),
                        vuids: &[
                            "VUID-VkSwapchainPresentScalingCreateInfoEXT-scalingBehavior-07770",
                        ],
                        ..Default::default()
                    }));
                }
            }

            if let Some(present_gravity) = present_gravity {
                for (axis_index, (present_gravity, supported_present_gravity)) in present_gravity
                    .into_iter()
                    .zip(surface_capabilities.supported_present_gravity)
                    .enumerate()
                {
                    if !supported_present_gravity.contains_enum(present_gravity) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`create_info.present_gravity[{0}]` is not present in the \
                                `supported_present_gravity[{0}]` value of the \
                                capabilities of `surface`",
                                axis_index,
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-07772",
                                "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityY-07774",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }
        } else {
            for (index, &present_mode) in present_modes.iter().enumerate() {
                if !surface_present_modes.contains(&present_mode) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`create_info.present_modes[{}]` is not supported for `surface`",
                            index,
                        )
                        .into(),
                        vuids: &["VUID-VkSwapchainPresentModesCreateInfoEXT-None-07762"],
                        ..Default::default()
                    }));
                }

                if !surface_capabilities
                    .compatible_present_modes
                    .contains(&present_mode)
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`create_info.present_modes[{}]` is not present in the \
                            `compatible_present_modes` value of the \
                            capabilities of `surface`",
                            index,
                        )
                        .into(),
                        vuids: &["VUID-VkSwapchainPresentModesCreateInfoEXT-pPresentModes-07763"],
                        ..Default::default()
                    }));
                }

                if scaling_behavior.is_some() || present_gravity.is_some() {
                    let surface_capabilities = unsafe {
                        device
                            .physical_device()
                            .surface_capabilities_unchecked(
                                surface,
                                SurfaceInfo {
                                    present_mode: Some(present_mode),
                                    full_screen_exclusive,
                                    win32_monitor,
                                    ..Default::default()
                                },
                            )
                            .map_err(|_err| {
                                Box::new(ValidationError {
                                    problem: "`PhysicalDevice::surface_capabilities` \
                                        returned an error"
                                        .into(),
                                    ..Default::default()
                                })
                            })?
                    };

                    if let Some(scaling_behavior) = scaling_behavior {
                        if !surface_capabilities
                            .supported_present_scaling
                            .contains_enum(scaling_behavior)
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`create_info.scaling_behavior` is not present in the \
                                    `supported_present_scaling` value of the \
                                    capabilities of `surface` for \
                                    `create_info.present_modes[{}]`",
                                    index,
                                )
                                .into(),
                                vuids: &[
                                    "VUID-VkSwapchainPresentScalingCreateInfoEXT-scalingBehavior-07771",
                                ],
                                ..Default::default()
                            }));
                        }
                    }

                    if let Some(present_gravity) = present_gravity {
                        for (axis_index, (present_gravity, supported_present_gravity)) in
                            present_gravity
                                .into_iter()
                                .zip(surface_capabilities.supported_present_gravity)
                                .enumerate()
                        {
                            if !supported_present_gravity.contains_enum(present_gravity) {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "`create_info.present_gravity[{0}]` is not present in the \
                                        `supported_present_gravity[{0}]` value of the \
                                        capabilities of `surface` for \
                                        `create_info.present_modes[{1}]`",
                                        axis_index, index,
                                    )
                                    .into(),
                                    vuids: &[
                                        "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-07773",
                                        "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityY-07775",
                                    ],
                                    ..Default::default()
                                }));
                            }
                        }
                    }
                }
            }
        }

        if scaling_behavior.is_some() {
            if let Some(min_scaled_image_extent) = surface_capabilities.min_scaled_image_extent {
                if image_extent[0] < min_scaled_image_extent[0]
                    || image_extent[1] < min_scaled_image_extent[1]
                {
                    return Err(Box::new(ValidationError {
                        problem: "`scaling_behavior` is `Some`, and an element of \
                            `create_info.image_extent` is less than the corresponding element \
                            of the `min_scaled_image_extent` value of the \
                            capabilities of `surface`"
                            .into(),
                        vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-07782"],
                        ..Default::default()
                    }));
                }
            }

            if let Some(max_scaled_image_extent) = surface_capabilities.max_scaled_image_extent {
                if image_extent[0] > max_scaled_image_extent[0]
                    || image_extent[1] > max_scaled_image_extent[1]
                {
                    return Err(Box::new(ValidationError {
                        problem: "`scaling_behavior` is `Some`, and an element of \
                            `create_info.image_extent` is greater than the corresponding element \
                            of the `max_scaled_image_extent` value of the \
                            capabilities of `surface`"
                            .into(),
                        vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-07782"],
                        ..Default::default()
                    }));
                }
            }
        } else {
            /*
            This check is in the spec, but in practice leads to race conditions.
            The window can be resized between calling `surface_capabilities` to get the
            min/max extent, and then creating the swapchain.

            See this discussion:
            https://github.com/KhronosGroup/Vulkan-Docs/issues/1144

            if image_extent[0] < surface_capabilities.min_image_extent[0]
                || image_extent[1] < surface_capabilities.min_image_extent[1]
            {
                return Err(Box::new(ValidationError {
                    problem: "`scaling_behavior` is `Some`, and an element of \
                        `create_info.image_extent` is less than the corresponding element \
                        of the `min_image_extent` value of the \
                        capabilities of `surface`"
                        .into(),
                    vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-07781"],
                    ..Default::default()
                }));
            }

            if image_extent[0] > surface_capabilities.max_image_extent[0]
                || image_extent[1] > surface_capabilities.max_image_extent[1]
            {
                return Err(Box::new(ValidationError {
                    problem: "`scaling_behavior` is `Some`, and an element of \
                        `create_info.image_extent` is greater than the corresponding element \
                        of the `max_image_extent` value of the \
                        capabilities of `surface`"
                        .into(),
                    vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-07781"],
                    ..Default::default()
                }));
            }
            */
        }

        Ok(())
    }

    unsafe fn new_inner_unchecked(
        device: &Device,
        surface: &Surface,
        create_info: &SwapchainCreateInfo,
        old_swapchain: Option<&Swapchain>,
    ) -> Result<(ash::vk::SwapchainKHR, Vec<ash::vk::Image>), VulkanError> {
        let &SwapchainCreateInfo {
            flags,
            min_image_count,
            image_format,
            ref image_view_formats,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            ref image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            ref present_modes,
            clipped,
            scaling_behavior,
            present_gravity,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = create_info;

        let (image_sharing_mode_vk, queue_family_index_count_vk, p_queue_family_indices_vk) =
            match image_sharing {
                Sharing::Exclusive => (ash::vk::SharingMode::EXCLUSIVE, 0, ptr::null()),
                Sharing::Concurrent(ref ids) => (
                    ash::vk::SharingMode::CONCURRENT,
                    ids.len() as u32,
                    ids.as_ptr(),
                ),
            };

        let mut create_info_vk = ash::vk::SwapchainCreateInfoKHR {
            flags: flags.into(),
            surface: surface.handle(),
            min_image_count,
            image_format: image_format.into(),
            image_color_space: image_color_space.into(),
            image_extent: ash::vk::Extent2D {
                width: image_extent[0],
                height: image_extent[1],
            },
            image_array_layers,
            image_usage: image_usage.into(),
            image_sharing_mode: image_sharing_mode_vk,
            queue_family_index_count: queue_family_index_count_vk,
            p_queue_family_indices: p_queue_family_indices_vk,
            pre_transform: pre_transform.into(),
            composite_alpha: composite_alpha.into(),
            present_mode: present_mode.into(),
            clipped: clipped as ash::vk::Bool32,
            old_swapchain: old_swapchain.map_or(ash::vk::SwapchainKHR::null(), |os| os.handle),
            ..Default::default()
        };
        let mut format_list_info_vk = None;
        let format_list_view_formats_vk: Vec<_>;
        let mut full_screen_exclusive_info_vk = None;
        let mut full_screen_exclusive_win32_info_vk = None;
        let mut present_modes_info_vk = None;
        let present_modes_vk: SmallVec<[ash::vk::PresentModeKHR; PresentMode::COUNT]>;
        let mut present_scaling_info_vk = None;

        if !image_view_formats.is_empty() {
            format_list_view_formats_vk = image_view_formats
                .iter()
                .copied()
                .map(ash::vk::Format::from)
                .collect();

            let next = format_list_info_vk.insert(ash::vk::ImageFormatListCreateInfo {
                view_format_count: format_list_view_formats_vk.len() as u32,
                p_view_formats: format_list_view_formats_vk.as_ptr(),
                ..Default::default()
            });

            next.p_next = create_info_vk.p_next;
            create_info_vk.p_next = next as *const _ as *const _;
        }

        if full_screen_exclusive != FullScreenExclusive::Default {
            let next =
                full_screen_exclusive_info_vk.insert(ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                    full_screen_exclusive: full_screen_exclusive.into(),
                    ..Default::default()
                });

            next.p_next = create_info_vk.p_next as *mut _;
            create_info_vk.p_next = next as *const _ as *const _;
        }

        if let Some(Win32Monitor(hmonitor)) = win32_monitor {
            let next = full_screen_exclusive_win32_info_vk.insert(
                ash::vk::SurfaceFullScreenExclusiveWin32InfoEXT {
                    hmonitor,
                    ..Default::default()
                },
            );

            next.p_next = create_info_vk.p_next as *mut _;
            create_info_vk.p_next = next as *const _ as *const _;
        }

        if !present_modes.is_empty() {
            present_modes_vk = present_modes.iter().copied().map(Into::into).collect();

            let next = present_modes_info_vk.insert(ash::vk::SwapchainPresentModesCreateInfoEXT {
                present_mode_count: present_modes_vk.len() as u32,
                p_present_modes: present_modes_vk.as_ptr(),
                ..Default::default()
            });

            next.p_next = create_info_vk.p_next as *mut _;
            create_info_vk.p_next = next as *const _ as *const _;
        }

        if scaling_behavior.is_some() || present_gravity.is_some() {
            let [present_gravity_x, present_gravity_y] =
                present_gravity.map_or_else(Default::default, |pg| pg.map(Into::into));
            let next =
                present_scaling_info_vk.insert(ash::vk::SwapchainPresentScalingCreateInfoEXT {
                    scaling_behavior: scaling_behavior.map_or_else(Default::default, Into::into),
                    present_gravity_x,
                    present_gravity_y,
                    ..Default::default()
                });

            next.p_next = create_info_vk.p_next as *mut _;
            create_info_vk.p_next = next as *const _ as *const _;
        }

        let fns = device.fns();

        let handle = {
            let mut output = MaybeUninit::uninit();
            (fns.khr_swapchain.create_swapchain_khr)(
                device.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        let image_handles = loop {
            let mut count = 0;
            (fns.khr_swapchain.get_swapchain_images_khr)(
                device.handle(),
                handle,
                &mut count,
                ptr::null_mut(),
            )
            .result()
            .map_err(VulkanError::from)?;

            let mut images = Vec::with_capacity(count as usize);
            let result = (fns.khr_swapchain.get_swapchain_images_khr)(
                device.handle(),
                handle,
                &mut count,
                images.as_mut_ptr(),
            );

            match result {
                ash::vk::Result::SUCCESS => {
                    images.set_len(count as usize);
                    break images;
                }
                ash::vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        };

        Ok((handle, image_handles))
    }

    /// Creates a new `Swapchain` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` and `image_handles` must be valid Vulkan object handles created from `device`.
    /// - `handle` must not be retired.
    /// - `image_handles` must be swapchain images owned by `handle`, in the same order as they
    ///   were returned by `vkGetSwapchainImagesKHR`.
    /// - `surface` and `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::SwapchainKHR,
        image_handles: impl IntoIterator<Item = ash::vk::Image>,
        surface: Arc<Surface>,
        create_info: SwapchainCreateInfo,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), VulkanError> {
        let SwapchainCreateInfo {
            flags,
            min_image_count,
            image_format,
            image_view_formats,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            present_modes,
            clipped,
            scaling_behavior,
            present_gravity,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = create_info;

        let swapchain = Arc::new(Swapchain {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            surface: InstanceOwnedDebugWrapper(surface),
            id: Self::next_id(),

            flags,
            min_image_count,
            image_format,
            image_view_formats,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            present_modes,
            clipped,
            scaling_behavior,
            present_gravity,
            full_screen_exclusive,
            win32_monitor,

            prev_present_id: Default::default(),
            full_screen_exclusive_held: AtomicBool::new(false),
            images: image_handles
                .into_iter()
                .map(|handle| ImageEntry {
                    handle,
                    layout_initialized: AtomicBool::new(false),
                })
                .collect(),
            is_retired: Mutex::new(false),
        });

        let swapchain_images = swapchain
            .images
            .iter()
            .enumerate()
            .map(|(image_index, entry)| unsafe {
                Ok(Arc::new(Image::from_swapchain(
                    entry.handle,
                    swapchain.clone(),
                    image_index as u32,
                )?))
            })
            .collect::<Result<_, VulkanError>>()?;

        Ok((swapchain, swapchain_images))
    }

    /// Returns the creation parameters of the swapchain.
    #[inline]
    pub fn create_info(&self) -> SwapchainCreateInfo {
        SwapchainCreateInfo {
            flags: self.flags,
            min_image_count: self.min_image_count,
            image_format: self.image_format,
            image_view_formats: self.image_view_formats.clone(),
            image_color_space: self.image_color_space,
            image_extent: self.image_extent,
            image_array_layers: self.image_array_layers,
            image_usage: self.image_usage,
            image_sharing: self.image_sharing.clone(),
            pre_transform: self.pre_transform,
            composite_alpha: self.composite_alpha,
            present_mode: self.present_mode,
            present_modes: self.present_modes.clone(),
            clipped: self.clipped,
            scaling_behavior: self.scaling_behavior,
            present_gravity: self.present_gravity,
            full_screen_exclusive: self.full_screen_exclusive,
            win32_monitor: self.win32_monitor,
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Returns the surface that the swapchain was created from.
    #[inline]
    pub fn surface(&self) -> &Arc<Surface> {
        &self.surface
    }

    /// Returns the flags that the swapchain was created with.
    #[inline]
    pub fn flags(&self) -> SwapchainCreateFlags {
        self.flags
    }

    /// If `image` is one of the images of this swapchain, returns its index within the swapchain.
    #[inline]
    pub fn index_of_image(&self, image: &Image) -> Option<u32> {
        self.images
            .iter()
            .position(|entry| entry.handle == image.handle())
            .map(|i| i as u32)
    }

    /// Returns the number of images of the swapchain.
    #[inline]
    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    /// Returns the format of the images of the swapchain.
    #[inline]
    pub fn image_format(&self) -> Format {
        self.image_format
    }

    /// Returns the formats that an image view created from a swapchain image can have.
    #[inline]
    pub fn image_view_formats(&self) -> &[Format] {
        &self.image_view_formats
    }

    /// Returns the color space of the images of the swapchain.
    #[inline]
    pub fn image_color_space(&self) -> ColorSpace {
        self.image_color_space
    }

    /// Returns the extent of the images of the swapchain.
    #[inline]
    pub fn image_extent(&self) -> [u32; 2] {
        self.image_extent
    }

    /// Returns the number of array layers of the images of the swapchain.
    #[inline]
    pub fn image_array_layers(&self) -> u32 {
        self.image_array_layers
    }

    /// Returns the usage of the images of the swapchain.
    #[inline]
    pub fn image_usage(&self) -> ImageUsage {
        self.image_usage
    }

    /// Returns the sharing of the images of the swapchain.
    #[inline]
    pub fn image_sharing(&self) -> &Sharing<SmallVec<[u32; 4]>> {
        &self.image_sharing
    }

    /// Returns the pre-transform that was passed when creating the swapchain.
    #[inline]
    pub fn pre_transform(&self) -> SurfaceTransform {
        self.pre_transform
    }

    /// Returns the alpha mode that was passed when creating the swapchain.
    #[inline]
    pub fn composite_alpha(&self) -> CompositeAlpha {
        self.composite_alpha
    }

    /// Returns the present mode that was passed when creating the swapchain.
    #[inline]
    pub fn present_mode(&self) -> PresentMode {
        self.present_mode
    }

    /// Returns the alternative present modes that were passed when creating the swapchain.
    #[inline]
    pub fn present_modes(&self) -> &[PresentMode] {
        &self.present_modes
    }

    /// Returns the value of `clipped` that was passed when creating the swapchain.
    #[inline]
    pub fn clipped(&self) -> bool {
        self.clipped
    }

    /// Returns the scaling behavior that was passed when creating the swapchain.
    #[inline]
    pub fn scaling_behavior(&self) -> Option<PresentScaling> {
        self.scaling_behavior
    }

    /// Returns the scaling behavior that was passed when creating the swapchain.
    #[inline]
    pub fn present_gravity(&self) -> Option<[PresentGravity; 2]> {
        self.present_gravity
    }

    /// Returns the value of 'full_screen_exclusive` that was passed when creating the swapchain.
    #[inline]
    pub fn full_screen_exclusive(&self) -> FullScreenExclusive {
        self.full_screen_exclusive
    }

    /// Acquires temporary ownership of a swapchain image.
    ///
    /// The function returns the index of the image in the array of images that was returned
    /// when creating the swapchain. The image will not be available immediately after the function
    /// returns successfully.
    ///
    /// When the image becomes available, a semaphore or fence will be signaled. The image can then
    /// be accessed by the host or device. After this, the image must be *presented* back to the
    /// swapchain, using the [`present`] queue command.
    ///
    /// # Safety
    ///
    /// - `self` must be kept alive until either `acquire_info.semaphore` or `acquire_info.fence`
    ///   is signaled.
    /// - If all images from `self` are currently acquired, and have not been presented yet, then
    ///   `acquire_info.timeout` must not be `None`.
    ///
    /// If `acquire_info.semaphore` is `Some`:
    /// - The semaphore must be kept alive until it is signaled.
    /// - When the signal operation is executed, the semaphore must be in the unsignaled state.
    ///
    /// If `acquire_info.fence` is `Some`:
    /// - The fence must be kept alive until it is signaled.
    /// - The fence must be unsignaled and must not be associated with any other command that is
    ///   still executing.
    ///
    /// [`present`]: crate::device::QueueGuard::present
    #[inline]
    pub unsafe fn acquire_next_image(
        &self,
        acquire_info: &AcquireNextImageInfo,
    ) -> Result<AcquiredImage, Validated<VulkanError>> {
        let is_retired_lock = self.is_retired.lock();
        self.validate_acquire_next_image(acquire_info, *is_retired_lock)?;

        Ok(self.acquire_next_image_unchecked(acquire_info)?)
    }

    fn validate_acquire_next_image(
        &self,
        acquire_info: &AcquireNextImageInfo,
        is_retired: bool,
    ) -> Result<(), Box<ValidationError>> {
        acquire_info
            .validate(&self.device)
            .map_err(|err| err.add_context("acquire_info"))?;

        if is_retired {
            return Err(Box::new(ValidationError {
                problem: "this swapchain is retired".into(),
                vuids: &["VUID-VkAcquireNextImageInfoKHR-swapchain-01675"],
                ..Default::default()
            }));
        }

        // unsafe
        // VUID-vkAcquireNextImage2KHR-surface-07784
        // VUID-VkAcquireNextImageInfoKHR-semaphore-01288
        // VUID-VkAcquireNextImageInfoKHR-semaphore-01781
        // VUID-VkAcquireNextImageInfoKHR-fence-01289

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn acquire_next_image_unchecked(
        &self,
        acquire_info: &AcquireNextImageInfo,
    ) -> Result<AcquiredImage, VulkanError> {
        let &AcquireNextImageInfo {
            timeout,
            ref semaphore,
            ref fence,
            _ne: _,
        } = acquire_info;

        let acquire_info_vk = ash::vk::AcquireNextImageInfoKHR {
            swapchain: self.handle,
            timeout: timeout.map_or(u64::MAX, |duration| {
                u64::try_from(duration.as_nanos()).unwrap()
            }),
            semaphore: semaphore
                .as_ref()
                .map(VulkanObject::handle)
                .unwrap_or_default(),
            fence: fence.as_ref().map(VulkanObject::handle).unwrap_or_default(),
            device_mask: self.device.device_mask(),
            ..Default::default()
        };

        let fns = self.device.fns();
        let mut output = MaybeUninit::uninit();

        let result = if self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_device_group
        {
            (fns.khr_swapchain.acquire_next_image2_khr)(
                self.device.handle(),
                &acquire_info_vk,
                output.as_mut_ptr(),
            )
        } else {
            debug_assert!(acquire_info_vk.p_next.is_null());
            (fns.khr_swapchain.acquire_next_image_khr)(
                self.device.handle(),
                acquire_info_vk.swapchain,
                acquire_info_vk.timeout,
                acquire_info_vk.semaphore,
                acquire_info_vk.fence,
                output.as_mut_ptr(),
            )
        };

        let is_suboptimal = match result {
            ash::vk::Result::SUCCESS => false,
            ash::vk::Result::SUBOPTIMAL_KHR => true,
            ash::vk::Result::NOT_READY => return Err(VulkanError::NotReady),
            ash::vk::Result::TIMEOUT => return Err(VulkanError::Timeout),
            err => {
                let err = VulkanError::from(err);

                if matches!(err, VulkanError::FullScreenExclusiveModeLost) {
                    self.full_screen_exclusive_held
                        .store(false, Ordering::SeqCst);
                }

                return Err(err);
            }
        };

        Ok(AcquiredImage {
            image_index: output.assume_init(),
            is_suboptimal,
        })
    }

    /// Waits for a swapchain image with a specific present ID to be presented to the user.
    ///
    /// For this to work, you must set [`SwapchainPresentInfo::present_id`] to `Some` when
    /// presenting. This function will then wait until the swapchain image with the specified ID is
    /// presented.
    ///
    /// Returns whether the presentation was suboptimal. This has the same meaning as in
    /// [`AcquiredImage::is_suboptimal`].
    #[inline]
    pub fn wait_for_present(
        &self,
        present_id: NonZeroU64,
        timeout: Option<Duration>,
    ) -> Result<bool, Validated<VulkanError>> {
        let is_retired_lock = self.is_retired.lock();
        self.validate_wait_for_present(present_id, timeout, *is_retired_lock)?;

        unsafe { Ok(self.wait_for_present_unchecked(present_id, timeout)?) }
    }

    fn validate_wait_for_present(
        &self,
        _present_id: NonZeroU64,
        timeout: Option<Duration>,
        is_retired: bool,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_features().present_wait {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "present_wait",
                )])]),
                vuids: &["VUID-vkWaitForPresentKHR-presentWait-06234"],
                ..Default::default()
            }));
        }

        if is_retired {
            return Err(Box::new(ValidationError {
                problem: "this swapchain is retired".into(),
                vuids: &["VUID-vkWaitForPresentKHR-swapchain-04997"],
                ..Default::default()
            }));
        }

        if let Some(timeout) = timeout {
            if timeout.as_nanos() >= u64::MAX as u128 {
                return Err(Box::new(ValidationError {
                    context: "timeout".into(),
                    problem: "is not less than `u64::MAX` nanoseconds".into(),
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wait_for_present_unchecked(
        &self,
        present_id: NonZeroU64,
        timeout: Option<Duration>,
    ) -> Result<bool, VulkanError> {
        let result = {
            let fns = self.device.fns();
            (fns.khr_present_wait.wait_for_present_khr)(
                self.device.handle(),
                self.handle,
                present_id.get(),
                timeout.map_or(u64::MAX, |duration| {
                    u64::try_from(duration.as_nanos()).unwrap()
                }),
            )
        };

        match result {
            ash::vk::Result::SUCCESS => Ok(false),
            ash::vk::Result::SUBOPTIMAL_KHR => Ok(true),
            ash::vk::Result::TIMEOUT => Err(VulkanError::Timeout),
            err => {
                let err = VulkanError::from(err);

                if matches!(err, VulkanError::FullScreenExclusiveModeLost) {
                    self.full_screen_exclusive_held
                        .store(false, Ordering::SeqCst);
                }

                Err(err)
            }
        }
    }

    /// Acquires full-screen exclusivity.
    ///
    /// The swapchain must have been created with [`FullScreenExclusive::ApplicationControlled`],
    /// and must not already hold full-screen exclusivity. Full-screen exclusivity is held until
    /// either the `release_full_screen_exclusive` is called, or if any of the the other
    /// `Swapchain` functions return `FullScreenExclusiveLost`.
    #[inline]
    pub fn acquire_full_screen_exclusive_mode(&self) -> Result<(), Validated<VulkanError>> {
        self.validate_acquire_full_screen_exclusive_mode()?;

        unsafe { Ok(self.acquire_full_screen_exclusive_mode_unchecked()?) }
    }

    fn validate_acquire_full_screen_exclusive_mode(&self) -> Result<(), Box<ValidationError>> {
        if *self.is_retired.lock() {
            return Err(Box::new(ValidationError {
                problem: "this swapchain is retired".into(),
                vuids: &["VUID-vkAcquireFullScreenExclusiveModeEXT-swapchain-02674"],
                ..Default::default()
            }));
        }

        if self.full_screen_exclusive != FullScreenExclusive::ApplicationControlled {
            return Err(Box::new(ValidationError {
                context: "self.full_screen_exclusive()".into(),
                problem: "is not `FullScreenExclusive::ApplicationControlled`".into(),
                vuids: &["VUID-vkAcquireFullScreenExclusiveModeEXT-swapchain-02675"],
                ..Default::default()
            }));
        }

        // This must be the last check in this function.
        if self
            .full_screen_exclusive_held
            .swap(true, Ordering::Acquire)
        {
            return Err(Box::new(ValidationError {
                problem: "this swapchain already holds full-screen exclusive access".into(),
                vuids: &["VUID-vkAcquireFullScreenExclusiveModeEXT-swapchain-02676"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn acquire_full_screen_exclusive_mode_unchecked(&self) -> Result<(), VulkanError> {
        self.full_screen_exclusive_held
            .store(true, Ordering::Relaxed);

        let fns = self.device.fns();
        (fns.ext_full_screen_exclusive
            .acquire_full_screen_exclusive_mode_ext)(self.device.handle(), self.handle)
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Releases full-screen exclusivity.
    ///
    /// The swapchain must have been created with [`FullScreenExclusive::ApplicationControlled`],
    /// and must currently hold full-screen exclusivity.
    #[inline]
    pub fn release_full_screen_exclusive_mode(&self) -> Result<(), Validated<VulkanError>> {
        self.validate_release_full_screen_exclusive_mode()?;

        unsafe { Ok(self.release_full_screen_exclusive_mode_unchecked()?) }
    }

    fn validate_release_full_screen_exclusive_mode(&self) -> Result<(), Box<ValidationError>> {
        if *self.is_retired.lock() {
            return Err(Box::new(ValidationError {
                problem: "this swapchain is retired".into(),
                vuids: &["VUID-vkReleaseFullScreenExclusiveModeEXT-swapchain-02677"],
                ..Default::default()
            }));
        }

        if self.full_screen_exclusive != FullScreenExclusive::ApplicationControlled {
            return Err(Box::new(ValidationError {
                context: "self.full_screen_exclusive()".into(),
                problem: "is not `FullScreenExclusive::ApplicationControlled`".into(),
                vuids: &["VUID-vkReleaseFullScreenExclusiveModeEXT-swapchain-02678"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn release_full_screen_exclusive_mode_unchecked(&self) -> Result<(), VulkanError> {
        self.full_screen_exclusive_held
            .store(false, Ordering::Release);

        let fns = self.device.fns();
        (fns.ext_full_screen_exclusive
            .release_full_screen_exclusive_mode_ext)(self.device.handle(), self.handle)
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }

    /// `FullScreenExclusive::AppControlled` is not the active full-screen exclusivity mode,
    /// then this function will always return false. If true is returned the swapchain
    /// is in `FullScreenExclusive::AppControlled` full-screen exclusivity mode and exclusivity
    /// is currently acquired.
    #[inline]
    pub fn is_full_screen_exclusive(&self) -> bool {
        if self.full_screen_exclusive != FullScreenExclusive::ApplicationControlled {
            false
        } else {
            self.full_screen_exclusive_held.load(Ordering::SeqCst)
        }
    }

    // This method is necessary to allow `SwapchainImage`s to signal when they have been
    // transitioned out of their initial `undefined` image layout.
    //
    // See the `ImageAccess::layout_initialized` method documentation for more details.
    pub(crate) fn image_layout_initialized(&self, image_index: u32) {
        let image_entry = self.images.get(image_index as usize);
        if let Some(image_entry) = image_entry {
            image_entry
                .layout_initialized
                .store(true, Ordering::Relaxed);
        }
    }

    pub(crate) fn is_image_layout_initialized(&self, image_index: u32) -> bool {
        let image_entry = self.images.get(image_index as usize);
        if let Some(image_entry) = image_entry {
            image_entry.layout_initialized.load(Ordering::Relaxed)
        } else {
            false
        }
    }

    #[inline]
    pub(crate) unsafe fn full_screen_exclusive_held(&self) -> &AtomicBool {
        &self.full_screen_exclusive_held
    }

    #[inline]
    pub(crate) unsafe fn try_claim_present_id(&self, present_id: NonZeroU64) -> bool {
        let present_id = u64::from(present_id);
        self.prev_present_id.fetch_max(present_id, Ordering::SeqCst) < present_id
    }
}

impl Drop for Swapchain {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.khr_swapchain.destroy_swapchain_khr)(
                self.device.handle(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

unsafe impl VulkanObject for Swapchain {
    type Handle = ash::vk::SwapchainKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Swapchain {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(Swapchain);

/// Parameters to create a new `Swapchain`.
///
/// Many of the values here must be supported by the physical device.
/// [`PhysicalDevice`](crate::device::physical::PhysicalDevice) has several
/// methods to query what is supported.
#[derive(Clone, Debug)]
pub struct SwapchainCreateInfo {
    /// Additional properties of the swapchain.
    ///
    /// The default value is empty.
    pub flags: SwapchainCreateFlags,

    /// The minimum number of images that will be created.
    ///
    /// The implementation is allowed to create more than this number, but never less.
    ///
    /// The default value is `2`.
    pub min_image_count: u32,

    /// The format of the created images.
    ///
    /// The default value is `Format::UNDEFINED`.
    pub image_format: Format,

    /// The formats that an image view can have when it is created from one of the swapchain
    /// images.
    ///
    /// If the list is not empty, and `flags` does not contain
    /// [`SwapchainCreateFlags::MUTABLE_FORMAT`], then the list must contain at most one element,
    /// otherwise any number of elements are allowed.
    /// The view formats must be compatible with `format`.
    ///
    /// If `flags` contains [`SwapchainCreateFlags::MUTABLE_FORMAT`], then the list must contain
    /// `image_format.
    ///
    /// If this is not empty, then the device API version must be at least 1.2, or the
    /// [`khr_image_format_list`] extension must be enabled on the device.
    ///
    /// The default value is empty.
    ///
    /// [`khr_image_format_list`]: crate::device::DeviceExtensions::khr_image_format_list
    pub image_view_formats: Vec<Format>,

    /// The color space of the created images.
    ///
    /// The default value is [`ColorSpace::SrgbNonLinear`].
    pub image_color_space: ColorSpace,

    /// The extent of the created images.
    ///
    /// Both values must be greater than zero. Note that on some platforms,
    /// [`SurfaceCapabilities::current_extent`] will be zero if the surface is minimized.
    /// Care must be taken to check for this, to avoid trying to create a zero-size swapchain.
    ///
    /// The default value is `[0, 0]`, which must be overridden.
    pub image_extent: [u32; 2],

    /// The number of array layers of the created images.
    ///
    /// The default value is `1`.
    pub image_array_layers: u32,

    /// How the created images will be used.
    ///
    /// The default value is [`ImageUsage::empty()`], which must be overridden.
    pub image_usage: ImageUsage,

    /// Whether the created images can be shared across multiple queues, or are limited to a single
    /// queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub image_sharing: Sharing<SmallVec<[u32; 4]>>,

    /// The transform that should be applied to an image before it is presented.
    ///
    /// The default value is [`SurfaceTransform::Identity`].
    pub pre_transform: SurfaceTransform,

    /// How alpha values of the pixels in the image are to be treated.
    ///
    /// The default value is [`CompositeAlpha::Opaque`].
    pub composite_alpha: CompositeAlpha,

    /// How the swapchain should behave when multiple images are waiting in the queue to be
    /// presented.
    ///
    /// The default value is [`PresentMode::Fifo`].
    pub present_mode: PresentMode,

    /// Alternative present modes that can be used with this swapchain. The mode specified in
    /// `present_mode` is the default mode, but can be changed for future present operations by
    /// specifying it when presenting.
    ///
    /// If this is not empty, then the
    /// [`ext_swapchain_maintenance1`](crate::device::DeviceExtensions::ext_swapchain_maintenance1)
    /// extension must be enabled on the device.
    /// It must always contain the mode specified in `present_mode`.
    ///
    /// The default value is empty.
    pub present_modes: SmallVec<[PresentMode; PresentMode::COUNT]>,

    /// Whether the implementation is allowed to discard rendering operations that affect regions
    /// of the surface which aren't visible. This is important to take into account if your
    /// fragment shader has side-effects or if you want to read back the content of the image
    /// afterwards.
    ///
    /// The default value is `true`.
    pub clipped: bool,

    /// The scaling method to use when the surface is not the same size as the swapchain image.
    ///
    /// `None` means the behavior is implementation defined.
    ///
    /// If this is `Some`, then the
    /// [`ext_swapchain_maintenance1`](crate::device::DeviceExtensions::ext_swapchain_maintenance1)
    /// extension must be enabled on the device.
    ///
    /// The default value is `None`.
    pub scaling_behavior: Option<PresentScaling>,

    /// The horizontal and vertical alignment to use when the swapchain image, after applying
    /// scaling, does not fill the whole surface.
    ///
    /// `None` means the behavior is implementation defined.
    ///
    /// If this is `Some`, then the
    /// [`ext_swapchain_maintenance1`](crate::device::DeviceExtensions::ext_swapchain_maintenance1)
    /// extension must be enabled on the device.
    ///
    /// The default value is `None`.
    pub present_gravity: Option<[PresentGravity; 2]>,

    /// How full-screen exclusivity is to be handled.
    ///
    /// If set to anything other than [`FullScreenExclusive::Default`], then the
    /// [`ext_full_screen_exclusive`](crate::device::DeviceExtensions::ext_full_screen_exclusive)
    /// extension must be enabled on the device.
    ///
    /// The default value is [`FullScreenExclusive::Default`].
    pub full_screen_exclusive: FullScreenExclusive,

    /// If `full_screen_exclusive` is not [`FullScreenExclusive::Default`], this specifies the
    /// monitor on which full-screen exclusivity should be used.
    ///
    /// For this case, the value must be `Some`, and for all others it must be `None`.
    ///
    /// The default value is `None`.
    pub win32_monitor: Option<Win32Monitor>,

    pub _ne: crate::NonExhaustive,
}

impl Default for SwapchainCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: SwapchainCreateFlags::empty(),
            min_image_count: 2,
            image_format: Format::UNDEFINED,
            image_view_formats: Vec::new(),
            image_color_space: ColorSpace::SrgbNonLinear,
            image_extent: [0, 0],
            image_array_layers: 1,
            image_usage: ImageUsage::empty(),
            image_sharing: Sharing::Exclusive,
            pre_transform: SurfaceTransform::Identity,
            composite_alpha: CompositeAlpha::Opaque,
            present_mode: PresentMode::Fifo,
            present_modes: SmallVec::new(),
            clipped: true,
            scaling_behavior: None,
            present_gravity: None,
            full_screen_exclusive: FullScreenExclusive::Default,
            win32_monitor: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SwapchainCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            min_image_count: _,
            image_format,
            ref image_view_formats,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            ref image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            ref present_modes,
            clipped: _,
            scaling_behavior,
            present_gravity,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-flags-parameter"])
        })?;

        image_format.validate_device(device).map_err(|err| {
            err.add_context("image_format")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-imageFormat-parameter"])
        })?;

        image_color_space.validate_device(device).map_err(|err| {
            err.add_context("image_color_space")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-imageColorSpace-parameter"])
        })?;

        image_usage.validate_device(device).map_err(|err| {
            err.add_context("image_usage")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-imageUsage-parameter"])
        })?;

        if image_usage.is_empty() {
            return Err(Box::new(ValidationError {
                context: "image_usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageUsage-requiredbitmask"],
                ..Default::default()
            }));
        }

        pre_transform.validate_device(device).map_err(|err| {
            err.add_context("pre_transform")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-preTransform-parameter"])
        })?;

        composite_alpha.validate_device(device).map_err(|err| {
            err.add_context("composite_alpha")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-compositeAlpha-parameter"])
        })?;

        present_mode.validate_device(device).map_err(|err| {
            err.add_context("present_mode")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-presentMode-parameter"])
        })?;

        if image_extent.contains(&0) {
            return Err(Box::new(ValidationError {
                context: "image_extent".into(),
                problem: "one or more elements are zero".into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageExtent-01689"],
                ..Default::default()
            }));
        }

        if image_array_layers == 0 {
            return Err(Box::new(ValidationError {
                context: "image_array_layers".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageArrayLayers-01275"],
                ..Default::default()
            }));
        }

        match image_sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                if queue_family_indices.len() < 2 {
                    return Err(Box::new(ValidationError {
                        context: "image_sharing".into(),
                        problem: "is `Sharing::Concurrent`, and contains less than 2 \
                            queue family indices"
                            .into(),
                        vuids: &["VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01278"],
                        ..Default::default()
                    }));
                }

                let queue_family_count =
                    device.physical_device().queue_family_properties().len() as u32;

                for (index, &queue_family_index) in queue_family_indices.iter().enumerate() {
                    if queue_family_indices[..index].contains(&queue_family_index) {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_indices".into(),
                            problem: format!(
                                "the queue family index in the list at index {} is contained in \
                                the list more than once",
                                index,
                            )
                            .into(),
                            vuids: &["VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01428"],
                            ..Default::default()
                        }));
                    }

                    if queue_family_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: format!("queue_family_indices[{}]", index).into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01428"],
                            ..Default::default()
                        }));
                    }
                }
            }
        };

        let image_format_properties = unsafe {
            device
                .physical_device()
                .image_format_properties_unchecked(ImageFormatInfo {
                    format: image_format,
                    image_type: ImageType::Dim2d,
                    tiling: ImageTiling::Optimal,
                    usage: image_usage,
                    ..Default::default()
                })
                .map_err(|_err| {
                    Box::new(ValidationError {
                        problem: "`PhysicalDevice::image_format_properties` \
                            returned an error"
                            .into(),
                        ..Default::default()
                    })
                })?
        };

        if image_format_properties.is_none() {
            return Err(Box::new(ValidationError {
                problem: "the combination of `image_format` and `image_usage` is not supported \
                    for images by the physical device"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-imageFormat-01778"],
                ..Default::default()
            }));
        }

        if flags.intersects(SwapchainCreateFlags::MUTABLE_FORMAT)
            && !image_view_formats.contains(&image_format)
        {
            return Err(Box::new(ValidationError {
                problem: "`flags` contains `SwapchainCreateFlags::MUTABLE_FORMAT`, but \
                    `image_view_formats` does not contain `image_format`"
                    .into(),
                vuids: &["VUID-VkSwapchainCreateInfoKHR-flags-03168"],
                ..Default::default()
            }));
        }

        if !image_view_formats.is_empty() {
            if !(device.api_version() >= Version::V1_2
                || device.enabled_extensions().khr_image_format_list)
            {
                return Err(Box::new(ValidationError {
                    context: "image_view_formats".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_image_format_list")]),
                    ]),
                    ..Default::default()
                }));
            }

            if !flags.intersects(SwapchainCreateFlags::MUTABLE_FORMAT)
                && image_view_formats.len() != 1
            {
                return Err(Box::new(ValidationError {
                    problem: "`flags` does not contain `SwapchainCreateFlags::MUTABLE_FORMAT`, \
                        but `image_view_formats` contains more than one element"
                        .into(),
                    vuids: &["VUID-VkSwapchainCreateInfoKHR-flags-04100"],
                    ..Default::default()
                }));
            }

            for (index, &image_view_format) in image_view_formats.iter().enumerate() {
                image_view_format.validate_device(device).map_err(|err| {
                    err.add_context(format!("image_view_formats[{}]", index))
                        .set_vuids(&["VUID-VkImageFormatListCreateInfo-pViewFormats-parameter"])
                })?;

                if image_view_format.compatibility() != image_format.compatibility() {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`image_view_formats[{}]` is not compatible with `image_format`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-04099"],
                        ..Default::default()
                    }));
                }
            }
        }

        if !present_modes.is_empty() {
            if !device.enabled_extensions().ext_swapchain_maintenance1 {
                return Err(Box::new(ValidationError {
                    context: "present_modes".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_swapchain_maintenance1",
                    )])]),
                    ..Default::default()
                }));
            }

            for (index, &present_mode) in present_modes.iter().enumerate() {
                present_mode.validate_device(device).map_err(|err| {
                    err.add_context(format!("present_modes[{}]", index))
                        .set_vuids(&[
                            "VUID-VkSwapchainPresentModesCreateInfoEXT-pPresentModes-parameter",
                        ])
                })?;
            }

            if !present_modes.contains(&present_mode) {
                return Err(Box::new(ValidationError {
                    problem: "`present_modes` is not empty, but does not contain `present_mode`"
                        .into(),
                    vuids: &["VUID-VkSwapchainPresentModesCreateInfoEXT-presentMode-07764"],
                    ..Default::default()
                }));
            }
        }

        if let Some(scaling_behavior) = scaling_behavior {
            if !device.enabled_extensions().ext_swapchain_maintenance1 {
                return Err(Box::new(ValidationError {
                    context: "scaling_behavior".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_swapchain_maintenance1",
                    )])]),
                    ..Default::default()
                }));
            }

            scaling_behavior.validate_device(device).map_err(|err| {
                err.add_context("scaling_behavior").set_vuids(&[
                    "VUID-VkSwapchainPresentScalingCreateInfoEXT-scalingBehavior-parameter",
                ])
            })?;

            // VUID-VkSwapchainPresentScalingCreateInfoEXT-scalingBehavior-07767
            // Ensured by the use of an enum.
        }

        if let Some(present_gravity) = present_gravity {
            if !device.enabled_extensions().ext_swapchain_maintenance1 {
                return Err(Box::new(ValidationError {
                    context: "present_gravity".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_swapchain_maintenance1",
                    )])]),
                    ..Default::default()
                }));
            }

            for (axis_index, present_gravity) in present_gravity.into_iter().enumerate() {
                present_gravity.validate_device(device).map_err(|err| {
                    err.add_context(format!("present_gravity[{}]", axis_index))
                        .set_vuids(&[
                            "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-parameter",
                            "VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityY-parameter",
                        ])
                })?;
            }

            // VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-07765
            // VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-07766
            // Ensured by the use of an array of enums wrapped in `Option`.

            // VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityX-07768
            // VUID-VkSwapchainPresentScalingCreateInfoEXT-presentGravityY-07769
            // Ensured by the use of an enum.
        }

        if full_screen_exclusive != FullScreenExclusive::Default {
            if !device.enabled_extensions().ext_full_screen_exclusive {
                return Err(Box::new(ValidationError {
                    context: "full_screen_exclusive".into(),
                    problem: "is not `FullScreenExclusive::Default`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_full_screen_exclusive",
                    )])]),
                    ..Default::default()
                }));
            }

            full_screen_exclusive
                .validate_device(device)
                .map_err(|err| {
                    err.add_context("full_screen_exclusive").set_vuids(&[
                        "VUID-VkSurfaceFullScreenExclusiveInfoEXT-fullScreenExclusive-parameter",
                    ])
                })?;

            if win32_monitor.is_none() {
                return Err(Box::new(ValidationError {
                    problem: "`full_screen_exclusive` is not `FullScreenExclusive::Default`, but \
                        `win32_monitor` is `None`"
                        .into(),
                    vuids: &["VUID-VkSwapchainCreateInfoKHR-pNext-02679"],
                    ..Default::default()
                }));
            }
        } else if win32_monitor.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`full_screen_exclusive` is `FullScreenExclusive::Default`, but \
                    `win32_monitor` is `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a swapchain.
    SwapchainCreateFlags = SwapchainCreateFlagsKHR(u32);

    /* TODO: enable
    /// Creates swapchain images with the [`ImageCreateFlags::SPLIT_INSTANCE_BIND_REGIONS`] flag.
    SPLIT_INSTANCE_BIND_REGIONS = SPLIT_INSTANCE_BIND_REGIONS
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_device_group)]),
    ]),*/

    /* TODO: enable
    /// Creates swapchain images with the [`ImageCreateFlags::PROTECTED`] flag.
    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),*/

    /// Creates swapchain images with both the [`ImageCreateFlags::MUTABLE_FORMAT`] and
    /// [`ImageCreateFlags::EXTENDED_USAGE`] flags.
    MUTABLE_FORMAT = MUTABLE_FORMAT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_swapchain_mutable_format)]),
    ]),

    /* TODO: enable
    // TODO: document
    DEFERRED_MEMORY_ALLOCATION = DEFERRED_MEMORY_ALLOCATION_EXT {
        device_extensions: [ext_swapchain_maintenance1],
    },*/
}

impl From<SwapchainCreateFlags> for ImageCreateFlags {
    #[inline]
    fn from(flags: SwapchainCreateFlags) -> Self {
        // Per https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html#_description
        let mut result = ImageCreateFlags::empty();

        /* TODO: enable
        if flags.intersects(SwapchainCreateFlags::SPLIT_INSTANCE_BIND_REGIONS) {
            result |= ImageCreateFlags::SPLIT_INSTANCE_BIND_REGIONS;
        } */

        /* TODO: enable
        if flags.intersects(SwapchainCreateFlags::PROTECTED) {
            result |= ImageCreateFlags::PROTECTED;
        } */

        if flags.intersects(SwapchainCreateFlags::MUTABLE_FORMAT) {
            result |= ImageCreateFlags::MUTABLE_FORMAT | ImageCreateFlags::EXTENDED_USAGE;
        }

        result
    }
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`PresentScaling`] values.
    PresentScalingFlags,

    /// The way a swapchain image is scaled, if it does not exactly fit the surface.
    PresentScaling,

    = PresentScalingFlagsEXT(u32);

    /// No scaling is performed; one swapchain image pixel maps to one surface pixel.
    ONE_TO_ONE, OneToOne = ONE_TO_ONE,

    /// Both axes of the image are scaled equally, without changing the aspect ratio of the image,
    /// to the largest size in which both axes fit inside the surface.
    ASPECT_RATIO_STRETCH, AspectRatioStretch = ASPECT_RATIO_STRETCH,

    /// Each axis of the image is scaled independently to fit the surface,
    /// which may change the aspect ratio of the image.
    STRETCH, Stretch = STRETCH,
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`PresentGravity`] values.
    PresentGravityFlags,

    /// The way a swapchain image is aligned, if it does not exactly fit the surface.
    PresentGravity,

    = PresentGravityFlagsEXT(u32);

    /// Aligned to the top or left side of the surface.
    MIN, Min = MIN,

    /// Aligned to the bottom or right side of the surface.
    MAX, Max = MAX,

    /// Aligned to the middle of the surface.
    CENTERED, Centered = CENTERED,
}

vulkan_enum! {
    #[non_exhaustive]

    /// The way full-screen exclusivity is handled.
    FullScreenExclusive = FullScreenExclusiveEXT(i32);

    /// Indicates that the driver should determine the appropriate full-screen method
    /// by whatever means it deems appropriate.
    Default = DEFAULT,

    /// Indicates that the driver may use full-screen exclusive mechanisms when available.
    /// Such mechanisms may result in better performance and/or the availability of
    /// different presentation capabilities, but may require a more disruptive transition
    // during swapchain initialization, first presentation and/or destruction.
    Allowed = ALLOWED,

    /// Indicates that the driver should avoid using full-screen mechanisms which rely
    /// on disruptive transitions.
    Disallowed = DISALLOWED,

    /// Indicates the application will manage full-screen exclusive mode by using the
    /// [`Swapchain::acquire_full_screen_exclusive_mode`] and
    /// [`Swapchain::release_full_screen_exclusive_mode`] functions.
    ApplicationControlled = APPLICATION_CONTROLLED,
}

/// A wrapper around a Win32 monitor handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Win32Monitor(pub(crate) ash::vk::HMONITOR);

impl Win32Monitor {
    /// Wraps a Win32 monitor handle.
    ///
    /// # Safety
    ///
    /// - `hmonitor` must be a valid handle as returned by the Win32 API.
    pub unsafe fn new(hmonitor: ash::vk::HMONITOR) -> Self {
        Self(hmonitor)
    }
}

// Winit's `MonitorHandle` is Send on Win32, so this seems safe.
unsafe impl Send for Win32Monitor {}
unsafe impl Sync for Win32Monitor {}
