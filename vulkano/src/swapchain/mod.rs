// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
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
//! There are 6 extensions that each allow you to create a surface from a type of window:
//!
//! - `VK_KHR_xlib_surface`
//! - `VK_KHR_xcb_surface`
//! - `VK_KHR_wayland_surface`
//! - `VK_KHR_mir_surface`
//! - `VK_KHR_android_surface`
//! - `VK_KHR_win32_surface`
//!
//! For example if you want to create a surface from an Android surface, you will have to enable
//! the `VK_KHR_android_surface` extension and use `Surface::from_anativewindow`.
//! See the documentation of `Surface` for all the possible constructors.
//!
//! Trying to use one of these functions without enabling the proper extension will result in an
//! error.
//!
//! **Note that the `Surface` object is unsafe**. It is your responsibility to keep the window
//! alive for at least as long as the surface exists.
//!
//! ### Example
//!
//! ```no_run
//! use std::ptr;
//! use vulkano::instance::Instance;
//! use vulkano::instance::InstanceExtensions;
//! use vulkano::swapchain::Surface;
//!
//! let instance = {
//!     let extensions = InstanceExtensions {
//!         khr_surface: true,
//!         khr_win32_surface: true,        // If you don't enable this, `from_hwnd` will fail.
//!         .. InstanceExtensions::none()
//!     };
//!
//!      match Instance::new(None, &extensions, None) {
//!         Ok(i) => i,
//!         Err(err) => panic!("Couldn't build instance: {:?}", err)
//!     }
//! };
//!
//! # fn build_window() -> *const u32 { ptr::null() }
//! let window = build_window();
//! let _surface = unsafe {
//!     let hinstance: *const () = ptr::null();     // Windows-specific object
//!     Surface::from_hwnd(&instance, hinstance, window).unwrap()
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
//! You can query the supported values of all these properties with `Surface::get_capabilities()`.
//!
//! ## Creating a swapchain
//!
//! In order to create a swapchain, you will first have to enable the `VK_KHR_swapchain` extension
//! on the device (and not on the instance like `VK_KHR_surface`).
//!
//! Then, you should query the capabilities of the surface with `Surface::get_capabilities()` and
//! choose which values you are going to use. Then, call `Swapchain::new`.
//!
//! TODO: add example here
//!
//! Creating a swapchain not only returns the swapchain object, but also all the images that belong
//! to it.
//!
//! ## Acquiring and presenting images
//!
//! Once you created a swapchain and retreived all the images that belong to it (see previous
//! section), you can draw on it. This is done in three steps:
//!
//!  - Call `Swapchain::acquire_next_image`. This function will return the index of the image
//!    (within the list returned by `Swapchain::new`) that is available to draw.
//!  - Draw on that image just like you would draw to any other image (see the documentation of
//!    the `pipeline` module).
//!  - Call `Swapchain::present` with the same index in order to tell the implementation that you
//!    are finished drawing to the image and that it can queue a command to present the image on
//!    the screen after the draw operations are finished.
//!
//! TODO: add example here
//! loop {
//!     let index = swapchain.acquire_next_image(Duration::from_millis(500)).unwrap();
//!     draw(images[index]);
//!     swapchain.present(queue, index).unwrap();
//! }
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
//! TODO: suboptimal stuff
//!
//! ```no_run
//! # use std::time::Duration;
//! use vulkano::swapchain::AcquireError;
//! use vulkano::swapchain::PresentError;
//!
//! // let mut swapchain = Swapchain::new(...);
//! # let mut swapchain: (::std::sync::Arc<::vulkano::swapchain::Swapchain>, _) = unsafe { ::std::mem::uninitialized() };
//! # let queue: ::std::sync::Arc<::vulkano::device::Queue> = unsafe { ::std::mem::uninitialized() };
//! let mut recreate_swapchain = false;
//!
//! loop {
//!     if recreate_swapchain {
//!         swapchain = swapchain.0.recreate_with_dimension([1024, 768]).unwrap();
//!         recreate_swapchain = false;
//!     }
//!
//!     let (ref swapchain, ref _images) = swapchain;
//!
//!     let index = match swapchain.acquire_next_image(Duration::from_millis(500)) {
//!         Ok(img) => img,
//!         Err(AcquireError::OutOfDate) => { recreate_swapchain = true; continue; },
//!         Err(err) => panic!("{:?}", err)
//!     };
//!
//!     // ...
//!
//!     match swapchain.present(&queue, index) {
//!         Ok(()) => (),
//!         Err(PresentError::OutOfDate) => { recreate_swapchain = true; },
//!         Err(err) => panic!("{:?}", err),
//!     }
//! }
//! ```
//!

use std::sync::atomic::AtomicBool;

pub use self::surface::Capabilities;
pub use self::surface::Surface;
pub use self::surface::PresentMode;
pub use self::surface::SurfaceTransform;
pub use self::surface::CompositeAlpha;
pub use self::surface::ColorSpace;
pub use self::surface::SurfaceCreationError;
pub use self::swapchain::Swapchain;
pub use self::swapchain::AcquireError;
pub use self::swapchain::PresentError;

pub mod display;
mod surface;
mod swapchain;

/// Internal trait so that creating/destroying a swapchain can access the surface's "has_swapchain"
/// flag.
unsafe trait SurfaceSwapchainLock {
    fn flag(&self) -> &AtomicBool;
}
