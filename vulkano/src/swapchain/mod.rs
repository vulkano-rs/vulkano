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
//! In order to draw on the screen or a window, you have to use two steps:
//! 
//! - Create a `Surface` object that represents the location where the image will show up.
//! - Create a `Swapchain` using that `Surface`.
//! 
//! Creating a surface can be done with only an `Instance` object. However creating a swapchain
//! requires a `Device` object.
//!
//! Once you have a swapchain, you can retreive `Image` objects from it and draw to them. However
//! due to double-buffering or other caching mechanism, the rendering will not automatically be
//! shown on screen. In order to show the output on screen, you have to *present* the swapchain
//! by using the method with the same name.
//!
//! # Extensions
//! 
//! Theses capabilities depend on some extensions:
//! 
//! - `VK_KHR_surface`
//! - `VK_KHR_swapchain`
//! - `VK_KHR_display`
//! - `VK_KHR_display_swapchain`
//! - `VK_KHR_xlib_surface`
//! - `VK_KHR_xcb_surface`
//! - `VK_KHR_wayland_surface`
//! - `VK_KHR_mir_surface`
//! - `VK_KHR_android_surface`
//! - `VK_KHR_win32_surface`
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
