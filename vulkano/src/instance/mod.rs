// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! API entry point.
//!
//! The first thing to do before you start using Vulkan is to create an `Instance` object.
//!
//! Creating an instance initializes everything and allows you to:
//! 
//!  - Enumerate physical devices, ie. all the Vulkan implementations that are available on
//!    the system.
//!  - Enumerate monitors.
//!  - Create surfaces (fullscreen or windowed) which will later be drawn upon.
//!
//! Enumerating monitors and creating surfaces can only be done if the proper extensions are
//! available and have been enabled. It is possible for a machine to support Vulkan without
//! support for rendering on a screen.
//!
//! # Application info
//! 
//! When you create an instance, you have the possibility to pass an `ApplicationInfo` struct. This
//! struct contains various information about your application, most notably its name and engine.
//! 
//! Passing such a structure allows for example the driver to let the user configure the driver's
//! behavior for your application alone through a control panel.
//!
//! # Enumerating physical devices
//!
//! After you have created an instance, the next step is to enumerate the physical devices that
//! are available on the system with `PhysicalDevice::enumerate()`.
//!
//! When choosing which physical device to use, keep in mind that physical devices may or may not
//! be able to draw to a certain surface (ie. to a window or a monitor), or may even not be able
//! to draw at all. See the `swapchain` module for more information about surfaces.
//!
//! A physical device can designate a video card, an integrated chip, but also multiple video
//! cards working together or a software implementation. Once you have chosen a physical device,
//! you can create a `Device` object from it. See the `device` module for more info.
//!
pub use features::Features;
pub use self::extensions::DeviceExtensions;
pub use self::extensions::InstanceExtensions;
pub use self::instance::Instance;
pub use self::instance::InstanceCreationError;
pub use self::instance::ApplicationInfo;
pub use self::instance::PhysicalDevice;
pub use self::instance::PhysicalDevicesIter;
pub use self::instance::PhysicalDeviceType;
pub use self::instance::QueueFamiliesIter;
pub use self::instance::QueueFamily;
pub use self::instance::MemoryTypesIter;
pub use self::instance::MemoryType;
pub use self::instance::MemoryHeapsIter;
pub use self::instance::MemoryHeap;
pub use self::instance::Limits;
pub use self::layers::layers_list;
pub use self::layers::LayerProperties;

pub mod debug;

mod extensions;
mod instance;
mod layers;
