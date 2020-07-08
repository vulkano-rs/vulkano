// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! API entry point.
//!
//! The first thing to do before you start using Vulkan is to create an `Instance` object.
//!
//! For example:
//!
//! ```no_run
//! use vulkano::instance::Instance;
//! use vulkano::instance::InstanceExtensions;
//!
//! let instance = match Instance::new(None, &InstanceExtensions::none(), None) {
//!     Ok(i) => i,
//!     Err(err) => panic!("Couldn't build instance: {:?}", err)
//! };
//! ```
//!
//! Creating an instance initializes everything and allows you to enumerate physical devices,
//! ie. all the Vulkan implementations that are available on the system.
//!
//! ```no_run
//! # use vulkano::instance::Instance;
//! # use vulkano::instance::InstanceExtensions;
//! use vulkano::instance::PhysicalDevice;
//!
//! # let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
//! for physical_device in PhysicalDevice::enumerate(&instance) {
//!     println!("Available device: {}", physical_device.name());
//! }
//! ```
//!
//! # Extensions
//!
//! Notice the second parameter of `Instance::new()`. It is an `InstanceExtensions` struct that
//! contains a list of extensions that must be enabled on the newly-created instance. Trying to
//! enable an extension that is not supported by the system will result in an error.
//!
//! Contrary to OpenGL, it is not possible to use the features of an extension if it was not
//! explicitly enabled.
//!
//! Extensions are especially important to take into account if you want to render images on the
//! screen, as the only way to do so is to use the `VK_KHR_surface` extension. More information
//! about this in the `swapchain` module.
//!
//! For example, here is how we create an instance with the `VK_KHR_surface` and
//! `VK_KHR_android_surface` extensions enabled, which will allow us to render images to an
//! Android screen. You can compile and run this code on any system, but it is highly unlikely to
//! succeed on anything else than an Android-running device.
//!
//! ```no_run
//! use vulkano::instance::Instance;
//! use vulkano::instance::InstanceExtensions;
//!
//! let extensions = InstanceExtensions {
//!     khr_surface: true,
//!     khr_android_surface: true,
//!     .. InstanceExtensions::none()
//! };
//!
//! let instance = match Instance::new(None, &extensions, None) {
//!     Ok(i) => i,
//!     Err(err) => panic!("Couldn't build instance: {:?}", err)
//! };
//! ```
//!
//! # Application info
//!
//! When you create an instance, you have the possibility to pass an `ApplicationInfo` struct as
//! the first parameter. This struct contains various information about your application, most
//! notably its name and engine.
//!
//! Passing such a structure allows for example the driver to let the user configure the driver's
//! behavior for your application alone through a control panel.
//!
//! ```no_run
//! # #[macro_use] extern crate vulkano;
//! # fn main() {
//! use vulkano::instance::{Instance, InstanceExtensions};
//!
//! // Builds an `ApplicationInfo` by looking at the content of the `Cargo.toml` file at
//! // compile-time.
//! let app_infos = app_info_from_cargo_toml!();
//!
//! let _instance = Instance::new(Some(&app_infos), &InstanceExtensions::none(), None).unwrap();
//! # }
//! ```
//!
//! # Enumerating physical devices and creating a device
//!
//! After you have created an instance, the next step is usually to enumerate the physical devices
//! that are available on the system with `PhysicalDevice::enumerate()` (see above).
//!
//! When choosing which physical device to use, keep in mind that physical devices may or may not
//! be able to draw to a certain surface (ie. to a window or a monitor), or may even not be able
//! to draw at all. See the `swapchain` module for more information about surfaces.
//!
//! Once you have chosen a physical device, you can create a `Device` object from it. See the
//! `device` module for more info.
//!

pub use self::extensions::InstanceExtensions;
pub use self::extensions::RawInstanceExtensions;
pub use self::instance::ApplicationInfo;
pub use self::instance::Instance;
pub use self::instance::InstanceCreationError;
pub use self::instance::MemoryHeap;
pub use self::instance::MemoryHeapsIter;
pub use self::instance::MemoryType;
pub use self::instance::MemoryTypesIter;
pub use self::instance::PhysicalDevice;
pub use self::instance::PhysicalDeviceType;
pub use self::instance::PhysicalDevicesIter;
pub use self::instance::QueueFamiliesIter;
pub use self::instance::QueueFamily;
pub use self::layers::layers_list;
pub use self::layers::LayerProperties;
pub use self::layers::LayersIterator;
pub use self::layers::LayersListError;
pub use self::limits::Limits;
pub use self::loader::LoadingError;
pub use version::Version;

pub mod debug;
pub mod loader;

mod extensions;
mod instance;
mod layers;
mod limits;
