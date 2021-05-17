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
//! use vulkano::Version;
//!
//! let instance = match Instance::new(None, Version::major_minor(1, 1), &InstanceExtensions::none(), None) {
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
//! # use vulkano::Version;
//! use vulkano::instance::PhysicalDevice;
//!
//! # let instance = Instance::new(None, Version::major_minor(1, 1), &InstanceExtensions::none(), None).unwrap();
//! for physical_device in PhysicalDevice::enumerate(&instance) {
//!     println!("Available device: {}", physical_device.name());
//! }
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

pub use self::extensions::InstanceExtensions;
pub use self::extensions::RawInstanceExtensions;
pub use self::instance::ApplicationInfo;
pub use self::instance::Instance;
pub use self::instance::InstanceCreationError;
pub use self::layers::layers_list;
pub use self::layers::LayerProperties;
pub use self::layers::LayersIterator;
pub use self::layers::LayersListError;
pub use self::limits::Limits;
pub use self::loader::LoadingError;
pub use self::physical_device::MemoryHeap;
pub use self::physical_device::MemoryHeapsIter;
pub use self::physical_device::MemoryType;
pub use self::physical_device::MemoryTypesIter;
pub use self::physical_device::PhysicalDevice;
pub use self::physical_device::PhysicalDeviceType;
pub use self::physical_device::PhysicalDevicesIter;
pub use self::physical_device::QueueFamiliesIter;
pub use self::physical_device::QueueFamily;
pub use crate::version::Version;

pub mod debug;
mod extensions;
mod instance;
mod layers;
mod limits;
pub mod loader;
mod physical_device;
