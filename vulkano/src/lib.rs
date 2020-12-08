// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]
//! Safe and rich Rust wrapper around the Vulkan API.
//!
//! # Brief summary of Vulkan
//!
//! - The [`Instance`](instance/struct.Instance.html) object is the API entry point. It is the
//!   first object you must create before starting to use Vulkan.
//!
//! - The [`PhysicalDevice`](instance/struct.PhysicalDevice.html) object represents an
//!   implementation of Vulkan available on the system (eg. a graphics card, a software
//!   implementation, etc.). Physical devices can be enumerated from an instance with
//!   [`PhysicalDevice::enumerate()`](instance/struct.PhysicalDevice.html#method.enumerate).
//!
//! - Once you have chosen a physical device to use, you can create a
//!   [`Device`](device/index.html) object from it. The `Device` is the most important
//!   object of Vulkan, as it represents an open channel of communication with a physical device.
//!   You always need to have one before you can do interesting things with Vulkan.
//!
//! - [*Buffers*](buffer/index.html) and [*images*](image/index.html) can be used to store data on
//!   memory accessible by the GPU (or more generally by the Vulkan implementation). Buffers are
//!   usually used to store information about vertices, lights, etc. or arbitrary data, while
//!   images are used to store textures or multi-dimensional data.
//!
//! - In order to show something on the screen, you need a [`Swapchain`](swapchain/index.html).
//!   A `Swapchain` contains special `Image`s that correspond to the content of the window or the
//!   monitor. When you *present* a swapchain, the content of one of these special images is shown
//!   on the screen.
//!
//! - In order to ask the GPU to do something, you must create a
//!   [*command buffer*](command_buffer/index.html). A command buffer contains a list of commands
//!   that the GPU must perform. This can include copies between buffers and images, compute
//!   operations, or graphics operations. For the work to start, the command buffer must then be
//!   submitted to a [`Queue`](device/struct.Queue.html), which is obtained when you create the
//!   `Device`.
//!
//! - In order to be able to add a compute operation or a graphics operation to a command buffer,
//!   you need to have created a [`ComputePipeline` or a `GraphicsPipeline`
//!   object](pipeline/index.html) that describes the operation you want. These objects are usually
//!   created during your program's initialization. `Shader`s are programs that the GPU will
//!   execute as part of a pipeline. [*Descriptors*](descriptor/index.html) can be used to access
//!   the content of buffers or images from within shaders.
//!
//! - For graphical operations, [`RenderPass`es and `Framebuffer`s](framebuffer/index.html)
//!   describe on which images the implementation must draw upon.
//!
//! - Once you have built a *command buffer* that contains a list of commands, submitting it to the
//!   GPU will return an object that implements [the `GpuFuture` trait](sync/index.html).
//!   `GpuFuture`s allow you to chain multiple submissions together and are essential to performing
//!   multiple operations on multiple different GPU queues.
//!

//#![warn(missing_docs)]        // TODO: activate
#![allow(dead_code)] // TODO: remove
#![allow(unused_variables)] // TODO: remove

extern crate crossbeam;
extern crate fnv;
#[macro_use]
extern crate lazy_static;
pub extern crate half;
extern crate parking_lot;
extern crate shared_library;
extern crate smallvec;
extern crate vk_sys as vk;

#[macro_use]
mod tests;

#[macro_use]
mod extensions;
mod features;
mod version;

pub mod buffer;
pub mod command_buffer;
pub mod descriptor;
pub mod device;
pub mod format;
#[macro_use]
pub mod framebuffer;
pub mod image;
pub mod instance;
pub mod memory;
pub mod pipeline;
pub mod query;
pub mod sampler;
pub mod swapchain;
pub mod sync;

use std::error;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::MutexGuard;

/// Alternative to the `Deref` trait. Contrary to `Deref`, must always return the same object.
pub unsafe trait SafeDeref: Deref {}
unsafe impl<'a, T: ?Sized> SafeDeref for &'a T {}
unsafe impl<T: ?Sized> SafeDeref for Arc<T> {}
unsafe impl<T: ?Sized> SafeDeref for Box<T> {}

pub trait VulkanHandle {
    fn value(&self) -> u64;
}

impl VulkanHandle for usize {
    #[inline]
    fn value(&self) -> u64 {
        *self as u64
    }
}
impl VulkanHandle for u64 {
    #[inline]
    fn value(&self) -> u64 {
        *self
    }
}

/// Gives access to the internal identifier of an object.
pub unsafe trait VulkanObject {
    /// The type of the object.
    type Object: VulkanHandle;

    /// The `ObjectType` of the internal Vulkan handle.
    const TYPE: vk::ObjectType;

    /// Returns a reference to the object.
    fn internal_object(&self) -> Self::Object;
}

/// Gives access to the internal identifier of an object.
// TODO: remove ; crappy design
pub unsafe trait SynchronizedVulkanObject {
    /// The type of the object.
    type Object: VulkanHandle;

    /// Returns a reference to the object.
    fn internal_object_guard(&self) -> MutexGuard<Self::Object>;
}

/// Error type returned by most Vulkan functions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OomError {
    /// There is no memory available on the host (ie. the CPU, RAM, etc.).
    OutOfHostMemory,
    /// There is no memory available on the device (ie. video memory).
    OutOfDeviceMemory,
}

impl error::Error for OomError {}

impl fmt::Display for OomError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                OomError::OutOfHostMemory => "no memory available on the host",
                OomError::OutOfDeviceMemory => "no memory available on the graphical device",
            }
        )
    }
}

impl From<Error> for OomError {
    #[inline]
    fn from(err: Error) -> OomError {
        match err {
            Error::OutOfHostMemory => OomError::OutOfHostMemory,
            Error::OutOfDeviceMemory => OomError::OutOfDeviceMemory,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// All possible success codes returned by any Vulkan function.
#[derive(Debug, Copy, Clone)]
#[repr(u32)]
enum Success {
    Success = vk::SUCCESS,
    NotReady = vk::NOT_READY,
    Timeout = vk::TIMEOUT,
    EventSet = vk::EVENT_SET,
    EventReset = vk::EVENT_RESET,
    Incomplete = vk::INCOMPLETE,
    Suboptimal = vk::SUBOPTIMAL_KHR,
}

/// All possible errors returned by any Vulkan function.
///
/// This type is not public. Instead all public error types should implement `From<Error>` and
/// panic for error code that aren't supposed to happen.
#[derive(Debug, Copy, Clone)]
#[repr(u32)]
// TODO: being pub is necessary because of the weird visibility rules in rustc
pub(crate) enum Error {
    OutOfHostMemory = vk::ERROR_OUT_OF_HOST_MEMORY,
    OutOfDeviceMemory = vk::ERROR_OUT_OF_DEVICE_MEMORY,
    InitializationFailed = vk::ERROR_INITIALIZATION_FAILED,
    DeviceLost = vk::ERROR_DEVICE_LOST,
    MemoryMapFailed = vk::ERROR_MEMORY_MAP_FAILED,
    LayerNotPresent = vk::ERROR_LAYER_NOT_PRESENT,
    ExtensionNotPresent = vk::ERROR_EXTENSION_NOT_PRESENT,
    FeatureNotPresent = vk::ERROR_FEATURE_NOT_PRESENT,
    IncompatibleDriver = vk::ERROR_INCOMPATIBLE_DRIVER,
    TooManyObjects = vk::ERROR_TOO_MANY_OBJECTS,
    FormatNotSupported = vk::ERROR_FORMAT_NOT_SUPPORTED,
    SurfaceLost = vk::ERROR_SURFACE_LOST_KHR,
    NativeWindowInUse = vk::ERROR_NATIVE_WINDOW_IN_USE_KHR,
    OutOfDate = vk::ERROR_OUT_OF_DATE_KHR,
    IncompatibleDisplay = vk::ERROR_INCOMPATIBLE_DISPLAY_KHR,
    ValidationFailed = vk::ERROR_VALIDATION_FAILED_EXT,
    OutOfPoolMemory = vk::ERROR_OUT_OF_POOL_MEMORY_KHR,
    FullscreenExclusiveLost = vk::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
}

/// Checks whether the result returned correctly.
fn check_errors(result: vk::Result) -> Result<Success, Error> {
    match result {
        vk::SUCCESS => Ok(Success::Success),
        vk::NOT_READY => Ok(Success::NotReady),
        vk::TIMEOUT => Ok(Success::Timeout),
        vk::EVENT_SET => Ok(Success::EventSet),
        vk::EVENT_RESET => Ok(Success::EventReset),
        vk::INCOMPLETE => Ok(Success::Incomplete),
        vk::ERROR_OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory),
        vk::ERROR_OUT_OF_DEVICE_MEMORY => Err(Error::OutOfDeviceMemory),
        vk::ERROR_INITIALIZATION_FAILED => Err(Error::InitializationFailed),
        vk::ERROR_DEVICE_LOST => Err(Error::DeviceLost),
        vk::ERROR_MEMORY_MAP_FAILED => Err(Error::MemoryMapFailed),
        vk::ERROR_LAYER_NOT_PRESENT => Err(Error::LayerNotPresent),
        vk::ERROR_EXTENSION_NOT_PRESENT => Err(Error::ExtensionNotPresent),
        vk::ERROR_FEATURE_NOT_PRESENT => Err(Error::FeatureNotPresent),
        vk::ERROR_INCOMPATIBLE_DRIVER => Err(Error::IncompatibleDriver),
        vk::ERROR_TOO_MANY_OBJECTS => Err(Error::TooManyObjects),
        vk::ERROR_FORMAT_NOT_SUPPORTED => Err(Error::FormatNotSupported),
        vk::ERROR_SURFACE_LOST_KHR => Err(Error::SurfaceLost),
        vk::ERROR_NATIVE_WINDOW_IN_USE_KHR => Err(Error::NativeWindowInUse),
        vk::SUBOPTIMAL_KHR => Ok(Success::Suboptimal),
        vk::ERROR_OUT_OF_DATE_KHR => Err(Error::OutOfDate),
        vk::ERROR_INCOMPATIBLE_DISPLAY_KHR => Err(Error::IncompatibleDisplay),
        vk::ERROR_VALIDATION_FAILED_EXT => Err(Error::ValidationFailed),
        vk::ERROR_OUT_OF_POOL_MEMORY_KHR => Err(Error::OutOfPoolMemory),
        vk::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => Err(Error::FullscreenExclusiveLost),
        vk::ERROR_INVALID_SHADER_NV => panic!(
            "Vulkan function returned \
                                               VK_ERROR_INVALID_SHADER_NV"
        ),
        c => unreachable!("Unexpected error code returned by Vulkan: {}", c),
    }
}
