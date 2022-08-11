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
//! - The [`VulkanLibrary`](crate::VulkanLibrary) represents a Vulkan library on the system.
//!   It must be loaded before you can do anything with Vulkan.
//!
//! - The [`Instance`](crate::instance::Instance) object is the API entry point, and represents
//!   an initialised Vulkan library. This is the first Vulkan object that you create.
//!
//! - The [`PhysicalDevice`](crate::device::physical::PhysicalDevice) object represents a
//!   Vulkan-capable device that is available on the system (eg. a graphics card, a software
//!   implementation, etc.). Physical devices can be enumerated from an instance with
//!   [`PhysicalDevice::enumerate`](crate::device::physical::PhysicalDevice::enumerate).
//!
//! - Once you have chosen a physical device to use, you can create a
//!   [`Device`](crate::device::Device) object from it. The `Device` is the most important
//!   object of Vulkan, as it represents an open channel of communication with a physical device.
//!   You always need to have one before you can do interesting things with Vulkan.
//!
//! - [*Buffers*](crate::buffer) and [*images*](crate::image) can be used to store data on
//!   memory accessible by the GPU (or more generally by the Vulkan implementation). Buffers are
//!   usually used to store information about vertices, lights, etc. or arbitrary data, while
//!   images are used to store textures or multi-dimensional data.
//!
//! - In order to show something on the screen, you need a
//!   [`Surface` and a `Swapchain`](crate::swapchain).
//!   A `Swapchain` contains special `Image`s that correspond to the content of the window or the
//!   monitor. When you *present* a swapchain, the content of one of these special images is shown
//!   on the screen.
//!
//! - For graphical operations, [`RenderPass`es and `Framebuffer`s](crate::render_pass)
//!   describe which images the device must draw upon.
//!
//! - In order to be able to perform operations on the device, you need to have created a
//!   [pipeline object](crate::pipeline) that describes the operation you want. These objects are usually
//!   created during your program's initialization. `Shader`s are programs that the GPU will
//!   execute as part of a pipeline. [*Descriptor sets*](crate::descriptor_set) can be used to access
//!   the content of buffers or images from within shaders.
//!
//! - To tell the GPU to do something, you must create a
//!   [*command buffer*](crate::command_buffer). A command buffer contains a list of commands
//!   that the GPU must perform. This can include copies between buffers and images, compute
//!   operations, or graphics operations. For the work to start, the command buffer must then be
//!   submitted to a [`Queue`](crate::device::Queue), which is obtained when you create the
//!   `Device`.
//!
//! - Once you have built a *command buffer* that contains a list of commands, submitting it to the
//!   GPU will return an object that implements [the `GpuFuture` trait](crate::sync::GpuFuture).
//!   `GpuFuture`s allow you to chain multiple submissions together and are essential to performing
//!   multiple operations on multiple different GPU queues.
//!

//#![warn(missing_docs)]        // TODO: activate
#![allow(dead_code)] // TODO: remove
#![allow(unused_variables)] // TODO: remove

pub use ash::vk::Handle;
pub use half;
pub use library::VulkanLibrary;
use parking_lot::MutexGuard;
use std::{error::Error, fmt, ops::Deref, sync::Arc};
pub use version::Version;

#[macro_use]
mod tests;
#[macro_use]
mod extensions;
pub mod buffer;
pub mod command_buffer;
pub mod descriptor_set;
pub mod device;
pub mod format;
mod version;
#[macro_use]
pub mod render_pass;
mod fns;
pub mod image;
pub mod instance;
pub mod library;
pub mod memory;
pub mod pipeline;
pub mod query;
mod range_map;
pub mod range_set;
pub mod sampler;
pub mod shader;
pub mod swapchain;
pub mod sync;

/// Represents memory size and offset values on a Vulkan device.
/// Analogous to the Rust `usize` type on the host.
pub use ash::vk::DeviceSize;

/// Alternative to the `Deref` trait. Contrary to `Deref`, must always return the same object.
pub unsafe trait SafeDeref: Deref {}
unsafe impl<'a, T: ?Sized> SafeDeref for &'a T {}
unsafe impl<T: ?Sized> SafeDeref for Arc<T> {}
unsafe impl<T: ?Sized> SafeDeref for Box<T> {}

/// Gives access to the internal identifier of an object.
pub unsafe trait VulkanObject {
    /// The type of the object.
    type Object: ash::vk::Handle;

    /// Returns a reference to the object.
    fn internal_object(&self) -> Self::Object;
}

/// Gives access to the internal identifier of an object.
// TODO: remove ; crappy design
pub unsafe trait SynchronizedVulkanObject {
    /// The type of the object.
    type Object: ash::vk::Handle;

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

impl Error for OomError {}

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

impl From<VulkanError> for OomError {
    #[inline]
    fn from(err: VulkanError) -> OomError {
        match err {
            VulkanError::OutOfHostMemory => OomError::OutOfHostMemory,
            VulkanError::OutOfDeviceMemory => OomError::OutOfDeviceMemory,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

// Generated by build.rs
include!(concat!(env!("OUT_DIR"), "/errors.rs"));

impl Error for VulkanError {}

impl fmt::Display for VulkanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VulkanError::OutOfHostMemory => write!(
                f,
                "A host memory allocation has failed.",
            ),
            VulkanError::OutOfDeviceMemory => write!(
                f,
                "A device memory allocation has failed.",
            ),
            VulkanError::InitializationFailed => write!(
                f,
                "Initialization of an object could not be completed for implementation-specific reasons.",
            ),
            VulkanError::DeviceLost => write!(
                f,
                "The logical or physical device has been lost.",
            ),
            VulkanError::MemoryMapFailed => write!(
                f,
                "Mapping of a memory object has failed.",
            ),
            VulkanError::LayerNotPresent => write!(
                f,
                "A requested layer is not present or could not be loaded.",
            ),
            VulkanError::ExtensionNotPresent => write!(
                f,
                "A requested extension is not supported.",
            ),
            VulkanError::FeatureNotPresent => write!(
                f,
                "A requested feature is not supported.",
            ),
            VulkanError::IncompatibleDriver => write!(
                f,
                "The requested version of Vulkan is not supported by the driver or is otherwise incompatible for implementation-specific reasons.",
            ),
            VulkanError::TooManyObjects => write!(
                f,
                "Too many objects of the type have already been created.",
            ),
            VulkanError::FormatNotSupported => write!(
                f,
                "A requested format is not supported on this device.",
            ),
            VulkanError::FragmentedPool => write!(
                f,
                "A pool allocation has failed due to fragmentation of the pool's memory.",
            ),
            VulkanError::Unknown => write!(
                f,
                "An unknown error has occurred; either the application has provided invalid input, or an implementation failure has occurred.",
            ),
            VulkanError::OutOfPoolMemory => write!(
                f,
                "A pool memory allocation has failed.",
            ),
            VulkanError::InvalidExternalHandle => write!(
                f,
                "An external handle is not a valid handle of the specified type.",
            ),
            VulkanError::Fragmentation => write!(
                f,
                "A descriptor pool creation has failed due to fragmentation.",
            ),
            VulkanError::InvalidOpaqueCaptureAddress => write!(
                f,
                "A buffer creation or memory allocation failed because the requested address is not available. A shader group handle assignment failed because the requested shader group handle information is no longer valid.",
            ),
            VulkanError::IncompatibleDisplay => write!(
                f,
                "The display used by a swapchain does not use the same presentable image layout, or is incompatible in a way that prevents sharing an image.",
            ),
            VulkanError::NotPermitted => write!(
                f,
                "A requested operation was not permitted.",
            ),
            VulkanError::SurfaceLost => write!(
                f,
                "A surface is no longer available.",
            ),
            VulkanError::NativeWindowInUse => write!(
                f,
                "The requested window is already in use by Vulkan or another API in a manner which prevents it from being used again.",
            ),
            VulkanError::OutOfDate => write!(
                f,
                "A surface has changed in such a way that it is no longer compatible with the swapchain, and further presentation requests using the swapchain will fail.",
            ),
            VulkanError::ValidationFailed => write!(
                f,
                "Validation failed.",
            ),
            VulkanError::FullScreenExclusiveModeLost => write!(
                f,
                "An operation on a swapchain created with application controlled full-screen access failed as it did not have exclusive full-screen access.",
            ),
            VulkanError::InvalidDrmFormatModifierPlaneLayout => write!(
                f,
                "The requested DRM format modifier plane layout is invalid.",
            ),
            VulkanError::InvalidShader => write!(
                f,
                "One or more shaders failed to compile or link.",
            ),
            VulkanError::Unnamed(result) => write!(
                f,
                "Unnamed error, VkResult value {}",
                result.as_raw(),
            ),
        }
    }
}

/// A helper type for non-exhaustive structs.
///
/// This type cannot be constructed outside Vulkano. Structures with a field of this type can only
/// be constructed by calling a constructor function or `Default::default()`. The effect is similar
/// to the standard Rust `#[non_exhaustive]` attribute, except that it does not prevent update
/// syntax from being used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] // add traits as needed
pub struct NonExhaustive(pub(crate) ());
