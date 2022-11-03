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
//!   [`Instance::enumerate_physical_devices`](crate::instance::Instance::enumerate_physical_devices).
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
#![warn(rust_2018_idioms, rust_2021_compatibility)]
// These lints are a bit too pedantic, so they're disabled here.
#![allow(
    clippy::collapsible_else_if,
    clippy::collapsible_if,
    clippy::large_enum_variant,
    clippy::len_without_is_empty,
    clippy::missing_safety_doc, // TODO: remove
    clippy::module_inception,
    clippy::mutable_key_type,
    clippy::new_without_default,
    clippy::nonminimal_bool,
    clippy::op_ref, // Seems to be bugged, the fixed code triggers a compile error
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::vec_box,
    clippy::wrong_self_convention
)]

pub use ash::vk::Handle;
pub use half;
pub use library::{LoadingError, VulkanLibrary};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    ops::Deref,
    sync::Arc,
};
pub use {extensions::ExtensionProperties, version::Version};

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
mod cache;
mod fns;
pub mod image;
pub mod instance;
pub mod library;
mod macros;
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
    type Handle: ash::vk::Handle;

    /// Returns the raw Vulkan handle of the object.
    fn handle(&self) -> Self::Handle;
}

unsafe impl<T, U> VulkanObject for T
where
    T: SafeDeref<Target = U>,
    U: VulkanObject + ?Sized,
{
    type Handle = U::Handle;

    #[inline]
    fn handle(&self) -> Self::Handle {
        (**self).handle()
    }
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

impl Display for OomError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
                OomError::OutOfHostMemory => "no memory available on the host",
                OomError::OutOfDeviceMemory => "no memory available on the graphical device",
            }
        )
    }
}

impl From<VulkanError> for OomError {
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

impl Display for VulkanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
                VulkanError::OutOfHostMemory => "a host memory allocation has failed",
                VulkanError::OutOfDeviceMemory => "a device memory allocation has failed",
                VulkanError::InitializationFailed => {
                    "initialization of an object could not be completed for \
                    implementation-specific reasons"
                }
                VulkanError::DeviceLost => "the logical or physical device has been lost",
                VulkanError::MemoryMapFailed => "mapping of a memory object has failed",
                VulkanError::LayerNotPresent => {
                    "a requested layer is not present or could not be loaded"
                }
                VulkanError::ExtensionNotPresent => "a requested extension is not supported",
                VulkanError::FeatureNotPresent => "a requested feature is not supported",
                VulkanError::IncompatibleDriver => {
                    "the requested version of Vulkan is not supported by the driver or is \
                    otherwise incompatible for implementation-specific reasons"
                }
                VulkanError::TooManyObjects => {
                    "too many objects of the type have already been created"
                }
                VulkanError::FormatNotSupported => {
                    "a requested format is not supported on this device"
                }
                VulkanError::FragmentedPool => {
                    "a pool allocation has failed due to fragmentation of the pool's memory"
                }
                VulkanError::Unknown => {
                    "an unknown error has occurred; either the application has provided invalid \
                    input, or an implementation failure has occurred"
                }
                VulkanError::OutOfPoolMemory => "a pool memory allocation has failed",
                VulkanError::InvalidExternalHandle => {
                    "an external handle is not a valid handle of the specified type"
                }
                VulkanError::Fragmentation => {
                    "a descriptor pool creation has failed due to fragmentation"
                }
                VulkanError::InvalidOpaqueCaptureAddress => {
                    "a buffer creation or memory allocation failed because the requested address \
                    is not available. A shader group handle assignment failed because the \
                    requested shader group handle information is no longer valid"
                }
                VulkanError::IncompatibleDisplay => {
                    "the display used by a swapchain does not use the same presentable image \
                    layout, or is incompatible in a way that prevents sharing an image"
                }
                VulkanError::NotPermitted => "a requested operation was not permitted",
                VulkanError::SurfaceLost => "a surface is no longer available",
                VulkanError::NativeWindowInUse => {
                    "the requested window is already in use by Vulkan or another API in a manner \
                    which prevents it from being used again"
                }
                VulkanError::OutOfDate => {
                    "a surface has changed in such a way that it is no longer compatible with the \
                    swapchain, and further presentation requests using the swapchain will fail"
                }
                VulkanError::ValidationFailed => "validation failed",
                VulkanError::FullScreenExclusiveModeLost => {
                    "an operation on a swapchain created with application controlled full-screen \
                    access failed as it did not have exclusive full-screen access"
                }
                VulkanError::InvalidDrmFormatModifierPlaneLayout => {
                    "the requested DRM format modifier plane layout is invalid"
                }
                VulkanError::InvalidShader => "one or more shaders failed to compile or link",
                VulkanError::Unnamed(result) =>
                    return write!(f, "unnamed error, VkResult value {}", result.as_raw()),
            }
        )
    }
}

/// Used in errors to indicate a set of alternatives that needs to be available/enabled to allow
/// a given operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequiresOneOf {
    /// A minimum Vulkan API version that would allow the operation.
    pub api_version: Option<Version>,

    /// Enabled features that would allow the operation.
    pub features: &'static [&'static str],

    /// Available/enabled device extensions that would allow the operation.
    pub device_extensions: &'static [&'static str],

    /// Available/enabled instance extensions that would allow the operation.
    pub instance_extensions: &'static [&'static str],
}

impl RequiresOneOf {
    /// Returns whether there is more than one possible requirement.
    pub fn len(&self) -> usize {
        self.api_version.map_or(0, |_| 1)
            + self.features.len()
            + self.device_extensions.len()
            + self.instance_extensions.len()
    }
}

impl Display for RequiresOneOf {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let mut members_written = 0;

        if let Some(version) = self.api_version {
            write!(f, "Vulkan API version {}.{}", version.major, version.minor)?;
            members_written += 1;
        }

        if let Some((last, rest)) = self.features.split_last() {
            if members_written != 0 {
                write!(f, ", ")?;
            }

            members_written += 1;

            if rest.is_empty() {
                write!(f, "feature {}", last)?;
            } else {
                write!(f, "features ")?;

                for feature in rest {
                    write!(f, "{}, ", feature)?;
                }

                write!(f, "{}", last)?;
            }
        }

        if let Some((last, rest)) = self.device_extensions.split_last() {
            if members_written != 0 {
                write!(f, ", ")?;
            }

            members_written += 1;

            if rest.is_empty() {
                write!(f, "device extension {}", last)?;
            } else {
                write!(f, "device extensions ")?;

                for feature in rest {
                    write!(f, "{}, ", feature)?;
                }

                write!(f, "{}", last)?;
            }
        }

        if let Some((last, rest)) = self.instance_extensions.split_last() {
            if members_written != 0 {
                write!(f, ", ")?;
            }

            if rest.is_empty() {
                write!(f, "instance extension {}", last)?;
            } else {
                write!(f, "instance extensions ")?;

                for feature in rest {
                    write!(f, "{}, ", feature)?;
                }

                write!(f, "{}", last)?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RequirementNotMet {
    pub(crate) required_for: &'static str,
    pub(crate) requires_one_of: RequiresOneOf,
}

/// A helper type for non-exhaustive structs.
///
/// This type cannot be constructed outside Vulkano. Structures with a field of this type can only
/// be constructed by calling a constructor function or `Default::default()`. The effect is similar
/// to the standard Rust `#[non_exhaustive]` attribute, except that it does not prevent update
/// syntax from being used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] // add traits as needed
pub struct NonExhaustive(pub(crate) ());

macro_rules! impl_id_counter {
    ($type:ident $(< $($param:ident $(: $bound:ident $(+ $bounds:ident)* )?),+ >)?) => {
        $crate::impl_id_counter!(
            @inner $type $(< $($param),+ >)?, $( $($param $(: $bound $(+ $bounds)* )?),+)?
        );
    };
    ($type:ident $(< $($param:ident $(: $bound:ident $(+ $bounds:ident)* )? + ?Sized),+ >)?) => {
        $crate::impl_id_counter!(
            @inner $type $(< $($param),+ >)?, $( $($param $(: $bound $(+ $bounds)* )? + ?Sized),+)?
        );
    };
    (@inner $type:ident $(< $($param:ident),+ >)?, $($bounds:tt)*) => {
        impl< $($bounds)* > $type $(< $($param),+ >)? {
            fn next_id() -> std::num::NonZeroU64 {
                use std::{
                    num::NonZeroU64,
                    sync::atomic::{AtomicU64, Ordering},
                };

                static COUNTER: AtomicU64 = AtomicU64::new(1);

                NonZeroU64::new(COUNTER.fetch_add(1, Ordering::Relaxed)).unwrap_or_else(|| {
                    println!("an ID counter has overflown ...somehow");
                    std::process::abort();
                })
            }
        }

        impl< $($bounds)* > PartialEq for $type $(< $($param),+ >)? {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.id == other.id
            }
        }

        impl< $($bounds)* > Eq for $type $(< $($param),+ >)? {}

        impl< $($bounds)* > std::hash::Hash for $type $(< $($param),+ >)? {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.id.hash(state);
            }
        }
    };
}

pub(crate) use impl_id_counter;
