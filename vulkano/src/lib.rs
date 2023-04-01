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
//! # Cargo features
//!
//! | Feature              | Description                                                                   |
//! |----------------------|-------------------------------------------------------------------------------|
//! | `document_unchecked` | Include `_unchecked` functions in the generated documentation.                |
//! | `serde`              | Enables (de)serialization of certain types using [`serde`].                   |
//!
//! # Starting off with Vulkano
//!
//! The steps for using Vulkan through Vulkano are in principle not any different from using
//! the raw Vulkan API, but the details may be different for the sake of idiomaticity, safety
//! and convenience.
//!
//! 1. Create a [`VulkanLibrary`]. This represents a Vulkan library on the system, which must be
//!    loaded before you can do anything with Vulkan.
//!
//! 2. Create an [`Instance`]. This is the API entry point, and represents an initialised Vulkan
//!    library.
//!
//! 3. If you intend to show graphics to the user on a window or a screen, create a [`Surface`].
//!    A `Surface` is created from a window identifier or handle, that is specific to the display or
//!    windowing system being used. The [`vulkano-win`] crate, which is part of the Vulkano
//!    project, can make this step easier.
//!
//! 4. [Enumerate the physical devices] that are available on the `Instance`, and choose one that
//!    is suitable for your program. A [`PhysicalDevice`] represents a Vulkan-capable device that
//!    is available on the system, such as a graphics card, a software implementation, etc.
//!
//! 6. Create a [`Device`] and accompanying [`Queue`]s from the selected `PhysicalDevice`.
//!    The `Device` is the most important object of Vulkan, and you need one to create almost
//!    every other object. `Queue`s are created together with the `Device`, and are used to submit
//!    work to the device to make it do something.
//!
//! 7. If you created a `Surface` earlier, create a [`Swapchain`]. This object contains special
//!    images that correspond to the contents of the surface. Whenever you want to
//!    change the contents (show something new to the user), you must first *acquire* one of these
//!    images from the swapchain, fill it with the new contents (by rendering, copying or any
//!    other means), and then *present* it back to the swapchain.
//!    A swapchain can become outdated if the properties of the surface change, such as when
//!    the size of the window changes. It then becomes necessary to create a new swapchain.
//!
//! 8. Record a [*command buffer*](crate::command_buffer), containing commands that the device must
//!    execute. Then build the command buffer and submit it to a `Queue`.
//!
//! Many different operations can be recorded to a command buffer, such as *draw*, *compute* and
//! *transfer* operations. To do any of these things, you will need to create several other objects,
//! depending on your specific needs. This includes:
//!
//! - [*Buffers*] store general-purpose data on memory accessible by the device. This can include
//!   mesh data (vertices, texture coordinates etc.), lighting information, matrices, and anything
//!   else you can think of.
//!
//! - [*Images*] store texel data, arranged in a grid of one or more dimensions. They can be used
//!   as textures, depth/stencil buffers, framebuffers and as part of a swapchain.
//!
//! - [*Pipelines*] describe operations on the device. They include one or more [*shader*]s, small
//!   programs that the device will execute as part of a pipeline.
//!   Pipelines come in several types:
//!   - A [`ComputePipeline`] describes how *dispatch* commands are to be performed.
//!   - A [`GraphicsPipeline`] describes how *draw* commands are to be performed.
//!
//! - [*Descriptor sets*] make buffers, images and other objects available to shaders. The
//!   arrangement of these resources in shaders is described by a [`DescriptorSetLayout`]. One or
//!   more of these layouts in turn forms a [`PipelineLayout`], which is used when creating a
//!   pipeline object.
//!
//! - For more complex, multi-stage draw operations, you can create a [`RenderPass`] object.
//!   This object describes the stages, known as subpasses, that draw operations consist of,
//!   how they interact with one another, and which types of images are available in each subpass.
//!   You must also create a [`Framebuffer`], which contains the image objects that are to be used
//!   in a render pass.
//!
//! # `_unchecked` functions
//!
//! Many functions in Vulkano have two versions: the normal function, which is usually safe to
//! call, and another function with `_unchecked` added onto the end of the name, which is unsafe
//! to call. The `_unchecked` functions skip all validation checks, so they are usually more
//! efficient, but you must ensure that you meet the validity/safety requirements of the function.
//!
//! For all `_unchecked` functions, a call to the function is valid, if a call to the
//! corresponding normal function with the same arguments would return without any error.
//! This includes following all the valid usage requirements of the Vulkan specification, but may
//! also include additional requirements specific to Vulkano.
//! **All other usage of `_unchecked` functions may be undefined behavior.**
//!
//! Because there are potentially many `_unchecked` functions, and because their name and operation
//! can be straightforwardly understood based on the corresponding normal function, they are hidden
//! from the Vulkano documentation by default. You can unhide them by enabling the
//! `document_unchecked` cargo feature, and then generating the documentation with the command
//! `cargo doc --open`.
//!
//! [`cgmath`]: https://crates.io/crates/cgmath
//! [`nalgebra`]: https://crates.io/crates/nalgebra
//! [`serde`]: https://crates.io/crates/serde
//! [`VulkanLibrary`]: crate::VulkanLibrary
//! [`Instance`]: crate::instance::Instance
//! [`Surface`]: crate::swapchain::Surface
//! [`vulkano-win`]: https://crates.io/crates/vulkano-win
//! [Enumerate the physical devices]: crate::instance::Instance::enumerate_physical_devices
//! [`PhysicalDevice`]: crate::device::physical::PhysicalDevice
//! [`Device`]: crate::device::Device
//! [`Queue`]: crate::device::Queue
//! [`Swapchain`]: crate::swapchain::Swapchain
//! [*command buffer*]: crate::command_buffer
//! [*Buffers*]: crate::buffer
//! [*Images*]: crate::image
//! [*Pipelines*]: crate::pipeline
//! [*shader*]: crate::shader
//! [`ComputePipeline`]: crate::pipeline::ComputePipeline
//! [`GraphicsPipeline`]: crate::pipeline::GraphicsPipeline
//! [*Descriptor sets*]: crate::descriptor_set
//! [`DescriptorSetLayout`]: crate::descriptor_set::layout
//! [`PipelineLayout`]: crate::pipeline::layout
//! [`RenderPass`]: crate::render_pass::RenderPass
//! [`Framebuffer`]: crate::render_pass::Framebuffer

//#![warn(missing_docs)]        // TODO: activate
#![warn(
    rust_2018_idioms,
    rust_2021_compatibility,
    clippy::trivially_copy_pass_by_ref
)]
// These lints are a bit too pedantic, so they're disabled here.
#![allow(
    clippy::collapsible_else_if,
    clippy::collapsible_if,
    clippy::large_enum_variant,
    clippy::len_without_is_empty,
    clippy::missing_safety_doc, // TODO: remove
    clippy::module_inception,
    clippy::mutable_key_type,
    clippy::needless_borrowed_reference,
    clippy::new_without_default,
    clippy::nonminimal_bool,
    clippy::op_ref, // Seems to be bugged, the fixed code triggers a compile error
    clippy::result_large_err,
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
    num::NonZeroU64,
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
pub mod padded;
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

/// A [`DeviceSize`] that is known not to equal zero.
pub type NonZeroDeviceSize = NonZeroU64;

// Allow refering to crate by its name to work around limitations of proc-macros
// in doctests.
// See https://github.com/rust-lang/cargo/issues/9886
// and https://github.com/bkchr/proc-macro-crate/issues/10
#[allow(unused_extern_crates)]
extern crate self as vulkano;

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
                VulkanError::ImageUsageNotSupported =>
                    "the requested `ImageUsage` are not supported",
                VulkanError::VideoPictureLayoutNotSupported =>
                    "the requested video picture layout is not supported",
                VulkanError::VideoProfileOperationNotSupported =>
                    "a video profile operation specified via \
                    `VideoProfileInfo::video_codec_operation` is not supported",
                VulkanError::VideoProfileFormatNotSupported =>
                    "format parameters in a requested `VideoProfileInfo` chain are not supported",
                VulkanError::VideoProfileCodecNotSupported =>
                    "codec-specific parameters in a requested `VideoProfileInfo` chain are not \
                    supported",
                VulkanError::VideoStdVersionNotSupported =>
                    "the specified video Std header version is not supported",
                VulkanError::CompressionExhausted =>
                    "an image creation failed because internal resources required for compression \
                    are exhausted",
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
