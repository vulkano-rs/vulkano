// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Images storage (1D, 2D, 3D, arrays, etc.).
//! 
//! An *image* is a location in memory whose purpose is to store multi-dimensional data. Its
//! most common usage is to store a 2D array of color pixels (in other words an *image* in the
//! everyday language), but it can also be used to store arbitrary data.
//!
//! The advantage of using an image compared to a buffer is that the memory layout is optimized
//! for locality. When reading a specific pixel of an image, reading the nearby pixels is really
//! fast. Most implementations have hardware dedicated to reading from images if you access them
//! through a sampler.
//!
//! # Properties of an image
//!
//! # Images and image views
//!
//! There is a distinction between *images* and *image views*. As its name tells, an image view
//! describes how the GPU must interpret the image.
//!
//! Transfer and memory operations operate on images themselves, while reading/writing an image
//! operates on image views. You can create multiple image views from the same image.
//!
//! # High-level wrappers
//!
//! In the vulkano library, an image is any object that implements the `Image` trait and an image
//! view is any object that implements the `ImageView` trait.
//!
//! Since these traits are low-level, you are encouraged to not implement them yourself but instead
//! use one of the provided implementations that are specialized depending on the way you are going
//! to use the image:
//!
//! - An `AttachmentImage` can be used when you want to draw to an image.
//! - An `ImmutableImage` stores data which never need be changed after the initial upload,
//!   like a texture.
//!
//! # Low-level informations
//!
//! To be written.
//!

pub use self::attachment::AttachmentImage;
pub use self::immutable::ImmutableImage;
pub use self::swapchain::SwapchainImage;
pub use self::sys::ImageCreationError;
pub use self::sys::Layout;
pub use self::sys::Usage;
pub use self::traits::Image;
pub use self::traits::ImageView;

pub mod attachment;     // TODO: make private
pub mod immutable;      // TODO: make private
pub mod swapchain;      // TODO: make private
pub mod sys;
pub mod traits;

/// Specifies how many mipmaps must be allocated.
///
/// Note that at least one mipmap must be allocated, to store the main level of the image.
#[derive(Debug, Copy, Clone)]
pub enum MipmapsCount {
    /// Allocates the number of mipmaps required to store all the mipmaps of the image where each
    /// mipmap is half the dimensions of the previous level. Guaranteed to be always supported.
    ///
    /// Note that this is not necessarily the maximum number of mipmaps, as the Vulkan
    /// implementation may report that it supports a greater value.
    Log2,

    /// Allocate one mipmap (ie. just the main level). Always supported.
    One,

    /// Allocate the given number of mipmaps. May result in an error if the value is out of range
    /// of what the implementation supports.
    Specific(u32),
}

impl From<u32> for MipmapsCount {
    #[inline]
    fn from(num: u32) -> MipmapsCount {
        MipmapsCount::Specific(num)
    }
}

/// Specifies how the components of an image must be swizzled.
///
/// When creating an image view, it is possible to ask the implementation to modify the value
/// returned when accessing a given component from within a shader.
///
/// If all the members are `Identity`, then the view is said to have identity swizzling. This is
/// what the `Default` trait implementation of this struct returns.
/// Views that don't have identity swizzling may not be supported for some operations. For example
/// attaching a view to a framebuffer is only possible if the view is identity-swizzled.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Swizzle {
    /// First component.
    pub r: ComponentSwizzle,
    /// Second component.
    pub g: ComponentSwizzle,
    /// Third component.
    pub b: ComponentSwizzle,
    /// Fourth component.
    pub a: ComponentSwizzle,
}

/// Describes the value that an individual component must return when being accessed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComponentSwizzle {
    /// Returns the value that this component should normally have.
    Identity,
    /// Always return zero.
    Zero,
    /// Always return one.
    One,
    /// Returns the value of the first component.
    Red,
    /// Returns the value of the second component.
    Green,
    /// Returns the value of the third component.
    Blue,
    /// Returns the value of the fourth component.
    Alpha,
}

impl Default for ComponentSwizzle {
    #[inline]
    fn default() -> ComponentSwizzle {
        ComponentSwizzle::Identity
    }
}
