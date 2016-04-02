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
//! # High-level wrappers
//! 
//! The vulkano library provides high-level wrappers around images that are specialized depending
//! on the way you are going to use the image:
//! 
//! - An `AttachmentImage` can be used when you want to draw to an image.
//! - An `ImmutableImage` stores data which never need be changed after the initial upload,
//!   like a texture.
//! 
//! If are a beginner, you are strongly encouraged to use one of these wrappers.
//! 
//! # Low-level informations
//! 
//! To be written.
//!

pub use self::sys::ImageCreationError;
pub use self::sys::Layout;
pub use self::sys::Usage;
pub use self::traits::Image;
pub use self::traits::ImageView;

pub mod attachment;
pub mod immutable;
pub mod swapchain;
pub mod sys;
pub mod traits;

/// Specifies how many mipmaps must be allocated.
///
/// Note that at least one mipmap must be allocated, to store the main level of the image.
#[derive(Debug, Copy, Clone)]
pub enum MipmapsCount {
    /// Allocate the given number of mipmaps. May result in an error if the value is out of range.
    Specific(u32),

    /// Allocates the number of mipmaps required to store all the mipmaps of the image where each
    /// mipmap is half the dimensions of the previous level. Always supported.
    ///
    /// Note that this is not necessarily the maximum number of mipmaps, as the Vulkan
    /// implementation may report that it supports a greater value.
    Log2,

    /// Allocate one mipmap (ie. just the main level). Always supported.
    One,
}

impl From<u32> for MipmapsCount {
    #[inline]
    fn from(num: u32) -> MipmapsCount {
        MipmapsCount::Specific(num)
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Swizzle {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComponentSwizzle {
    Identity,
    Zero,
    One,
    Red,
    Green,
    Blue,
    Alpha,
}

impl Default for ComponentSwizzle {
    #[inline]
    fn default() -> ComponentSwizzle {
        ComponentSwizzle::Identity
    }
}
