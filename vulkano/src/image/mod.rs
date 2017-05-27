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
pub use self::layout::ImageLayout;
pub use self::storage::StorageImage;
pub use self::swapchain::SwapchainImage;
pub use self::sys::ImageCreationError;
pub use self::traits::ImageAccess;
pub use self::traits::ImageViewAccess;
pub use self::traits::Image;
pub use self::traits::ImageView;
pub use self::usage::ImageUsage;

pub mod attachment;     // TODO: make private
pub mod immutable;      // TODO: make private
mod layout;
mod storage;
pub mod swapchain;      // TODO: make private
pub mod sys;
pub mod traits;
mod usage;

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dimensions {
    Dim1d { width: u32 },
    Dim1dArray { width: u32, array_layers: u32 },
    Dim2d { width: u32, height: u32 },
    Dim2dArray { width: u32, height: u32, array_layers: u32 },
    Dim3d { width: u32, height: u32, depth: u32 },
    Cubemap { size: u32 },
    CubemapArray { size: u32, array_layers: u32 },
}

impl Dimensions {
    #[inline]
    pub fn width(&self) -> u32 {
        match *self {
            Dimensions::Dim1d { width } => width,
            Dimensions::Dim1dArray { width, .. } => width,
            Dimensions::Dim2d { width, .. } => width,
            Dimensions::Dim2dArray { width, .. } => width,
            Dimensions::Dim3d { width, .. } => width,
            Dimensions::Cubemap { size } => size,
            Dimensions::CubemapArray { size, .. } => size,
        }
    }

    #[inline]
    pub fn height(&self) -> u32 {
        match *self {
            Dimensions::Dim1d { .. } => 1,
            Dimensions::Dim1dArray { .. } => 1,
            Dimensions::Dim2d { height, .. } => height,
            Dimensions::Dim2dArray { height, .. } => height,
            Dimensions::Dim3d { height, .. }  => height,
            Dimensions::Cubemap { size } => size,
            Dimensions::CubemapArray { size, .. } => size,
        }
    }

    #[inline]
    pub fn width_height(&self) -> [u32; 2] {
        [self.width(), self.height()]
    }

    #[inline]
    pub fn depth(&self) -> u32 {
        match *self {
            Dimensions::Dim1d { .. } => 1,
            Dimensions::Dim1dArray { .. } => 1,
            Dimensions::Dim2d { .. } => 1,
            Dimensions::Dim2dArray { .. } => 1,
            Dimensions::Dim3d { depth, .. }  => depth,
            Dimensions::Cubemap { .. } => 1,
            Dimensions::CubemapArray { .. } => 1,
        }
    }

    #[inline]
    pub fn width_height_depth(&self) -> [u32; 3] {
        [self.width(), self.height(), self.depth()]
    }

    #[inline]
    pub fn array_layers(&self) -> u32 {
        match *self {
            Dimensions::Dim1d { .. } => 1,
            Dimensions::Dim1dArray { array_layers, .. } => array_layers,
            Dimensions::Dim2d { .. } => 1,
            Dimensions::Dim2dArray { array_layers, .. } => array_layers,
            Dimensions::Dim3d { .. }  => 1,
            Dimensions::Cubemap { .. } => 1,
            Dimensions::CubemapArray { array_layers, .. } => array_layers,
        }
    }

    #[inline]
    pub fn array_layers_with_cube(&self) -> u32 {
        match *self {
            Dimensions::Dim1d { .. } => 1,
            Dimensions::Dim1dArray { array_layers, .. } => array_layers,
            Dimensions::Dim2d { .. } => 1,
            Dimensions::Dim2dArray { array_layers, .. } => array_layers,
            Dimensions::Dim3d { .. }  => 1,
            Dimensions::Cubemap { .. } => 6,
            Dimensions::CubemapArray { array_layers, .. } => array_layers * 6,
        }
    }

    /// Builds the corresponding `ImageDimensions`.
    #[inline]
    pub fn to_image_dimensions(&self) -> ImageDimensions {
        match *self {
            Dimensions::Dim1d { width } => {
                ImageDimensions::Dim1d { width: width, array_layers: 1 }
            },
            Dimensions::Dim1dArray { width, array_layers } => {
                ImageDimensions::Dim1d { width: width, array_layers: array_layers }
            },
            Dimensions::Dim2d { width, height } => {
                ImageDimensions::Dim2d { width: width, height: height, array_layers: 1,
                                         cubemap_compatible: false }
            },
            Dimensions::Dim2dArray { width, height, array_layers } => {
                ImageDimensions::Dim2d { width: width, height: height,
                                         array_layers: array_layers, cubemap_compatible: false }
            },
            Dimensions::Dim3d { width, height, depth } => {
                ImageDimensions::Dim3d { width: width, height: height, depth: depth }
            },
            Dimensions::Cubemap { size } => {
                ImageDimensions::Dim2d { width: size, height: size, array_layers: 6,
                                         cubemap_compatible: true }
            },
            Dimensions::CubemapArray { size, array_layers } => {
                ImageDimensions::Dim2d { width: size, height: size, array_layers: array_layers * 6,
                                         cubemap_compatible: true }
            },
        }
    }

    /// Builds the corresponding `ViewType`.
    #[inline]
    pub fn to_view_type(&self) -> ViewType {
        match *self {
            Dimensions::Dim1d { .. } => ViewType::Dim1d, 
            Dimensions::Dim1dArray { .. } => ViewType::Dim1dArray, 
            Dimensions::Dim2d { .. } => ViewType::Dim2d, 
            Dimensions::Dim2dArray { .. } => ViewType::Dim2dArray, 
            Dimensions::Dim3d { .. } => ViewType::Dim3d, 
            Dimensions::Cubemap { .. } => ViewType::Cubemap, 
            Dimensions::CubemapArray { .. } => ViewType::CubemapArray, 
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ViewType {
    Dim1d,
    Dim1dArray,
    Dim2d,
    Dim2dArray,
    Dim3d,
    Cubemap,
    CubemapArray,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageDimensions {
    Dim1d { width: u32, array_layers: u32 },
    Dim2d { width: u32, height: u32, array_layers: u32, cubemap_compatible: bool },
    Dim3d { width: u32, height: u32, depth: u32 }
}

impl ImageDimensions {
    #[inline]
    pub fn width(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { width, .. } => width,
            ImageDimensions::Dim2d { width, .. } => width,
            ImageDimensions::Dim3d { width, .. }  => width,
        }
    }

    #[inline]
    pub fn height(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { .. } => 1,
            ImageDimensions::Dim2d { height, .. } => height,
            ImageDimensions::Dim3d { height, .. }  => height,
        }
    }

    #[inline]
    pub fn width_height(&self) -> [u32; 2] {
        [self.width(), self.height()]
    }

    #[inline]
    pub fn depth(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { .. } => 1,
            ImageDimensions::Dim2d { .. } => 1,
            ImageDimensions::Dim3d { depth, .. }  => depth,
        }
    }

    #[inline]
    pub fn width_height_depth(&self) -> [u32; 3] {
        [self.width(), self.height(), self.depth()]
    }

    #[inline]
    pub fn array_layers(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { array_layers, .. } => array_layers,
            ImageDimensions::Dim2d { array_layers, .. } => array_layers,
            ImageDimensions::Dim3d { .. }  => 1,
        }
    }
}
