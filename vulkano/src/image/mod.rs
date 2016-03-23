//! Images storage (1D, 2D, 3D, arrays, etc.).
//! 
//! # Strong typing
//! 
//! Images in vulkano are strong-typed. Their signature is `Image<Ty, F, M>`.
//! 
//! The `Ty` parameter describes the type of image: 1D, 2D, 3D, 1D array, 2D array. All these come
//! in two variants: with or without multisampling. The actual type of `Ty` must be one of the
//! marker structs of this module that start with the `Ty` prefix.
//! 
//! The `F` parameter describes the format of each pixel of the image. It must be one of the marker
//! structs of the `formats` module.
//! 
//! The `M` parameter describes where the image's memory was allocated from. It is similar to
//! buffers.
//!

pub use self::sys::Layout;
pub use self::sys::Usage;
pub use self::traits::Image;
pub use self::traits::ImageView;

pub mod attachment;
pub mod swapchain;
pub mod sys;
pub mod traits;

/// Specifies how many mipmaps must be allocated.
///
/// Note that at least one mipmap must be allocated, to store the main level of the image.
#[derive(Debug, Copy, Clone)]
pub enum MipmapsCount {
    /// Allocate the given number of mipmaps.
    Specific(u32),

    /// Allocates the number of mipmaps required to store all the mipmaps of the image where each
    /// mipmap is half the dimensions of the previous level.
    Max,

    /// Allocate one mipmap (ie. just the main level).
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
