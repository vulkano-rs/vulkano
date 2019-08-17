// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Declares all the formats of data and images supported by Vulkan.
//!
//! # Content of this module
//!
//! This module contains three things:
//!
//! - The `Format` enumeration, which contains all the available formats.
//! - The `FormatDesc` trait.
//! - One struct for each format.
//!
//! # Formats
//!
//! List of suffixes:
//!
//! - `Unorm` means that the values are unsigned integers that are converted into floating points.
//!   The maximum possible representable value becomes `1.0`, and the minimum representable value
//!   becomes `0.0`. For example the value `255` in a `R8Unorm` will be interpreted as `1.0`.
//!
//! - `Snorm` is the same as `Unorm`, but the integers are signed and the range is from `-1.0` to
//!   `1.0` instead.
//!
//! - `Uscaled` means that the values are unsigned integers that are converted into floating points.
//!   No change in the value is done. For example the value `255` in a `R8Uscaled` will be
//!   interpreted as `255.0`.
//!
//! - `Sscaled` is the same as `Uscaled` expect that the integers are signed.
//!
//! - `Uint` means that the values are unsigned integers. No conversion is performed.
//!
//! - `Sint` means that the values are signed integers. No conversion is performed.
//!
//! - `Ufloat` means that the values are unsigned floating points. No conversion is performed. This
//!   format is very unusual.
//!
//! - `Sfloat` means that the values are regular floating points. No conversion is performed.
//!
//! - `Srgb` is the same as `Unorm`, except that the value is interpreted as being in the sRGB
//!   color space. This means that its value will be converted to fit in the RGB color space when
//!   it is read. The fourth channel (usually used for alpha), if present, is not concerned by the
//!   conversion.
//!
//! # Choosing a format
//!
//! The following formats are guaranteed to be supported for everything that is related to
//! texturing (ie. blitting source and sampling them linearly). You should choose one of these
//! formats if you have an image that you are going to sample from:
//!
//! // TODO: use vulkano enums
//! - B4G4R4A4_UNORM_PACK16
//! - R5G6B5_UNORM_PACK16
//! - A1R5G5B5_UNORM_PACK16
//! - R8_UNORM
//! - R8_SNORM
//! - R8G8_UNORM
//! - R8G8_SNORM
//! - R8G8B8A8_UNORM
//! - R8G8B8A8_SNORM
//! - R8G8B8A8_SRGB
//! - B8G8R8A8_UNORM
//! - B8G8R8A8_SRGB
//! - A8B8G8R8_UNORM_PACK32
//! - A8B8G8R8_SNORM_PACK32
//! - A8B8G8R8_SRGB_PACK32
//! - A2B10G10R10_UNORM_PACK32
//! - R16_SFLOAT
//! - R16G16_SFLOAT
//! - R16G16B16A16_SFLOAT
//! - B10G11R11_UFLOAT_PACK32
//! - E5B9G9R9_UFLOAT_PACK32
//!
//! The following formats are guaranteed to be supported for everything that is related to
//! intermediate render targets (ie. blitting destination, color attachment and sampling linearly):
//!
//! // TODO: use vulkano enums
//! - R5G6B5_UNORM_PACK16
//! - A1R5G5B5_UNORM_PACK16
//! - R8_UNORM
//! - R8G8_UNORM
//! - R8G8B8A8_UNORM
//! - R8G8B8A8_SRGB
//! - B8G8R8A8_UNORM
//! - B8G8R8A8_SRGB
//! - A8B8G8R8_UNORM_PACK32
//! - A8B8G8R8_SRGB_PACK32
//! - A2B10G10R10_UNORM_PACK32
//! - R16_SFLOAT
//! - R16G16_SFLOAT
//! - R16G16B16A16_SFLOAT
//!
//! For depth images, only `D16Unorm` is guaranteed to be supported. For depth-stencil images,
//! it is guaranteed that either `D24Unorm_S8Uint` or `D32Sfloat_S8Uint` are supported.
//!
//! // TODO: storage formats
//!

use std::{error, fmt, mem};
use std::vec::IntoIter as VecIntoIter;

use half::f16;

use vk;

// TODO: add enumerations for color, depth, stencil and depthstencil formats

/// Some data whose type must be known by the library.
///
/// This trait is unsafe to implement because bad things will happen if `ty()` returns a wrong
/// value.
pub unsafe trait Data {
    /// Returns the type of the data from an enum.
    fn ty() -> Format;

    // TODO "is_supported" functions that redirect to `Self::ty().is_supported()`
}

// TODO: that's just an example ; implement for all common data types
unsafe impl Data for i8 {
    #[inline]
    fn ty() -> Format {
        Format::R8Sint
    }
}
unsafe impl Data for u8 {
    #[inline]
    fn ty() -> Format {
        Format::R8Uint
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IncompatiblePixelsType;

impl error::Error for IncompatiblePixelsType {
    #[inline]
    fn description(&self) -> &str {
        "supplied pixels' type is incompatible with this format"
    }
}

impl fmt::Display for IncompatiblePixelsType {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

pub unsafe trait AcceptsPixels<T> {
    /// Returns an error if `T` cannot be used as a source of pixels for `Self`.
    fn ensure_accepts(&self) -> Result<(), IncompatiblePixelsType>;

    /// The number of `T`s which make up a single pixel.
    ///
    /// ```
    /// use vulkano::format::{AcceptsPixels, R8G8B8A8Srgb};
    /// assert_eq!(<R8G8B8A8Srgb as AcceptsPixels<[u8; 4]>>::rate(&R8G8B8A8Srgb), 1);
    /// assert_eq!(<R8G8B8A8Srgb as AcceptsPixels<u8>>::rate(&R8G8B8A8Srgb), 4);
    /// ```
    ///
    /// # Panics
    ///
    /// May panic if `ensure_accepts` would not return `Ok(())`.
    fn rate(&self) -> u32 {
        1
    }
}

macro_rules! formats {
    ($($name:ident => $vk:ident [$bdim:expr] [$sz:expr] [$($f_ty:tt)*] {$($d_ty:tt)*},)+) => (
        /// An enumeration of all the possible formats.
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        #[repr(u32)]
        #[allow(missing_docs)]
        #[allow(non_camel_case_types)]
        pub enum Format {
            $($name = vk::$vk,)+
        }

        impl Format {
            /*pub fn is_supported_for_vertex_attributes(&self) -> bool {

            }

            .. other functions ..
            */

            /// Returns the size in bytes of an element of this format. For block based formats
            /// this will be the size of a single block. Returns `None` if the
            /// size is irrelevant.
            #[inline]
            pub fn size(&self) -> Option<usize> {
                match *self {
                    $(
                        Format::$name => $sz,
                    )+
                }
            }

            /// Returns (width, heigh) of the dimensions for block based formats. For
            /// non block formats will return (1,1)
            #[inline]
            pub fn block_dimensions(&self) -> (u32, u32) {
                match *self {
                    $(
                        Format::$name => $bdim,
                    )+
                }
            }

            /// Returns the `Format` corresponding to a Vulkan constant.
            pub(crate) fn from_vulkan_num(val: u32) -> Option<Format> {
                match val {
                    $(
                        vk::$vk => Some(Format::$name),
                    )+
                    _ => None,
                }
            }

            #[inline]
            pub fn ty(&self) -> FormatTy {
                match *self {
                    $(
                        Format::$name => formats!(__inner_ty__ $name $($f_ty)*),
                    )+
                }
            }
        }

        $(
            #[derive(Debug, Copy, Clone, Default)]
            #[allow(missing_docs)]
            #[allow(non_camel_case_types)]
            pub struct $name;

            formats!(__inner_impl__ $name $($f_ty)*);
            formats!(__inner_strongstorage__ $name $($d_ty)*);
        )+
    );

    (__inner_impl__ $name:ident float=$num:expr) => {
        unsafe impl FormatDesc for $name {
            type ClearValue = [f32; $num];

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }

            #[inline]
            fn decode_clear_value(&self, val: Self::ClearValue) -> ClearValue {
                val.into()
            }
        }
        unsafe impl PossibleFloatFormatDesc for $name {
            #[inline(always)]
            fn is_float(&self) -> bool { true }
        }
        unsafe impl PossibleFloatOrCompressedFormatDesc for $name {
            #[inline(always)]
            fn is_float_or_compressed(&self) -> bool { true }
        }
    };

    (__inner_impl__ $name:ident uint=$num:expr) => {
        unsafe impl FormatDesc for $name {
            type ClearValue = [u32; $num];

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }

            #[inline]
            fn decode_clear_value(&self, val: Self::ClearValue) -> ClearValue {
                val.into()
            }
        }

        unsafe impl PossibleUintFormatDesc for $name {
            #[inline(always)]
            fn is_uint(&self) -> bool { true }
        }
    };

    (__inner_impl__ $name:ident sint=$num:expr) => {
        unsafe impl FormatDesc for $name {
            type ClearValue = [i32; $num];

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }

            #[inline]
            fn decode_clear_value(&self, val: Self::ClearValue) -> ClearValue {
                val.into()
            }
        }

        unsafe impl PossibleSintFormatDesc for $name {
            #[inline(always)]
            fn is_sint(&self) -> bool { true }
        }
    };

    (__inner_impl__ $name:ident depth) => {
        unsafe impl FormatDesc for $name {
            type ClearValue = f32;

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }

            #[inline]
            fn decode_clear_value(&self, val: Self::ClearValue) -> ClearValue {
                val.into()
            }
        }

        unsafe impl PossibleDepthFormatDesc for $name {
            #[inline(always)]
            fn is_depth(&self) -> bool { true }
        }
    };

    (__inner_impl__ $name:ident stencil) => {
        unsafe impl FormatDesc for $name {
            type ClearValue = u32;      // FIXME: shouldn't stencil be i32?

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }

            #[inline]
            fn decode_clear_value(&self, val: Self::ClearValue) -> ClearValue {
                val.into()
            }
        }

        unsafe impl PossibleStencilFormatDesc for $name {
            #[inline(always)]
            fn is_stencil(&self) -> bool { true }
        }
    };

    (__inner_impl__ $name:ident depthstencil) => {
        unsafe impl FormatDesc for $name {
            type ClearValue = (f32, u32);       // FIXME: shouldn't stencil be i32?

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }

            #[inline]
            fn decode_clear_value(&self, val: Self::ClearValue) -> ClearValue {
                val.into()
            }
        }

        unsafe impl PossibleDepthStencilFormatDesc for $name {
            #[inline(always)]
            fn is_depth_stencil(&self) -> bool { true }
        }
    };

    (__inner_impl__ $name:ident compressed = $feature:ident) => {
        unsafe impl FormatDesc for $name {
            type ClearValue = [f32; 4];

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }

            #[inline]
            fn decode_clear_value(&self, val: Self::ClearValue) -> ClearValue {
                val.into()
            }
        }

        unsafe impl PossibleCompressedFormatDesc for $name {
            #[inline(always)]
            fn is_compressed(&self) -> bool { true }
        }
        unsafe impl PossibleFloatOrCompressedFormatDesc for $name {
            #[inline(always)]
            fn is_float_or_compressed(&self) -> bool { true }
        }
    };

    (__inner_ty__ $name:ident float=$num:tt) => { FormatTy::Float };
    (__inner_ty__ $name:ident uint=$num:tt) => { FormatTy::Uint };
    (__inner_ty__ $name:ident sint=$num:tt) => { FormatTy::Sint };
    (__inner_ty__ $name:ident depth) => { FormatTy::Depth };
    (__inner_ty__ $name:ident stencil) => { FormatTy::Stencil };
    (__inner_ty__ $name:ident depthstencil) => { FormatTy::DepthStencil };
    (__inner_ty__ $name:ident compressed=$f:tt) => { FormatTy::Compressed };


    (__inner_strongstorage__ $name:ident [$ty:ty; $dim:expr]) => {
        formats!(__inner_strongstorage_common__ $name [$ty; $dim]);
        unsafe impl AcceptsPixels<$ty> for $name {
            fn ensure_accepts(&self) -> Result<(), IncompatiblePixelsType> { Ok(()) }
            fn rate(&self) -> u32 { $dim }
        }
    };
    (__inner_strongstorage__ $name:ident $ty:ty) => {
        formats!(__inner_strongstorage_common__ $name $ty);
    };
    (__inner_strongstorage__ $name:ident ) => {};

    (__inner_strongstorage_common__ $name:ident $ty:ty) => {
        unsafe impl StrongStorage for $name {
            type Pixel = $ty;
        }
        unsafe impl AcceptsPixels<$ty> for $name {
            fn ensure_accepts(&self) -> Result<(), IncompatiblePixelsType> { Ok(()) }
        }
    };
}

formats! {
    R4G4UnormPack8 => FORMAT_R4G4_UNORM_PACK8 [(1, 1)] [Some(1)] [float=2] {u8},
    R4G4B4A4UnormPack16 => FORMAT_R4G4B4A4_UNORM_PACK16 [(1, 1)] [Some(2)] [float=4] {u16},
    B4G4R4A4UnormPack16 => FORMAT_B4G4R4A4_UNORM_PACK16 [(1, 1)] [Some(2)] [float=4] {u16},
    R5G6B5UnormPack16 => FORMAT_R5G6B5_UNORM_PACK16 [(1, 1)] [Some(2)] [float=3] {u16},
    B5G6R5UnormPack16 => FORMAT_B5G6R5_UNORM_PACK16 [(1, 1)] [Some(2)] [float=3] {u16},
    R5G5B5A1UnormPack16 => FORMAT_R5G5B5A1_UNORM_PACK16 [(1, 1)] [Some(2)] [float=4] {u16},
    B5G5R5A1UnormPack16 => FORMAT_B5G5R5A1_UNORM_PACK16 [(1, 1)] [Some(2)] [float=4] {u16},
    A1R5G5B5UnormPack16 => FORMAT_A1R5G5B5_UNORM_PACK16 [(1, 1)] [Some(2)] [float=4] {u16},
    R8Unorm => FORMAT_R8_UNORM [(1, 1)] [Some(1)] [float=1] {u8},
    R8Snorm => FORMAT_R8_SNORM [(1, 1)] [Some(1)] [float=1] {i8},
    R8Uscaled => FORMAT_R8_USCALED [(1, 1)] [Some(1)] [float=1] {u8},
    R8Sscaled => FORMAT_R8_SSCALED [(1, 1)] [Some(1)] [float=1] {i8},
    R8Uint => FORMAT_R8_UINT [(1, 1)] [Some(1)] [uint=1] {u8},
    R8Sint => FORMAT_R8_SINT [(1, 1)] [Some(1)] [sint=1] {i8},
    R8Srgb => FORMAT_R8_SRGB [(1, 1)] [Some(1)] [float=1] {u8},
    R8G8Unorm => FORMAT_R8G8_UNORM [(1, 1)] [Some(2)] [float=2] {[u8; 2]},
    R8G8Snorm => FORMAT_R8G8_SNORM [(1, 1)] [Some(2)] [float=2] {[i8; 2]},
    R8G8Uscaled => FORMAT_R8G8_USCALED [(1, 1)] [Some(2)] [float=2] {[u8; 2]},
    R8G8Sscaled => FORMAT_R8G8_SSCALED [(1, 1)] [Some(2)] [float=2] {[i8; 2]},
    R8G8Uint => FORMAT_R8G8_UINT [(1, 1)] [Some(2)] [uint=2] {[u8; 2]},
    R8G8Sint => FORMAT_R8G8_SINT [(1, 1)] [Some(2)] [sint=2] {[i8; 2]},
    R8G8Srgb => FORMAT_R8G8_SRGB [(1, 1)] [Some(2)] [float=2] {[u8; 2]},
    R8G8B8Unorm => FORMAT_R8G8B8_UNORM [(1, 1)] [Some(3)] [float=3] {[u8; 3]},
    R8G8B8Snorm => FORMAT_R8G8B8_SNORM [(1, 1)] [Some(3)] [float=3] {[i8; 3]},
    R8G8B8Uscaled => FORMAT_R8G8B8_USCALED [(1, 1)] [Some(3)] [float=3] {[u8; 3]},
    R8G8B8Sscaled => FORMAT_R8G8B8_SSCALED [(1, 1)] [Some(3)] [float=3] {[i8; 3]},
    R8G8B8Uint => FORMAT_R8G8B8_UINT [(1, 1)] [Some(3)] [uint=3] {[u8; 3]},
    R8G8B8Sint => FORMAT_R8G8B8_SINT [(1, 1)] [Some(3)] [sint=3] {[i8; 3]},
    R8G8B8Srgb => FORMAT_R8G8B8_SRGB [(1, 1)] [Some(3)] [float=3] {[u8; 3]},
    B8G8R8Unorm => FORMAT_B8G8R8_UNORM [(1, 1)] [Some(3)] [float=3] {[u8; 3]},
    B8G8R8Snorm => FORMAT_B8G8R8_SNORM [(1, 1)] [Some(3)] [float=3] {[i8; 3]},
    B8G8R8Uscaled => FORMAT_B8G8R8_USCALED [(1, 1)] [Some(3)] [float=3] {[u8; 3]},
    B8G8R8Sscaled => FORMAT_B8G8R8_SSCALED [(1, 1)] [Some(3)] [float=3] {[i8; 3]},
    B8G8R8Uint => FORMAT_B8G8R8_UINT [(1, 1)] [Some(3)] [uint=3] {[u8; 3]},
    B8G8R8Sint => FORMAT_B8G8R8_SINT [(1, 1)] [Some(3)] [sint=3] {[i8; 3]},
    B8G8R8Srgb => FORMAT_B8G8R8_SRGB [(1, 1)] [Some(3)] [float=3] {[u8; 3]},
    R8G8B8A8Unorm => FORMAT_R8G8B8A8_UNORM [(1, 1)] [Some(4)] [float=4] {[u8; 4]},
    R8G8B8A8Snorm => FORMAT_R8G8B8A8_SNORM [(1, 1)] [Some(4)] [float=4] {[i8; 4]},
    R8G8B8A8Uscaled => FORMAT_R8G8B8A8_USCALED [(1, 1)] [Some(4)] [float=4] {[u8; 4]},
    R8G8B8A8Sscaled => FORMAT_R8G8B8A8_SSCALED [(1, 1)] [Some(4)] [float=4] {[i8; 4]},
    R8G8B8A8Uint => FORMAT_R8G8B8A8_UINT [(1, 1)] [Some(4)] [uint=4] {[u8; 4]},
    R8G8B8A8Sint => FORMAT_R8G8B8A8_SINT [(1, 1)] [Some(4)] [sint=4] {[i8; 4]},
    R8G8B8A8Srgb => FORMAT_R8G8B8A8_SRGB [(1, 1)] [Some(4)] [float=4] {[u8; 4]},
    B8G8R8A8Unorm => FORMAT_B8G8R8A8_UNORM [(1, 1)] [Some(4)] [float=4] {[u8; 4]},
    B8G8R8A8Snorm => FORMAT_B8G8R8A8_SNORM [(1, 1)] [Some(4)] [float=4] {[i8; 4]},
    B8G8R8A8Uscaled => FORMAT_B8G8R8A8_USCALED [(1, 1)] [Some(4)] [float=4] {[u8; 4]},
    B8G8R8A8Sscaled => FORMAT_B8G8R8A8_SSCALED [(1, 1)] [Some(4)] [float=4] {[i8; 4]},
    B8G8R8A8Uint => FORMAT_B8G8R8A8_UINT [(1, 1)] [Some(4)] [uint=4] {[u8; 4]},
    B8G8R8A8Sint => FORMAT_B8G8R8A8_SINT [(1, 1)] [Some(4)] [sint=4] {[i8; 4]},
    B8G8R8A8Srgb => FORMAT_B8G8R8A8_SRGB [(1, 1)] [Some(4)] [float=4] {[u8; 4]},
    A8B8G8R8UnormPack32 => FORMAT_A8B8G8R8_UNORM_PACK32 [(1, 1)] [Some(4)] [float=4] {[u8; 4]},
    A8B8G8R8SnormPack32 => FORMAT_A8B8G8R8_SNORM_PACK32 [(1, 1)] [Some(4)] [float=4] {[i8; 4]},
    A8B8G8R8UscaledPack32 => FORMAT_A8B8G8R8_USCALED_PACK32 [(1, 1)] [Some(4)] [float=4] {[u8; 4]},
    A8B8G8R8SscaledPack32 => FORMAT_A8B8G8R8_SSCALED_PACK32 [(1, 1)] [Some(4)] [float=4] {[i8; 4]},
    A8B8G8R8UintPack32 => FORMAT_A8B8G8R8_UINT_PACK32 [(1, 1)] [Some(4)] [uint=4] {[u8; 4]},
    A8B8G8R8SintPack32 => FORMAT_A8B8G8R8_SINT_PACK32 [(1, 1)] [Some(4)] [sint=4] {[i8; 4]},
    A8B8G8R8SrgbPack32 => FORMAT_A8B8G8R8_SRGB_PACK32 [(1, 1)] [Some(4)] [float=4] {[u8; 4]},
    A2R10G10B10UnormPack32 => FORMAT_A2R10G10B10_UNORM_PACK32 [(1, 1)] [Some(4)] [float=4] {u32},
    A2R10G10B10SnormPack32 => FORMAT_A2R10G10B10_SNORM_PACK32 [(1, 1)] [Some(4)] [float=4] {u32},
    A2R10G10B10UscaledPack32 => FORMAT_A2R10G10B10_USCALED_PACK32 [(1, 1)] [Some(4)] [float=4] {u32},
    A2R10G10B10SscaledPack32 => FORMAT_A2R10G10B10_SSCALED_PACK32 [(1, 1)] [Some(4)] [float=4] {u32},
    A2R10G10B10UintPack32 => FORMAT_A2R10G10B10_UINT_PACK32 [(1, 1)] [Some(4)] [uint=4] {u32},
    A2R10G10B10SintPack32 => FORMAT_A2R10G10B10_SINT_PACK32 [(1, 1)] [Some(4)] [sint=4] {u32},
    A2B10G10R10UnormPack32 => FORMAT_A2B10G10R10_UNORM_PACK32 [(1, 1)] [Some(4)] [float=4] {u32},
    A2B10G10R10SnormPack32 => FORMAT_A2B10G10R10_SNORM_PACK32 [(1, 1)] [Some(4)] [float=4] {u32},
    A2B10G10R10UscaledPack32 => FORMAT_A2B10G10R10_USCALED_PACK32 [(1, 1)] [Some(4)] [float=4] {u32},
    A2B10G10R10SscaledPack32 => FORMAT_A2B10G10R10_SSCALED_PACK32 [(1, 1)] [Some(4)] [float=4] {u32},
    A2B10G10R10UintPack32 => FORMAT_A2B10G10R10_UINT_PACK32 [(1, 1)] [Some(4)] [uint=4] {u32},
    A2B10G10R10SintPack32 => FORMAT_A2B10G10R10_SINT_PACK32 [(1, 1)] [Some(4)] [sint=4] {u32},
    R16Unorm => FORMAT_R16_UNORM [(1, 1)] [Some(2)] [float=1] {u16},
    R16Snorm => FORMAT_R16_SNORM [(1, 1)] [Some(2)] [float=1] {i16},
    R16Uscaled => FORMAT_R16_USCALED [(1, 1)] [Some(2)] [float=1] {u16},
    R16Sscaled => FORMAT_R16_SSCALED [(1, 1)] [Some(2)] [float=1] {i16},
    R16Uint => FORMAT_R16_UINT [(1, 1)] [Some(2)] [uint=1] {u16},
    R16Sint => FORMAT_R16_SINT [(1, 1)] [Some(2)] [sint=1] {i16},
    R16Sfloat => FORMAT_R16_SFLOAT [(1, 1)] [Some(2)] [float=1] {f16},
    R16G16Unorm => FORMAT_R16G16_UNORM [(1, 1)] [Some(4)] [float=2] {[u16; 2]},
    R16G16Snorm => FORMAT_R16G16_SNORM [(1, 1)] [Some(4)] [float=2] {[i16; 2]},
    R16G16Uscaled => FORMAT_R16G16_USCALED [(1, 1)] [Some(4)] [float=2] {[u16; 2]},
    R16G16Sscaled => FORMAT_R16G16_SSCALED [(1, 1)] [Some(4)] [float=2] {[i16; 2]},
    R16G16Uint => FORMAT_R16G16_UINT [(1, 1)] [Some(4)] [uint=2] {[u16; 2]},
    R16G16Sint => FORMAT_R16G16_SINT [(1, 1)] [Some(4)] [sint=2] {[i16; 2]},
    R16G16Sfloat => FORMAT_R16G16_SFLOAT [(1, 1)] [Some(4)] [float=2] {[f16; 2]},
    R16G16B16Unorm => FORMAT_R16G16B16_UNORM [(1, 1)] [Some(6)] [float=3] {[u16; 3]},
    R16G16B16Snorm => FORMAT_R16G16B16_SNORM [(1, 1)] [Some(6)] [float=3] {[i16; 3]},
    R16G16B16Uscaled => FORMAT_R16G16B16_USCALED [(1, 1)] [Some(6)] [float=3] {[u16; 3]},
    R16G16B16Sscaled => FORMAT_R16G16B16_SSCALED [(1, 1)] [Some(6)] [float=3] {[i16; 3]},
    R16G16B16Uint => FORMAT_R16G16B16_UINT [(1, 1)] [Some(6)] [uint=3] {[u16; 3]},
    R16G16B16Sint => FORMAT_R16G16B16_SINT [(1, 1)] [Some(6)] [sint=3] {[i16; 3]},
    R16G16B16Sfloat => FORMAT_R16G16B16_SFLOAT [(1, 1)] [Some(6)] [float=3] {[f16; 3]},
    R16G16B16A16Unorm => FORMAT_R16G16B16A16_UNORM [(1, 1)] [Some(8)] [float=4] {[u16; 4]},
    R16G16B16A16Snorm => FORMAT_R16G16B16A16_SNORM [(1, 1)] [Some(8)] [float=4] {[i16; 4]},
    R16G16B16A16Uscaled => FORMAT_R16G16B16A16_USCALED [(1, 1)] [Some(8)] [float=4] {[u16; 4]},
    R16G16B16A16Sscaled => FORMAT_R16G16B16A16_SSCALED [(1, 1)] [Some(8)] [float=4] {[i16; 4]},
    R16G16B16A16Uint => FORMAT_R16G16B16A16_UINT [(1, 1)] [Some(8)] [uint=4] {[u16; 4]},
    R16G16B16A16Sint => FORMAT_R16G16B16A16_SINT [(1, 1)] [Some(8)] [sint=4] {[i16; 4]},
    R16G16B16A16Sfloat => FORMAT_R16G16B16A16_SFLOAT [(1, 1)] [Some(8)] [float=4] {[f16; 4]},
    R32Uint => FORMAT_R32_UINT [(1, 1)] [Some(4)] [uint=1] {u32},
    R32Sint => FORMAT_R32_SINT [(1, 1)] [Some(4)] [sint=1] {i32},
    R32Sfloat => FORMAT_R32_SFLOAT [(1, 1)] [Some(4)] [float=1] {f32},
    R32G32Uint => FORMAT_R32G32_UINT [(1, 1)] [Some(8)] [uint=2] {[u32; 2]},
    R32G32Sint => FORMAT_R32G32_SINT [(1, 1)] [Some(8)] [sint=2] {[i32; 2]},
    R32G32Sfloat => FORMAT_R32G32_SFLOAT [(1, 1)] [Some(8)] [float=2] {[f32; 2]},
    R32G32B32Uint => FORMAT_R32G32B32_UINT [(1, 1)] [Some(12)] [uint=3] {[u32; 3]},
    R32G32B32Sint => FORMAT_R32G32B32_SINT [(1, 1)] [Some(12)] [sint=3] {[i32; 3]},
    R32G32B32Sfloat => FORMAT_R32G32B32_SFLOAT [(1, 1)] [Some(12)] [float=3] {[f32; 3]},
    R32G32B32A32Uint => FORMAT_R32G32B32A32_UINT [(1, 1)] [Some(16)] [uint=4] {[u32; 4]},
    R32G32B32A32Sint => FORMAT_R32G32B32A32_SINT [(1, 1)] [Some(16)] [sint=4] {[i32; 4]},
    R32G32B32A32Sfloat => FORMAT_R32G32B32A32_SFLOAT [(1, 1)] [Some(16)] [float=4] {[f32; 4]},
    R64Uint => FORMAT_R64_UINT [(1, 1)] [Some(8)] [uint=1] {u64},
    R64Sint => FORMAT_R64_SINT [(1, 1)] [Some(8)] [sint=1] {i64},
    R64Sfloat => FORMAT_R64_SFLOAT [(1, 1)] [Some(8)] [float=1] {f64},
    R64G64Uint => FORMAT_R64G64_UINT [(1, 1)] [Some(16)] [uint=2] {[u64; 2]},
    R64G64Sint => FORMAT_R64G64_SINT [(1, 1)] [Some(16)] [sint=2] {[i64; 2]},
    R64G64Sfloat => FORMAT_R64G64_SFLOAT [(1, 1)] [Some(16)] [float=2] {[f64; 2]},
    R64G64B64Uint => FORMAT_R64G64B64_UINT [(1, 1)] [Some(24)] [uint=3] {[u64; 3]},
    R64G64B64Sint => FORMAT_R64G64B64_SINT [(1, 1)] [Some(24)] [sint=3] {[i64; 3]},
    R64G64B64Sfloat => FORMAT_R64G64B64_SFLOAT [(1, 1)] [Some(24)] [float=3] {[f64; 3]},
    R64G64B64A64Uint => FORMAT_R64G64B64A64_UINT [(1, 1)] [Some(32)] [uint=4] {[u64; 4]},
    R64G64B64A64Sint => FORMAT_R64G64B64A64_SINT [(1, 1)] [Some(32)] [sint=4] {[i64; 4]},
    R64G64B64A64Sfloat => FORMAT_R64G64B64A64_SFLOAT [(1, 1)] [Some(32)] [float=4] {[f64; 4]},
    B10G11R11UfloatPack32 => FORMAT_B10G11R11_UFLOAT_PACK32 [(1, 1)] [Some(4)] [float=3] {u32},
    E5B9G9R9UfloatPack32 => FORMAT_E5B9G9R9_UFLOAT_PACK32 [(1, 1)] [Some(4)] [float=3] {u32},
    D16Unorm => FORMAT_D16_UNORM [(1, 1)] [Some(2)] [depth] {},
    X8_D24UnormPack32 => FORMAT_X8_D24_UNORM_PACK32 [(1, 1)] [Some(4)] [depth] {},
    D32Sfloat => FORMAT_D32_SFLOAT [(1, 1)] [Some(4)] [depth] {},
    S8Uint => FORMAT_S8_UINT [(1, 1)] [Some(1)] [stencil] {},
    D16Unorm_S8Uint => FORMAT_D16_UNORM_S8_UINT [(1, 1)] [None] [depthstencil] {},
    D24Unorm_S8Uint => FORMAT_D24_UNORM_S8_UINT [(1, 1)] [None] [depthstencil] {},
    D32Sfloat_S8Uint => FORMAT_D32_SFLOAT_S8_UINT [(1, 1)] [None] [depthstencil] {},
    BC1_RGBUnormBlock => FORMAT_BC1_RGB_UNORM_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_bc] {u8},
    BC1_RGBSrgbBlock => FORMAT_BC1_RGB_SRGB_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_bc] {u8},
    BC1_RGBAUnormBlock => FORMAT_BC1_RGBA_UNORM_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_bc] {u8},
    BC1_RGBASrgbBlock => FORMAT_BC1_RGBA_SRGB_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_bc] {u8},
    BC2UnormBlock => FORMAT_BC2_UNORM_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_bc] {u8},
    BC2SrgbBlock => FORMAT_BC2_SRGB_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_bc] {u8},
    BC3UnormBlock => FORMAT_BC3_UNORM_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_bc] {u8},
    BC3SrgbBlock => FORMAT_BC3_SRGB_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_bc] {u8},
    BC4UnormBlock => FORMAT_BC4_UNORM_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_bc] {u8},
    BC4SnormBlock => FORMAT_BC4_SNORM_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_bc] {u8},
    BC5UnormBlock => FORMAT_BC5_UNORM_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_bc] {u8},
    BC5SnormBlock => FORMAT_BC5_SNORM_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_bc] {u8},
    BC6HUfloatBlock => FORMAT_BC6H_UFLOAT_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_bc] {u8},
    BC6HSfloatBlock => FORMAT_BC6H_SFLOAT_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_bc] {u8},
    BC7UnormBlock => FORMAT_BC7_UNORM_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_bc] {u8},
    BC7SrgbBlock => FORMAT_BC7_SRGB_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_bc] {u8},
    ETC2_R8G8B8UnormBlock => FORMAT_ETC2_R8G8B8_UNORM_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_etc2] {u8},
    ETC2_R8G8B8SrgbBlock => FORMAT_ETC2_R8G8B8_SRGB_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_etc2] {u8},
    ETC2_R8G8B8A1UnormBlock => FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_etc2] {u8},
    ETC2_R8G8B8A1SrgbBlock => FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_etc2] {u8},
    ETC2_R8G8B8A8UnormBlock => FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_etc2] {u8},
    ETC2_R8G8B8A8SrgbBlock => FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_etc2] {u8},
    EAC_R11UnormBlock => FORMAT_EAC_R11_UNORM_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_etc2] {u8},
    EAC_R11SnormBlock => FORMAT_EAC_R11_SNORM_BLOCK [(4, 4)] [Some(8)] [compressed=texture_compression_etc2] {u8},
    EAC_R11G11UnormBlock => FORMAT_EAC_R11G11_UNORM_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_etc2] {u8},
    EAC_R11G11SnormBlock => FORMAT_EAC_R11G11_SNORM_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_etc2] {u8},
    ASTC_4x4UnormBlock => FORMAT_ASTC_4x4_UNORM_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_4x4SrgbBlock => FORMAT_ASTC_4x4_SRGB_BLOCK [(4, 4)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_5x4UnormBlock => FORMAT_ASTC_5x4_UNORM_BLOCK [(5, 4)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_5x4SrgbBlock => FORMAT_ASTC_5x4_SRGB_BLOCK [(5, 4)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_5x5UnormBlock => FORMAT_ASTC_5x5_UNORM_BLOCK [(5, 5)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_5x5SrgbBlock => FORMAT_ASTC_5x5_SRGB_BLOCK [(5, 5)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_6x5UnormBlock => FORMAT_ASTC_6x5_UNORM_BLOCK [(6, 5)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_6x5SrgbBlock => FORMAT_ASTC_6x5_SRGB_BLOCK [(6, 5)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_6x6UnormBlock => FORMAT_ASTC_6x6_UNORM_BLOCK [(6, 6)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_6x6SrgbBlock => FORMAT_ASTC_6x6_SRGB_BLOCK [(6, 6)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_8x5UnormBlock => FORMAT_ASTC_8x5_UNORM_BLOCK [(8, 5)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_8x5SrgbBlock => FORMAT_ASTC_8x5_SRGB_BLOCK [(8, 5)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_8x6UnormBlock => FORMAT_ASTC_8x6_UNORM_BLOCK [(8, 6)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_8x6SrgbBlock => FORMAT_ASTC_8x6_SRGB_BLOCK [(8, 6)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_8x8UnormBlock => FORMAT_ASTC_8x8_UNORM_BLOCK [(8, 8)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_8x8SrgbBlock => FORMAT_ASTC_8x8_SRGB_BLOCK [(8, 8)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_10x5UnormBlock => FORMAT_ASTC_10x5_UNORM_BLOCK [(10, 5)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_10x5SrgbBlock => FORMAT_ASTC_10x5_SRGB_BLOCK [(10, 5)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_10x6UnormBlock => FORMAT_ASTC_10x6_UNORM_BLOCK [(10, 6)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_10x6SrgbBlock => FORMAT_ASTC_10x6_SRGB_BLOCK [(10, 6)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_10x8UnormBlock => FORMAT_ASTC_10x8_UNORM_BLOCK [(10, 8)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_10x8SrgbBlock => FORMAT_ASTC_10x8_SRGB_BLOCK [(10, 8)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_10x10UnormBlock => FORMAT_ASTC_10x10_UNORM_BLOCK [(10, 10)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_10x10SrgbBlock => FORMAT_ASTC_10x10_SRGB_BLOCK [(10, 10)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_12x10UnormBlock => FORMAT_ASTC_12x10_UNORM_BLOCK [(12, 10)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_12x10SrgbBlock => FORMAT_ASTC_12x10_SRGB_BLOCK [(12, 10)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_12x12UnormBlock => FORMAT_ASTC_12x12_UNORM_BLOCK [(12, 12)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
    ASTC_12x12SrgbBlock => FORMAT_ASTC_12x12_SRGB_BLOCK [(12, 12)] [Some(16)] [compressed=texture_compression_astc_ldr] {u8},
}

pub unsafe trait FormatDesc {
    type ClearValue;

    fn format(&self) -> Format;

    fn decode_clear_value(&self, Self::ClearValue) -> ClearValue;
}

unsafe impl FormatDesc for Format {
    type ClearValue = ClearValue;

    #[inline]
    fn format(&self) -> Format {
        *self
    }

    fn decode_clear_value(&self, value: Self::ClearValue) -> ClearValue {
        match (self.ty(), value) {
            (FormatTy::Float, f @ ClearValue::Float(_)) => f,
            (FormatTy::Compressed, f @ ClearValue::Float(_)) => f,
            (FormatTy::Sint, f @ ClearValue::Int(_)) => f,
            (FormatTy::Uint, f @ ClearValue::Uint(_)) => f,
            (FormatTy::Depth, f @ ClearValue::Depth(_)) => f,
            (FormatTy::Stencil, f @ ClearValue::Stencil(_)) => f,
            (FormatTy::DepthStencil, f @ ClearValue::DepthStencil(_)) => f,
            _ => panic!("Wrong clear value"),
        }
    }
}

/// Trait for types that can possibly describe a float attachment.
pub unsafe trait PossibleFloatFormatDesc: FormatDesc {
    /// Returns true if the format is a float format.
    fn is_float(&self) -> bool;
}

unsafe impl PossibleFloatFormatDesc for Format {
    #[inline]
    fn is_float(&self) -> bool {
        self.ty() == FormatTy::Float
    }
}

pub unsafe trait PossibleUintFormatDesc: FormatDesc {
    fn is_uint(&self) -> bool;
}

unsafe impl PossibleUintFormatDesc for Format {
    #[inline]
    fn is_uint(&self) -> bool {
        self.ty() == FormatTy::Uint
    }
}

pub unsafe trait PossibleSintFormatDesc: FormatDesc {
    fn is_sint(&self) -> bool;
}

unsafe impl PossibleSintFormatDesc for Format {
    #[inline]
    fn is_sint(&self) -> bool {
        self.ty() == FormatTy::Sint
    }
}

pub unsafe trait PossibleDepthFormatDesc: FormatDesc {
    fn is_depth(&self) -> bool;
}

unsafe impl PossibleDepthFormatDesc for Format {
    #[inline]
    fn is_depth(&self) -> bool {
        self.ty() == FormatTy::Depth
    }
}

pub unsafe trait PossibleStencilFormatDesc: FormatDesc {
    fn is_stencil(&self) -> bool;
}

unsafe impl PossibleStencilFormatDesc for Format {
    #[inline]
    fn is_stencil(&self) -> bool {
        self.ty() == FormatTy::Stencil
    }
}

pub unsafe trait PossibleDepthStencilFormatDesc: FormatDesc {
    fn is_depth_stencil(&self) -> bool;
}

unsafe impl PossibleDepthStencilFormatDesc for Format {
    #[inline]
    fn is_depth_stencil(&self) -> bool {
        self.ty() == FormatTy::DepthStencil
    }
}

pub unsafe trait PossibleCompressedFormatDesc: FormatDesc {
    fn is_compressed(&self) -> bool;
}

unsafe impl PossibleCompressedFormatDesc for Format {
    #[inline]
    fn is_compressed(&self) -> bool {
        self.ty() == FormatTy::Compressed
    }
}

/// Trait for types that can possibly describe a float or compressed attachment.
pub unsafe trait PossibleFloatOrCompressedFormatDesc: FormatDesc {
    /// Returns true if the format is a float or compressed format.
    fn is_float_or_compressed(&self) -> bool;
}

unsafe impl PossibleFloatOrCompressedFormatDesc for Format {
    #[inline]
    fn is_float_or_compressed(&self) -> bool {
        self.ty() == FormatTy::Float || self.ty() == FormatTy::Compressed
    }
}

macro_rules! impl_pixel {
    {$($ty:ty;)+} => {
        $(impl_pixel!(inner $ty);)*
        $(impl_pixel!(inner [$ty; 1]);)*
        $(impl_pixel!(inner [$ty; 2]);)*
        $(impl_pixel!(inner [$ty; 3]);)*
        $(impl_pixel!(inner [$ty; 4]);)*
        $(impl_pixel!(inner ($ty,));)*
        $(impl_pixel!(inner ($ty, $ty));)*
        $(impl_pixel!(inner ($ty, $ty, $ty));)*
        $(impl_pixel!(inner ($ty, $ty, $ty, $ty));)*
    };
    (inner $ty:ty) => {
        unsafe impl AcceptsPixels<$ty> for Format {
            fn ensure_accepts(&self) -> Result<(), IncompatiblePixelsType> {
                // TODO: Be more strict: accept only if the format has a matching AcceptsPixels impl.
                if self.size().map_or(false, |x| x % mem::size_of::<$ty>() == 0) {
                    Ok(())
                } else {
                    Err(IncompatiblePixelsType)
                }
            }
            fn rate(&self) -> u32 {
                (self.size().expect("this format cannot accept pixels") / mem::size_of::<$ty>()) as u32
            }
        }
    }
}

impl_pixel! {
    u8; i8; u16; i16; u32; i32; u64; i64; f16; f32; f64;
}

pub unsafe trait StrongStorage: FormatDesc {
    type Pixel: Copy;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FormatTy {
    Float,
    Uint,
    Sint,
    Depth,
    Stencil,
    DepthStencil,
    Compressed,
}

impl FormatTy {
    /// Returns true if `Depth`, `Stencil`, `DepthStencil`. False otherwise.
    #[inline]
    pub fn is_depth_and_or_stencil(&self) -> bool {
        match *self {
            FormatTy::Depth => true,
            FormatTy::Stencil => true,
            FormatTy::DepthStencil => true,
            _ => false,
        }
    }
}

/// Describes a uniform value that will be used to fill an image.
// TODO: should have the same layout as `vk::ClearValue` for performance
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ClearValue {
    /// Entry for attachments that aren't cleared.
    None,
    /// Value for floating-point attachments, including `Unorm`, `Snorm`, `Sfloat`.
    Float([f32; 4]),
    /// Value for integer attachments, including `Int`.
    Int([i32; 4]),
    /// Value for unsigned integer attachments, including `Uint`.
    Uint([u32; 4]),
    /// Value for depth attachments.
    Depth(f32),
    /// Value for stencil attachments.
    Stencil(u32),
    /// Value for depth and stencil attachments.
    DepthStencil((f32, u32)),
}

// TODO: remove all these From implementations once they are no longer needed

impl From<[f32; 1]> for ClearValue {
    #[inline]
    fn from(val: [f32; 1]) -> ClearValue {
        ClearValue::Float([val[0], 0.0, 0.0, 1.0])
    }
}

impl From<[f32; 2]> for ClearValue {
    #[inline]
    fn from(val: [f32; 2]) -> ClearValue {
        ClearValue::Float([val[0], val[1], 0.0, 1.0])
    }
}

impl From<[f32; 3]> for ClearValue {
    #[inline]
    fn from(val: [f32; 3]) -> ClearValue {
        ClearValue::Float([val[0], val[1], val[2], 1.0])
    }
}

impl From<[f32; 4]> for ClearValue {
    #[inline]
    fn from(val: [f32; 4]) -> ClearValue {
        ClearValue::Float(val)
    }
}

impl From<[u32; 1]> for ClearValue {
    #[inline]
    fn from(val: [u32; 1]) -> ClearValue {
        ClearValue::Uint([val[0], 0, 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 2]> for ClearValue {
    #[inline]
    fn from(val: [u32; 2]) -> ClearValue {
        ClearValue::Uint([val[0], val[1], 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 3]> for ClearValue {
    #[inline]
    fn from(val: [u32; 3]) -> ClearValue {
        ClearValue::Uint([val[0], val[1], val[2], 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 4]> for ClearValue {
    #[inline]
    fn from(val: [u32; 4]) -> ClearValue {
        ClearValue::Uint(val)
    }
}

impl From<[i32; 1]> for ClearValue {
    #[inline]
    fn from(val: [i32; 1]) -> ClearValue {
        ClearValue::Int([val[0], 0, 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 2]> for ClearValue {
    #[inline]
    fn from(val: [i32; 2]) -> ClearValue {
        ClearValue::Int([val[0], val[1], 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 3]> for ClearValue {
    #[inline]
    fn from(val: [i32; 3]) -> ClearValue {
        ClearValue::Int([val[0], val[1], val[2], 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 4]> for ClearValue {
    #[inline]
    fn from(val: [i32; 4]) -> ClearValue {
        ClearValue::Int(val)
    }
}

impl From<f32> for ClearValue {
    #[inline]
    fn from(val: f32) -> ClearValue {
        ClearValue::Depth(val)
    }
}

impl From<u32> for ClearValue {
    #[inline]
    fn from(val: u32) -> ClearValue {
        ClearValue::Stencil(val)
    }
}

impl From<(f32, u32)> for ClearValue {
    #[inline]
    fn from(val: (f32, u32)) -> ClearValue {
        ClearValue::DepthStencil(val)
    }
}


// TODO: remove once no longer needed
pub unsafe trait ClearValuesTuple {
    type Iter: Iterator<Item = ClearValue>;
    fn iter(self) -> Self::Iter;
}

macro_rules! impl_clear_values_tuple {
    ($first:ident $($others:ident)+) => (
        #[allow(non_snake_case)]
        unsafe impl<$first $(, $others)*> ClearValuesTuple for ($first, $($others,)+)
            where $first: Into<ClearValue> $(, $others: Into<ClearValue>)*
        {
            type Iter = VecIntoIter<ClearValue>;
            #[inline]
            fn iter(self) -> VecIntoIter<ClearValue> {
                let ($first, $($others,)+) = self;
                vec![
                    $first.into() $(, $others.into())+
                ].into_iter()
            }
        }

        impl_clear_values_tuple!($($others)*);
    );

    ($first:ident) => (
        unsafe impl<$first> ClearValuesTuple for ($first,)
            where $first: Into<ClearValue>
        {
            type Iter = VecIntoIter<ClearValue>;
            #[inline]
            fn iter(self) -> VecIntoIter<ClearValue> {
                vec![self.0.into()].into_iter()
            }
        }
    );
}

impl_clear_values_tuple!(A B C D E F G H I J K L M N O P Q R S T U V W X Y Z);
