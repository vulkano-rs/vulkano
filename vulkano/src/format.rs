// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
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

use crate::instance::PhysicalDevice;
use crate::vk;
use crate::VulkanObject;
use half::f16;
use std::convert::TryFrom;
use std::mem::MaybeUninit;
use std::vec::IntoIter as VecIntoIter;
use std::{error, fmt, mem};

macro_rules! formats {
    ($($name:ident => { vk: $vk:ident, bdim: $bdim:expr, size: $sz:expr, ty: $($f_ty:tt)*},)+) => (
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

            /// Returns (width, height) of the dimensions for block based formats. For
            /// non block formats will return (1,1)
            #[inline]
            pub fn block_dimensions(&self) -> (u32, u32) {
                match *self {
                    $(
                        Format::$name => $bdim,
                    )+
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

        impl TryFrom<vk::Format> for Format {
            type Error = ();

            #[inline]
            fn try_from(val: vk::Format) -> Result<Format, ()> {
                match val {
                    $(
                        vk::$vk => Ok(Format::$name),
                    )+
                    _ => Err(()),
                }
            }
        }

        impl From<Format> for vk::Format {
            #[inline]
            fn from(val: Format) -> Self {
                match val {
                    $(
                        Format::$name => vk::$vk,
                    )+
                }
            }
        }
    );

    (__inner_ty__ $name:ident [float; $num:tt]) => { FormatTy::Float };
    (__inner_ty__ $name:ident [uint; $num:tt]) => { FormatTy::Uint };
    (__inner_ty__ $name:ident [sint; $num:tt]) => { FormatTy::Sint };
    (__inner_ty__ $name:ident depth) => { FormatTy::Depth };
    (__inner_ty__ $name:ident stencil) => { FormatTy::Stencil };
    (__inner_ty__ $name:ident depthstencil) => { FormatTy::DepthStencil };
    (__inner_ty__ $name:ident compressed($f:tt)) => { FormatTy::Compressed };
    (__inner_ty__ $name:ident ycbcr) => { FormatTy::Ycbcr };
}

formats! {
    R4G4UnormPack8 => {vk: FORMAT_R4G4_UNORM_PACK8, bdim: (1, 1), size: Some(1), ty: [float; 2]},
    R4G4B4A4UnormPack16 => {vk: FORMAT_R4G4B4A4_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: [float; 4]},
    B4G4R4A4UnormPack16 => {vk: FORMAT_B4G4R4A4_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: [float; 4]},
    R5G6B5UnormPack16 => {vk: FORMAT_R5G6B5_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: [float; 3]},
    B5G6R5UnormPack16 => {vk: FORMAT_B5G6R5_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: [float; 3]},
    R5G5B5A1UnormPack16 => {vk: FORMAT_R5G5B5A1_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: [float; 4]},
    B5G5R5A1UnormPack16 => {vk: FORMAT_B5G5R5A1_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: [float; 4]},
    A1R5G5B5UnormPack16 => {vk: FORMAT_A1R5G5B5_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: [float; 4]},
    R8Unorm => {vk: FORMAT_R8_UNORM, bdim: (1, 1), size: Some(1), ty: [float; 1]},
    R8Snorm => {vk: FORMAT_R8_SNORM, bdim: (1, 1), size: Some(1), ty: [float; 1]},
    R8Uscaled => {vk: FORMAT_R8_USCALED, bdim: (1, 1), size: Some(1), ty: [float; 1]},
    R8Sscaled => {vk: FORMAT_R8_SSCALED, bdim: (1, 1), size: Some(1), ty: [float; 1]},
    R8Uint => {vk: FORMAT_R8_UINT, bdim: (1, 1), size: Some(1), ty: [uint; 1]},
    R8Sint => {vk: FORMAT_R8_SINT, bdim: (1, 1), size: Some(1), ty: [sint; 1]},
    R8Srgb => {vk: FORMAT_R8_SRGB, bdim: (1, 1), size: Some(1), ty: [float; 1]},
    R8G8Unorm => {vk: FORMAT_R8G8_UNORM, bdim: (1, 1), size: Some(2), ty: [float; 2]},
    R8G8Snorm => {vk: FORMAT_R8G8_SNORM, bdim: (1, 1), size: Some(2), ty: [float; 2]},
    R8G8Uscaled => {vk: FORMAT_R8G8_USCALED, bdim: (1, 1), size: Some(2), ty: [float; 2]},
    R8G8Sscaled => {vk: FORMAT_R8G8_SSCALED, bdim: (1, 1), size: Some(2), ty: [float; 2]},
    R8G8Uint => {vk: FORMAT_R8G8_UINT, bdim: (1, 1), size: Some(2), ty: [uint; 2]},
    R8G8Sint => {vk: FORMAT_R8G8_SINT, bdim: (1, 1), size: Some(2), ty: [sint; 2]},
    R8G8Srgb => {vk: FORMAT_R8G8_SRGB, bdim: (1, 1), size: Some(2), ty: [float; 2]},
    R8G8B8Unorm => {vk: FORMAT_R8G8B8_UNORM, bdim: (1, 1), size: Some(3), ty: [float; 3]},
    R8G8B8Snorm => {vk: FORMAT_R8G8B8_SNORM, bdim: (1, 1), size: Some(3), ty: [float; 3]},
    R8G8B8Uscaled => {vk: FORMAT_R8G8B8_USCALED, bdim: (1, 1), size: Some(3), ty: [float; 3]},
    R8G8B8Sscaled => {vk: FORMAT_R8G8B8_SSCALED, bdim: (1, 1), size: Some(3), ty: [float; 3]},
    R8G8B8Uint => {vk: FORMAT_R8G8B8_UINT, bdim: (1, 1), size: Some(3), ty: [uint; 3]},
    R8G8B8Sint => {vk: FORMAT_R8G8B8_SINT, bdim: (1, 1), size: Some(3), ty: [sint; 3]},
    R8G8B8Srgb => {vk: FORMAT_R8G8B8_SRGB, bdim: (1, 1), size: Some(3), ty: [float; 3]},
    B8G8R8Unorm => {vk: FORMAT_B8G8R8_UNORM, bdim: (1, 1), size: Some(3), ty: [float; 3]},
    B8G8R8Snorm => {vk: FORMAT_B8G8R8_SNORM, bdim: (1, 1), size: Some(3), ty: [float; 3]},
    B8G8R8Uscaled => {vk: FORMAT_B8G8R8_USCALED, bdim: (1, 1), size: Some(3), ty: [float; 3]},
    B8G8R8Sscaled => {vk: FORMAT_B8G8R8_SSCALED, bdim: (1, 1), size: Some(3), ty: [float; 3]},
    B8G8R8Uint => {vk: FORMAT_B8G8R8_UINT, bdim: (1, 1), size: Some(3), ty: [uint; 3]},
    B8G8R8Sint => {vk: FORMAT_B8G8R8_SINT, bdim: (1, 1), size: Some(3), ty: [sint; 3]},
    B8G8R8Srgb => {vk: FORMAT_B8G8R8_SRGB, bdim: (1, 1), size: Some(3), ty: [float; 3]},
    R8G8B8A8Unorm => {vk: FORMAT_R8G8B8A8_UNORM, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    R8G8B8A8Snorm => {vk: FORMAT_R8G8B8A8_SNORM, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    R8G8B8A8Uscaled => {vk: FORMAT_R8G8B8A8_USCALED, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    R8G8B8A8Sscaled => {vk: FORMAT_R8G8B8A8_SSCALED, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    R8G8B8A8Uint => {vk: FORMAT_R8G8B8A8_UINT, bdim: (1, 1), size: Some(4), ty: [uint; 4]},
    R8G8B8A8Sint => {vk: FORMAT_R8G8B8A8_SINT, bdim: (1, 1), size: Some(4), ty: [sint; 4]},
    R8G8B8A8Srgb => {vk: FORMAT_R8G8B8A8_SRGB, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    B8G8R8A8Unorm => {vk: FORMAT_B8G8R8A8_UNORM, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    B8G8R8A8Snorm => {vk: FORMAT_B8G8R8A8_SNORM, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    B8G8R8A8Uscaled => {vk: FORMAT_B8G8R8A8_USCALED, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    B8G8R8A8Sscaled => {vk: FORMAT_B8G8R8A8_SSCALED, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    B8G8R8A8Uint => {vk: FORMAT_B8G8R8A8_UINT, bdim: (1, 1), size: Some(4), ty: [uint; 4]},
    B8G8R8A8Sint => {vk: FORMAT_B8G8R8A8_SINT, bdim: (1, 1), size: Some(4), ty: [sint; 4]},
    B8G8R8A8Srgb => {vk: FORMAT_B8G8R8A8_SRGB, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A8B8G8R8UnormPack32 => {vk: FORMAT_A8B8G8R8_UNORM_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A8B8G8R8SnormPack32 => {vk: FORMAT_A8B8G8R8_SNORM_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A8B8G8R8UscaledPack32 => {vk: FORMAT_A8B8G8R8_USCALED_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A8B8G8R8SscaledPack32 => {vk: FORMAT_A8B8G8R8_SSCALED_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A8B8G8R8UintPack32 => {vk: FORMAT_A8B8G8R8_UINT_PACK32, bdim: (1, 1), size: Some(4), ty: [uint; 4]},
    A8B8G8R8SintPack32 => {vk: FORMAT_A8B8G8R8_SINT_PACK32, bdim: (1, 1), size: Some(4), ty: [sint; 4]},
    A8B8G8R8SrgbPack32 => {vk: FORMAT_A8B8G8R8_SRGB_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A2R10G10B10UnormPack32 => {vk: FORMAT_A2R10G10B10_UNORM_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A2R10G10B10SnormPack32 => {vk: FORMAT_A2R10G10B10_SNORM_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A2R10G10B10UscaledPack32 => {vk: FORMAT_A2R10G10B10_USCALED_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A2R10G10B10SscaledPack32 => {vk: FORMAT_A2R10G10B10_SSCALED_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A2R10G10B10UintPack32 => {vk: FORMAT_A2R10G10B10_UINT_PACK32, bdim: (1, 1), size: Some(4), ty: [uint; 4]},
    A2R10G10B10SintPack32 => {vk: FORMAT_A2R10G10B10_SINT_PACK32, bdim: (1, 1), size: Some(4), ty: [sint; 4]},
    A2B10G10R10UnormPack32 => {vk: FORMAT_A2B10G10R10_UNORM_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A2B10G10R10SnormPack32 => {vk: FORMAT_A2B10G10R10_SNORM_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A2B10G10R10UscaledPack32 => {vk: FORMAT_A2B10G10R10_USCALED_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A2B10G10R10SscaledPack32 => {vk: FORMAT_A2B10G10R10_SSCALED_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 4]},
    A2B10G10R10UintPack32 => {vk: FORMAT_A2B10G10R10_UINT_PACK32, bdim: (1, 1), size: Some(4), ty: [uint; 4]},
    A2B10G10R10SintPack32 => {vk: FORMAT_A2B10G10R10_SINT_PACK32, bdim: (1, 1), size: Some(4), ty: [sint; 4]},
    R16Unorm => {vk: FORMAT_R16_UNORM, bdim: (1, 1), size: Some(2), ty: [float; 1]},
    R16Snorm => {vk: FORMAT_R16_SNORM, bdim: (1, 1), size: Some(2), ty: [float; 1]},
    R16Uscaled => {vk: FORMAT_R16_USCALED, bdim: (1, 1), size: Some(2), ty: [float; 1]},
    R16Sscaled => {vk: FORMAT_R16_SSCALED, bdim: (1, 1), size: Some(2), ty: [float; 1]},
    R16Uint => {vk: FORMAT_R16_UINT, bdim: (1, 1), size: Some(2), ty: [uint; 1]},
    R16Sint => {vk: FORMAT_R16_SINT, bdim: (1, 1), size: Some(2), ty: [sint; 1]},
    R16Sfloat => {vk: FORMAT_R16_SFLOAT, bdim: (1, 1), size: Some(2), ty: [float; 1]},
    R16G16Unorm => {vk: FORMAT_R16G16_UNORM, bdim: (1, 1), size: Some(4), ty: [float; 2]},
    R16G16Snorm => {vk: FORMAT_R16G16_SNORM, bdim: (1, 1), size: Some(4), ty: [float; 2]},
    R16G16Uscaled => {vk: FORMAT_R16G16_USCALED, bdim: (1, 1), size: Some(4), ty: [float; 2]},
    R16G16Sscaled => {vk: FORMAT_R16G16_SSCALED, bdim: (1, 1), size: Some(4), ty: [float; 2]},
    R16G16Uint => {vk: FORMAT_R16G16_UINT, bdim: (1, 1), size: Some(4), ty: [uint; 2]},
    R16G16Sint => {vk: FORMAT_R16G16_SINT, bdim: (1, 1), size: Some(4), ty: [sint; 2]},
    R16G16Sfloat => {vk: FORMAT_R16G16_SFLOAT, bdim: (1, 1), size: Some(4), ty: [float; 2]},
    R16G16B16Unorm => {vk: FORMAT_R16G16B16_UNORM, bdim: (1, 1), size: Some(6), ty: [float; 3]},
    R16G16B16Snorm => {vk: FORMAT_R16G16B16_SNORM, bdim: (1, 1), size: Some(6), ty: [float; 3]},
    R16G16B16Uscaled => {vk: FORMAT_R16G16B16_USCALED, bdim: (1, 1), size: Some(6), ty: [float; 3]},
    R16G16B16Sscaled => {vk: FORMAT_R16G16B16_SSCALED, bdim: (1, 1), size: Some(6), ty: [float; 3]},
    R16G16B16Uint => {vk: FORMAT_R16G16B16_UINT, bdim: (1, 1), size: Some(6), ty: [uint; 3]},
    R16G16B16Sint => {vk: FORMAT_R16G16B16_SINT, bdim: (1, 1), size: Some(6), ty: [sint; 3]},
    R16G16B16Sfloat => {vk: FORMAT_R16G16B16_SFLOAT, bdim: (1, 1), size: Some(6), ty: [float; 3]},
    R16G16B16A16Unorm => {vk: FORMAT_R16G16B16A16_UNORM, bdim: (1, 1), size: Some(8), ty: [float; 4]},
    R16G16B16A16Snorm => {vk: FORMAT_R16G16B16A16_SNORM, bdim: (1, 1), size: Some(8), ty: [float; 4]},
    R16G16B16A16Uscaled => {vk: FORMAT_R16G16B16A16_USCALED, bdim: (1, 1), size: Some(8), ty: [float; 4]},
    R16G16B16A16Sscaled => {vk: FORMAT_R16G16B16A16_SSCALED, bdim: (1, 1), size: Some(8), ty: [float; 4]},
    R16G16B16A16Uint => {vk: FORMAT_R16G16B16A16_UINT, bdim: (1, 1), size: Some(8), ty: [uint; 4]},
    R16G16B16A16Sint => {vk: FORMAT_R16G16B16A16_SINT, bdim: (1, 1), size: Some(8), ty: [sint; 4]},
    R16G16B16A16Sfloat => {vk: FORMAT_R16G16B16A16_SFLOAT, bdim: (1, 1), size: Some(8), ty: [float; 4]},
    R32Uint => {vk: FORMAT_R32_UINT, bdim: (1, 1), size: Some(4), ty: [uint; 1]},
    R32Sint => {vk: FORMAT_R32_SINT, bdim: (1, 1), size: Some(4), ty: [sint; 1]},
    R32Sfloat => {vk: FORMAT_R32_SFLOAT, bdim: (1, 1), size: Some(4), ty: [float; 1]},
    R32G32Uint => {vk: FORMAT_R32G32_UINT, bdim: (1, 1), size: Some(8), ty: [uint; 2]},
    R32G32Sint => {vk: FORMAT_R32G32_SINT, bdim: (1, 1), size: Some(8), ty: [sint; 2]},
    R32G32Sfloat => {vk: FORMAT_R32G32_SFLOAT, bdim: (1, 1), size: Some(8), ty: [float; 2]},
    R32G32B32Uint => {vk: FORMAT_R32G32B32_UINT, bdim: (1, 1), size: Some(12), ty: [uint; 3]},
    R32G32B32Sint => {vk: FORMAT_R32G32B32_SINT, bdim: (1, 1), size: Some(12), ty: [sint; 3]},
    R32G32B32Sfloat => {vk: FORMAT_R32G32B32_SFLOAT, bdim: (1, 1), size: Some(12), ty: [float; 3]},
    R32G32B32A32Uint => {vk: FORMAT_R32G32B32A32_UINT, bdim: (1, 1), size: Some(16), ty: [uint; 4]},
    R32G32B32A32Sint => {vk: FORMAT_R32G32B32A32_SINT, bdim: (1, 1), size: Some(16), ty: [sint; 4]},
    R32G32B32A32Sfloat => {vk: FORMAT_R32G32B32A32_SFLOAT, bdim: (1, 1), size: Some(16), ty: [float; 4]},
    R64Uint => {vk: FORMAT_R64_UINT, bdim: (1, 1), size: Some(8), ty: [uint; 1]},
    R64Sint => {vk: FORMAT_R64_SINT, bdim: (1, 1), size: Some(8), ty: [sint; 1]},
    R64Sfloat => {vk: FORMAT_R64_SFLOAT, bdim: (1, 1), size: Some(8), ty: [float; 1]},
    R64G64Uint => {vk: FORMAT_R64G64_UINT, bdim: (1, 1), size: Some(16), ty: [uint; 2]},
    R64G64Sint => {vk: FORMAT_R64G64_SINT, bdim: (1, 1), size: Some(16), ty: [sint; 2]},
    R64G64Sfloat => {vk: FORMAT_R64G64_SFLOAT, bdim: (1, 1), size: Some(16), ty: [float; 2]},
    R64G64B64Uint => {vk: FORMAT_R64G64B64_UINT, bdim: (1, 1), size: Some(24), ty: [uint; 3]},
    R64G64B64Sint => {vk: FORMAT_R64G64B64_SINT, bdim: (1, 1), size: Some(24), ty: [sint; 3]},
    R64G64B64Sfloat => {vk: FORMAT_R64G64B64_SFLOAT, bdim: (1, 1), size: Some(24), ty: [float; 3]},
    R64G64B64A64Uint => {vk: FORMAT_R64G64B64A64_UINT, bdim: (1, 1), size: Some(32), ty: [uint; 4]},
    R64G64B64A64Sint => {vk: FORMAT_R64G64B64A64_SINT, bdim: (1, 1), size: Some(32), ty: [sint; 4]},
    R64G64B64A64Sfloat => {vk: FORMAT_R64G64B64A64_SFLOAT, bdim: (1, 1), size: Some(32), ty: [float; 4]},
    B10G11R11UfloatPack32 => {vk: FORMAT_B10G11R11_UFLOAT_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 3]},
    E5B9G9R9UfloatPack32 => {vk: FORMAT_E5B9G9R9_UFLOAT_PACK32, bdim: (1, 1), size: Some(4), ty: [float; 3]},
    D16Unorm => {vk: FORMAT_D16_UNORM, bdim: (1, 1), size: Some(2), ty: depth},
    X8_D24UnormPack32 => {vk: FORMAT_X8_D24_UNORM_PACK32, bdim: (1, 1), size: Some(4), ty: depth},
    D32Sfloat => {vk: FORMAT_D32_SFLOAT, bdim: (1, 1), size: Some(4), ty: depth},
    S8Uint => {vk: FORMAT_S8_UINT, bdim: (1, 1), size: Some(1), ty: stencil},
    D16Unorm_S8Uint => {vk: FORMAT_D16_UNORM_S8_UINT, bdim: (1, 1), size: None, ty: depthstencil},
    D24Unorm_S8Uint => {vk: FORMAT_D24_UNORM_S8_UINT, bdim: (1, 1), size: None, ty: depthstencil},
    D32Sfloat_S8Uint => {vk: FORMAT_D32_SFLOAT_S8_UINT, bdim: (1, 1), size: None, ty: depthstencil},
    BC1_RGBUnormBlock => {vk: FORMAT_BC1_RGB_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_bc)},
    BC1_RGBSrgbBlock => {vk: FORMAT_BC1_RGB_SRGB_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_bc)},
    BC1_RGBAUnormBlock => {vk: FORMAT_BC1_RGBA_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_bc)},
    BC1_RGBASrgbBlock => {vk: FORMAT_BC1_RGBA_SRGB_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_bc)},
    BC2UnormBlock => {vk: FORMAT_BC2_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_bc)},
    BC2SrgbBlock => {vk: FORMAT_BC2_SRGB_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_bc)},
    BC3UnormBlock => {vk: FORMAT_BC3_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_bc)},
    BC3SrgbBlock => {vk: FORMAT_BC3_SRGB_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_bc)},
    BC4UnormBlock => {vk: FORMAT_BC4_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_bc)},
    BC4SnormBlock => {vk: FORMAT_BC4_SNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_bc)},
    BC5UnormBlock => {vk: FORMAT_BC5_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_bc)},
    BC5SnormBlock => {vk: FORMAT_BC5_SNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_bc)},
    BC6HUfloatBlock => {vk: FORMAT_BC6H_UFLOAT_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_bc)},
    BC6HSfloatBlock => {vk: FORMAT_BC6H_SFLOAT_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_bc)},
    BC7UnormBlock => {vk: FORMAT_BC7_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_bc)},
    BC7SrgbBlock => {vk: FORMAT_BC7_SRGB_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_bc)},
    ETC2_R8G8B8UnormBlock => {vk: FORMAT_ETC2_R8G8B8_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_etc2)},
    ETC2_R8G8B8SrgbBlock => {vk: FORMAT_ETC2_R8G8B8_SRGB_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_etc2)},
    ETC2_R8G8B8A1UnormBlock => {vk: FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_etc2)},
    ETC2_R8G8B8A1SrgbBlock => {vk: FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_etc2)},
    ETC2_R8G8B8A8UnormBlock => {vk: FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_etc2)},
    ETC2_R8G8B8A8SrgbBlock => {vk: FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_etc2)},
    EAC_R11UnormBlock => {vk: FORMAT_EAC_R11_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_etc2)},
    EAC_R11SnormBlock => {vk: FORMAT_EAC_R11_SNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: compressed(texture_compression_etc2)},
    EAC_R11G11UnormBlock => {vk: FORMAT_EAC_R11G11_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_etc2)},
    EAC_R11G11SnormBlock => {vk: FORMAT_EAC_R11G11_SNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_etc2)},
    ASTC_4x4UnormBlock => {vk: FORMAT_ASTC_4x4_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_4x4SrgbBlock => {vk: FORMAT_ASTC_4x4_SRGB_BLOCK, bdim: (4, 4), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_5x4UnormBlock => {vk: FORMAT_ASTC_5x4_UNORM_BLOCK, bdim: (5, 4), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_5x4SrgbBlock => {vk: FORMAT_ASTC_5x4_SRGB_BLOCK, bdim: (5, 4), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_5x5UnormBlock => {vk: FORMAT_ASTC_5x5_UNORM_BLOCK, bdim: (5, 5), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_5x5SrgbBlock => {vk: FORMAT_ASTC_5x5_SRGB_BLOCK, bdim: (5, 5), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_6x5UnormBlock => {vk: FORMAT_ASTC_6x5_UNORM_BLOCK, bdim: (6, 5), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_6x5SrgbBlock => {vk: FORMAT_ASTC_6x5_SRGB_BLOCK, bdim: (6, 5), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_6x6UnormBlock => {vk: FORMAT_ASTC_6x6_UNORM_BLOCK, bdim: (6, 6), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_6x6SrgbBlock => {vk: FORMAT_ASTC_6x6_SRGB_BLOCK, bdim: (6, 6), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_8x5UnormBlock => {vk: FORMAT_ASTC_8x5_UNORM_BLOCK, bdim: (8, 5), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_8x5SrgbBlock => {vk: FORMAT_ASTC_8x5_SRGB_BLOCK, bdim: (8, 5), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_8x6UnormBlock => {vk: FORMAT_ASTC_8x6_UNORM_BLOCK, bdim: (8, 6), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_8x6SrgbBlock => {vk: FORMAT_ASTC_8x6_SRGB_BLOCK, bdim: (8, 6), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_8x8UnormBlock => {vk: FORMAT_ASTC_8x8_UNORM_BLOCK, bdim: (8, 8), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_8x8SrgbBlock => {vk: FORMAT_ASTC_8x8_SRGB_BLOCK, bdim: (8, 8), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_10x5UnormBlock => {vk: FORMAT_ASTC_10x5_UNORM_BLOCK, bdim: (10, 5), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_10x5SrgbBlock => {vk: FORMAT_ASTC_10x5_SRGB_BLOCK, bdim: (10, 5), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_10x6UnormBlock => {vk: FORMAT_ASTC_10x6_UNORM_BLOCK, bdim: (10, 6), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_10x6SrgbBlock => {vk: FORMAT_ASTC_10x6_SRGB_BLOCK, bdim: (10, 6), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_10x8UnormBlock => {vk: FORMAT_ASTC_10x8_UNORM_BLOCK, bdim: (10, 8), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_10x8SrgbBlock => {vk: FORMAT_ASTC_10x8_SRGB_BLOCK, bdim: (10, 8), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_10x10UnormBlock => {vk: FORMAT_ASTC_10x10_UNORM_BLOCK, bdim: (10, 10), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_10x10SrgbBlock => {vk: FORMAT_ASTC_10x10_SRGB_BLOCK, bdim: (10, 10), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_12x10UnormBlock => {vk: FORMAT_ASTC_12x10_UNORM_BLOCK, bdim: (12, 10), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_12x10SrgbBlock => {vk: FORMAT_ASTC_12x10_SRGB_BLOCK, bdim: (12, 10), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_12x12UnormBlock => {vk: FORMAT_ASTC_12x12_UNORM_BLOCK, bdim: (12, 12), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    ASTC_12x12SrgbBlock => {vk: FORMAT_ASTC_12x12_SRGB_BLOCK, bdim: (12, 12), size: Some(16), ty: compressed(texture_compression_astc_ldr)},
    G8B8R8_3PLANE420Unorm => {vk: FORMAT_G8_B8_R8_3PLANE_420_UNORM, bdim: (1, 1), size: None, ty: ycbcr},
    G8B8R8_2PLANE420Unorm => {vk: FORMAT_G8_B8R8_2PLANE_420_UNORM, bdim: (1, 1), size: None, ty: ycbcr},
}

impl Format {
    /// Retrieves the properties of a format when used by a certain device.
    #[inline]
    pub fn properties(&self, physical_device: PhysicalDevice) -> FormatProperties {
        let vk_properties = unsafe {
            let vk_i = physical_device.instance().pointers();
            let mut output = MaybeUninit::uninit();
            vk_i.GetPhysicalDeviceFormatProperties(
                physical_device.internal_object(),
                (*self).into(),
                output.as_mut_ptr(),
            );
            output.assume_init()
        };

        FormatProperties {
            linear_tiling_features: vk_properties.linearTilingFeatures.into(),
            optimal_tiling_features: vk_properties.optimalTilingFeatures.into(),
            buffer_features: vk_properties.bufferFeatures.into(),
        }
    }

    #[inline]
    pub fn decode_clear_value(&self, value: ClearValue) -> ClearValue {
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

pub unsafe trait AcceptsPixels<T> {
    /// Returns an error if `T` cannot be used as a source of pixels for `Self`.
    fn ensure_accepts(&self) -> Result<(), IncompatiblePixelsType>;

    /// The number of `T`s which make up a single pixel.
    ///
    /// # Panics
    ///
    /// May panic if `ensure_accepts` would not return `Ok(())`.
    fn rate(&self) -> u32 {
        1
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IncompatiblePixelsType;

impl error::Error for IncompatiblePixelsType {}

impl fmt::Display for IncompatiblePixelsType {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            "supplied pixels' type is incompatible with this format"
        )
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FormatTy {
    Float,
    Uint,
    Sint,
    Depth,
    Stencil,
    DepthStencil,
    Compressed,
    Ycbcr,
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

/// The properties of an image format that are supported by a physical device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct FormatProperties {
    /// Features available for images with linear tiling.
    pub linear_tiling_features: FormatFeatures,

    /// Features available for images with optimal tiling.
    pub optimal_tiling_features: FormatFeatures,

    /// Features available for buffers.
    pub buffer_features: FormatFeatures,
}

/// The features supported by images with a particular format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(missing_docs)]
pub struct FormatFeatures {
    pub sampled_image: bool,
    pub storage_image: bool,
    pub storage_image_atomic: bool,
    pub uniform_texel_buffer: bool,
    pub storage_texel_buffer: bool,
    pub storage_texel_buffer_atomic: bool,
    pub vertex_buffer: bool,
    pub color_attachment: bool,
    pub color_attachment_blend: bool,
    pub depth_stencil_attachment: bool,
    pub blit_src: bool,
    pub blit_dst: bool,
    pub sampled_image_filter_linear: bool,
    pub transfer_src: bool,
    pub transfer_dst: bool,
    pub midpoint_chroma_samples: bool,
    pub sampled_image_ycbcr_conversion_linear_filter: bool,
    pub sampled_image_ycbcr_conversion_separate_reconstruction_filter: bool,
    pub sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: bool,
    pub sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: bool,
    pub disjoint: bool,
    pub cosited_chroma_samples: bool,
    pub sampled_image_filter_minmax: bool,
    pub img_sampled_image_filter_cubic: bool,
    pub khr_acceleration_structure_vertex_buffer: bool,
    pub ext_fragment_density_map: bool,
}

impl From<vk::FormatFeatureFlags> for FormatFeatures {
    #[inline]
    #[rustfmt::skip]
    fn from(val: vk::FormatFeatureFlags) -> FormatFeatures {
        FormatFeatures {
            sampled_image: (val & vk::FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0,
            storage_image: (val & vk::FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0,
            storage_image_atomic: (val & vk::FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) != 0,
            uniform_texel_buffer: (val & vk::FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) != 0,
            storage_texel_buffer: (val & vk::FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT) != 0,
            storage_texel_buffer_atomic: (val & vk::FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT) != 0,
            vertex_buffer: (val & vk::FORMAT_FEATURE_VERTEX_BUFFER_BIT) != 0,
            color_attachment: (val & vk::FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0,
            color_attachment_blend: (val & vk::FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0,
            depth_stencil_attachment: (val & vk::FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0,
            blit_src: (val & vk::FORMAT_FEATURE_BLIT_SRC_BIT) != 0,
            blit_dst: (val & vk::FORMAT_FEATURE_BLIT_DST_BIT) != 0,
            sampled_image_filter_linear: (val & vk::FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0,
            transfer_src: (val & vk::FORMAT_FEATURE_TRANSFER_SRC_BIT) != 0,
            transfer_dst: (val & vk::FORMAT_FEATURE_TRANSFER_DST_BIT) != 0,
            midpoint_chroma_samples: (val & vk::FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT) != 0,
            sampled_image_ycbcr_conversion_linear_filter: (val & vk::FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT) != 0,
            sampled_image_ycbcr_conversion_separate_reconstruction_filter: (val & vk::FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT) != 0,
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: (val & vk::FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT) != 0,
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: (val & vk::FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT) != 0,
            disjoint: (val & vk::FORMAT_FEATURE_DISJOINT_BIT) != 0,
            cosited_chroma_samples: (val & vk::FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT) != 0,
            sampled_image_filter_minmax: (val & vk::FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT) != 0,
            img_sampled_image_filter_cubic: (val & vk::FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG) != 0,
            khr_acceleration_structure_vertex_buffer: (val & vk::FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR) != 0,
            ext_fragment_density_map: (val & vk::FORMAT_FEATURE_FRAGMENT_DENSITY_MAP_BIT_EXT) != 0,
        }
    }
}
