// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! All the formats of images supported by Vulkan.
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
//! - B4G4R4A4UnormPack16
//! - R5G6B5UnormPack16
//! - A1R5G5B5UnormPack16
//! - R8Unorm
//! - R8Snorm
//! - R8G8Unorm
//! - R8G8Snorm
//! - R8G8B8A8Unorm
//! - R8G8B8A8Snorm
//! - R8G8B8A8Srgb
//! - B8G8R8A8Unorm
//! - B8G8R8A8Srgb
//! - A8B8G8R8UnormPack32
//! - A8B8G8R8SnormPack32
//! - A8B8G8R8SrgbPack32
//! - A2B10G10R10UnormPack32
//! - R16Sfloat
//! - R16G16Sfloat
//! - R16G16B16A16Sfloat
//! - B10G11R11UfloatPack32
//! - E5B9G9R9UfloatPack32
//!
//! The following formats are guaranteed to be supported for everything that is related to
//! intermediate render targets (ie. blitting destination, color attachment and sampling linearly):
//!
//! - R5G6B5UnormPack16
//! - A1R5G5B5UnormPack16
//! - R8Unorm
//! - R8G8Unorm
//! - R8G8B8A8Unorm
//! - R8G8B8A8Srgb
//! - B8G8R8A8Unorm
//! - B8G8R8A8Srgb
//! - A8B8G8R8UnormPack32
//! - A8B8G8R8SrgbPack32
//! - A2B10G10R10UnormPack32
//! - R16Sfloat
//! - R16G16Sfloat
//! - R16G16B16A16Sfloat
//!
//! For depth images, only `D16Unorm` is guaranteed to be supported. For depth-stencil images,
//! it is guaranteed that either `D24Unorm_S8Uint` or `D32Sfloat_S8Uint` are supported.
//!
//! // TODO: storage formats
//!

use crate::device::physical::PhysicalDevice;
use crate::image::ImageAspects;
use crate::DeviceSize;
use crate::VulkanObject;
use half::f16;
use std::convert::TryFrom;
use std::mem::MaybeUninit;
use std::vec::IntoIter as VecIntoIter;
use std::{error, fmt, mem};

macro_rules! formats {
    ($($name:ident => { vk: $vk:ident, bdim: $bdim:expr, size: $sz:expr, ty: $f_ty:ident$(, planes: $planes:expr)?},)+) => (
        /// An enumeration of all the possible formats.
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        #[repr(i32)]
        #[allow(missing_docs)]
        #[allow(non_camel_case_types)]
        pub enum Format {
            $($name = ash::vk::Format::$vk.as_raw(),)+
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
            pub const fn size(&self) -> Option<DeviceSize> {
                match *self {
                    $(
                        Format::$name => $sz,
                    )+
                }
            }

            /// Returns (width, height) of the dimensions for block based formats. For
            /// non block formats will return (1,1)
            #[inline]
            pub const fn block_dimensions(&self) -> (u32, u32) {
                match *self {
                    $(
                        Format::$name => $bdim,
                    )+
                }
            }

            /// Returns the data type of the format.
            #[inline]
            pub const fn ty(&self) -> FormatTy {
                match *self {
                    $(
                        Format::$name => FormatTy::$f_ty,
                    )+
                }
            }

            /// Returns the number of planes that images of this format have.
            ///
            /// Returns 0 if the format is not multi-planar.
            #[inline]
            pub const fn planes(&self) -> u8 {
                match *self {
                    $(
                        $(Format::$name => $planes,)?
                    )+
                    _ => 0,
                }
            }
        }

        impl TryFrom<ash::vk::Format> for Format {
            type Error = ();

            #[inline]
            fn try_from(val: ash::vk::Format) -> Result<Format, ()> {
                match val {
                    $(
                        ash::vk::Format::$vk => Ok(Format::$name),
                    )+
                    _ => Err(()),
                }
            }
        }

        impl From<Format> for ash::vk::Format {
            #[inline]
            fn from(val: Format) -> Self {
                ash::vk::Format::from_raw(val as i32)
            }
        }
    );
}

formats! {
    R4G4UnormPack8 => {vk: R4G4_UNORM_PACK8, bdim: (1, 1), size: Some(1), ty: Float},
    R4G4B4A4UnormPack16 => {vk: R4G4B4A4_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: Float},
    B4G4R4A4UnormPack16 => {vk: B4G4R4A4_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: Float},
    R5G6B5UnormPack16 => {vk: R5G6B5_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: Float},
    B5G6R5UnormPack16 => {vk: B5G6R5_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: Float},
    R5G5B5A1UnormPack16 => {vk: R5G5B5A1_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: Float},
    B5G5R5A1UnormPack16 => {vk: B5G5R5A1_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: Float},
    A1R5G5B5UnormPack16 => {vk: A1R5G5B5_UNORM_PACK16, bdim: (1, 1), size: Some(2), ty: Float},
    R8Unorm => {vk: R8_UNORM, bdim: (1, 1), size: Some(1), ty: Float},
    R8Snorm => {vk: R8_SNORM, bdim: (1, 1), size: Some(1), ty: Float},
    R8Uscaled => {vk: R8_USCALED, bdim: (1, 1), size: Some(1), ty: Float},
    R8Sscaled => {vk: R8_SSCALED, bdim: (1, 1), size: Some(1), ty: Float},
    R8Uint => {vk: R8_UINT, bdim: (1, 1), size: Some(1), ty: Uint},
    R8Sint => {vk: R8_SINT, bdim: (1, 1), size: Some(1), ty: Sint},
    R8Srgb => {vk: R8_SRGB, bdim: (1, 1), size: Some(1), ty: Float},
    R8G8Unorm => {vk: R8G8_UNORM, bdim: (1, 1), size: Some(2), ty: Float},
    R8G8Snorm => {vk: R8G8_SNORM, bdim: (1, 1), size: Some(2), ty: Float},
    R8G8Uscaled => {vk: R8G8_USCALED, bdim: (1, 1), size: Some(2), ty: Float},
    R8G8Sscaled => {vk: R8G8_SSCALED, bdim: (1, 1), size: Some(2), ty: Float},
    R8G8Uint => {vk: R8G8_UINT, bdim: (1, 1), size: Some(2), ty: Uint},
    R8G8Sint => {vk: R8G8_SINT, bdim: (1, 1), size: Some(2), ty: Sint},
    R8G8Srgb => {vk: R8G8_SRGB, bdim: (1, 1), size: Some(2), ty: Float},
    R8G8B8Unorm => {vk: R8G8B8_UNORM, bdim: (1, 1), size: Some(3), ty: Float},
    R8G8B8Snorm => {vk: R8G8B8_SNORM, bdim: (1, 1), size: Some(3), ty: Float},
    R8G8B8Uscaled => {vk: R8G8B8_USCALED, bdim: (1, 1), size: Some(3), ty: Float},
    R8G8B8Sscaled => {vk: R8G8B8_SSCALED, bdim: (1, 1), size: Some(3), ty: Float},
    R8G8B8Uint => {vk: R8G8B8_UINT, bdim: (1, 1), size: Some(3), ty: Uint},
    R8G8B8Sint => {vk: R8G8B8_SINT, bdim: (1, 1), size: Some(3), ty: Sint},
    R8G8B8Srgb => {vk: R8G8B8_SRGB, bdim: (1, 1), size: Some(3), ty: Float},
    B8G8R8Unorm => {vk: B8G8R8_UNORM, bdim: (1, 1), size: Some(3), ty: Float},
    B8G8R8Snorm => {vk: B8G8R8_SNORM, bdim: (1, 1), size: Some(3), ty: Float},
    B8G8R8Uscaled => {vk: B8G8R8_USCALED, bdim: (1, 1), size: Some(3), ty: Float},
    B8G8R8Sscaled => {vk: B8G8R8_SSCALED, bdim: (1, 1), size: Some(3), ty: Float},
    B8G8R8Uint => {vk: B8G8R8_UINT, bdim: (1, 1), size: Some(3), ty: Uint},
    B8G8R8Sint => {vk: B8G8R8_SINT, bdim: (1, 1), size: Some(3), ty: Sint},
    B8G8R8Srgb => {vk: B8G8R8_SRGB, bdim: (1, 1), size: Some(3), ty: Float},
    R8G8B8A8Unorm => {vk: R8G8B8A8_UNORM, bdim: (1, 1), size: Some(4), ty: Float},
    R8G8B8A8Snorm => {vk: R8G8B8A8_SNORM, bdim: (1, 1), size: Some(4), ty: Float},
    R8G8B8A8Uscaled => {vk: R8G8B8A8_USCALED, bdim: (1, 1), size: Some(4), ty: Float},
    R8G8B8A8Sscaled => {vk: R8G8B8A8_SSCALED, bdim: (1, 1), size: Some(4), ty: Float},
    R8G8B8A8Uint => {vk: R8G8B8A8_UINT, bdim: (1, 1), size: Some(4), ty: Uint},
    R8G8B8A8Sint => {vk: R8G8B8A8_SINT, bdim: (1, 1), size: Some(4), ty: Sint},
    R8G8B8A8Srgb => {vk: R8G8B8A8_SRGB, bdim: (1, 1), size: Some(4), ty: Float},
    B8G8R8A8Unorm => {vk: B8G8R8A8_UNORM, bdim: (1, 1), size: Some(4), ty: Float},
    B8G8R8A8Snorm => {vk: B8G8R8A8_SNORM, bdim: (1, 1), size: Some(4), ty: Float},
    B8G8R8A8Uscaled => {vk: B8G8R8A8_USCALED, bdim: (1, 1), size: Some(4), ty: Float},
    B8G8R8A8Sscaled => {vk: B8G8R8A8_SSCALED, bdim: (1, 1), size: Some(4), ty: Float},
    B8G8R8A8Uint => {vk: B8G8R8A8_UINT, bdim: (1, 1), size: Some(4), ty: Uint},
    B8G8R8A8Sint => {vk: B8G8R8A8_SINT, bdim: (1, 1), size: Some(4), ty: Sint},
    B8G8R8A8Srgb => {vk: B8G8R8A8_SRGB, bdim: (1, 1), size: Some(4), ty: Float},
    A8B8G8R8UnormPack32 => {vk: A8B8G8R8_UNORM_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A8B8G8R8SnormPack32 => {vk: A8B8G8R8_SNORM_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A8B8G8R8UscaledPack32 => {vk: A8B8G8R8_USCALED_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A8B8G8R8SscaledPack32 => {vk: A8B8G8R8_SSCALED_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A8B8G8R8UintPack32 => {vk: A8B8G8R8_UINT_PACK32, bdim: (1, 1), size: Some(4), ty: Uint},
    A8B8G8R8SintPack32 => {vk: A8B8G8R8_SINT_PACK32, bdim: (1, 1), size: Some(4), ty: Sint},
    A8B8G8R8SrgbPack32 => {vk: A8B8G8R8_SRGB_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A2R10G10B10UnormPack32 => {vk: A2R10G10B10_UNORM_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A2R10G10B10SnormPack32 => {vk: A2R10G10B10_SNORM_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A2R10G10B10UscaledPack32 => {vk: A2R10G10B10_USCALED_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A2R10G10B10SscaledPack32 => {vk: A2R10G10B10_SSCALED_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A2R10G10B10UintPack32 => {vk: A2R10G10B10_UINT_PACK32, bdim: (1, 1), size: Some(4), ty: Uint},
    A2R10G10B10SintPack32 => {vk: A2R10G10B10_SINT_PACK32, bdim: (1, 1), size: Some(4), ty: Sint},
    A2B10G10R10UnormPack32 => {vk: A2B10G10R10_UNORM_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A2B10G10R10SnormPack32 => {vk: A2B10G10R10_SNORM_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A2B10G10R10UscaledPack32 => {vk: A2B10G10R10_USCALED_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A2B10G10R10SscaledPack32 => {vk: A2B10G10R10_SSCALED_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    A2B10G10R10UintPack32 => {vk: A2B10G10R10_UINT_PACK32, bdim: (1, 1), size: Some(4), ty: Uint},
    A2B10G10R10SintPack32 => {vk: A2B10G10R10_SINT_PACK32, bdim: (1, 1), size: Some(4), ty: Sint},
    R16Unorm => {vk: R16_UNORM, bdim: (1, 1), size: Some(2), ty: Float},
    R16Snorm => {vk: R16_SNORM, bdim: (1, 1), size: Some(2), ty: Float},
    R16Uscaled => {vk: R16_USCALED, bdim: (1, 1), size: Some(2), ty: Float},
    R16Sscaled => {vk: R16_SSCALED, bdim: (1, 1), size: Some(2), ty: Float},
    R16Uint => {vk: R16_UINT, bdim: (1, 1), size: Some(2), ty: Uint},
    R16Sint => {vk: R16_SINT, bdim: (1, 1), size: Some(2), ty: Sint},
    R16Sfloat => {vk: R16_SFLOAT, bdim: (1, 1), size: Some(2), ty: Float},
    R16G16Unorm => {vk: R16G16_UNORM, bdim: (1, 1), size: Some(4), ty: Float},
    R16G16Snorm => {vk: R16G16_SNORM, bdim: (1, 1), size: Some(4), ty: Float},
    R16G16Uscaled => {vk: R16G16_USCALED, bdim: (1, 1), size: Some(4), ty: Float},
    R16G16Sscaled => {vk: R16G16_SSCALED, bdim: (1, 1), size: Some(4), ty: Float},
    R16G16Uint => {vk: R16G16_UINT, bdim: (1, 1), size: Some(4), ty: Uint},
    R16G16Sint => {vk: R16G16_SINT, bdim: (1, 1), size: Some(4), ty: Sint},
    R16G16Sfloat => {vk: R16G16_SFLOAT, bdim: (1, 1), size: Some(4), ty: Float},
    R16G16B16Unorm => {vk: R16G16B16_UNORM, bdim: (1, 1), size: Some(6), ty: Float},
    R16G16B16Snorm => {vk: R16G16B16_SNORM, bdim: (1, 1), size: Some(6), ty: Float},
    R16G16B16Uscaled => {vk: R16G16B16_USCALED, bdim: (1, 1), size: Some(6), ty: Float},
    R16G16B16Sscaled => {vk: R16G16B16_SSCALED, bdim: (1, 1), size: Some(6), ty: Float},
    R16G16B16Uint => {vk: R16G16B16_UINT, bdim: (1, 1), size: Some(6), ty: Uint},
    R16G16B16Sint => {vk: R16G16B16_SINT, bdim: (1, 1), size: Some(6), ty: Sint},
    R16G16B16Sfloat => {vk: R16G16B16_SFLOAT, bdim: (1, 1), size: Some(6), ty: Float},
    R16G16B16A16Unorm => {vk: R16G16B16A16_UNORM, bdim: (1, 1), size: Some(8), ty: Float},
    R16G16B16A16Snorm => {vk: R16G16B16A16_SNORM, bdim: (1, 1), size: Some(8), ty: Float},
    R16G16B16A16Uscaled => {vk: R16G16B16A16_USCALED, bdim: (1, 1), size: Some(8), ty: Float},
    R16G16B16A16Sscaled => {vk: R16G16B16A16_SSCALED, bdim: (1, 1), size: Some(8), ty: Float},
    R16G16B16A16Uint => {vk: R16G16B16A16_UINT, bdim: (1, 1), size: Some(8), ty: Uint},
    R16G16B16A16Sint => {vk: R16G16B16A16_SINT, bdim: (1, 1), size: Some(8), ty: Sint},
    R16G16B16A16Sfloat => {vk: R16G16B16A16_SFLOAT, bdim: (1, 1), size: Some(8), ty: Float},
    R32Uint => {vk: R32_UINT, bdim: (1, 1), size: Some(4), ty: Uint},
    R32Sint => {vk: R32_SINT, bdim: (1, 1), size: Some(4), ty: Sint},
    R32Sfloat => {vk: R32_SFLOAT, bdim: (1, 1), size: Some(4), ty: Float},
    R32G32Uint => {vk: R32G32_UINT, bdim: (1, 1), size: Some(8), ty: Uint},
    R32G32Sint => {vk: R32G32_SINT, bdim: (1, 1), size: Some(8), ty: Sint},
    R32G32Sfloat => {vk: R32G32_SFLOAT, bdim: (1, 1), size: Some(8), ty: Float},
    R32G32B32Uint => {vk: R32G32B32_UINT, bdim: (1, 1), size: Some(12), ty: Uint},
    R32G32B32Sint => {vk: R32G32B32_SINT, bdim: (1, 1), size: Some(12), ty: Sint},
    R32G32B32Sfloat => {vk: R32G32B32_SFLOAT, bdim: (1, 1), size: Some(12), ty: Float},
    R32G32B32A32Uint => {vk: R32G32B32A32_UINT, bdim: (1, 1), size: Some(16), ty: Uint},
    R32G32B32A32Sint => {vk: R32G32B32A32_SINT, bdim: (1, 1), size: Some(16), ty: Sint},
    R32G32B32A32Sfloat => {vk: R32G32B32A32_SFLOAT, bdim: (1, 1), size: Some(16), ty: Float},
    R64Uint => {vk: R64_UINT, bdim: (1, 1), size: Some(8), ty: Uint},
    R64Sint => {vk: R64_SINT, bdim: (1, 1), size: Some(8), ty: Sint},
    R64Sfloat => {vk: R64_SFLOAT, bdim: (1, 1), size: Some(8), ty: Float},
    R64G64Uint => {vk: R64G64_UINT, bdim: (1, 1), size: Some(16), ty: Uint},
    R64G64Sint => {vk: R64G64_SINT, bdim: (1, 1), size: Some(16), ty: Sint},
    R64G64Sfloat => {vk: R64G64_SFLOAT, bdim: (1, 1), size: Some(16), ty: Float},
    R64G64B64Uint => {vk: R64G64B64_UINT, bdim: (1, 1), size: Some(24), ty: Uint},
    R64G64B64Sint => {vk: R64G64B64_SINT, bdim: (1, 1), size: Some(24), ty: Sint},
    R64G64B64Sfloat => {vk: R64G64B64_SFLOAT, bdim: (1, 1), size: Some(24), ty: Float},
    R64G64B64A64Uint => {vk: R64G64B64A64_UINT, bdim: (1, 1), size: Some(32), ty: Uint},
    R64G64B64A64Sint => {vk: R64G64B64A64_SINT, bdim: (1, 1), size: Some(32), ty: Sint},
    R64G64B64A64Sfloat => {vk: R64G64B64A64_SFLOAT, bdim: (1, 1), size: Some(32), ty: Float},
    B10G11R11UfloatPack32 => {vk: B10G11R11_UFLOAT_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    E5B9G9R9UfloatPack32 => {vk: E5B9G9R9_UFLOAT_PACK32, bdim: (1, 1), size: Some(4), ty: Float},
    D16Unorm => {vk: D16_UNORM, bdim: (1, 1), size: Some(2), ty: Depth},
    X8_D24UnormPack32 => {vk: X8_D24_UNORM_PACK32, bdim: (1, 1), size: Some(4), ty: Depth},
    D32Sfloat => {vk: D32_SFLOAT, bdim: (1, 1), size: Some(4), ty: Depth},
    S8Uint => {vk: S8_UINT, bdim: (1, 1), size: Some(1), ty: Stencil},
    D16Unorm_S8Uint => {vk: D16_UNORM_S8_UINT, bdim: (1, 1), size: None, ty: DepthStencil},
    D24Unorm_S8Uint => {vk: D24_UNORM_S8_UINT, bdim: (1, 1), size: None, ty: DepthStencil},
    D32Sfloat_S8Uint => {vk: D32_SFLOAT_S8_UINT, bdim: (1, 1), size: None, ty: DepthStencil},
    BC1_RGBUnormBlock => {vk: BC1_RGB_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    BC1_RGBSrgbBlock => {vk: BC1_RGB_SRGB_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    BC1_RGBAUnormBlock => {vk: BC1_RGBA_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    BC1_RGBASrgbBlock => {vk: BC1_RGBA_SRGB_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    BC2UnormBlock => {vk: BC2_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    BC2SrgbBlock => {vk: BC2_SRGB_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    BC3UnormBlock => {vk: BC3_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    BC3SrgbBlock => {vk: BC3_SRGB_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    BC4UnormBlock => {vk: BC4_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    BC4SnormBlock => {vk: BC4_SNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    BC5UnormBlock => {vk: BC5_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    BC5SnormBlock => {vk: BC5_SNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    BC6HUfloatBlock => {vk: BC6H_UFLOAT_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    BC6HSfloatBlock => {vk: BC6H_SFLOAT_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    BC7UnormBlock => {vk: BC7_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    BC7SrgbBlock => {vk: BC7_SRGB_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    ETC2_R8G8B8UnormBlock => {vk: ETC2_R8G8B8_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    ETC2_R8G8B8SrgbBlock => {vk: ETC2_R8G8B8_SRGB_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    ETC2_R8G8B8A1UnormBlock => {vk: ETC2_R8G8B8A1_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    ETC2_R8G8B8A1SrgbBlock => {vk: ETC2_R8G8B8A1_SRGB_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    ETC2_R8G8B8A8UnormBlock => {vk: ETC2_R8G8B8A8_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    ETC2_R8G8B8A8SrgbBlock => {vk: ETC2_R8G8B8A8_SRGB_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    EAC_R11UnormBlock => {vk: EAC_R11_UNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    EAC_R11SnormBlock => {vk: EAC_R11_SNORM_BLOCK, bdim: (4, 4), size: Some(8), ty: Compressed},
    EAC_R11G11UnormBlock => {vk: EAC_R11G11_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    EAC_R11G11SnormBlock => {vk: EAC_R11G11_SNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    ASTC_4x4UnormBlock => {vk: ASTC_4X4_UNORM_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    ASTC_4x4SrgbBlock => {vk: ASTC_4X4_SRGB_BLOCK, bdim: (4, 4), size: Some(16), ty: Compressed},
    ASTC_5x4UnormBlock => {vk: ASTC_5X4_UNORM_BLOCK, bdim: (5, 4), size: Some(16), ty: Compressed},
    ASTC_5x4SrgbBlock => {vk: ASTC_5X4_SRGB_BLOCK, bdim: (5, 4), size: Some(16), ty: Compressed},
    ASTC_5x5UnormBlock => {vk: ASTC_5X5_UNORM_BLOCK, bdim: (5, 5), size: Some(16), ty: Compressed},
    ASTC_5x5SrgbBlock => {vk: ASTC_5X5_SRGB_BLOCK, bdim: (5, 5), size: Some(16), ty: Compressed},
    ASTC_6x5UnormBlock => {vk: ASTC_6X5_UNORM_BLOCK, bdim: (6, 5), size: Some(16), ty: Compressed},
    ASTC_6x5SrgbBlock => {vk: ASTC_6X5_SRGB_BLOCK, bdim: (6, 5), size: Some(16), ty: Compressed},
    ASTC_6x6UnormBlock => {vk: ASTC_6X6_UNORM_BLOCK, bdim: (6, 6), size: Some(16), ty: Compressed},
    ASTC_6x6SrgbBlock => {vk: ASTC_6X6_SRGB_BLOCK, bdim: (6, 6), size: Some(16), ty: Compressed},
    ASTC_8x5UnormBlock => {vk: ASTC_8X5_UNORM_BLOCK, bdim: (8, 5), size: Some(16), ty: Compressed},
    ASTC_8x5SrgbBlock => {vk: ASTC_8X5_SRGB_BLOCK, bdim: (8, 5), size: Some(16), ty: Compressed},
    ASTC_8x6UnormBlock => {vk: ASTC_8X6_UNORM_BLOCK, bdim: (8, 6), size: Some(16), ty: Compressed},
    ASTC_8x6SrgbBlock => {vk: ASTC_8X6_SRGB_BLOCK, bdim: (8, 6), size: Some(16), ty: Compressed},
    ASTC_8x8UnormBlock => {vk: ASTC_8X8_UNORM_BLOCK, bdim: (8, 8), size: Some(16), ty: Compressed},
    ASTC_8x8SrgbBlock => {vk: ASTC_8X8_SRGB_BLOCK, bdim: (8, 8), size: Some(16), ty: Compressed},
    ASTC_10x5UnormBlock => {vk: ASTC_10X5_UNORM_BLOCK, bdim: (10, 5), size: Some(16), ty: Compressed},
    ASTC_10x5SrgbBlock => {vk: ASTC_10X5_SRGB_BLOCK, bdim: (10, 5), size: Some(16), ty: Compressed},
    ASTC_10x6UnormBlock => {vk: ASTC_10X6_UNORM_BLOCK, bdim: (10, 6), size: Some(16), ty: Compressed},
    ASTC_10x6SrgbBlock => {vk: ASTC_10X6_SRGB_BLOCK, bdim: (10, 6), size: Some(16), ty: Compressed},
    ASTC_10x8UnormBlock => {vk: ASTC_10X8_UNORM_BLOCK, bdim: (10, 8), size: Some(16), ty: Compressed},
    ASTC_10x8SrgbBlock => {vk: ASTC_10X8_SRGB_BLOCK, bdim: (10, 8), size: Some(16), ty: Compressed},
    ASTC_10x10UnormBlock => {vk: ASTC_10X10_UNORM_BLOCK, bdim: (10, 10), size: Some(16), ty: Compressed},
    ASTC_10x10SrgbBlock => {vk: ASTC_10X10_SRGB_BLOCK, bdim: (10, 10), size: Some(16), ty: Compressed},
    ASTC_12x10UnormBlock => {vk: ASTC_12X10_UNORM_BLOCK, bdim: (12, 10), size: Some(16), ty: Compressed},
    ASTC_12x10SrgbBlock => {vk: ASTC_12X10_SRGB_BLOCK, bdim: (12, 10), size: Some(16), ty: Compressed},
    ASTC_12x12UnormBlock => {vk: ASTC_12X12_UNORM_BLOCK, bdim: (12, 12), size: Some(16), ty: Compressed},
    ASTC_12x12SrgbBlock => {vk: ASTC_12X12_SRGB_BLOCK, bdim: (12, 12), size: Some(16), ty: Compressed},
    G8B8R8_3PLANE420Unorm => {vk: G8_B8_R8_3PLANE_420_UNORM, bdim: (1, 1), size: None, ty: Ycbcr, planes: 3},
    G8B8R8_2PLANE420Unorm => {vk: G8_B8R8_2PLANE_420_UNORM, bdim: (1, 1), size: None, ty: Ycbcr, planes: 2},
}

impl Format {
    /// Returns the aspects that images of this format have.
    #[inline]
    pub const fn aspects(&self) -> ImageAspects {
        let ty = self.ty();
        let planes = self.planes();
        ImageAspects {
            color: matches!(
                ty,
                FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed
            ),
            depth: matches!(ty, FormatTy::Depth | FormatTy::DepthStencil),
            stencil: matches!(ty, FormatTy::Stencil | FormatTy::DepthStencil),
            plane0: planes >= 1,
            plane1: planes >= 2,
            plane2: planes >= 3,
            ..ImageAspects::none()
        }
    }

    /// Retrieves the properties of a format when used by a certain device.
    #[inline]
    pub fn properties(&self, physical_device: PhysicalDevice) -> FormatProperties {
        let vk_properties = unsafe {
            let fns_i = physical_device.instance().fns();
            let mut output = MaybeUninit::uninit();
            fns_i.v1_0.get_physical_device_format_properties(
                physical_device.internal_object(),
                (*self).into(),
                output.as_mut_ptr(),
            );
            output.assume_init()
        };

        FormatProperties {
            linear_tiling_features: vk_properties.linear_tiling_features.into(),
            optimal_tiling_features: vk_properties.optimal_tiling_features.into(),
            buffer_features: vk_properties.buffer_features.into(),
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

/// Trait for Rust types that can represent a pixel in an image.
pub unsafe trait Pixel {
    /// Returns an error if `Self` cannot be used as a source of pixels for `format`.
    fn ensure_accepts(format: Format) -> Result<(), IncompatiblePixelsType>;

    /// The number of `Self`s which make up a single pixel.
    ///
    /// # Panics
    ///
    /// May panic if `ensure_accepts` would not return `Ok(())`.
    fn rate(format: Format) -> u32;
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
        unsafe impl Pixel for $ty {
            fn ensure_accepts(format: Format) -> Result<(), IncompatiblePixelsType> {
                // TODO: Be more strict: accept only if the format has a matching AcceptsPixels impl.
                if format.size().map_or(false, |x| x % mem::size_of::<$ty>() as DeviceSize == 0) {
                    Ok(())
                } else {
                    Err(IncompatiblePixelsType)
                }
            }

            fn rate(format: Format) -> u32 {
                (format.size().expect("this format cannot accept pixels") / mem::size_of::<$ty>() as DeviceSize) as u32
            }
        }
    }
}

impl_pixel! {
    u8; i8; u16; i16; u32; i32; u64; i64; f16; f32; f64;
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

impl From<ash::vk::FormatFeatureFlags> for FormatFeatures {
    #[inline]
    #[rustfmt::skip]
    fn from(val: ash::vk::FormatFeatureFlags) -> FormatFeatures {
        FormatFeatures {
            sampled_image: !(val & ash::vk::FormatFeatureFlags::SAMPLED_IMAGE).is_empty(),
            storage_image: !(val & ash::vk::FormatFeatureFlags::STORAGE_IMAGE).is_empty(),
            storage_image_atomic: !(val & ash::vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC).is_empty(),
            uniform_texel_buffer: !(val & ash::vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER).is_empty(),
            storage_texel_buffer: !(val & ash::vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER).is_empty(),
            storage_texel_buffer_atomic: !(val & ash::vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER_ATOMIC).is_empty(),
            vertex_buffer: !(val & ash::vk::FormatFeatureFlags::VERTEX_BUFFER).is_empty(),
            color_attachment: !(val & ash::vk::FormatFeatureFlags::COLOR_ATTACHMENT).is_empty(),
            color_attachment_blend: !(val & ash::vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND).is_empty(),
            depth_stencil_attachment: !(val & ash::vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT).is_empty(),
            blit_src: !(val & ash::vk::FormatFeatureFlags::BLIT_SRC).is_empty(),
            blit_dst: !(val & ash::vk::FormatFeatureFlags::BLIT_DST).is_empty(),
            sampled_image_filter_linear: !(val & ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR).is_empty(),
            transfer_src: !(val & ash::vk::FormatFeatureFlags::TRANSFER_SRC).is_empty(),
            transfer_dst: !(val & ash::vk::FormatFeatureFlags::TRANSFER_DST).is_empty(),
            midpoint_chroma_samples: !(val & ash::vk::FormatFeatureFlags::MIDPOINT_CHROMA_SAMPLES).is_empty(),
            sampled_image_ycbcr_conversion_linear_filter: !(val & ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER).is_empty(),
            sampled_image_ycbcr_conversion_separate_reconstruction_filter: !(val & ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER).is_empty(),
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: !(val & ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT).is_empty(),
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: !(val & ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE).is_empty(),
            disjoint: !(val & ash::vk::FormatFeatureFlags::DISJOINT).is_empty(),
            cosited_chroma_samples: !(val & ash::vk::FormatFeatureFlags::COSITED_CHROMA_SAMPLES).is_empty(),
            sampled_image_filter_minmax: !(val & ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_MINMAX).is_empty(),
            img_sampled_image_filter_cubic: !(val & ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_CUBIC_IMG).is_empty(),
            khr_acceleration_structure_vertex_buffer: !(val & ash::vk::FormatFeatureFlags::ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR).is_empty(),
            ext_fragment_density_map: !(val & ash::vk::FormatFeatureFlags::FRAGMENT_DENSITY_MAP_EXT).is_empty(),
        }
    }
}
