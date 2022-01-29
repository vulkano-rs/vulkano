// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! All the formats supported by Vulkan.
//!
//! A format is mostly used to describe the texel data of an image. However, formats also show up in
//! a few other places, most notably to describe the format of vertex buffers.
//!
//! # Format support
//!
//! Not all formats are supported by every device. Those that devices do support may only be
//! supported for certain use cases. It is an error to use a format where it is not supported, but
//! you can query a device beforehand for its support by calling the `properties` method on a format
//! value. You can use this to select a usable format from one or more suitable alternatives.
//! Some formats are required to be always supported for a particular usage. These are listed in the
//! [tables in the Vulkan specification](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap43.html#features-required-format-support).
//!
//! # Special format types
//!
//! ## Depth/stencil formats
//!
//! Depth/stencil formats can be identified by the `D` and `S` components in their names. They are
//! used primarily as the format for framebuffer attachments, for the purposes of depth and stencil
//! testing.
//!
//! Some formats have only a depth or stencil component, while others combine both. The two
//! components are represented as separate *aspects*, which means that they can be accessed
//! individually as separate images. These pseudo-images have the same resolution, but different
//! bit depth and numeric representation.
//!
//! Depth/stencil formats deviate from the others in a few more ways. Their data representation is
//! considered opaque, meaning that they do not have a fixed layout in memory nor a fixed size per
//! texel. They also have special limitations in several operations such as copying; a depth/stencil
//! format is not compatible with any other format, only with itself.
//!
//! ## Block-compressed formats
//!
//! A block-compressed format uses compression to encode a larger block of texels into a smaller
//! number of bytes. Individual texels are no longer represented in memory, only the block as a
//! whole. An image must consist of a whole number of blocks, so the dimensions of an image must be
//! a whole multiple of the block dimensions. Vulkan supports several different compression schemes,
//! represented in Vulkano by the `CompressionType` enum.
//!
//! Overall, block-compressed formats do not behave significantly differently from regular formats.
//! They are mostly restricted in terms of compatibility. Because of the compression, the notion of
//! bits per component does not apply, so the `components` method will only return whether a
//! component is present or not.
//!
//! ## YCbCr formats
//!
//! YCbCr, also known as YUV, is an alternative image representation with three components:
//! Y for luminance or *luma* (overall brightness) and two color or *chroma* components Cb and Cr
//! encoding the blueness and redness respectively. YCbCr formats are primarily used in video
//! applications. In Vulkan, the formats used to encode YCbCr data use the green channel to
//! represent the luma component, while the blue and red components hold the chroma.
//!
//! To use most YCbCr formats in an [image view](crate::image::view), a feature known as
//! *sampler YCbCr conversion* is needed. It must be enabled on both the image view and any
//! combined image samplers in shaders that the image view is attached to. This feature handles
//! the correct conversion between YCbCr input data and RGB data inside the shader. To query whether
//! a format requires the conversion, you can call `requires_sampler_ycbcr_conversion` on a format.
//! As a rule, any format with `444`, `422`, `420`, `3PACK` or `4PACK` in the name requires it.
//!
//! Almost all YCbCr formats make use of **chroma subsampling**. This is a technique whereby the two
//! chroma components are encoded using a lower resolution than the luma component. The human eye is
//! less sensitive to color detail than to detail in brightness, so this allows more detail to be
//! encoded in less data. Chroma subsampling is indicated with one of three numbered suffixes in a
//! format name:
//! - `444` indicates a YCbCr format without chroma subsampling. All components have the same
//!   resolution.
//! - `422` indicates horizontal chroma subsampling. The horizontal resolution of the chroma
//!   components is halved, so a single value is shared within a 2x1 block of texels.
//! - `420` indicates horizontal and vertical chroma subsampling. Both dimensions of the chroma
//!   components are halved, so a single value is shared within a 2x2 block of texels.
//!
//! Most YCbCr formats, including all of the `444` and `420` formats, are **multi-planar**. Instead
//! of storing the components of a single texel together in memory, the components are separated
//! into *planes*, which act like independent images. In 3-plane formats, the planes hold the Y,
//! Cb and Cr components respectively, while in 2-plane formats, Cb and Cr are combined into a
//! two-component plane. Where chroma subsampling is applied, plane 0 has the full resolution, while
//! planes 1 and 2 have reduced resolution. Effectively, they are standalone images with half the
//! resolution of the original.
//!
//! The texels of multi-planar images cannot be accessed individually, for example to copy or blit,
//! since the components of each texel are split across the planes. Instead, you must access each
//! plane as an individual *aspect* of the image. A single-plane aspect of a multi-planar image
//! behaves as a regular image, and even has its own format, which can be queried with the `plane`
//! method on a format.

use crate::device::physical::FormatProperties;
use crate::device::physical::PhysicalDevice;
use crate::image::ImageAspects;
use crate::shader::spirv::ImageFormat;
use crate::DeviceSize;
use half::f16;
use std::vec::IntoIter as VecIntoIter;
use std::{error, fmt, mem};

// Generated by build.rs
include!(concat!(env!("OUT_DIR"), "/formats.rs"));

impl Format {
    /// Retrieves the properties of a format when used by a certain device.
    #[deprecated(since = "0.28", note = "Use PhysicalDevice::format_properties instead")]
    #[inline]
    pub fn properties(&self, physical_device: PhysicalDevice) -> FormatProperties {
        physical_device.format_properties(*self)
    }

    /// Returns whether the format can be used with a storage image, without specifying
    /// the format in the shader, if the
    /// [`shader_storage_image_read_without_format`](crate::device::Features::shader_storage_image_read_without_format)
    /// and/or
    /// [`shader_storage_image_write_without_format`](crate::device::Features::shader_storage_image_write_without_format)
    /// features are enabled on the device.
    #[inline]
    pub fn shader_storage_image_without_format(&self) -> bool {
        matches!(
            *self,
            Format::R8G8B8A8_UNORM
                | Format::R8G8B8A8_SNORM
                | Format::R8G8B8A8_UINT
                | Format::R8G8B8A8_SINT
                | Format::R32_UINT
                | Format::R32_SINT
                | Format::R32_SFLOAT
                | Format::R32G32_UINT
                | Format::R32G32_SINT
                | Format::R32G32_SFLOAT
                | Format::R32G32B32A32_UINT
                | Format::R32G32B32A32_SINT
                | Format::R32G32B32A32_SFLOAT
                | Format::R16G16B16A16_UINT
                | Format::R16G16B16A16_SINT
                | Format::R16G16B16A16_SFLOAT
                | Format::R16G16_SFLOAT
                | Format::B10G11R11_UFLOAT_PACK32
                | Format::R16_SFLOAT
                | Format::R16G16B16A16_UNORM
                | Format::A2B10G10R10_UNORM_PACK32
                | Format::R16G16_UNORM
                | Format::R8G8_UNORM
                | Format::R16_UNORM
                | Format::R8_UNORM
                | Format::R16G16B16A16_SNORM
                | Format::R16G16_SNORM
                | Format::R8G8_SNORM
                | Format::R16_SNORM
                | Format::R8_SNORM
                | Format::R16G16_SINT
                | Format::R8G8_SINT
                | Format::R16_SINT
                | Format::R8_SINT
                | Format::A2B10G10R10_UINT_PACK32
                | Format::R16G16_UINT
                | Format::R8G8_UINT
                | Format::R16_UINT
                | Format::R8_UINT
        )
    }

    #[inline]
    pub fn decode_clear_value(&self, value: ClearValue) -> ClearValue {
        let aspects = self.aspects();

        if aspects.depth && aspects.stencil {
            assert!(matches!(value, ClearValue::DepthStencil(_)));
        } else if aspects.depth {
            assert!(matches!(value, ClearValue::Depth(_)));
        } else if aspects.stencil {
            assert!(matches!(value, ClearValue::Stencil(_)));
        } else if let Some(numeric_type) = self.type_color() {
            match numeric_type {
                NumericType::SFLOAT
                | NumericType::UFLOAT
                | NumericType::SNORM
                | NumericType::UNORM
                | NumericType::SSCALED
                | NumericType::USCALED
                | NumericType::SRGB => {
                    assert!(matches!(value, ClearValue::Float(_)));
                }
                NumericType::SINT => {
                    assert!(matches!(value, ClearValue::Int(_)));
                }
                NumericType::UINT => {
                    assert!(matches!(value, ClearValue::Uint(_)));
                }
            }
        } else {
            panic!("Shouldn't happen!");
        }

        value
    }
}

impl From<Format> for ash::vk::Format {
    #[inline]
    fn from(val: Format) -> Self {
        ash::vk::Format::from_raw(val as i32)
    }
}

// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap46.html#spirvenv-image-formats
impl From<ImageFormat> for Option<Format> {
    fn from(val: ImageFormat) -> Self {
        match val {
            ImageFormat::Unknown => None,
            ImageFormat::Rgba32f => Some(Format::R32G32B32A32_SFLOAT),
            ImageFormat::Rgba16f => Some(Format::R16G16B16A16_SFLOAT),
            ImageFormat::R32f => Some(Format::R32_SFLOAT),
            ImageFormat::Rgba8 => Some(Format::R8G8B8A8_UNORM),
            ImageFormat::Rgba8Snorm => Some(Format::R8G8B8A8_SNORM),
            ImageFormat::Rg32f => Some(Format::R32G32_SFLOAT),
            ImageFormat::Rg16f => Some(Format::R16G16_SFLOAT),
            ImageFormat::R11fG11fB10f => Some(Format::B10G11R11_UFLOAT_PACK32),
            ImageFormat::R16f => Some(Format::R16_SFLOAT),
            ImageFormat::Rgba16 => Some(Format::R16G16B16A16_UNORM),
            ImageFormat::Rgb10A2 => Some(Format::A2B10G10R10_UNORM_PACK32),
            ImageFormat::Rg16 => Some(Format::R16G16_UNORM),
            ImageFormat::Rg8 => Some(Format::R8G8_UNORM),
            ImageFormat::R16 => Some(Format::R16_UNORM),
            ImageFormat::R8 => Some(Format::R8_UNORM),
            ImageFormat::Rgba16Snorm => Some(Format::R16G16B16A16_SNORM),
            ImageFormat::Rg16Snorm => Some(Format::R16G16_SNORM),
            ImageFormat::Rg8Snorm => Some(Format::R8G8_SNORM),
            ImageFormat::R16Snorm => Some(Format::R16_SNORM),
            ImageFormat::R8Snorm => Some(Format::R8_SNORM),
            ImageFormat::Rgba32i => Some(Format::R32G32B32A32_SINT),
            ImageFormat::Rgba16i => Some(Format::R16G16B16A16_SINT),
            ImageFormat::Rgba8i => Some(Format::R8G8B8A8_SINT),
            ImageFormat::R32i => Some(Format::R32_SINT),
            ImageFormat::Rg32i => Some(Format::R32G32_SINT),
            ImageFormat::Rg16i => Some(Format::R16G16_SINT),
            ImageFormat::Rg8i => Some(Format::R8G8_SINT),
            ImageFormat::R16i => Some(Format::R16_SINT),
            ImageFormat::R8i => Some(Format::R8_SINT),
            ImageFormat::Rgba32ui => Some(Format::R32G32B32A32_UINT),
            ImageFormat::Rgba16ui => Some(Format::R16G16B16A16_UINT),
            ImageFormat::Rgba8ui => Some(Format::R8G8B8A8_UINT),
            ImageFormat::R32ui => Some(Format::R32_UINT),
            ImageFormat::Rgb10a2ui => Some(Format::A2B10G10R10_UINT_PACK32),
            ImageFormat::Rg32ui => Some(Format::R32G32_UINT),
            ImageFormat::Rg16ui => Some(Format::R16G16_UINT),
            ImageFormat::Rg8ui => Some(Format::R8G8_UINT),
            ImageFormat::R16ui => Some(Format::R16_UINT),
            ImageFormat::R8ui => Some(Format::R8_UINT),
            ImageFormat::R64ui => Some(Format::R64_UINT),
            ImageFormat::R64i => Some(Format::R64_SINT),
        }
    }
}

/// The block compression scheme used in a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum CompressionType {
    /// Adaptive Scalable Texture Compression, low dynamic range.
    ASTC_LDR,
    /// Adaptive Scalable Texture Compression, high dynamic range.
    ASTC_HDR,
    /// S3TC Block Compression.
    BC,
    /// Ericsson Texture Compression 2.
    ETC2,
    /// ETC2 Alpha Compression.
    EAC,
    /// PowerVR Texture Compression.
    PVRTC,
}

/// For YCbCr formats, the type of chroma sampling used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChromaSampling {
    /// The chroma components are represented at the same resolution as the luma component.
    Mode444,
    /// The chroma components have half the horizontal resolution as the luma component.
    Mode422,
    /// The chroma components have half the horizontal and vertical resolution as the luma
    /// component.
    Mode420,
}

/// The numeric type that represents data of a format in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericType {
    /// Signed floating-point number.
    SFLOAT,
    /// Unsigned floating-point number.
    UFLOAT,
    /// Signed integer.
    SINT,
    /// Unsigned integer.
    UINT,
    /// Signed integer that represents a normalized floating-point value in the range \[-1,1].
    SNORM,
    /// Unsigned integer that represents a normalized floating-point value in the range \[0,1].
    UNORM,
    /// Signed integer that is converted to a floating-point value directly.
    SSCALED,
    /// Unsigned integer that is converted to a floating-point value directly.
    USCALED,
    /// Unsigned integer where R, G, B components represent a normalized floating-point value in the
    /// sRGB color space, while the A component is a simple normalized value as in `UNORM`.
    SRGB,
}

/// An opaque type that represents a format compatibility class.
///
/// Two formats are compatible if their compatibility classes compare equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FormatCompatibility(pub(crate) &'static FormatCompatibilityInner);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub(crate) enum FormatCompatibilityInner {
    Class_8bit,
    Class_16bit,
    Class_24bit,
    Class_32bit,
    Class_48bit,
    Class_64bit,
    Class_96bit,
    Class_128bit,
    Class_192bit,
    Class_256bit,
    Class_D16,
    Class_D24,
    Class_D32,
    Class_S8,
    Class_D16S8,
    Class_D24S8,
    Class_D32S8,
    Class_64bit_R10G10B10A10,
    Class_64bit_R12G12B12A12,
    Class_BC1_RGB,
    Class_BC1_RGBA,
    Class_BC2,
    Class_BC3,
    Class_BC4,
    Class_BC5,
    Class_BC6H,
    Class_BC7,
    Class_ETC2_RGB,
    Class_ETC2_RGBA,
    Class_ETC2_EAC_RGBA,
    Class_EAC_R,
    Class_EAC_RG,
    Class_ASTC_4x4,
    Class_ASTC_5x4,
    Class_ASTC_5x5,
    Class_ASTC_6x5,
    Class_ASTC_6x6,
    Class_ASTC_8x5,
    Class_ASTC_8x6,
    Class_ASTC_8x8,
    Class_ASTC_10x5,
    Class_ASTC_10x6,
    Class_ASTC_10x8,
    Class_ASTC_10x10,
    Class_ASTC_12x10,
    Class_ASTC_12x12,
    Class_PVRTC1_2BPP,
    Class_PVRTC1_4BPP,
    Class_PVRTC2_2BPP,
    Class_PVRTC2_4BPP,
    Class_32bit_G8B8G8R8,
    Class_32bit_B8G8R8G8,
    Class_64bit_G10B10G10R10,
    Class_64bit_B10G10R10G10,
    Class_64bit_G12B12G12R12,
    Class_64bit_B12G12R12G12,
    Class_64bit_G16B16G16R16,
    Class_64bit_B16G16R16G16,
    Class_8bit_3plane_420,
    Class_8bit_2plane_420,
    Class_10bit_3plane_420,
    Class_10bit_2plane_420,
    Class_12bit_3plane_420,
    Class_12bit_2plane_420,
    Class_16bit_3plane_420,
    Class_16bit_2plane_420,
    Class_8bit_3plane_422,
    Class_8bit_2plane_422,
    Class_10bit_3plane_422,
    Class_10bit_2plane_422,
    Class_12bit_3plane_422,
    Class_12bit_2plane_422,
    Class_16bit_3plane_422,
    Class_16bit_2plane_422,
    Class_8bit_3plane_444,
    Class_10bit_3plane_444,
    Class_12bit_3plane_444,
    Class_16bit_3plane_444,
    Class_8bit_2plane_444,
    Class_10bit_2plane_444,
    Class_12bit_2plane_444,
    Class_16bit_2plane_444,
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
                if format.block_size().map_or(false, |x| x % mem::size_of::<$ty>() as DeviceSize == 0) {
                    Ok(())
                } else {
                    Err(IncompatiblePixelsType)
                }
            }

            fn rate(format: Format) -> u32 {
                (format.block_size().expect("this format cannot accept pixels") / mem::size_of::<$ty>() as DeviceSize) as u32
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
    /// Value for floating-point attachments, including `UNORM`, `SNORM`, `SFLOAT`.
    Float([f32; 4]),
    /// Value for integer attachments, including `SINT`.
    Int([i32; 4]),
    /// Value for unsigned integer attachments, including `UINT`.
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
