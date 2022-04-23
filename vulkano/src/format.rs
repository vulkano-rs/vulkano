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
//! you can query a device beforehand for its support by calling `format_properties` on the physical
//! device. You can use this to select a usable format from one or more suitable alternatives.
//! Some formats are required to be always supported for a particular usage. These are listed in the
//! [tables in the Vulkan specification](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap43.html#features-required-format-support).
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
//! To use most YCbCr formats in an [image view](crate::image::view), a
//! [sampler YCbCr conversion](crate::sampler::ycbcr) object must be created, and attached to both
//! the image view and the sampler. To query whether a format requires the conversion, you can call
//! `ycbcr_chroma_sampling` on a format. As a rule, any format with `444`, `422`, `420`,
//! `3PACK` or `4PACK` in the name requires it.
//!
//! Many YCbCr formats make use of **chroma subsampling**. This is a technique whereby the two
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

use crate::{
    device::physical::PhysicalDevice, image::ImageAspects, shader::spirv::ImageFormat, DeviceSize,
};
use std::ops::BitOr;

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
}

impl From<Format> for ash::vk::Format {
    #[inline]
    fn from(val: Format) -> Self {
        ash::vk::Format::from_raw(val as i32)
    }
}

// https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap46.html#spirvenv-image-formats
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

impl ChromaSampling {
    pub fn subsampled_extent(&self, mut extent: [u32; 3]) -> [u32; 3] {
        match self {
            ChromaSampling::Mode444 => (),
            ChromaSampling::Mode422 => {
                debug_assert!(extent[0] % 2 == 0);
                extent[0] /= 2;
            }
            ChromaSampling::Mode420 => {
                debug_assert!(extent[0] % 2 == 0 && extent[1] % 2 == 0);
                extent[0] /= 2;
                extent[1] /= 2;
            }
        }

        extent
    }
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

/// Describes a uniform value that will be used to fill an image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearValue {
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

impl From<ClearValue> for ash::vk::ClearValue {
    #[inline]
    fn from(val: ClearValue) -> Self {
        match val {
            ClearValue::Float(val) => Self {
                color: ash::vk::ClearColorValue { float32: val },
            },
            ClearValue::Int(val) => Self {
                color: ash::vk::ClearColorValue { int32: val },
            },
            ClearValue::Uint(val) => Self {
                color: ash::vk::ClearColorValue { uint32: val },
            },
            ClearValue::Depth(depth) => Self {
                depth_stencil: ash::vk::ClearDepthStencilValue { depth, stencil: 0 },
            },
            ClearValue::Stencil(stencil) => Self {
                depth_stencil: ash::vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil,
                },
            },
            ClearValue::DepthStencil((depth, stencil)) => Self {
                depth_stencil: ash::vk::ClearDepthStencilValue { depth, stencil },
            },
        }
    }
}

impl From<ClearColorValue> for ClearValue {
    #[inline]
    fn from(val: ClearColorValue) -> Self {
        match val {
            ClearColorValue::Float(val) => Self::Float(val),
            ClearColorValue::Int(val) => Self::Int(val),
            ClearColorValue::Uint(val) => Self::Uint(val),
        }
    }
}

impl From<[f32; 1]> for ClearValue {
    #[inline]
    fn from(val: [f32; 1]) -> Self {
        Self::Float([val[0], 0.0, 0.0, 1.0])
    }
}

impl From<[f32; 2]> for ClearValue {
    #[inline]
    fn from(val: [f32; 2]) -> Self {
        Self::Float([val[0], val[1], 0.0, 1.0])
    }
}

impl From<[f32; 3]> for ClearValue {
    #[inline]
    fn from(val: [f32; 3]) -> Self {
        Self::Float([val[0], val[1], val[2], 1.0])
    }
}

impl From<[f32; 4]> for ClearValue {
    #[inline]
    fn from(val: [f32; 4]) -> Self {
        Self::Float(val)
    }
}

impl From<[u32; 1]> for ClearValue {
    #[inline]
    fn from(val: [u32; 1]) -> Self {
        Self::Uint([val[0], 0, 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 2]> for ClearValue {
    #[inline]
    fn from(val: [u32; 2]) -> Self {
        Self::Uint([val[0], val[1], 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 3]> for ClearValue {
    #[inline]
    fn from(val: [u32; 3]) -> Self {
        Self::Uint([val[0], val[1], val[2], 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 4]> for ClearValue {
    #[inline]
    fn from(val: [u32; 4]) -> Self {
        Self::Uint(val)
    }
}

impl From<[i32; 1]> for ClearValue {
    #[inline]
    fn from(val: [i32; 1]) -> Self {
        Self::Int([val[0], 0, 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 2]> for ClearValue {
    #[inline]
    fn from(val: [i32; 2]) -> Self {
        Self::Int([val[0], val[1], 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 3]> for ClearValue {
    #[inline]
    fn from(val: [i32; 3]) -> Self {
        Self::Int([val[0], val[1], val[2], 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 4]> for ClearValue {
    #[inline]
    fn from(val: [i32; 4]) -> Self {
        Self::Int(val)
    }
}

impl From<f32> for ClearValue {
    #[inline]
    fn from(val: f32) -> Self {
        Self::Depth(val)
    }
}

impl From<u32> for ClearValue {
    #[inline]
    fn from(val: u32) -> Self {
        Self::Stencil(val)
    }
}

impl From<(f32, u32)> for ClearValue {
    #[inline]
    fn from(val: (f32, u32)) -> Self {
        Self::DepthStencil(val)
    }
}

/// A value that will be used to clear a color image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearColorValue {
    /// Value for formats with a numeric type that is not `SINT` or `UINT`.
    Float([f32; 4]),
    /// Value for formats with a numeric type of `SINT`.
    Int([i32; 4]),
    /// Value for formats with a numeric type of `UINT`.
    Uint([u32; 4]),
}

impl From<ClearColorValue> for ash::vk::ClearColorValue {
    #[inline]
    fn from(val: ClearColorValue) -> Self {
        match val {
            ClearColorValue::Float(float32) => Self { float32 },
            ClearColorValue::Int(int32) => Self { int32 },
            ClearColorValue::Uint(uint32) => Self { uint32 },
        }
    }
}

impl From<[f32; 1]> for ClearColorValue {
    #[inline]
    fn from(val: [f32; 1]) -> Self {
        Self::Float([val[0], 0.0, 0.0, 1.0])
    }
}

impl From<[f32; 2]> for ClearColorValue {
    #[inline]
    fn from(val: [f32; 2]) -> Self {
        Self::Float([val[0], val[1], 0.0, 1.0])
    }
}

impl From<[f32; 3]> for ClearColorValue {
    #[inline]
    fn from(val: [f32; 3]) -> Self {
        Self::Float([val[0], val[1], val[2], 1.0])
    }
}

impl From<[f32; 4]> for ClearColorValue {
    #[inline]
    fn from(val: [f32; 4]) -> Self {
        Self::Float(val)
    }
}

impl From<[i32; 1]> for ClearColorValue {
    #[inline]
    fn from(val: [i32; 1]) -> Self {
        Self::Int([val[0], 0, 0, 1])
    }
}

impl From<[i32; 2]> for ClearColorValue {
    #[inline]
    fn from(val: [i32; 2]) -> Self {
        Self::Int([val[0], val[1], 0, 1])
    }
}

impl From<[i32; 3]> for ClearColorValue {
    #[inline]
    fn from(val: [i32; 3]) -> Self {
        Self::Int([val[0], val[1], val[2], 1])
    }
}

impl From<[i32; 4]> for ClearColorValue {
    #[inline]
    fn from(val: [i32; 4]) -> Self {
        Self::Int(val)
    }
}

impl From<[u32; 1]> for ClearColorValue {
    #[inline]
    fn from(val: [u32; 1]) -> Self {
        Self::Uint([val[0], 0, 0, 1])
    }
}

impl From<[u32; 2]> for ClearColorValue {
    #[inline]
    fn from(val: [u32; 2]) -> Self {
        Self::Uint([val[0], val[1], 0, 1])
    }
}

impl From<[u32; 3]> for ClearColorValue {
    #[inline]
    fn from(val: [u32; 3]) -> Self {
        Self::Uint([val[0], val[1], val[2], 1])
    }
}

impl From<[u32; 4]> for ClearColorValue {
    #[inline]
    fn from(val: [u32; 4]) -> Self {
        Self::Uint(val)
    }
}

/// A value that will be used to clear a depth/stencil image.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClearDepthStencilValue {
    /// Value for the depth component.
    pub depth: f32,
    /// Value for the stencil component.
    pub stencil: u32,
}

impl From<ClearDepthStencilValue> for ash::vk::ClearDepthStencilValue {
    #[inline]
    fn from(val: ClearDepthStencilValue) -> Self {
        Self {
            depth: val.depth,
            stencil: val.stencil,
        }
    }
}

impl From<f32> for ClearDepthStencilValue {
    #[inline]
    fn from(depth: f32) -> Self {
        Self { depth, stencil: 0 }
    }
}

impl From<u32> for ClearDepthStencilValue {
    #[inline]
    fn from(stencil: u32) -> Self {
        Self {
            depth: 0.0,
            stencil,
        }
    }
}

impl From<(f32, u32)> for ClearDepthStencilValue {
    #[inline]
    fn from((depth, stencil): (f32, u32)) -> Self {
        Self { depth, stencil }
    }
}

/// The properties of a format that are supported by a physical device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FormatProperties {
    /// Features available for images with linear tiling.
    pub linear_tiling_features: FormatFeatures,

    /// Features available for images with optimal tiling.
    pub optimal_tiling_features: FormatFeatures,

    /// Features available for buffers.
    pub buffer_features: FormatFeatures,

    pub _ne: crate::NonExhaustive,
}

impl FormatProperties {
    /// Returns the potential format features, following the definition of
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap43.html#potential-format-features>.
    #[inline]
    pub fn potential_format_features(&self) -> FormatFeatures {
        &self.linear_tiling_features | &self.optimal_tiling_features
    }
}

/// The features supported by a device for an image or buffer with a particular format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(missing_docs)]
pub struct FormatFeatures {
    // Image usage
    /// Can be used with a sampled image descriptor.
    pub sampled_image: bool,
    /// Can be used with a storage image descriptor.
    pub storage_image: bool,
    /// Can be used with a storage image descriptor with atomic operations in a shader.
    pub storage_image_atomic: bool,
    /// Can be used with a storage image descriptor for reading, without specifying a format on the
    /// image view.
    pub storage_read_without_format: bool,
    /// Can be used with a storage image descriptor for writing, without specifying a format on the
    /// image view.
    pub storage_write_without_format: bool,
    /// Can be used with a color attachment in a framebuffer, or with an input attachment
    /// descriptor.
    pub color_attachment: bool,
    /// Can be used with a color attachment in a framebuffer with blending, or with an input
    /// attachment descriptor.
    pub color_attachment_blend: bool,
    /// Can be used with a depth/stencil attachment in a framebuffer, or with an input attachment
    /// descriptor.
    pub depth_stencil_attachment: bool,
    /// Can be used with a fragment density map attachment in a framebuffer.
    pub fragment_density_map: bool,
    /// Can be used with a fragment shading rate attachment in a framebuffer.
    pub fragment_shading_rate_attachment: bool,
    /// Can be used with the source image in a transfer (copy) operation.
    pub transfer_src: bool,
    /// Can be used with the destination image in a transfer (copy) operation.
    pub transfer_dst: bool,
    /// Can be used with the source image in a blit operation.
    pub blit_src: bool,
    /// Can be used with the destination image in a blit operation.
    pub blit_dst: bool,

    // Sampling
    /// Can be used with samplers or as a blit source, using the
    /// [`Linear`](crate::sampler::Filter::Linear) filter.
    pub sampled_image_filter_linear: bool,
    /// Can be used with samplers or as a blit source, using the
    /// [`Cubic`](crate::sampler::Filter::Cubic) filter.
    pub sampled_image_filter_cubic: bool,
    /// Can be used with samplers using a reduction mode of
    /// [`Min`](crate::sampler::SamplerReductionMode::Min) or
    /// [`Max`](crate::sampler::SamplerReductionMode::Max).
    pub sampled_image_filter_minmax: bool,
    /// Can be used with sampler YCbCr conversions using a chroma offset of
    /// [`Midpoint`](crate::sampler::ycbcr::ChromaLocation::Midpoint).
    pub midpoint_chroma_samples: bool,
    /// Can be used with sampler YCbCr conversions using a chroma offset of
    /// [`CositedEven`](crate::sampler::ycbcr::ChromaLocation::CositedEven).
    pub cosited_chroma_samples: bool,
    /// Can be used with sampler YCbCr conversions using the
    /// [`Linear`](crate::sampler::Filter::Linear) chroma filter.
    pub sampled_image_ycbcr_conversion_linear_filter: bool,
    /// Can be used with sampler YCbCr conversions whose chroma filter differs from the filters of
    /// the base sampler.
    pub sampled_image_ycbcr_conversion_separate_reconstruction_filter: bool,
    /// When used with a sampler YCbCr conversion, the implementation will always perform
    /// explicit chroma reconstruction.
    pub sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: bool,
    /// Can be used with sampler YCbCr conversions with forced explicit reconstruction.
    pub sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: bool,
    /// Can be used with samplers using depth comparison.
    pub sampled_image_depth_comparison: bool,

    // Video
    /// Can be used with the output image of a video decode operation.
    pub video_decode_output: bool,
    /// Can be used with the DPB image of a video decode operation.
    pub video_decode_dpb: bool,
    /// Can be used with the input image of a video encode operation.
    pub video_encode_input: bool,
    /// Can be used with the DPB image of a video encode operation.
    pub video_encode_dpb: bool,

    // Misc image features
    /// For multi-planar formats, can be used with images created with the `disjoint` flag.
    pub disjoint: bool,

    // Buffer usage
    /// Can be used with a uniform texel buffer descriptor.
    pub uniform_texel_buffer: bool,
    /// Can be used with a storage texel buffer descriptor.
    pub storage_texel_buffer: bool,
    /// Can be used with a storage texel buffer descriptor with atomic operations in a shader.
    pub storage_texel_buffer_atomic: bool,
    /// Can be used as the format of a vertex attribute in the vertex input state of a graphics
    /// pipeline.
    pub vertex_buffer: bool,
    /// Can be used with the vertex buffer of an acceleration structure.
    pub acceleration_structure_vertex_buffer: bool,

    pub _ne: crate::NonExhaustive,
}

impl BitOr for &FormatFeatures {
    type Output = FormatFeatures;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::Output {
            sampled_image: self.sampled_image || rhs.sampled_image,
            storage_image: self.storage_image || rhs.storage_image,
            storage_image_atomic: self.storage_image_atomic || rhs.storage_image_atomic,
            storage_read_without_format: self.storage_read_without_format
                || rhs.storage_read_without_format,
            storage_write_without_format: self.storage_write_without_format
                || rhs.storage_write_without_format,
            color_attachment: self.color_attachment || rhs.color_attachment,
            color_attachment_blend: self.color_attachment_blend || rhs.color_attachment_blend,
            depth_stencil_attachment: self.depth_stencil_attachment || rhs.depth_stencil_attachment,
            fragment_density_map: self.fragment_density_map || rhs.fragment_density_map,
            fragment_shading_rate_attachment: self.fragment_shading_rate_attachment
                || rhs.fragment_shading_rate_attachment,
            transfer_src: self.transfer_src || rhs.transfer_src,
            transfer_dst: self.transfer_dst || rhs.transfer_dst,
            blit_src: self.blit_src || rhs.blit_src,
            blit_dst: self.blit_dst || rhs.blit_dst,

            sampled_image_filter_linear: self.sampled_image_filter_linear
                || rhs.sampled_image_filter_linear,
            sampled_image_filter_cubic: self.sampled_image_filter_cubic
                || rhs.sampled_image_filter_cubic,
            sampled_image_filter_minmax: self.sampled_image_filter_minmax
                || rhs.sampled_image_filter_minmax,
            midpoint_chroma_samples: self.midpoint_chroma_samples || rhs.midpoint_chroma_samples,
            cosited_chroma_samples: self.cosited_chroma_samples || rhs.cosited_chroma_samples,
            sampled_image_ycbcr_conversion_linear_filter: self
                .sampled_image_ycbcr_conversion_linear_filter
                || rhs.sampled_image_ycbcr_conversion_linear_filter,
            sampled_image_ycbcr_conversion_separate_reconstruction_filter: self
                .sampled_image_ycbcr_conversion_separate_reconstruction_filter
                || rhs.sampled_image_ycbcr_conversion_separate_reconstruction_filter,
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: self
                .sampled_image_ycbcr_conversion_chroma_reconstruction_explicit
                || rhs.sampled_image_ycbcr_conversion_chroma_reconstruction_explicit,
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: self
                .sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable
                || rhs.sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable,
            sampled_image_depth_comparison: self.sampled_image_depth_comparison
                || rhs.sampled_image_depth_comparison,

            video_decode_output: self.video_decode_output || rhs.video_decode_output,
            video_decode_dpb: self.video_decode_dpb || rhs.video_decode_dpb,
            video_encode_input: self.video_encode_input || rhs.video_encode_input,
            video_encode_dpb: self.video_encode_dpb || rhs.video_encode_dpb,

            disjoint: self.disjoint || rhs.disjoint,

            uniform_texel_buffer: self.uniform_texel_buffer || rhs.uniform_texel_buffer,
            storage_texel_buffer: self.storage_texel_buffer || rhs.storage_texel_buffer,
            storage_texel_buffer_atomic: self.storage_texel_buffer_atomic
                || rhs.storage_texel_buffer_atomic,
            vertex_buffer: self.vertex_buffer || rhs.vertex_buffer,
            acceleration_structure_vertex_buffer: self.acceleration_structure_vertex_buffer
                || rhs.acceleration_structure_vertex_buffer,

            _ne: crate::NonExhaustive(()),
        }
    }
}

impl From<ash::vk::FormatFeatureFlags> for FormatFeatures {
    #[inline]
    #[rustfmt::skip]
    fn from(val: ash::vk::FormatFeatureFlags) -> FormatFeatures {
        FormatFeatures {
            sampled_image: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE),
            storage_image: val.intersects(ash::vk::FormatFeatureFlags::STORAGE_IMAGE),
            storage_image_atomic: val.intersects(ash::vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC),
            storage_read_without_format: false, // FormatFeatureFlags2 only
            storage_write_without_format: false, // FormatFeatureFlags2 only
            color_attachment: val.intersects(ash::vk::FormatFeatureFlags::COLOR_ATTACHMENT),
            color_attachment_blend: val.intersects(ash::vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
            depth_stencil_attachment: val.intersects(ash::vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
            fragment_density_map: val.intersects(ash::vk::FormatFeatureFlags::FRAGMENT_DENSITY_MAP_EXT),
            fragment_shading_rate_attachment: val.intersects(ash::vk::FormatFeatureFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR),
            transfer_src: val.intersects(ash::vk::FormatFeatureFlags::TRANSFER_SRC),
            transfer_dst: val.intersects(ash::vk::FormatFeatureFlags::TRANSFER_DST),
            blit_src: val.intersects(ash::vk::FormatFeatureFlags::BLIT_SRC),
            blit_dst: val.intersects(ash::vk::FormatFeatureFlags::BLIT_DST),

            sampled_image_filter_linear: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
            sampled_image_filter_cubic: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_CUBIC_EXT),
            sampled_image_filter_minmax: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_MINMAX),
            midpoint_chroma_samples: val.intersects(ash::vk::FormatFeatureFlags::MIDPOINT_CHROMA_SAMPLES),
            cosited_chroma_samples: val.intersects(ash::vk::FormatFeatureFlags::COSITED_CHROMA_SAMPLES),
            sampled_image_ycbcr_conversion_linear_filter: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER),
            sampled_image_ycbcr_conversion_separate_reconstruction_filter: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER),
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT),
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE),
            sampled_image_depth_comparison: false, // FormatFeatureFlags2 only

            video_decode_output: val.intersects(ash::vk::FormatFeatureFlags::VIDEO_DECODE_OUTPUT_KHR),
            video_decode_dpb: val.intersects(ash::vk::FormatFeatureFlags::VIDEO_DECODE_DPB_KHR),
            video_encode_input: val.intersects(ash::vk::FormatFeatureFlags::VIDEO_ENCODE_INPUT_KHR),
            video_encode_dpb: val.intersects(ash::vk::FormatFeatureFlags::VIDEO_ENCODE_DPB_KHR),

            disjoint: val.intersects(ash::vk::FormatFeatureFlags::DISJOINT),

            uniform_texel_buffer: val.intersects(ash::vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER),
            storage_texel_buffer: val.intersects(ash::vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER),
            storage_texel_buffer_atomic: val.intersects(ash::vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER_ATOMIC),
            vertex_buffer: val.intersects(ash::vk::FormatFeatureFlags::VERTEX_BUFFER),
            acceleration_structure_vertex_buffer: val.intersects(ash::vk::FormatFeatureFlags::ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR),

            _ne: crate::NonExhaustive(()),
        }
    }
}

impl From<ash::vk::FormatFeatureFlags2> for FormatFeatures {
    #[inline]
    #[rustfmt::skip]
    fn from(val: ash::vk::FormatFeatureFlags2) -> FormatFeatures {
        FormatFeatures {
            sampled_image: val.intersects(ash::vk::FormatFeatureFlags2::SAMPLED_IMAGE),
            storage_image: val.intersects(ash::vk::FormatFeatureFlags2::STORAGE_IMAGE),
            storage_image_atomic: val.intersects(ash::vk::FormatFeatureFlags2::STORAGE_IMAGE_ATOMIC),
            storage_read_without_format: val.intersects(ash::vk::FormatFeatureFlags2::STORAGE_READ_WITHOUT_FORMAT),
            storage_write_without_format: val.intersects(ash::vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT),
            color_attachment: val.intersects(ash::vk::FormatFeatureFlags2::COLOR_ATTACHMENT),
            color_attachment_blend: val.intersects(ash::vk::FormatFeatureFlags2::COLOR_ATTACHMENT_BLEND),
            depth_stencil_attachment: val.intersects(ash::vk::FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT),
            fragment_density_map: val.intersects(ash::vk::FormatFeatureFlags2::FRAGMENT_DENSITY_MAP_EXT),
            fragment_shading_rate_attachment: val.intersects(ash::vk::FormatFeatureFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR),
            transfer_src: val.intersects(ash::vk::FormatFeatureFlags2::TRANSFER_SRC),
            transfer_dst: val.intersects(ash::vk::FormatFeatureFlags2::TRANSFER_DST),
            blit_src: val.intersects(ash::vk::FormatFeatureFlags2::BLIT_SRC),
            blit_dst: val.intersects(ash::vk::FormatFeatureFlags2::BLIT_DST),

            sampled_image_filter_linear: val.intersects(ash::vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR),
            sampled_image_filter_cubic: val.intersects(ash::vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_CUBIC_EXT),
            sampled_image_filter_minmax: val.intersects(ash::vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_MINMAX),
            midpoint_chroma_samples: val.intersects(ash::vk::FormatFeatureFlags2::MIDPOINT_CHROMA_SAMPLES),
            cosited_chroma_samples: val.intersects(ash::vk::FormatFeatureFlags2::COSITED_CHROMA_SAMPLES),
            sampled_image_ycbcr_conversion_linear_filter: val.intersects(ash::vk::FormatFeatureFlags2::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER),
            sampled_image_ycbcr_conversion_separate_reconstruction_filter: val.intersects(ash::vk::FormatFeatureFlags2::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER),
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: val.intersects(ash::vk::FormatFeatureFlags2::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT),
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: val.intersects(ash::vk::FormatFeatureFlags2::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE),
            sampled_image_depth_comparison: val.intersects(ash::vk::FormatFeatureFlags2::SAMPLED_IMAGE_DEPTH_COMPARISON),

            video_decode_output: val.intersects(ash::vk::FormatFeatureFlags2::VIDEO_DECODE_OUTPUT_KHR),
            video_decode_dpb: val.intersects(ash::vk::FormatFeatureFlags2::VIDEO_DECODE_DPB_KHR),
            video_encode_input: val.intersects(ash::vk::FormatFeatureFlags2::VIDEO_ENCODE_INPUT_KHR),
            video_encode_dpb: val.intersects(ash::vk::FormatFeatureFlags2::VIDEO_ENCODE_DPB_KHR),

            disjoint: val.intersects(ash::vk::FormatFeatureFlags2::DISJOINT),

            uniform_texel_buffer: val.intersects(ash::vk::FormatFeatureFlags2::UNIFORM_TEXEL_BUFFER),
            storage_texel_buffer: val.intersects(ash::vk::FormatFeatureFlags2::STORAGE_TEXEL_BUFFER),
            storage_texel_buffer_atomic: val.intersects(ash::vk::FormatFeatureFlags2::STORAGE_TEXEL_BUFFER_ATOMIC),
            vertex_buffer: val.intersects(ash::vk::FormatFeatureFlags2::VERTEX_BUFFER),
            acceleration_structure_vertex_buffer: val.intersects(ash::vk::FormatFeatureFlags2::ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR),

            _ne: crate::NonExhaustive(()),
        }
    }
}
