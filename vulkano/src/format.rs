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

use crate::device::physical::PhysicalDevice;
use crate::image::ImageAspects;
use crate::shader::spirv::ImageFormat;
use crate::DeviceSize;
use crate::VulkanObject;
use half::f16;
use std::mem::MaybeUninit;
use std::vec::IntoIter as VecIntoIter;
use std::{error, fmt, mem};

// Generated by build.rs
include!(concat!(env!("OUT_DIR"), "/formats.rs"));

impl Format {
    /// Returns whether sampler YCbCr conversion is required for image views of this format.
    #[inline]
    pub fn requires_sampler_ycbcr_conversion(&self) -> bool {
        matches!(
            self.compatibility().0,
            FormatCompatibilityInner::YCbCrRGBA { .. }
                | FormatCompatibilityInner::YCbCr1Plane { .. }
                | FormatCompatibilityInner::YCbCr2Plane { .. }
                | FormatCompatibilityInner::YCbCr3Plane { .. }
        )
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
pub enum CompressionType {
    /// Adaptive Scalable Texture Compression.
    ASTC,
    /// S3TC Block Compression 1, also known as DXT1.
    BC1,
    /// S3TC Block Compression 2,
    /// also known as DXT2 (with premultiplied alpha) and DXT3 (no premultiplied alpha).
    BC2,
    /// S3TC Block Compression 3,
    /// also known as DXT4 (with premultiplied alpha) and DXT5 (no premultiplied alpha).
    BC3,
    /// S3TC Block Compression 4.
    BC4,
    /// S3TC Block Compression 5.
    BC5,
    /// S3TC Block Compression 6 or 6H.
    BC6H,
    /// S3TC Block Compression 7.
    BC7,
    /// Ericsson Texture Compression 2.
    ETC2,
    /// ETC2 Alpha Compression.
    EAC,
    /// PowerVR Texture Compression 1.
    PVRTC1,
    /// PowerVR Texture Compression 2.
    PVRTC2,
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
pub(crate) enum FormatCompatibilityInner {
    Normal {
        size: u8,
    },
    DepthStencil {
        ty: u8,
    },
    Compressed {
        compression: CompressionType,
        subtype: u8,
    },
    YCbCrRGBA {
        bits: u8,
    },
    YCbCr1Plane {
        bits: u8,
        g_even: bool,
    },
    YCbCr2Plane {
        bits: u8,
        block_texels: u8,
    },
    YCbCr3Plane {
        bits: u8,
        block_texels: u8,
    },
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

/// The features supported by a device for images with a particular format.
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
