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

pub use crate::autogen::Format;
use crate::device::physical::PhysicalDevice;
use crate::DeviceSize;
use crate::VulkanObject;
use half::f16;
use std::mem::MaybeUninit;
use std::vec::IntoIter as VecIntoIter;
use std::{error, fmt, mem};

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
    /// Signed integer that represents a normalized floating-point value in the range [-1,1].
    SNORM,
    /// Unsigned integer that represents a normalized floating-point value in the range [0,1].
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
