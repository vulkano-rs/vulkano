//! Declares all the formats of data and images supported by Vulkan.
//!
//! # Content of this module
//!
//! This module contains three things:
//!
//! - The `Format` enumeration, which contains all the available formats.
//! - The `FormatMarker` trait.
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
//!   format is very unsual.
//!
//! - `Sfloat` means that the values are regular floating points. No conversion is performed.
//!
//! - `Srgb` is the same as `Unorm`, except that the value is interpreted as being in the sRGB
//!   color space. This means that its value will be converted to fit in the RGB color space when
//!   it is read. The fourth channel (usually used for alpha), if present, is not concerned by the
//!   conversion.
//!
use std::vec::IntoIter as VecIntoIter;
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
unsafe impl Data for u8 {
    #[inline]
    fn ty() -> Format { Format::R8Uint }
}

macro_rules! formats {
    ($($name:ident => $vk:ident [$($f_ty:tt)*],)+) => (
        /// An enumeration of all the possible formats.
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

            /// Returns the `Format` corresponding to a Vulkan constant.
            #[doc(hidden)]
            pub fn from_num(val: u32) -> Option<Format> {
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
            #[derive(Debug, Copy, Clone)]
            #[allow(missing_docs)]
            #[allow(non_camel_case_types)]
            pub struct $name;

            formats!(__inner_impl__ $name $($f_ty)*);
        )+
    );

    (__inner_impl__ $name:ident float=$num:expr) => {
        unsafe impl FormatMarker for $name {
            type ClearValue = [f32; $num];

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }
        }
        unsafe impl FloatFormatMarker for $name {}
        unsafe impl FloatOrCompressedFormatMarker for $name {}
    };

    (__inner_impl__ $name:ident uint=$num:expr) => {
        unsafe impl FormatMarker for $name {
            type ClearValue = [u32; $num];

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }
        }

        unsafe impl UintFormatMarker for $name {}
    };

    (__inner_impl__ $name:ident sint=$num:expr) => {
        unsafe impl FormatMarker for $name {
            type ClearValue = [i32; $num];

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }
        }

        unsafe impl SintFormatMarker for $name {}
    };

    (__inner_impl__ $name:ident depth) => {
        unsafe impl FormatMarker for $name {
            type ClearValue = f32;

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }
        }

        unsafe impl DepthFormatMarker for $name {}
    };

    (__inner_impl__ $name:ident stencil) => {
        unsafe impl FormatMarker for $name {
            type ClearValue = u32;      // FIXME: shouldn't stencil be i32?

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }
        }

        unsafe impl StencilFormatMarker for $name {}
    };

    (__inner_impl__ $name:ident depthstencil) => {
        unsafe impl FormatMarker for $name {
            type ClearValue = (f32, u32);       // FIXME: shouldn't stencil be i32?

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }
        }

        unsafe impl DepthStencilFormatMarker for $name {}
    };

    (__inner_impl__ $name:ident compressed) => {
        unsafe impl FormatMarker for $name {
            type ClearValue = [f32; 4];

            #[inline]
            fn format(&self) -> Format {
                Format::$name
            }
        }

        unsafe impl CompressedFormatMarker for $name {}
        unsafe impl FloatOrCompressedFormatMarker for $name {}
    };

    (__inner_ty__ $name:ident float=$num:tt) => { FormatTy::Float };
    (__inner_ty__ $name:ident uint=$num:tt) => { FormatTy::Uint };
    (__inner_ty__ $name:ident sint=$num:tt) => { FormatTy::Sint };
    (__inner_ty__ $name:ident depth) => { FormatTy::Depth };
    (__inner_ty__ $name:ident stencil) => { FormatTy::Stencil };
    (__inner_ty__ $name:ident depthstencil) => { FormatTy::DepthStencil };
    (__inner_ty__ $name:ident compressed) => { FormatTy::Compressed };
}

formats! {
    //Undefined => FORMAT_UNDEFINED [float],      // FIXME: what to do with this one?
    R4G4UnormPack8 => FORMAT_R4G4_UNORM_PACK8 [float=2],
    R4G4B4A4UnormPack16 => FORMAT_R4G4B4A4_UNORM_PACK16 [float=4],
    B4G4R4A4UnormPack16 => FORMAT_B4G4R4A4_UNORM_PACK16 [float=4],
    R5G6B5UnormPack16 => FORMAT_R5G6B5_UNORM_PACK16 [float=3],
    B5G6R5UnormPack16 => FORMAT_B5G6R5_UNORM_PACK16 [float=3],
    R5G5B5A1UnormPack16 => FORMAT_R5G5B5A1_UNORM_PACK16 [float=4],
    B5G5R5A1UnormPack16 => FORMAT_B5G5R5A1_UNORM_PACK16 [float=4],
    A1R5G5B5UnormPack16 => FORMAT_A1R5G5B5_UNORM_PACK16 [float=4],
    R8Unorm => FORMAT_R8_UNORM [float=1],
    R8Snorm => FORMAT_R8_SNORM [float=1],
    R8Uscaled => FORMAT_R8_USCALED [float=1],
    R8Sscaled => FORMAT_R8_SSCALED [float=1],
    R8Uint => FORMAT_R8_UINT [uint=1],
    R8Sint => FORMAT_R8_SINT [sint=1],
    R8Srgb => FORMAT_R8_SRGB [float=1],
    R8G8Unorm => FORMAT_R8G8_UNORM [float=2],
    R8G8Snorm => FORMAT_R8G8_SNORM [float=2],
    R8G8Uscaled => FORMAT_R8G8_USCALED [float=2],
    R8G8Sscaled => FORMAT_R8G8_SSCALED [float=2],
    R8G8Uint => FORMAT_R8G8_UINT [uint=2],
    R8G8Sint => FORMAT_R8G8_SINT [sint=2],
    R8G8Srgb => FORMAT_R8G8_SRGB [float=2],
    R8G8B8Unorm => FORMAT_R8G8B8_UNORM [float=3],
    R8G8B8Snorm => FORMAT_R8G8B8_SNORM [float=3],
    R8G8B8Uscaled => FORMAT_R8G8B8_USCALED [float=3],
    R8G8B8Sscaled => FORMAT_R8G8B8_SSCALED [float=3],
    R8G8B8Uint => FORMAT_R8G8B8_UINT [uint=3],
    R8G8B8Sint => FORMAT_R8G8B8_SINT [sint=3],
    R8G8B8Srgb => FORMAT_R8G8B8_SRGB [float=3],
    B8G8R8Unorm => FORMAT_B8G8R8_UNORM [float=3],
    B8G8R8Snorm => FORMAT_B8G8R8_SNORM [float=3],
    B8G8R8Uscaled => FORMAT_B8G8R8_USCALED [float=3],
    B8G8R8Sscaled => FORMAT_B8G8R8_SSCALED [float=3],
    B8G8R8Uint => FORMAT_B8G8R8_UINT [uint=3],
    B8G8R8Sint => FORMAT_B8G8R8_SINT [sint=3],
    B8G8R8Srgb => FORMAT_B8G8R8_SRGB [float=3],
    R8G8B8A8Unorm => FORMAT_R8G8B8A8_UNORM [float=4],
    R8G8B8A8Snorm => FORMAT_R8G8B8A8_SNORM [float=4],
    R8G8B8A8Uscaled => FORMAT_R8G8B8A8_USCALED [float=4],
    R8G8B8A8Sscaled => FORMAT_R8G8B8A8_SSCALED [float=4],
    R8G8B8A8Uint => FORMAT_R8G8B8A8_UINT [uint=4],
    R8G8B8A8Sint => FORMAT_R8G8B8A8_SINT [sint=4],
    R8G8B8A8Srgb => FORMAT_R8G8B8A8_SRGB [float=4],
    B8G8R8A8Unorm => FORMAT_B8G8R8A8_UNORM [float=4],
    B8G8R8A8Snorm => FORMAT_B8G8R8A8_SNORM [float=4],
    B8G8R8A8Uscaled => FORMAT_B8G8R8A8_USCALED [float=4],
    B8G8R8A8Sscaled => FORMAT_B8G8R8A8_SSCALED [float=4],
    B8G8R8A8Uint => FORMAT_B8G8R8A8_UINT [uint=4],
    B8G8R8A8Sint => FORMAT_B8G8R8A8_SINT [sint=4],
    B8G8R8A8Srgb => FORMAT_B8G8R8A8_SRGB [float=4],
    A8B8G8R8UnormPack32 => FORMAT_A8B8G8R8_UNORM_PACK32 [float=4],
    A8B8G8R8SnormPack32 => FORMAT_A8B8G8R8_SNORM_PACK32 [float=4],
    A8B8G8R8UscaledPack32 => FORMAT_A8B8G8R8_USCALED_PACK32 [float=4],
    A8B8G8R8SscaledPack32 => FORMAT_A8B8G8R8_SSCALED_PACK32 [float=4],
    A8B8G8R8UintPack32 => FORMAT_A8B8G8R8_UINT_PACK32 [uint=4],
    A8B8G8R8SintPack32 => FORMAT_A8B8G8R8_SINT_PACK32 [sint=4],
    A8B8G8R8SrgbPack32 => FORMAT_A8B8G8R8_SRGB_PACK32 [float=4],
    A2R10G10B10UnormPack32 => FORMAT_A2R10G10B10_UNORM_PACK32 [float=4],
    A2R10G10B10SnormPack32 => FORMAT_A2R10G10B10_SNORM_PACK32 [float=4],
    A2R10G10B10UscaledPack32 => FORMAT_A2R10G10B10_USCALED_PACK32 [float=4],
    A2R10G10B10SscaledPack32 => FORMAT_A2R10G10B10_SSCALED_PACK32 [float=4],
    A2R10G10B10UintPack32 => FORMAT_A2R10G10B10_UINT_PACK32 [uint=4],
    A2R10G10B10SintPack32 => FORMAT_A2R10G10B10_SINT_PACK32 [sint=4],
    A2B10G10R10UnormPack32 => FORMAT_A2B10G10R10_UNORM_PACK32 [float=4],
    A2B10G10R10SnormPack32 => FORMAT_A2B10G10R10_SNORM_PACK32 [float=4],
    A2B10G10R10UscaledPack32 => FORMAT_A2B10G10R10_USCALED_PACK32 [float=4],
    A2B10G10R10SscaledPack32 => FORMAT_A2B10G10R10_SSCALED_PACK32 [float=4],
    A2B10G10R10UintPack32 => FORMAT_A2B10G10R10_UINT_PACK32 [uint=4],
    A2B10G10R10SintPack32 => FORMAT_A2B10G10R10_SINT_PACK32 [sint=4],
    R16Unorm => FORMAT_R16_UNORM [float=1],
    R16Snorm => FORMAT_R16_SNORM [float=1],
    R16Uscaled => FORMAT_R16_USCALED [float=1],
    R16Sscaled => FORMAT_R16_SSCALED [float=1],
    R16Uint => FORMAT_R16_UINT [uint=1],
    R16Sint => FORMAT_R16_SINT [sint=1],
    R16Sfloat => FORMAT_R16_SFLOAT [float=1],
    R16G16Unorm => FORMAT_R16G16_UNORM [float=2],
    R16G16Snorm => FORMAT_R16G16_SNORM [float=2],
    R16G16Uscaled => FORMAT_R16G16_USCALED [float=2],
    R16G16Sscaled => FORMAT_R16G16_SSCALED [float=2],
    R16G16Uint => FORMAT_R16G16_UINT [uint=2],
    R16G16Sint => FORMAT_R16G16_SINT [sint=2],
    R16G16Sfloat => FORMAT_R16G16_SFLOAT [float=2],
    R16G16B16Unorm => FORMAT_R16G16B16_UNORM [float=3],
    R16G16B16Snorm => FORMAT_R16G16B16_SNORM [float=3],
    R16G16B16Uscaled => FORMAT_R16G16B16_USCALED [float=3],
    R16G16B16Sscaled => FORMAT_R16G16B16_SSCALED [float=3],
    R16G16B16Uint => FORMAT_R16G16B16_UINT [uint=3],
    R16G16B16Sint => FORMAT_R16G16B16_SINT [sint=3],
    R16G16B16Sfloat => FORMAT_R16G16B16_SFLOAT [float=3],
    R16G16B16A16Unorm => FORMAT_R16G16B16A16_UNORM [float=4],
    R16G16B16A16Snorm => FORMAT_R16G16B16A16_SNORM [float=4],
    R16G16B16A16Uscaled => FORMAT_R16G16B16A16_USCALED [float=4],
    R16G16B16A16Sscaled => FORMAT_R16G16B16A16_SSCALED [float=4],
    R16G16B16A16Uint => FORMAT_R16G16B16A16_UINT [uint=4],
    R16G16B16A16Sint => FORMAT_R16G16B16A16_SINT [sint=4],
    R16G16B16A16Sfloat => FORMAT_R16G16B16A16_SFLOAT [float=4],
    R32Uint => FORMAT_R32_UINT [uint=1],
    R32Sint => FORMAT_R32_SINT [sint=1],
    R32Sfloat => FORMAT_R32_SFLOAT [float=1],
    R32G32Uint => FORMAT_R32G32_UINT [uint=2],
    R32G32Sint => FORMAT_R32G32_SINT [sint=2],
    R32G32Sfloat => FORMAT_R32G32_SFLOAT [float=2],
    R32G32B32Uint => FORMAT_R32G32B32_UINT [uint=3],
    R32G32B32Sint => FORMAT_R32G32B32_SINT [sint=3],
    R32G32B32Sfloat => FORMAT_R32G32B32_SFLOAT [float=3],
    R32G32B32A32Uint => FORMAT_R32G32B32A32_UINT [uint=4],
    R32G32B32A32Sint => FORMAT_R32G32B32A32_SINT [sint=4],
    R32G32B32A32Sfloat => FORMAT_R32G32B32A32_SFLOAT [float=4],
    R64Uint => FORMAT_R64_UINT [uint=1],
    R64Sint => FORMAT_R64_SINT [sint=1],
    R64Sfloat => FORMAT_R64_SFLOAT [float=1],
    R64G64Uint => FORMAT_R64G64_UINT [uint=2],
    R64G64Sint => FORMAT_R64G64_SINT [sint=2],
    R64G64Sfloat => FORMAT_R64G64_SFLOAT [float=2],
    R64G64B64Uint => FORMAT_R64G64B64_UINT [uint=3],
    R64G64B64Sint => FORMAT_R64G64B64_SINT [sint=3],
    R64G64B64Sfloat => FORMAT_R64G64B64_SFLOAT [float=3],
    R64G64B64A64Uint => FORMAT_R64G64B64A64_UINT [uint=4],
    R64G64B64A64Sint => FORMAT_R64G64B64A64_SINT [sint=4],
    R64G64B64A64Sfloat => FORMAT_R64G64B64A64_SFLOAT [float=4],
    B10G11R11UfloatPack32 => FORMAT_B10G11R11_UFLOAT_PACK32 [float=3],
    E5B9G9R9UfloatPack32 => FORMAT_E5B9G9R9_UFLOAT_PACK32 [float=3],
    D16Unorm => FORMAT_D16_UNORM [depth],
    X8_D24UnormPack32 => FORMAT_X8_D24_UNORM_PACK32 [depth],
    D32Sfloat => FORMAT_D32_SFLOAT [depth],
    S8Uint => FORMAT_S8_UINT [stencil],
    D16Unorm_S8Uint => FORMAT_D16_UNORM_S8_UINT [depthstencil],
    D24Unorm_S8Uint => FORMAT_D24_UNORM_S8_UINT [depthstencil],
    D32Sfloat_S8Uint => FORMAT_D32_SFLOAT_S8_UINT [depthstencil],
    BC1_RGBUnormBlock => FORMAT_BC1_RGB_UNORM_BLOCK [compressed],
    BC1_RGBSrgbBlock => FORMAT_BC1_RGB_SRGB_BLOCK [compressed],
    BC1_RGBAUnormBlock => FORMAT_BC1_RGBA_UNORM_BLOCK [compressed],
    BC1_RGBASrgbBlock => FORMAT_BC1_RGBA_SRGB_BLOCK [compressed],
    BC2UnormBlock => FORMAT_BC2_UNORM_BLOCK [compressed],
    BC2SrgbBlock => FORMAT_BC2_SRGB_BLOCK [compressed],
    BC3UnormBlock => FORMAT_BC3_UNORM_BLOCK [compressed],
    BC3SrgbBlock => FORMAT_BC3_SRGB_BLOCK [compressed],
    BC4UnormBlock => FORMAT_BC4_UNORM_BLOCK [compressed],
    BC4SnormBlock => FORMAT_BC4_SNORM_BLOCK [compressed],
    BC5UnormBlock => FORMAT_BC5_UNORM_BLOCK [compressed],
    BC5SnormBlock => FORMAT_BC5_SNORM_BLOCK [compressed],
    BC6HUfloatBlock => FORMAT_BC6H_UFLOAT_BLOCK [compressed],
    BC6HSfloatBlock => FORMAT_BC6H_SFLOAT_BLOCK [compressed],
    BC7UnormBlock => FORMAT_BC7_UNORM_BLOCK [compressed],
    BC7SrgbBlock => FORMAT_BC7_SRGB_BLOCK [compressed],
    ETC2_R8G8B8UnormBlock => FORMAT_ETC2_R8G8B8_UNORM_BLOCK [compressed],
    ETC2_R8G8B8SrgbBlock => FORMAT_ETC2_R8G8B8_SRGB_BLOCK [compressed],
    ETC2_R8G8B8A1UnormBlock => FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK [compressed],
    ETC2_R8G8B8A1SrgbBlock => FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK [compressed],
    ETC2_R8G8B8A8UnormBlock => FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK [compressed],
    ETC2_R8G8B8A8SrgbBlock => FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK [compressed],
    EAC_R11UnormBlock => FORMAT_EAC_R11_UNORM_BLOCK [compressed],
    EAC_R11SnormBlock => FORMAT_EAC_R11_SNORM_BLOCK [compressed],
    EAC_R11G11UnormBlock => FORMAT_EAC_R11G11_UNORM_BLOCK [compressed],
    EAC_R11G11SnormBlock => FORMAT_EAC_R11G11_SNORM_BLOCK [compressed],
    ASTC_4x4UnormBlock => FORMAT_ASTC_4x4_UNORM_BLOCK [compressed],
    ASTC_4x4SrgbBlock => FORMAT_ASTC_4x4_SRGB_BLOCK [compressed],
    ASTC_5x4UnormBlock => FORMAT_ASTC_5x4_UNORM_BLOCK [compressed],
    ASTC_5x4SrgbBlock => FORMAT_ASTC_5x4_SRGB_BLOCK [compressed],
    ASTC_5x5UnormBlock => FORMAT_ASTC_5x5_UNORM_BLOCK [compressed],
    ASTC_5x5SrgbBlock => FORMAT_ASTC_5x5_SRGB_BLOCK [compressed],
    ASTC_6x5UnormBlock => FORMAT_ASTC_6x5_UNORM_BLOCK [compressed],
    ASTC_6x5SrgbBlock => FORMAT_ASTC_6x5_SRGB_BLOCK [compressed],
    ASTC_6x6UnormBlock => FORMAT_ASTC_6x6_UNORM_BLOCK [compressed],
    ASTC_6x6SrgbBlock => FORMAT_ASTC_6x6_SRGB_BLOCK [compressed],
    ASTC_8x5UnormBlock => FORMAT_ASTC_8x5_UNORM_BLOCK [compressed],
    ASTC_8x5SrgbBlock => FORMAT_ASTC_8x5_SRGB_BLOCK [compressed],
    ASTC_8x6UnormBlock => FORMAT_ASTC_8x6_UNORM_BLOCK [compressed],
    ASTC_8x6SrgbBlock => FORMAT_ASTC_8x6_SRGB_BLOCK [compressed],
    ASTC_8x8UnormBlock => FORMAT_ASTC_8x8_UNORM_BLOCK [compressed],
    ASTC_8x8SrgbBlock => FORMAT_ASTC_8x8_SRGB_BLOCK [compressed],
    ASTC_10x5UnormBlock => FORMAT_ASTC_10x5_UNORM_BLOCK [compressed],
    ASTC_10x5SrgbBlock => FORMAT_ASTC_10x5_SRGB_BLOCK [compressed],
    ASTC_10x6UnormBlock => FORMAT_ASTC_10x6_UNORM_BLOCK [compressed],
    ASTC_10x6SrgbBlock => FORMAT_ASTC_10x6_SRGB_BLOCK [compressed],
    ASTC_10x8UnormBlock => FORMAT_ASTC_10x8_UNORM_BLOCK [compressed],
    ASTC_10x8SrgbBlock => FORMAT_ASTC_10x8_SRGB_BLOCK [compressed],
    ASTC_10x10UnormBlock => FORMAT_ASTC_10x10_UNORM_BLOCK [compressed],
    ASTC_10x10SrgbBlock => FORMAT_ASTC_10x10_SRGB_BLOCK [compressed],
    ASTC_12x10UnormBlock => FORMAT_ASTC_12x10_UNORM_BLOCK [compressed],
    ASTC_12x10SrgbBlock => FORMAT_ASTC_12x10_SRGB_BLOCK [compressed],
    ASTC_12x12UnormBlock => FORMAT_ASTC_12x12_UNORM_BLOCK [compressed],
    ASTC_12x12SrgbBlock => FORMAT_ASTC_12x12_SRGB_BLOCK [compressed],
}

pub unsafe trait FormatMarker {
    type ClearValue: Into<ClearValue>;

    fn format(&self) -> Format;
}

pub unsafe trait FloatFormatMarker: FormatMarker {}
pub unsafe trait UintFormatMarker: FormatMarker {}
pub unsafe trait SintFormatMarker: FormatMarker {}
pub unsafe trait DepthFormatMarker: FormatMarker {}
pub unsafe trait StencilFormatMarker: FormatMarker {}
pub unsafe trait DepthStencilFormatMarker: FormatMarker {}
pub unsafe trait CompressedFormatMarker: FormatMarker {}
pub unsafe trait FloatOrCompressedFormatMarker: FormatMarker {}

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

/// Describes a uniform value that will be used to fill an image.
// TODO: should have the same layout as `vk::ClearValue` for performances
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
        ClearValue::Uint([val[0], 0, 0, 0])     // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 2]> for ClearValue {
    #[inline]
    fn from(val: [u32; 2]) -> ClearValue {
        ClearValue::Uint([val[0], val[1], 0, 0])        // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 3]> for ClearValue {
    #[inline]
    fn from(val: [u32; 3]) -> ClearValue {
        ClearValue::Uint([val[0], val[1], val[2], 0])       // TODO: is alpha value 0 correct?
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
        ClearValue::Int([val[0], 0, 0, 0])      // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 2]> for ClearValue {
    #[inline]
    fn from(val: [i32; 2]) -> ClearValue {
        ClearValue::Int([val[0], val[1], 0, 0])     // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 3]> for ClearValue {
    #[inline]
    fn from(val: [i32; 3]) -> ClearValue {
        ClearValue::Int([val[0], val[1], val[2], 0])        // TODO: is alpha value 0 correct?
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
