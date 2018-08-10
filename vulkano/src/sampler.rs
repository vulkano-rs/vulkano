// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! How to retrieve data from an image within a shader.
//!
//! When you retrieve data from an image, you have to pass the coordinates of the pixel you want
//! to retrieve. The implementation then performs various calculations, and these operations are
//! what the `Sampler` struct describes.
//!
//! Sampling is a very complex topic but that hasn't changed much since the beginnings of 3D
//! rendering. Documentation here is missing, but any tutorial about OpenGL or DirectX can teach
//! you how it works.
//!
//! # Examples
//!
//! A simple sampler for most usages:
//!
//! ```
//! use vulkano::sampler::Sampler;
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! let _sampler = Sampler::simple_repeat_linear_no_mipmap(device.clone());
//! ```
//!
//! More detailed sampler creation:
//!
//! ```
//! use vulkano::sampler;
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! let _sampler = sampler::Sampler::new(device.clone(), sampler::Filter::Linear,
//!                                      sampler::Filter::Linear,
//!                                      sampler::MipmapMode::Nearest,
//!                                      sampler::SamplerAddressMode::Repeat,
//!                                      sampler::SamplerAddressMode::Repeat,
//!                                      sampler::SamplerAddressMode::Repeat, 1.0, 1.0,
//!                                      0.0, 100.0).unwrap();;
//! ```
//!
//! # About border colors
//!
//! One of the possible values of `SamplerAddressMode` and `UnnormalizedSamplerAddressMode` is
//! `ClampToBorder`. This value indicates that accessing an image outside of its range must return
//! the specified color.
//!
//! However this comes with restrictions. When using a floating-point border color, the sampler can
//! only be used with floating-point or depth image views. When using an integer border color, the
//! sampler can only be used with integer or stencil image views. In addition to this, you can't
//! use an opaque black border color with an image view that uses components swizzling.
//!
//! > **Note**: The reason for this restriction about opaque black borders is that the value of the
//! > alpha is 1.0 while the value of the color components is 0.0. In the other border colors, the
//! > value of all the components is the same.
//!
//! Samplers that don't use `ClampToBorder` are not concerned by these restrictions.
//!
// FIXME: restrictions aren't checked yet

use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;

use Error;
use OomError;
use VulkanObject;
use check_errors;
use device::Device;
use device::DeviceOwned;
use vk;

pub use pipeline::depth_stencil::Compare;

/// Describes how to retrieve data from an image within a shader.
pub struct Sampler {
    sampler: vk::Sampler,
    device: Arc<Device>,
    compare_mode: bool,
    unnormalized: bool,
    usable_with_float_formats: bool,
    usable_with_int_formats: bool,
    usable_with_swizzling: bool,
}

impl Sampler {
    /// Shortcut for creating a sampler with linear sampling, linear mipmaps, and with the repeat
    /// mode for borders.
    ///
    /// Useful for prototyping, but can also be used in real projects.
    ///
    /// # Panic
    ///
    /// - Panics if out of memory or the maximum number of samplers has exceeded.
    ///
    #[inline]
    pub fn simple_repeat_linear(device: Arc<Device>) -> Arc<Sampler> {
        Sampler::new(device,
                     Filter::Linear,
                     Filter::Linear,
                     MipmapMode::Linear,
                     SamplerAddressMode::Repeat,
                     SamplerAddressMode::Repeat,
                     SamplerAddressMode::Repeat,
                     0.0,
                     1.0,
                     0.0,
                     1_000.0)
            .unwrap()
    }

    /// Shortcut for creating a sampler with linear sampling, that only uses the main level of
    /// images, and with the repeat mode for borders.
    ///
    /// Useful for prototyping, but can also be used in real projects.
    ///
    /// # Panic
    ///
    /// - Panics if out of memory or the maximum number of samplers has exceeded.
    ///
    #[inline]
    pub fn simple_repeat_linear_no_mipmap(device: Arc<Device>) -> Arc<Sampler> {
        Sampler::new(device,
                     Filter::Linear,
                     Filter::Linear,
                     MipmapMode::Nearest,
                     SamplerAddressMode::Repeat,
                     SamplerAddressMode::Repeat,
                     SamplerAddressMode::Repeat,
                     0.0,
                     1.0,
                     0.0,
                     1.0)
            .unwrap()
    }

    /// Creates a new `Sampler` with the given behavior.
    ///
    /// `mag_filter` and `min_filter` define how the implementation should sample from the image
    /// when it is respectively larger and smaller than the original.
    ///
    /// `mipmap_mode` defines how the implementation should choose which mipmap to use.
    ///
    /// `address_u`, `address_v` and `address_w` define how the implementation should behave when
    /// sampling outside of the texture coordinates range `[0.0, 1.0]`.
    ///
    /// `mip_lod_bias` is a value to add to .
    ///
    /// `max_anisotropy` must be greater than or equal to 1.0. If greater than 1.0, the
    /// implementation will use anisotropic filtering. Using a value greater than 1.0 requires
    /// the `sampler_anisotropy` feature to be enabled when creating the device.
    ///
    /// `min_lod` and `max_lod` are respectively the minimum and maximum mipmap level to use.
    /// `max_lod` must always be greater than or equal to `min_lod`.
    ///
    /// # Panic
    ///
    /// - Panics if multiple `ClampToBorder` values are passed and the border color is different.
    /// - Panics if `max_anisotropy < 1.0`.
    /// - Panics if `min_lod > max_lod`.
    ///
    #[inline(always)]
    pub fn new(device: Arc<Device>, mag_filter: Filter, min_filter: Filter,
               mipmap_mode: MipmapMode, address_u: SamplerAddressMode,
               address_v: SamplerAddressMode, address_w: SamplerAddressMode, mip_lod_bias: f32,
               max_anisotropy: f32, min_lod: f32, max_lod: f32)
               -> Result<Arc<Sampler>, SamplerCreationError> {
        Sampler::new_impl(device,
                          mag_filter,
                          min_filter,
                          mipmap_mode,
                          address_u,
                          address_v,
                          address_w,
                          mip_lod_bias,
                          max_anisotropy,
                          min_lod,
                          max_lod,
                          None)
    }

    /// Creates a new `Sampler` with the given behavior.
    ///
    /// Contrary to `new`, this creates a sampler that is used to compare depth values.
    ///
    /// A sampler like this can only operate on depth or depth-stencil textures. Instead of
    /// returning the value of the texture, this sampler will return a value between 0.0 and 1.0
    /// indicating how much the reference value (passed by the shader) compares to the value in the
    /// texture.
    ///
    /// Note that it doesn't make sense to create a compare-mode sampler with an integer border
    /// color, as such a sampler would be unusable.
    ///
    /// # Panic
    ///
    /// Same panic reasons as `new`.
    ///
    #[inline(always)]
    pub fn compare(device: Arc<Device>, mag_filter: Filter, min_filter: Filter,
                   mipmap_mode: MipmapMode, address_u: SamplerAddressMode,
                   address_v: SamplerAddressMode, address_w: SamplerAddressMode,
                   mip_lod_bias: f32, max_anisotropy: f32, min_lod: f32, max_lod: f32,
                   compare: Compare)
                   -> Result<Arc<Sampler>, SamplerCreationError> {
        Sampler::new_impl(device,
                          mag_filter,
                          min_filter,
                          mipmap_mode,
                          address_u,
                          address_v,
                          address_w,
                          mip_lod_bias,
                          max_anisotropy,
                          min_lod,
                          max_lod,
                          Some(compare))
    }

    fn new_impl(device: Arc<Device>, mag_filter: Filter, min_filter: Filter,
                mipmap_mode: MipmapMode, address_u: SamplerAddressMode,
                address_v: SamplerAddressMode, address_w: SamplerAddressMode, mip_lod_bias: f32,
                max_anisotropy: f32, min_lod: f32, max_lod: f32, compare: Option<Compare>)
                -> Result<Arc<Sampler>, SamplerCreationError> {
        assert!(max_anisotropy >= 1.0);
        assert!(min_lod <= max_lod);

        // Check max anisotropy.
        if max_anisotropy > 1.0 {
            if !device.enabled_features().sampler_anisotropy {
                return Err(SamplerCreationError::SamplerAnisotropyFeatureNotEnabled);
            }

            let limit = device.physical_device().limits().max_sampler_anisotropy();
            if max_anisotropy > limit {
                return Err(SamplerCreationError::AnisotropyLimitExceeded {
                               requested: max_anisotropy,
                               maximum: limit,
                           });
            }
        }

        // Check mip_lod_bias value.
        {
            let limit = device.physical_device().limits().max_sampler_lod_bias();
            if mip_lod_bias > limit {
                return Err(SamplerCreationError::MipLodBiasLimitExceeded {
                               requested: mip_lod_bias,
                               maximum: limit,
                           });
            }
        }

        // Check MirrorClampToEdge extension support
        if [address_u, address_v, address_w]
            .iter()
            .any(|&mode| mode == SamplerAddressMode::MirrorClampToEdge)
        {
            if !device.loaded_extensions().khr_sampler_mirror_clamp_to_edge {
                return Err(SamplerCreationError::SamplerMirrorClampToEdgeExtensionNotEnabled);
            }
        }

        // Handling border color.
        let border_color = address_u.border_color();
        let border_color = match (border_color, address_v.border_color()) {
            (Some(b1), Some(b2)) => {
                assert_eq!(b1, b2);
                Some(b1)
            },
            (None, b) => b,
            (b, None) => b,
        };
        let border_color = match (border_color, address_w.border_color()) {
            (Some(b1), Some(b2)) => {
                assert_eq!(b1, b2);
                Some(b1)
            },
            (None, b) => b,
            (b, None) => b,
        };

        let vk = device.pointers();
        let sampler = unsafe {
            let infos = vk::SamplerCreateInfo {
                sType: vk::STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                magFilter: mag_filter as u32,
                minFilter: min_filter as u32,
                mipmapMode: mipmap_mode as u32,
                addressModeU: address_u.to_vk(),
                addressModeV: address_v.to_vk(),
                addressModeW: address_w.to_vk(),
                mipLodBias: mip_lod_bias,
                anisotropyEnable: if max_anisotropy > 1.0 {
                    vk::TRUE
                } else {
                    vk::FALSE
                },
                maxAnisotropy: max_anisotropy,
                compareEnable: if compare.is_some() {
                    vk::TRUE
                } else {
                    vk::FALSE
                },
                compareOp: compare.map(|c| c as u32).unwrap_or(0),
                minLod: min_lod,
                maxLod: max_lod,
                borderColor: border_color.map(|b| b as u32).unwrap_or(0),
                unnormalizedCoordinates: vk::FALSE,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateSampler(device.internal_object(),
                                          &infos,
                                          ptr::null(),
                                          &mut output))?;
            output
        };

        Ok(Arc::new(Sampler {
                        sampler: sampler,
                        device: device.clone(),
                        compare_mode: compare.is_some(),
                        unnormalized: false,
                        usable_with_float_formats: match border_color {
                            Some(BorderColor::FloatTransparentBlack) => true,
                            Some(BorderColor::FloatOpaqueBlack) => true,
                            Some(BorderColor::FloatOpaqueWhite) => true,
                            Some(_) => false,
                            None => true,
                        },
                        usable_with_int_formats: compare.is_none() &&
                            match border_color {
                                Some(BorderColor::IntTransparentBlack) => true,
                                Some(BorderColor::IntOpaqueBlack) => true,
                                Some(BorderColor::IntOpaqueWhite) => true,
                                Some(_) => false,
                                None => true,
                            },
                        usable_with_swizzling: match border_color {
                            Some(BorderColor::FloatOpaqueBlack) => false,
                            Some(BorderColor::IntOpaqueBlack) => false,
                            _ => true,
                        },
                    }))
    }

    /// Creates a sampler with unnormalized coordinates. This means that texture coordinates won't
    /// range between `0.0` and `1.0` but use plain pixel offsets.
    ///
    /// Using an unnormalized sampler adds a few restrictions:
    ///
    /// - It can only be used with non-array 1D or 2D images.
    /// - It can only be used with images with a single mipmap.
    /// - Projection and offsets can't be used by shaders. Only the first mipmap can be accessed.
    ///
    /// # Panic
    ///
    /// - Panics if multiple `ClampToBorder` values are passed and the border color is different.
    ///
    pub fn unnormalized(device: Arc<Device>, filter: Filter,
                        address_u: UnnormalizedSamplerAddressMode,
                        address_v: UnnormalizedSamplerAddressMode)
                        -> Result<Arc<Sampler>, SamplerCreationError> {
        let vk = device.pointers();

        let border_color = address_u.border_color();
        let border_color = match (border_color, address_v.border_color()) {
            (Some(b1), Some(b2)) => {
                assert_eq!(b1, b2);
                Some(b1)
            },
            (None, b) => b,
            (b, None) => b,
        };

        let sampler = unsafe {
            let infos = vk::SamplerCreateInfo {
                sType: vk::STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                magFilter: filter as u32,
                minFilter: filter as u32,
                mipmapMode: vk::SAMPLER_MIPMAP_MODE_NEAREST,
                addressModeU: address_u.to_vk(),
                addressModeV: address_v.to_vk(),
                addressModeW: vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, // unused by the impl
                mipLodBias: 0.0,
                anisotropyEnable: vk::FALSE,
                maxAnisotropy: 1.0,
                compareEnable: vk::FALSE,
                compareOp: vk::COMPARE_OP_NEVER,
                minLod: 0.0,
                maxLod: 0.0,
                borderColor: border_color.map(|b| b as u32).unwrap_or(0),
                unnormalizedCoordinates: vk::TRUE,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateSampler(device.internal_object(),
                                          &infos,
                                          ptr::null(),
                                          &mut output))?;
            output
        };

        Ok(Arc::new(Sampler {
                        sampler: sampler,
                        device: device.clone(),
                        compare_mode: false,
                        unnormalized: true,
                        usable_with_float_formats: match border_color {
                            Some(BorderColor::FloatTransparentBlack) => true,
                            Some(BorderColor::FloatOpaqueBlack) => true,
                            Some(BorderColor::FloatOpaqueWhite) => true,
                            Some(_) => false,
                            None => true,
                        },
                        usable_with_int_formats: match border_color {
                            Some(BorderColor::IntTransparentBlack) => true,
                            Some(BorderColor::IntOpaqueBlack) => true,
                            Some(BorderColor::IntOpaqueWhite) => true,
                            Some(_) => false,
                            None => true,
                        },
                        usable_with_swizzling: match border_color {
                            Some(BorderColor::FloatOpaqueBlack) => false,
                            Some(BorderColor::IntOpaqueBlack) => false,
                            _ => true,
                        },
                    }))
    }

    /// Returns true if the sampler is a compare-mode sampler.
    #[inline]
    pub fn compare_mode(&self) -> bool {
        self.compare_mode
    }

    /// Returns true if the sampler is unnormalized.
    #[inline]
    pub fn is_unnormalized(&self) -> bool {
        self.unnormalized
    }

    /// Returns true if the sampler can be used with floating-point image views. See the
    /// documentation of the `sampler` module for more info.
    #[inline]
    pub fn usable_with_float_formats(&self) -> bool {
        self.usable_with_float_formats
    }

    /// Returns true if the sampler can be used with integer image views. See the documentation of
    /// the `sampler` module for more info.
    #[inline]
    pub fn usable_with_int_formats(&self) -> bool {
        self.usable_with_int_formats
    }

    /// Returns true if the sampler can be used with image views that have non-identity swizzling.
    /// See the documentation of the `sampler` module for more info.
    #[inline]
    pub fn usable_with_swizzling(&self) -> bool {
        self.usable_with_swizzling
    }
}

unsafe impl DeviceOwned for Sampler {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for Sampler {
    type Object = vk::Sampler;

    const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT;

    #[inline]
    fn internal_object(&self) -> vk::Sampler {
        self.sampler
    }
}

impl fmt::Debug for Sampler {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan sampler {:?}>", self.sampler)
    }
}

impl Drop for Sampler {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroySampler(self.device.internal_object(), self.sampler, ptr::null());
        }
    }
}

/// Describes how the color of each pixel should be determined.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Filter {
    /// The four pixels whose center surround the requested coordinates are taken, then their
    /// values are interpolated.
    Linear = vk::FILTER_LINEAR,

    /// The pixel whose center is nearest to the requested coordinates is taken from the source
    /// and its value is returned as-is.
    Nearest = vk::FILTER_NEAREST,
}

/// Describes which mipmap from the source to use.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MipmapMode {
    /// Use the mipmap whose dimensions are the nearest to the dimensions of the destination.
    Nearest = vk::SAMPLER_MIPMAP_MODE_NEAREST,

    /// Take the mipmap whose dimensions are no greater than that of the destination together
    /// with the next higher level mipmap, calculate the value for both, and interpolate them.
    Linear = vk::SAMPLER_MIPMAP_MODE_LINEAR,
}

/// How the sampler should behave when it needs to access a pixel that is out of range of the
/// texture.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SamplerAddressMode {
    /// Repeat the texture. In other words, the pixel at coordinate `x + 1.0` is the same as the
    /// one at coordinate `x`.
    Repeat,

    /// Repeat the texture but mirror it at every repetition. In other words, the pixel at
    /// coordinate `x + 1.0` is the same as the one at coordinate `1.0 - x`.
    MirroredRepeat,

    /// The coordinates are clamped to the valid range. Coordinates below 0.0 have the same value
    /// as coordinate 0.0. Coordinates over 1.0 have the same value as coordinate 1.0.
    ClampToEdge,

    /// Any pixel out of range is considered to be part of the "border" of the image, which has a
    /// specific color of your choice.
    ///
    /// Note that if you use `ClampToBorder` multiple times, they must all have the same border
    /// color.
    ClampToBorder(BorderColor),

    /// Similar to `MirroredRepeat`, except that coordinates are clamped to the range
    /// `[-1.0, 1.0]`.
    MirrorClampToEdge,
}

impl SamplerAddressMode {
    #[inline]
    fn to_vk(self) -> vk::SamplerAddressMode {
        match self {
            SamplerAddressMode::Repeat => vk::SAMPLER_ADDRESS_MODE_REPEAT,
            SamplerAddressMode::MirroredRepeat => vk::SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
            SamplerAddressMode::ClampToEdge => vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            SamplerAddressMode::ClampToBorder(_) => vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            SamplerAddressMode::MirrorClampToEdge => vk::SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
        }
    }

    #[inline]
    fn border_color(self) -> Option<BorderColor> {
        match self {
            SamplerAddressMode::ClampToBorder(c) => Some(c),
            _ => None,
        }
    }
}

/// How the sampler should behave when it needs to access a pixel that is out of range of the
/// texture.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum UnnormalizedSamplerAddressMode {
    /// The coordinates are clamped to the valid range. Coordinates below 0 have the same value
    /// as coordinate 0. Coordinates over *size of texture* have the same value as coordinate
    /// *size of texture*.
    ClampToEdge,

    /// Any pixel out of range is considered to be part of the "border" of the image, which has a
    /// specific color of your choice.
    ///
    /// Note that if you use `ClampToBorder` multiple times, they must all have the same border
    /// color.
    ClampToBorder(BorderColor),
}

impl UnnormalizedSamplerAddressMode {
    #[inline]
    fn to_vk(self) -> vk::SamplerAddressMode {
        match self {
            UnnormalizedSamplerAddressMode::ClampToEdge => vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            UnnormalizedSamplerAddressMode::ClampToBorder(_) => {
                vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
            },
        }
    }

    #[inline]
    fn border_color(self) -> Option<BorderColor> {
        match self {
            UnnormalizedSamplerAddressMode::ClampToEdge => None,
            UnnormalizedSamplerAddressMode::ClampToBorder(c) => Some(c),
        }
    }
}

/// The color to use for the border of an image.
///
/// Only relevant if you use `ClampToBorder`.
///
/// Using a border color restricts the sampler to either floating-point images or integer images.
/// See the documentation of the `sampler` module for more info.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum BorderColor {
    /// The value `(0.0, 0.0, 0.0, 0.0)`. Can only be used with floating-point images.
    FloatTransparentBlack = vk::BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,

    /// The value `(0, 0, 0, 0)`. Can only be used with integer images.
    IntTransparentBlack = vk::BORDER_COLOR_INT_TRANSPARENT_BLACK,

    /// The value `(0.0, 0.0, 0.0, 1.0)`. Can only be used with floating-point identity-swizzled
    /// images.
    FloatOpaqueBlack = vk::BORDER_COLOR_FLOAT_OPAQUE_BLACK,

    /// The value `(0, 0, 0, 1)`. Can only be used with integer identity-swizzled images.
    IntOpaqueBlack = vk::BORDER_COLOR_INT_OPAQUE_BLACK,

    /// The value `(1.0, 1.0, 1.0, 1.0)`. Can only be used with floating-point images.
    FloatOpaqueWhite = vk::BORDER_COLOR_FLOAT_OPAQUE_WHITE,

    /// The value `(1, 1, 1, 1)`. Can only be used with integer images.
    IntOpaqueWhite = vk::BORDER_COLOR_INT_OPAQUE_WHITE,
}

/// Error that can happen when creating an instance.
#[derive(Clone, Debug, PartialEq)]
pub enum SamplerCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// Too many sampler objects have been created. You must destroy some before creating new ones.
    /// Note the specs guarantee that at least 4000 samplers can exist simultaneously.
    TooManyObjects,

    /// Using an anisotropy greater than 1.0 requires enabling the `sampler_anisotropy` feature
    /// when creating the device.
    SamplerAnisotropyFeatureNotEnabled,

    /// The requested anisotropy level exceeds the device's limits.
    AnisotropyLimitExceeded {
        /// The value that was requested.
        requested: f32,
        /// The maximum supported value.
        maximum: f32,
    },

    /// The requested mip lod bias exceeds the device's limits.
    MipLodBiasLimitExceeded {
        /// The value that was requested.
        requested: f32,
        /// The maximum supported value.
        maximum: f32,
    },

    /// Using `MirrorClampToEdge` requires enabling the `VK_KHR_sampler_mirror_clamp_to_edge`
    /// extension when creating the device.
    SamplerMirrorClampToEdgeExtensionNotEnabled,
}

impl error::Error for SamplerCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            SamplerCreationError::OomError(_) => "not enough memory available",
            SamplerCreationError::TooManyObjects => "too many simultaneous sampler objects",
            SamplerCreationError::SamplerAnisotropyFeatureNotEnabled =>
                "the `sampler_anisotropy` feature is not enabled",
            SamplerCreationError::AnisotropyLimitExceeded { .. } => "anisotropy limit exceeded",
            SamplerCreationError::MipLodBiasLimitExceeded { .. } => "mip lod bias limit exceeded",
            SamplerCreationError::SamplerMirrorClampToEdgeExtensionNotEnabled =>
                "the device extension `VK_KHR_sampler_mirror_clamp_to_edge` is not enabled",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            SamplerCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SamplerCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for SamplerCreationError {
    #[inline]
    fn from(err: OomError) -> SamplerCreationError {
        SamplerCreationError::OomError(err)
    }
}

impl From<Error> for SamplerCreationError {
    #[inline]
    fn from(err: Error) -> SamplerCreationError {
        match err {
            err @ Error::OutOfHostMemory => SamplerCreationError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => SamplerCreationError::OomError(OomError::from(err)),
            Error::TooManyObjects => SamplerCreationError::TooManyObjects,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use sampler;

    #[test]
    fn create_regular() {
        let (device, queue) = gfx_dev_and_queue!();

        let s = sampler::Sampler::new(device,
                                      sampler::Filter::Linear,
                                      sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      1.0,
                                      1.0,
                                      0.0,
                                      2.0)
            .unwrap();
        assert!(!s.compare_mode());
        assert!(!s.is_unnormalized());
    }

    #[test]
    fn create_compare() {
        let (device, queue) = gfx_dev_and_queue!();

        let s = sampler::Sampler::compare(device,
                                          sampler::Filter::Linear,
                                          sampler::Filter::Linear,
                                          sampler::MipmapMode::Nearest,
                                          sampler::SamplerAddressMode::Repeat,
                                          sampler::SamplerAddressMode::Repeat,
                                          sampler::SamplerAddressMode::Repeat,
                                          1.0,
                                          1.0,
                                          0.0,
                                          2.0,
                                          sampler::Compare::Less)
            .unwrap();

        assert!(s.compare_mode());
        assert!(!s.is_unnormalized());
    }

    #[test]
    fn create_unnormalized() {
        let (device, queue) = gfx_dev_and_queue!();

        let s =
            sampler::Sampler::unnormalized(device,
                                           sampler::Filter::Linear,
                                           sampler::UnnormalizedSamplerAddressMode::ClampToEdge,
                                           sampler::UnnormalizedSamplerAddressMode::ClampToEdge)
                .unwrap();

        assert!(!s.compare_mode());
        assert!(s.is_unnormalized());
    }

    #[test]
    fn simple_repeat_linear() {
        let (device, queue) = gfx_dev_and_queue!();
        let _ = sampler::Sampler::simple_repeat_linear(device);
    }

    #[test]
    fn simple_repeat_linear_no_mipmap() {
        let (device, queue) = gfx_dev_and_queue!();
        let _ = sampler::Sampler::simple_repeat_linear_no_mipmap(device);
    }

    #[test]
    fn min_lod_inferior() {
        let (device, queue) = gfx_dev_and_queue!();

        assert_should_panic!({
                                 let _ = sampler::Sampler::new(device,
                                                               sampler::Filter::Linear,
                                                               sampler::Filter::Linear,
                                                               sampler::MipmapMode::Nearest,
                                                               sampler::SamplerAddressMode::Repeat,
                                                               sampler::SamplerAddressMode::Repeat,
                                                               sampler::SamplerAddressMode::Repeat,
                                                               1.0,
                                                               1.0,
                                                               5.0,
                                                               2.0);
                             });
    }

    #[test]
    fn max_anisotropy() {
        let (device, queue) = gfx_dev_and_queue!();

        assert_should_panic!({
                                 let _ = sampler::Sampler::new(device,
                                                               sampler::Filter::Linear,
                                                               sampler::Filter::Linear,
                                                               sampler::MipmapMode::Nearest,
                                                               sampler::SamplerAddressMode::Repeat,
                                                               sampler::SamplerAddressMode::Repeat,
                                                               sampler::SamplerAddressMode::Repeat,
                                                               1.0,
                                                               0.5,
                                                               0.0,
                                                               2.0);
                             });
    }

    #[test]
    fn different_borders() {
        let (device, queue) = gfx_dev_and_queue!();

        let b1 = sampler::BorderColor::IntTransparentBlack;
        let b2 = sampler::BorderColor::FloatOpaqueWhite;

        assert_should_panic!({
            let _ = sampler::Sampler::new(device,
                                          sampler::Filter::Linear,
                                          sampler::Filter::Linear,
                                          sampler::MipmapMode::Nearest,
                                          sampler::SamplerAddressMode::ClampToBorder(b1),
                                          sampler::SamplerAddressMode::ClampToBorder(b2),
                                          sampler::SamplerAddressMode::Repeat,
                                          1.0,
                                          1.0,
                                          5.0,
                                          2.0);
        });
    }

    #[test]
    fn anisotropy_feature() {
        let (device, queue) = gfx_dev_and_queue!();

        let r = sampler::Sampler::new(device,
                                      sampler::Filter::Linear,
                                      sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      1.0,
                                      2.0,
                                      0.0,
                                      2.0);

        match r {
            Err(sampler::SamplerCreationError::SamplerAnisotropyFeatureNotEnabled) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn anisotropy_limit() {
        let (device, queue) = gfx_dev_and_queue!(sampler_anisotropy);

        let r = sampler::Sampler::new(device,
                                      sampler::Filter::Linear,
                                      sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      1.0,
                                      100000000.0,
                                      0.0,
                                      2.0);

        match r {
            Err(sampler::SamplerCreationError::AnisotropyLimitExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn mip_lod_bias_limit() {
        let (device, queue) = gfx_dev_and_queue!();

        let r = sampler::Sampler::new(device,
                                      sampler::Filter::Linear,
                                      sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      100000000.0,
                                      1.0,
                                      0.0,
                                      2.0);

        match r {
            Err(sampler::SamplerCreationError::MipLodBiasLimitExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn sampler_mirror_clamp_to_edge_extension() {
        let (device, queue) = gfx_dev_and_queue!();

        let r = sampler::Sampler::new(device,
                                      sampler::Filter::Linear,
                                      sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::MirrorClampToEdge,
                                      sampler::SamplerAddressMode::MirrorClampToEdge,
                                      sampler::SamplerAddressMode::MirrorClampToEdge,
                                      1.0,
                                      1.0,
                                      0.0,
                                      2.0);

        match r {
            Err(sampler::SamplerCreationError::SamplerMirrorClampToEdgeExtensionNotEnabled) => (),
            _ => panic!(),
        }
    }
}
