// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! How to retrieve data from a sampled image within a shader.
//!
//! When you retrieve data from a sampled image, you have to pass the coordinates of the pixel you
//! want to retrieve. The implementation then performs various calculations, and these operations
//! are what the `Sampler` object describes.
//!
//! # Level of detail
//!
//! The level-of-detail (LOD) is a floating-point value that expresses a sense of how much texture
//! detail is visible to the viewer. It is used in texture filtering and mipmapping calculations.
//!
//! LOD is calculated through one or more steps to produce a final value. The base LOD is
//! determined by one of two ways:
//! - Implicitly, by letting Vulkan calculate it automatically, based on factors such as number of
//!   pixels, distance and viewing angle. This is done using an `ImplicitLod` SPIR-V sampling
//!   operation, which corresponds to the `texture*` functions not suffixed with `Lod` in GLSL.
//! - Explicitly, specified in the shader. This is done using an `ExplicitLod` SPIR-V sampling
//!   operation, which corresponds to the `texture*Lod` functions in GLSL.
//!
//! It is possible to provide a *bias* to the base LOD value, which is simply added to it.
//! An LOD bias can be provided both in the sampler object and as part of the sampling operation in
//! the shader, and are combined by addition to produce the final bias value, which is then added to
//! the base LOD.
//!
//! Once LOD bias has been applied, the resulting value may be *clamped* to a minimum and maximum
//! value to provide the final LOD. A maximum may be specified by the sampler, while a minimum
//! can be specified by the sampler or the shader sampling operation.
//!
//! # Texel filtering
//!
//! Texel filtering operations determine how the color value to be sampled from each mipmap is
//! calculated. The filtering mode can be set independently for different signs of the LOD value:
//! - Negative or zero: **magnification**. The rendered object is closer to the viewer, and each
//!   pixel in the texture corresponds to exactly one or more than one framebuffer pixel.
//! - Positive: **minification**. The rendered object is further from the viewer, and each pixel in
//!   the texture corresponds to less than one framebuffer pixel.

use crate::check_errors;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::pipeline::graphics::depth_stencil::CompareOp;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use std::error;
use std::fmt;
use std::mem::MaybeUninit;
use std::ops::RangeInclusive;
use std::ptr;
use std::sync::Arc;

/// Describes how to retrieve data from a sampled image within a shader.
///
/// # Examples
///
/// A simple sampler for most usages:
///
/// ```
/// use vulkano::sampler::Sampler;
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let _sampler = Sampler::simple_repeat_linear_no_mipmap(device.clone());
/// ```
///
/// More detailed sampler creation:
///
/// ```
/// use vulkano::sampler::{Filter, Sampler, SamplerAddressMode};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let _sampler = Sampler::start(device.clone())
///     .filter(Filter::Linear)
///     .address_mode(SamplerAddressMode::Repeat)
///     .mip_lod_bias(1.0)
///     .lod(0.0..=100.0)
///     .build()
///     .unwrap();
/// ```
pub struct Sampler {
    handle: ash::vk::Sampler,
    device: Arc<Device>,
    compare_mode: bool,
    unnormalized: bool,
    usable_with_float_formats: bool,
    usable_with_int_formats: bool,
    usable_with_swizzling: bool,
}

impl Sampler {
    /// Starts constructing a new `Sampler`.
    pub fn start(device: Arc<Device>) -> SamplerBuilder {
        SamplerBuilder {
            device,

            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mipmap_mode: SamplerMipmapMode::Nearest,
            address_mode_u: SamplerAddressMode::ClampToEdge,
            address_mode_v: SamplerAddressMode::ClampToEdge,
            address_mode_w: SamplerAddressMode::ClampToEdge,
            mip_lod_bias: 0.0,
            anisotropy: None,
            compare: None,
            lod: 0.0..=0.0,
            border_color: BorderColor::FloatTransparentBlack,
            unnormalized_coordinates: false,
            reduction_mode: SamplerReductionMode::WeightedAverage,
        }
    }

    /// Shortcut for creating a sampler with linear sampling, linear mipmaps, and with the repeat
    /// mode for borders.
    ///
    /// Useful for prototyping, but can also be used in real projects.
    #[inline]
    pub fn simple_repeat_linear(device: Arc<Device>) -> Result<Arc<Sampler>, SamplerCreationError> {
        Sampler::start(device.clone())
            .filter(Filter::Linear)
            .mipmap_mode(SamplerMipmapMode::Linear)
            .address_mode(SamplerAddressMode::Repeat)
            .min_lod(0.0)
            .build()
    }

    /// Shortcut for creating a sampler with linear sampling, that only uses the main level of
    /// images, and with the repeat mode for borders.
    ///
    /// Useful for prototyping, but can also be used in real projects.
    #[inline]
    pub fn simple_repeat_linear_no_mipmap(
        device: Arc<Device>,
    ) -> Result<Arc<Sampler>, SamplerCreationError> {
        Sampler::start(device.clone())
            .filter(Filter::Linear)
            .address_mode(SamplerAddressMode::Repeat)
            .lod(0.0..=1.0)
            .build()
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

    /// Returns true if the sampler can be used with floating-point image views.
    #[inline]
    pub fn usable_with_float_formats(&self) -> bool {
        self.usable_with_float_formats
    }

    /// Returns true if the sampler can be used with integer image views.
    #[inline]
    pub fn usable_with_int_formats(&self) -> bool {
        self.usable_with_int_formats
    }

    /// Returns true if the sampler can be used with image views that have non-identity swizzling.
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
    type Object = ash::vk::Sampler;

    #[inline]
    fn internal_object(&self) -> ash::vk::Sampler {
        self.handle
    }
}

impl fmt::Debug for Sampler {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan sampler {:?}>", self.handle)
    }
}

impl PartialEq for Sampler {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl Eq for Sampler {}

impl Drop for Sampler {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_sampler(self.device.internal_object(), self.handle, ptr::null());
        }
    }
}

/// Used to construct a new `Sampler`.
#[derive(Clone, Debug)]
pub struct SamplerBuilder {
    device: Arc<Device>,

    mag_filter: Filter,
    min_filter: Filter,
    mipmap_mode: SamplerMipmapMode,
    address_mode_u: SamplerAddressMode,
    address_mode_v: SamplerAddressMode,
    address_mode_w: SamplerAddressMode,
    mip_lod_bias: f32,
    anisotropy: Option<f32>,
    compare: Option<CompareOp>,
    lod: RangeInclusive<f32>,
    border_color: BorderColor,
    unnormalized_coordinates: bool,
    reduction_mode: SamplerReductionMode,
}

impl SamplerBuilder {
    pub fn build(self) -> Result<Arc<Sampler>, SamplerCreationError> {
        let device = self.device;

        if [
            self.address_mode_u,
            self.address_mode_v,
            self.address_mode_w,
        ]
        .into_iter()
        .any(|mode| mode == SamplerAddressMode::MirrorClampToEdge)
        {
            if !device.enabled_features().sampler_mirror_clamp_to_edge
                && !device.enabled_extensions().khr_sampler_mirror_clamp_to_edge
            {
                if device
                    .physical_device()
                    .supported_features()
                    .sampler_mirror_clamp_to_edge
                {
                    return Err(SamplerCreationError::FeatureNotEnabled {
                        feature: "sampler_mirror_clamp_to_edge",
                        reason: "one or more address modes were MirrorClampToEdge",
                    });
                } else {
                    return Err(SamplerCreationError::ExtensionNotEnabled {
                        extension: "khr_sampler_mirror_clamp_to_edge",
                        reason: "one or more address modes were MirrorClampToEdge",
                    });
                }
            }
        }

        {
            let limit = device.physical_device().properties().max_sampler_lod_bias;
            if self.mip_lod_bias.abs() > limit {
                return Err(SamplerCreationError::MaxSamplerLodBiasExceeded {
                    requested: self.mip_lod_bias,
                    maximum: limit,
                });
            }
        }

        let (anisotropy_enable, max_anisotropy) = if let Some(max_anisotropy) = self.anisotropy {
            if !device.enabled_features().sampler_anisotropy {
                return Err(SamplerCreationError::FeatureNotEnabled {
                    feature: "sampler_anisotropy",
                    reason: "anisotropy was set to `Some`",
                });
            }

            let limit = device.physical_device().properties().max_sampler_anisotropy;
            if max_anisotropy > limit {
                return Err(SamplerCreationError::MaxSamplerAnisotropyExceeded {
                    requested: max_anisotropy,
                    maximum: limit,
                });
            }

            if [self.mag_filter, self.min_filter]
                .into_iter()
                .any(|filter| filter == Filter::Cubic)
            {
                return Err(SamplerCreationError::AnisotropyInvalidFilter {
                    mag_filter: self.mag_filter,
                    min_filter: self.min_filter,
                });
            }

            (ash::vk::TRUE, max_anisotropy)
        } else {
            (ash::vk::FALSE, 1.0)
        };

        let (compare_enable, compare_op) = if let Some(compare_op) = self.compare {
            if self.reduction_mode != SamplerReductionMode::WeightedAverage {
                return Err(SamplerCreationError::CompareInvalidReductionMode {
                    reduction_mode: self.reduction_mode,
                });
            }

            (ash::vk::TRUE, compare_op)
        } else {
            (ash::vk::FALSE, CompareOp::Never)
        };

        let border_color_used = [
            self.address_mode_u,
            self.address_mode_v,
            self.address_mode_w,
        ]
        .into_iter()
        .any(|mode| mode == SamplerAddressMode::ClampToBorder);

        if self.unnormalized_coordinates {
            if self.min_filter != self.mag_filter {
                return Err(
                    SamplerCreationError::UnnormalizedCoordinatesFiltersNotEqual {
                        mag_filter: self.mag_filter,
                        min_filter: self.min_filter,
                    },
                );
            }

            if self.mipmap_mode != SamplerMipmapMode::Nearest {
                return Err(
                    SamplerCreationError::UnnormalizedCoordinatesInvalidMipmapMode {
                        mipmap_mode: self.mipmap_mode,
                    },
                );
            }

            if self.lod != (0.0..=0.0) {
                return Err(SamplerCreationError::UnnormalizedCoordinatesNonzeroLod {
                    lod: self.lod.clone(),
                });
            }

            if [self.address_mode_u, self.address_mode_v]
                .into_iter()
                .any(|mode| {
                    !matches!(
                        mode,
                        SamplerAddressMode::ClampToEdge | SamplerAddressMode::ClampToBorder
                    )
                })
            {
                return Err(
                    SamplerCreationError::UnnormalizedCoordinatesInvalidAddressMode {
                        address_mode_u: self.address_mode_u,
                        address_mode_v: self.address_mode_v,
                    },
                );
            }

            if self.anisotropy.is_some() {
                return Err(SamplerCreationError::UnnormalizedCoordinatesAnisotropyEnabled);
            }

            if self.compare.is_some() {
                return Err(SamplerCreationError::UnnormalizedCoordinatesCompareEnabled);
            }
        }

        let mut sampler_reduction_mode_create_info =
            if device.enabled_features().sampler_filter_minmax
                || device.enabled_extensions().ext_sampler_filter_minmax
            {
                Some(ash::vk::SamplerReductionModeCreateInfo {
                    reduction_mode: self.reduction_mode.into(),
                    ..Default::default()
                })
            } else {
                if self.reduction_mode != SamplerReductionMode::WeightedAverage {
                    if device
                        .physical_device()
                        .supported_features()
                        .sampler_filter_minmax
                    {
                        return Err(SamplerCreationError::FeatureNotEnabled {
                            feature: "sampler_filter_minmax",
                            reason: "reduction_mode was not WeightedAverage",
                        });
                    } else {
                        return Err(SamplerCreationError::ExtensionNotEnabled {
                            extension: "ext_sampler_filter_minmax",
                            reason: "reduction_mode was not WeightedAverage",
                        });
                    }
                }

                None
            };

        let fns = device.fns();
        let handle = unsafe {
            let mut create_info = ash::vk::SamplerCreateInfo {
                flags: ash::vk::SamplerCreateFlags::empty(),
                mag_filter: self.mag_filter.into(),
                min_filter: self.min_filter.into(),
                mipmap_mode: self.mipmap_mode.into(),
                address_mode_u: self.address_mode_u.into(),
                address_mode_v: self.address_mode_v.into(),
                address_mode_w: self.address_mode_w.into(),
                mip_lod_bias: self.mip_lod_bias,
                anisotropy_enable,
                max_anisotropy,
                compare_enable,
                compare_op: compare_op.into(),
                min_lod: *self.lod.start(),
                max_lod: *self.lod.end(),
                border_color: self.border_color.into(),
                unnormalized_coordinates: self.unnormalized_coordinates as ash::vk::Bool32,
                ..Default::default()
            };

            if let Some(sampler_reduction_mode_create_info) =
                sampler_reduction_mode_create_info.as_mut()
            {
                sampler_reduction_mode_create_info.p_next = create_info.p_next;
                create_info.p_next = sampler_reduction_mode_create_info as *const _ as *const _;
            }

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_sampler(
                device.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Sampler {
            handle,
            device,
            compare_mode: self.compare.is_some(),
            unnormalized: self.unnormalized_coordinates,
            usable_with_float_formats: !border_color_used
                || matches!(
                    self.border_color,
                    BorderColor::FloatTransparentBlack
                        | BorderColor::FloatOpaqueBlack
                        | BorderColor::FloatOpaqueWhite
                ),
            usable_with_int_formats: (!border_color_used
                || matches!(
                    self.border_color,
                    BorderColor::IntTransparentBlack
                        | BorderColor::IntOpaqueBlack
                        | BorderColor::IntOpaqueWhite
                ))
                && self.compare.is_none(),
            usable_with_swizzling: !border_color_used
                || !matches!(
                    self.border_color,
                    BorderColor::FloatOpaqueBlack | BorderColor::IntOpaqueBlack
                ),
        }))
    }

    /// How the sampled value of a single mipmap should be calculated,
    /// for both magnification and minification.
    ///
    /// The default value is [`Nearest`](Filter::Nearest).
    #[inline]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.mag_filter = filter;
        self.min_filter = filter;
        self
    }

    /// How the sampled value of a single mipmap should be calculated,
    /// when magnification is applied (LOD <= 0.0).
    ///
    /// The default value is [`Nearest`](Filter::Nearest).
    #[inline]
    pub fn mag_filter(mut self, filter: Filter) -> Self {
        self.mag_filter = filter;
        self
    }

    /// How the sampled value of a single mipmap should be calculated,
    /// when minification is applied (LOD > 0.0).
    ///
    /// The default value is [`Nearest`](Filter::Nearest).
    #[inline]
    pub fn min_filter(mut self, filter: Filter) -> Self {
        self.min_filter = filter;
        self
    }

    /// How the final sampled value should be calculated from the samples of individual
    /// mipmaps.
    ///
    /// The default value is [`Nearest`](MipmapMode::Nearest).
    #[inline]
    pub fn mipmap_mode(mut self, mode: SamplerMipmapMode) -> Self {
        self.mipmap_mode = mode;
        self
    }

    /// How out-of-range texture coordinates should be treated, for all texture coordinate indices.
    ///
    /// The default value is [`ClampToEdge`](SamplerAddressMode::ClampToEdge).
    #[inline]
    pub fn address_mode(mut self, mode: SamplerAddressMode) -> Self {
        self.address_mode_u = mode;
        self.address_mode_v = mode;
        self.address_mode_w = mode;
        self
    }

    /// How out-of-range texture coordinates should be treated, for the u coordinate.
    ///
    /// The default value is [`ClampToEdge`](SamplerAddressMode::ClampToEdge).
    #[inline]
    pub fn address_mode_u(mut self, mode: SamplerAddressMode) -> Self {
        self.address_mode_u = mode;
        self
    }

    /// How out-of-range texture coordinates should be treated, for the v coordinate.
    ///
    /// The default value is [`ClampToEdge`](SamplerAddressMode::ClampToEdge).
    #[inline]
    pub fn address_mode_v(mut self, mode: SamplerAddressMode) -> Self {
        self.address_mode_v = mode;
        self
    }

    /// How out-of-range texture coordinates should be treated, for the w coordinate.
    ///
    /// The default value is [`ClampToEdge`](SamplerAddressMode::ClampToEdge).
    #[inline]
    pub fn address_mode_w(mut self, mode: SamplerAddressMode) -> Self {
        self.address_mode_w = mode;
        self
    }

    /// The bias value to be added to the base LOD before clamping.
    ///
    /// The absolute value of the provided value must not exceed the
    /// [`max_sampler_lod_bias`](crate::device::Properties::max_sampler_lod_bias) limit of the
    /// device.
    ///
    /// The default value is `0.0`.
    #[inline]
    pub fn mip_lod_bias(mut self, bias: f32) -> Self {
        self.mip_lod_bias = bias;
        self
    }

    /// Sets whether anisotropic texel filtering is enabled (`Some`) and provides the maximum
    /// anisotropy value if it is enabled.
    ///
    /// Anisotropic filtering is a special filtering mode that takes into account the differences in
    /// scaling between the horizontal and vertical framebuffer axes.
    ///
    /// If set to `Some`, the [`sampler_anisotropy`](crate::device::Features::sampler_anisotropy)
    /// feature must be enabled on the device, the provided maximum value must not exceed the
    /// [`max_sampler_anisotropy`](crate::device::Properties::max_sampler_anisotropy) limit, and
    /// the [`Cubic`](Filter::Cubic) filter must not be used.
    ///
    /// The default value is `None`.
    ///
    /// # Panics
    /// - Panics if `anisotropy` is `Some` and contains a value less than 1.0.
    #[inline]
    pub fn anisotropy(mut self, anisotropy: Option<f32>) -> Self {
        if let Some(max_anisotropy) = anisotropy {
            assert!(max_anisotropy >= 1.0);
        }

        self.anisotropy = anisotropy;
        self
    }

    /// Sets whether depth comparison is enabled (`Some`) and provides a comparison operator if it
    /// is enabled.
    ///
    /// Depth comparison is an alternative mode for samplers that can be used in combination with
    /// image views specifying the depth aspect. Instead of returning a value that is sampled from
    /// the image directly, a comparison operation is applied between the sampled value and a
    /// reference value that is specified as part of the operation. The result is binary: 1.0 if the
    /// operation returns `true`, 0.0 if it returns `false`.
    ///
    /// If set to `Some`, the `reduction_mode` must be set to
    /// [`WeightedAverage`](SamplerReductionMode::WeightedAverage).
    ///
    /// The default value is `None`.
    #[inline]
    pub fn compare(mut self, compare: Option<CompareOp>) -> Self {
        self.compare = compare;
        self
    }

    /// The range that LOD values must be clamped to.
    ///
    /// The default value is `0.0..`.
    ///
    /// # Panics
    /// - Panics if `range` is empty.
    #[inline]
    pub fn lod(mut self, range: RangeInclusive<f32>) -> Self {
        assert!(!range.is_empty());
        self.lod = range;
        self
    }

    /// The minimum value that LOD values must be clamped to. The maximum LOD is left unbounded.
    ///
    /// The default value is `0.0..`.
    ///
    /// # Panics
    /// - Panics if `min` is greater than 1000.0.
    #[inline]
    pub fn min_lod(mut self, min: f32) -> Self {
        assert!(min <= ash::vk::LOD_CLAMP_NONE);
        self.lod = min..=ash::vk::LOD_CLAMP_NONE;
        self
    }

    /// The border color to use if `address_mode` is set to
    /// [`ClampToBorder`](SamplerAddressMode::ClampToBorder).
    ///
    /// The default value is [`FloatTransparentBlack`](BorderColor::FloatTransparentBlack).
    #[inline]
    pub fn border_color(mut self, border_color: BorderColor) -> Self {
        self.border_color = border_color;
        self
    }

    /// Sets whether unnormalized texture coordinates are enabled.
    ///
    /// When a sampler is set to use unnormalized coordinates as input, the texture coordinates are
    /// not scaled by the size of the image, and therefore range up to the size of the image rather
    /// than 1.0. Enabling this comes with several restrictions:
    /// - `min_filter` and `mag_filter` must be equal.
    /// - `mipmap_mode` must be [`Nearest`](MipmapMode::Nearest).
    /// - The `lod` range must be `0.0..=0.0`.
    /// - `address_mode` for u and v must be either
    ///   [`ClampToEdge`](`SamplerAddressMode::ClampToEdge`) or
    ///   [`ClampToBorder`](`SamplerAddressMode::ClampToBorder`).
    /// - Anisotropy and depth comparison must be disabled.
    ///
    /// Some restrictions also apply to the image view being sampled:
    /// - The view type must be [`Dim1d`](crate::image::view::ImageViewType::Dim1d) or
    ///   [`Dim2d`](crate::image::view::ImageViewType::Dim2d). Arrayed types are not allowed.
    /// - It must have a single mipmap level.
    ///
    /// Finally, restrictions apply to the sampling operations that can be used in a shader:
    /// - Only explicit LOD operations are allowed, implicit LOD operations are not.
    /// - Sampling with projection is not allowed.
    /// - Sampling with an LOD bias is not allowed.
    /// - Sampling with an offset is not allowed.
    ///
    /// The default value is `false`.
    #[inline]
    pub fn unnormalized_coordinates(mut self, enable: bool) -> Self {
        self.unnormalized_coordinates = enable;
        self
    }

    /// Sets how the value sampled from a mipmap should be calculated from the selected
    /// pixels, for the `Linear` and `Cubic` filters.
    ///
    /// The default value is [`WeightedAverage`](SamplerReductionMode::WeightedAverage).
    #[inline]
    pub fn reduction_mode(mut self, mode: SamplerReductionMode) -> Self {
        self.reduction_mode = mode;
        self
    }
}

/// Describes how the color of each pixel should be determined.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Filter {
    /// The pixel whose center is nearest to the requested coordinates is taken from the source
    /// and its value is returned as-is.
    Nearest = ash::vk::Filter::NEAREST.as_raw(),

    /// The 8/4/2 pixels (depending on view dimensionality) whose center surround the requested
    /// coordinates are taken, then their values are combined according to the chosen
    /// `reduction_mode`.
    Linear = ash::vk::Filter::LINEAR.as_raw(),

    /// The 64/16/4 pixels (depending on the view dimensionality) whose center surround the
    /// requested coordinates are taken, then their values are combined according to the chosen
    /// `reduction_mode`.
    ///
    /// The [`ext_filter_cubic`](crate::device::DeviceExtensions::ext_filter_cubic) extension must
    /// be enabled on the device, and anisotropy must be disabled. Sampled image views must have
    /// a type of [`Dim2d`](crate::image::view::ImageViewType::Dim2d).
    Cubic = ash::vk::Filter::CUBIC_EXT.as_raw(),
}

impl From<Filter> for ash::vk::Filter {
    #[inline]
    fn from(val: Filter) -> Self {
        Self::from_raw(val as i32)
    }
}

/// Describes which mipmap from the source to use.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SamplerMipmapMode {
    /// Use the mipmap whose dimensions are the nearest to the dimensions of the destination.
    Nearest = ash::vk::SamplerMipmapMode::NEAREST.as_raw(),

    /// Take the mipmap whose dimensions are no greater than that of the destination together
    /// with the next higher level mipmap, calculate the value for both, and interpolate them.
    Linear = ash::vk::SamplerMipmapMode::LINEAR.as_raw(),
}

impl From<SamplerMipmapMode> for ash::vk::SamplerMipmapMode {
    #[inline]
    fn from(val: SamplerMipmapMode) -> Self {
        Self::from_raw(val as i32)
    }
}

/// How the sampler should behave when it needs to access a pixel that is out of range of the
/// texture.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SamplerAddressMode {
    /// Repeat the texture. In other words, the pixel at coordinate `x + 1.0` is the same as the
    /// one at coordinate `x`.
    Repeat = ash::vk::SamplerAddressMode::REPEAT.as_raw(),

    /// Repeat the texture but mirror it at every repetition. In other words, the pixel at
    /// coordinate `x + 1.0` is the same as the one at coordinate `1.0 - x`.
    MirroredRepeat = ash::vk::SamplerAddressMode::MIRRORED_REPEAT.as_raw(),

    /// The coordinates are clamped to the valid range. Coordinates below 0.0 have the same value
    /// as coordinate 0.0. Coordinates over 1.0 have the same value as coordinate 1.0.
    ClampToEdge = ash::vk::SamplerAddressMode::CLAMP_TO_EDGE.as_raw(),

    /// Any pixel out of range is colored using the colour selected with the `border_color` on the
    /// `SamplerBuilder`.
    ///
    /// When this mode is chosen, the numeric type of the image view's format must match the border
    /// color. When using a floating-point border color, the sampler can only be used with
    /// floating-point or depth image views. When using an integer border color, the sampler can
    /// only be used with integer or stencil image views. In addition to this, you can't use an
    /// opaque black border color with an image view that uses component swizzling.
    ClampToBorder = ash::vk::SamplerAddressMode::CLAMP_TO_BORDER.as_raw(),

    /// Similar to `MirroredRepeat`, except that coordinates are clamped to the range
    /// `[-1.0, 1.0]`.
    ///
    /// The [`sampler_mirror_clamp_to_edge`](crate::device::Features::sampler_mirror_clamp_to_edge)
    /// feature or the
    /// [`khr_sampler_mirror_clamp_to_edge`](crate::device::DeviceExtensions::khr_sampler_mirror_clamp_to_edge)
    /// extension must be enabled on the device.
    MirrorClampToEdge = ash::vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE.as_raw(),
}

impl From<SamplerAddressMode> for ash::vk::SamplerAddressMode {
    #[inline]
    fn from(val: SamplerAddressMode) -> Self {
        Self::from_raw(val as i32)
    }
}

/// The color to use for the border of an image.
///
/// Only relevant if you use `ClampToBorder`.
///
/// Using a border color restricts the sampler to either floating-point images or integer images.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum BorderColor {
    /// The value `(0.0, 0.0, 0.0, 0.0)`. Can only be used with floating-point images.
    FloatTransparentBlack = ash::vk::BorderColor::FLOAT_TRANSPARENT_BLACK.as_raw(),

    /// The value `(0, 0, 0, 0)`. Can only be used with integer images.
    IntTransparentBlack = ash::vk::BorderColor::INT_TRANSPARENT_BLACK.as_raw(),

    /// The value `(0.0, 0.0, 0.0, 1.0)`. Can only be used with floating-point identity-swizzled
    /// images.
    FloatOpaqueBlack = ash::vk::BorderColor::FLOAT_OPAQUE_BLACK.as_raw(),

    /// The value `(0, 0, 0, 1)`. Can only be used with integer identity-swizzled images.
    IntOpaqueBlack = ash::vk::BorderColor::INT_OPAQUE_BLACK.as_raw(),

    /// The value `(1.0, 1.0, 1.0, 1.0)`. Can only be used with floating-point images.
    FloatOpaqueWhite = ash::vk::BorderColor::FLOAT_OPAQUE_WHITE.as_raw(),

    /// The value `(1, 1, 1, 1)`. Can only be used with integer images.
    IntOpaqueWhite = ash::vk::BorderColor::INT_OPAQUE_WHITE.as_raw(),
}

impl From<BorderColor> for ash::vk::BorderColor {
    #[inline]
    fn from(val: BorderColor) -> Self {
        Self::from_raw(val as i32)
    }
}

/// Describes how the value sampled from a mipmap should be calculated from the selected
/// pixels, for the `Linear` and `Cubic` filters.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SamplerReductionMode {
    /// Calculates a weighted average of the selected pixels. For `Linear` filtering the pixels
    /// are evenly weighted, for `Cubic` filtering they use Catmull-Rom weights.
    WeightedAverage = ash::vk::SamplerReductionMode::WEIGHTED_AVERAGE.as_raw(),

    /// Calculates the minimum of the selected pixels.
    ///
    /// The [`sampler_filter_minmax`](crate::device::Features::sampler_filter_minmax)
    /// feature or the
    /// [`ext_sampler_filter_minmax`](crate::device::DeviceExtensions::ext_sampler_filter_minmax)
    /// extension must be enabled on the device.
    Min = ash::vk::SamplerReductionMode::MIN.as_raw(),

    /// Calculates the maximum of the selected pixels.
    ///
    /// The [`sampler_filter_minmax`](crate::device::Features::sampler_filter_minmax)
    /// feature or the
    /// [`ext_sampler_filter_minmax`](crate::device::DeviceExtensions::ext_sampler_filter_minmax)
    /// extension must be enabled on the device.
    Max = ash::vk::SamplerReductionMode::MAX.as_raw(),
}

impl From<SamplerReductionMode> for ash::vk::SamplerReductionMode {
    #[inline]
    fn from(val: SamplerReductionMode) -> Self {
        Self::from_raw(val as i32)
    }
}

/// Error that can happen when creating an instance.
#[derive(Clone, Debug, PartialEq)]
pub enum SamplerCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// Too many sampler objects have been created. You must destroy some before creating new ones.
    /// Note the specs guarantee that at least 4000 samplers can exist simultaneously.
    TooManyObjects,

    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },
    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
    },

    /// Anisotropy was enabled with an invalid filter.
    AnisotropyInvalidFilter {
        mag_filter: Filter,
        min_filter: Filter,
    },

    /// Depth comparison was enabled with an invalid reduction mode.
    CompareInvalidReductionMode {
        reduction_mode: SamplerReductionMode,
    },

    /// The requested anisotropy level exceeds the device's limits.
    MaxSamplerAnisotropyExceeded {
        /// The value that was requested.
        requested: f32,
        /// The maximum supported value.
        maximum: f32,
    },

    /// The requested mip lod bias exceeds the device's limits.
    MaxSamplerLodBiasExceeded {
        /// The value that was requested.
        requested: f32,
        /// The maximum supported value.
        maximum: f32,
    },

    /// Unnormalized coordinates were enabled together with anisotropy.
    UnnormalizedCoordinatesAnisotropyEnabled,

    /// Unnormalized coordinates were enabled together with depth comparison.
    UnnormalizedCoordinatesCompareEnabled,

    /// Unnormalized coordinates were enabled, but the min and mag filters were not equal.
    UnnormalizedCoordinatesFiltersNotEqual {
        mag_filter: Filter,
        min_filter: Filter,
    },

    /// Unnormalized coordinates were enabled, but the address mode for u or v was something other
    /// than `ClampToEdge` or `ClampToBorder`.
    UnnormalizedCoordinatesInvalidAddressMode {
        address_mode_u: SamplerAddressMode,
        address_mode_v: SamplerAddressMode,
    },

    /// Unnormalized coordinates were enabled, but the mipmap mode was not `Nearest`.
    UnnormalizedCoordinatesInvalidMipmapMode { mipmap_mode: SamplerMipmapMode },

    /// Unnormalized coordinates were enabled, but the LOD range was not zero.
    UnnormalizedCoordinatesNonzeroLod { lod: RangeInclusive<f32> },
}

impl error::Error for SamplerCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            SamplerCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SamplerCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::OomError(_) => write!(fmt, "not enough memory available"),
            Self::TooManyObjects => write!(fmt, "too many simultaneous sampler objects",),
            Self::ExtensionNotEnabled { extension, reason } => write!(
                fmt,
                "the extension {} must be enabled: {}",
                extension, reason
            ),
            Self::FeatureNotEnabled { feature, reason } => {
                write!(fmt, "the feature {} must be enabled: {}", feature, reason)
            }
            Self::AnisotropyInvalidFilter { .. } => write!(fmt, "anisotropy was enabled with an invalid filter"),
            Self::CompareInvalidReductionMode { .. } => write!(fmt, "depth comparison was enabled with an invalid reduction mode"),
            Self::MaxSamplerAnisotropyExceeded { .. } => {
                write!(fmt, "max_sampler_anisotropy limit exceeded")
            }
            Self::MaxSamplerLodBiasExceeded { .. } => write!(fmt, "mip lod bias limit exceeded"),
            Self::UnnormalizedCoordinatesAnisotropyEnabled => write!(
                fmt,
                "unnormalized coordinates were enabled together with anisotropy"
            ),
            Self::UnnormalizedCoordinatesCompareEnabled => write!(
                fmt,
                "unnormalized coordinates were enabled together with depth comparison"
            ),
            Self::UnnormalizedCoordinatesFiltersNotEqual { .. } => write!(
                fmt,
                "unnormalized coordinates were enabled, but the min and mag filters were not equal"
            ),
            Self::UnnormalizedCoordinatesInvalidAddressMode { .. } => write!(
                fmt,
                "unnormalized coordinates were enabled, but the address mode for u or v was something other than `ClampToEdge` or `ClampToBorder`"
            ),
            Self::UnnormalizedCoordinatesInvalidMipmapMode { .. } => write!(
                fmt,
                "unnormalized coordinates were enabled, but the mipmap mode was not `Nearest`"
            ),
            Self::UnnormalizedCoordinatesNonzeroLod { .. } => write!(
                fmt,
                "unnormalized coordinates were enabled, but the LOD range was not zero"
            ),
        }
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
    use crate::{
        pipeline::graphics::depth_stencil::CompareOp,
        sampler::{
            Filter, Sampler, SamplerAddressMode, SamplerCreationError, SamplerReductionMode,
        },
    };

    #[test]
    fn create_regular() {
        let (device, queue) = gfx_dev_and_queue!();

        let s = Sampler::start(device)
            .filter(Filter::Linear)
            .address_mode(SamplerAddressMode::Repeat)
            .mip_lod_bias(1.0)
            .lod(0.0..=2.0)
            .build()
            .unwrap();
        assert!(!s.compare_mode());
        assert!(!s.is_unnormalized());
    }

    #[test]
    fn create_compare() {
        let (device, queue) = gfx_dev_and_queue!();

        let s = Sampler::start(device)
            .filter(Filter::Linear)
            .address_mode(SamplerAddressMode::Repeat)
            .mip_lod_bias(1.0)
            .compare(Some(CompareOp::Less))
            .lod(0.0..=2.0)
            .build()
            .unwrap();
        assert!(s.compare_mode());
        assert!(!s.is_unnormalized());
    }

    #[test]
    fn create_unnormalized() {
        let (device, queue) = gfx_dev_and_queue!();

        let s = Sampler::start(device)
            .filter(Filter::Linear)
            .unnormalized_coordinates(true)
            .build()
            .unwrap();
        assert!(!s.compare_mode());
        assert!(s.is_unnormalized());
    }

    #[test]
    fn simple_repeat_linear() {
        let (device, queue) = gfx_dev_and_queue!();
        let _ = Sampler::simple_repeat_linear(device);
    }

    #[test]
    fn simple_repeat_linear_no_mipmap() {
        let (device, queue) = gfx_dev_and_queue!();
        let _ = Sampler::simple_repeat_linear_no_mipmap(device);
    }

    #[test]
    fn min_lod_inferior() {
        let (device, queue) = gfx_dev_and_queue!();

        assert_should_panic!({
            let _ = Sampler::start(device)
                .filter(Filter::Linear)
                .address_mode(SamplerAddressMode::Repeat)
                .mip_lod_bias(1.0)
                .lod(5.0..=2.0)
                .build();
        });
    }

    #[test]
    fn max_anisotropy() {
        let (device, queue) = gfx_dev_and_queue!();

        assert_should_panic!({
            let _ = Sampler::start(device)
                .filter(Filter::Linear)
                .address_mode(SamplerAddressMode::Repeat)
                .mip_lod_bias(1.0)
                .anisotropy(Some(0.5))
                .lod(0.0..=2.0)
                .build();
        });
    }

    #[test]
    fn anisotropy_feature() {
        let (device, queue) = gfx_dev_and_queue!();

        let r = Sampler::start(device)
            .filter(Filter::Linear)
            .address_mode(SamplerAddressMode::Repeat)
            .mip_lod_bias(1.0)
            .anisotropy(Some(2.0))
            .lod(0.0..=2.0)
            .build();

        match r {
            Err(SamplerCreationError::FeatureNotEnabled {
                feature: "sampler_anisotropy",
                ..
            }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn anisotropy_limit() {
        let (device, queue) = gfx_dev_and_queue!(sampler_anisotropy);

        let r = Sampler::start(device)
            .filter(Filter::Linear)
            .address_mode(SamplerAddressMode::Repeat)
            .mip_lod_bias(1.0)
            .anisotropy(Some(100000000.0))
            .lod(0.0..=2.0)
            .build();

        match r {
            Err(SamplerCreationError::MaxSamplerAnisotropyExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn mip_lod_bias_limit() {
        let (device, queue) = gfx_dev_and_queue!();

        let r = Sampler::start(device)
            .filter(Filter::Linear)
            .address_mode(SamplerAddressMode::Repeat)
            .mip_lod_bias(100000000.0)
            .lod(0.0..=2.0)
            .build();

        match r {
            Err(SamplerCreationError::MaxSamplerLodBiasExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn sampler_mirror_clamp_to_edge_extension() {
        let (device, queue) = gfx_dev_and_queue!();

        let r = Sampler::start(device)
            .filter(Filter::Linear)
            .address_mode(SamplerAddressMode::MirrorClampToEdge)
            .mip_lod_bias(1.0)
            .lod(0.0..=2.0)
            .build();

        match r {
            Err(
                SamplerCreationError::FeatureNotEnabled {
                    feature: "sampler_mirror_clamp_to_edge",
                    ..
                }
                | SamplerCreationError::ExtensionNotEnabled {
                    extension: "khr_sampler_mirror_clamp_to_edge",
                    ..
                },
            ) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn sampler_filter_minmax_extension() {
        let (device, queue) = gfx_dev_and_queue!();

        let r = Sampler::start(device)
            .filter(Filter::Linear)
            .reduction_mode(SamplerReductionMode::Min)
            .build();

        match r {
            Err(
                SamplerCreationError::FeatureNotEnabled {
                    feature: "sampler_filter_minmax",
                    ..
                }
                | SamplerCreationError::ExtensionNotEnabled {
                    extension: "ext_sampler_filter_minmax",
                    ..
                },
            ) => (),
            _ => panic!(),
        }
    }
}
