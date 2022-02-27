// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Conversion from sampled YCbCr image data to RGB shader data.
//!
//! A sampler YCbCr conversion is an object that assists a sampler when converting from YCbCr
//! formats and/or YCbCr texel input data. It is used to read frames of video data within a shader,
//! possibly to apply it as texture on a rendered primitive. Sampler YCbCr conversion can only be
//! used with certain formats, and conversely, some formats require the use of a sampler YCbCr
//! conversion to be sampled at all.
//!
//! A sampler YCbCr conversion can only be used with a combined image sampler descriptor in a
//! descriptor set. The conversion must be attached on both the image view and sampler in the
//! descriptor, and the sampler must be included in the descriptor set layout as an immutable
//! sampler.
//!
//! # Examples
//!
//! ```
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! # let image_data: Vec<u8> = return;
//! # let queue: std::sync::Arc<vulkano::device::Queue> = return;
//! use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
//! use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
//! use vulkano::format::Format;
//! use vulkano::image::{ImmutableImage, ImageCreateFlags, ImageDimensions, ImageUsage, MipmapsCount};
//! use vulkano::image::view::{ImageView, ImageViewCreateInfo};
//! use vulkano::sampler::{Sampler, SamplerCreateInfo};
//! use vulkano::sampler::ycbcr::{SamplerYcbcrConversion, SamplerYcbcrConversionCreateInfo, SamplerYcbcrModelConversion};
//! use vulkano::shader::ShaderStage;
//!
//! let conversion = SamplerYcbcrConversion::new(device.clone(), SamplerYcbcrConversionCreateInfo {
//!     format: Some(Format::G8_B8_R8_3PLANE_420_UNORM),
//!     ycbcr_model: SamplerYcbcrModelConversion::YcbcrIdentity,
//!     ..Default::default()
//! })
//! .unwrap();
//!
//! let sampler = Sampler::new(device.clone(), SamplerCreateInfo {
//!     sampler_ycbcr_conversion: Some(conversion.clone()),
//!     ..Default::default()
//! })
//! .unwrap();
//!
//! let descriptor_set_layout = DescriptorSetLayout::new(
//!     device.clone(),
//!         DescriptorSetLayoutCreateInfo {
//!         bindings: [(
//!             0,
//!             DescriptorSetLayoutBinding {
//!                 stages: ShaderStage::Fragment.into(),
//!                 immutable_samplers: vec![sampler],
//!                 ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
//!             },
//!         )]
//!         .into(),
//!         ..Default::default()
//!     },
//! ).unwrap();
//!
//! let (image, future) = ImmutableImage::from_iter(
//!     image_data,
//!     ImageDimensions::Dim2d { width: 1920, height: 1080, array_layers: 1 },
//!     MipmapsCount::One,
//!     Format::G8_B8_R8_3PLANE_420_UNORM,
//!     queue.clone(),
//! ).unwrap();
//!
//! let create_info = ImageViewCreateInfo {
//!     sampler_ycbcr_conversion: Some(conversion.clone()),
//!     ..ImageViewCreateInfo::from_image(&image)
//! };
//! let image_view = ImageView::new(image, create_info).unwrap();
//!
//! let descriptor_set = PersistentDescriptorSet::new(
//!     descriptor_set_layout.clone(),
//!     [WriteDescriptorSet::image_view(0, image_view)],
//! ).unwrap();
//! ```

use crate::{
    check_errors,
    device::{Device, DeviceOwned},
    format::{ChromaSampling, Format, NumericType},
    sampler::{ComponentMapping, ComponentSwizzle, Filter},
    Error, OomError, Version, VulkanObject,
};
use std::{
    error, fmt,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ptr,
    sync::Arc,
};

/// Describes how sampled image data should converted from a YCbCr representation to an RGB one.
#[derive(Debug)]
pub struct SamplerYcbcrConversion {
    handle: ash::vk::SamplerYcbcrConversion,
    device: Arc<Device>,

    format: Option<Format>,
    ycbcr_model: SamplerYcbcrModelConversion,
    ycbcr_range: SamplerYcbcrRange,
    component_mapping: ComponentMapping,
    chroma_offset: [ChromaLocation; 2],
    chroma_filter: Filter,
    force_explicit_reconstruction: bool,
}

impl SamplerYcbcrConversion {
    /// Creates a new `SamplerYcbcrConversion`.
    ///
    /// The [`sampler_ycbcr_conversion`](crate::device::Features::sampler_ycbcr_conversion)
    /// feature must be enabled on the device.
    pub fn new(
        device: Arc<Device>,
        create_info: SamplerYcbcrConversionCreateInfo,
    ) -> Result<Arc<SamplerYcbcrConversion>, SamplerYcbcrConversionCreationError> {
        let SamplerYcbcrConversionCreateInfo {
            format,
            ycbcr_model,
            ycbcr_range,
            component_mapping,
            chroma_offset,
            chroma_filter,
            force_explicit_reconstruction,
            _ne: _,
        } = create_info;

        if !device.enabled_features().sampler_ycbcr_conversion {
            return Err(SamplerYcbcrConversionCreationError::FeatureNotEnabled {
                feature: "sampler_ycbcr_conversion",
                reason: "tried to create a SamplerYcbcrConversion",
            });
        }

        let format = match format {
            Some(f) => f,
            None => {
                return Err(SamplerYcbcrConversionCreationError::FormatMissing);
            }
        };

        // VUID-VkSamplerYcbcrConversionCreateInfo-format-04061
        if !format
            .type_color()
            .map_or(false, |ty| ty == NumericType::UNORM)
        {
            return Err(SamplerYcbcrConversionCreationError::FormatNotUnorm);
        }

        let potential_format_features = device
            .physical_device()
            .format_properties(format)
            .potential_format_features();

        // VUID-VkSamplerYcbcrConversionCreateInfo-format-01650
        if !(potential_format_features.midpoint_chroma_samples
            || potential_format_features.cosited_chroma_samples)
        {
            return Err(SamplerYcbcrConversionCreationError::FormatNotSupported);
        }

        if let Some(chroma_sampling @ (ChromaSampling::Mode422 | ChromaSampling::Mode420)) =
            format.ycbcr_chroma_sampling()
        {
            let chroma_offsets_to_check = match chroma_sampling {
                ChromaSampling::Mode420 => &chroma_offset[0..2],
                ChromaSampling::Mode422 => &chroma_offset[0..1],
                _ => unreachable!(),
            };

            for offset in chroma_offsets_to_check {
                match offset {
                    ChromaLocation::CositedEven => {
                        // VUID-VkSamplerYcbcrConversionCreateInfo-xChromaOffset-01651
                        if !potential_format_features.cosited_chroma_samples {
                            return Err(
                                SamplerYcbcrConversionCreationError::FormatChromaOffsetNotSupported,
                            );
                        }
                    }
                    ChromaLocation::Midpoint => {
                        // VUID-VkSamplerYcbcrConversionCreateInfo-xChromaOffset-01652
                        if !potential_format_features.midpoint_chroma_samples {
                            return Err(
                                SamplerYcbcrConversionCreationError::FormatChromaOffsetNotSupported,
                            );
                        }
                    }
                }
            }

            // VUID-VkSamplerYcbcrConversionCreateInfo-components-02581
            let g_ok = component_mapping.g_is_identity();

            // VUID-VkSamplerYcbcrConversionCreateInfo-components-02582
            let a_ok = component_mapping.a_is_identity()
                || matches!(
                    component_mapping.a,
                    ComponentSwizzle::One | ComponentSwizzle::Zero
                );

            // VUID-VkSamplerYcbcrConversionCreateInfo-components-02583
            // VUID-VkSamplerYcbcrConversionCreateInfo-components-02584
            // VUID-VkSamplerYcbcrConversionCreateInfo-components-02585
            let rb_ok1 = component_mapping.r_is_identity() && component_mapping.b_is_identity();
            let rb_ok2 = matches!(component_mapping.r, ComponentSwizzle::Blue)
                && matches!(component_mapping.b, ComponentSwizzle::Red);

            if !(g_ok && a_ok && (rb_ok1 || rb_ok2)) {
                return Err(SamplerYcbcrConversionCreationError::FormatInvalidComponentMapping);
            }
        }

        let components_bits = {
            let bits = format.components();
            component_mapping
                .component_map()
                .map(move |i| i.map(|i| bits[i]))
        };

        // VUID-VkSamplerYcbcrConversionCreateInfo-ycbcrModel-01655
        if ycbcr_model != SamplerYcbcrModelConversion::RgbIdentity
            && !components_bits[0..3]
                .iter()
                .all(|b| b.map_or(false, |b| b != 0))
        {
            return Err(SamplerYcbcrConversionCreationError::YcbcrModelInvalidComponentMapping);
        }

        // VUID-VkSamplerYcbcrConversionCreateInfo-ycbcrRange-02748
        if ycbcr_range == SamplerYcbcrRange::ItuNarrow {
            // TODO: Spec doesn't say how many bits `Zero` and `One` are considered to have, so
            // just skip them for now.
            for &bits in components_bits[0..3].iter().flatten() {
                if bits < 8 {
                    return Err(SamplerYcbcrConversionCreationError::YcbcrRangeFormatNotEnoughBits);
                }
            }
        }

        // VUID-VkSamplerYcbcrConversionCreateInfo-forceExplicitReconstruction-01656
        if force_explicit_reconstruction
            && !potential_format_features
                .sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable
        {
            return Err(
                SamplerYcbcrConversionCreationError::FormatForceExplicitReconstructionNotSupported,
            );
        }

        match chroma_filter {
            Filter::Nearest => (),
            Filter::Linear => {
                // VUID-VkSamplerYcbcrConversionCreateInfo-chromaFilter-01657
                if !potential_format_features.sampled_image_ycbcr_conversion_linear_filter {
                    return Err(
                        SamplerYcbcrConversionCreationError::FormatLinearFilterNotSupported,
                    );
                }
            }
            Filter::Cubic => {
                return Err(SamplerYcbcrConversionCreationError::CubicFilterNotSupported);
            }
        }

        let create_info = ash::vk::SamplerYcbcrConversionCreateInfo {
            format: format.into(),
            ycbcr_model: ycbcr_model.into(),
            ycbcr_range: ycbcr_range.into(),
            components: component_mapping.into(),
            x_chroma_offset: chroma_offset[0].into(),
            y_chroma_offset: chroma_offset[1].into(),
            chroma_filter: chroma_filter.into(),
            force_explicit_reconstruction: force_explicit_reconstruction as ash::vk::Bool32,
            ..Default::default()
        };

        let handle = unsafe {
            let fns = device.fns();
            let create_sampler_ycbcr_conversion = if device.api_version() >= Version::V1_1 {
                fns.v1_1.create_sampler_ycbcr_conversion
            } else {
                fns.khr_sampler_ycbcr_conversion
                    .create_sampler_ycbcr_conversion_khr
            };

            let mut output = MaybeUninit::uninit();
            check_errors(create_sampler_ycbcr_conversion(
                device.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(SamplerYcbcrConversion {
            handle,
            device,
            format: Some(format),
            ycbcr_model,
            ycbcr_range,
            component_mapping,
            chroma_offset,
            chroma_filter,
            force_explicit_reconstruction,
        }))
    }

    /// Returns the chroma filter used by the conversion.
    #[inline]
    pub fn chroma_filter(&self) -> Filter {
        self.chroma_filter
    }

    /// Returns the chroma offsets used by the conversion.
    #[inline]
    pub fn chroma_offset(&self) -> [ChromaLocation; 2] {
        self.chroma_offset
    }

    /// Returns the component mapping of the conversion.
    #[inline]
    pub fn component_mapping(&self) -> ComponentMapping {
        self.component_mapping
    }

    /// Returns whether the conversion has forced explicit reconstruction to be enabled.
    #[inline]
    pub fn force_explicit_reconstruction(&self) -> bool {
        self.force_explicit_reconstruction
    }

    /// Returns the format that the conversion was created for.
    #[inline]
    pub fn format(&self) -> Option<Format> {
        self.format
    }

    /// Returns the YCbCr model of the conversion.
    #[inline]
    pub fn ycbcr_model(&self) -> SamplerYcbcrModelConversion {
        self.ycbcr_model
    }

    /// Returns the YCbCr range of the conversion.
    #[inline]
    pub fn ycbcr_range(&self) -> SamplerYcbcrRange {
        self.ycbcr_range
    }

    /// Returns whether `self` is equal or identically defined to `other`.
    #[inline]
    pub fn is_identical(&self, other: &SamplerYcbcrConversion) -> bool {
        self.handle == other.handle || {
            let &Self {
                handle: _,
                device: _,
                format,
                ycbcr_model,
                ycbcr_range,
                component_mapping,
                chroma_offset,
                chroma_filter,
                force_explicit_reconstruction,
            } = self;

            format == other.format
                && ycbcr_model == other.ycbcr_model
                && ycbcr_range == other.ycbcr_range
                && component_mapping == other.component_mapping
                && chroma_offset == other.chroma_offset
                && chroma_filter == other.chroma_filter
                && force_explicit_reconstruction == other.force_explicit_reconstruction
        }
    }
}

impl Drop for SamplerYcbcrConversion {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            let destroy_sampler_ycbcr_conversion = if self.device.api_version() >= Version::V1_1 {
                fns.v1_1.destroy_sampler_ycbcr_conversion
            } else {
                fns.khr_sampler_ycbcr_conversion
                    .destroy_sampler_ycbcr_conversion_khr
            };

            destroy_sampler_ycbcr_conversion(
                self.device.internal_object(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

unsafe impl VulkanObject for SamplerYcbcrConversion {
    type Object = ash::vk::SamplerYcbcrConversion;

    #[inline]
    fn internal_object(&self) -> ash::vk::SamplerYcbcrConversion {
        self.handle
    }
}

unsafe impl DeviceOwned for SamplerYcbcrConversion {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for SamplerYcbcrConversion {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for SamplerYcbcrConversion {}

impl Hash for SamplerYcbcrConversion {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Error that can happen when creating a `SamplerYcbcrConversion`.
#[derive(Clone, Debug, PartialEq)]
pub enum SamplerYcbcrConversionCreationError {
    /// Not enough memory.
    OomError(OomError),

    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
    },

    /// The `Cubic` filter was specified.
    CubicFilterNotSupported,

    /// No format was specified when one was required.
    FormatMissing,

    /// The format has a color type other than `UNORM`.
    FormatNotUnorm,

    /// The format does not support sampler YCbCr conversion.
    FormatNotSupported,

    /// The format does not support the chosen chroma offsets.
    FormatChromaOffsetNotSupported,

    /// The component mapping was not valid for use with the chosen format.
    FormatInvalidComponentMapping,

    /// The format does not support `force_explicit_reconstruction`.
    FormatForceExplicitReconstructionNotSupported,

    /// The format does not support the `Linear` filter.
    FormatLinearFilterNotSupported,

    /// The component mapping was not valid for use with the chosen YCbCr model.
    YcbcrModelInvalidComponentMapping,

    /// For the chosen `ycbcr_range`, the R, G or B components being read from the `format` do not
    /// have the minimum number of required bits.
    YcbcrRangeFormatNotEnoughBits,
}

impl error::Error for SamplerYcbcrConversionCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            SamplerYcbcrConversionCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SamplerYcbcrConversionCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::OomError(_) => write!(fmt, "not enough memory available"),
            Self::FeatureNotEnabled { feature, reason } => {
                write!(fmt, "the feature {} must be enabled: {}", feature, reason)
            }
            Self::CubicFilterNotSupported => {
                write!(fmt, "the `Cubic` filter was specified")
            }
            Self::FormatMissing => {
                write!(fmt, "no format was specified when one was required")
            }
            Self::FormatNotUnorm => {
                write!(fmt, "the format has a color type other than `UNORM`")
            }
            Self::FormatNotSupported => {
                write!(fmt, "the format does not support sampler YCbCr conversion")
            }
            Self::FormatChromaOffsetNotSupported => {
                write!(fmt, "the format does not support the chosen chroma offsets")
            }
            Self::FormatInvalidComponentMapping => {
                write!(
                    fmt,
                    "the component mapping was not valid for use with the chosen format"
                )
            }
            Self::FormatForceExplicitReconstructionNotSupported => {
                write!(
                    fmt,
                    "the format does not support `force_explicit_reconstruction`"
                )
            }
            Self::FormatLinearFilterNotSupported => {
                write!(fmt, "the format does not support the `Linear` filter")
            }
            Self::YcbcrModelInvalidComponentMapping => {
                write!(
                    fmt,
                    "the component mapping was not valid for use with the chosen YCbCr model"
                )
            }
            Self::YcbcrRangeFormatNotEnoughBits => {
                write!(fmt, "for the chosen `ycbcr_range`, the R, G or B components being read from the `format` do not have the minimum number of required bits")
            }
        }
    }
}

impl From<OomError> for SamplerYcbcrConversionCreationError {
    #[inline]
    fn from(err: OomError) -> SamplerYcbcrConversionCreationError {
        SamplerYcbcrConversionCreationError::OomError(err)
    }
}

impl From<Error> for SamplerYcbcrConversionCreationError {
    #[inline]
    fn from(err: Error) -> SamplerYcbcrConversionCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                SamplerYcbcrConversionCreationError::OomError(OomError::from(err))
            }
            err @ Error::OutOfDeviceMemory => {
                SamplerYcbcrConversionCreationError::OomError(OomError::from(err))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// Parameters to create a new `SamplerYcbcrConversion`.
#[derive(Clone, Debug)]
pub struct SamplerYcbcrConversionCreateInfo {
    /// The image view format that this conversion will read data from. The conversion cannot be
    /// used with image views of any other format.
    ///
    /// The format must support YCbCr conversions, meaning that its `FormatFeatures` must support
    /// at least one of `cosited_chroma_samples` or `midpoint_chroma_samples`.
    ///
    /// If this is set to a format that has chroma subsampling (contains `422` or `420` in the name)
    /// then `component_mapping` is restricted as follows:
    /// - `g` must be identity swizzled.
    /// - `a` must be identity swizzled or `Zero` or `One`.
    /// - `r` and `b` must be identity swizzled or mapped to each other.
    ///
    /// Compatibility notice: currently, this value must be `Some`, but future additions may allow
    /// `None` as a valid value as well.
    ///
    /// The default value is `None`.
    pub format: Option<Format>,

    /// The conversion between the input color model and the output RGB color model.
    ///
    /// If this is not set to `RgbIdentity`, then the `r`, `g` and `b` components of
    /// `component_mapping` must not be `Zero` or `One`, and the component being read must exist in
    /// `format` (must be represented as a nonzero number of bits).
    ///
    /// The default value is [`RgbIdentity`](SamplerYcbcrModelConversion::RgbIdentity).
    pub ycbcr_model: SamplerYcbcrModelConversion,

    /// If `ycbcr_model` is not `RgbIdentity`, specifies the range expansion of the input values
    /// that should be used.
    ///
    /// If this is set to `ItuNarrow`, then the `r`, `g` and `b` components of `component_mapping`
    /// must each map to a component of `format` that is represented with at least 8 bits.
    ///
    /// The default value is [`ItuFull`](SamplerYcbcrRange::ItuFull).
    pub ycbcr_range: SamplerYcbcrRange,

    /// The mapping to apply to the components of the input format, before color model conversion
    /// and range expansion.
    ///
    /// The default value is [`ComponentMapping::identity()`].
    pub component_mapping: ComponentMapping,

    /// For formats with chroma subsampling and a `Linear` filter, specifies the sampled location
    /// for the subsampled components, in the x and y direction.
    ///
    /// The value is ignored if the filter is `Nearest` or the corresponding axis is not chroma
    /// subsampled. If the value is not ignored, the format must support the chosen mode.
    ///
    /// The default value is [`CositedEven`](ChromaLocation::CositedEven) for both axes.
    pub chroma_offset: [ChromaLocation; 2],

    /// For formats with chroma subsampling, specifies the filter used for reconstructing the chroma
    /// components to full resolution.
    ///
    /// The `Cubic` filter is not supported. If `Linear` is used, the format must support it.
    ///
    /// The default value is [`Nearest`](Filter::Nearest).
    pub chroma_filter: Filter,

    /// Forces explicit reconstruction if the implementation does not use it by default. The format
    /// must support it. See
    /// [the spec](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap16.html#textures-chroma-reconstruction)
    /// for more information.
    ///
    /// The default value is `false`.
    pub force_explicit_reconstruction: bool,

    pub _ne: crate::NonExhaustive,
}

impl Default for SamplerYcbcrConversionCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            format: None,
            ycbcr_model: SamplerYcbcrModelConversion::RgbIdentity,
            ycbcr_range: SamplerYcbcrRange::ItuFull,
            component_mapping: ComponentMapping::identity(),
            chroma_offset: [ChromaLocation::CositedEven; 2],
            chroma_filter: Filter::Nearest,
            force_explicit_reconstruction: false,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The conversion between the color model of the source image and the color model of the shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SamplerYcbcrModelConversion {
    /// The input values are already in the shader's model, and are passed through unmodified.
    RgbIdentity = ash::vk::SamplerYcbcrModelConversion::RGB_IDENTITY.as_raw(),

    /// The input values are only range expanded, no other modifications are done.
    YcbcrIdentity = ash::vk::SamplerYcbcrModelConversion::YCBCR_IDENTITY.as_raw(),

    /// The input values are converted according to the
    /// [ITU-R BT.709](https://en.wikipedia.org/wiki/Rec._709) standard.
    Ycbcr709 = ash::vk::SamplerYcbcrModelConversion::YCBCR_709.as_raw(),

    /// The input values are converted according to the
    /// [ITU-R BT.601](https://en.wikipedia.org/wiki/Rec._601) standard.
    Ycbcr601 = ash::vk::SamplerYcbcrModelConversion::YCBCR_601.as_raw(),

    /// The input values are converted according to the
    /// [ITU-R BT.2020](https://en.wikipedia.org/wiki/Rec._2020) standard.
    Ycbcr2020 = ash::vk::SamplerYcbcrModelConversion::YCBCR_2020.as_raw(),
}

impl From<SamplerYcbcrModelConversion> for ash::vk::SamplerYcbcrModelConversion {
    #[inline]
    fn from(val: SamplerYcbcrModelConversion) -> Self {
        Self::from_raw(val as i32)
    }
}

/// How the numeric range of the input data is converted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SamplerYcbcrRange {
    /// The input values cover the full numeric range, and are interpreted according to the ITU
    /// "full range" rules.
    ItuFull = ash::vk::SamplerYcbcrRange::ITU_FULL.as_raw(),

    /// The input values cover only a subset of the numeric range, with the remainder reserved as
    /// headroom/footroom. The values are interpreted according to the ITU "narrow range" rules.
    ItuNarrow = ash::vk::SamplerYcbcrRange::ITU_NARROW.as_raw(),
}

impl From<SamplerYcbcrRange> for ash::vk::SamplerYcbcrRange {
    #[inline]
    fn from(val: SamplerYcbcrRange) -> Self {
        Self::from_raw(val as i32)
    }
}

/// For formats with chroma subsampling, the location where the chroma components are sampled,
/// relative to the luma component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ChromaLocation {
    /// The chroma components are sampled at the even luma coordinate.
    CositedEven = ash::vk::ChromaLocation::COSITED_EVEN.as_raw(),

    /// The chroma components are sampled at the midpoint between the even luma coordinate and
    /// the next higher odd luma coordinate.
    Midpoint = ash::vk::ChromaLocation::MIDPOINT.as_raw(),
}

impl From<ChromaLocation> for ash::vk::ChromaLocation {
    #[inline]
    fn from(val: ChromaLocation) -> Self {
        Self::from_raw(val as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::{SamplerYcbcrConversion, SamplerYcbcrConversionCreationError};

    #[test]
    fn feature_not_enabled() {
        let (device, queue) = gfx_dev_and_queue!();

        let r = SamplerYcbcrConversion::new(device, Default::default());

        match r {
            Err(SamplerYcbcrConversionCreationError::FeatureNotEnabled {
                feature: "sampler_ycbcr_conversion",
                ..
            }) => (),
            _ => panic!(),
        }
    }
}
