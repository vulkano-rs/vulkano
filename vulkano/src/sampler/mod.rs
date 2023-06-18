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

pub mod ycbcr;

use self::ycbcr::SamplerYcbcrConversion;
use crate::{
    device::{Device, DeviceOwned},
    format::FormatFeatures,
    image::{view::ImageViewType, ImageAspects, ImageViewAbstract},
    macros::{impl_id_counter, vulkan_enum},
    pipeline::graphics::depth_stencil::CompareOp,
    shader::ShaderScalarType,
    OomError, RequirementNotMet, Requires, RequiresAllOf, RequiresOneOf, RuntimeError,
    ValidationError, VulkanObject,
};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::MaybeUninit,
    num::NonZeroU64,
    ops::RangeInclusive,
    ptr,
    sync::Arc,
};

/// Describes how to retrieve data from a sampled image within a shader.
///
/// # Examples
///
/// A simple sampler for most usages:
///
/// ```
/// use vulkano::sampler::{Sampler, SamplerCreateInfo};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let _sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear_no_mipmap());
/// ```
///
/// More detailed sampler creation:
///
/// ```
/// use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let _sampler = Sampler::new(device.clone(), SamplerCreateInfo {
///     mag_filter: Filter::Linear,
///     min_filter: Filter::Linear,
///     address_mode: [SamplerAddressMode::Repeat; 3],
///     mip_lod_bias: 1.0,
///     lod: 0.0..=100.0,
///     ..Default::default()
/// })
/// .unwrap();
/// ```
#[derive(Debug)]
pub struct Sampler {
    handle: ash::vk::Sampler,
    device: Arc<Device>,
    id: NonZeroU64,

    address_mode: [SamplerAddressMode; 3],
    anisotropy: Option<f32>,
    border_color: Option<BorderColor>,
    compare: Option<CompareOp>,
    lod: RangeInclusive<f32>,
    mag_filter: Filter,
    min_filter: Filter,
    mip_lod_bias: f32,
    mipmap_mode: SamplerMipmapMode,
    reduction_mode: SamplerReductionMode,
    sampler_ycbcr_conversion: Option<Arc<SamplerYcbcrConversion>>,
    unnormalized_coordinates: bool,
}

impl Sampler {
    /// Creates a new `Sampler`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.anisotropy` is `Some` and contains a value less than 1.0.
    /// - Panics if `create_info.lod` is empty.
    pub fn new(
        device: Arc<Device>,
        create_info: SamplerCreateInfo,
    ) -> Result<Arc<Sampler>, SamplerCreationError> {
        let SamplerCreateInfo {
            mag_filter,
            min_filter,
            mipmap_mode,
            address_mode,
            mip_lod_bias,
            anisotropy,
            compare,
            lod,
            border_color,
            unnormalized_coordinates,
            reduction_mode,
            sampler_ycbcr_conversion,
            _ne: _,
        } = create_info;

        for filter in [mag_filter, min_filter] {
            // VUID-VkSamplerCreateInfo-magFilter-parameter
            // VUID-VkSamplerCreateInfo-minFilter-parameter
            filter.validate_device(&device)?;
        }

        // VUID-VkSamplerCreateInfo-mipmapMode-parameter
        mipmap_mode.validate_device(&device)?;

        for mode in address_mode {
            // VUID-VkSamplerCreateInfo-addressModeU-parameter
            // VUID-VkSamplerCreateInfo-addressModeV-parameter
            // VUID-VkSamplerCreateInfo-addressModeW-parameter
            mode.validate_device(&device)?;

            if mode == SamplerAddressMode::ClampToBorder {
                // VUID-VkSamplerCreateInfo-addressModeU-01078
                border_color.validate_device(&device)?;
            }
        }

        if address_mode.contains(&SamplerAddressMode::MirrorClampToEdge) {
            if !(device.enabled_features().sampler_mirror_clamp_to_edge
                || device.enabled_extensions().khr_sampler_mirror_clamp_to_edge)
            {
                return Err(SamplerCreationError::RequirementNotMet {
                    required_for: "`create_info.address_mode` contains \
                        `SamplerAddressMode::MirrorClampToEdge`",
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::Feature("sampler_mirror_clamp_to_edge")]),
                        RequiresAllOf(&[Requires::DeviceExtension(
                            "khr_sampler_mirror_clamp_to_edge",
                        )]),
                    ]),
                });
            }
        }

        {
            assert!(!lod.is_empty());
            let limit = device.physical_device().properties().max_sampler_lod_bias;
            if mip_lod_bias.abs() > limit {
                return Err(SamplerCreationError::MaxSamplerLodBiasExceeded {
                    requested: mip_lod_bias,
                    maximum: limit,
                });
            }
        }

        // VUID-VkSamplerCreateInfo-samplerMipLodBias-04467
        if device.enabled_extensions().khr_portability_subset
            && !device.enabled_features().sampler_mip_lod_bias
            && mip_lod_bias != 0.0
        {
            return Err(SamplerCreationError::RequirementNotMet {
                required_for: "this device is a portability subset device, and \
                    `create_info.mip_lod_bias` is not zero",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "sampler_mip_lod_bias",
                )])]),
            });
        }

        let (anisotropy_enable, max_anisotropy) = if let Some(max_anisotropy) = anisotropy {
            assert!(max_anisotropy >= 1.0);

            if !device.enabled_features().sampler_anisotropy {
                return Err(SamplerCreationError::RequirementNotMet {
                    required_for: "`create_info.anisotropy` is `Some`",
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "sampler_anisotropy",
                    )])]),
                });
            }

            let limit = device.physical_device().properties().max_sampler_anisotropy;
            if max_anisotropy > limit {
                return Err(SamplerCreationError::MaxSamplerAnisotropyExceeded {
                    requested: max_anisotropy,
                    maximum: limit,
                });
            }

            if [mag_filter, min_filter]
                .into_iter()
                .any(|filter| filter == Filter::Cubic)
            {
                return Err(SamplerCreationError::AnisotropyInvalidFilter {
                    mag_filter,
                    min_filter,
                });
            }

            (ash::vk::TRUE, max_anisotropy)
        } else {
            (ash::vk::FALSE, 1.0)
        };

        let (compare_enable, compare_op) = if let Some(compare_op) = compare {
            // VUID-VkSamplerCreateInfo-compareEnable-01080
            compare_op.validate_device(&device)?;

            if reduction_mode != SamplerReductionMode::WeightedAverage {
                return Err(SamplerCreationError::CompareInvalidReductionMode { reduction_mode });
            }

            (ash::vk::TRUE, compare_op)
        } else {
            (ash::vk::FALSE, CompareOp::Never)
        };

        if unnormalized_coordinates {
            if min_filter != mag_filter {
                return Err(
                    SamplerCreationError::UnnormalizedCoordinatesFiltersNotEqual {
                        mag_filter,
                        min_filter,
                    },
                );
            }

            if mipmap_mode != SamplerMipmapMode::Nearest {
                return Err(
                    SamplerCreationError::UnnormalizedCoordinatesInvalidMipmapMode { mipmap_mode },
                );
            }

            if lod != (0.0..=0.0) {
                return Err(SamplerCreationError::UnnormalizedCoordinatesNonzeroLod { lod });
            }

            if address_mode[0..2].iter().any(|mode| {
                !matches!(
                    mode,
                    SamplerAddressMode::ClampToEdge | SamplerAddressMode::ClampToBorder
                )
            }) {
                return Err(
                    SamplerCreationError::UnnormalizedCoordinatesInvalidAddressMode {
                        address_mode: [address_mode[0], address_mode[1]],
                    },
                );
            }

            if anisotropy.is_some() {
                return Err(SamplerCreationError::UnnormalizedCoordinatesAnisotropyEnabled);
            }

            if compare.is_some() {
                return Err(SamplerCreationError::UnnormalizedCoordinatesCompareEnabled);
            }
        }

        let mut sampler_reduction_mode_create_info = if reduction_mode
            != SamplerReductionMode::WeightedAverage
        {
            if !(device.enabled_features().sampler_filter_minmax
                || device.enabled_extensions().ext_sampler_filter_minmax)
            {
                return Err(SamplerCreationError::RequirementNotMet {
                    required_for: "`create_info.reduction_mode` is not \
                            `SamplerReductionMode::WeightedAverage`",
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::Feature("sampler_filter_minmax")]),
                        RequiresAllOf(&[Requires::DeviceExtension("ext_sampler_filter_minmax")]),
                    ]),
                });
            }

            // VUID-VkSamplerReductionModeCreateInfo-reductionMode-parameter
            reduction_mode.validate_device(&device)?;

            Some(ash::vk::SamplerReductionModeCreateInfo {
                reduction_mode: reduction_mode.into(),
                ..Default::default()
            })
        } else {
            None
        };

        // Don't need to check features because you can't create a conversion object without the
        // feature anyway.
        let mut sampler_ycbcr_conversion_info = if let Some(sampler_ycbcr_conversion) =
            &sampler_ycbcr_conversion
        {
            assert_eq!(&device, sampler_ycbcr_conversion.device());

            // Use unchecked, because all validation has been done by the SamplerYcbcrConversion.
            let potential_format_features = unsafe {
                device
                    .physical_device()
                    .format_properties_unchecked(sampler_ycbcr_conversion.format().unwrap())
                    .potential_format_features()
            };

            // VUID-VkSamplerCreateInfo-minFilter-01645
            if !potential_format_features.intersects(
                FormatFeatures::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER,
            ) && !(mag_filter == sampler_ycbcr_conversion.chroma_filter()
                && min_filter == sampler_ycbcr_conversion.chroma_filter())
            {
                return Err(
                    SamplerCreationError::SamplerYcbcrConversionChromaFilterMismatch {
                        chroma_filter: sampler_ycbcr_conversion.chroma_filter(),
                        mag_filter,
                        min_filter,
                    },
                );
            }

            // VUID-VkSamplerCreateInfo-addressModeU-01646
            if address_mode
                .into_iter()
                .any(|mode| !matches!(mode, SamplerAddressMode::ClampToEdge))
            {
                return Err(
                    SamplerCreationError::SamplerYcbcrConversionInvalidAddressMode { address_mode },
                );
            }

            // VUID-VkSamplerCreateInfo-addressModeU-01646
            if anisotropy.is_some() {
                return Err(SamplerCreationError::SamplerYcbcrConversionAnisotropyEnabled);
            }

            // VUID-VkSamplerCreateInfo-addressModeU-01646
            if unnormalized_coordinates {
                return Err(
                    SamplerCreationError::SamplerYcbcrConversionUnnormalizedCoordinatesEnabled,
                );
            }

            // VUID-VkSamplerCreateInfo-None-01647
            if reduction_mode != SamplerReductionMode::WeightedAverage {
                return Err(
                    SamplerCreationError::SamplerYcbcrConversionInvalidReductionMode {
                        reduction_mode,
                    },
                );
            }

            Some(ash::vk::SamplerYcbcrConversionInfo {
                conversion: sampler_ycbcr_conversion.handle(),
                ..Default::default()
            })
        } else {
            None
        };

        let mut create_info = ash::vk::SamplerCreateInfo {
            flags: ash::vk::SamplerCreateFlags::empty(),
            mag_filter: mag_filter.into(),
            min_filter: min_filter.into(),
            mipmap_mode: mipmap_mode.into(),
            address_mode_u: address_mode[0].into(),
            address_mode_v: address_mode[1].into(),
            address_mode_w: address_mode[2].into(),
            mip_lod_bias,
            anisotropy_enable,
            max_anisotropy,
            compare_enable,
            compare_op: compare_op.into(),
            min_lod: *lod.start(),
            max_lod: *lod.end(),
            border_color: border_color.into(),
            unnormalized_coordinates: unnormalized_coordinates as ash::vk::Bool32,
            ..Default::default()
        };

        if let Some(sampler_reduction_mode_create_info) =
            sampler_reduction_mode_create_info.as_mut()
        {
            sampler_reduction_mode_create_info.p_next = create_info.p_next;
            create_info.p_next = sampler_reduction_mode_create_info as *const _ as *const _;
        }

        if let Some(sampler_ycbcr_conversion_info) = sampler_ycbcr_conversion_info.as_mut() {
            sampler_ycbcr_conversion_info.p_next = create_info.p_next;
            create_info.p_next = sampler_ycbcr_conversion_info as *const _ as *const _;
        }

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_sampler)(
                device.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Sampler {
            handle,
            device,
            id: Self::next_id(),
            address_mode,
            anisotropy,
            border_color: address_mode
                .into_iter()
                .any(|mode| mode == SamplerAddressMode::ClampToBorder)
                .then_some(border_color),
            compare,
            lod,
            mag_filter,
            min_filter,
            mip_lod_bias,
            mipmap_mode,
            reduction_mode,
            sampler_ycbcr_conversion,
            unnormalized_coordinates,
        }))
    }

    /// Creates a new `Sampler` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Sampler,
        create_info: SamplerCreateInfo,
    ) -> Arc<Sampler> {
        let SamplerCreateInfo {
            mag_filter,
            min_filter,
            mipmap_mode,
            address_mode,
            mip_lod_bias,
            anisotropy,
            compare,
            lod,
            border_color,
            unnormalized_coordinates,
            reduction_mode,
            sampler_ycbcr_conversion,
            _ne: _,
        } = create_info;

        Arc::new(Sampler {
            handle,
            device,
            id: Self::next_id(),
            address_mode,
            anisotropy,
            border_color: address_mode
                .into_iter()
                .any(|mode| mode == SamplerAddressMode::ClampToBorder)
                .then_some(border_color),
            compare,
            lod,
            mag_filter,
            min_filter,
            mip_lod_bias,
            mipmap_mode,
            reduction_mode,
            sampler_ycbcr_conversion,
            unnormalized_coordinates,
        })
    }

    /// Checks whether this sampler is compatible with `image_view`.
    pub fn check_can_sample(
        &self,
        image_view: &(impl ImageViewAbstract + ?Sized),
    ) -> Result<(), ValidationError> {
        /*
            NOTE: Most of these checks come from the Instruction/Sampler/Image View Validation
            section, and are not strictly VUIDs.
            https://registry.khronos.org/vulkan/specs/1.2-extensions/html/chap16.html#textures-input-validation
        */

        if self.compare.is_some() {
            // VUID-vkCmdDispatch-None-06479
            if !image_view
                .format_features()
                .intersects(FormatFeatures::SAMPLED_IMAGE_DEPTH_COMPARISON)
            {
                return Err(ValidationError {
                    problem: "the sampler has depth comparison enabled, and \
                        the image view's format features do not include \
                        FormatFeatures::SAMPLED_IMAGE_DEPTH_COMPARISON"
                        .into(),
                    ..Default::default()
                });
            }

            // The SPIR-V instruction is one of the OpImage*Dref* instructions, the image
            // view format is one of the depth/stencil formats, and the image view aspect
            // is not VK_IMAGE_ASPECT_DEPTH_BIT.
            if !image_view
                .subresource_range()
                .aspects
                .intersects(ImageAspects::DEPTH)
            {
                return Err(ValidationError {
                    problem: "the sampler has depth comparison enabled, and \
                        the image view's aspects do not include ImageAspects::DEPTH"
                        .into(),
                    ..Default::default()
                });
            }
        } else {
            if !image_view
                .format_features()
                .intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR)
            {
                // VUID-vkCmdDispatch-magFilter-04553
                if self.mag_filter == Filter::Linear || self.min_filter == Filter::Linear {
                    return Err(ValidationError {
                        problem: "the sampler's mag_filter or min_filter is Filter::Linear, and \
                            the image view's format features do not include \
                            FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR"
                            .into(),
                        ..Default::default()
                    });
                }

                // VUID-vkCmdDispatch-mipmapMode-04770
                if self.mipmap_mode == SamplerMipmapMode::Linear {
                    return Err(ValidationError {
                        problem: "the sampler's mipmap_mode is SamplerMipmapMpde::Linear, and \
                            the image view's format features do not include \
                            FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR"
                            .into(),
                        ..Default::default()
                    });
                }
            }
        }

        if self.mag_filter == Filter::Cubic || self.min_filter == Filter::Cubic {
            // VUID-vkCmdDispatch-None-02692
            if !image_view
                .format_features()
                .intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_CUBIC)
            {
                return Err(ValidationError {
                    problem: "the sampler's mag_filter or min_filter is Filter::Cubic, and \
                        the image view's format features do not include \
                        FormatFeatures::SAMPLED_IMAGE_FILTER_CUBIC"
                        .into(),
                    ..Default::default()
                });
            }

            // VUID-vkCmdDispatch-filterCubic-02694
            if !image_view.filter_cubic() {
                return Err(ValidationError {
                    problem: "the sampler's mag_filter or min_filter is Filter::Cubic, and \
                        the image view does not support this, as returned by \
                        PhysicalDevice::image_format_properties"
                        .into(),
                    ..Default::default()
                });
            }

            // VUID-vkCmdDispatch-filterCubicMinmax-02695
            if matches!(
                self.reduction_mode,
                SamplerReductionMode::Min | SamplerReductionMode::Max
            ) && !image_view.filter_cubic_minmax()
            {
                return Err(ValidationError {
                    problem: "the sampler's mag_filter or min_filter is Filter::Cubic, and \
                        the its reduction_mode is SamplerReductionMode::Min or \
                        SamplerReductionMode::Max, and
                        the image view does not support this, as returned by \
                        PhysicalDevice::image_format_properties"
                        .into(),
                    ..Default::default()
                });
            }
        }

        if let Some(border_color) = self.border_color {
            let aspects = image_view.subresource_range().aspects;
            let view_scalar_type = ShaderScalarType::from(
                if aspects.intersects(
                    ImageAspects::COLOR
                        | ImageAspects::PLANE_0
                        | ImageAspects::PLANE_1
                        | ImageAspects::PLANE_2,
                ) {
                    image_view.format().unwrap().type_color().unwrap()
                } else if aspects.intersects(ImageAspects::DEPTH) {
                    image_view.format().unwrap().type_depth().unwrap()
                } else if aspects.intersects(ImageAspects::STENCIL) {
                    image_view.format().unwrap().type_stencil().unwrap()
                } else {
                    // Per `ImageViewBuilder::aspects` and
                    // VUID-VkDescriptorImageInfo-imageView-01976
                    unreachable!()
                },
            );

            match border_color {
                BorderColor::IntTransparentBlack
                | BorderColor::IntOpaqueBlack
                | BorderColor::IntOpaqueWhite => {
                    // The sampler borderColor is an integer type and the image view
                    // format is not one of the VkFormat integer types or a stencil
                    // component of a depth/stencil format.
                    if !matches!(
                        view_scalar_type,
                        ShaderScalarType::Sint | ShaderScalarType::Uint
                    ) {
                        return Err(ValidationError {
                            problem: "the sampler has an integer border color, and \
                                the image view does not have an integer format"
                                .into(),
                            ..Default::default()
                        });
                    }
                }
                BorderColor::FloatTransparentBlack
                | BorderColor::FloatOpaqueBlack
                | BorderColor::FloatOpaqueWhite => {
                    // The sampler borderColor is a float type and the image view
                    // format is not one of the VkFormat float types or a depth
                    // component of a depth/stencil format.
                    if !matches!(view_scalar_type, ShaderScalarType::Float) {
                        return Err(ValidationError {
                            problem: "the sampler has an floating-point border color, and \
                                the image view does not have a floating-point format"
                                .into(),
                            ..Default::default()
                        });
                    }
                }
            }

            // The sampler borderColor is one of the opaque black colors
            // (VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK or VK_BORDER_COLOR_INT_OPAQUE_BLACK)
            // and the image view VkComponentSwizzle for any of the VkComponentMapping
            // components is not the identity swizzle, and
            // VkPhysicalDeviceBorderColorSwizzleFeaturesEXT::borderColorSwizzleFromImage
            // feature is not enabled, and
            // VkSamplerBorderColorComponentMappingCreateInfoEXT is not specified.
            if matches!(
                border_color,
                BorderColor::FloatOpaqueBlack | BorderColor::IntOpaqueBlack
            ) && !image_view.component_mapping().is_identity()
            {
                return Err(ValidationError {
                    problem: "the sampler has an opaque black border color, and \
                        the image view is not identity swizzled"
                        .into(),
                    ..Default::default()
                });
            }
        }

        // The sampler unnormalizedCoordinates is VK_TRUE and any of the limitations of
        // unnormalized coordinates are violated.
        // https://registry.khronos.org/vulkan/specs/1.2-extensions/html/chap13.html#samplers-unnormalizedCoordinates
        if self.unnormalized_coordinates {
            // The viewType must be either VK_IMAGE_VIEW_TYPE_1D or
            // VK_IMAGE_VIEW_TYPE_2D.
            // VUID-vkCmdDispatch-None-02702
            if !matches!(
                image_view.view_type(),
                ImageViewType::Dim1d | ImageViewType::Dim2d
            ) {
                return Err(ValidationError {
                    problem: "the sampler uses unnormalized coordinates, and \
                        the image view's type is not ImageViewtype::Dim1d or ImageViewType::Dim2d"
                        .into(),
                    ..Default::default()
                });
            }

            // The image view must have a single layer and a single mip level.
            if image_view.subresource_range().mip_levels.end
                - image_view.subresource_range().mip_levels.start
                != 1
            {
                return Err(ValidationError {
                    problem: "the sampler uses unnormalized coordinates, and \
                        the image view has more than one mip level"
                        .into(),
                    ..Default::default()
                });
            }
        }

        Ok(())
    }

    /// Returns the address modes for the u, v and w coordinates.
    #[inline]
    pub fn address_mode(&self) -> [SamplerAddressMode; 3] {
        self.address_mode
    }

    /// Returns the anisotropy mode.
    #[inline]
    pub fn anisotropy(&self) -> Option<f32> {
        self.anisotropy
    }

    /// Returns the border color if one is used by this sampler.
    #[inline]
    pub fn border_color(&self) -> Option<BorderColor> {
        self.border_color
    }

    /// Returns the compare operation if the sampler is a compare-mode sampler.
    #[inline]
    pub fn compare(&self) -> Option<CompareOp> {
        self.compare
    }

    /// Returns the LOD range.
    #[inline]
    pub fn lod(&self) -> RangeInclusive<f32> {
        self.lod.clone()
    }

    /// Returns the magnification filter.
    #[inline]
    pub fn mag_filter(&self) -> Filter {
        self.mag_filter
    }

    /// Returns the minification filter.
    #[inline]
    pub fn min_filter(&self) -> Filter {
        self.min_filter
    }

    /// Returns the mip LOD bias.
    #[inline]
    pub fn mip_lod_bias(&self) -> f32 {
        self.mip_lod_bias
    }

    /// Returns the mipmap mode.
    #[inline]
    pub fn mipmap_mode(&self) -> SamplerMipmapMode {
        self.mipmap_mode
    }

    /// Returns the reduction mode.
    #[inline]
    pub fn reduction_mode(&self) -> SamplerReductionMode {
        self.reduction_mode
    }

    /// Returns a reference to the sampler YCbCr conversion of this sampler, if any.
    #[inline]
    pub fn sampler_ycbcr_conversion(&self) -> Option<&Arc<SamplerYcbcrConversion>> {
        self.sampler_ycbcr_conversion.as_ref()
    }

    /// Returns true if the sampler uses unnormalized coordinates.
    #[inline]
    pub fn unnormalized_coordinates(&self) -> bool {
        self.unnormalized_coordinates
    }
}

impl Drop for Sampler {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_sampler)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for Sampler {
    type Handle = ash::vk::Sampler;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Sampler {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(Sampler);

/// Error that can happen when creating an instance.
#[derive(Clone, Debug, PartialEq)]
pub enum SamplerCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// Too many sampler objects have been created. You must destroy some before creating new ones.
    /// Note the specs guarantee that at least 4000 samplers can exist simultaneously.
    TooManyObjects,

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
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

    /// Sampler YCbCr conversion was enabled together with anisotropy.
    SamplerYcbcrConversionAnisotropyEnabled,

    /// Sampler YCbCr conversion was enabled, and its format does not support
    /// `sampled_image_ycbcr_conversion_separate_reconstruction_filter`, but `mag_filter` or
    /// `min_filter` did not match the conversion's `chroma_filter`.
    SamplerYcbcrConversionChromaFilterMismatch {
        chroma_filter: Filter,
        mag_filter: Filter,
        min_filter: Filter,
    },

    /// Sampler YCbCr conversion was enabled, but the address mode for `u`, `v` or `w` was
    /// something other than `ClampToEdge`.
    SamplerYcbcrConversionInvalidAddressMode {
        address_mode: [SamplerAddressMode; 3],
    },

    /// Sampler YCbCr conversion was enabled, but the reduction mode was something other than
    /// `WeightedAverage`.
    SamplerYcbcrConversionInvalidReductionMode {
        reduction_mode: SamplerReductionMode,
    },

    /// Sampler YCbCr conversion was enabled together with unnormalized coordinates.
    SamplerYcbcrConversionUnnormalizedCoordinatesEnabled,

    /// Unnormalized coordinates were enabled together with anisotropy.
    UnnormalizedCoordinatesAnisotropyEnabled,

    /// Unnormalized coordinates were enabled together with depth comparison.
    UnnormalizedCoordinatesCompareEnabled,

    /// Unnormalized coordinates were enabled, but the min and mag filters were not equal.
    UnnormalizedCoordinatesFiltersNotEqual {
        mag_filter: Filter,
        min_filter: Filter,
    },

    /// Unnormalized coordinates were enabled, but the address mode for `u` or `v` was something
    /// other than `ClampToEdge` or `ClampToBorder`.
    UnnormalizedCoordinatesInvalidAddressMode {
        address_mode: [SamplerAddressMode; 2],
    },

    /// Unnormalized coordinates were enabled, but the mipmap mode was not `Nearest`.
    UnnormalizedCoordinatesInvalidMipmapMode { mipmap_mode: SamplerMipmapMode },

    /// Unnormalized coordinates were enabled, but the LOD range was not zero.
    UnnormalizedCoordinatesNonzeroLod { lod: RangeInclusive<f32> },
}

impl Error for SamplerCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            SamplerCreationError::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for SamplerCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::TooManyObjects => write!(f, "too many simultaneous sampler objects"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::AnisotropyInvalidFilter { .. } => {
                write!(f, "anisotropy was enabled with an invalid filter")
            }
            Self::CompareInvalidReductionMode { .. } => write!(
                f,
                "depth comparison was enabled with an invalid reduction mode",
            ),
            Self::MaxSamplerAnisotropyExceeded { .. } => {
                write!(f, "max_sampler_anisotropy limit exceeded")
            }
            Self::MaxSamplerLodBiasExceeded { .. } => write!(f, "mip lod bias limit exceeded"),
            Self::SamplerYcbcrConversionAnisotropyEnabled => write!(
                f,
                "sampler YCbCr conversion was enabled together with anisotropy",
            ),
            Self::SamplerYcbcrConversionChromaFilterMismatch { .. } => write!(
                f,
                "sampler YCbCr conversion was enabled, and its format does not support
                `sampled_image_ycbcr_conversion_separate_reconstruction_filter`, but `mag_filter`
                or `min_filter` did not match the conversion's `chroma_filter`",
            ),
            Self::SamplerYcbcrConversionInvalidAddressMode { .. } => write!(
                f,
                "sampler YCbCr conversion was enabled, but the address mode for u, v or w was
                something other than `ClampToEdge`",
            ),
            Self::SamplerYcbcrConversionInvalidReductionMode { .. } => write!(
                f,
                "sampler YCbCr conversion was enabled, but the reduction mode was something other \
                than `WeightedAverage`",
            ),
            Self::SamplerYcbcrConversionUnnormalizedCoordinatesEnabled => write!(
                f,
                "sampler YCbCr conversion was enabled together with unnormalized coordinates",
            ),
            Self::UnnormalizedCoordinatesAnisotropyEnabled => write!(
                f,
                "unnormalized coordinates were enabled together with anisotropy",
            ),
            Self::UnnormalizedCoordinatesCompareEnabled => write!(
                f,
                "unnormalized coordinates were enabled together with depth comparison",
            ),
            Self::UnnormalizedCoordinatesFiltersNotEqual { .. } => write!(
                f,
                "unnormalized coordinates were enabled, but the min and mag filters were not equal",
            ),
            Self::UnnormalizedCoordinatesInvalidAddressMode { .. } => write!(
                f,
                "unnormalized coordinates were enabled, but the address mode for u or v was \
                something other than `ClampToEdge` or `ClampToBorder`",
            ),
            Self::UnnormalizedCoordinatesInvalidMipmapMode { .. } => write!(
                f,
                "unnormalized coordinates were enabled, but the mipmap mode was not `Nearest`",
            ),
            Self::UnnormalizedCoordinatesNonzeroLod { .. } => write!(
                f,
                "unnormalized coordinates were enabled, but the LOD range was not zero",
            ),
        }
    }
}

impl From<OomError> for SamplerCreationError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<RuntimeError> for SamplerCreationError {
    fn from(err: RuntimeError) -> Self {
        match err {
            err @ RuntimeError::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ RuntimeError::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            RuntimeError::TooManyObjects => Self::TooManyObjects,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<RequirementNotMet> for SamplerCreationError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

/// Parameters to create a new `Sampler`.
#[derive(Clone, Debug)]
pub struct SamplerCreateInfo {
    /// How the sampled value of a single mipmap should be calculated,
    /// when magnification is applied (LOD <= 0.0).
    ///
    /// The default value is [`Nearest`](Filter::Nearest).
    pub mag_filter: Filter,

    /// How the sampled value of a single mipmap should be calculated,
    /// when minification is applied (LOD > 0.0).
    ///
    /// The default value is [`Nearest`](Filter::Nearest).
    pub min_filter: Filter,

    /// How the final sampled value should be calculated from the samples of individual
    /// mipmaps.
    ///
    /// The default value is [`Nearest`](SamplerMipmapMode::Nearest).
    pub mipmap_mode: SamplerMipmapMode,

    /// How out-of-range texture coordinates should be treated, for the `u`, `v` and `w` texture
    /// coordinate indices respectively.
    ///
    /// The default value is [`ClampToEdge`](SamplerAddressMode::ClampToEdge).
    pub address_mode: [SamplerAddressMode; 3],

    /// The bias value to be added to the base LOD before clamping.
    ///
    /// The absolute value of the provided value must not exceed the
    /// [`max_sampler_lod_bias`](crate::device::Properties::max_sampler_lod_bias) limit of the
    /// device.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if `mip_lod_bias` is not `0.0`, the
    /// [`sampler_mip_lod_bias`](crate::device::Features::sampler_mip_lod_bias)
    /// feature must be enabled on the device.
    ///
    /// The default value is `0.0`.
    pub mip_lod_bias: f32,

    /// Whether anisotropic texel filtering is enabled (`Some`), and the maximum anisotropy value
    /// to use if it is enabled.
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
    pub anisotropy: Option<f32>,

    /// Whether depth comparison is enabled (`Some`), and the comparison operator to use if it is
    /// enabled.
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
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if the sampler is going to be used as a mutable sampler (written to descriptor sets
    /// rather than being an immutable part of a descriptor set layout), the
    /// [`mutable_comparison_samplers`](crate::device::Features::mutable_comparison_samplers)
    /// feature must be enabled on the device.
    ///
    /// The default value is `None`.
    pub compare: Option<CompareOp>,

    /// The range that LOD values must be clamped to.
    ///
    /// If the end of the range is set to [`LOD_CLAMP_NONE`], it is unbounded.
    ///
    /// The default value is `0.0..=0.0`.
    pub lod: RangeInclusive<f32>,

    /// The border color to use if `address_mode` is set to
    /// [`ClampToBorder`](SamplerAddressMode::ClampToBorder).
    ///
    /// The default value is [`FloatTransparentBlack`](BorderColor::FloatTransparentBlack).
    pub border_color: BorderColor,

    /// Whether unnormalized texture coordinates are enabled.
    ///
    /// When a sampler is set to use unnormalized coordinates as input, the texture coordinates are
    /// not scaled by the size of the image, and therefore range up to the size of the image rather
    /// than 1.0. Enabling this comes with several restrictions:
    /// - `min_filter` and `mag_filter` must be equal.
    /// - `mipmap_mode` must be [`Nearest`](SamplerMipmapMode::Nearest).
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
    pub unnormalized_coordinates: bool,

    /// How the value sampled from a mipmap should be calculated from the selected
    /// pixels, for the `Linear` and `Cubic` filters.
    ///
    /// The default value is [`WeightedAverage`](SamplerReductionMode::WeightedAverage).
    pub reduction_mode: SamplerReductionMode,

    /// Adds a sampler YCbCr conversion to the sampler.
    ///
    /// If set to `Some`, several restrictions apply:
    /// - If the `format` of `conversion` does not support
    ///   `sampled_image_ycbcr_conversion_separate_reconstruction_filter`, then `mag_filter` and
    ///   `min_filter` must be equal to the `chroma_filter` of `conversion`.
    /// - `address_mode` for u, v and w must be [`ClampToEdge`](`SamplerAddressMode::ClampToEdge`).
    /// - Anisotropy and unnormalized coordinates must be disabled.
    /// - The `reduction_mode` must be [`WeightedAverage`](SamplerReductionMode::WeightedAverage).
    ///
    /// In addition, the sampler must only be used as an immutable sampler within a descriptor set
    /// layout, and only in a combined image sampler descriptor.
    ///
    /// The default value is `None`.
    pub sampler_ycbcr_conversion: Option<Arc<SamplerYcbcrConversion>>,

    pub _ne: crate::NonExhaustive,
}

impl Default for SamplerCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mipmap_mode: SamplerMipmapMode::Nearest,
            address_mode: [SamplerAddressMode::ClampToEdge; 3],
            mip_lod_bias: 0.0,
            anisotropy: None,
            compare: None,
            lod: 0.0..=0.0,
            border_color: BorderColor::FloatTransparentBlack,
            unnormalized_coordinates: false,
            reduction_mode: SamplerReductionMode::WeightedAverage,
            sampler_ycbcr_conversion: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SamplerCreateInfo {
    /// Shortcut for creating a sampler with linear sampling, linear mipmaps, and with the repeat
    /// mode for borders.
    #[inline]
    pub fn simple_repeat_linear() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mipmap_mode: SamplerMipmapMode::Linear,
            address_mode: [SamplerAddressMode::Repeat; 3],
            lod: 0.0..=LOD_CLAMP_NONE,
            ..Default::default()
        }
    }

    /// Shortcut for creating a sampler with linear sampling, that only uses the main level of
    /// images, and with the repeat mode for borders.
    #[inline]
    pub fn simple_repeat_linear_no_mipmap() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::Repeat; 3],
            lod: 0.0..=1.0,
            ..Default::default()
        }
    }
}

/// A special value to indicate that the maximum LOD should not be clamped.
pub const LOD_CLAMP_NONE: f32 = ash::vk::LOD_CLAMP_NONE;

/// A mapping between components of a source format and components read by a shader.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ComponentMapping {
    /// First component.
    pub r: ComponentSwizzle,
    /// Second component.
    pub g: ComponentSwizzle,
    /// Third component.
    pub b: ComponentSwizzle,
    /// Fourth component.
    pub a: ComponentSwizzle,
}

impl ComponentMapping {
    /// Creates a `ComponentMapping` with all components identity swizzled.
    #[inline]
    pub fn identity() -> Self {
        Self::default()
    }

    /// Returns `true` if all components are identity swizzled,
    /// meaning that all the members are `Identity` or the name of that member.
    ///
    /// Certain operations require views that are identity swizzled, and will return an error
    /// otherwise. For example, attaching a view to a framebuffer is only possible if the view is
    /// identity swizzled.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.r_is_identity() && self.g_is_identity() && self.b_is_identity() && self.a_is_identity()
    }

    /// Returns `true` if the red component mapping is identity swizzled.
    #[inline]
    pub fn r_is_identity(&self) -> bool {
        matches!(self.r, ComponentSwizzle::Identity | ComponentSwizzle::Red)
    }

    /// Returns `true` if the green component mapping is identity swizzled.
    #[inline]
    pub fn g_is_identity(&self) -> bool {
        matches!(self.g, ComponentSwizzle::Identity | ComponentSwizzle::Green)
    }

    /// Returns `true` if the blue component mapping is identity swizzled.
    #[inline]
    pub fn b_is_identity(&self) -> bool {
        matches!(self.b, ComponentSwizzle::Identity | ComponentSwizzle::Blue)
    }

    /// Returns `true` if the alpha component mapping is identity swizzled.
    #[inline]
    pub fn a_is_identity(&self) -> bool {
        matches!(self.a, ComponentSwizzle::Identity | ComponentSwizzle::Alpha)
    }

    /// Returns the component indices that each component reads from. The index is `None` if the
    /// component has a fixed value and is not read from anywhere (`Zero` or `One`).
    #[inline]
    pub fn component_map(&self) -> [Option<usize>; 4] {
        [
            match self.r {
                ComponentSwizzle::Identity => Some(0),
                ComponentSwizzle::Zero => None,
                ComponentSwizzle::One => None,
                ComponentSwizzle::Red => Some(0),
                ComponentSwizzle::Green => Some(1),
                ComponentSwizzle::Blue => Some(2),
                ComponentSwizzle::Alpha => Some(3),
            },
            match self.g {
                ComponentSwizzle::Identity => Some(1),
                ComponentSwizzle::Zero => None,
                ComponentSwizzle::One => None,
                ComponentSwizzle::Red => Some(0),
                ComponentSwizzle::Green => Some(1),
                ComponentSwizzle::Blue => Some(2),
                ComponentSwizzle::Alpha => Some(3),
            },
            match self.b {
                ComponentSwizzle::Identity => Some(2),
                ComponentSwizzle::Zero => None,
                ComponentSwizzle::One => None,
                ComponentSwizzle::Red => Some(0),
                ComponentSwizzle::Green => Some(1),
                ComponentSwizzle::Blue => Some(2),
                ComponentSwizzle::Alpha => Some(3),
            },
            match self.a {
                ComponentSwizzle::Identity => Some(3),
                ComponentSwizzle::Zero => None,
                ComponentSwizzle::One => None,
                ComponentSwizzle::Red => Some(0),
                ComponentSwizzle::Green => Some(1),
                ComponentSwizzle::Blue => Some(2),
                ComponentSwizzle::Alpha => Some(3),
            },
        ]
    }
}

impl From<ComponentMapping> for ash::vk::ComponentMapping {
    #[inline]
    fn from(value: ComponentMapping) -> Self {
        Self {
            r: value.r.into(),
            g: value.g.into(),
            b: value.b.into(),
            a: value.a.into(),
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes the value that an individual component must return when being accessed.
    ComponentSwizzle = ComponentSwizzle(i32);

    /// Returns the value that this component should normally have.
    ///
    /// This is the `Default` value.
    Identity = IDENTITY,

    /// Always return zero.
    Zero = ZERO,

    /// Always return one.
    One = ONE,

    /// Returns the value of the first component.
    Red = R,

    /// Returns the value of the second component.
    Green = G,

    /// Returns the value of the third component.
    Blue = B,

    /// Returns the value of the fourth component.
    Alpha = A,
}

impl Default for ComponentSwizzle {
    #[inline]
    fn default() -> ComponentSwizzle {
        ComponentSwizzle::Identity
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes how the color of each pixel should be determined.
    Filter = Filter(i32);

    /// The pixel whose center is nearest to the requested coordinates is taken from the source
    /// and its value is returned as-is.
    Nearest = NEAREST,

    /// The 8/4/2 pixels (depending on view dimensionality) whose center surround the requested
    /// coordinates are taken, then their values are combined according to the chosen
    /// `reduction_mode`.
    Linear = LINEAR,

    /// The 64/16/4 pixels (depending on the view dimensionality) whose center surround the
    /// requested coordinates are taken, then their values are combined according to the chosen
    /// `reduction_mode`.
    ///
    /// The [`ext_filter_cubic`](crate::device::DeviceExtensions::ext_filter_cubic) extension must
    /// be enabled on the device, and anisotropy must be disabled. Sampled image views must have
    /// a type of [`Dim2d`](crate::image::view::ImageViewType::Dim2d).
    Cubic = CUBIC_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_filter_cubic)]),
        RequiresAllOf([DeviceExtension(img_filter_cubic)]),
    ]),
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes which mipmap from the source to use.
    SamplerMipmapMode = SamplerMipmapMode(i32);

    /// Use the mipmap whose dimensions are the nearest to the dimensions of the destination.
    Nearest = NEAREST,

    /// Take the mipmap whose dimensions are no greater than that of the destination together
    /// with the next higher level mipmap, calculate the value for both, and interpolate them.
    Linear = LINEAR,
}

vulkan_enum! {
    #[non_exhaustive]

    /// How the sampler should behave when it needs to access a pixel that is out of range of the
    /// texture.
    SamplerAddressMode = SamplerAddressMode(i32);

    /// Repeat the texture. In other words, the pixel at coordinate `x + 1.0` is the same as the
    /// one at coordinate `x`.
    Repeat = REPEAT,

    /// Repeat the texture but mirror it at every repetition. In other words, the pixel at
    /// coordinate `x + 1.0` is the same as the one at coordinate `1.0 - x`.
    MirroredRepeat = MIRRORED_REPEAT,

    /// The coordinates are clamped to the valid range. Coordinates below 0.0 have the same value
    /// as coordinate 0.0. Coordinates over 1.0 have the same value as coordinate 1.0.
    ClampToEdge = CLAMP_TO_EDGE,

    /// Any pixel out of range is colored using the colour selected with the `border_color` on the
    /// `SamplerBuilder`.
    ///
    /// When this mode is chosen, the numeric type of the image view's format must match the border
    /// color. When using a floating-point border color, the sampler can only be used with
    /// floating-point or depth image views. When using an integer border color, the sampler can
    /// only be used with integer or stencil image views. In addition to this, you can't use an
    /// opaque black border color with an image view that uses component swizzling.
    ClampToBorder = CLAMP_TO_BORDER,

    /// Similar to `MirroredRepeat`, except that coordinates are clamped to the range
    /// `[-1.0, 1.0]`.
    ///
    /// The [`sampler_mirror_clamp_to_edge`](crate::device::Features::sampler_mirror_clamp_to_edge)
    /// feature or the
    /// [`khr_sampler_mirror_clamp_to_edge`](crate::device::DeviceExtensions::khr_sampler_mirror_clamp_to_edge)
    /// extension must be enabled on the device.
    MirrorClampToEdge = MIRROR_CLAMP_TO_EDGE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(khr_sampler_mirror_clamp_to_edge)]),
    ]),
}

vulkan_enum! {
    #[non_exhaustive]

    /// The color to use for the border of an image.
    ///
    /// Only relevant if you use `ClampToBorder`.
    ///
    /// Using a border color restricts the sampler to either floating-point images or integer images.
    BorderColor = BorderColor(i32);

    /// The value `(0.0, 0.0, 0.0, 0.0)`. Can only be used with floating-point images.
    FloatTransparentBlack = FLOAT_TRANSPARENT_BLACK,

    /// The value `(0, 0, 0, 0)`. Can only be used with integer images.
    IntTransparentBlack = INT_TRANSPARENT_BLACK,

    /// The value `(0.0, 0.0, 0.0, 1.0)`. Can only be used with floating-point identity-swizzled
    /// images.
    FloatOpaqueBlack = FLOAT_OPAQUE_BLACK,

    /// The value `(0, 0, 0, 1)`. Can only be used with integer identity-swizzled images.
    IntOpaqueBlack = INT_OPAQUE_BLACK,

    /// The value `(1.0, 1.0, 1.0, 1.0)`. Can only be used with floating-point images.
    FloatOpaqueWhite = FLOAT_OPAQUE_WHITE,

    /// The value `(1, 1, 1, 1)`. Can only be used with integer images.
    IntOpaqueWhite = INT_OPAQUE_WHITE,

    /* TODO: enable
    // TODO: document
    FloatCustom = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_custom_border_color)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    IntCustom = INT_CUSTOM_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_custom_border_color)]),
    ]),*/
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes how the value sampled from a mipmap should be calculated from the selected
    /// pixels, for the `Linear` and `Cubic` filters.
    SamplerReductionMode = SamplerReductionMode(i32);

    /// Calculates a weighted average of the selected pixels. For `Linear` filtering the pixels
    /// are evenly weighted, for `Cubic` filtering they use Catmull-Rom weights.
    WeightedAverage = WEIGHTED_AVERAGE,

    /// Calculates the minimum of the selected pixels.
    ///
    /// The [`sampler_filter_minmax`](crate::device::Features::sampler_filter_minmax)
    /// feature or the
    /// [`ext_sampler_filter_minmax`](crate::device::DeviceExtensions::ext_sampler_filter_minmax)
    /// extension must be enabled on the device.
    Min = MIN,

    /// Calculates the maximum of the selected pixels.
    ///
    /// The [`sampler_filter_minmax`](crate::device::Features::sampler_filter_minmax)
    /// feature or the
    /// [`ext_sampler_filter_minmax`](crate::device::DeviceExtensions::ext_sampler_filter_minmax)
    /// extension must be enabled on the device.
    Max = MAX,
}

#[cfg(test)]
mod tests {
    use crate::{
        pipeline::graphics::depth_stencil::CompareOp,
        sampler::{
            Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerCreationError,
            SamplerReductionMode,
        },
        Requires, RequiresAllOf, RequiresOneOf,
    };

    #[test]
    fn create_regular() {
        let (device, _queue) = gfx_dev_and_queue!();

        let s = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 1.0,
                lod: 0.0..=2.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.compare().is_none());
        assert!(!s.unnormalized_coordinates());
    }

    #[test]
    fn create_compare() {
        let (device, _queue) = gfx_dev_and_queue!();

        let s = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 1.0,
                compare: Some(CompareOp::Less),
                lod: 0.0..=2.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.compare().is_some());
        assert!(!s.unnormalized_coordinates());
    }

    #[test]
    fn create_unnormalized() {
        let (device, _queue) = gfx_dev_and_queue!();

        let s = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                unnormalized_coordinates: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.compare().is_none());
        assert!(s.unnormalized_coordinates());
    }

    #[test]
    fn simple_repeat_linear() {
        let (device, _queue) = gfx_dev_and_queue!();
        let _ = Sampler::new(device, SamplerCreateInfo::simple_repeat_linear());
    }

    #[test]
    fn simple_repeat_linear_no_mipmap() {
        let (device, _queue) = gfx_dev_and_queue!();
        let _ = Sampler::new(device, SamplerCreateInfo::simple_repeat_linear_no_mipmap());
    }

    #[test]
    fn min_lod_inferior() {
        let (device, _queue) = gfx_dev_and_queue!();

        assert_should_panic!({
            let _ = Sampler::new(
                device,
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::Repeat; 3],
                    mip_lod_bias: 1.0,
                    lod: 5.0..=2.0,
                    ..Default::default()
                },
            );
        });
    }

    #[test]
    fn max_anisotropy() {
        let (device, _queue) = gfx_dev_and_queue!();

        assert_should_panic!({
            let _ = Sampler::new(
                device,
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::Repeat; 3],
                    mip_lod_bias: 1.0,
                    anisotropy: Some(0.5),
                    lod: 0.0..=2.0,
                    ..Default::default()
                },
            );
        });
    }

    #[test]
    fn anisotropy_feature() {
        let (device, _queue) = gfx_dev_and_queue!();

        let r = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 1.0,
                anisotropy: Some(2.0),
                lod: 0.0..=2.0,
                ..Default::default()
            },
        );

        match r {
            Err(SamplerCreationError::RequirementNotMet {
                requires_one_of:
                    RequiresOneOf([RequiresAllOf([Requires::Feature("sampler_anisotropy")])]),
                ..
            }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn anisotropy_limit() {
        let (device, _queue) = gfx_dev_and_queue!(sampler_anisotropy);

        let r = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 1.0,
                anisotropy: Some(100000000.0),
                lod: 0.0..=2.0,
                ..Default::default()
            },
        );

        match r {
            Err(SamplerCreationError::MaxSamplerAnisotropyExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn mip_lod_bias_limit() {
        let (device, _queue) = gfx_dev_and_queue!();

        let r = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 100000000.0,
                lod: 0.0..=2.0,
                ..Default::default()
            },
        );

        match r {
            Err(SamplerCreationError::MaxSamplerLodBiasExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn sampler_mirror_clamp_to_edge_extension() {
        let (device, _queue) = gfx_dev_and_queue!();

        let r = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::MirrorClampToEdge; 3],
                mip_lod_bias: 1.0,
                lod: 0.0..=2.0,
                ..Default::default()
            },
        );

        match r {
            Err(SamplerCreationError::RequirementNotMet {
                requires_one_of:
                    RequiresOneOf(
                        [RequiresAllOf([Requires::Feature("sampler_mirror_clamp_to_edge")]), RequiresAllOf(
                            [Requires::DeviceExtension("khr_sampler_mirror_clamp_to_edge")],
                        )],
                    ),
                ..
            }) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn sampler_filter_minmax_extension() {
        let (device, _queue) = gfx_dev_and_queue!();

        let r = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                reduction_mode: SamplerReductionMode::Min,
                ..Default::default()
            },
        );

        match r {
            Err(SamplerCreationError::RequirementNotMet {
                requires_one_of:
                    RequiresOneOf(
                        [RequiresAllOf([Requires::Feature("sampler_filter_minmax")]), RequiresAllOf([Requires::DeviceExtension("ext_sampler_filter_minmax")])],
                    ),
                ..
            }) => {}
            _ => panic!(),
        }
    }
}
