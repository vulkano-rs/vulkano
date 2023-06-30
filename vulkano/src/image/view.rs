// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Image views.
//!
//! This module contains types related to image views. An image view wraps around
//! an image and describes how the GPU should interpret the data. It is needed when an image is
//! to be used in a shader descriptor or as a framebuffer attachment.

use super::{Image, ImageDimensions, ImageFormatInfo, ImageSubresourceRange, ImageUsage};
use crate::{
    device::{Device, DeviceOwned},
    format::{ChromaSampling, Format, FormatFeatures},
    image::{
        sampler::{ycbcr::SamplerYcbcrConversion, ComponentMapping},
        ImageAspects, ImageCreateFlags, ImageTiling, ImageType, SampleCount,
    },
    macros::{impl_id_counter, vulkan_enum},
    OomError, RequirementNotMet, Requires, RequiresAllOf, RequiresOneOf, RuntimeError, Version,
    VulkanObject,
};
use std::{
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    hash::Hash,
    mem::MaybeUninit,
    num::NonZeroU64,
    ptr,
    sync::Arc,
};

/// A wrapper around an image that makes it available to shaders or framebuffers.
#[derive(Debug)]
pub struct ImageView {
    handle: ash::vk::ImageView,
    image: Arc<Image>,
    id: NonZeroU64,

    view_type: ImageViewType,
    format: Option<Format>,
    component_mapping: ComponentMapping,
    subresource_range: ImageSubresourceRange,
    usage: ImageUsage,
    sampler_ycbcr_conversion: Option<Arc<SamplerYcbcrConversion>>,

    format_features: FormatFeatures,
    filter_cubic: bool,
    filter_cubic_minmax: bool,
}

impl ImageView {
    /// Creates a new `ImageView`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.array_layers` is empty.
    /// - Panics if `create_info.mip_levels` is empty.
    /// - Panics if `create_info.aspects` contains any aspects other than `color`, `depth`,
    ///   `stencil`, `plane0`, `plane1` or `plane2`.
    /// - Panics if `create_info.aspects` contains more more than one aspect, unless `depth` and
    ///   `stencil` are the only aspects selected.
    pub fn new(
        image: Arc<Image>,
        create_info: ImageViewCreateInfo,
    ) -> Result<Arc<ImageView>, ImageViewCreationError> {
        let format_features = Self::validate_new(&image, &create_info)?;

        unsafe {
            Ok(Self::new_unchecked_with_format_features(
                image,
                create_info,
                format_features,
            )?)
        }
    }

    fn validate_new(
        image: &Image,
        create_info: &ImageViewCreateInfo,
    ) -> Result<FormatFeatures, ImageViewCreationError> {
        let &ImageViewCreateInfo {
            view_type,
            format,
            component_mapping,
            ref subresource_range,
            mut usage,
            ref sampler_ycbcr_conversion,
            _ne: _,
        } = create_info;

        let device = image.device();
        let format = format.unwrap();

        let level_count = subresource_range.mip_levels.end - subresource_range.mip_levels.start;
        let layer_count = subresource_range.array_layers.end - subresource_range.array_layers.start;

        // VUID-VkImageSubresourceRange-aspectMask-requiredbitmask
        assert!(!subresource_range.aspects.is_empty());

        // VUID-VkImageSubresourceRange-levelCount-01720
        assert!(level_count != 0);

        // VUID-VkImageSubresourceRange-layerCount-01721
        assert!(layer_count != 0);

        let default_usage = Self::get_default_usage(subresource_range.aspects, image);

        let has_non_default_usage = if usage.is_empty() {
            usage = default_usage;
            false
        } else {
            usage == default_usage
        };

        // VUID-VkImageViewCreateInfo-viewType-parameter
        view_type.validate_device(device)?;

        // VUID-VkImageViewCreateInfo-format-parameter
        format.validate_device(device)?;

        // VUID-VkComponentMapping-r-parameter
        component_mapping.r.validate_device(device)?;

        // VUID-VkComponentMapping-g-parameter
        component_mapping.g.validate_device(device)?;

        // VUID-VkComponentMapping-b-parameter
        component_mapping.b.validate_device(device)?;

        // VUID-VkComponentMapping-a-parameter
        component_mapping.a.validate_device(device)?;

        // VUID-VkImageSubresourceRange-aspectMask-parameter
        subresource_range.aspects.validate_device(device)?;

        assert!(!subresource_range.aspects.intersects(
            ImageAspects::METADATA
                | ImageAspects::MEMORY_PLANE_0
                | ImageAspects::MEMORY_PLANE_1
                | ImageAspects::MEMORY_PLANE_2
        ));
        assert!({
            subresource_range.aspects.count() == 1
                || subresource_range
                    .aspects
                    .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                    && !subresource_range.aspects.intersects(
                        ImageAspects::COLOR
                            | ImageAspects::PLANE_0
                            | ImageAspects::PLANE_1
                            | ImageAspects::PLANE_2,
                    )
        });

        // Get format features
        let format_features = unsafe { Self::get_format_features(format, image) };

        // No VUID apparently, but this seems like something we want to check?
        if !image
            .format()
            .unwrap()
            .aspects()
            .contains(subresource_range.aspects)
        {
            return Err(ImageViewCreationError::ImageAspectsNotCompatible {
                aspects: subresource_range.aspects,
                image_aspects: image.format().unwrap().aspects(),
            });
        }

        // VUID-VkImageViewCreateInfo-None-02273
        if format_features == FormatFeatures::default() {
            return Err(ImageViewCreationError::FormatNotSupported);
        }

        // Check for compatibility with the image
        let image_type = image.dimensions().image_type();

        // VUID-VkImageViewCreateInfo-subResourceRange-01021
        if !view_type.is_compatible_with(image_type) {
            return Err(ImageViewCreationError::ImageTypeNotCompatible);
        }

        // VUID-VkImageViewCreateInfo-image-01003
        if (view_type == ImageViewType::Cube || view_type == ImageViewType::CubeArray)
            && !image.flags().intersects(ImageCreateFlags::CUBE_COMPATIBLE)
        {
            return Err(ImageViewCreationError::ImageNotCubeCompatible);
        }

        // VUID-VkImageViewCreateInfo-viewType-01004
        if view_type == ImageViewType::CubeArray && !device.enabled_features().image_cube_array {
            return Err(ImageViewCreationError::RequirementNotMet {
                required_for: "`create_info.viewtype` is `ImageViewType::CubeArray`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "image_cube_array",
                )])]),
            });
        }

        // VUID-VkImageViewCreateInfo-subresourceRange-01718
        if subresource_range.mip_levels.end > image.mip_levels() {
            return Err(ImageViewCreationError::MipLevelsOutOfRange {
                range_end: subresource_range.mip_levels.end,
                max: image.mip_levels(),
            });
        }

        if image_type == ImageType::Dim3d
            && (view_type == ImageViewType::Dim2d || view_type == ImageViewType::Dim2dArray)
        {
            // VUID-VkImageViewCreateInfo-image-01005
            if !image
                .flags()
                .intersects(ImageCreateFlags::ARRAY_2D_COMPATIBLE)
            {
                return Err(ImageViewCreationError::ImageNotArray2dCompatible);
            }

            // VUID-VkImageViewCreateInfo-image-04970
            if level_count != 1 {
                return Err(ImageViewCreationError::Array2dCompatibleMultipleMipLevels);
            }

            // VUID-VkImageViewCreateInfo-image-02724
            // VUID-VkImageViewCreateInfo-subresourceRange-02725
            // We're using the depth dimension as array layers, but because of mip scaling, the
            // depth, and therefore number of layers available, shrinks as the mip level gets
            // higher.
            let max = image
                .dimensions()
                .mip_level_dimensions(subresource_range.mip_levels.start)
                .unwrap()
                .depth();
            if subresource_range.array_layers.end > max {
                return Err(ImageViewCreationError::ArrayLayersOutOfRange {
                    range_end: subresource_range.array_layers.end,
                    max,
                });
            }
        } else {
            // VUID-VkImageViewCreateInfo-image-01482
            // VUID-VkImageViewCreateInfo-subresourceRange-01483
            if subresource_range.array_layers.end > image.dimensions().array_layers() {
                return Err(ImageViewCreationError::ArrayLayersOutOfRange {
                    range_end: subresource_range.array_layers.end,
                    max: image.dimensions().array_layers(),
                });
            }
        }

        // VUID-VkImageViewCreateInfo-image-04972
        if image.samples() != SampleCount::Sample1
            && !(view_type == ImageViewType::Dim2d || view_type == ImageViewType::Dim2dArray)
        {
            return Err(ImageViewCreationError::MultisamplingNot2d);
        }

        /* Check usage requirements */

        if has_non_default_usage {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_maintenance2)
            {
                return Err(ImageViewCreationError::RequirementNotMet {
                    required_for: "`create_info.usage` is not the default value",
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance2")]),
                    ]),
                });
            }

            // VUID-VkImageViewUsageCreateInfo-usage-parameter
            usage.validate_device(device)?;

            // VUID-VkImageViewUsageCreateInfo-usage-requiredbitmask
            assert!(!usage.is_empty());

            // VUID-VkImageViewCreateInfo-pNext-02662
            // VUID-VkImageViewCreateInfo-pNext-02663
            // VUID-VkImageViewCreateInfo-pNext-02664
            if !default_usage.contains(usage) {
                return Err(ImageViewCreationError::UsageNotSupportedByImage {
                    usage,
                    supported_usage: default_usage,
                });
            }
        }

        // VUID-VkImageViewCreateInfo-image-04441
        if !image.usage().intersects(
            ImageUsage::SAMPLED
                | ImageUsage::STORAGE
                | ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                | ImageUsage::INPUT_ATTACHMENT
                | ImageUsage::TRANSIENT_ATTACHMENT,
        ) {
            return Err(ImageViewCreationError::ImageMissingUsage);
        }

        // VUID-VkImageViewCreateInfo-usage-02274
        if usage.intersects(ImageUsage::SAMPLED)
            && !format_features.intersects(FormatFeatures::SAMPLED_IMAGE)
        {
            return Err(ImageViewCreationError::FormatUsageNotSupported { usage: "sampled" });
        }

        // VUID-VkImageViewCreateInfo-usage-02275
        if usage.intersects(ImageUsage::STORAGE)
            && !format_features.intersects(FormatFeatures::STORAGE_IMAGE)
        {
            return Err(ImageViewCreationError::FormatUsageNotSupported { usage: "storage" });
        }

        // VUID-VkImageViewCreateInfo-usage-02276
        if usage.intersects(ImageUsage::COLOR_ATTACHMENT)
            && !format_features.intersects(FormatFeatures::COLOR_ATTACHMENT)
        {
            return Err(ImageViewCreationError::FormatUsageNotSupported {
                usage: "color_attachment",
            });
        }

        // VUID-VkImageViewCreateInfo-usage-02277
        if usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            && !format_features.intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
        {
            return Err(ImageViewCreationError::FormatUsageNotSupported {
                usage: "depth_stencil_attachment",
            });
        }

        // VUID-VkImageViewCreateInfo-usage-02652
        if usage.intersects(ImageUsage::INPUT_ATTACHMENT)
            && !format_features.intersects(
                FormatFeatures::COLOR_ATTACHMENT | FormatFeatures::DEPTH_STENCIL_ATTACHMENT,
            )
        {
            return Err(ImageViewCreationError::FormatUsageNotSupported {
                usage: "input_attachment",
            });
        }

        /* Check flags requirements */

        if Some(format) != image.format() {
            // VUID-VkImageViewCreateInfo-image-01762
            if !image.flags().intersects(ImageCreateFlags::MUTABLE_FORMAT)
                || !image.format().unwrap().planes().is_empty()
                    && subresource_range.aspects.intersects(ImageAspects::COLOR)
            {
                return Err(ImageViewCreationError::FormatNotCompatible);
            }

            // VUID-VkImageViewCreateInfo-imageViewFormatReinterpretation-04466
            // TODO: it is unclear what the number of bits is for compressed formats.
            // See https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2361
            if device.enabled_extensions().khr_portability_subset
                && !device.enabled_features().image_view_format_reinterpretation
                && format.components() != image.format().unwrap().components()
            {
                return Err(ImageViewCreationError::RequirementNotMet {
                    required_for: "this device is a portability subset device, and the format of \
                        the image view does not have the same components and number of bits per \
                        component as the parent image",
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "image_view_format_reinterpretation",
                    )])]),
                });
            }

            if image
                .flags()
                .intersects(ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE)
            {
                // VUID-VkImageViewCreateInfo-image-01583
                if !(format.compatibility() == image.format().unwrap().compatibility()
                    || format.block_size() == image.format().unwrap().block_size())
                {
                    return Err(ImageViewCreationError::FormatNotCompatible);
                }

                if format.compression().is_none() {
                    // VUID-VkImageViewCreateInfo-image-01584
                    if layer_count != 1 {
                        return Err(
                            ImageViewCreationError::BlockTexelViewCompatibleMultipleArrayLayers,
                        );
                    }

                    // VUID-VkImageViewCreateInfo-image-01584
                    if level_count != 1 {
                        return Err(
                            ImageViewCreationError::BlockTexelViewCompatibleMultipleMipLevels,
                        );
                    }
                }
            } else {
                if image.format().unwrap().planes().is_empty() {
                    // VUID-VkImageViewCreateInfo-image-01761
                    if format.compatibility() != image.format().unwrap().compatibility() {
                        return Err(ImageViewCreationError::FormatNotCompatible);
                    }
                } else {
                    let plane = if subresource_range.aspects.intersects(ImageAspects::PLANE_0) {
                        0
                    } else if subresource_range.aspects.intersects(ImageAspects::PLANE_1) {
                        1
                    } else if subresource_range.aspects.intersects(ImageAspects::PLANE_2) {
                        2
                    } else {
                        unreachable!()
                    };
                    let plane_format = image.format().unwrap().planes()[plane];

                    // VUID-VkImageViewCreateInfo-image-01586
                    if format.compatibility() != plane_format.compatibility() {
                        return Err(ImageViewCreationError::FormatNotCompatible);
                    }
                }
            }
        }

        // VUID-VkImageViewCreateInfo-imageViewType-04973
        if (view_type == ImageViewType::Dim1d
            || view_type == ImageViewType::Dim2d
            || view_type == ImageViewType::Dim3d)
            && layer_count != 1
        {
            return Err(ImageViewCreationError::TypeNonArrayedMultipleArrayLayers);
        }
        // VUID-VkImageViewCreateInfo-viewType-02960
        else if view_type == ImageViewType::Cube && layer_count != 6 {
            return Err(ImageViewCreationError::TypeCubeNot6ArrayLayers);
        }
        // VUID-VkImageViewCreateInfo-viewType-02961
        else if view_type == ImageViewType::CubeArray && layer_count % 6 != 0 {
            return Err(ImageViewCreationError::TypeCubeArrayNotMultipleOf6ArrayLayers);
        }

        // VUID-VkImageViewCreateInfo-imageViewFormatSwizzle-04465
        if device.enabled_extensions().khr_portability_subset
            && !device.enabled_features().image_view_format_swizzle
            && !component_mapping.is_identity()
        {
            return Err(ImageViewCreationError::RequirementNotMet {
                required_for: "this device is a portability subset device, and \
                    `create_info.component_mapping` is not the identity mapping",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "image_view_format_swizzle",
                )])]),
            });
        }

        // VUID-VkImageViewCreateInfo-format-04714
        // VUID-VkImageViewCreateInfo-format-04715
        match format.ycbcr_chroma_sampling() {
            Some(ChromaSampling::Mode422) => {
                if image.dimensions().width() % 2 != 0 {
                    return Err(
                        ImageViewCreationError::FormatChromaSubsamplingInvalidImageDimensions,
                    );
                }
            }
            Some(ChromaSampling::Mode420) => {
                if image.dimensions().width() % 2 != 0 || image.dimensions().height() % 2 != 0 {
                    return Err(
                        ImageViewCreationError::FormatChromaSubsamplingInvalidImageDimensions,
                    );
                }
            }
            _ => (),
        }

        // Don't need to check features because you can't create a conversion object without the
        // feature anyway.
        if let Some(conversion) = &sampler_ycbcr_conversion {
            assert_eq!(device, conversion.device());

            // VUID-VkImageViewCreateInfo-pNext-01970
            if !component_mapping.is_identity() {
                return Err(
                    ImageViewCreationError::SamplerYcbcrConversionComponentMappingNotIdentity {
                        component_mapping,
                    },
                );
            }
        } else {
            // VUID-VkImageViewCreateInfo-format-06415
            if format.ycbcr_chroma_sampling().is_some() {
                return Err(
                    ImageViewCreationError::FormatRequiresSamplerYcbcrConversion { format },
                );
            }
        }

        Ok(format_features)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        image: Arc<Image>,
        create_info: ImageViewCreateInfo,
    ) -> Result<Arc<Self>, RuntimeError> {
        let format_features = Self::get_format_features(create_info.format.unwrap(), &image);

        Self::new_unchecked_with_format_features(image, create_info, format_features)
    }

    unsafe fn new_unchecked_with_format_features(
        image: Arc<Image>,
        create_info: ImageViewCreateInfo,
        format_features: FormatFeatures,
    ) -> Result<Arc<Self>, RuntimeError> {
        let &ImageViewCreateInfo {
            view_type,
            format,
            component_mapping,
            ref subresource_range,
            mut usage,
            ref sampler_ycbcr_conversion,
            _ne: _,
        } = &create_info;

        let device = image.device();

        let default_usage = Self::get_default_usage(subresource_range.aspects, &image);

        let has_non_default_usage = if usage.is_empty() {
            usage = default_usage;
            false
        } else {
            usage == default_usage
        };

        let mut info_vk = ash::vk::ImageViewCreateInfo {
            flags: ash::vk::ImageViewCreateFlags::empty(),
            image: image.handle(),
            view_type: view_type.into(),
            format: format.unwrap().into(),
            components: component_mapping.into(),
            subresource_range: subresource_range.clone().into(),
            ..Default::default()
        };
        let mut image_view_usage_info_vk = None;
        let mut sampler_ycbcr_conversion_info_vk = None;

        if has_non_default_usage {
            let next = image_view_usage_info_vk.insert(ash::vk::ImageViewUsageCreateInfo {
                usage: usage.into(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *const _ as *const _;
        }

        if let Some(conversion) = sampler_ycbcr_conversion {
            let next =
                sampler_ycbcr_conversion_info_vk.insert(ash::vk::SamplerYcbcrConversionInfo {
                    conversion: conversion.handle(),
                    ..Default::default()
                });

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *const _ as *const _;
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_image_view)(
                device.handle(),
                &info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        Self::from_handle_with_format_features(image, handle, create_info, format_features)
    }

    /// Creates a default `ImageView`. Equivalent to
    /// `ImageView::new(image, ImageViewCreateInfo::from_image(image))`.
    pub fn new_default(image: Arc<Image>) -> Result<Arc<ImageView>, ImageViewCreationError> {
        let create_info = ImageViewCreateInfo::from_image(&image);

        Self::new(image, create_info)
    }

    /// Creates a new `ImageView` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `image`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        image: Arc<Image>,
        handle: ash::vk::ImageView,
        create_info: ImageViewCreateInfo,
    ) -> Result<Arc<Self>, RuntimeError> {
        let format_features = Self::get_format_features(create_info.format.unwrap(), &image);

        Self::from_handle_with_format_features(image, handle, create_info, format_features)
    }

    unsafe fn from_handle_with_format_features(
        image: Arc<Image>,
        handle: ash::vk::ImageView,
        create_info: ImageViewCreateInfo,
        format_features: FormatFeatures,
    ) -> Result<Arc<Self>, RuntimeError> {
        let ImageViewCreateInfo {
            view_type,
            format,
            component_mapping,
            subresource_range,
            mut usage,
            sampler_ycbcr_conversion,
            _ne: _,
        } = create_info;

        let device = image.device();

        if usage.is_empty() {
            usage = Self::get_default_usage(subresource_range.aspects, &image);
        }

        let mut filter_cubic = false;
        let mut filter_cubic_minmax = false;

        if device
            .physical_device()
            .supported_extensions()
            .ext_filter_cubic
        {
            // Use unchecked, because all validation has been done above or is validated by the
            // image.
            let properties =
                device
                    .physical_device()
                    .image_format_properties_unchecked(ImageFormatInfo {
                        flags: image.flags(),
                        format: image.format(),
                        image_type: image.dimensions().image_type(),
                        tiling: image.tiling(),
                        usage: image.usage(),
                        image_view_type: Some(view_type),
                        ..Default::default()
                    })?;

            if let Some(properties) = properties {
                filter_cubic = properties.filter_cubic;
                filter_cubic_minmax = properties.filter_cubic_minmax;
            }
        }

        Ok(Arc::new(ImageView {
            handle,
            image,
            id: Self::next_id(),
            view_type,
            format,
            component_mapping,
            subresource_range,
            usage,
            sampler_ycbcr_conversion,
            format_features,
            filter_cubic,
            filter_cubic_minmax,
        }))
    }

    // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkImageViewCreateInfo.html#_description
    fn get_default_usage(aspects: ImageAspects, image: &Image) -> ImageUsage {
        let has_stencil_aspect = aspects.intersects(ImageAspects::STENCIL);
        let has_non_stencil_aspect = !(aspects - ImageAspects::STENCIL).is_empty();

        if has_stencil_aspect && has_non_stencil_aspect {
            image.usage() & image.stencil_usage()
        } else if has_stencil_aspect {
            image.stencil_usage()
        } else if has_non_stencil_aspect {
            image.usage()
        } else {
            unreachable!()
        }
    }

    // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap12.html#resources-image-view-format-features
    unsafe fn get_format_features(format: Format, image: &Image) -> FormatFeatures {
        let device = image.device();

        let mut format_features = if Some(format) != image.format() {
            // Use unchecked, because all validation should have been done before calling.
            let format_properties = device.physical_device().format_properties_unchecked(format);

            match image.tiling() {
                ImageTiling::Optimal => format_properties.optimal_tiling_features,
                ImageTiling::Linear => format_properties.linear_tiling_features,
                ImageTiling::DrmFormatModifier => format_properties.linear_tiling_features,
            }
        } else {
            image.format_features()
        };

        if !device.enabled_extensions().khr_format_feature_flags2 {
            if format.type_color().is_none()
                && format_features.intersects(FormatFeatures::SAMPLED_IMAGE)
            {
                format_features |= FormatFeatures::SAMPLED_IMAGE_DEPTH_COMPARISON;
            }

            if format.shader_storage_image_without_format() {
                if device
                    .enabled_features()
                    .shader_storage_image_read_without_format
                {
                    format_features |= FormatFeatures::STORAGE_READ_WITHOUT_FORMAT;
                }

                if device
                    .enabled_features()
                    .shader_storage_image_write_without_format
                {
                    format_features |= FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT;
                }
            }
        }

        format_features
    }

    /// Returns the wrapped image that this image view was created from.
    #[inline]
    pub fn image(&self) -> &Arc<Image> {
        &self.image
    }

    /// Returns the [`ImageViewType`] of this image view.
    #[inline]
    pub fn view_type(&self) -> ImageViewType {
        self.view_type
    }

    /// Returns the format of this view. This can be different from the parent's format.
    #[inline]
    pub fn format(&self) -> Option<Format> {
        self.format
    }

    /// Returns the component mapping of this view.
    #[inline]
    pub fn component_mapping(&self) -> ComponentMapping {
        self.component_mapping
    }

    /// Returns the subresource range of the wrapped image that this view exposes.
    #[inline]
    pub fn subresource_range(&self) -> &ImageSubresourceRange {
        &self.subresource_range
    }

    /// Returns the usage of the image view.
    #[inline]
    pub fn usage(&self) -> ImageUsage {
        self.usage
    }

    /// Returns the sampler YCbCr conversion that this image view was created with, if any.
    #[inline]
    pub fn sampler_ycbcr_conversion(&self) -> Option<&Arc<SamplerYcbcrConversion>> {
        self.sampler_ycbcr_conversion.as_ref()
    }

    /// Returns the dimensions of this view.
    #[inline]
    pub fn dimensions(&self) -> ImageDimensions {
        let array_layers =
            self.subresource_range.array_layers.end - self.subresource_range.array_layers.start;

        match self.image().dimensions() {
            ImageDimensions::Dim1d { width, .. } => ImageDimensions::Dim1d {
                width,
                array_layers,
            },
            ImageDimensions::Dim2d { width, height, .. } => ImageDimensions::Dim2d {
                width,
                height,
                array_layers,
            },
            ImageDimensions::Dim3d {
                width,
                height,
                depth,
            } => ImageDimensions::Dim3d {
                width,
                height,
                depth,
            },
        }
    }

    /// Returns the features supported by the image view's format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.format_features
    }

    /// Returns whether the image view supports sampling with a
    /// [`Cubic`](crate::sampler::Filter::Cubic) `mag_filter` or `min_filter`.
    #[inline]
    pub fn filter_cubic(&self) -> bool {
        self.filter_cubic
    }

    /// Returns whether the image view supports sampling with a
    /// [`Cubic`](crate::sampler::Filter::Cubic) `mag_filter` or `min_filter`, and with a
    /// [`Min`](crate::sampler::SamplerReductionMode::Min) or
    /// [`Max`](crate::sampler::SamplerReductionMode::Max) `reduction_mode`.
    #[inline]
    pub fn filter_cubic_minmax(&self) -> bool {
        self.filter_cubic_minmax
    }
}

impl Drop for ImageView {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let device = self.device();
            let fns = device.fns();
            (fns.v1_0.destroy_image_view)(device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for ImageView {
    type Handle = ash::vk::ImageView;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for ImageView {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.image.device()
    }
}

impl_id_counter!(ImageView);

/// Parameters to create a new `ImageView`.
#[derive(Debug)]
pub struct ImageViewCreateInfo {
    /// The image view type.
    ///
    /// The view type must be compatible with the dimensions of the image and the selected array
    /// layers.
    ///
    /// The default value is [`ImageViewType::Dim2d`].
    pub view_type: ImageViewType,

    /// The format of the image view.
    ///
    /// If this is set to a format that is different from the image, the image must be created with
    /// the `mutable_format` flag.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if `format` does not have the same number of components and bits per component as
    /// the parent image's format, the
    /// [`image_view_format_reinterpretation`](crate::device::Features::image_view_format_reinterpretation)
    /// feature must be enabled on the device.
    ///
    /// The default value is `None`, which must be overridden.
    pub format: Option<Format>,

    /// How to map components of each pixel.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if `component_mapping` is not the identity mapping, the
    /// [`image_view_format_swizzle`](crate::device::Features::image_view_format_swizzle)
    /// feature must be enabled on the device.
    ///
    /// The default value is [`ComponentMapping::identity()`].
    pub component_mapping: ComponentMapping,

    /// The subresource range of the image that the view should cover.
    ///
    /// The default value is empty, which must be overridden.
    pub subresource_range: ImageSubresourceRange,

    /// How the image view is going to be used.
    ///
    /// If `usage` is empty, then a default value is used based on the parent image's usages.
    /// Depending on the image aspects selected in `subresource_range`,
    /// the default `usage` will be equal to the parent image's `usage`, its `stencil_usage`,
    /// or the intersection of the two.
    ///
    /// If you set `usage` to a different value from the default, then the device API version must
    /// be at least 1.1, or the [`khr_maintenance2`](crate::device::DeviceExtensions::khr_maintenance2)
    /// extension must be enabled on the device. The specified `usage` must be a subset of the
    /// default value; usages that are not set for the parent image are not allowed.
    ///
    /// The default value is [`ImageUsage::empty()`].
    pub usage: ImageUsage,

    /// The sampler YCbCr conversion to be used with the image view.
    ///
    /// If set to `Some`, several restrictions apply:
    /// - The `component_mapping` must be the identity swizzle for all components.
    /// - If the image view is to be used in a shader, it must be in a combined image sampler
    ///   descriptor, a separate sampled image descriptor is not allowed.
    /// - The corresponding sampler must have the same sampler YCbCr object or an identically
    ///   created one, and must be used as an immutable sampler within a descriptor set layout.
    ///
    /// The default value is `None`.
    pub sampler_ycbcr_conversion: Option<Arc<SamplerYcbcrConversion>>,

    pub _ne: crate::NonExhaustive,
}

impl Default for ImageViewCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            view_type: ImageViewType::Dim2d,
            format: None,
            component_mapping: ComponentMapping::identity(),
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::empty(),
                array_layers: 0..0,
                mip_levels: 0..0,
            },
            usage: ImageUsage::empty(),
            sampler_ycbcr_conversion: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl ImageViewCreateInfo {
    /// Returns an `ImageViewCreateInfo` with the `view_type` determined from the image type and
    /// array layers, and `subresource_range` determined from the image format and covering the
    /// whole image.
    #[inline]
    pub fn from_image(image: &Image) -> Self {
        Self {
            view_type: match image.dimensions() {
                ImageDimensions::Dim1d {
                    array_layers: 1, ..
                } => ImageViewType::Dim1d,
                ImageDimensions::Dim1d { .. } => ImageViewType::Dim1dArray,
                ImageDimensions::Dim2d {
                    array_layers: 1, ..
                } => ImageViewType::Dim2d,
                ImageDimensions::Dim2d { .. } => ImageViewType::Dim2dArray,
                ImageDimensions::Dim3d { .. } => ImageViewType::Dim3d,
            },
            format: image.format(),
            subresource_range: image.subresource_range(),
            ..Default::default()
        }
    }
}

/// Error that can happen when creating an image view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageViewCreationError {
    /// Allocating memory failed.
    OomError(OomError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// A 2D image view was requested from a 3D image, but a range of multiple mip levels was
    /// specified.
    Array2dCompatibleMultipleMipLevels,

    /// The specified range of array layers was not a subset of those in the image.
    ArrayLayersOutOfRange { range_end: u32, max: u32 },

    /// The image has the `block_texel_view_compatible` flag, but a range of multiple array layers
    /// was specified.
    BlockTexelViewCompatibleMultipleArrayLayers,

    /// The image has the `block_texel_view_compatible` flag, but a range of multiple mip levels
    /// was specified.
    BlockTexelViewCompatibleMultipleMipLevels,

    /// The requested format has chroma subsampling, but the width and/or height of the image was
    /// not a multiple of 2.
    FormatChromaSubsamplingInvalidImageDimensions,

    /// The requested format was not compatible with the image.
    FormatNotCompatible,

    /// The given format was not supported by the device.
    FormatNotSupported,

    /// The format requires a sampler YCbCr conversion, but none was provided.
    FormatRequiresSamplerYcbcrConversion { format: Format },

    /// A requested usage flag was not supported by the given format.
    FormatUsageNotSupported { usage: &'static str },

    /// An aspect was selected that was not present in the image.
    ImageAspectsNotCompatible {
        aspects: ImageAspects,
        image_aspects: ImageAspects,
    },

    /// The image was not created with
    /// [one of the required usages](https://registry.khronos.org/vulkan/specs/1.2-extensions/html/vkspec.html#valid-imageview-imageusage)
    /// for image views.
    ImageMissingUsage,

    /// A 2D image view was requested from a 3D image, but the image was not created with the
    /// `array_2d_compatible` flag.
    ImageNotArray2dCompatible,

    /// A cube image view type was requested, but the image was not created with the
    /// `cube_compatible` flag.
    ImageNotCubeCompatible,

    /// The given image view type was not compatible with the type of the image.
    ImageTypeNotCompatible,

    /// The requested [`ImageViewType`] was not compatible with the image, or with the specified
    /// ranges of array layers and mipmap levels.
    IncompatibleType,

    /// The specified range of mip levels was not a subset of those in the image.
    MipLevelsOutOfRange { range_end: u32, max: u32 },

    /// The image has multisampling enabled, but the image view type was not `Dim2d` or
    /// `Dim2dArray`.
    MultisamplingNot2d,

    /// Sampler YCbCr conversion was enabled, but `component_mapping` was not the identity mapping.
    SamplerYcbcrConversionComponentMappingNotIdentity { component_mapping: ComponentMapping },

    /// The `CubeArray` image view type was specified, but the range of array layers did not have a
    /// size that is a multiple 6.
    TypeCubeArrayNotMultipleOf6ArrayLayers,

    /// The `Cube` image view type was specified, but the range of array layers did not have a size
    /// of 6.
    TypeCubeNot6ArrayLayers,

    /// A non-arrayed image view type was specified, but a range of multiple array layers was
    /// specified.
    TypeNonArrayedMultipleArrayLayers,

    /// The provided `usage` is not supported by the parent image.
    UsageNotSupportedByImage {
        usage: ImageUsage,
        supported_usage: ImageUsage,
    },
}

impl Error for ImageViewCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ImageViewCreationError::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for ImageViewCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "allocating memory failed",),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::Array2dCompatibleMultipleMipLevels => write!(
                f,
                "a 2D image view was requested from a 3D image, but a range of multiple mip levels \
                was specified",
            ),
            Self::ArrayLayersOutOfRange { .. } => write!(
                f,
                "the specified range of array layers was not a subset of those in the image",
            ),
            Self::BlockTexelViewCompatibleMultipleArrayLayers => write!(
                f,
                "the image has the `block_texel_view_compatible` flag, but a range of multiple \
                array layers was specified",
            ),
            Self::BlockTexelViewCompatibleMultipleMipLevels => write!(
                f,
                "the image has the `block_texel_view_compatible` flag, but a range of multiple mip \
                levels was specified",
            ),
            Self::FormatChromaSubsamplingInvalidImageDimensions => write!(
                f,
                "the requested format has chroma subsampling, but the width and/or height of the \
                image was not a multiple of 2",
            ),
            Self::FormatNotCompatible => {
                write!(f, "the requested format was not compatible with the image")
            }
            Self::FormatNotSupported => {
                write!(f, "the given format was not supported by the device")
            }
            Self::FormatRequiresSamplerYcbcrConversion { .. } => write!(
                f,
                "the format requires a sampler YCbCr conversion, but none was provided",
            ),
            Self::FormatUsageNotSupported { .. } => write!(
                f,
                "a requested usage flag was not supported by the given format",
            ),
            Self::ImageAspectsNotCompatible { .. } => write!(
                f,
                "an aspect was selected that was not present in the image",
            ),
            Self::ImageMissingUsage => write!(
                f,
                "the image was not created with one of the required usages for image views",
            ),
            Self::ImageNotArray2dCompatible => write!(
                f,
                "a 2D image view was requested from a 3D image, but the image was not created with \
                the `array_2d_compatible` flag",
            ),
            Self::ImageNotCubeCompatible => write!(
                f,
                "a cube image view type was requested, but the image was not created with the \
                `cube_compatible` flag",
            ),
            Self::ImageTypeNotCompatible => write!(
                f,
                "the given image view type was not compatible with the type of the image",
            ),
            Self::IncompatibleType => write!(
                f,
                "image view type is not compatible with image, array layers or mipmap levels",
            ),
            Self::MipLevelsOutOfRange { .. } => write!(
                f,
                "the specified range of mip levels was not a subset of those in the image",
            ),
            Self::MultisamplingNot2d => write!(
                f,
                "the image has multisampling enabled, but the image view type was not `Dim2d` or \
                `Dim2dArray`",
            ),
            Self::SamplerYcbcrConversionComponentMappingNotIdentity { .. } => write!(
                f,
                "sampler YCbCr conversion was enabled, but `component_mapping` was not the \
                identity mapping",
            ),
            Self::TypeCubeArrayNotMultipleOf6ArrayLayers => write!(
                f,
                "the `CubeArray` image view type was specified, but the range of array layers did \
                not have a size that is a multiple 6",
            ),
            Self::TypeCubeNot6ArrayLayers => write!(
                f,
                "the `Cube` image view type was specified, but the range of array layers did not \
                have a size of 6",
            ),
            Self::TypeNonArrayedMultipleArrayLayers => write!(
                f,
                "a non-arrayed image view type was specified, but a range of multiple array layers \
                was specified",
            ),
            Self::UsageNotSupportedByImage {
                usage: _,
                supported_usage: _,
            } => write!(
                f,
                "the provided `usage` is not supported by the parent image",
            ),
        }
    }
}

impl From<OomError> for ImageViewCreationError {
    fn from(err: OomError) -> ImageViewCreationError {
        ImageViewCreationError::OomError(err)
    }
}

impl From<RuntimeError> for ImageViewCreationError {
    fn from(err: RuntimeError) -> ImageViewCreationError {
        match err {
            err @ RuntimeError::OutOfHostMemory => OomError::from(err).into(),
            err @ RuntimeError::OutOfDeviceMemory => OomError::from(err).into(),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<RequirementNotMet> for ImageViewCreationError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The geometry type of an image view.
    ImageViewType impl {
        /// Returns whether the type is arrayed.
        #[inline]
        pub fn is_arrayed(self) -> bool {
            match self {
                Self::Dim1d | Self::Dim2d | Self::Dim3d | Self::Cube => false,
                Self::Dim1dArray | Self::Dim2dArray | Self::CubeArray => true,
            }
        }

        /// Returns whether `self` is compatible with the given `image_type`.
        #[inline]
        pub fn is_compatible_with(self, image_type: ImageType) -> bool {
            matches!(
                (self, image_type,),
                (
                    ImageViewType::Dim1d | ImageViewType::Dim1dArray,
                    ImageType::Dim1d
                ) | (
                    ImageViewType::Dim2d | ImageViewType::Dim2dArray,
                    ImageType::Dim2d | ImageType::Dim3d
                ) | (
                    ImageViewType::Cube | ImageViewType::CubeArray,
                    ImageType::Dim2d
                ) | (ImageViewType::Dim3d, ImageType::Dim3d)
            )
        }
    }
    = ImageViewType(i32);

    // TODO: document
    Dim1d = TYPE_1D,

    // TODO: document
    Dim2d = TYPE_2D,

    // TODO: document
    Dim3d = TYPE_3D,

    // TODO: document
    Cube = CUBE,

    // TODO: document
    Dim1dArray = TYPE_1D_ARRAY,

    // TODO: document
    Dim2dArray = TYPE_2D_ARRAY,

    // TODO: document
    CubeArray = CUBE_ARRAY,
}
