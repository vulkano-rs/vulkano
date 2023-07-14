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
//! This module contains types related to image views. An image view wraps around an image and
//! describes how the GPU should interpret the data. It is needed when an image is to be used in a
//! shader descriptor or as a framebuffer attachment.
//!
//! See also [the parent module-level documentation] for more information about images.
//!
//! [the parent module-level documentation]: super

use super::{mip_level_extent, Image, ImageFormatInfo, ImageSubresourceRange, ImageUsage};
use crate::{
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    format::{ChromaSampling, Format, FormatFeatures},
    image::{
        sampler::{ycbcr::SamplerYcbcrConversion, ComponentMapping},
        ImageAspects, ImageCreateFlags, ImageTiling, ImageType, SampleCount,
    },
    macros::{impl_id_counter, vulkan_enum},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use smallvec::{smallvec, SmallVec};
use std::{fmt::Debug, hash::Hash, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

/// A wrapper around an image that makes it available to shaders or framebuffers.
///
/// See also [the parent module-level documentation] for more information about images.
///
/// [the parent module-level documentation]: super
#[derive(Debug)]
pub struct ImageView {
    handle: ash::vk::ImageView,
    image: DeviceOwnedDebugWrapper<Arc<Image>>,
    id: NonZeroU64,

    view_type: ImageViewType,
    format: Format,
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
    #[inline]
    pub fn new(
        image: Arc<Image>,
        create_info: ImageViewCreateInfo,
    ) -> Result<Arc<ImageView>, Validated<VulkanError>> {
        Self::validate_new(&image, &create_info)?;

        unsafe { Ok(Self::new_unchecked(image, create_info)?) }
    }

    fn validate_new(
        image: &Image,
        create_info: &ImageViewCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        let device = image.device();

        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        let &ImageViewCreateInfo {
            view_type,
            format,
            component_mapping: _,
            ref subresource_range,
            mut usage,
            sampler_ycbcr_conversion: _,
            _ne: _,
        } = create_info;

        let format_features = unsafe { get_format_features(format, image) };

        if format_features.is_empty() {
            return Err(Box::new(ValidationError {
                context: "create_info.format".into(),
                problem: "the format features are empty".into(),
                vuids: &["VUID-VkImageViewCreateInfo-None-02273"],
                ..Default::default()
            }));
        }

        let image_type = image.image_type();

        if !view_type.is_compatible_with(image_type) {
            return Err(Box::new(ValidationError {
                problem: "`create_info.view_type` is not compatible with \
                    `image.image_type()`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-subResourceRange-01021"],
                ..Default::default()
            }));
        }

        if subresource_range.mip_levels.end > image.mip_levels() {
            return Err(Box::new(ValidationError {
                problem: "`create_info.subresource_range.mip_levels.end` is greater than \
                    `image.mip_levels()`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-subresourceRange-01718"],
                ..Default::default()
            }));
        }

        if matches!(view_type, ImageViewType::Cube | ImageViewType::CubeArray)
            && !image.flags().intersects(ImageCreateFlags::CUBE_COMPATIBLE)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.view_type` is `ImageViewType::Cube` or \
                    `ImageViewType::CubeArray`, but \
                    `image.flags()` does not contain `ImageCreateFlags::CUBE_COMPATIBLE`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-image-01003"],
                ..Default::default()
            }));
        }

        if matches!(view_type, ImageViewType::Dim2d | ImageViewType::Dim2dArray)
            && image_type == ImageType::Dim3d
        {
            match view_type {
                ImageViewType::Dim2d => {
                    if !image
                        .flags()
                        .intersects(ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`create_info.view_type` is `ImageViewType::Dim2d`, and \
                                `image.image_type()` is `ImageType::Dim3d`, but \
                                `image.flags()` does not contain \
                                `ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE` or \
                                `ImageCreateFlags::DIM2D_VIEW_COMPATIBLE`"
                                .into(),
                            vuids: &["VUID-VkImageViewCreateInfo-image-06728"],
                            ..Default::default()
                        }));
                    }
                }
                ImageViewType::Dim2dArray => {
                    if !image
                        .flags()
                        .intersects(ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`create_info.view_type` is `ImageViewType::Dim2dArray`, and \
                                `image.image_type()` is `ImageType::Dim3d`, but \
                                `image.flags()` does not contain \
                                `ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE`"
                                .into(),
                            vuids: &["VUID-VkImageViewCreateInfo-image-06723"],
                            ..Default::default()
                        }));
                    }
                }
                _ => unreachable!(),
            }

            if subresource_range.mip_levels.len() != 1 {
                return Err(Box::new(ValidationError {
                    problem: "`create_info.view_type` is `ImageViewType::Dim2d` or \
                        `ImageViewType::Dim2dArray`, and \
                        `image.image_type()` is `ImageType::Dim3d`, but \
                        the length of `create_info.subresource_range.mip_levels` is not 1"
                        .into(),
                    vuids: &["VUID-VkImageViewCreateInfo-image-04970"],
                    ..Default::default()
                }));
            }

            // We're using the depth dimension as array layers, but because of mip scaling, the
            // depth, and therefore number of layers available, shrinks as the mip level gets
            // higher.
            let mip_level_extent =
                mip_level_extent(image.extent(), subresource_range.mip_levels.start).unwrap();

            if subresource_range.array_layers.end > mip_level_extent[2] {
                return Err(Box::new(ValidationError {
                    problem: "`create_info.view_type` is `ImageViewType::Dim2d` or \
                        `ImageViewType::Dim2dArray`, and \
                        `image.image_type()` is `ImageType::Dim3d`, but \
                        `create_info.subresource_range.array_layers.end` is greater than \
                        the depth of the mip level \
                        `create_info.subresource_range.mip_levels` of `image`"
                        .into(),
                    vuids: &[
                        "VUID-VkImageViewCreateInfo-image-02724",
                        "VUID-VkImageViewCreateInfo-subresourceRange-02725",
                    ],
                    ..Default::default()
                }));
            }
        } else {
            if subresource_range.array_layers.end > image.array_layers() {
                return Err(Box::new(ValidationError {
                    problem: "`create_info.subresource_range.array_layers.end` is greater than \
                        `image.array_layers()`"
                        .into(),
                    vuids: &[
                        "VUID-VkImageViewCreateInfo-image-06724",
                        "VUID-VkImageViewCreateInfo-subresourceRange-06725",
                    ],
                    ..Default::default()
                }));
            }
        }

        if image.samples() != SampleCount::Sample1
            && !matches!(view_type, ImageViewType::Dim2d | ImageViewType::Dim2dArray)
        {
            return Err(Box::new(ValidationError {
                problem: "`image.samples()` is not `SampleCount::Sample1`, but \
                    `create_info.view_type` is not `ImageViewType::Dim2d` or \
                    `ImageViewType::Dim2dArray`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-image-04972"],
                ..Default::default()
            }));
        }

        /* Check usage requirements */

        let implicit_default_usage = get_implicit_default_usage(subresource_range.aspects, image);

        let has_non_implicit_usage = if usage.is_empty() {
            usage = implicit_default_usage;
            false
        } else {
            usage == implicit_default_usage
        };

        if has_non_implicit_usage {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_maintenance2)
            {
                return Err(Box::new(ValidationError {
                    problem: "`create_info.usage` is not the implicit default usage \
                        (calculated from `image` and `create_info.subresource_range.aspects`)"
                        .into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance2")]),
                    ]),
                    ..Default::default()
                }));
            }

            // VUID-VkImageViewUsageCreateInfo-usage-requiredbitmask
            // Ensured because if it's empty, we copy the image usage
            // (which is already validated to not be empty).

            if !implicit_default_usage.contains(usage) {
                return Err(Box::new(ValidationError {
                    problem: "`create_info.usage` is not a subset of the implicit default usage \
                        (calculated from `image` and `create_info.subresource_range.aspects`)"
                        .into(),
                    vuids: &[
                        "VUID-VkImageViewCreateInfo-pNext-02662",
                        "VUID-VkImageViewCreateInfo-pNext-02663",
                        "VUID-VkImageViewCreateInfo-pNext-02664",
                    ],
                    ..Default::default()
                }));
            }
        }

        if !image.usage().intersects(
            ImageUsage::SAMPLED
                | ImageUsage::STORAGE
                | ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                | ImageUsage::INPUT_ATTACHMENT
                | ImageUsage::TRANSIENT_ATTACHMENT,
        ) {
            return Err(Box::new(ValidationError {
                context: "image.usage()".into(),
                problem: "does not contain one of `ImageUsage::SAMPLED`, `ImageUsage::STORAGE`, \
                    `ImageUsage::COLOR_ATTACHMENT`, `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, \
                    `ImageUsage::INPUT_ATTACHMENT` or `ImageUsage::TRANSIENT_ATTACHMENT`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-image-04441"],
                ..Default::default()
            }));
        }

        if usage.intersects(ImageUsage::SAMPLED)
            && !format_features.intersects(FormatFeatures::SAMPLED_IMAGE)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.usage` or the implicit default usage \
                    (calculated from `image` and `create_info.subresource_range.aspects`) \
                    contains `ImageUsage::SAMPLED`, but \
                    the format features of `create_info.format` do not contain \
                    `FormatFeatures::SAMPLED_IMAGE`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-usage-02274"],
                ..Default::default()
            }));
        }

        if usage.intersects(ImageUsage::STORAGE)
            && !format_features.intersects(FormatFeatures::STORAGE_IMAGE)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.usage` or the implicit default usage \
                    (calculated from `image` and `create_info.subresource_range.aspects`) \
                    contains `ImageUsage::STORAGE`, but \
                    the format features of `create_info.format` do not contain \
                    `FormatFeatures::STORAGE_IMAGE`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-usage-02275"],
                ..Default::default()
            }));
        }

        if usage.intersects(ImageUsage::COLOR_ATTACHMENT)
            && !format_features.intersects(FormatFeatures::COLOR_ATTACHMENT)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.usage` or the implicit default usage \
                    (calculated from `image` and `create_info.subresource_range.aspects`) \
                    contains `ImageUsage::COLOR_ATTACHMENT`, but \
                    the format features of `create_info.format` do not contain \
                    `FormatFeatures::COLOR_ATTACHMENT`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-usage-02276"],
                ..Default::default()
            }));
        }

        if usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            && !format_features.intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.usage` or the implicit default usage \
                    (calculated from `image` and `create_info.subresource_range.aspects`) \
                    contains `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, but \
                    the format features of `create_info.format` do not contain \
                    `FormatFeatures::DEPTH_STENCIL_ATTACHMENT`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-usage-02277"],
                ..Default::default()
            }));
        }

        if usage.intersects(ImageUsage::INPUT_ATTACHMENT)
            && !format_features.intersects(
                FormatFeatures::COLOR_ATTACHMENT | FormatFeatures::DEPTH_STENCIL_ATTACHMENT,
            )
        {
            return Err(Box::new(ValidationError {
                problem: "`create_info.usage` or the implicit default usage \
                    (calculated from `image` and `create_info.subresource_range.aspects`) \
                    contains `ImageUsage::INPUT_ATTACHMENT`, but \
                    the format features of `create_info.format` do not contain \
                    `FormatFeatures::COLOR_ATTACHMENT` or \
                    `FormatFeatures::DEPTH_STENCIL_ATTACHMENT`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-usage-02652"],
                ..Default::default()
            }));
        }

        /* Check flags requirements */

        if format != image.format() {
            if !image.format().planes().is_empty()
                && subresource_range.aspects.intersects(ImageAspects::COLOR)
            {
                return Err(Box::new(ValidationError {
                    problem: "`image.format()` is a multi-planar format, and \
                        `create_info.subresource_range.aspects` contains `ImageAspects::COLOR`, \
                        but `create_info.format` does not equal `image.format()`"
                        .into(),
                    vuids: &["VUID-VkImageViewCreateInfo-image-01762"],
                    ..Default::default()
                }));
            }

            if !image.flags().intersects(ImageCreateFlags::MUTABLE_FORMAT) {
                return Err(Box::new(ValidationError {
                    problem: "`create_info.format` does not equal `image.format()`, but \
                        `image.flags()` does not contain `ImageCreateFlags::MUTABLE_FORMAT`"
                        .into(),
                    vuids: &["VUID-VkImageViewCreateInfo-image-01762"],
                    ..Default::default()
                }));
            }

            // TODO: it is unclear what the number of bits is for compressed formats.
            // See https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2361
            if device.enabled_extensions().khr_portability_subset
                && !device.enabled_features().image_view_format_reinterpretation
                && format.components() != image.format().components()
            {
                return Err(Box::new(ValidationError {
                    problem: "this device is a portability subset device, and \
                        `create_info.format` does not have the same components and \
                        number of bits per component as `image.format()`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "image_view_format_reinterpretation",
                    )])]),
                    vuids: &["VUID-VkImageViewCreateInfo-imageViewFormatReinterpretation-04466"],
                    ..Default::default()
                }));
            }

            if image
                .flags()
                .intersects(ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE)
            {
                if !(format.compatibility() == image.format().compatibility()
                    || format.block_size() == image.format().block_size())
                {
                    return Err(Box::new(ValidationError {
                        problem: "`image.flags()` contains \
                            `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, but \
                            `create_info.format` is not compatible with `image.format()`, or \
                            does not have an equal block size"
                            .into(),
                        vuids: &["VUID-VkImageViewCreateInfo-image-01583"],
                        ..Default::default()
                    }));
                }

                if format.compression().is_none() {
                    if subresource_range.array_layers.len() != 1 {
                        return Err(Box::new(ValidationError {
                            problem: "`image.flags()` contains \
                                `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, and \
                                `create_info.format` is not a compressed format, but \
                                the length of `create_info.subresource_range.array_layers` \
                                is not 1"
                                .into(),
                            vuids: &["VUID-VkImageViewCreateInfo-image-07072"],
                            ..Default::default()
                        }));
                    }

                    if subresource_range.mip_levels.len() != 1 {
                        return Err(Box::new(ValidationError {
                            problem: "`image.flags()` contains \
                                `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, and \
                                `create_info.format` is not a compressed format, but \
                                the length of `create_info.subresource_range.mip_levels` \
                                is not 1"
                                .into(),
                            vuids: &["VUID-VkImageViewCreateInfo-image-07072"],
                            ..Default::default()
                        }));
                    }
                }
            } else {
                if image.format().planes().is_empty() {
                    if format.compatibility() != image.format().compatibility() {
                        return Err(Box::new(ValidationError {
                            problem: "`image.flags()` does not contain \
                                `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, and \
                                `image.format()` is not a multi-planar format, but \
                                is not compatible with `create_info.format`"
                                .into(),
                            vuids: &["VUID-VkImageViewCreateInfo-image-01761"],
                            ..Default::default()
                        }));
                    }
                } else if subresource_range.aspects.intersects(
                    ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2,
                ) {
                    let plane = if subresource_range.aspects.intersects(ImageAspects::PLANE_0) {
                        0
                    } else if subresource_range.aspects.intersects(ImageAspects::PLANE_1) {
                        1
                    } else if subresource_range.aspects.intersects(ImageAspects::PLANE_2) {
                        2
                    } else {
                        unreachable!()
                    };
                    let plane_format = image.format().planes()[plane];

                    if format.compatibility() != plane_format.compatibility() {
                        return Err(Box::new(ValidationError {
                            problem: "`image.flags()` does not contain \
                                `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, and \
                                `image.format()` is a multi-planar format, but \
                                `create_info.format` is not compatible with the format of the \
                                plane that is selected by `create_info.subresource_range.aspects`"
                                .into(),
                            vuids: &["VUID-VkImageViewCreateInfo-image-01586"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        if let Some(chroma_sampling) = format.ycbcr_chroma_sampling() {
            match chroma_sampling {
                ChromaSampling::Mode444 => (),
                ChromaSampling::Mode422 => {
                    if image.extent()[0] % 2 != 0 {
                        return Err(Box::new(ValidationError {
                            problem: "`create_info.format` is a YCbCr format with horizontal \
                                chroma subsampling, but \
                                `image.extent()[0]` is not \
                                a multiple of 2"
                                .into(),
                            vuids: &["VUID-VkImageViewCreateInfo-format-04714"],
                            ..Default::default()
                        }));
                    }
                }
                ChromaSampling::Mode420 => {
                    if !(image.extent()[0] % 2 == 0 && image.extent()[1] % 2 == 0) {
                        return Err(Box::new(ValidationError {
                            problem: "`create_info.format` is a YCbCr format with horizontal \
                                and vertical chroma subsampling, but \
                                `image.extent()[0]` and `image.extent()[1]` \
                                are not both a multiple of 2"
                                .into(),
                            vuids: &[
                                "VUID-VkImageViewCreateInfo-format-04714",
                                "VUID-VkImageViewCreateInfo-format-04715",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        image: Arc<Image>,
        create_info: ImageViewCreateInfo,
    ) -> Result<Arc<Self>, VulkanError> {
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
        let implicit_default_usage = get_implicit_default_usage(subresource_range.aspects, &image);

        let has_non_default_usage = if usage.is_empty() {
            usage = implicit_default_usage;
            false
        } else {
            usage == implicit_default_usage
        };

        let mut info_vk = ash::vk::ImageViewCreateInfo {
            flags: ash::vk::ImageViewCreateFlags::empty(),
            image: image.handle(),
            view_type: view_type.into(),
            format: format.into(),
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
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Self::from_handle(image, handle, create_info)
    }

    /// Creates a default `ImageView`. Equivalent to
    /// `ImageView::new(image, ImageViewCreateInfo::from_image(image))`.
    pub fn new_default(image: Arc<Image>) -> Result<Arc<ImageView>, Validated<VulkanError>> {
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
    ) -> Result<Arc<Self>, VulkanError> {
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
            usage = get_implicit_default_usage(subresource_range.aspects, &image);
        }

        let format_features = get_format_features(format, &image);

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
                        image_type: image.image_type(),
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
            image: DeviceOwnedDebugWrapper(image),
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
    pub fn format(&self) -> Format {
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

    /// Returns the features supported by the image view's format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.format_features
    }

    /// Returns whether the image view supports sampling with a
    /// [`Cubic`](crate::image::sampler::Filter::Cubic) `mag_filter` or `min_filter`.
    #[inline]
    pub fn filter_cubic(&self) -> bool {
        self.filter_cubic
    }

    /// Returns whether the image view supports sampling with a
    /// [`Cubic`](crate::image::sampler::Filter::Cubic) `mag_filter` or `min_filter`, and with a
    /// [`Min`](crate::image::sampler::SamplerReductionMode::Min) or
    /// [`Max`](crate::image::sampler::SamplerReductionMode::Max) `reduction_mode`.
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
    /// The default value is `Format::UNDEFINED`.
    pub format: Format,

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
    /// If `usage` is empty, then an implicit default value is used based on the parent image's
    /// usages.  Depending on the image aspects selected in `subresource_range`, the implicit
    /// `usage` will be equal to the parent image's `usage`, its `stencil_usage`,
    /// or the intersection of the two.
    ///
    /// If you set `usage` to a different value from the implicit default, then
    /// the device API version must be at least 1.1, or the
    /// [`khr_maintenance2`](crate::device::DeviceExtensions::khr_maintenance2)
    /// extension must be enabled on the device. The specified `usage` must be a subset of the
    /// implicit default value; usages that are not set for the parent image are not allowed.
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
            format: Format::UNDEFINED,
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
            view_type: match image.image_type() {
                ImageType::Dim1d => {
                    if image.array_layers() == 1 {
                        ImageViewType::Dim1d
                    } else {
                        ImageViewType::Dim1dArray
                    }
                }
                ImageType::Dim2d => {
                    if image.array_layers() == 1 {
                        ImageViewType::Dim2d
                    } else {
                        ImageViewType::Dim2dArray
                    }
                }
                ImageType::Dim3d => ImageViewType::Dim3d,
            },
            format: image.format(),
            subresource_range: image.subresource_range(),
            ..Default::default()
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            view_type,
            format,
            component_mapping,
            ref subresource_range,
            usage,
            ref sampler_ycbcr_conversion,
            _ne: _,
        } = self;

        view_type
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "view_type".into(),
                vuids: &["VUID-VkImageViewCreateInfo-viewType-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        format
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "format".into(),
                vuids: &["VUID-VkImageViewCreateInfo-format-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        component_mapping
            .validate(device)
            .map_err(|err| err.add_context("component_mapping"))?;

        subresource_range
            .validate(device)
            .map_err(|err| err.add_context("subresource_range"))?;

        usage
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "usage".into(),
                vuids: &["VUID-VkImageViewUsageCreateInfo-usage-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        match view_type {
            ImageViewType::Dim1d | ImageViewType::Dim2d | ImageViewType::Dim3d => {
                if subresource_range.array_layers.len() != 1 {
                    return Err(Box::new(ValidationError {
                        problem: "`view_type` is `ImageViewType::Dim1d`, \
                            `ImageViewType::Dim2d` or `ImageViewType::Dim3d`, but \
                            the length of `subresource_range.array_layers` is not 1"
                            .into(),
                        vuids: &["VUID-VkImageViewCreateInfo-imageViewType-04973"],
                        ..Default::default()
                    }));
                }
            }
            ImageViewType::Cube => {
                if subresource_range.array_layers.len() != 6 {
                    return Err(Box::new(ValidationError {
                        problem: "`view_type` is `ImageViewType::Cube`, but \
                            the length of `subresource_range.array_layers` is not 6"
                            .into(),
                        vuids: &["VUID-VkImageViewCreateInfo-viewType-02960"],
                        ..Default::default()
                    }));
                }
            }
            ImageViewType::CubeArray => {
                if !device.enabled_features().image_cube_array {
                    return Err(Box::new(ValidationError {
                        context: "view_type".into(),
                        problem: "is `ImageViewType::CubeArray`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "image_cube_array",
                        )])]),
                        vuids: &["VUID-VkImageViewCreateInfo-viewType-01004"],
                    }));
                }

                if subresource_range.array_layers.len() % 6 != 0 {
                    return Err(Box::new(ValidationError {
                        problem: "`view_type` is `ImageViewType::CubeArray`, but \
                            the length of `subresource_range.array_layers` is not \
                            a multiple of 6"
                            .into(),
                        vuids: &["VUID-VkImageViewCreateInfo-viewType-02961"],
                        ..Default::default()
                    }));
                }
            }
            _ => (),
        }

        if subresource_range
            .aspects
            .intersection(ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2)
            .count()
            > 1
        {
            return Err(Box::new(ValidationError {
                context: "subresource_range.aspects".into(),
                problem: "contains more than one of `ImageAspects::PLANE_0`, \
                    `ImageAspects::PLANE_1` and `ImageAspects::PLANE_2`"
                    .into(),
                vuids: &["VUID-VkImageViewCreateInfo-subresourceRange-07818"],
                ..Default::default()
            }));
        }

        if device.enabled_extensions().khr_portability_subset
            && !device.enabled_features().image_view_format_swizzle
            && !component_mapping.is_identity()
        {
            return Err(Box::new(ValidationError {
                problem: "this device is a portability subset device, and \
                    `component_mapping` is not the identity mapping"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "image_view_format_swizzle",
                )])]),
                vuids: &["VUID-VkImageViewCreateInfo-imageViewFormatSwizzle-04465"],
                ..Default::default()
            }));
        }

        // Don't need to check features because you can't create a conversion object without the
        // feature anyway.
        if let Some(conversion) = &sampler_ycbcr_conversion {
            assert_eq!(device, conversion.device().as_ref());

            if !component_mapping.is_identity() {
                return Err(Box::new(ValidationError {
                    problem: "`sampler_ycbcr_conversion` is `Some`, but \
                        `component_mapping` is not the identity mapping"
                        .into(),
                    vuids: &["VUID-VkImageViewCreateInfo-pNext-01970"],
                    ..Default::default()
                }));
            }
        } else {
            if format.ycbcr_chroma_sampling().is_some() {
                return Err(Box::new(ValidationError {
                    problem: "`sampler_ycbcr_conversion` is `None`, but \
                        `format.ycbcr_chroma_sampling()` is `Some`"
                        .into(),
                    vuids: &["VUID-VkImageViewCreateInfo-format-06415"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The geometry type of an image view.
    ImageViewType = ImageViewType(i32);

    /// A one-dimensional image view with a single array layer.
    Dim1d = TYPE_1D,

    /// A two-dimensional image view with a single array layer.
    Dim2d = TYPE_2D,

    /// A three-dimensional image view with a single array layer.
    Dim3d = TYPE_3D,

    /// A cube map image view, with six array layers.
    /// One array layer is mapped to each of the faces of the cube.
    Cube = CUBE,

    /// An arrayed one-dimensional image view, with one array layer per element.
    Dim1dArray = TYPE_1D_ARRAY,

    /// An arrayed two-dimensional image view, with one array layer per element.
    Dim2dArray = TYPE_2D_ARRAY,

    /// An arrayed cube map image view, with a multiple of six array layers.
    /// For each array element, one array layer is mapped to each of the faces of the cube.
    CubeArray = CUBE_ARRAY,
}

impl ImageViewType {
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
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#resources-image-views-compatibility
        matches!(
            (self, image_type),
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

// https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkImageViewCreateInfo.html#_description
fn get_implicit_default_usage(aspects: ImageAspects, image: &Image) -> ImageUsage {
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
unsafe fn get_format_features(view_format: Format, image: &Image) -> FormatFeatures {
    let device = image.device();

    let mut format_features = {
        // Use unchecked, because all validation should have been done before calling.
        let format_properties = device
            .physical_device()
            .format_properties_unchecked(view_format);
        let drm_format_modifiers: SmallVec<[_; 1]> = image
            .drm_format_modifier()
            .map_or_else(Default::default, |(m, _)| smallvec![m]);
        format_properties.format_features(image.tiling(), &drm_format_modifiers)
    };

    if !(device.api_version() >= Version::V1_3
        || device.enabled_extensions().khr_format_feature_flags2)
        && matches!(image.tiling(), ImageTiling::Linear | ImageTiling::Optimal)
    {
        if view_format.type_color().is_none()
            && format_features.intersects(FormatFeatures::SAMPLED_IMAGE)
        {
            format_features |= FormatFeatures::SAMPLED_IMAGE_DEPTH_COMPARISON;
        }

        if view_format.shader_storage_image_without_format() {
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
