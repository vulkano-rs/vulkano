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

use crate::device::physical::FormatFeatures;
use crate::device::{Device, DeviceOwned};
use crate::format::{ChromaSampling, Format};
use crate::image::{
    ImageAccess, ImageAspects, ImageDimensions, ImageTiling, ImageType, ImageUsage, SampleCount,
};
use crate::sampler::ycbcr::SamplerYcbcrConversion;
use crate::sampler::ComponentMapping;
use crate::OomError;
use crate::VulkanObject;
use crate::{check_errors, Error};
use std::error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

/// A wrapper around an image that makes it available to shaders or framebuffers.
pub struct ImageView<I>
where
    I: ImageAccess,
{
    handle: ash::vk::ImageView,
    image: Arc<I>,

    array_layers: Range<u32>,
    aspects: ImageAspects,
    component_mapping: ComponentMapping,
    format: Format,
    format_features: FormatFeatures,
    mip_levels: Range<u32>,
    sampler_ycbcr_conversion: Option<Arc<SamplerYcbcrConversion>>,
    ty: ImageViewType,
    usage: ImageUsage,

    filter_cubic: bool,
    filter_cubic_minmax: bool,
}

impl<I> ImageView<I>
where
    I: ImageAccess,
{
    /// Creates a default `ImageView`. Equivalent to `ImageView::start(image).build()`.
    #[inline]
    pub fn new(image: Arc<I>) -> Result<Arc<ImageView<I>>, ImageViewCreationError> {
        Self::start(image).build()
    }

    /// Begins building an `ImageView`.
    pub fn start(image: Arc<I>) -> ImageViewBuilder<I> {
        let array_layers = 0..image.dimensions().array_layers();
        let aspects = {
            let aspects = image.format().aspects();
            if aspects.depth || aspects.stencil {
                debug_assert!(!aspects.color);
                ImageAspects {
                    depth: aspects.depth,
                    stencil: aspects.stencil,
                    ..Default::default()
                }
            } else {
                debug_assert!(aspects.color);
                ImageAspects {
                    color: true,
                    ..Default::default()
                }
            }
        };
        let format = image.format();
        let mip_levels = 0..image.mip_levels();
        let ty = match image.dimensions() {
            ImageDimensions::Dim1d {
                array_layers: 1, ..
            } => ImageViewType::Dim1d,
            ImageDimensions::Dim1d { .. } => ImageViewType::Dim1dArray,
            ImageDimensions::Dim2d {
                array_layers: 1, ..
            } => ImageViewType::Dim2d,
            ImageDimensions::Dim2d { .. } => ImageViewType::Dim2dArray,
            ImageDimensions::Dim3d { .. } => ImageViewType::Dim3d,
        };

        ImageViewBuilder {
            image,

            array_layers,
            aspects,
            component_mapping: ComponentMapping::default(),
            format,
            mip_levels,
            sampler_ycbcr_conversion: None,
            ty,
        }
    }

    /// Returns the wrapped image that this image view was created from.
    pub fn image(&self) -> &Arc<I> {
        &self.image
    }
}

unsafe impl<I> VulkanObject for ImageView<I>
where
    I: ImageAccess,
{
    type Object = ash::vk::ImageView;

    #[inline]
    fn internal_object(&self) -> ash::vk::ImageView {
        self.handle
    }
}

unsafe impl<I> DeviceOwned for ImageView<I>
where
    I: ImageAccess,
{
    fn device(&self) -> &Arc<Device> {
        self.image.inner().image.device()
    }
}

impl<I> fmt::Debug for ImageView<I>
where
    I: ImageAccess,
{
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan image view {:?}>", self.handle)
    }
}

impl<I> Drop for ImageView<I>
where
    I: ImageAccess,
{
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let device = self.device();
            let fns = device.fns();
            fns.v1_0
                .destroy_image_view(device.internal_object(), self.handle, ptr::null());
        }
    }
}

impl<I> PartialEq for ImageView<I>
where
    I: ImageAccess,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl<I> Eq for ImageView<I> where I: ImageAccess {}

impl<I> Hash for ImageView<I>
where
    I: ImageAccess,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

#[derive(Debug)]
pub struct ImageViewBuilder<I> {
    image: Arc<I>,

    array_layers: Range<u32>,
    aspects: ImageAspects,
    component_mapping: ComponentMapping,
    format: Format,
    mip_levels: Range<u32>,
    sampler_ycbcr_conversion: Option<Arc<SamplerYcbcrConversion>>,
    ty: ImageViewType,
}

impl<I> ImageViewBuilder<I>
where
    I: ImageAccess,
{
    /// Builds the `ImageView`.
    pub fn build(self) -> Result<Arc<ImageView<I>>, ImageViewCreationError> {
        let Self {
            array_layers,
            aspects,
            component_mapping,
            format,
            mip_levels,
            ty,
            sampler_ycbcr_conversion,
            image,
        } = self;

        let image_inner = image.inner().image;
        let level_count = mip_levels.end - mip_levels.start;
        let layer_count = array_layers.end - array_layers.start;

        // Get format features
        let format_features = {
            let format_features = if format != image_inner.format() {
                let format_properties = image_inner
                    .device()
                    .physical_device()
                    .format_properties(self.format);

                match image_inner.tiling() {
                    ImageTiling::Optimal => format_properties.optimal_tiling_features,
                    ImageTiling::Linear => format_properties.linear_tiling_features,
                }
            } else {
                *image_inner.format_features()
            };

            // Per https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap12.html#resources-image-view-format-features
            if image_inner
                .device()
                .enabled_extensions()
                .khr_format_feature_flags2
            {
                format_features
            } else {
                let is_without_format = format.shader_storage_image_without_format();

                FormatFeatures {
                    sampled_image_depth_comparison: format.type_color().is_none()
                        && format_features.sampled_image,
                    storage_read_without_format: is_without_format
                        && image_inner
                            .device()
                            .enabled_features()
                            .shader_storage_image_read_without_format,
                    storage_write_without_format: is_without_format
                        && image_inner
                            .device()
                            .enabled_features()
                            .shader_storage_image_write_without_format,
                    ..format_features
                }
            }
        };

        // No VUID apparently, but this seems like something we want to check?
        if !image_inner.format().aspects().contains(&aspects) {
            return Err(ImageViewCreationError::ImageAspectsNotCompatible {
                aspects,
                image_aspects: image_inner.format().aspects(),
            });
        }

        // VUID-VkImageViewCreateInfo-None-02273
        if format_features == FormatFeatures::default() {
            return Err(ImageViewCreationError::FormatNotSupported);
        }

        // Get usage
        // Can be different from image usage, see
        // https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkImageViewCreateInfo.html#_description
        let usage = *image_inner.usage();

        // Check for compatibility with the image
        let image_type = image.dimensions().image_type();

        // VUID-VkImageViewCreateInfo-subResourceRange-01021
        if !ty.is_compatible_with(image_type) {
            return Err(ImageViewCreationError::ImageTypeNotCompatible);
        }

        // VUID-VkImageViewCreateInfo-image-01003
        if (ty == ImageViewType::Cube || ty == ImageViewType::CubeArray)
            && !image_inner.flags().cube_compatible
        {
            return Err(ImageViewCreationError::ImageNotCubeCompatible);
        }

        // VUID-VkImageViewCreateInfo-viewType-01004
        if ty == ImageViewType::CubeArray
            && !image_inner.device().enabled_features().image_cube_array
        {
            return Err(ImageViewCreationError::FeatureNotEnabled {
                feature: "image_cube_array",
                reason: "the `CubeArray` view type was requested",
            });
        }

        // VUID-VkImageViewCreateInfo-subresourceRange-01718
        if mip_levels.end > image_inner.mip_levels() {
            return Err(ImageViewCreationError::MipLevelsOutOfRange {
                range_end: mip_levels.end,
                max: image_inner.mip_levels(),
            });
        }

        if image_type == ImageType::Dim3d
            && (ty == ImageViewType::Dim2d || ty == ImageViewType::Dim2dArray)
        {
            // VUID-VkImageViewCreateInfo-image-01005
            if !image_inner.flags().array_2d_compatible {
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
            let max = image_inner
                .dimensions()
                .mip_level_dimensions(mip_levels.start)
                .unwrap()
                .depth();
            if array_layers.end > max {
                return Err(ImageViewCreationError::ArrayLayersOutOfRange {
                    range_end: array_layers.end,
                    max,
                });
            }
        } else {
            // VUID-VkImageViewCreateInfo-image-01482
            // VUID-VkImageViewCreateInfo-subresourceRange-01483
            if array_layers.end > image_inner.dimensions().array_layers() {
                return Err(ImageViewCreationError::ArrayLayersOutOfRange {
                    range_end: array_layers.end,
                    max: image_inner.dimensions().array_layers(),
                });
            }
        }

        // VUID-VkImageViewCreateInfo-image-04972
        if image_inner.samples() != SampleCount::Sample1
            && !(ty == ImageViewType::Dim2d || ty == ImageViewType::Dim2dArray)
        {
            return Err(ImageViewCreationError::MultisamplingNot2d);
        }

        /* Check usage requirements */

        // VUID-VkImageViewCreateInfo-image-04441
        if !(image_inner.usage().sampled
            || image_inner.usage().storage
            || image_inner.usage().color_attachment
            || image_inner.usage().depth_stencil_attachment
            || image_inner.usage().input_attachment
            || image_inner.usage().transient_attachment)
        {
            return Err(ImageViewCreationError::InvalidImageUsage);
        }

        // VUID-VkImageViewCreateInfo-usage-02274
        if usage.sampled && !format_features.sampled_image {
            return Err(ImageViewCreationError::FormatUsageNotSupported { usage: "sampled" });
        }

        // VUID-VkImageViewCreateInfo-usage-02275
        if usage.storage && !format_features.storage_image {
            return Err(ImageViewCreationError::FormatUsageNotSupported { usage: "storage" });
        }

        // VUID-VkImageViewCreateInfo-usage-02276
        if usage.color_attachment && !format_features.color_attachment {
            return Err(ImageViewCreationError::FormatUsageNotSupported {
                usage: "color_attachment",
            });
        }

        // VUID-VkImageViewCreateInfo-usage-02277
        if usage.depth_stencil_attachment && !format_features.depth_stencil_attachment {
            return Err(ImageViewCreationError::FormatUsageNotSupported {
                usage: "depth_stencil_attachment",
            });
        }

        // VUID-VkImageViewCreateInfo-usage-02652
        if usage.input_attachment
            && !(format_features.color_attachment || format_features.depth_stencil_attachment)
        {
            return Err(ImageViewCreationError::FormatUsageNotSupported {
                usage: "input_attachment",
            });
        }

        /* Check flags requirements */

        if image_inner.flags().block_texel_view_compatible {
            // VUID-VkImageViewCreateInfo-image-01583
            if !(format.compatibility() == image_inner.format().compatibility()
                || format.block_size() == image_inner.format().block_size())
            {
                return Err(ImageViewCreationError::FormatNotCompatible);
            }

            // VUID-VkImageViewCreateInfo-image-01584
            if layer_count != 1 {
                return Err(ImageViewCreationError::BlockTexelViewCompatibleMultipleArrayLayers);
            }

            // VUID-VkImageViewCreateInfo-image-01584
            if level_count != 1 {
                return Err(ImageViewCreationError::BlockTexelViewCompatibleMultipleMipLevels);
            }

            // VUID-VkImageViewCreateInfo-image-04739
            if format.compression().is_none() && ty == ImageViewType::Dim3d {
                return Err(ImageViewCreationError::BlockTexelViewCompatibleUncompressedIs3d);
            }
        }
        // VUID-VkImageViewCreateInfo-image-01761
        else if image_inner.flags().mutable_format
            && image_inner.format().planes().is_empty()
            && format.compatibility() != image_inner.format().compatibility()
        {
            return Err(ImageViewCreationError::FormatNotCompatible);
        }

        if image_inner.flags().mutable_format
            && !image_inner.format().planes().is_empty()
            && !aspects.color
        {
            let plane = if aspects.plane0 {
                0
            } else if aspects.plane1 {
                1
            } else if aspects.plane2 {
                2
            } else {
                unreachable!()
            };
            let plane_format = image_inner.format().planes()[plane];

            // VUID-VkImageViewCreateInfo-image-01586
            if format.compatibility() != plane_format.compatibility() {
                return Err(ImageViewCreationError::FormatNotCompatible);
            }
        }
        // VUID-VkImageViewCreateInfo-image-01762
        else if format != image_inner.format() {
            return Err(ImageViewCreationError::FormatNotCompatible);
        }

        // VUID-VkImageViewCreateInfo-imageViewType-04973
        if (ty == ImageViewType::Dim1d || ty == ImageViewType::Dim2d || ty == ImageViewType::Dim3d)
            && layer_count != 1
        {
            return Err(ImageViewCreationError::TypeNonArrayedMultipleArrayLayers);
        }
        // VUID-VkImageViewCreateInfo-viewType-02960
        else if ty == ImageViewType::Cube && layer_count != 6 {
            return Err(ImageViewCreationError::TypeCubeNot6ArrayLayers);
        }
        // VUID-VkImageViewCreateInfo-viewType-02961
        else if ty == ImageViewType::CubeArray && layer_count % 6 != 0 {
            return Err(ImageViewCreationError::TypeCubeArrayNotMultipleOf6ArrayLayers);
        }

        // VUID-VkImageViewCreateInfo-format-04714
        // VUID-VkImageViewCreateInfo-format-04715
        match format.ycbcr_chroma_sampling() {
            Some(ChromaSampling::Mode422) => {
                if image_inner.dimensions().width() % 2 != 0 {
                    return Err(
                        ImageViewCreationError::FormatChromaSubsamplingInvalidImageDimensions,
                    );
                }
            }
            Some(ChromaSampling::Mode420) => {
                if image_inner.dimensions().width() % 2 != 0
                    || image_inner.dimensions().height() % 2 != 0
                {
                    return Err(
                        ImageViewCreationError::FormatChromaSubsamplingInvalidImageDimensions,
                    );
                }
            }
            _ => (),
        }

        // Don't need to check features because you can't create a conversion object without the
        // feature anyway.
        let mut sampler_ycbcr_conversion_info = if let Some(conversion) = &sampler_ycbcr_conversion
        {
            assert_eq!(image_inner.device(), conversion.device());

            // VUID-VkImageViewCreateInfo-pNext-01970
            if !component_mapping.is_identity() {
                return Err(
                    ImageViewCreationError::SamplerYcbcrConversionComponentMappingNotIdentity {
                        component_mapping,
                    },
                );
            }

            Some(ash::vk::SamplerYcbcrConversionInfo {
                conversion: conversion.internal_object(),
                ..Default::default()
            })
        } else {
            // VUID-VkImageViewCreateInfo-format-06415
            if format.ycbcr_chroma_sampling().is_some() {
                return Err(
                    ImageViewCreationError::FormatRequiresSamplerYcbcrConversion { format },
                );
            }

            None
        };

        let mut create_info = ash::vk::ImageViewCreateInfo {
            flags: ash::vk::ImageViewCreateFlags::empty(),
            image: image_inner.internal_object(),
            view_type: ty.into(),
            format: format.into(),
            components: component_mapping.into(),
            subresource_range: ash::vk::ImageSubresourceRange {
                aspect_mask: aspects.into(),
                base_mip_level: mip_levels.start,
                level_count,
                base_array_layer: array_layers.start,
                layer_count,
            },
            ..Default::default()
        };

        if let Some(sampler_ycbcr_conversion_info) = sampler_ycbcr_conversion_info.as_mut() {
            sampler_ycbcr_conversion_info.p_next = create_info.p_next;
            create_info.p_next = sampler_ycbcr_conversion_info as *const _ as *const _;
        }

        let handle = unsafe {
            let fns = image_inner.device().fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_image_view(
                image_inner.device().internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        let (filter_cubic, filter_cubic_minmax) = if let Some(properties) = image_inner
            .device()
            .physical_device()
            .image_format_properties(
                image_inner.format(),
                image_type,
                image_inner.tiling(),
                *image_inner.usage(),
                image_inner.flags(),
                None,
                Some(ty),
            )? {
            (properties.filter_cubic, properties.filter_cubic_minmax)
        } else {
            (false, false)
        };

        Ok(Arc::new(ImageView {
            handle,
            image,

            array_layers,
            aspects,
            component_mapping,
            format,
            format_features,
            mip_levels,
            sampler_ycbcr_conversion,
            ty,
            usage,

            filter_cubic,
            filter_cubic_minmax,
        }))
    }

    /// The range of array layers of the image that the view should cover.
    ///
    /// The default value is the full range of array layers present in the image.
    ///
    /// # Panics
    ///
    /// - Panics if `array_layers` is empty.
    #[inline]
    pub fn array_layers(mut self, array_layers: Range<u32>) -> Self {
        assert!(!array_layers.is_empty());
        self.array_layers = array_layers;
        self
    }

    /// The aspects of the image that the view should cover.
    ///
    /// The default value is `color` if the image is a color format, `depth` and/or `stencil` if
    /// the image is a depth/stencil format.
    ///
    /// # Panics
    ///
    /// - Panics if aspects other than `color`, `depth`, `stencil`, `plane0`, `plane1` or `plane2`
    ///   are selected.
    /// - Panics if more than one aspect is selected, unless `depth` and `stencil` are the only
    ///   aspects selected.
    #[inline]
    pub fn aspects(mut self, aspects: ImageAspects) -> Self {
        let ImageAspects {
            color,
            depth,
            stencil,
            metadata,
            plane0,
            plane1,
            plane2,
            memory_plane0,
            memory_plane1,
            memory_plane2,
        } = aspects;

        assert!(!(metadata || memory_plane0 || memory_plane1 || memory_plane2));
        assert!({
            let num_bits = color as u8
                + depth as u8
                + stencil as u8
                + plane0 as u8
                + plane1 as u8
                + plane2 as u8;
            num_bits == 1 || depth && stencil && !(color || plane0 || plane1 || plane2)
        });

        self.aspects = aspects;
        self
    }

    /// How to map components of each pixel.
    ///
    /// The default value is [`ComponentMapping::identity()`].
    #[inline]
    pub fn component_mapping(mut self, component_mapping: ComponentMapping) -> Self {
        self.component_mapping = component_mapping;
        self
    }

    /// The format of the image view.
    ///
    /// If this is set to a format that is different from the image, the image must be created with
    /// the `mutable_format` flag.
    ///
    /// The default value is the format of the image.
    #[inline]
    pub fn format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    /// The range of mipmap levels of the image that the view should cover.
    ///
    /// The default value is the full range of mipmaps present in the image.
    ///
    /// # Panics
    ///
    /// - Panics if `mip_levels` is empty.
    #[inline]
    pub fn mip_levels(mut self, mip_levels: Range<u32>) -> Self {
        assert!(!mip_levels.is_empty());
        self.mip_levels = mip_levels;
        self
    }

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
    #[inline]
    pub fn sampler_ycbcr_conversion(
        mut self,
        conversion: Option<Arc<SamplerYcbcrConversion>>,
    ) -> Self {
        self.sampler_ycbcr_conversion = conversion;
        self
    }

    /// The image view type.
    ///
    /// The view type must be compatible with the dimensions of the image and the selected array
    /// layers.
    ///
    /// The default value is determined from the image, based on its dimensions and number of
    /// layers.
    #[inline]
    pub fn ty(mut self, ty: ImageViewType) -> Self {
        self.ty = ty;
        self
    }
}

/// Error that can happen when creating an image view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageViewCreationError {
    /// Allocating memory failed.
    OomError(OomError),

    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
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

    /// The image has the `block_texel_view_compatible` flag, and an uncompressed format was
    /// requested, and the image view type was `Dim3d`.
    BlockTexelViewCompatibleUncompressedIs3d,

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

    /// The image was not created with
    /// [one of the required usages](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#valid-imageview-imageusage)
    /// for image views.
    InvalidImageUsage,

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
}

impl error::Error for ImageViewCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            ImageViewCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for ImageViewCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::OomError(err) => write!(
                fmt,
                "allocating memory failed",
            ),
            Self::FeatureNotEnabled { feature, reason } => {
                write!(fmt, "the feature {} must be enabled: {}", feature, reason)
            }
            Self::Array2dCompatibleMultipleMipLevels => write!(
                fmt,
                "a 2D image view was requested from a 3D image, but a range of multiple mip levels was specified",
            ),
            Self::ArrayLayersOutOfRange { .. } => write!(
                fmt,
                "the specified range of array layers was not a subset of those in the image",
            ),
            Self::BlockTexelViewCompatibleMultipleArrayLayers => write!(
                fmt,
                "the image has the `block_texel_view_compatible` flag, but a range of multiple array layers was specified",
            ),
            Self::BlockTexelViewCompatibleMultipleMipLevels => write!(
                fmt,
                "the image has the `block_texel_view_compatible` flag, but a range of multiple mip levels was specified",
            ),
            Self::BlockTexelViewCompatibleUncompressedIs3d => write!(
                fmt,
                "the image has the `block_texel_view_compatible` flag, and an uncompressed format was requested, and the image view type was `Dim3d`",
            ),
            Self::FormatChromaSubsamplingInvalidImageDimensions => write!(
                fmt,
                "the requested format has chroma subsampling, but the width and/or height of the image was not a multiple of 2",
            ),
            Self::FormatNotCompatible => write!(
                fmt,
                "the requested format was not compatible with the image",
            ),
            Self::FormatNotSupported => write!(
                fmt,
                "the given format was not supported by the device"
            ),
            Self::FormatRequiresSamplerYcbcrConversion { .. } => write!(
                fmt,
                "the format requires a sampler YCbCr conversion, but none was provided",
            ),
            Self::FormatUsageNotSupported { usage } => write!(
                fmt,
                "a requested usage flag was not supported by the given format"
            ),
            Self::ImageAspectsNotCompatible { .. } => write!(
                fmt,
                "an aspect was selected that was not present in the image",
            ),
            Self::ImageNotArray2dCompatible => write!(
                fmt,
                "a 2D image view was requested from a 3D image, but the image was not created with the `array_2d_compatible` flag",
            ),
            Self::ImageNotCubeCompatible => write!(
                fmt,
                "a cube image view type was requested, but the image was not created with the `cube_compatible` flag",
            ),
            Self::ImageTypeNotCompatible => write!(
                fmt,
                "the given image view type was not compatible with the type of the image",
            ),
            Self::IncompatibleType => write!(
                fmt,
                "image view type is not compatible with image, array layers or mipmap levels",
            ),
            Self::InvalidImageUsage => write!(
                fmt,
                "the usage of the image is not compatible with image views",
            ),
            Self::MipLevelsOutOfRange { .. } => write!(
                fmt,
                "the specified range of mip levels was not a subset of those in the image",
            ),
            Self::MultisamplingNot2d => write!(
                fmt,
                "the image has multisampling enabled, but the image view type was not `Dim2d` or `Dim2dArray`",
            ),
            Self::SamplerYcbcrConversionComponentMappingNotIdentity { .. } => write!(
                fmt,
                "sampler YCbCr conversion was enabled, but `component_mapping` was not the identity mapping",
            ),
            Self::TypeCubeArrayNotMultipleOf6ArrayLayers => write!(
                fmt,
                "the `CubeArray` image view type was specified, but the range of array layers did not have a size that is a multiple 6"
            ),
            Self::TypeCubeNot6ArrayLayers => write!(
                fmt,
                "the `Cube` image view type was specified, but the range of array layers did not have a size of 6"
            ),
            Self::TypeNonArrayedMultipleArrayLayers => write!(
                fmt,
                "a non-arrayed image view type was specified, but a range of multiple array layers was specified"
            )
        }
    }
}

impl From<OomError> for ImageViewCreationError {
    #[inline]
    fn from(err: OomError) -> ImageViewCreationError {
        ImageViewCreationError::OomError(err)
    }
}

impl From<Error> for ImageViewCreationError {
    #[inline]
    fn from(err: Error) -> ImageViewCreationError {
        match err {
            err @ Error::OutOfHostMemory => OomError::from(err).into(),
            err @ Error::OutOfDeviceMemory => OomError::from(err).into(),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// The geometry type of an image view.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ImageViewType {
    Dim1d = ash::vk::ImageViewType::TYPE_1D.as_raw(),
    Dim1dArray = ash::vk::ImageViewType::TYPE_1D_ARRAY.as_raw(),
    Dim2d = ash::vk::ImageViewType::TYPE_2D.as_raw(),
    Dim2dArray = ash::vk::ImageViewType::TYPE_2D_ARRAY.as_raw(),
    Dim3d = ash::vk::ImageViewType::TYPE_3D.as_raw(),
    Cube = ash::vk::ImageViewType::CUBE.as_raw(),
    CubeArray = ash::vk::ImageViewType::CUBE_ARRAY.as_raw(),
}

impl ImageViewType {
    /// Returns whether the type is arrayed.
    #[inline]
    pub fn is_arrayed(&self) -> bool {
        match self {
            Self::Dim1d | Self::Dim2d | Self::Dim3d | Self::Cube => false,
            Self::Dim1dArray | Self::Dim2dArray | Self::CubeArray => true,
        }
    }

    /// Returns whether `self` is compatible with the given `image_type`.
    #[inline]
    pub fn is_compatible_with(&self, image_type: ImageType) -> bool {
        matches!(
            (*self, image_type,),
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

impl From<ImageViewType> for ash::vk::ImageViewType {
    fn from(val: ImageViewType) -> Self {
        Self::from_raw(val as i32)
    }
}

/// Trait for types that represent the GPU can access an image view.
pub unsafe trait ImageViewAbstract:
    VulkanObject<Object = ash::vk::ImageView> + DeviceOwned + Send + Sync
{
    /// Returns the wrapped image that this image view was created from.
    fn image(&self) -> Arc<dyn ImageAccess>;

    /// Returns the range of array layers of the wrapped image that this view exposes.
    fn array_layers(&self) -> Range<u32>;

    /// Returns the aspects of the wrapped image that this view exposes.
    fn aspects(&self) -> &ImageAspects;

    /// Returns the component mapping of this view.
    fn component_mapping(&self) -> ComponentMapping;

    /// Returns whether the image view supports sampling with a
    /// [`Cubic`](crate::sampler::Filter::Cubic) `mag_filter` or `min_filter`.
    fn filter_cubic(&self) -> bool;

    /// Returns whether the image view supports sampling with a
    /// [`Cubic`](crate::sampler::Filter::Cubic) `mag_filter` or `min_filter`, and with a
    /// [`Min`](crate::sampler::SamplerReductionMode::Min) or
    /// [`Max`](crate::sampler::SamplerReductionMode::Max) `reduction_mode`.
    fn filter_cubic_minmax(&self) -> bool;

    /// Returns the format of this view. This can be different from the parent's format.
    fn format(&self) -> Format;

    /// Returns the features supported by the image view's format.
    fn format_features(&self) -> &FormatFeatures;

    /// Returns the range of mip levels of the wrapped image that this view exposes.
    fn mip_levels(&self) -> Range<u32>;

    /// Returns the sampler YCbCr conversion that this image view was created with, if any.
    fn sampler_ycbcr_conversion(&self) -> Option<&Arc<SamplerYcbcrConversion>>;

    /// Returns the [`ImageViewType`] of this image view.
    fn ty(&self) -> ImageViewType;

    /// Returns the usage of the image view.
    fn usage(&self) -> &ImageUsage;
}

unsafe impl<I> ImageViewAbstract for ImageView<I>
where
    I: ImageAccess + 'static,
{
    #[inline]
    fn image(&self) -> Arc<dyn ImageAccess> {
        self.image.clone() as Arc<_>
    }

    #[inline]
    fn array_layers(&self) -> Range<u32> {
        self.array_layers.clone()
    }

    #[inline]
    fn aspects(&self) -> &ImageAspects {
        &self.aspects
    }

    #[inline]
    fn component_mapping(&self) -> ComponentMapping {
        self.component_mapping
    }

    #[inline]
    fn filter_cubic(&self) -> bool {
        self.filter_cubic
    }

    #[inline]
    fn filter_cubic_minmax(&self) -> bool {
        self.filter_cubic_minmax
    }

    #[inline]
    fn format(&self) -> Format {
        self.format
    }

    #[inline]
    fn format_features(&self) -> &FormatFeatures {
        &self.format_features
    }

    #[inline]
    fn mip_levels(&self) -> Range<u32> {
        self.mip_levels.clone()
    }

    #[inline]
    fn sampler_ycbcr_conversion(&self) -> Option<&Arc<SamplerYcbcrConversion>> {
        self.sampler_ycbcr_conversion.as_ref()
    }

    #[inline]
    fn ty(&self) -> ImageViewType {
        self.ty
    }

    #[inline]
    fn usage(&self) -> &ImageUsage {
        &self.usage
    }
}

impl PartialEq for dyn ImageViewAbstract {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.internal_object() == other.internal_object() && self.device() == other.device()
    }
}

impl Eq for dyn ImageViewAbstract {}

impl Hash for dyn ImageViewAbstract {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.internal_object().hash(state);
        self.device().hash(state);
    }
}
