// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level implementation of images.
//!
//! This module contains low-level wrappers around the Vulkan image types. All
//! other image types of this library, and all custom image types
//! that you create must wrap around the types in this module.

use super::{
    ImageAspect, ImageAspects, ImageCreateFlags, ImageDimensions, ImageLayout,
    ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageUsage, SampleCount,
    SampleCounts, SparseImageMemoryRequirements,
};
use crate::{
    buffer::cpu_access::{ReadLockError, WriteLockError},
    cache::OnceCache,
    device::{Device, DeviceOwned},
    format::{ChromaSampling, Format, FormatFeatures, NumericType},
    image::{
        view::ImageViewCreationError, ImageFormatInfo, ImageFormatProperties, ImageType,
        SparseImageFormatProperties,
    },
    memory::{
        allocator::{AllocationCreationError, MemoryAlloc},
        DedicatedTo, ExternalMemoryHandleType, ExternalMemoryHandleTypes, MemoryPropertyFlags,
        MemoryRequirements,
    },
    range_map::RangeMap,
    swapchain::Swapchain,
    sync::{AccessError, CurrentAccess, Sharing},
    DeviceSize, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use smallvec::{smallvec, SmallVec};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    iter::{FusedIterator, Peekable},
    mem::{size_of_val, MaybeUninit},
    num::NonZeroU64,
    ops::Range,
    ptr,
    sync::Arc,
};

/// A raw image, with no memory backing it.
///
/// This is the basic image type, a direct translation of a `VkImage` object, but it is mostly
/// useless in this form. After creating a raw image, you must call `bind_memory` to make a
/// complete image object.
#[derive(Debug)]
pub struct RawImage {
    handle: ash::vk::Image,
    device: Arc<Device>,
    id: NonZeroU64,

    flags: ImageCreateFlags,
    dimensions: ImageDimensions,
    format: Option<Format>,
    format_features: FormatFeatures,
    initial_layout: ImageLayout,
    mip_levels: u32,
    samples: SampleCount,
    tiling: ImageTiling,
    usage: ImageUsage,
    sharing: Sharing<SmallVec<[u32; 4]>>,
    stencil_usage: ImageUsage,
    external_memory_handle_types: ExternalMemoryHandleTypes,

    memory_requirements: SmallVec<[MemoryRequirements; 3]>,
    needs_destruction: bool, // `vkDestroyImage` is called only if true.
    subresource_layout: OnceCache<(ImageAspect, u32, u32), SubresourceLayout>,
}

impl RawImage {
    /// Creates a new `RawImage`.
    ///
    /// # Panics
    ///
    /// - Panics if one of the values in `create_info.dimensions` is zero.
    /// - Panics if `create_info.format` is `None`.
    /// - Panics if `create_info.block_texel_view_compatible` is set but not
    ///   `create_info.mutable_format`.
    /// - Panics if `create_info.mip_levels` is `0`.
    /// - Panics if `create_info.sharing` is [`Sharing::Concurrent`] with less than 2 items.
    /// - Panics if `create_info.initial_layout` is something other than
    ///   [`ImageLayout::Undefined`] or [`ImageLayout::Preinitialized`].
    /// - Panics if `create_info.usage` is empty.
    /// - Panics if `create_info.usage` contains `transient_attachment`, but does not also contain
    ///   at least one of `color_attachment`, `depth_stencil_attachment`, `input_attachment`, or
    ///   if it contains values other than these.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        mut create_info: ImageCreateInfo,
    ) -> Result<RawImage, ImageError> {
        match &mut create_info.sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                // VUID-VkImageCreateInfo-sharingMode-01420
                queue_family_indices.sort_unstable();
                queue_family_indices.dedup();
            }
        }

        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(RawImage::new_unchecked(device, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        create_info: &ImageCreateInfo,
    ) -> Result<FormatFeatures, ImageError> {
        let &ImageCreateInfo {
            flags,
            dimensions,
            format,
            mip_levels,
            samples,
            tiling,
            usage,
            mut stencil_usage,
            ref sharing,
            initial_layout,
            external_memory_handle_types,
            _ne: _,
        } = create_info;

        let physical_device = device.physical_device();
        let device_properties = physical_device.properties();

        let format = format.unwrap(); // Can be None for "external formats" but Vulkano doesn't support that yet
        let aspects = format.aspects();

        let has_separate_stencil_usage = if stencil_usage.is_empty()
            || !aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
        {
            stencil_usage = usage;
            false
        } else {
            stencil_usage == usage
        };

        // VUID-VkImageCreateInfo-flags-parameter
        flags.validate_device(device)?;

        // VUID-VkImageCreateInfo-format-parameter
        format.validate_device(device)?;

        // VUID-VkImageCreateInfo-samples-parameter
        samples.validate_device(device)?;

        // VUID-VkImageCreateInfo-tiling-parameter
        tiling.validate_device(device)?;

        // VUID-VkImageCreateInfo-usage-parameter
        usage.validate_device(device)?;

        // VUID-VkImageCreateInfo-usage-requiredbitmask
        assert!(!usage.is_empty());

        if has_separate_stencil_usage {
            if !(device.api_version() >= Version::V1_2
                || device.enabled_extensions().ext_separate_stencil_usage)
            {
                return Err(ImageError::RequirementNotMet {
                    required_for: "`create_info.stencil_usage` is `Some` and `create_info.format` \
                        has both a depth and a stencil aspect",
                    requires_one_of: RequiresOneOf {
                        api_version: Some(Version::V1_2),
                        device_extensions: &["ext_separate_stencil_usage"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkImageStencilUsageCreateInfo-stencilUsage-parameter
            stencil_usage.validate_device(device)?;

            // VUID-VkImageStencilUsageCreateInfo-usage-requiredbitmask
            assert!(!stencil_usage.is_empty());
        }

        // VUID-VkImageCreateInfo-initialLayout-parameter
        initial_layout.validate_device(device)?;

        // VUID-VkImageCreateInfo-initialLayout-00993
        assert!(matches!(
            initial_layout,
            ImageLayout::Undefined | ImageLayout::Preinitialized
        ));

        // VUID-VkImageCreateInfo-flags-01573
        assert!(!flags.intersects(
            ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE | ImageCreateFlags::MUTABLE_FORMAT
        ));

        // Get format features
        let format_features = {
            // Use unchecked, because all validation has been done above.
            let format_properties = unsafe { physical_device.format_properties_unchecked(format) };
            match tiling {
                ImageTiling::Linear => format_properties.linear_tiling_features,
                ImageTiling::Optimal => format_properties.optimal_tiling_features,
            }
        };

        // Format isn't supported at all?
        if format_features.is_empty() {
            return Err(ImageError::FormatNotSupported);
        }

        // Decode the dimensions
        let (image_type, extent, array_layers) = match dimensions {
            ImageDimensions::Dim1d {
                width,
                array_layers,
            } => (ImageType::Dim1d, [width, 1, 1], array_layers),
            ImageDimensions::Dim2d {
                width,
                height,
                array_layers,
            } => (ImageType::Dim2d, [width, height, 1], array_layers),
            ImageDimensions::Dim3d {
                width,
                height,
                depth,
            } => (ImageType::Dim3d, [width, height, depth], 1),
        };

        // VUID-VkImageCreateInfo-extent-00944
        assert!(extent[0] != 0);

        // VUID-VkImageCreateInfo-extent-00945
        assert!(extent[1] != 0);

        // VUID-VkImageCreateInfo-extent-00946
        assert!(extent[2] != 0);

        // VUID-VkImageCreateInfo-arrayLayers-00948
        assert!(array_layers != 0);

        // VUID-VkImageCreateInfo-mipLevels-00947
        assert!(mip_levels != 0);

        // Check mip levels

        let max_mip_levels = dimensions.max_mip_levels();
        debug_assert!(max_mip_levels >= 1);

        // VUID-VkImageCreateInfo-mipLevels-00958
        if mip_levels > max_mip_levels {
            return Err(ImageError::MaxMipLevelsExceeded {
                mip_levels,
                max: max_mip_levels,
            });
        }

        // VUID-VkImageCreateInfo-samples-02257
        if samples != SampleCount::Sample1 {
            if image_type != ImageType::Dim2d {
                return Err(ImageError::MultisampleNot2d);
            }

            if flags.intersects(ImageCreateFlags::CUBE_COMPATIBLE) {
                return Err(ImageError::MultisampleCubeCompatible);
            }

            if mip_levels != 1 {
                return Err(ImageError::MultisampleMultipleMipLevels);
            }

            if tiling == ImageTiling::Linear {
                return Err(ImageError::MultisampleLinearTiling);
            }

            // VUID-VkImageCreateInfo-multisampleArrayImage-04460
            if device.enabled_extensions().khr_portability_subset
                && !device.enabled_features().multisample_array_image
                && array_layers != 1
            {
                return Err(ImageError::RequirementNotMet {
                    required_for: "this device is a portability subset device, \
                        `create_info.samples` is not `SampleCount::Sample1` and \
                        `create_info.dimensions.array_layers()` is greater than `1`",
                    requires_one_of: RequiresOneOf {
                        features: &["multisample_array_image"],
                        ..Default::default()
                    },
                });
            }
        }

        // Check limits for YCbCr formats
        if let Some(chroma_sampling) = format.ycbcr_chroma_sampling() {
            // VUID-VkImageCreateInfo-format-06410
            if mip_levels != 1 {
                return Err(ImageError::YcbcrFormatMultipleMipLevels);
            }

            // VUID-VkImageCreateInfo-format-06411
            if samples != SampleCount::Sample1 {
                return Err(ImageError::YcbcrFormatMultisampling);
            }

            // VUID-VkImageCreateInfo-format-06412
            if image_type != ImageType::Dim2d {
                return Err(ImageError::YcbcrFormatNot2d);
            }

            // VUID-VkImageCreateInfo-format-06413
            if array_layers > 1 && !device.enabled_features().ycbcr_image_arrays {
                return Err(ImageError::RequirementNotMet {
                    required_for: "`create_info.format.ycbcr_chroma_sampling()` is `Some` and \
                        `create_info.dimensions.array_layers()` is greater than `1`",
                    requires_one_of: RequiresOneOf {
                        features: &["ycbcr_image_arrays"],
                        ..Default::default()
                    },
                });
            }

            match chroma_sampling {
                ChromaSampling::Mode444 => (),
                ChromaSampling::Mode422 => {
                    // VUID-VkImageCreateInfo-format-04712
                    if extent[0] % 2 != 0 {
                        return Err(ImageError::YcbcrFormatInvalidDimensions);
                    }
                }
                ChromaSampling::Mode420 => {
                    // VUID-VkImageCreateInfo-format-04712
                    // VUID-VkImageCreateInfo-format-04713
                    if !(extent[0] % 2 == 0 && extent[1] % 2 == 0) {
                        return Err(ImageError::YcbcrFormatInvalidDimensions);
                    }
                }
            }
        }

        /* Check usage requirements */

        let combined_usage = usage | stencil_usage;

        if combined_usage.intersects(ImageUsage::SAMPLED)
            && !format_features.intersects(FormatFeatures::SAMPLED_IMAGE)
        {
            return Err(ImageError::FormatUsageNotSupported { usage: "sampled" });
        }

        if combined_usage.intersects(ImageUsage::COLOR_ATTACHMENT)
            && !format_features.intersects(FormatFeatures::COLOR_ATTACHMENT)
        {
            return Err(ImageError::FormatUsageNotSupported {
                usage: "color_attachment",
            });
        }

        if combined_usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            && !format_features.intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
        {
            return Err(ImageError::FormatUsageNotSupported {
                usage: "depth_stencil_attachment",
            });
        }

        if combined_usage.intersects(ImageUsage::INPUT_ATTACHMENT)
            && !format_features.intersects(
                FormatFeatures::COLOR_ATTACHMENT | FormatFeatures::DEPTH_STENCIL_ATTACHMENT,
            )
        {
            return Err(ImageError::FormatUsageNotSupported {
                usage: "input_attachment",
            });
        }

        if combined_usage.intersects(
            ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                | ImageUsage::INPUT_ATTACHMENT
                | ImageUsage::TRANSIENT_ATTACHMENT,
        ) {
            // VUID-VkImageCreateInfo-usage-00964
            // VUID-VkImageCreateInfo-usage-00965
            // VUID-VkImageCreateInfo-Format-02536
            // VUID-VkImageCreateInfo-format-02537
            if extent[0] > device_properties.max_framebuffer_width
                || extent[1] > device_properties.max_framebuffer_height
            {
                return Err(ImageError::MaxFramebufferDimensionsExceeded {
                    extent: [extent[0], extent[1]],
                    max: [
                        device_properties.max_framebuffer_width,
                        device_properties.max_framebuffer_height,
                    ],
                });
            }
        }

        if combined_usage.intersects(ImageUsage::STORAGE) {
            if !format_features.intersects(FormatFeatures::STORAGE_IMAGE) {
                return Err(ImageError::FormatUsageNotSupported { usage: "storage" });
            }

            // VUID-VkImageCreateInfo-usage-00968
            // VUID-VkImageCreateInfo-format-02538
            if !device.enabled_features().shader_storage_image_multisample
                && samples != SampleCount::Sample1
            {
                return Err(ImageError::RequirementNotMet {
                    required_for: "`create_info.usage` or `create_info.stencil_usage` contains \
                        `ImageUsage::STORAGE`, and `create_info.samples` is not \
                        `SampleCount::Sample1`",
                    requires_one_of: RequiresOneOf {
                        features: &["shader_storage_image_multisample"],
                        ..Default::default()
                    },
                });
            }
        }

        // These flags only exist in later versions, ignore them otherwise
        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if combined_usage.intersects(ImageUsage::TRANSFER_SRC)
                && !format_features.intersects(FormatFeatures::TRANSFER_SRC)
            {
                return Err(ImageError::FormatUsageNotSupported {
                    usage: "transfer_src",
                });
            }

            if combined_usage.intersects(ImageUsage::TRANSFER_DST)
                && !format_features.intersects(FormatFeatures::TRANSFER_DST)
            {
                return Err(ImageError::FormatUsageNotSupported {
                    usage: "transfer_dst",
                });
            }
        }

        if usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT) {
            // VUID-VkImageCreateInfo-usage-00966
            assert!(usage.intersects(
                ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT
            ));

            // VUID-VkImageCreateInfo-usage-00963
            assert!((usage
                - (ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT))
                .is_empty())
        }

        if has_separate_stencil_usage {
            // VUID-VkImageCreateInfo-format-02795
            // VUID-VkImageCreateInfo-format-02796
            if usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                != stencil_usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(ImageError::StencilUsageMismatch {
                    usage,
                    stencil_usage,
                });
            }

            // VUID-VkImageCreateInfo-format-02797
            // VUID-VkImageCreateInfo-format-02798
            if usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
                != stencil_usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
            {
                return Err(ImageError::StencilUsageMismatch {
                    usage,
                    stencil_usage,
                });
            }

            if stencil_usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT) {
                // VUID-VkImageStencilUsageCreateInfo-stencilUsage-02539
                assert!((stencil_usage
                    - (ImageUsage::TRANSIENT_ATTACHMENT
                        | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT))
                    .is_empty())
            }
        }

        /* Check flags requirements */

        if flags.intersects(ImageCreateFlags::CUBE_COMPATIBLE) {
            // VUID-VkImageCreateInfo-flags-00949
            if image_type != ImageType::Dim2d {
                return Err(ImageError::CubeCompatibleNot2d);
            }

            // VUID-VkImageCreateInfo-imageType-00954
            if extent[0] != extent[1] {
                return Err(ImageError::CubeCompatibleNotSquare);
            }

            // VUID-VkImageCreateInfo-imageType-00954
            if array_layers < 6 {
                return Err(ImageError::CubeCompatibleNotEnoughArrayLayers);
            }
        }

        if flags.intersects(ImageCreateFlags::ARRAY_2D_COMPATIBLE) {
            // VUID-VkImageCreateInfo-flags-00950
            if image_type != ImageType::Dim3d {
                return Err(ImageError::Array2dCompatibleNot3d);
            }

            // VUID-VkImageCreateInfo-imageView2DOn3DImage-04459
            if device.enabled_extensions().khr_portability_subset
                && !device.enabled_features().image_view2_d_on3_d_image
            {
                return Err(ImageError::RequirementNotMet {
                    required_for: "this device is a portability subset device, and \
                        `create_info.flags` contains `ImageCreateFlags::ARRAY_2D_COMPATIBLE`",
                    requires_one_of: RequiresOneOf {
                        features: &["image_view2_d_on3_d_image"],
                        ..Default::default()
                    },
                });
            }
        }

        if flags.intersects(ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE) {
            // VUID-VkImageCreateInfo-flags-01572
            if format.compression().is_none() {
                return Err(ImageError::BlockTexelViewCompatibleNotCompressed);
            }
        }

        if flags.intersects(ImageCreateFlags::DISJOINT) {
            // VUID-VkImageCreateInfo-format-01577
            if format.planes().len() < 2 {
                return Err(ImageError::DisjointFormatNotSupported);
            }

            // VUID-VkImageCreateInfo-imageCreateFormatFeatures-02260
            if !format_features.intersects(FormatFeatures::DISJOINT) {
                return Err(ImageError::DisjointFormatNotSupported);
            }
        }

        /* Check sharing mode and queue families */

        match sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                // VUID-VkImageCreateInfo-sharingMode-00942
                assert!(queue_family_indices.len() >= 2);

                for &queue_family_index in queue_family_indices {
                    // VUID-VkImageCreateInfo-sharingMode-01420
                    if queue_family_index
                        >= device.physical_device().queue_family_properties().len() as u32
                    {
                        return Err(ImageError::SharingQueueFamilyIndexOutOfRange {
                            queue_family_index,
                            queue_family_count: device
                                .physical_device()
                                .queue_family_properties()
                                .len() as u32,
                        });
                    }
                }
            }
        }

        /* External memory handles */

        if !external_memory_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_memory)
            {
                return Err(ImageError::RequirementNotMet {
                    required_for: "`create_info.external_memory_handle_types` is not empty",
                    requires_one_of: RequiresOneOf {
                        api_version: Some(Version::V1_1),
                        device_extensions: &["khr_external_memory"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkExternalMemoryImageCreateInfo-handleTypes-parameter
            external_memory_handle_types.validate_device(device)?;

            // VUID-VkImageCreateInfo-pNext-01443
            if initial_layout != ImageLayout::Undefined {
                return Err(ImageError::ExternalMemoryInvalidInitialLayout);
            }
        }

        /*
            Some device limits can be exceeded, but only for particular image configurations, which
            must be queried with `image_format_properties`. See:
            https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap44.html#capabilities-image
            First, we check if this is the case, then query the device if so.
        */

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap44.html#features-extentperimagetype
        let extent_must_query = || match image_type {
            ImageType::Dim1d => {
                let limit = device.physical_device().properties().max_image_dimension1_d;
                extent[0] > limit
            }
            ImageType::Dim2d if flags.intersects(ImageCreateFlags::CUBE_COMPATIBLE) => {
                let limit = device
                    .physical_device()
                    .properties()
                    .max_image_dimension_cube;
                extent[0] > limit
            }
            ImageType::Dim2d => {
                let limit = device.physical_device().properties().max_image_dimension2_d;
                extent[0] > limit || extent[1] > limit
            }
            ImageType::Dim3d => {
                let limit = device.physical_device().properties().max_image_dimension3_d;
                extent[0] > limit || extent[1] > limit || extent[2] > limit
            }
        };
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageFormatProperties.html
        let mip_levels_must_query = || {
            if mip_levels > 1 {
                // TODO: for external memory, the spec says:
                // "handle type included in the handleTypes member for which mipmap image support is
                // not required". But which handle types are those?
                !external_memory_handle_types.is_empty()
            } else {
                false
            }
        };
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageFormatProperties.html
        let array_layers_must_query = || {
            if array_layers > device.physical_device().properties().max_image_array_layers {
                true
            } else if array_layers > 1 {
                image_type == ImageType::Dim3d
            } else {
                false
            }
        };
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap44.html#features-supported-sample-counts
        let samples_must_query = || {
            if samples == SampleCount::Sample1 {
                return false;
            }

            if combined_usage.intersects(ImageUsage::COLOR_ATTACHMENT)
                && !device_properties
                    .framebuffer_color_sample_counts
                    .contains_enum(samples)
            {
                // TODO: how to handle framebuffer_integer_color_sample_counts limit, which only
                // exists >= Vulkan 1.2
                return true;
            }

            if combined_usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT) {
                if aspects.intersects(ImageAspects::DEPTH)
                    && !device_properties
                        .framebuffer_depth_sample_counts
                        .contains_enum(samples)
                {
                    return true;
                }

                if aspects.intersects(ImageAspects::STENCIL)
                    && !device_properties
                        .framebuffer_stencil_sample_counts
                        .contains_enum(samples)
                {
                    return true;
                }
            }

            if combined_usage.intersects(ImageUsage::SAMPLED) {
                if let Some(numeric_type) = format.type_color() {
                    match numeric_type {
                        NumericType::UINT | NumericType::SINT => {
                            if !device_properties
                                .sampled_image_integer_sample_counts
                                .contains_enum(samples)
                            {
                                return true;
                            }
                        }
                        NumericType::SFLOAT
                        | NumericType::UFLOAT
                        | NumericType::SNORM
                        | NumericType::UNORM
                        | NumericType::SSCALED
                        | NumericType::USCALED
                        | NumericType::SRGB => {
                            if !device_properties
                                .sampled_image_color_sample_counts
                                .contains_enum(samples)
                            {
                                return true;
                            }
                        }
                    }
                } else {
                    if aspects.intersects(ImageAspects::DEPTH)
                        && !device_properties
                            .sampled_image_depth_sample_counts
                            .contains_enum(samples)
                    {
                        return true;
                    }

                    if aspects.intersects(ImageAspects::STENCIL)
                        && device_properties
                            .sampled_image_stencil_sample_counts
                            .contains_enum(samples)
                    {
                        return true;
                    }
                }
            }

            if combined_usage.intersects(ImageUsage::STORAGE)
                && !device_properties
                    .storage_image_sample_counts
                    .contains_enum(samples)
            {
                return true;
            }

            false
        };
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html#_description
        let linear_must_query = || {
            if tiling == ImageTiling::Linear {
                !(image_type == ImageType::Dim2d
                    && format.type_color().is_some()
                    && mip_levels == 1
                    && array_layers == 1
                    // VUID-VkImageCreateInfo-samples-02257 already states that multisampling+linear
                    // is invalid so no need to check for that here.
                    && (usage - (ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST)).is_empty())
            } else {
                false
            }
        };

        let must_query_device = extent_must_query()
            || mip_levels_must_query()
            || array_layers_must_query()
            || samples_must_query()
            || linear_must_query();

        // We determined that we must query the device in order to be sure that the image
        // configuration is supported.
        if must_query_device {
            let external_memory_handle_types: SmallVec<[Option<ExternalMemoryHandleType>; 4]> =
                if !external_memory_handle_types.is_empty() {
                    // If external memory handles are used, the properties need to be queried
                    // individually for each handle type.
                    external_memory_handle_types.into_iter().map(Some).collect()
                } else {
                    smallvec![None]
                };

            for external_memory_handle_type in external_memory_handle_types {
                // Use unchecked, because all validation has been done above.
                let image_format_properties = unsafe {
                    device
                        .physical_device()
                        .image_format_properties_unchecked(ImageFormatInfo {
                            flags,
                            format: Some(format),
                            image_type,
                            tiling,
                            usage,
                            external_memory_handle_type,
                            ..Default::default()
                        })?
                };

                let ImageFormatProperties {
                    max_extent,
                    max_mip_levels,
                    max_array_layers,
                    sample_counts,
                    max_resource_size: _,
                    ..
                } = match image_format_properties {
                    Some(x) => x,
                    None => return Err(ImageError::ImageFormatPropertiesNotSupported),
                };

                // VUID-VkImageCreateInfo-extent-02252
                // VUID-VkImageCreateInfo-extent-02253
                // VUID-VkImageCreateInfo-extent-02254
                if extent[0] > max_extent[0]
                    || extent[1] > max_extent[1]
                    || extent[2] > max_extent[2]
                {
                    return Err(ImageError::MaxDimensionsExceeded {
                        extent,
                        max: max_extent,
                    });
                }

                // VUID-VkImageCreateInfo-mipLevels-02255
                if mip_levels > max_mip_levels {
                    return Err(ImageError::MaxMipLevelsExceeded {
                        mip_levels,
                        max: max_mip_levels,
                    });
                }

                // VUID-VkImageCreateInfo-arrayLayers-02256
                if array_layers > max_array_layers {
                    return Err(ImageError::MaxArrayLayersExceeded {
                        array_layers,
                        max: max_array_layers,
                    });
                }

                // VUID-VkImageCreateInfo-samples-02258
                if !sample_counts.contains_enum(samples) {
                    return Err(ImageError::SampleCountNotSupported {
                        samples,
                        supported: sample_counts,
                    });
                }

                // TODO: check resource size?
            }
        }

        Ok(format_features)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: ImageCreateInfo,
    ) -> Result<Self, VulkanError> {
        let &ImageCreateInfo {
            flags,
            dimensions,
            format,
            mip_levels,
            samples,
            tiling,
            usage,
            mut stencil_usage,
            ref sharing,
            initial_layout,
            external_memory_handle_types,
            _ne: _,
        } = &create_info;

        let aspects = format.map_or_else(Default::default, |format| format.aspects());

        let has_separate_stencil_usage = if stencil_usage.is_empty()
            || !aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
        {
            stencil_usage = usage;
            false
        } else {
            stencil_usage == usage
        };

        let (image_type, extent, array_layers) = match dimensions {
            ImageDimensions::Dim1d {
                width,
                array_layers,
            } => (ImageType::Dim1d, [width, 1, 1], array_layers),
            ImageDimensions::Dim2d {
                width,
                height,
                array_layers,
            } => (ImageType::Dim2d, [width, height, 1], array_layers),
            ImageDimensions::Dim3d {
                width,
                height,
                depth,
            } => (ImageType::Dim3d, [width, height, depth], 1),
        };

        let (sharing_mode, queue_family_index_count, p_queue_family_indices) = match sharing {
            Sharing::Exclusive => (ash::vk::SharingMode::EXCLUSIVE, 0, &[] as _),
            Sharing::Concurrent(queue_family_indices) => (
                ash::vk::SharingMode::CONCURRENT,
                queue_family_indices.len() as u32,
                queue_family_indices.as_ptr(),
            ),
        };

        let mut info_vk = ash::vk::ImageCreateInfo {
            flags: flags.into(),
            image_type: image_type.into(),
            format: format.map(Into::into).unwrap_or_default(),
            extent: ash::vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: extent[2],
            },
            mip_levels,
            array_layers,
            samples: samples.into(),
            tiling: tiling.into(),
            usage: usage.into(),
            sharing_mode,
            queue_family_index_count,
            p_queue_family_indices,
            initial_layout: initial_layout.into(),
            ..Default::default()
        };
        let mut external_memory_info_vk = None;
        let mut stencil_usage_info_vk = None;

        if !external_memory_handle_types.is_empty() {
            let next = external_memory_info_vk.insert(ash::vk::ExternalMemoryImageCreateInfo {
                handle_types: external_memory_handle_types.into(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *const _ as *const _;
        }

        if has_separate_stencil_usage {
            let next = stencil_usage_info_vk.insert(ash::vk::ImageStencilUsageCreateInfo {
                stencil_usage: stencil_usage.into(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *const _ as *const _;
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_image)(device.handle(), &info_vk, ptr::null(), output.as_mut_ptr())
                .result()
                .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(device, handle, create_info))
    }

    /// Creates a new `RawImage` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `handle` must refer to an image that has not yet had memory bound to it.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Image,
        create_info: ImageCreateInfo,
    ) -> Self {
        Self::from_handle_with_destruction(device, handle, create_info, true)
    }

    unsafe fn from_handle_with_destruction(
        device: Arc<Device>,
        handle: ash::vk::Image,
        create_info: ImageCreateInfo,
        needs_destruction: bool,
    ) -> Self {
        let ImageCreateInfo {
            flags,
            dimensions,
            format,
            mip_levels,
            samples,
            tiling,
            usage,
            mut stencil_usage,
            sharing,
            initial_layout,
            external_memory_handle_types,
            _ne: _,
        } = create_info;

        let aspects = format.map_or_else(Default::default, |format| format.aspects());

        if stencil_usage.is_empty()
            || !aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
        {
            stencil_usage = usage;
        }

        // Get format features
        let format_features = {
            // Use unchecked, because `create_info` is assumed to match the info of the handle, and
            // therefore already valid.
            let format_properties = device
                .physical_device()
                .format_properties_unchecked(format.unwrap());
            match tiling {
                ImageTiling::Linear => format_properties.linear_tiling_features,
                ImageTiling::Optimal => format_properties.optimal_tiling_features,
            }
        };

        let memory_requirements = if flags.intersects(ImageCreateFlags::DISJOINT) {
            (0..format.unwrap().planes().len())
                .map(|plane| Self::get_memory_requirements(&device, handle, Some(plane)))
                .collect()
        } else {
            smallvec![Self::get_memory_requirements(&device, handle, None)]
        };

        RawImage {
            handle,
            device,
            id: Self::next_id(),
            flags,
            dimensions,
            format,
            format_features,
            mip_levels,
            initial_layout,
            samples,
            tiling,
            usage,
            stencil_usage,
            sharing,
            external_memory_handle_types,
            memory_requirements,
            needs_destruction,
            subresource_layout: OnceCache::new(),
        }
    }

    fn get_memory_requirements(
        device: &Device,
        handle: ash::vk::Image,
        plane: Option<usize>,
    ) -> MemoryRequirements {
        let mut info_vk = ash::vk::ImageMemoryRequirementsInfo2 {
            image: handle,
            ..Default::default()
        };
        let mut plane_info_vk = None;

        if let Some(plane) = plane {
            debug_assert!(
                device.api_version() >= Version::V1_1
                    || device.enabled_extensions().khr_get_memory_requirements2
                        && device.enabled_extensions().khr_sampler_ycbcr_conversion
            );

            let next = plane_info_vk.insert(ash::vk::ImagePlaneMemoryRequirementsInfo {
                plane_aspect: match plane {
                    0 => ash::vk::ImageAspectFlags::PLANE_0,
                    1 => ash::vk::ImageAspectFlags::PLANE_1,
                    2 => ash::vk::ImageAspectFlags::PLANE_2,
                    _ => unreachable!(),
                },
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *mut _ as *mut _;
        }

        let mut memory_requirements2_vk = ash::vk::MemoryRequirements2::default();
        let mut memory_dedicated_requirements_vk = None;

        if device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation
        {
            debug_assert!(
                device.api_version() >= Version::V1_1
                    || device.enabled_extensions().khr_get_memory_requirements2
            );

            let next = memory_dedicated_requirements_vk
                .insert(ash::vk::MemoryDedicatedRequirements::default());

            next.p_next = memory_requirements2_vk.p_next;
            memory_requirements2_vk.p_next = next as *mut _ as *mut _;
        }

        unsafe {
            let fns = device.fns();

            if device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_get_memory_requirements2
            {
                if device.api_version() >= Version::V1_1 {
                    (fns.v1_1.get_image_memory_requirements2)(
                        device.handle(),
                        &info_vk,
                        &mut memory_requirements2_vk,
                    );
                } else {
                    (fns.khr_get_memory_requirements2
                        .get_image_memory_requirements2_khr)(
                        device.handle(),
                        &info_vk,
                        &mut memory_requirements2_vk,
                    );
                }
            } else {
                (fns.v1_0.get_image_memory_requirements)(
                    device.handle(),
                    handle,
                    &mut memory_requirements2_vk.memory_requirements,
                );
            }
        }

        MemoryRequirements {
            size: memory_requirements2_vk.memory_requirements.size,
            alignment: memory_requirements2_vk.memory_requirements.alignment,
            memory_type_bits: memory_requirements2_vk.memory_requirements.memory_type_bits,
            prefers_dedicated_allocation: memory_dedicated_requirements_vk
                .map_or(false, |dreqs| dreqs.prefers_dedicated_allocation != 0),
            requires_dedicated_allocation: memory_dedicated_requirements_vk
                .map_or(false, |dreqs| dreqs.requires_dedicated_allocation != 0),
        }
    }

    #[inline]
    #[allow(dead_code)] // Remove when sparse memory is implemented
    fn get_sparse_memory_requirements(&self) -> Vec<SparseImageMemoryRequirements> {
        let device = &self.device;

        unsafe {
            let fns = self.device.fns();

            if device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_get_memory_requirements2
            {
                let info2 = ash::vk::ImageSparseMemoryRequirementsInfo2 {
                    image: self.handle,
                    ..Default::default()
                };

                let mut count = 0;

                if device.api_version() >= Version::V1_1 {
                    (fns.v1_1.get_image_sparse_memory_requirements2)(
                        device.handle(),
                        &info2,
                        &mut count,
                        ptr::null_mut(),
                    );
                } else {
                    (fns.khr_get_memory_requirements2
                        .get_image_sparse_memory_requirements2_khr)(
                        device.handle(),
                        &info2,
                        &mut count,
                        ptr::null_mut(),
                    );
                }

                let mut sparse_image_memory_requirements2 =
                    vec![ash::vk::SparseImageMemoryRequirements2::default(); count as usize];

                if device.api_version() >= Version::V1_1 {
                    (fns.v1_1.get_image_sparse_memory_requirements2)(
                        self.device.handle(),
                        &info2,
                        &mut count,
                        sparse_image_memory_requirements2.as_mut_ptr(),
                    );
                } else {
                    (fns.khr_get_memory_requirements2
                        .get_image_sparse_memory_requirements2_khr)(
                        self.device.handle(),
                        &info2,
                        &mut count,
                        sparse_image_memory_requirements2.as_mut_ptr(),
                    );
                }

                sparse_image_memory_requirements2.set_len(count as usize);

                sparse_image_memory_requirements2
                    .into_iter()
                    .map(
                        |sparse_image_memory_requirements2| SparseImageMemoryRequirements {
                            format_properties: SparseImageFormatProperties {
                                aspects: sparse_image_memory_requirements2
                                    .memory_requirements
                                    .format_properties
                                    .aspect_mask
                                    .into(),
                                image_granularity: [
                                    sparse_image_memory_requirements2
                                        .memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .width,
                                    sparse_image_memory_requirements2
                                        .memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .height,
                                    sparse_image_memory_requirements2
                                        .memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .depth,
                                ],
                                flags: sparse_image_memory_requirements2
                                    .memory_requirements
                                    .format_properties
                                    .flags
                                    .into(),
                            },
                            image_mip_tail_first_lod: sparse_image_memory_requirements2
                                .memory_requirements
                                .image_mip_tail_first_lod,
                            image_mip_tail_size: sparse_image_memory_requirements2
                                .memory_requirements
                                .image_mip_tail_size,
                            image_mip_tail_offset: sparse_image_memory_requirements2
                                .memory_requirements
                                .image_mip_tail_offset,
                            image_mip_tail_stride: (!sparse_image_memory_requirements2
                                .memory_requirements
                                .format_properties
                                .flags
                                .intersects(ash::vk::SparseImageFormatFlags::SINGLE_MIPTAIL))
                            .then_some(
                                sparse_image_memory_requirements2
                                    .memory_requirements
                                    .image_mip_tail_stride,
                            ),
                        },
                    )
                    .collect()
            } else {
                let mut count = 0;

                (fns.v1_0.get_image_sparse_memory_requirements)(
                    device.handle(),
                    self.handle,
                    &mut count,
                    ptr::null_mut(),
                );

                let mut sparse_image_memory_requirements =
                    vec![ash::vk::SparseImageMemoryRequirements::default(); count as usize];

                (fns.v1_0.get_image_sparse_memory_requirements)(
                    device.handle(),
                    self.handle,
                    &mut count,
                    sparse_image_memory_requirements.as_mut_ptr(),
                );

                sparse_image_memory_requirements.set_len(count as usize);

                sparse_image_memory_requirements
                    .into_iter()
                    .map(
                        |sparse_image_memory_requirements| SparseImageMemoryRequirements {
                            format_properties: SparseImageFormatProperties {
                                aspects: sparse_image_memory_requirements
                                    .format_properties
                                    .aspect_mask
                                    .into(),
                                image_granularity: [
                                    sparse_image_memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .width,
                                    sparse_image_memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .height,
                                    sparse_image_memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .depth,
                                ],
                                flags: sparse_image_memory_requirements
                                    .format_properties
                                    .flags
                                    .into(),
                            },
                            image_mip_tail_first_lod: sparse_image_memory_requirements
                                .image_mip_tail_first_lod,
                            image_mip_tail_size: sparse_image_memory_requirements
                                .image_mip_tail_size,
                            image_mip_tail_offset: sparse_image_memory_requirements
                                .image_mip_tail_offset,
                            image_mip_tail_stride: (!sparse_image_memory_requirements
                                .format_properties
                                .flags
                                .intersects(ash::vk::SparseImageFormatFlags::SINGLE_MIPTAIL))
                            .then_some(sparse_image_memory_requirements.image_mip_tail_stride),
                        },
                    )
                    .collect()
            }
        }
    }

    pub(crate) fn id(&self) -> NonZeroU64 {
        self.id
    }

    /// Binds device memory to this image.
    ///
    /// - If `self.flags().disjoint` is not set, then `allocations` must contain exactly one
    ///   element. This element may be a dedicated allocation.
    /// - If `self.flags().disjoint` is set, then `allocations` must contain exactly
    ///   `self.format().unwrap().planes().len()` elements. These elements must not be dedicated
    ///   allocations.
    pub fn bind_memory(
        self,
        allocations: impl IntoIterator<Item = MemoryAlloc>,
    ) -> Result<
        Image,
        (
            ImageError,
            RawImage,
            impl ExactSizeIterator<Item = MemoryAlloc>,
        ),
    > {
        let allocations: SmallVec<[_; 3]> = allocations.into_iter().collect();

        if let Err(err) = self.validate_bind_memory(&allocations) {
            return Err((err, self, allocations.into_iter()));
        }

        unsafe { self.bind_memory_unchecked(allocations) }.map_err(|(err, image, allocations)| {
            (
                err.into(),
                image,
                allocations
                    .into_iter()
                    .collect::<SmallVec<[_; 3]>>()
                    .into_iter(),
            )
        })
    }

    fn validate_bind_memory(&self, allocations: &[MemoryAlloc]) -> Result<(), ImageError> {
        if self.flags.intersects(ImageCreateFlags::DISJOINT) {
            if allocations.len() != self.format.unwrap().planes().len() {
                return Err(ImageError::AllocationsWrongNumberOfElements {
                    provided: allocations.len(),
                    required: self.format.unwrap().planes().len(),
                });
            }
        } else {
            if allocations.len() != 1 {
                return Err(ImageError::AllocationsWrongNumberOfElements {
                    provided: allocations.len(),
                    required: 1,
                });
            }
        }

        for (allocations_index, (allocation, memory_requirements)) in (allocations.iter())
            .zip(self.memory_requirements.iter())
            .enumerate()
        {
            let memory = allocation.device_memory();
            let memory_offset = allocation.offset();
            let memory_type = &self
                .device
                .physical_device()
                .memory_properties()
                .memory_types[memory.memory_type_index() as usize];

            // VUID-VkBindImageMemoryInfo-commonparent
            assert_eq!(self.device(), memory.device());

            // VUID-VkBindImageMemoryInfo-image-07460
            // Ensured by taking ownership of `RawImage`.

            // VUID-VkBindImageMemoryInfo-image-01045
            // Currently ensured by not having sparse binding flags, but this needs to be checked
            // once those are enabled.

            // VUID-VkBindImageMemoryInfo-memoryOffset-01046
            // Assume that `allocation` was created correctly.

            if let Some(dedicated_to) = memory.dedicated_to() {
                // VUID-VkBindImageMemoryInfo-memory-02628
                match dedicated_to {
                    DedicatedTo::Image(id) if id == self.id => {}
                    _ => return Err(ImageError::DedicatedAllocationMismatch),
                }
                debug_assert!(memory_offset == 0); // This should be ensured by the allocator
            } else {
                // VUID-VkBindImageMemoryInfo-image-01445
                if memory_requirements.requires_dedicated_allocation {
                    return Err(ImageError::DedicatedAllocationRequired);
                }
            }

            // VUID-VkBindImageMemoryInfo-None-01901
            if memory_type
                .property_flags
                .intersects(MemoryPropertyFlags::PROTECTED)
            {
                return Err(ImageError::MemoryProtectedMismatch {
                    allocations_index,
                    image_protected: false,
                    memory_protected: true,
                });
            }

            // VUID-VkBindImageMemoryInfo-memory-02728
            if !memory.export_handle_types().is_empty()
                && !memory
                    .export_handle_types()
                    .intersects(self.external_memory_handle_types)
            {
                return Err(ImageError::MemoryExternalHandleTypesDisjoint {
                    allocations_index,
                    image_handle_types: self.external_memory_handle_types,
                    memory_export_handle_types: memory.export_handle_types(),
                });
            }

            if let Some(handle_type) = memory.imported_handle_type() {
                // VUID-VkBindImageMemoryInfo-memory-02989
                if !ExternalMemoryHandleTypes::from(handle_type)
                    .intersects(self.external_memory_handle_types)
                {
                    return Err(ImageError::MemoryImportedHandleTypeNotEnabled {
                        allocations_index,
                        image_handle_types: self.external_memory_handle_types,
                        memory_imported_handle_type: handle_type,
                    });
                }
            }

            // VUID-VkBindImageMemoryInfo-pNext-01615
            // VUID-VkBindImageMemoryInfo-pNext-01619
            if memory_requirements.memory_type_bits & (1 << memory.memory_type_index()) == 0 {
                return Err(ImageError::MemoryTypeNotAllowed {
                    allocations_index,
                    provided_memory_type_index: memory.memory_type_index(),
                    allowed_memory_type_bits: memory_requirements.memory_type_bits,
                });
            }

            // VUID-VkBindImageMemoryInfo-pNext-01616
            // VUID-VkBindImageMemoryInfo-pNext-01620
            if memory_offset % memory_requirements.alignment != 0 {
                return Err(ImageError::MemoryAllocationNotAligned {
                    allocations_index,
                    allocation_offset: memory_offset,
                    required_alignment: memory_requirements.alignment,
                });
            }

            // VUID-VkBindImageMemoryInfo-pNext-01617
            // VUID-VkBindImageMemoryInfo-pNext-01621
            if allocation.size() < memory_requirements.size {
                return Err(ImageError::MemoryAllocationTooSmall {
                    allocations_index,
                    allocation_size: allocation.size(),
                    required_size: memory_requirements.size,
                });
            }
        }

        Ok(())
    }

    /// # Safety
    ///
    /// - If `self.flags().disjoint` is not set, then `allocations` must contain exactly one
    ///   element.
    /// - If `self.flags().disjoint` is set, then `allocations` must contain exactly
    ///   `self.format().unwrap().planes().len()` elements.
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_memory_unchecked(
        self,
        allocations: impl IntoIterator<Item = MemoryAlloc>,
    ) -> Result<
        Image,
        (
            VulkanError,
            RawImage,
            impl ExactSizeIterator<Item = MemoryAlloc>,
        ),
    > {
        let allocations: SmallVec<[_; 3]> = allocations.into_iter().collect();
        let fns = self.device.fns();

        let result = if self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_bind_memory2
        {
            let mut infos_vk: SmallVec<[_; 3]> = SmallVec::with_capacity(3);
            let mut plane_infos_vk: SmallVec<[_; 3]> = SmallVec::with_capacity(3);

            if self.flags.intersects(ImageCreateFlags::DISJOINT) {
                debug_assert_eq!(allocations.len(), self.format.unwrap().planes().len());

                for (plane, allocation) in allocations.iter().enumerate() {
                    let memory = allocation.device_memory();
                    let memory_offset = allocation.offset();

                    infos_vk.push(ash::vk::BindImageMemoryInfo {
                        image: self.handle,
                        memory: memory.handle(),
                        memory_offset,
                        ..Default::default()
                    });
                    // VUID-VkBindImageMemoryInfo-pNext-01618
                    plane_infos_vk.push(ash::vk::BindImagePlaneMemoryInfo {
                        plane_aspect: match plane {
                            0 => ash::vk::ImageAspectFlags::PLANE_0,
                            1 => ash::vk::ImageAspectFlags::PLANE_1,
                            2 => ash::vk::ImageAspectFlags::PLANE_2,
                            _ => unreachable!(),
                        },
                        ..Default::default()
                    });
                }
            } else {
                debug_assert_eq!(allocations.len(), 1);

                let allocation = &allocations[0];
                let memory = allocation.device_memory();
                let memory_offset = allocation.offset();

                infos_vk.push(ash::vk::BindImageMemoryInfo {
                    image: self.handle,
                    memory: memory.handle(),
                    memory_offset,
                    ..Default::default()
                });
            };

            for (info_vk, plane_info_vk) in (infos_vk.iter_mut()).zip(plane_infos_vk.iter_mut()) {
                info_vk.p_next = plane_info_vk as *mut _ as *mut _;
            }

            if self.device.api_version() >= Version::V1_1 {
                (fns.v1_1.bind_image_memory2)(
                    self.device.handle(),
                    infos_vk.len() as u32,
                    infos_vk.as_ptr(),
                )
            } else {
                (fns.khr_bind_memory2.bind_image_memory2_khr)(
                    self.device.handle(),
                    infos_vk.len() as u32,
                    infos_vk.as_ptr(),
                )
            }
        } else {
            debug_assert_eq!(allocations.len(), 1);

            let allocation = &allocations[0];
            let memory = allocation.device_memory();
            let memory_offset = allocation.offset();

            (fns.v1_0.bind_image_memory)(
                self.device.handle(),
                self.handle,
                memory.handle(),
                memory_offset,
            )
        }
        .result();

        if let Err(err) = result {
            return Err((VulkanError::from(err), self, allocations.into_iter()));
        }

        Ok(Image::from_raw(self, ImageMemory::Normal(allocations)))
    }

    /// Returns the memory requirements for this image.
    ///
    /// - If `self.flags().disjoint` is not set, this returns a slice with a length of 1.
    /// - If `self.flags().disjoint` is set, this returns a slice with a length equal to
    ///   `self.format().unwrap().planes().len()`.
    #[inline]
    pub fn memory_requirements(&self) -> &[MemoryRequirements] {
        &self.memory_requirements
    }

    /// Returns the flags the image was created with.
    #[inline]
    pub fn flags(&self) -> ImageCreateFlags {
        self.flags
    }

    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> ImageDimensions {
        self.dimensions
    }

    /// Returns the image's format.
    #[inline]
    pub fn format(&self) -> Option<Format> {
        self.format
    }

    /// Returns the features supported by the image's format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.format_features
    }

    /// Returns the number of mipmap levels in the image.
    #[inline]
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Returns the initial layout of the image.
    #[inline]
    pub fn initial_layout(&self) -> ImageLayout {
        self.initial_layout
    }

    /// Returns the number of samples for the image.
    #[inline]
    pub fn samples(&self) -> SampleCount {
        self.samples
    }

    /// Returns the tiling of the image.
    #[inline]
    pub fn tiling(&self) -> ImageTiling {
        self.tiling
    }

    /// Returns the usage the image was created with.
    #[inline]
    pub fn usage(&self) -> ImageUsage {
        self.usage
    }

    /// Returns the stencil usage the image was created with.
    #[inline]
    pub fn stencil_usage(&self) -> ImageUsage {
        self.stencil_usage
    }

    /// Returns the sharing the image was created with.
    #[inline]
    pub fn sharing(&self) -> &Sharing<SmallVec<[u32; 4]>> {
        &self.sharing
    }

    /// Returns the external memory handle types that are supported with this image.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.external_memory_handle_types
    }

    /// Returns an `ImageSubresourceLayers` covering the first mip level of the image. All aspects
    /// of the image are selected, or `plane0` if the image is multi-planar.
    #[inline]
    pub fn subresource_layers(&self) -> ImageSubresourceLayers {
        ImageSubresourceLayers {
            aspects: {
                let aspects = self.format.unwrap().aspects();

                if aspects.intersects(ImageAspects::PLANE_0) {
                    ImageAspects::PLANE_0
                } else {
                    aspects
                }
            },
            mip_level: 0,
            array_layers: 0..self.dimensions.array_layers(),
        }
    }

    /// Returns an `ImageSubresourceRange` covering the whole image. If the image is multi-planar,
    /// only the `color` aspect is selected.
    #[inline]
    pub fn subresource_range(&self) -> ImageSubresourceRange {
        ImageSubresourceRange {
            aspects: self.format.unwrap().aspects()
                - (ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2),
            mip_levels: 0..self.mip_levels,
            array_layers: 0..self.dimensions.array_layers(),
        }
    }

    /// Queries the memory layout of a single subresource of the image.
    ///
    /// Only images with linear tiling are supported, if they do not have a format with both a
    /// depth and a stencil format. Images with optimal tiling have an opaque image layout that is
    /// not suitable for direct memory accesses, and likewise for combined depth/stencil formats.
    /// Multi-planar formats are supported, but you must specify one of the planes as the `aspect`,
    /// not [`ImageAspect::Color`].
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    pub fn subresource_layout(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<SubresourceLayout, ImageError> {
        self.validate_subresource_layout(aspect, mip_level, array_layer)?;

        unsafe { Ok(self.subresource_layout_unchecked(aspect, mip_level, array_layer)) }
    }

    fn validate_subresource_layout(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<(), ImageError> {
        // VUID-VkImageSubresource-aspectMask-parameter
        aspect.validate_device(&self.device)?;

        // VUID-VkImageSubresource-aspectMask-requiredbitmask
        // VUID-vkGetImageSubresourceLayout-aspectMask-00997
        // Ensured by use of enum `ImageAspect`.

        // VUID-vkGetImageSubresourceLayout-image-02270
        if !matches!(self.tiling, ImageTiling::Linear) {
            return Err(ImageError::OptimalTilingNotSupported);
        }

        // VUID-vkGetImageSubresourceLayout-mipLevel-01716
        if mip_level >= self.mip_levels {
            return Err(ImageError::MipLevelOutOfRange {
                provided_mip_level: mip_level,
                image_mip_levels: self.mip_levels,
            });
        }

        // VUID-vkGetImageSubresourceLayout-arrayLayer-01717
        if array_layer >= self.dimensions.array_layers() {
            return Err(ImageError::ArrayLayerOutOfRange {
                provided_array_layer: array_layer,
                image_array_layers: self.dimensions.array_layers(),
            });
        }

        let mut allowed_aspects = self.format.unwrap().aspects();

        // Follows from the combination of these three VUIDs. See:
        // https://github.com/KhronosGroup/Vulkan-Docs/issues/1942
        // VUID-vkGetImageSubresourceLayout-aspectMask-00997
        // VUID-vkGetImageSubresourceLayout-format-04462
        // VUID-vkGetImageSubresourceLayout-format-04463
        if allowed_aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            return Err(ImageError::DepthStencilFormatsNotSupported);
        }

        if allowed_aspects
            .intersects(ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2)
        {
            allowed_aspects -= ImageAspects::COLOR;
        }

        // VUID-vkGetImageSubresourceLayout-format-04461
        // VUID-vkGetImageSubresourceLayout-format-04462
        // VUID-vkGetImageSubresourceLayout-format-04463
        // VUID-vkGetImageSubresourceLayout-format-04464
        // VUID-vkGetImageSubresourceLayout-format-01581
        // VUID-vkGetImageSubresourceLayout-format-01582
        if !allowed_aspects.contains(aspect.into()) {
            return Err(ImageError::AspectNotAllowed {
                provided_aspect: aspect,
                allowed_aspects,
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn subresource_layout_unchecked(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> SubresourceLayout {
        self.subresource_layout.get_or_insert(
            (aspect, mip_level, array_layer),
            |&(aspect, mip_level, array_layer)| {
                let fns = self.device.fns();

                let subresource = ash::vk::ImageSubresource {
                    aspect_mask: aspect.into(),
                    mip_level,
                    array_layer,
                };

                let mut output = MaybeUninit::uninit();
                (fns.v1_0.get_image_subresource_layout)(
                    self.device.handle(),
                    self.handle,
                    &subresource,
                    output.as_mut_ptr(),
                );
                let output = output.assume_init();

                SubresourceLayout {
                    offset: output.offset,
                    size: output.size,
                    row_pitch: output.row_pitch,
                    array_pitch: (self.dimensions.array_layers() > 1).then_some(output.array_pitch),
                    depth_pitch: matches!(self.dimensions, ImageDimensions::Dim3d { .. })
                        .then_some(output.depth_pitch),
                }
            },
        )
    }
}

impl Drop for RawImage {
    #[inline]
    fn drop(&mut self) {
        if !self.needs_destruction {
            return;
        }

        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_image)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for RawImage {
    type Handle = ash::vk::Image;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for RawImage {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

crate::impl_id_counter!(RawImage);

/// Parameters to create a new `Image`.
#[derive(Clone, Debug)]
pub struct ImageCreateInfo {
    /// Flags to enable.
    ///
    /// The default value is [`ImageCreateFlags::empty()`].
    pub flags: ImageCreateFlags,

    /// The type, extent and number of array layers to create the image with.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if `samples` is not [`SampleCount::Sample1`] and `dimensions.array_layers()` is
    /// not 1, the [`multisample_array_image`](crate::device::Features::multisample_array_image)
    /// feature must be enabled on the device.
    ///
    /// The default value is `ImageDimensions::Dim2d { width: 0, height: 0, array_layers: 1 }`,
    /// which must be overridden.
    pub dimensions: ImageDimensions,

    /// The format used to store the image data.
    ///
    /// The default value is `None`, which must be overridden.
    pub format: Option<Format>,

    /// The number of mip levels to create the image with.
    ///
    /// The default value is `1`.
    pub mip_levels: u32,

    /// The number of samples per texel that the image should use.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if `samples` is not [`SampleCount::Sample1`] and `dimensions.array_layers()` is
    /// not 1, the [`multisample_array_image`](crate::device::Features::multisample_array_image)
    /// feature must be enabled on the device.
    ///
    /// The default value is [`SampleCount::Sample1`].
    pub samples: SampleCount,

    /// The memory arrangement of the texel blocks.
    ///
    /// The default value is [`ImageTiling::Optimal`].
    pub tiling: ImageTiling,

    /// How the image is going to be used.
    ///
    /// The default value is [`ImageUsage::empty()`], which must be overridden.
    pub usage: ImageUsage,

    /// How the stencil aspect of the image is going to be used, if any.
    ///
    /// If `stencil_usage` is empty or if `format` does not have both a depth and a stencil aspect,
    /// then it is automatically set to equal `usage`.
    ///
    /// If after this, `stencil_usage` does not equal `usage`,
    /// then the device API version must be at least 1.2, or the
    /// [`ext_separate_stencil_usage`](crate::device::DeviceExtensions::ext_separate_stencil_usage)
    /// extension must be enabled on the device.
    ///
    /// The default value is [`ImageUsage::empty()`].
    pub stencil_usage: ImageUsage,

    /// Whether the image can be shared across multiple queues, or is limited to a single queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub sharing: Sharing<SmallVec<[u32; 4]>>,

    /// The image layout that the image will have when it is created.
    ///
    /// The default value is [`ImageLayout::Undefined`].
    pub initial_layout: ImageLayout,

    /// The external memory handle types that are going to be used with the image.
    ///
    /// If any of the fields in this value are set, the device must either support API version 1.1
    /// or the [`khr_external_memory`](crate::device::DeviceExtensions::khr_external_memory)
    /// extension must be enabled, and `initial_layout` must be set to
    /// [`ImageLayout::Undefined`].
    ///
    /// The default value is [`ExternalMemoryHandleTypes::empty()`].
    pub external_memory_handle_types: ExternalMemoryHandleTypes,

    pub _ne: crate::NonExhaustive,
}

impl Default for ImageCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: ImageCreateFlags::empty(),
            dimensions: ImageDimensions::Dim2d {
                width: 0,
                height: 0,
                array_layers: 1,
            },
            format: None,
            mip_levels: 1,
            samples: SampleCount::Sample1,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::empty(),
            stencil_usage: ImageUsage::empty(),
            sharing: Sharing::Exclusive,
            initial_layout: ImageLayout::Undefined,
            external_memory_handle_types: ExternalMemoryHandleTypes::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A multi-dimensioned storage for texel data.
///
/// Unlike [`RawImage`], an `Image` has memory backing it, and can be used normally.
#[derive(Debug)]
pub struct Image {
    inner: RawImage,
    memory: ImageMemory,

    aspect_list: SmallVec<[ImageAspect; 4]>,
    aspect_size: DeviceSize,
    mip_level_size: DeviceSize,
    range_size: DeviceSize,
    state: Mutex<ImageState>,
}

/// The type of backing memory that an image can have.
#[derive(Debug)]
pub enum ImageMemory {
    /// The image is backed by normal memory, bound with [`bind_memory`].
    ///
    /// [`bind_memory`]: RawImage::bind_memory
    Normal(SmallVec<[MemoryAlloc; 3]>),

    /// The image is backed by sparse memory, bound with [`bind_sparse`].
    ///
    /// [`bind_sparse`]: crate::device::QueueGuard::bind_sparse
    Sparse(Vec<SparseImageMemoryRequirements>),

    /// The image is backed by memory owned by a [`Swapchain`].
    Swapchain {
        swapchain: Arc<Swapchain>,
        image_index: u32,
    },
}

impl Image {
    fn from_raw(inner: RawImage, memory: ImageMemory) -> Self {
        let aspects = inner.format.unwrap().aspects();
        let aspect_list: SmallVec<[ImageAspect; 4]> = aspects.into_iter().collect();
        let mip_level_size = inner.dimensions.array_layers() as DeviceSize;
        let aspect_size = mip_level_size * inner.mip_levels as DeviceSize;
        let range_size = aspect_list.len() as DeviceSize * aspect_size;
        let state = Mutex::new(ImageState::new(range_size, inner.initial_layout));

        Image {
            inner,
            memory,

            aspect_list,
            aspect_size,
            mip_level_size,
            range_size,
            state,
        }
    }

    pub(crate) unsafe fn from_swapchain(
        handle: ash::vk::Image,
        swapchain: Arc<Swapchain>,
        image_index: u32,
    ) -> Self {
        let create_info = ImageCreateInfo {
            flags: ImageCreateFlags::empty(),
            dimensions: ImageDimensions::Dim2d {
                width: swapchain.image_extent()[0],
                height: swapchain.image_extent()[1],
                array_layers: swapchain.image_array_layers(),
            },
            format: Some(swapchain.image_format()),
            initial_layout: ImageLayout::Undefined,
            mip_levels: 1,
            samples: SampleCount::Sample1,
            tiling: ImageTiling::Optimal,
            usage: swapchain.image_usage(),
            stencil_usage: swapchain.image_usage(),
            sharing: swapchain.image_sharing().clone(),
            ..Default::default()
        };

        Self::from_raw(
            RawImage::from_handle_with_destruction(
                swapchain.device().clone(),
                handle,
                create_info,
                false,
            ),
            ImageMemory::Swapchain {
                swapchain,
                image_index,
            },
        )
    }

    /// Returns the type of memory that is backing this image.
    #[inline]
    pub fn memory(&self) -> &ImageMemory {
        &self.memory
    }

    /// Returns the memory requirements for this image.
    ///
    /// - If `self.flags().disjoint` is not set, this returns a slice with a length of 1.
    /// - If `self.flags().disjoint` is set, this returns a slice with a length equal to
    ///   `self.format().unwrap().planes().len()`.
    #[inline]
    pub fn memory_requirements(&self) -> &[MemoryRequirements] {
        &self.inner.memory_requirements
    }

    /// Returns the flags the image was created with.
    #[inline]
    pub fn flags(&self) -> ImageCreateFlags {
        self.inner.flags
    }

    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> ImageDimensions {
        self.inner.dimensions
    }

    /// Returns the image's format.
    #[inline]
    pub fn format(&self) -> Option<Format> {
        self.inner.format
    }

    /// Returns the features supported by the image's format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.inner.format_features
    }

    /// Returns the number of mipmap levels in the image.
    #[inline]
    pub fn mip_levels(&self) -> u32 {
        self.inner.mip_levels
    }

    /// Returns the initial layout of the image.
    #[inline]
    pub fn initial_layout(&self) -> ImageLayout {
        self.inner.initial_layout
    }

    /// Returns the number of samples for the image.
    #[inline]
    pub fn samples(&self) -> SampleCount {
        self.inner.samples
    }

    /// Returns the tiling of the image.
    #[inline]
    pub fn tiling(&self) -> ImageTiling {
        self.inner.tiling
    }

    /// Returns the usage the image was created with.
    #[inline]
    pub fn usage(&self) -> ImageUsage {
        self.inner.usage
    }

    /// Returns the stencil usage the image was created with.
    #[inline]
    pub fn stencil_usage(&self) -> ImageUsage {
        self.inner.stencil_usage
    }

    /// Returns the sharing the image was created with.
    #[inline]
    pub fn sharing(&self) -> &Sharing<SmallVec<[u32; 4]>> {
        &self.inner.sharing
    }

    /// Returns the external memory handle types that are supported with this image.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.inner.external_memory_handle_types
    }

    /// Returns an `ImageSubresourceLayers` covering the first mip level of the image. All aspects
    /// of the image are selected, or `plane0` if the image is multi-planar.
    #[inline]
    pub fn subresource_layers(&self) -> ImageSubresourceLayers {
        self.inner.subresource_layers()
    }

    /// Returns an `ImageSubresourceRange` covering the whole image. If the image is multi-planar,
    /// only the `color` aspect is selected.
    #[inline]
    pub fn subresource_range(&self) -> ImageSubresourceRange {
        self.inner.subresource_range()
    }

    /// Queries the memory layout of a single subresource of the image.
    ///
    /// Only images with linear tiling are supported, if they do not have a format with both a
    /// depth and a stencil format. Images with optimal tiling have an opaque image layout that is
    /// not suitable for direct memory accesses, and likewise for combined depth/stencil formats.
    /// Multi-planar formats are supported, but you must specify one of the planes as the `aspect`,
    /// not [`ImageAspect::Color`].
    ///
    /// The layout is invariant for each image. However it is not cached, as this would waste
    /// memory in the case of non-linear-tiling images. You are encouraged to store the layout
    /// somewhere in order to avoid calling this semi-expensive function at every single memory
    /// access.
    #[inline]
    pub fn subresource_layout(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<SubresourceLayout, ImageError> {
        self.inner
            .subresource_layout(aspect, mip_level, array_layer)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn subresource_layout_unchecked(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> SubresourceLayout {
        self.inner
            .subresource_layout_unchecked(aspect, mip_level, array_layer)
    }

    pub(crate) fn range_size(&self) -> DeviceSize {
        self.range_size
    }

    /// Returns an iterator over subresource ranges.
    ///
    /// In ranges, the subresources are "flattened" to `DeviceSize`, where each index in the range
    /// is a single array layer. The layers are arranged hierarchically: aspects at the top level,
    /// with the mip levels in that aspect, and the array layers in that mip level.
    pub(crate) fn iter_ranges(
        &self,
        subresource_range: ImageSubresourceRange,
    ) -> SubresourceRangeIterator {
        assert!(self
            .format()
            .unwrap()
            .aspects()
            .contains(subresource_range.aspects));
        assert!(subresource_range.mip_levels.end <= self.inner.mip_levels);
        assert!(subresource_range.array_layers.end <= self.inner.dimensions.array_layers());

        SubresourceRangeIterator::new(
            subresource_range,
            &self.aspect_list,
            self.aspect_size,
            self.inner.mip_levels,
            self.mip_level_size,
            self.inner.dimensions.array_layers(),
        )
    }

    pub(crate) fn range_to_subresources(
        &self,
        mut range: Range<DeviceSize>,
    ) -> ImageSubresourceRange {
        debug_assert!(!range.is_empty());
        debug_assert!(range.end <= self.range_size);

        if range.end - range.start > self.aspect_size {
            debug_assert!(range.start % self.aspect_size == 0);
            debug_assert!(range.end % self.aspect_size == 0);

            let start_aspect_num = (range.start / self.aspect_size) as usize;
            let end_aspect_num = (range.end / self.aspect_size) as usize;

            ImageSubresourceRange {
                aspects: self.aspect_list[start_aspect_num..end_aspect_num]
                    .iter()
                    .copied()
                    .collect(),
                mip_levels: 0..self.inner.mip_levels,
                array_layers: 0..self.inner.dimensions.array_layers(),
            }
        } else {
            let aspect_num = (range.start / self.aspect_size) as usize;
            range.start %= self.aspect_size;
            range.end %= self.aspect_size;

            // Wraparound
            if range.end == 0 {
                range.end = self.aspect_size;
            }

            if range.end - range.start > self.mip_level_size {
                debug_assert!(range.start % self.mip_level_size == 0);
                debug_assert!(range.end % self.mip_level_size == 0);

                let start_mip_level = (range.start / self.mip_level_size) as u32;
                let end_mip_level = (range.end / self.mip_level_size) as u32;

                ImageSubresourceRange {
                    aspects: self.aspect_list[aspect_num].into(),
                    mip_levels: start_mip_level..end_mip_level,
                    array_layers: 0..self.inner.dimensions.array_layers(),
                }
            } else {
                let mip_level = (range.start / self.mip_level_size) as u32;
                range.start %= self.mip_level_size;
                range.end %= self.mip_level_size;

                // Wraparound
                if range.end == 0 {
                    range.end = self.mip_level_size;
                }

                let start_array_layer = range.start as u32;
                let end_array_layer = range.end as u32;

                ImageSubresourceRange {
                    aspects: self.aspect_list[aspect_num].into(),
                    mip_levels: mip_level..mip_level + 1,
                    array_layers: start_array_layer..end_array_layer,
                }
            }
        }
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, ImageState> {
        self.state.lock()
    }
}

unsafe impl VulkanObject for Image {
    type Handle = ash::vk::Image;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle
    }
}

unsafe impl DeviceOwned for Image {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }
}

impl PartialEq for Image {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for Image {}

impl Hash for Image {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

/// The current state of an image.
#[derive(Debug)]
pub(crate) struct ImageState {
    ranges: RangeMap<DeviceSize, ImageRangeState>,
}

impl ImageState {
    fn new(size: DeviceSize, initial_layout: ImageLayout) -> Self {
        ImageState {
            ranges: [(
                0..size,
                ImageRangeState {
                    current_access: CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads: 0,
                    },
                    layout: initial_layout,
                },
            )]
            .into_iter()
            .collect(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn check_cpu_read(&self, range: Range<DeviceSize>) -> Result<(), ReadLockError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::CpuExclusive { .. } => return Err(ReadLockError::CpuWriteLocked),
                CurrentAccess::GpuExclusive { .. } => return Err(ReadLockError::GpuWriteLocked),
                CurrentAccess::Shared { .. } => (),
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cpu_read_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::Shared { cpu_reads, .. } => {
                    *cpu_reads += 1;
                }
                _ => unreachable!("Image is being written by the CPU or GPU"),
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cpu_read_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::Shared { cpu_reads, .. } => *cpu_reads -= 1,
                _ => unreachable!("Image was not locked for CPU read"),
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn check_cpu_write(&self, range: Range<DeviceSize>) -> Result<(), WriteLockError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::CpuExclusive => return Err(WriteLockError::CpuLocked),
                CurrentAccess::GpuExclusive { .. } => return Err(WriteLockError::GpuLocked),
                CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads: 0,
                } => (),
                CurrentAccess::Shared { cpu_reads, .. } if *cpu_reads > 0 => {
                    return Err(WriteLockError::CpuLocked)
                }
                CurrentAccess::Shared { .. } => return Err(WriteLockError::GpuLocked),
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cpu_write_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            state.current_access = CurrentAccess::CpuExclusive;
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cpu_write_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::CpuExclusive => {
                    state.current_access = CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads: 0,
                    }
                }
                _ => unreachable!("Image was not locked for CPU write"),
            }
        }
    }

    pub(crate) fn check_gpu_read(
        &self,
        range: Range<DeviceSize>,
        expected_layout: ImageLayout,
    ) -> Result<(), AccessError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::Shared { .. } => (),
                _ => return Err(AccessError::AlreadyInUse),
            }

            if expected_layout != ImageLayout::Undefined && state.layout != expected_layout {
                return Err(AccessError::UnexpectedImageLayout {
                    allowed: state.layout,
                    requested: expected_layout,
                });
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn gpu_read_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_reads, .. }
                | CurrentAccess::Shared { gpu_reads, .. } => *gpu_reads += 1,
                _ => unreachable!("Image is being written by the CPU"),
            }
        }
    }

    pub(crate) unsafe fn gpu_read_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_reads, .. } => *gpu_reads -= 1,
                CurrentAccess::Shared { gpu_reads, .. } => *gpu_reads -= 1,
                _ => unreachable!("Buffer was not locked for GPU read"),
            }
        }
    }

    pub(crate) fn check_gpu_write(
        &self,
        range: Range<DeviceSize>,
        expected_layout: ImageLayout,
    ) -> Result<(), AccessError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads: 0,
                } => (),
                _ => return Err(AccessError::AlreadyInUse),
            }

            if expected_layout != ImageLayout::Undefined && state.layout != expected_layout {
                return Err(AccessError::UnexpectedImageLayout {
                    allowed: state.layout,
                    requested: expected_layout,
                });
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn gpu_write_lock(
        &mut self,
        range: Range<DeviceSize>,
        destination_layout: ImageLayout,
    ) {
        debug_assert!(!matches!(
            destination_layout,
            ImageLayout::Undefined | ImageLayout::Preinitialized
        ));

        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_writes, .. } => *gpu_writes += 1,
                &mut CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads,
                } => {
                    state.current_access = CurrentAccess::GpuExclusive {
                        gpu_reads,
                        gpu_writes: 1,
                    }
                }
                _ => unreachable!("Image is being accessed by the CPU"),
            }

            state.layout = destination_layout;
        }
    }

    pub(crate) unsafe fn gpu_write_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                &mut CurrentAccess::GpuExclusive {
                    gpu_reads,
                    gpu_writes: 1,
                } => {
                    state.current_access = CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads,
                    }
                }
                CurrentAccess::GpuExclusive { gpu_writes, .. } => *gpu_writes -= 1,
                _ => unreachable!("Image was not locked for GPU write"),
            }
        }
    }
}

/// The current state of a specific subresource range in an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ImageRangeState {
    current_access: CurrentAccess,
    layout: ImageLayout,
}

#[derive(Clone)]
pub(crate) struct SubresourceRangeIterator {
    next_fn: fn(&mut Self) -> Option<Range<DeviceSize>>,
    image_aspect_size: DeviceSize,
    image_mip_level_size: DeviceSize,
    mip_levels: Range<u32>,
    array_layers: Range<u32>,

    aspect_nums: Peekable<smallvec::IntoIter<[usize; 4]>>,
    current_aspect_num: Option<usize>,
    current_mip_level: u32,
}

impl SubresourceRangeIterator {
    fn new(
        subresource_range: ImageSubresourceRange,
        image_aspect_list: &[ImageAspect],
        image_aspect_size: DeviceSize,
        image_mip_levels: u32,
        image_mip_level_size: DeviceSize,
        image_array_layers: u32,
    ) -> Self {
        assert!(!subresource_range.mip_levels.is_empty());
        assert!(!subresource_range.array_layers.is_empty());

        let next_fn = if subresource_range.array_layers.start != 0
            || subresource_range.array_layers.end != image_array_layers
        {
            Self::next_some_layers
        } else if subresource_range.mip_levels.start != 0
            || subresource_range.mip_levels.end != image_mip_levels
        {
            Self::next_some_levels_all_layers
        } else {
            Self::next_all_levels_all_layers
        };

        let mut aspect_nums = subresource_range
            .aspects
            .into_iter()
            .map(|aspect| image_aspect_list.iter().position(|&a| a == aspect).unwrap())
            .collect::<SmallVec<[usize; 4]>>()
            .into_iter()
            .peekable();
        assert!(aspect_nums.len() != 0);
        let current_aspect_num = aspect_nums.next();
        let current_mip_level = subresource_range.mip_levels.start;

        Self {
            next_fn,
            image_aspect_size,
            image_mip_level_size,
            mip_levels: subresource_range.mip_levels,
            array_layers: subresource_range.array_layers,

            aspect_nums,
            current_aspect_num,
            current_mip_level,
        }
    }

    /// Used when the requested range contains only a subset of the array layers in the image.
    /// The iterator returns one range for each mip level and aspect, each covering the range of
    /// array layers of that mip level and aspect.
    fn next_some_layers(&mut self) -> Option<Range<DeviceSize>> {
        self.current_aspect_num.map(|aspect_num| {
            let mip_level_offset = aspect_num as DeviceSize * self.image_aspect_size
                + self.current_mip_level as DeviceSize * self.image_mip_level_size;
            self.current_mip_level += 1;

            if self.current_mip_level >= self.mip_levels.end {
                self.current_mip_level = self.mip_levels.start;
                self.current_aspect_num = self.aspect_nums.next();
            }

            let start = mip_level_offset + self.array_layers.start as DeviceSize;
            let end = mip_level_offset + self.array_layers.end as DeviceSize;
            start..end
        })
    }

    /// Used when the requested range contains all array layers in the image, but not all mip
    /// levels. The iterator returns one range for each aspect, each covering all layers of the
    /// range of mip levels of that aspect.
    fn next_some_levels_all_layers(&mut self) -> Option<Range<DeviceSize>> {
        self.current_aspect_num.map(|aspect_num| {
            let aspect_offset = aspect_num as DeviceSize * self.image_aspect_size;
            self.current_aspect_num = self.aspect_nums.next();

            let start =
                aspect_offset + self.mip_levels.start as DeviceSize * self.image_mip_level_size;
            let end = aspect_offset + self.mip_levels.end as DeviceSize * self.image_mip_level_size;
            start..end
        })
    }

    /// Used when the requested range contains all array layers and mip levels in the image.
    /// The iterator returns one range for each series of adjacent aspect numbers, each covering
    /// all mip levels and all layers of those aspects. If the range contains the whole image, then
    /// exactly one range is returned since all aspect numbers will be adjacent.
    fn next_all_levels_all_layers(&mut self) -> Option<Range<DeviceSize>> {
        self.current_aspect_num.map(|aspect_num_start| {
            self.current_aspect_num = self.aspect_nums.next();
            let mut aspect_num_end = aspect_num_start + 1;

            while self.current_aspect_num == Some(aspect_num_end) {
                self.current_aspect_num = self.aspect_nums.next();
                aspect_num_end += 1;
            }

            let start = aspect_num_start as DeviceSize * self.image_aspect_size;
            let end = aspect_num_end as DeviceSize * self.image_aspect_size;
            start..end
        })
    }
}

impl Iterator for SubresourceRangeIterator {
    type Item = Range<DeviceSize>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.next_fn)(self)
    }
}

impl FusedIterator for SubresourceRangeIterator {}

/// Describes the memory layout of a single subresource of an image.
///
/// The address of a texel at `(x, y, z, layer)` is `layer * array_pitch + z * depth_pitch +
/// y * row_pitch + x * size_of_each_texel + offset`. `size_of_each_texel` must be determined
/// depending on the format. The same formula applies for compressed formats, except that the
/// coordinates must be in number of blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubresourceLayout {
    /// The number of bytes from the start of the memory to the start of the queried subresource.
    pub offset: DeviceSize,

    /// The total number of bytes for the queried subresource.
    pub size: DeviceSize,

    /// The number of bytes between two texels or two blocks in adjacent rows.
    pub row_pitch: DeviceSize,

    /// For images with more than one array layer, the number of bytes between two texels or two
    /// blocks in adjacent array layers.
    pub array_pitch: Option<DeviceSize>,

    /// For 3D images, the number of bytes between two texels or two blocks in adjacent depth
    /// layers.
    pub depth_pitch: Option<DeviceSize>,
}

/// Error that can happen in image functions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageError {
    VulkanError(VulkanError),

    /// Allocating memory failed.
    AllocError(AllocationCreationError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The provided number of elements in `allocations` is not what is required for `image`.
    AllocationsWrongNumberOfElements {
        provided: usize,
        required: usize,
    },

    /// The `array_2d_compatible` flag was enabled, but the image type was not 3D.
    Array2dCompatibleNot3d,

    /// The provided array layer is not less than the number of array layers in the image.
    ArrayLayerOutOfRange {
        provided_array_layer: u32,
        image_array_layers: u32,
    },

    /// The provided aspect is not present in the image, or is not allowed.
    AspectNotAllowed {
        provided_aspect: ImageAspect,
        allowed_aspects: ImageAspects,
    },

    /// The `block_texel_view_compatible` flag was enabled, but the given format was not compressed.
    BlockTexelViewCompatibleNotCompressed,

    /// The `cube_compatible` flag was enabled, but the image type was not 2D.
    CubeCompatibleNot2d,

    /// The `cube_compatible` flag was enabled, but the number of array layers was less than 6.
    CubeCompatibleNotEnoughArrayLayers,

    /// The `cube_compatible` flag was enabled, but the image dimensions were not square.
    CubeCompatibleNotSquare,

    /// The `cube_compatible` flag was enabled together with multisampling.
    CubeCompatibleMultisampling,

    /// The memory was created dedicated to a resource, but not to this image.
    DedicatedAllocationMismatch,

    /// A dedicated allocation is required for this image, but one was not provided.
    DedicatedAllocationRequired,

    /// The image has a format with both a depth and a stencil aspect, which is not supported for
    /// this operation.
    DepthStencilFormatsNotSupported,

    /// The `disjoint` flag was enabled, but the given format is either not multi-planar, or does
    /// not support disjoint images.
    DisjointFormatNotSupported,

    /// One or more external memory handle types were provided, but the initial layout was not
    /// `Undefined`.
    ExternalMemoryInvalidInitialLayout,

    /// The given format was not supported by the device.
    FormatNotSupported,

    /// A requested usage flag was not supported by the given format.
    FormatUsageNotSupported {
        usage: &'static str,
    },

    /// The image configuration as queried through the `image_format_properties` function was not
    /// supported by the device.
    ImageFormatPropertiesNotSupported,

    /// The number of array layers exceeds the maximum supported by the device for this image
    /// configuration.
    MaxArrayLayersExceeded {
        array_layers: u32,
        max: u32,
    },

    /// The specified dimensions exceed the maximum supported by the device for this image
    /// configuration.
    MaxDimensionsExceeded {
        extent: [u32; 3],
        max: [u32; 3],
    },

    /// The usage included one of the attachment types, and the specified width and height exceeded
    /// the `max_framebuffer_width` or `max_framebuffer_height` limits.
    MaxFramebufferDimensionsExceeded {
        extent: [u32; 2],
        max: [u32; 2],
    },

    /// The maximum number of mip levels for the given dimensions has been exceeded.
    MaxMipLevelsExceeded {
        mip_levels: u32,
        max: u32,
    },

    /// In an `allocations` element, the offset of the allocation does not have the required
    /// alignment.
    MemoryAllocationNotAligned {
        allocations_index: usize,
        allocation_offset: DeviceSize,
        required_alignment: DeviceSize,
    },

    /// In an `allocations` element, the size of the allocation is smaller than what is required.
    MemoryAllocationTooSmall {
        allocations_index: usize,
        allocation_size: DeviceSize,
        required_size: DeviceSize,
    },

    /// In an `allocations` element, the memory was created with export handle types, but none of
    /// these handle types were enabled on the image.
    MemoryExternalHandleTypesDisjoint {
        allocations_index: usize,
        image_handle_types: ExternalMemoryHandleTypes,
        memory_export_handle_types: ExternalMemoryHandleTypes,
    },

    /// In an `allocations` element, the memory was created with an import, but the import's handle
    /// type was not enabled on the image.
    MemoryImportedHandleTypeNotEnabled {
        allocations_index: usize,
        image_handle_types: ExternalMemoryHandleTypes,
        memory_imported_handle_type: ExternalMemoryHandleType,
    },

    /// In an `allocations` element, the protection of image and memory are not equal.
    MemoryProtectedMismatch {
        allocations_index: usize,
        image_protected: bool,
        memory_protected: bool,
    },

    /// In an `allocations` element, the provided memory type is not one of the allowed memory
    /// types that can be bound to this image or image plane.
    MemoryTypeNotAllowed {
        allocations_index: usize,
        provided_memory_type_index: u32,
        allowed_memory_type_bits: u32,
    },

    /// The provided mip level is not less than the number of mip levels in the image.
    MipLevelOutOfRange {
        provided_mip_level: u32,
        image_mip_levels: u32,
    },

    /// Multisampling was enabled, and the `cube_compatible` flag was set.
    MultisampleCubeCompatible,

    /// Multisampling was enabled, and tiling was `Linear`.
    MultisampleLinearTiling,

    /// Multisampling was enabled, and multiple mip levels were specified.
    MultisampleMultipleMipLevels,

    /// Multisampling was enabled, but the image type was not 2D.
    MultisampleNot2d,

    /// The image has optimal tiling, which is not supported for this operation.
    OptimalTilingNotSupported,

    /// The sample count is not supported by the device for this image configuration.
    SampleCountNotSupported {
        samples: SampleCount,
        supported: SampleCounts,
    },

    /// The sharing mode was set to `Concurrent`, but one of the specified queue family indices was
    /// out of range.
    SharingQueueFamilyIndexOutOfRange {
        queue_family_index: u32,
        queue_family_count: u32,
    },

    /// The provided `usage` and `stencil_usage` have different values for
    /// `depth_stencil_attachment` or `transient_attachment`.
    StencilUsageMismatch {
        usage: ImageUsage,
        stencil_usage: ImageUsage,
    },

    /// A YCbCr format was given, but the specified width and/or height was not a multiple of 2
    /// as required by the format's chroma subsampling.
    YcbcrFormatInvalidDimensions,

    /// A YCbCr format was given, and multiple mip levels were specified.
    YcbcrFormatMultipleMipLevels,

    /// A YCbCr format was given, and multisampling was enabled.
    YcbcrFormatMultisampling,

    /// A YCbCr format was given, but the image type was not 2D.
    YcbcrFormatNot2d,

    DirectImageViewCreationFailed(ImageViewCreationError),
}

impl Error for ImageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ImageError::AllocError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for ImageError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::VulkanError(_) => write!(f, "a runtime error occurred"),
            Self::AllocError(_) => write!(f, "allocating memory failed"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::AllocationsWrongNumberOfElements { provided, required } => write!(
                f,
                "the provided number of elements in `allocations` ({}) is not what is required for \
                `image` ({})",
                provided, required,
            ),
            Self::Array2dCompatibleNot3d => write!(
                f,
                "the `array_2d_compatible` flag was enabled, but the image type was not 3D",
            ),
            Self::ArrayLayerOutOfRange {
                provided_array_layer,
                image_array_layers,
            } => write!(
                f,
                "the provided array layer ({}) is not less than the number of array layers in the image ({})",
                provided_array_layer, image_array_layers,
            ),
            Self::AspectNotAllowed {
                provided_aspect,
                allowed_aspects,
            } => write!(
                f,
                "the provided aspect ({:?}) is not present in the image, or is not allowed ({:?})",
                provided_aspect, allowed_aspects,
            ),
            Self::BlockTexelViewCompatibleNotCompressed => write!(
                f,
                "the `block_texel_view_compatible` flag was enabled, but the given format was not \
                compressed",
            ),
            Self::CubeCompatibleNot2d => write!(
                f,
                "the `cube_compatible` flag was enabled, but the image type was not 2D",
            ),
            Self::CubeCompatibleNotEnoughArrayLayers => write!(
                f,
                "the `cube_compatible` flag was enabled, but the number of array layers was less \
                than 6",
            ),
            Self::CubeCompatibleNotSquare => write!(
                f,
                "the `cube_compatible` flag was enabled, but the image dimensions were not square",
            ),
            Self::CubeCompatibleMultisampling => write!(
                f,
                "the `cube_compatible` flag was enabled together with multisampling",
            ),
            Self::DedicatedAllocationMismatch => write!(
                f,
                "the memory was created dedicated to a resource, but not to this image",
            ),
            Self::DedicatedAllocationRequired => write!(
                f,
                "a dedicated allocation is required for this image, but one was not provided"
            ),
            Self::DepthStencilFormatsNotSupported => write!(
                f,
                "the image has a format with both a depth and a stencil aspect, which is not \
                supported for this operation",
            ),
            Self::DisjointFormatNotSupported => write!(
                f,
                "the `disjoint` flag was enabled, but the given format is either not multi-planar, \
                or does not support disjoint images",
            ),
            Self::ExternalMemoryInvalidInitialLayout => write!(
                f,
                "one or more external memory handle types were provided, but the initial layout \
                was not `Undefined`",
            ),
            Self::FormatNotSupported => {
                write!(f, "the given format was not supported by the device")
            }
            Self::FormatUsageNotSupported { .. } => write!(
                f,
                "a requested usage flag was not supported by the given format",
            ),
            Self::ImageFormatPropertiesNotSupported => write!(
                f,
                "the image configuration as queried through the `image_format_properties` function \
                was not supported by the device",
            ),
            Self::MaxArrayLayersExceeded { .. } => write!(
                f,
                "the number of array layers exceeds the maximum supported by the device for this \
                image configuration",
            ),
            Self::MaxDimensionsExceeded { .. } => write!(
                f,
                "the specified dimensions exceed the maximum supported by the device for this \
                image configuration",
            ),
            Self::MaxFramebufferDimensionsExceeded { .. } => write!(
                f,
                "the usage included one of the attachment types, and the specified width and \
                height exceeded the `max_framebuffer_width` or `max_framebuffer_height` limits",
            ),
            Self::MaxMipLevelsExceeded { .. } => write!(
                f,
                "the maximum number of mip levels for the given dimensions has been exceeded",
            ),
            Self::MemoryAllocationNotAligned {
                allocations_index,
                allocation_offset,
                required_alignment,
            } => write!(
                f,
                "in `allocations` element {}, the offset of the allocation ({}) does not have the \
                required alignment ({})",
                allocations_index, allocation_offset, required_alignment,
            ),
            Self::MemoryAllocationTooSmall {
                allocations_index,
                allocation_size,
                required_size,
            } => write!(
                f,
                "in `allocations` element {}, the size of the allocation ({}) is smaller than what \
                is required ({})",
                allocations_index, allocation_size, required_size,
            ),
            Self::MemoryExternalHandleTypesDisjoint {
                allocations_index, ..
            } => write!(
                f,
                "in `allocations` element {}, the memory was created with export handle types, but \
                none of these handle types were enabled on the image",
                allocations_index,
            ),
            Self::MemoryImportedHandleTypeNotEnabled {
                allocations_index, ..
            } => write!(
                f,
                "in `allocations` element {}, the memory was created with an import, but the \
                import's handle type was not enabled on the image",
                allocations_index,
            ),
            Self::MemoryProtectedMismatch {
                allocations_index,
                image_protected,
                memory_protected,
            } => write!(
                f,
                "in `allocations` element {}, the protection of image ({}) and memory ({}) are not \
                equal",
                allocations_index, image_protected, memory_protected,
            ),
            Self::MemoryTypeNotAllowed {
                allocations_index,
                provided_memory_type_index,
                allowed_memory_type_bits,
            } => write!(
                f,
                "in `allocations` element {}, the provided memory type ({}) is not one of the \
                allowed memory types (",
                allocations_index, provided_memory_type_index,
            )
            .and_then(|_| {
                let mut first = true;

                for i in (0..size_of_val(allowed_memory_type_bits))
                    .filter(|i| allowed_memory_type_bits & (1 << i) != 0)
                {
                    if first {
                        write!(f, "{}", i)?;
                        first = false;
                    } else {
                        write!(f, ", {}", i)?;
                    }
                }

                Ok(())
            })
            .and_then(|_| write!(f, ") that can be bound to this buffer")),
            Self::MipLevelOutOfRange {
                provided_mip_level,
                image_mip_levels,
            } => write!(
                f,
                "the provided mip level ({}) is not less than the number of mip levels in the image ({})",
                provided_mip_level, image_mip_levels,
            ),
            Self::MultisampleCubeCompatible => write!(
                f,
                "multisampling was enabled, and the `cube_compatible` flag was set",
            ),
            Self::MultisampleLinearTiling => {
                write!(f, "multisampling was enabled, and tiling was `Linear`")
            }
            Self::MultisampleMultipleMipLevels => write!(
                f,
                "multisampling was enabled, and multiple mip levels were specified",
            ),
            Self::MultisampleNot2d => write!(
                f,
                "multisampling was enabled, but the image type was not 2D",
            ),
            Self::OptimalTilingNotSupported => write!(
                f,
                "the image has optimal tiling, which is not supported for this operation",
            ),
            Self::SampleCountNotSupported { .. } => write!(
                f,
                "the sample count is not supported by the device for this image configuration",
            ),
            Self::SharingQueueFamilyIndexOutOfRange { .. } => write!(
                f,
                "the sharing mode was set to `Concurrent`, but one of the specified queue family \
                indices was out of range",
            ),
            Self::StencilUsageMismatch {
                usage: _,
                stencil_usage: _,
            } => write!(
                f,
                "the provided `usage` and `stencil_usage` have different values for \
                `depth_stencil_attachment` or `transient_attachment`",
            ),
            Self::YcbcrFormatInvalidDimensions => write!(
                f,
                "a YCbCr format was given, but the specified width and/or height was not a \
                multiple of 2 as required by the format's chroma subsampling",
            ),
            Self::YcbcrFormatMultipleMipLevels => write!(
                f,
                "a YCbCr format was given, and multiple mip levels were specified",
            ),
            Self::YcbcrFormatMultisampling => {
                write!(f, "a YCbCr format was given, and multisampling was enabled")
            }
            Self::YcbcrFormatNot2d => {
                write!(f, "a YCbCr format was given, but the image type was not 2D")
            }
            Self::DirectImageViewCreationFailed(e) => write!(f, "Image view creation failed {}", e),
        }
    }
}

impl From<VulkanError> for ImageError {
    fn from(err: VulkanError) -> Self {
        Self::VulkanError(err)
    }
}

impl From<AllocationCreationError> for ImageError {
    fn from(err: AllocationCreationError) -> Self {
        Self::AllocError(err)
    }
}

impl From<RequirementNotMet> for ImageError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageCreateInfo, ImageError, ImageUsage, RawImage};
    use crate::{
        format::Format,
        image::{
            sys::SubresourceRangeIterator, ImageAspect, ImageAspects, ImageCreateFlags,
            ImageDimensions, ImageSubresourceRange, SampleCount,
        },
        DeviceSize, RequiresOneOf,
    };
    use smallvec::SmallVec;

    #[test]
    fn create_sampled() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = RawImage::new(
            device,
            ImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                format: Some(Format::R8G8B8A8_UNORM),
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn create_transient() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = RawImage::new(
            device,
            ImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                format: Some(Format::R8G8B8A8_UNORM),
                usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn zero_mipmap() {
        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            let _ = RawImage::new(
                device,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 32,
                        height: 32,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    mip_levels: 0,
                    usage: ImageUsage::SAMPLED,
                    ..Default::default()
                },
            );
        });
    }

    #[test]
    fn mipmaps_too_high() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::new(
            device,
            ImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                format: Some(Format::R8G8B8A8_UNORM),
                mip_levels: u32::MAX,
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        );

        match res {
            Err(ImageError::MaxMipLevelsExceeded { .. }) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn shader_storage_image_multisample() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::new(
            device,
            ImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                format: Some(Format::R8G8B8A8_UNORM),
                samples: SampleCount::Sample2,
                usage: ImageUsage::STORAGE,
                ..Default::default()
            },
        );

        match res {
            Err(ImageError::RequirementNotMet {
                requires_one_of: RequiresOneOf { features, .. },
                ..
            }) if features.contains(&"shader_storage_image_multisample") => (),
            Err(ImageError::SampleCountNotSupported { .. }) => (), // unlikely but possible
            _ => panic!(),
        };
    }

    #[test]
    fn compressed_not_color_attachment() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::new(
            device,
            ImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                format: Some(Format::ASTC_5x4_UNORM_BLOCK),
                usage: ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
        );

        match res {
            Err(ImageError::FormatNotSupported) => (),
            Err(ImageError::FormatUsageNotSupported {
                usage: "color_attachment",
            }) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn transient_forbidden_with_some_usages() {
        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            let _ = RawImage::new(
                device,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 32,
                        height: 32,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::SAMPLED,
                    ..Default::default()
                },
            );
        })
    }

    #[test]
    fn cubecompatible_dims_mismatch() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::new(
            device,
            ImageCreateInfo {
                flags: ImageCreateFlags::CUBE_COMPATIBLE,
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 64,
                    array_layers: 1,
                },
                format: Some(Format::R8G8B8A8_UNORM),
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        );

        match res {
            Err(ImageError::CubeCompatibleNotEnoughArrayLayers) => (),
            Err(ImageError::CubeCompatibleNotSquare) => (),
            _ => panic!(),
        };
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn subresource_range_iterator() {
        // A fictitious set of aspects that no real image would actually ever have.
        let image_aspect_list: SmallVec<[ImageAspect; 4]> = (ImageAspects::COLOR
            | ImageAspects::DEPTH
            | ImageAspects::STENCIL
            | ImageAspects::PLANE_0)
            .into_iter()
            .collect();
        let image_mip_levels = 6;
        let image_array_layers = 8;

        let mip = image_array_layers as DeviceSize;
        let asp = mip * image_mip_levels as DeviceSize;

        // Whole image
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR
                    | ImageAspects::DEPTH
                    | ImageAspects::STENCIL
                    | ImageAspects::PLANE_0,
                mip_levels: 0..6,
                array_layers: 0..8,
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );

        assert_eq!(iter.next(), Some(0 * asp..4 * asp));
        assert_eq!(iter.next(), None);

        // Only some aspects
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR | ImageAspects::DEPTH | ImageAspects::PLANE_0,
                mip_levels: 0..6,
                array_layers: 0..8,
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );

        assert_eq!(iter.next(), Some(0 * asp..2 * asp));
        assert_eq!(iter.next(), Some(3 * asp..4 * asp));
        assert_eq!(iter.next(), None);

        // Two aspects, and only some of the mip levels
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::DEPTH | ImageAspects::STENCIL,
                mip_levels: 2..4,
                array_layers: 0..8,
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );
        assert_eq!(iter.next(), Some(1 * asp + 2 * mip..1 * asp + 4 * mip));
        assert_eq!(iter.next(), Some(2 * asp + 2 * mip..2 * asp + 4 * mip));
        assert_eq!(iter.next(), None);

        // One aspect, one mip level, only some of the array layers
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR,

                mip_levels: 0..1,
                array_layers: 2..4,
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );

        assert_eq!(
            iter.next(),
            Some(0 * asp + 0 * mip + 2..0 * asp + 0 * mip + 4)
        );
        assert_eq!(iter.next(), None);

        // Two aspects, two mip levels, only some of the array layers
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::DEPTH | ImageAspects::STENCIL,
                mip_levels: 2..4,
                array_layers: 6..8,
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );
        assert_eq!(
            iter.next(),
            Some(1 * asp + 2 * mip + 6..1 * asp + 2 * mip + 8)
        );
        assert_eq!(
            iter.next(),
            Some(1 * asp + 3 * mip + 6..1 * asp + 3 * mip + 8)
        );
        assert_eq!(
            iter.next(),
            Some(2 * asp + 2 * mip + 6..2 * asp + 2 * mip + 8)
        );
        assert_eq!(
            iter.next(),
            Some(2 * asp + 3 * mip + 6..2 * asp + 3 * mip + 8)
        );
        assert_eq!(iter.next(), None);
    }
}
