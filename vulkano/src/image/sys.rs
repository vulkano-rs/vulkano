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
    device::{Device, DeviceOwned},
    format::{ChromaSampling, Format, FormatFeatures, NumericType},
    image::{
        view::ImageViewCreationError, ImageFormatInfo, ImageFormatProperties, ImageType,
        SparseImageFormatProperties,
    },
    memory::{
        allocator::AllocationCreationError, DeviceMemory, ExternalMemoryHandleType,
        ExternalMemoryHandleTypes, MemoryRequirements,
    },
    range_map::RangeMap,
    sync::{AccessError, CurrentAccess, Sharing},
    DeviceSize, OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use ash::vk::Handle;
use parking_lot::{Mutex, MutexGuard};
use smallvec::{smallvec, SmallVec};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    iter::{FusedIterator, Peekable},
    mem::MaybeUninit,
    ops::Range,
    ptr,
    sync::Arc,
};

/// A storage for pixels or arbitrary data.
///
/// # Safety
///
/// `UnsafeImage` is not unsafe to *create*, but it is unsafe to *use*:
///
/// - You must manually bind memory to the image with `bind_memory`. The memory must respect the
///   requirements returned by `memory_requirements`.
/// - The memory that you bind to the image must be manually kept alive.
/// - The queue family ownership must be manually enforced.
/// - The usage must be manually enforced.
/// - The image layout must be manually enforced and transitioned.
#[derive(Debug)]
pub struct UnsafeImage {
    handle: ash::vk::Image,
    device: Arc<Device>,

    dimensions: ImageDimensions,
    format: Option<Format>,
    format_features: FormatFeatures,
    initial_layout: ImageLayout,
    mip_levels: u32,
    samples: SampleCount,
    tiling: ImageTiling,
    usage: ImageUsage,
    stencil_usage: ImageUsage,
    external_memory_handle_types: ExternalMemoryHandleTypes,
    mutable_format: bool,
    cube_compatible: bool,
    array_2d_compatible: bool,
    block_texel_view_compatible: bool,

    aspect_list: SmallVec<[ImageAspect; 4]>,
    aspect_size: DeviceSize,
    mip_level_size: DeviceSize,
    needs_destruction: bool, // `vkDestroyImage` is called only if true.
    range_size: DeviceSize,
    state: Mutex<ImageState>,
}

impl UnsafeImage {
    /// Creates a new `UnsafeImage`.
    ///
    /// # Panics
    ///
    /// - Panics if one of the values in `create_info.dimensions` is zero.
    /// - Panics if `create_info.format` is `None`.
    /// - Panics if `create_info.block_texel_view_compatible` is set but not
    ///   `create_info.mutable_format`.
    /// - Panics if `create_info.mip_levels` is `0`.
    /// - Panics if `create_info.sharing` is [`Sharing::Concurrent`] with less than 2 items.
    /// - Panics if `create_info.initial)layout` is something other than
    ///   [`ImageLayout::Undefined`] or [`ImageLayout::Preinitialized`].
    /// - Panics if `create_info.usage` is empty.
    /// - Panics if `create_info.usage` contains `transient_attachment`, but does not also contain
    ///   at least one of `color_attachment`, `depth_stencil_attachment`, `input_attachment`, or
    ///   if it contains values other than these.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        mut create_info: UnsafeImageCreateInfo,
    ) -> Result<Arc<UnsafeImage>, ImageCreationError> {
        match &mut create_info.sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                // VUID-VkImageCreateInfo-sharingMode-01420
                queue_family_indices.sort_unstable();
                queue_family_indices.dedup();
            }
        }

        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(UnsafeImage::new_unchecked(device, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        create_info: &UnsafeImageCreateInfo,
    ) -> Result<FormatFeatures, ImageCreationError> {
        let &UnsafeImageCreateInfo {
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
            mutable_format,
            cube_compatible,
            array_2d_compatible,
            block_texel_view_compatible,
            _ne: _,
        } = create_info;

        let physical_device = device.physical_device();
        let device_properties = physical_device.properties();

        let format = format.unwrap(); // Can be None for "external formats" but Vulkano doesn't support that yet
        let aspects = format.aspects();

        let has_separate_stencil_usage =
            if stencil_usage.is_empty() || !(aspects.depth && aspects.stencil) {
                stencil_usage = usage;
                false
            } else {
                stencil_usage == usage
            };

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
                return Err(ImageCreationError::RequirementNotMet {
                    required_for: "`create_info.stencil_usage` is `Some` and `create_info.format` has both a depth and a stencil aspect",
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
        assert!(!block_texel_view_compatible || mutable_format);

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
        if format_features == FormatFeatures::default() {
            return Err(ImageCreationError::FormatNotSupported);
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
            return Err(ImageCreationError::MaxMipLevelsExceeded {
                mip_levels,
                max: max_mip_levels,
            });
        }

        // VUID-VkImageCreateInfo-samples-02257
        if samples != SampleCount::Sample1 {
            if image_type != ImageType::Dim2d {
                return Err(ImageCreationError::MultisampleNot2d);
            }

            if cube_compatible {
                return Err(ImageCreationError::MultisampleCubeCompatible);
            }

            if mip_levels != 1 {
                return Err(ImageCreationError::MultisampleMultipleMipLevels);
            }

            if tiling == ImageTiling::Linear {
                return Err(ImageCreationError::MultisampleLinearTiling);
            }
        }

        // Check limits for YCbCr formats
        if let Some(chroma_sampling) = format.ycbcr_chroma_sampling() {
            // VUID-VkImageCreateInfo-format-06410
            if mip_levels != 1 {
                return Err(ImageCreationError::YcbcrFormatMultipleMipLevels);
            }

            // VUID-VkImageCreateInfo-format-06411
            if samples != SampleCount::Sample1 {
                return Err(ImageCreationError::YcbcrFormatMultisampling);
            }

            // VUID-VkImageCreateInfo-format-06412
            if image_type != ImageType::Dim2d {
                return Err(ImageCreationError::YcbcrFormatNot2d);
            }

            // VUID-VkImageCreateInfo-format-06413
            if array_layers > 1 && !device.enabled_features().ycbcr_image_arrays {
                return Err(ImageCreationError::RequirementNotMet {
                    required_for: "`create_info.format.ycbcr_chroma_sampling()` is `Some` and `create_info.dimensions.array_layers()` is greater than `1`",
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
                        return Err(ImageCreationError::YcbcrFormatInvalidDimensions);
                    }
                }
                ChromaSampling::Mode420 => {
                    // VUID-VkImageCreateInfo-format-04712
                    // VUID-VkImageCreateInfo-format-04713
                    if !(extent[0] % 2 == 0 && extent[1] % 2 == 0) {
                        return Err(ImageCreationError::YcbcrFormatInvalidDimensions);
                    }
                }
            }
        }

        /* Check usage requirements */

        let combined_usage = usage | stencil_usage;

        if combined_usage.sampled && !format_features.sampled_image {
            return Err(ImageCreationError::FormatUsageNotSupported { usage: "sampled" });
        }

        if combined_usage.color_attachment && !format_features.color_attachment {
            return Err(ImageCreationError::FormatUsageNotSupported {
                usage: "color_attachment",
            });
        }

        if combined_usage.depth_stencil_attachment && !format_features.depth_stencil_attachment {
            return Err(ImageCreationError::FormatUsageNotSupported {
                usage: "depth_stencil_attachment",
            });
        }

        if combined_usage.input_attachment
            && !(format_features.color_attachment || format_features.depth_stencil_attachment)
        {
            return Err(ImageCreationError::FormatUsageNotSupported {
                usage: "input_attachment",
            });
        }

        if combined_usage.color_attachment
            || combined_usage.depth_stencil_attachment
            || combined_usage.input_attachment
            || combined_usage.transient_attachment
        {
            // VUID-VkImageCreateInfo-usage-00964
            // VUID-VkImageCreateInfo-usage-00965
            // VUID-VkImageCreateInfo-Format-02536
            // VUID-VkImageCreateInfo-format-02537
            if extent[0] > device_properties.max_framebuffer_width
                || extent[1] > device_properties.max_framebuffer_height
            {
                return Err(ImageCreationError::MaxFramebufferDimensionsExceeded {
                    extent: [extent[0], extent[1]],
                    max: [
                        device_properties.max_framebuffer_width,
                        device_properties.max_framebuffer_height,
                    ],
                });
            }
        }

        if combined_usage.storage {
            if !format_features.storage_image {
                return Err(ImageCreationError::FormatUsageNotSupported { usage: "storage" });
            }

            // VUID-VkImageCreateInfo-usage-00968
            // VUID-VkImageCreateInfo-format-02538
            if !device.enabled_features().shader_storage_image_multisample
                && samples != SampleCount::Sample1
            {
                return Err(ImageCreationError::RequirementNotMet {
                    required_for: "`create_info.usage.storage` or `create_info.stencil_usage.storage` is set and `create_info.samples` is not `SampleCount::Sample1`",
                    requires_one_of: RequiresOneOf {
                        features: &["shader_storage_image_multisample"],
                        ..Default::default()
                    },
                });
            }
        }

        // These flags only exist in later versions, ignore them otherwise
        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            if combined_usage.transfer_src && !format_features.transfer_src {
                return Err(ImageCreationError::FormatUsageNotSupported {
                    usage: "transfer_src",
                });
            }

            if combined_usage.transfer_dst && !format_features.transfer_dst {
                return Err(ImageCreationError::FormatUsageNotSupported {
                    usage: "transfer_dst",
                });
            }
        }

        if usage.transient_attachment {
            // VUID-VkImageCreateInfo-usage-00966
            assert!(
                usage.color_attachment || usage.depth_stencil_attachment || usage.input_attachment
            );

            // VUID-VkImageCreateInfo-usage-00963
            assert!(ImageUsage {
                transient_attachment: false,
                color_attachment: false,
                depth_stencil_attachment: false,
                input_attachment: false,
                ..usage
            }
            .is_empty())
        }

        if has_separate_stencil_usage {
            // VUID-VkImageCreateInfo-format-02795
            // VUID-VkImageCreateInfo-format-02796
            if usage.depth_stencil_attachment != stencil_usage.depth_stencil_attachment {
                return Err(ImageCreationError::StencilUsageMismatch {
                    usage,
                    stencil_usage,
                });
            }

            // VUID-VkImageCreateInfo-format-02797
            // VUID-VkImageCreateInfo-format-02798
            if usage.transient_attachment != stencil_usage.transient_attachment {
                return Err(ImageCreationError::StencilUsageMismatch {
                    usage,
                    stencil_usage,
                });
            }

            if stencil_usage.transient_attachment {
                // VUID-VkImageStencilUsageCreateInfo-stencilUsage-02539
                assert!(ImageUsage {
                    transient_attachment: false,
                    depth_stencil_attachment: false,
                    input_attachment: false,
                    ..stencil_usage
                }
                .is_empty())
            }
        }

        /* Check flags requirements */

        if cube_compatible {
            // VUID-VkImageCreateInfo-flags-00949
            if image_type != ImageType::Dim2d {
                return Err(ImageCreationError::CubeCompatibleNot2d);
            }

            // VUID-VkImageCreateInfo-imageType-00954
            if extent[0] != extent[1] {
                return Err(ImageCreationError::CubeCompatibleNotSquare);
            }

            // VUID-VkImageCreateInfo-imageType-00954
            if array_layers < 6 {
                return Err(ImageCreationError::CubeCompatibleNotEnoughArrayLayers);
            }
        }

        if array_2d_compatible {
            // VUID-VkImageCreateInfo-flags-00950
            if image_type != ImageType::Dim3d {
                return Err(ImageCreationError::Array2dCompatibleNot3d);
            }
        }

        if block_texel_view_compatible {
            // VUID-VkImageCreateInfo-flags-01572
            if format.compression().is_none() {
                return Err(ImageCreationError::BlockTexelViewCompatibleNotCompressed);
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
                        return Err(ImageCreationError::SharingQueueFamilyIndexOutOfRange {
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
                return Err(ImageCreationError::RequirementNotMet {
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
                return Err(ImageCreationError::ExternalMemoryInvalidInitialLayout);
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
            ImageType::Dim2d if cube_compatible => {
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

            if combined_usage.color_attachment
                && !device_properties
                    .framebuffer_color_sample_counts
                    .contains_count(samples)
            {
                // TODO: how to handle framebuffer_integer_color_sample_counts limit, which only
                // exists >= Vulkan 1.2
                return true;
            }

            if combined_usage.depth_stencil_attachment {
                if aspects.depth
                    && !device_properties
                        .framebuffer_depth_sample_counts
                        .contains_count(samples)
                {
                    return true;
                }

                if aspects.stencil
                    && !device_properties
                        .framebuffer_stencil_sample_counts
                        .contains_count(samples)
                {
                    return true;
                }
            }

            if combined_usage.sampled {
                if let Some(numeric_type) = format.type_color() {
                    match numeric_type {
                        NumericType::UINT | NumericType::SINT => {
                            if !device_properties
                                .sampled_image_integer_sample_counts
                                .contains_count(samples)
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
                                .contains_count(samples)
                            {
                                return true;
                            }
                        }
                    }
                } else {
                    if aspects.depth
                        && !device_properties
                            .sampled_image_depth_sample_counts
                            .contains_count(samples)
                    {
                        return true;
                    }

                    if aspects.stencil
                        && device_properties
                            .sampled_image_stencil_sample_counts
                            .contains_count(samples)
                    {
                        return true;
                    }
                }
            }

            if combined_usage.storage
                && !device_properties
                    .storage_image_sample_counts
                    .contains_count(samples)
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
                    && ImageUsage {
                        transfer_src: false,
                        transfer_dst: false,
                        ..usage
                    }.is_empty())
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
                    external_memory_handle_types.iter().map(Some).collect()
                } else {
                    smallvec![None]
                };

            for external_memory_handle_type in external_memory_handle_types {
                // Use unchecked, because all validation has been done above.
                let image_format_properties = unsafe {
                    device
                        .physical_device()
                        .image_format_properties_unchecked(ImageFormatInfo {
                            format: Some(format),
                            image_type,
                            tiling,
                            usage,
                            mutable_format,
                            cube_compatible,
                            array_2d_compatible,
                            block_texel_view_compatible,
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
                    None => return Err(ImageCreationError::ImageFormatPropertiesNotSupported),
                };

                // VUID-VkImageCreateInfo-extent-02252
                // VUID-VkImageCreateInfo-extent-02253
                // VUID-VkImageCreateInfo-extent-02254
                if extent[0] > max_extent[0]
                    || extent[1] > max_extent[1]
                    || extent[2] > max_extent[2]
                {
                    return Err(ImageCreationError::MaxDimensionsExceeded {
                        extent,
                        max: max_extent,
                    });
                }

                // VUID-VkImageCreateInfo-mipLevels-02255
                if mip_levels > max_mip_levels {
                    return Err(ImageCreationError::MaxMipLevelsExceeded {
                        mip_levels,
                        max: max_mip_levels,
                    });
                }

                // VUID-VkImageCreateInfo-arrayLayers-02256
                if array_layers > max_array_layers {
                    return Err(ImageCreationError::MaxArrayLayersExceeded {
                        array_layers,
                        max: max_array_layers,
                    });
                }

                // VUID-VkImageCreateInfo-samples-02258
                if !sample_counts.contains_count(samples) {
                    return Err(ImageCreationError::SampleCountNotSupported {
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
        create_info: UnsafeImageCreateInfo,
    ) -> Result<Arc<Self>, VulkanError> {
        let &UnsafeImageCreateInfo {
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
            mutable_format,
            cube_compatible,
            array_2d_compatible,
            block_texel_view_compatible,
            _ne: _,
        } = &create_info;

        let aspects = format.map_or_else(Default::default, |format| format.aspects());

        let has_separate_stencil_usage =
            if stencil_usage.is_empty() || !(aspects.depth && aspects.stencil) {
                stencil_usage = usage;
                false
            } else {
                stencil_usage == usage
            };

        let flags = ImageCreateFlags {
            mutable_format,
            cube_compatible,
            array_2d_compatible,
            block_texel_view_compatible,
            ..ImageCreateFlags::empty()
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

    /// Creates a new `UnsafeImage` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Image,
        create_info: UnsafeImageCreateInfo,
    ) -> Arc<Self> {
        let UnsafeImageCreateInfo {
            dimensions,
            format,
            mip_levels,
            samples,
            tiling,
            usage,
            mut stencil_usage,
            sharing: _,
            initial_layout,
            external_memory_handle_types,
            mutable_format,
            cube_compatible,
            array_2d_compatible,
            block_texel_view_compatible,
            _ne: _,
        } = create_info;

        let aspects = format.map_or_else(Default::default, |format| format.aspects());

        if stencil_usage.is_empty() || !(aspects.depth && aspects.stencil) {
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
        let aspects = format.unwrap().aspects();
        let aspect_list: SmallVec<[ImageAspect; 4]> = aspects.iter().collect();
        let mip_level_size = dimensions.array_layers() as DeviceSize;
        let aspect_size = mip_level_size * mip_levels as DeviceSize;
        let range_size = aspect_list.len() as DeviceSize * aspect_size;

        Arc::new(UnsafeImage {
            device,
            handle,

            dimensions,
            format,
            format_features,
            mip_levels,
            initial_layout,
            samples,
            tiling,
            usage,
            stencil_usage,
            external_memory_handle_types,
            mutable_format,
            cube_compatible,
            array_2d_compatible,
            block_texel_view_compatible,

            aspect_list,
            aspect_size,
            mip_level_size,
            needs_destruction: true,
            range_size,
            state: Mutex::new(ImageState::new(range_size, initial_layout)),
        })
    }

    /// Creates an image from a raw handle. The image won't be destroyed.
    ///
    /// This function is for example used at the swapchain's initialization.
    pub(crate) unsafe fn from_raw(
        device: Arc<Device>,
        handle: ash::vk::Image,
        usage: ImageUsage,
        format: Format,
        flags: ImageCreateFlags,
        dimensions: ImageDimensions,
        samples: SampleCount,
        mip_levels: u32,
    ) -> Arc<UnsafeImage> {
        let tiling = ImageTiling::Optimal;

        // Use unchecked, the parameters are assumed to match the info of the swapchain, and
        // therefore already valid.
        let format_features = device
            .physical_device()
            .format_properties_unchecked(format)
            .optimal_tiling_features;

        assert!(ImageCreateFlags {
            mutable_format: false,
            cube_compatible: false,
            array_2d_compatible: false,
            block_texel_view_compatible: false,
            ..ImageCreateFlags::empty()
        }
        .is_empty());

        // TODO: check that usage is correct in regard to `output`?

        let aspects = format.aspects();
        let initial_layout = ImageLayout::Undefined; // TODO: Maybe this should be passed in?

        let aspect_list: SmallVec<[ImageAspect; 4]> = aspects.iter().collect();
        let mip_level_size = dimensions.array_layers() as DeviceSize;
        let aspect_size = mip_level_size * mip_levels as DeviceSize;
        let range_size = aspect_list.len() as DeviceSize * aspect_size;

        let image = UnsafeImage {
            handle,
            device,

            dimensions,
            format: Some(format),
            format_features,
            initial_layout,
            mip_levels,
            samples,
            tiling,
            usage,
            stencil_usage: usage,
            external_memory_handle_types: ExternalMemoryHandleTypes::empty(),
            mutable_format: flags.mutable_format,
            cube_compatible: flags.cube_compatible,
            array_2d_compatible: flags.array_2d_compatible,
            block_texel_view_compatible: flags.block_texel_view_compatible,

            aspect_list,
            aspect_size,
            mip_level_size,
            needs_destruction: false, // TODO: pass as parameter
            range_size,
            state: Mutex::new(ImageState::new(range_size, initial_layout)),
        };

        Arc::new(image)
    }

    /// Returns the memory requirements for this image.
    #[inline]
    pub fn memory_requirements(&self) -> MemoryRequirements {
        let image_memory_requirements_info2 = ash::vk::ImageMemoryRequirementsInfo2 {
            image: self.handle,
            ..Default::default()
        };
        let mut memory_requirements2 = ash::vk::MemoryRequirements2::default();

        let mut memory_dedicated_requirements = if self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_dedicated_allocation
        {
            Some(ash::vk::MemoryDedicatedRequirements::default())
        } else {
            None
        };

        if let Some(next) = memory_dedicated_requirements.as_mut() {
            next.p_next = memory_requirements2.p_next;
            memory_requirements2.p_next = next as *mut _ as *mut _;
        }

        unsafe {
            let fns = self.device.fns();

            if self.device.api_version() >= Version::V1_1
                || self
                    .device
                    .enabled_extensions()
                    .khr_get_memory_requirements2
            {
                if self.device.api_version() >= Version::V1_1 {
                    (fns.v1_1.get_image_memory_requirements2)(
                        self.device.handle(),
                        &image_memory_requirements_info2,
                        &mut memory_requirements2,
                    );
                } else {
                    (fns.khr_get_memory_requirements2
                        .get_image_memory_requirements2_khr)(
                        self.device.handle(),
                        &image_memory_requirements_info2,
                        &mut memory_requirements2,
                    );
                }
            } else {
                (fns.v1_0.get_image_memory_requirements)(
                    self.device.handle(),
                    self.handle,
                    &mut memory_requirements2.memory_requirements,
                );
            }
        }

        MemoryRequirements {
            prefer_dedicated: memory_dedicated_requirements
                .map_or(false, |dreqs| dreqs.prefers_dedicated_allocation != 0),
            ..MemoryRequirements::from(memory_requirements2.memory_requirements)
        }
    }

    /// Returns the sparse memory requirements for this image.
    #[inline]
    pub fn sparse_memory_requirements(&self) -> Vec<SparseImageMemoryRequirements> {
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

    #[inline]
    pub unsafe fn bind_memory(
        &self,
        memory: &DeviceMemory,
        offset: DeviceSize,
    ) -> Result<(), OomError> {
        let fns = self.device.fns();

        // We check for correctness in debug mode.
        debug_assert!({
            let mut mem_reqs = MaybeUninit::uninit();
            (fns.v1_0.get_image_memory_requirements)(
                self.device.handle(),
                self.handle,
                mem_reqs.as_mut_ptr(),
            );

            let mem_reqs = mem_reqs.assume_init();
            mem_reqs.size <= memory.allocation_size() - offset
                && offset % mem_reqs.alignment == 0
                && mem_reqs.memory_type_bits & (1 << memory.memory_type_index()) != 0
        });

        (fns.v1_0.bind_image_memory)(self.device.handle(), self.handle, memory.handle(), offset)
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
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
            .contains(&subresource_range.aspects));
        assert!(subresource_range.mip_levels.end <= self.mip_levels);
        assert!(subresource_range.array_layers.end <= self.dimensions.array_layers());

        SubresourceRangeIterator::new(
            subresource_range,
            &self.aspect_list,
            self.aspect_size,
            self.mip_levels,
            self.mip_level_size,
            self.dimensions.array_layers(),
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
                mip_levels: 0..self.mip_levels,
                array_layers: 0..self.dimensions.array_layers(),
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
                    array_layers: 0..self.dimensions.array_layers(),
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
    pub fn format_features(&self) -> &FormatFeatures {
        &self.format_features
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
    pub fn usage(&self) -> &ImageUsage {
        &self.usage
    }

    /// Returns the stencil usage the image was created with.
    #[inline]
    pub fn stencil_usage(&self) -> &ImageUsage {
        &self.stencil_usage
    }

    /// Returns the external memory handle types that are supported with this image.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.external_memory_handle_types
    }

    /// Returns whether `mutable_format` is enabled on the image.
    #[inline]
    pub fn mutable_format(&self) -> bool {
        self.mutable_format
    }

    /// Returns whether `cube_compatible` is enabled on the image.
    #[inline]
    pub fn cube_compatible(&self) -> bool {
        self.cube_compatible
    }

    /// Returns whether `array_2d_compatible` is enabled on the image.
    #[inline]
    pub fn array_2d_compatible(&self) -> bool {
        self.array_2d_compatible
    }

    /// Returns whether `block_texel_view_compatible` is enabled on the image.
    #[inline]
    pub fn block_texel_view_compatible(&self) -> bool {
        self.block_texel_view_compatible
    }

    /// Returns an `ImageSubresourceLayers` covering the first mip level of the image. All aspects
    /// of the image are selected, or `plane0` if the image is multi-planar.
    #[inline]
    pub fn subresource_layers(&self) -> ImageSubresourceLayers {
        ImageSubresourceLayers {
            aspects: {
                let aspects = self.format.unwrap().aspects();

                if aspects.plane0 {
                    ImageAspects {
                        plane0: true,
                        ..ImageAspects::empty()
                    }
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
            aspects: ImageAspects {
                plane0: false,
                plane1: false,
                plane2: false,
                ..self.format.unwrap().aspects()
            },
            mip_levels: 0..self.mip_levels,
            array_layers: 0..self.dimensions.array_layers(),
        }
    }

    /// Returns a key unique to each `UnsafeImage`. Can be used for the `conflicts_key` method.
    #[inline]
    pub fn key(&self) -> u64 {
        self.handle.as_raw()
    }

    /// Queries the layout of an image in memory. Only valid for images with linear tiling.
    ///
    /// This function is only valid for images with a color format. See the other similar functions
    /// for the other aspects.
    ///
    /// The layout is invariant for each image. However it is not cached, as this would waste
    /// memory in the case of non-linear-tiling images. You are encouraged to store the layout
    /// somewhere in order to avoid calling this semi-expensive function at every single memory
    /// access.
    ///
    /// Note that while Vulkan allows querying the array layers other than 0, it is redundant as
    /// you can easily calculate the position of any layer.
    ///
    /// # Panics
    ///
    /// - Panics if the mipmap level is out of range.
    ///
    /// # Safety
    ///
    /// - The image must *not* have a depth, stencil or depth-stencil format.
    /// - The image must have been created with linear tiling.
    ///
    #[inline]
    pub unsafe fn color_linear_layout(&self, mip_level: u32) -> LinearLayout {
        self.linear_layout_impl(mip_level, ImageAspect::Color)
    }

    /// Same as `color_linear_layout`, except that it retrieves the depth component of the image.
    ///
    /// # Panics
    ///
    /// - Panics if the mipmap level is out of range.
    ///
    /// # Safety
    ///
    /// - The image must have a depth or depth-stencil format.
    /// - The image must have been created with linear tiling.
    ///
    #[inline]
    pub unsafe fn depth_linear_layout(&self, mip_level: u32) -> LinearLayout {
        self.linear_layout_impl(mip_level, ImageAspect::Depth)
    }

    /// Same as `color_linear_layout`, except that it retrieves the stencil component of the image.
    ///
    /// # Panics
    ///
    /// - Panics if the mipmap level is out of range.
    ///
    /// # Safety
    ///
    /// - The image must have a stencil or depth-stencil format.
    /// - The image must have been created with linear tiling.
    ///
    #[inline]
    pub unsafe fn stencil_linear_layout(&self, mip_level: u32) -> LinearLayout {
        self.linear_layout_impl(mip_level, ImageAspect::Stencil)
    }

    /// Same as `color_linear_layout`, except that it retrieves layout for the requested YCbCr
    /// component too if the format is a YCbCr format.
    ///
    /// # Panics
    ///
    /// - Panics if plane aspect is out of range.
    /// - Panics if the aspect is not a color or planar aspect.
    /// - Panics if the number of mipmaps is not 1.
    #[inline]
    pub unsafe fn multiplane_color_layout(&self, aspect: ImageAspect) -> LinearLayout {
        // This function only supports color and planar aspects currently.
        assert!(matches!(
            aspect,
            ImageAspect::Color | ImageAspect::Plane0 | ImageAspect::Plane1 | ImageAspect::Plane2
        ));
        assert!(self.mip_levels == 1);

        if matches!(
            aspect,
            ImageAspect::Plane0 | ImageAspect::Plane1 | ImageAspect::Plane2
        ) {
            debug_assert!(self.format.unwrap().ycbcr_chroma_sampling().is_some());
        }

        self.linear_layout_impl(0, aspect)
    }

    // Implementation of the `*_layout` functions.
    unsafe fn linear_layout_impl(&self, mip_level: u32, aspect: ImageAspect) -> LinearLayout {
        let fns = self.device.fns();

        assert!(mip_level < self.mip_levels);

        let subresource = ash::vk::ImageSubresource {
            aspect_mask: ash::vk::ImageAspectFlags::from(aspect),
            mip_level,
            array_layer: 0,
        };

        let mut out = MaybeUninit::uninit();
        (fns.v1_0.get_image_subresource_layout)(
            self.device.handle(),
            self.handle,
            &subresource,
            out.as_mut_ptr(),
        );

        let out = out.assume_init();
        LinearLayout {
            offset: out.offset,
            size: out.size,
            row_pitch: out.row_pitch,
            array_pitch: out.array_pitch,
            depth_pitch: out.depth_pitch,
        }
    }
}

impl Drop for UnsafeImage {
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

unsafe impl VulkanObject for UnsafeImage {
    type Handle = ash::vk::Image;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for UnsafeImage {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for UnsafeImage {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for UnsafeImage {}

impl Hash for UnsafeImage {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Parameters to create a new `UnsafeImage`.
#[derive(Clone, Debug)]
pub struct UnsafeImageCreateInfo {
    /// The type, extent and number of array layers to create the image with.
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

    /// For non-multi-planar formats, whether an image view wrapping the image can have a
    /// different format.
    ///
    /// For multi-planar formats, whether an image view wrapping the image can be created from a
    /// single plane of the image.
    ///
    /// The default value is `false`.
    pub mutable_format: bool,

    /// For 2D images, whether an image view of type
    /// [`ImageViewType::Cube`](crate::image::view::ImageViewType::Cube) or
    /// [`ImageViewType::CubeArray`](crate::image::view::ImageViewType::CubeArray) can be created
    /// from the image.
    ///
    /// The default value is `false`.
    pub cube_compatible: bool,

    /// For 3D images, whether an image view of type
    /// [`ImageViewType::Dim2d`](crate::image::view::ImageViewType::Dim2d) or
    /// [`ImageViewType::Dim2dArray`](crate::image::view::ImageViewType::Dim2dArray) can be created
    /// from the image.
    ///
    /// The default value is `false`.
    pub array_2d_compatible: bool,

    /// For images with a compressed format, whether an image view with an uncompressed
    /// format can be created from the image, where each texel in the view will correspond to a
    /// compressed texel block in the image.
    ///
    /// Requires `mutable_format`.
    ///
    /// The default value is `false`.
    pub block_texel_view_compatible: bool,

    pub _ne: crate::NonExhaustive,
}

impl Default for UnsafeImageCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
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
            mutable_format: false,
            cube_compatible: false,
            array_2d_compatible: false,
            block_texel_view_compatible: false,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Error that can happen when creating an image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageCreationError {
    /// Allocating memory failed.
    AllocError(AllocationCreationError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The array_2d_compatible flag was enabled, but the image type was not 3D.
    Array2dCompatibleNot3d,

    /// The block_texel_view_compatible flag was enabled, but the given format was not compressed.
    BlockTexelViewCompatibleNotCompressed,

    /// The cube_compatible flag was enabled, but the image type was not 2D.
    CubeCompatibleNot2d,

    /// The cube_compatible flag was enabled, but the number of array layers was less than 6.
    CubeCompatibleNotEnoughArrayLayers,

    /// The cube_compatible flag was enabled, but the image dimensions were not square.
    CubeCompatibleNotSquare,

    /// The cube_compatible flag was enabled together with multisampling.
    CubeCompatibleMultisampling,

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

    /// Multisampling was enabled, and the `cube_compatible` flag was set.
    MultisampleCubeCompatible,

    /// Multisampling was enabled, and tiling was `Linear`.
    MultisampleLinearTiling,

    /// Multisampling was enabled, and multiple mip levels were specified.
    MultisampleMultipleMipLevels,

    /// Multisampling was enabled, but the image type was not 2D.
    MultisampleNot2d,

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

impl Error for ImageCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ImageCreationError::AllocError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for ImageCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::AllocError(_) => write!(f, "allocating memory failed"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::Array2dCompatibleNot3d => write!(
                f,
                "the array_2d_compatible flag was enabled, but the image type was not 3D",
            ),
            Self::BlockTexelViewCompatibleNotCompressed => write!(
                f,
                "the block_texel_view_compatible flag was enabled, but the given format was not \
                compressed",
            ),
            Self::CubeCompatibleNot2d => write!(
                f,
                "the cube_compatible flag was enabled, but the image type was not 2D",
            ),
            Self::CubeCompatibleNotEnoughArrayLayers => write!(
                f,
                "the cube_compatible flag was enabled, but the number of array layers was less \
                than 6",
            ),
            Self::CubeCompatibleNotSquare => write!(
                f,
                "the cube_compatible flag was enabled, but the image dimensions were not square",
            ),
            Self::CubeCompatibleMultisampling => write!(
                f,
                "the cube_compatible flag was enabled together with multisampling",
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

impl From<OomError> for ImageCreationError {
    fn from(err: OomError) -> Self {
        Self::AllocError(err.into())
    }
}

impl From<AllocationCreationError> for ImageCreationError {
    fn from(err: AllocationCreationError) -> Self {
        Self::AllocError(err)
    }
}

impl From<VulkanError> for ImageCreationError {
    fn from(err: VulkanError) -> Self {
        match err {
            VulkanError::OutOfHostMemory => {
                Self::AllocError(AllocationCreationError::OutOfHostMemory)
            }
            VulkanError::OutOfDeviceMemory => {
                Self::AllocError(AllocationCreationError::OutOfDeviceMemory)
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<RequirementNotMet> for ImageCreationError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

/// Describes the memory layout of an image with linear tiling.
///
/// Obtained by calling `*_linear_layout` on the image.
///
/// The address of a texel at `(x, y, z, layer)` is `layer * array_pitch + z * depth_pitch +
/// y * row_pitch + x * size_of_each_texel + offset`. `size_of_each_texel` must be determined
/// depending on the format. The same formula applies for compressed formats, except that the
/// coordinates must be in number of blocks.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LinearLayout {
    /// Number of bytes from the start of the memory and the start of the queried subresource.
    pub offset: DeviceSize,
    /// Total number of bytes for the queried subresource. Can be used for a safety check.
    pub size: DeviceSize,
    /// Number of bytes between two texels or two blocks in adjacent rows.
    pub row_pitch: DeviceSize,
    /// Number of bytes between two texels or two blocks in adjacent array layers. This value is
    /// undefined for images with only one array layer.
    pub array_pitch: DeviceSize,
    /// Number of bytes between two texels or two blocks in adjacent depth layers. This value is
    /// undefined for images that are not three-dimensional.
    pub depth_pitch: DeviceSize,
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
    pub(crate) fn check_cpu_read(&mut self, range: Range<DeviceSize>) -> Result<(), ReadLockError> {
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
    pub(crate) fn check_cpu_write(
        &mut self,
        range: Range<DeviceSize>,
    ) -> Result<(), WriteLockError> {
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
        &mut self,
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
        &mut self,
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
            .iter()
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

#[cfg(test)]
mod tests {
    use super::{ImageCreationError, ImageUsage, UnsafeImage, UnsafeImageCreateInfo};
    use crate::{
        format::Format,
        image::{
            sys::SubresourceRangeIterator, ImageAspect, ImageAspects, ImageDimensions,
            ImageSubresourceRange, SampleCount,
        },
        DeviceSize, RequiresOneOf,
    };
    use smallvec::SmallVec;

    #[test]
    fn create_sampled() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = UnsafeImage::new(
            device,
            UnsafeImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                format: Some(Format::R8G8B8A8_UNORM),
                usage: ImageUsage {
                    sampled: true,
                    ..ImageUsage::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn create_transient() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = UnsafeImage::new(
            device,
            UnsafeImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                format: Some(Format::R8G8B8A8_UNORM),
                usage: ImageUsage {
                    transient_attachment: true,
                    color_attachment: true,
                    ..ImageUsage::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn zero_mipmap() {
        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            let _ = UnsafeImage::new(
                device,
                UnsafeImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 32,
                        height: 32,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    mip_levels: 0,
                    usage: ImageUsage {
                        sampled: true,
                        ..ImageUsage::empty()
                    },
                    ..Default::default()
                },
            );
        });
    }

    #[test]
    fn mipmaps_too_high() {
        let (device, _) = gfx_dev_and_queue!();

        let res = UnsafeImage::new(
            device,
            UnsafeImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                format: Some(Format::R8G8B8A8_UNORM),
                mip_levels: u32::MAX,
                usage: ImageUsage {
                    sampled: true,
                    ..ImageUsage::empty()
                },
                ..Default::default()
            },
        );

        match res {
            Err(ImageCreationError::MaxMipLevelsExceeded { .. }) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn shader_storage_image_multisample() {
        let (device, _) = gfx_dev_and_queue!();

        let res = UnsafeImage::new(
            device,
            UnsafeImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                format: Some(Format::R8G8B8A8_UNORM),
                samples: SampleCount::Sample2,
                usage: ImageUsage {
                    storage: true,
                    ..ImageUsage::empty()
                },
                ..Default::default()
            },
        );

        match res {
            Err(ImageCreationError::RequirementNotMet {
                requires_one_of: RequiresOneOf { features, .. },
                ..
            }) if features.contains(&"shader_storage_image_multisample") => (),
            Err(ImageCreationError::SampleCountNotSupported { .. }) => (), // unlikely but possible
            _ => panic!(),
        };
    }

    #[test]
    fn compressed_not_color_attachment() {
        let (device, _) = gfx_dev_and_queue!();

        let res = UnsafeImage::new(
            device,
            UnsafeImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                format: Some(Format::ASTC_5x4_UNORM_BLOCK),
                usage: ImageUsage {
                    color_attachment: true,
                    ..ImageUsage::empty()
                },
                ..Default::default()
            },
        );

        match res {
            Err(ImageCreationError::FormatNotSupported) => (),
            Err(ImageCreationError::FormatUsageNotSupported {
                usage: "color_attachment",
            }) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn transient_forbidden_with_some_usages() {
        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            let _ = UnsafeImage::new(
                device,
                UnsafeImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 32,
                        height: 32,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    usage: ImageUsage {
                        transient_attachment: true,
                        sampled: true,
                        ..ImageUsage::empty()
                    },
                    ..Default::default()
                },
            );
        })
    }

    #[test]
    fn cubecompatible_dims_mismatch() {
        let (device, _) = gfx_dev_and_queue!();

        let res = UnsafeImage::new(
            device,
            UnsafeImageCreateInfo {
                dimensions: ImageDimensions::Dim2d {
                    width: 32,
                    height: 64,
                    array_layers: 1,
                },
                format: Some(Format::R8G8B8A8_UNORM),
                usage: ImageUsage {
                    sampled: true,
                    ..ImageUsage::empty()
                },
                cube_compatible: true,
                ..Default::default()
            },
        );

        match res {
            Err(ImageCreationError::CubeCompatibleNotEnoughArrayLayers) => (),
            Err(ImageCreationError::CubeCompatibleNotSquare) => (),
            _ => panic!(),
        };
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn subresource_range_iterator() {
        // A fictitious set of aspects that no real image would actually ever have.
        let image_aspect_list: SmallVec<[ImageAspect; 4]> = ImageAspects {
            color: true,
            depth: true,
            stencil: true,
            plane0: true,
            ..ImageAspects::empty()
        }
        .iter()
        .collect();
        let image_mip_levels = 6;
        let image_array_layers = 8;

        let mip = image_array_layers as DeviceSize;
        let asp = mip * image_mip_levels as DeviceSize;

        // Whole image
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects {
                    color: true,
                    depth: true,
                    stencil: true,
                    plane0: true,
                    ..ImageAspects::empty()
                },
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
                aspects: ImageAspects {
                    color: true,
                    depth: true,
                    stencil: false,
                    plane0: true,
                    ..ImageAspects::empty()
                },
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
                aspects: ImageAspects {
                    color: false,
                    depth: true,
                    stencil: true,
                    plane0: false,
                    ..ImageAspects::empty()
                },
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
                aspects: ImageAspects {
                    color: true,
                    depth: false,
                    stencil: false,
                    plane0: false,
                    ..ImageAspects::empty()
                },
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
                aspects: ImageAspects {
                    color: false,
                    depth: true,
                    stencil: true,
                    plane0: false,
                    ..ImageAspects::empty()
                },
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
