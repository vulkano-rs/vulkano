// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub(super) mod bind_push;
pub(super) mod debug;
pub(super) mod dynamic_state;
pub(super) mod image;
pub(super) mod pipeline;
pub(super) mod query;
pub(super) mod render_pass;
pub(super) mod secondary;
pub(super) mod sync;
pub(super) mod transfer;

use super::synced::SyncCommandBufferBuilderError;
use crate::{
    format::Format,
    image::{ImageAspects, ImageLayout, SampleCount, SampleCounts},
    DeviceSize,
};
use std::{error, fmt};

/// Error that can happen when recording a copy command.
#[derive(Clone, Debug)]
pub enum CopyError {
    SyncCommandBufferBuilderError(SyncCommandBufferBuilderError),

    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },

    /// Operation forbidden inside of a render pass.
    ForbiddenInsideRenderPass,

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,

    /// The array layer counts of the source and destination subresource ranges of a region do not
    /// match.
    ArrayLayerCountMismatch {
        region_index: usize,
        src_layer_count: u32,
        dst_layer_count: u32,
    },

    /// The end of the range of accessed array layers of the subresource range of a region is
    /// greater than the number of array layers in the image.
    ArrayLayersOutOfRange {
        resource: CopyErrorResource,
        region_index: usize,
        array_layers_range_end: u32,
        image_array_layers: u32,
    },

    /// The aspects of the source and destination subresource ranges of a region do not match.
    AspectsMismatch {
        region_index: usize,
        src_aspects: ImageAspects,
        dst_aspects: ImageAspects,
    },

    /// The aspects of the subresource range of a region contain aspects that are not present
    /// in the image, or that are not allowed.
    AspectsNotAllowed {
        resource: CopyErrorResource,
        region_index: usize,
        aspects: ImageAspects,
        allowed_aspects: ImageAspects,
    },

    /// The buffer image height of a region is not a multiple of the required buffer alignment.
    BufferImageHeightNotAligned {
        resource: CopyErrorResource,
        region_index: usize,
        image_height: u32,
        required_alignment: u32,
    },

    /// The buffer image height of a region is smaller than the image extent height.
    BufferImageHeightTooSmall {
        resource: CopyErrorResource,
        region_index: usize,
        image_height: u32,
        min: u32,
    },

    /// The buffer row length of a region is not a multiple of the required buffer alignment.
    BufferRowLengthNotAligned {
        resource: CopyErrorResource,
        region_index: usize,
        row_length: u32,
        required_alignment: u32,
    },

    /// The buffer row length of a region specifies a row of texels that is greater than 0x7FFFFFFF
    /// bytes in size.
    BufferRowLengthTooLarge {
        resource: CopyErrorResource,
        region_index: usize,
        buffer_row_length: u32,
    },

    /// The buffer row length of a region is smaller than the image extent width.
    BufferRowLengthTooSmall {
        resource: CopyErrorResource,
        region_index: usize,
        row_length: u32,
        min: u32,
    },

    /// The provided data has a size larger than the maximum allowed.
    DataTooLarge {
        size: DeviceSize,
        max: DeviceSize,
    },

    /// Depth/stencil images are not supported by the queue family of this command buffer; a graphics queue family is required.
    DepthStencilNotSupportedByQueueFamily,

    /// The image extent of a region is not a multiple of the required image alignment.
    ExtentNotAlignedForImage {
        resource: CopyErrorResource,
        region_index: usize,
        extent: [u32; 3],
        required_alignment: [u32; 3],
    },

    /// The chosen filter type does not support the dimensionality of the source image.
    FilterNotSupportedForImageType,

    /// The chosen filter type does not support the format of the source image.
    FilterNotSupportedByFormat,

    /// The format of an image is not supported for this operation.
    FormatNotSupported {
        resource: CopyErrorResource,
        format: Format,
    },

    /// The format of the source image does not match the format of the destination image.
    FormatsMismatch {
        src_format: Format,
        dst_format: Format,
    },

    /// The format of the source image subresource is not compatible with the format of the
    /// destination image subresource.
    FormatsNotCompatible {
        src_format: Format,
        dst_format: Format,
    },

    /// A specified image layout is not valid for this operation.
    ImageLayoutInvalid {
        resource: CopyErrorResource,
        image_layout: ImageLayout,
    },

    /// The end of the range of accessed mip levels of the subresource range of a region is greater
    /// than the number of mip levels in the image.
    MipLevelsOutOfRange {
        resource: CopyErrorResource,
        region_index: usize,
        mip_levels_range_end: u32,
        image_mip_levels: u32,
    },

    /// An image does not have a required format feature.
    MissingFormatFeature {
        resource: CopyErrorResource,
        format_feature: &'static str,
    },

    /// A resource did not have a required usage enabled.
    MissingUsage {
        resource: CopyErrorResource,
        usage: &'static str,
    },

    /// A subresource range of a region specifies multiple aspects, but only one aspect can be
    /// selected for the image.
    MultipleAspectsNotAllowed {
        resource: CopyErrorResource,
        region_index: usize,
        aspects: ImageAspects,
    },

    /// The buffer offset of a region is not a multiple of the required buffer alignment.
    OffsetNotAlignedForBuffer {
        resource: CopyErrorResource,
        region_index: usize,
        offset: DeviceSize,
        required_alignment: DeviceSize,
    },

    /// The image offset of a region is not a multiple of the required image alignment.
    OffsetNotAlignedForImage {
        resource: CopyErrorResource,
        region_index: usize,
        offset: [u32; 3],
        required_alignment: [u32; 3],
    },

    /// The image offsets of a region are not the values required for that axis ([0, 1]) for the
    /// type of the image.
    OffsetsInvalidForImageType {
        resource: CopyErrorResource,
        region_index: usize,
        offsets: [u32; 2],
    },

    /// The source bounds of a region overlap with the destination bounds of a region.
    OverlappingRegions {
        src_region_index: usize,
        dst_region_index: usize,
    },

    /// The source subresources of a region overlap with the destination subresources of a region,
    /// but the source image layout does not equal the destination image layout.
    OverlappingSubresourcesLayoutMismatch {
        src_region_index: usize,
        dst_region_index: usize,
        src_image_layout: ImageLayout,
        dst_image_layout: ImageLayout,
    },

    /// The end of the range of accessed byte offsets of a region is greater than the size of the
    /// buffer.
    RegionOutOfBufferBounds {
        resource: CopyErrorResource,
        region_index: usize,
        offset_range_end: DeviceSize,
        buffer_size: DeviceSize,
    },

    /// The end of the range of accessed texel offsets of a region is greater than the extent of
    /// the selected subresource of the image.
    RegionOutOfImageBounds {
        resource: CopyErrorResource,
        region_index: usize,
        offset_range_end: [u32; 3],
        subresource_extent: [u32; 3],
    },

    /// An image has a sample count that is not valid for this operation.
    SampleCountInvalid {
        resource: CopyErrorResource,
        sample_count: SampleCount,
        allowed_sample_counts: SampleCounts,
    },

    /// The source image has a different sample count than the destination image.
    SampleCountMismatch {
        src_sample_count: SampleCount,
        dst_sample_count: SampleCount,
    },

    /// The buffer size of a region is not a multiple of the required buffer alignment.
    SizeNotAlignedForBuffer {
        resource: CopyErrorResource,
        region_index: usize,
        size: DeviceSize,
        required_alignment: DeviceSize,
    },
}

impl error::Error for CopyError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::SyncCommandBufferBuilderError(err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for CopyError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::SyncCommandBufferBuilderError(_) => write!(f, "a SyncCommandBufferBuilderError"),
            Self::ExtensionNotEnabled { extension, reason } => write!(
                f,
                "the extension {} must be enabled: {}",
                extension, reason
            ),
            Self::ForbiddenInsideRenderPass => {
                write!(f, "operation forbidden inside of a render pass")
            }
            Self::NotSupportedByQueueFamily => {
                write!(f, "the queue family doesn't allow this operation")
            }

            Self::ArrayLayerCountMismatch {
                region_index,
                src_layer_count,
                dst_layer_count,
            } => write!(
                f,
                "the array layer counts of the source and destination subresource ranges of region {} do not match (source: {}; destination: {})",
                region_index, src_layer_count, dst_layer_count,
            ),
            Self::ArrayLayersOutOfRange {
                resource,
                region_index,
                array_layers_range_end,
                image_array_layers,
            } => write!(
                f,
                "the end of the range of accessed array layers ({}) of the {} subresource range of region {} is greater than the number of array layers in the {} image ({})",
                array_layers_range_end, resource, region_index, resource, image_array_layers,
            ),
            Self::AspectsMismatch {
                region_index,
                src_aspects,
                dst_aspects,
            } => write!(
                f,
                "the aspects of the source and destination subresource ranges of region {} do not match (source: {:?}; destination: {:?})",
                region_index, src_aspects, dst_aspects,
            ),
            Self::AspectsNotAllowed {
                resource,
                region_index,
                aspects,
                allowed_aspects,
            } => write!(
                f,
                "the aspects ({:?}) of the {} subresource range of region {} contain aspects that are not present in the {} image, or that are not allowed ({:?})",
                aspects, resource, region_index, resource, allowed_aspects,
            ),
            Self::BufferImageHeightNotAligned {
                resource,
                region_index,
                image_height,
                required_alignment,
            } => write!(
                f,
                "the {} buffer image height ({}) of region {} is not a multiple of the required {} buffer alignment ({})",
                resource, image_height, region_index, resource, required_alignment,
            ),
            Self::BufferRowLengthTooLarge {
                resource,
                region_index,
                buffer_row_length,
            } => write!(
                f,
                "the {} buffer row length ({}) of region {} specifies a row of texels that is greater than 0x7FFFFFFF bytes in size",
                resource, buffer_row_length, region_index,
            ),
            Self::BufferImageHeightTooSmall {
                resource,
                region_index,
                image_height,
                min,
            } => write!(
                f,
                "the {} buffer image height ({}) of region {} is smaller than the {} image extent height ({})",
                resource, image_height, region_index, resource, min,
            ),
            Self::BufferRowLengthNotAligned {
                resource,
                region_index,
                row_length,
                required_alignment,
            } => write!(
                f,
                "the {} buffer row length ({}) of region {} is not a multiple of the required {} buffer alignment ({})",
                resource, row_length, region_index, resource, required_alignment,
            ),
            Self::BufferRowLengthTooSmall {
                resource,
                region_index,
                row_length,
                min,
            } => write!(
                f,
                "the {} buffer row length length ({}) of region {} is smaller than the {} image extent width ({})",
                resource, row_length, region_index, resource, min,
            ),
            Self::DataTooLarge {
                size,
                max,
            } => write!(
                f,
                "the provided data has a size ({}) greater than the maximum allowed ({})",
                size, max,
            ),
            Self::DepthStencilNotSupportedByQueueFamily => write!(
                f,
                "depth/stencil images are not supported by the queue family of this command buffer; a graphics queue family is required",
            ),
            Self::ExtentNotAlignedForImage {
                resource,
                region_index,
                extent,
                required_alignment,
            } => write!(
                f,
                "the {} image extent ({:?}) of region {} is not a multiple of the required {} image alignment ({:?})",
                resource, extent, region_index, resource, required_alignment,
            ),
            Self::FilterNotSupportedForImageType => write!(
                f,
                "the chosen filter is not supported for the source image type"
            ),
            Self::FilterNotSupportedByFormat => write!(
                f,
                "the chosen filter is not supported by the format of the source image"
            ),
            Self::FormatNotSupported {
                resource,
                format,
            } => write!(
                f,
                "the format of the {} image ({:?}) is not supported for this operation",
                resource, format,
            ),
            Self::FormatsMismatch {
                src_format,
                dst_format,
            } => write!(
                f,
                "the format of the source image ({:?}) does not match the format of the destination image ({:?})",
                src_format, dst_format,
            ),
            Self::FormatsNotCompatible {
                src_format,
                dst_format,
            } => write!(
                f,
                "the format of the source image subresource ({:?}) is not compatible with the format of the destination image subresource ({:?})",
                src_format, dst_format,
            ),
            Self::ImageLayoutInvalid {
                resource,
                image_layout,
            } => write!(
                f,
                "the specified {} image layout {:?} is not valid for this operation",
                resource, image_layout,
            ),
            Self::MipLevelsOutOfRange {
                resource,
                region_index,
                mip_levels_range_end,
                image_mip_levels,
            } => write!(
                f,
                "the end of the range of accessed mip levels ({}) of the {} subresource range of region {} is not less than the number of mip levels in the {} image ({})",
                mip_levels_range_end, resource, region_index, resource, image_mip_levels,
            ),
            Self::MissingFormatFeature {
                resource,
                format_feature,
            } => write!(
                f,
                "the {} image does not have the required format feature {}",
                resource, format_feature,
            ),
            Self::MissingUsage { resource, usage } => write!(
                f,
                "the {} resource did not have the required usage {} enabled",
                resource, usage,
            ),
            Self::MultipleAspectsNotAllowed {
                resource,
                region_index,
                aspects,
            } => write!(
                f,
                "the {} subresource range of region {} specifies multiple aspects ({:?}), but only one aspect can be selected for the {} image",
                resource, region_index, aspects, resource,
            ),
            Self::OffsetNotAlignedForBuffer {
                resource,
                region_index,
                offset,
                required_alignment,
            } => write!(
                f,
                "the {} buffer offset ({}) of region {} is not a multiple of the required {} buffer alignment ({})",
                resource, offset, region_index, resource, required_alignment,
            ),
            Self::OffsetNotAlignedForImage {
                resource,
                region_index,
                offset,
                required_alignment,
            } => write!(
                f,
                "the {} image offset ({:?}) of region {} is not a multiple of the required {} image alignment ({:?})",
                resource, offset, region_index, resource, required_alignment,
            ),
            Self::OffsetsInvalidForImageType {
                resource,
                region_index,
                offsets,
            } => write!(
                f,
                "the {} image offsets ({:?}) of region {} are not the values required for that axis ([0, 1]) for the type of the {} image",
                resource, offsets, region_index, resource,
            ),
            Self::OverlappingRegions {
                src_region_index,
                dst_region_index,
            } => write!(
                f,
                "the source bounds of region {} overlap with the destination bounds of region {}",
                src_region_index, dst_region_index,
            ),
            Self::OverlappingSubresourcesLayoutMismatch {
                src_region_index,
                dst_region_index,
                src_image_layout,
                dst_image_layout,
            } => write!(
                f,
                "the source subresources of region {} overlap with the destination subresources of region {}, but the source image layout ({:?}) does not equal the destination image layout ({:?})",
                src_region_index, dst_region_index, src_image_layout, dst_image_layout,
            ),
            Self::RegionOutOfBufferBounds {
                resource,
                region_index,
                offset_range_end,
                buffer_size,
            } => write!(
                f,
                "the end of the range of accessed {} byte offsets ({}) of region {} is greater than the size of the {} buffer ({})",
                resource, offset_range_end, region_index, resource, buffer_size,
            ),
            Self::RegionOutOfImageBounds {
                resource,
                region_index,
                offset_range_end,
                subresource_extent,
            } => write!(
                f,
                "the end of the range of accessed {} texel offsets ({:?}) of region {} is greater than the extent of the selected subresource of the {} image ({:?})",
                resource, offset_range_end, region_index, resource, subresource_extent,
            ),
            Self::SampleCountInvalid {
                resource,
                sample_count,
                allowed_sample_counts,
            } => write!(
                f,
                "the {} image has a sample count ({:?}) that is not valid for this operation ({:?})",
                resource, sample_count, allowed_sample_counts,
            ),
            Self::SampleCountMismatch {
                src_sample_count,
                dst_sample_count,
            } => write!(
                f,
                "the source image has a different sample count ({:?}) than the destination image ({:?})",
                src_sample_count, dst_sample_count,
            ),
            Self::SizeNotAlignedForBuffer {
                resource,
                region_index,
                size,
                required_alignment,
            } => write!(
                f,
                "the {} buffer size ({}) of region {} is not a multiple of the required {} buffer alignment ({})",
                resource, size, region_index, resource, required_alignment,
            ),
        }
    }
}

impl From<SyncCommandBufferBuilderError> for CopyError {
    #[inline]
    fn from(err: SyncCommandBufferBuilderError) -> Self {
        Self::SyncCommandBufferBuilderError(err)
    }
}

/// Indicates which resource a `CopyError` applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyErrorResource {
    Source,
    Destination,
}

impl fmt::Display for CopyErrorResource {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::Source => write!(f, "source"),
            Self::Destination => write!(f, "destination"),
        }
    }
}
