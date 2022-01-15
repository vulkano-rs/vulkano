// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::ranges::{is_overlapping_ranges, is_overlapping_regions};
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::format::NumericType;
use crate::image::ImageAccess;
use crate::image::ImageDimensions;
use crate::image::SampleCount;
use crate::sampler::Filter;
use crate::VulkanObject;
use std::error;
use std::fmt;

/// Checks whether a blit image command is valid.
///
/// Note that this doesn't check whether `layer_count` is equal to 0. TODO: change that?
///
/// # Panic
///
/// - Panics if the source or the destination was not created with `device`.
///
pub fn check_blit_image<S, D>(
    device: &Device,
    source: &S,
    source_top_left: [i32; 3],
    source_bottom_right: [i32; 3],
    source_base_array_layer: u32,
    source_mip_level: u32,
    destination: &D,
    destination_top_left: [i32; 3],
    destination_bottom_right: [i32; 3],
    destination_base_array_layer: u32,
    destination_mip_level: u32,
    layer_count: u32,
    filter: Filter,
) -> Result<(), CheckBlitImageError>
where
    S: ?Sized + ImageAccess,
    D: ?Sized + ImageAccess,
{
    let source_inner = source.inner();
    let destination_inner = destination.inner();

    assert_eq!(
        source_inner.image.device().internal_object(),
        device.internal_object()
    );
    assert_eq!(
        destination_inner.image.device().internal_object(),
        device.internal_object()
    );

    if !source_inner.image.usage().transfer_source {
        return Err(CheckBlitImageError::MissingTransferSourceUsage);
    }

    if !destination_inner.image.usage().transfer_destination {
        return Err(CheckBlitImageError::MissingTransferDestinationUsage);
    }

    if !source_inner.image.format_features().blit_src {
        return Err(CheckBlitImageError::SourceFormatNotSupported);
    }

    if !destination_inner.image.format_features().blit_dst {
        return Err(CheckBlitImageError::DestinationFormatNotSupported);
    }

    if source.samples() != SampleCount::Sample1 || destination.samples() != SampleCount::Sample1 {
        return Err(CheckBlitImageError::UnexpectedMultisampled);
    }

    if let (Some(source_type), Some(destination_type)) = (
        source.format().type_color(),
        destination.format().type_color(),
    ) {
        let types_should_be_same = source_type == NumericType::UINT
            || destination_type == NumericType::UINT
            || source_type == NumericType::SINT
            || destination_type == NumericType::SINT;
        if types_should_be_same && (source_type != destination_type) {
            return Err(CheckBlitImageError::IncompatibleFormatTypes {
                source_type,
                destination_type,
            });
        }
    } else {
        if source.format() != destination.format() {
            return Err(CheckBlitImageError::DepthStencilFormatMismatch);
        }

        if filter != Filter::Nearest {
            return Err(CheckBlitImageError::DepthStencilNearestMandatory);
        }
    }

    let source_dimensions = match source.dimensions().mipmap_dimensions(source_mip_level) {
        Some(d) => d,
        None => return Err(CheckBlitImageError::SourceCoordinatesOutOfRange),
    };

    let destination_dimensions = match destination
        .dimensions()
        .mipmap_dimensions(destination_mip_level)
    {
        Some(d) => d,
        None => return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange),
    };

    if source_base_array_layer + layer_count > source_dimensions.array_layers() {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if destination_base_array_layer + layer_count > destination_dimensions.array_layers() {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if source_top_left[0] < 0 || source_top_left[0] > source_dimensions.width() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if source_top_left[1] < 0 || source_top_left[1] > source_dimensions.height() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if source_top_left[2] < 0 || source_top_left[2] > source_dimensions.depth() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if source_bottom_right[0] < 0 || source_bottom_right[0] > source_dimensions.width() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if source_bottom_right[1] < 0 || source_bottom_right[1] > source_dimensions.height() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if source_bottom_right[2] < 0 || source_bottom_right[2] > source_dimensions.depth() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if destination_top_left[0] < 0
        || destination_top_left[0] > destination_dimensions.width() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_top_left[1] < 0
        || destination_top_left[1] > destination_dimensions.height() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_top_left[2] < 0
        || destination_top_left[2] > destination_dimensions.depth() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_bottom_right[0] < 0
        || destination_bottom_right[0] > destination_dimensions.width() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_bottom_right[1] < 0
        || destination_bottom_right[1] > destination_dimensions.height() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_bottom_right[2] < 0
        || destination_bottom_right[2] > destination_dimensions.depth() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    match source_dimensions {
        ImageDimensions::Dim1d { .. } => {
            if source_top_left[1] != 0 || source_bottom_right[1] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
            if source_top_left[2] != 0 || source_bottom_right[2] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim2d { .. } => {
            if source_top_left[2] != 0 || source_bottom_right[2] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim3d { .. } => {}
    }

    match destination_dimensions {
        ImageDimensions::Dim1d { .. } => {
            if destination_top_left[1] != 0 || destination_bottom_right[1] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
            if destination_top_left[2] != 0 || destination_bottom_right[2] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim2d { .. } => {
            if destination_top_left[2] != 0 || destination_bottom_right[2] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim3d { .. } => {}
    }

    if source.conflict_key() == destination.conflict_key() {
        if source_mip_level == destination_mip_level
            && is_overlapping_ranges(
                source_base_array_layer as u64,
                layer_count as u64,
                destination_base_array_layer as u64,
                layer_count as u64,
            )
        {
            // we get the top left coordinate of the source in relation to the resulting image,
            // because in blit we can do top_left = [100, 100] and bottom_right = [0, 0]
            // which would result in flipped image and thats ok, but we can't use these values to compute
            // extent, because it would result in negative size.
            let mut source_render_top_left = [0; 3];
            let mut source_extent = [0; 3];
            let mut destination_render_top_left = [0; 3];
            let mut destination_extent = [0; 3];
            for i in 0..3 {
                if source_top_left[i] < source_bottom_right[i] {
                    source_render_top_left[i] = source_top_left[i];
                    source_extent[i] = (source_bottom_right[i] - source_top_left[i]) as u32;
                } else {
                    source_render_top_left[i] = source_bottom_right[i];
                    source_extent[i] = (source_top_left[i] - source_bottom_right[i]) as u32;
                }
                if destination_top_left[i] < destination_bottom_right[i] {
                    destination_render_top_left[i] = destination_top_left[i];
                    destination_extent[i] =
                        (destination_bottom_right[i] - destination_top_left[i]) as u32;
                } else {
                    destination_render_top_left[i] = destination_bottom_right[i];
                    destination_extent[i] =
                        (destination_top_left[i] - destination_bottom_right[i]) as u32;
                }
            }

            if is_overlapping_regions(
                source_render_top_left,
                source_extent,
                destination_render_top_left,
                destination_extent,
                // since both images are the same, we can use any dimensions type
                source_dimensions,
            ) {
                return Err(CheckBlitImageError::OverlappingRegions);
            }
        }
    }

    match filter {
        Filter::Nearest => (),
        Filter::Linear => {
            if !source_inner
                .image
                .format_features()
                .sampled_image_filter_linear
            {
                return Err(CheckBlitImageError::FilterFormatNotSupported);
            }
        }
        Filter::Cubic => {
            if !device.enabled_extensions().ext_filter_cubic {
                return Err(CheckBlitImageError::ExtensionNotEnabled {
                    extension: "ext_filter_cubic",
                    reason: "the specified filter was Cubic",
                });
            }

            if !source_inner
                .image
                .format_features()
                .sampled_image_filter_cubic
            {
                return Err(CheckBlitImageError::FilterFormatNotSupported);
            }

            if !matches!(source.dimensions(), ImageDimensions::Dim2d { .. }) {
                return Err(CheckBlitImageError::FilterDimensionalityNotSupported);
            }
        }
    }

    Ok(())
}

/// Error that can happen from `check_clear_color_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckBlitImageError {
    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },

    /// The chosen filter type does not support the dimensionality of the source image.
    FilterDimensionalityNotSupported,
    /// The chosen filter type does not support the format of the source image.
    FilterFormatNotSupported,
    /// The source is missing the transfer source usage.
    MissingTransferSourceUsage,
    /// The destination is missing the transfer destination usage.
    MissingTransferDestinationUsage,
    /// The format of the source image doesn't support blit operations.
    SourceFormatNotSupported,
    /// The format of the destination image doesn't support blit operations.
    DestinationFormatNotSupported,
    /// You must use the nearest filter when blitting depth/stencil images.
    DepthStencilNearestMandatory,
    /// The format of the source and destination must be equal when blitting depth/stencil images.
    DepthStencilFormatMismatch,
    /// The types of the source format and the destination format aren't compatible.
    IncompatibleFormatTypes {
        source_type: NumericType,
        destination_type: NumericType,
    },
    /// Blitting between multisampled images is forbidden.
    UnexpectedMultisampled,
    /// The offsets, array layers and/or mipmap levels are out of range in the source image.
    SourceCoordinatesOutOfRange,
    /// The offsets, array layers and/or mipmap levels are out of range in the destination image.
    DestinationCoordinatesOutOfRange,
    /// The top-left and/or bottom-right coordinates are incompatible with the image type.
    IncompatibleRangeForImageType,
    /// The source and destination regions are overlapping.
    OverlappingRegions,
}

impl error::Error for CheckBlitImageError {}

impl fmt::Display for CheckBlitImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::ExtensionNotEnabled { extension, reason } => write!(
                fmt,
                "the extension {} must be enabled: {}",
                extension, reason
            ),
            Self::FilterDimensionalityNotSupported => write!(
                fmt,
                "the chosen filter type does not support the dimensionality of the source image"
            ),
            Self::FilterFormatNotSupported => write!(
                fmt,
                "the chosen filter type does not support the format of the source image"
            ),
            Self::MissingTransferSourceUsage => {
                write!(fmt, "the source is missing the transfer source usage")
            }
            Self::MissingTransferDestinationUsage => {
                write!(
                    fmt,
                    "the destination is missing the transfer destination usage"
                )
            }
            Self::SourceFormatNotSupported => {
                write!(
                    fmt,
                    "the format of the source image doesn't support blit operations"
                )
            }
            Self::DestinationFormatNotSupported => {
                write!(
                    fmt,
                    "the format of the destination image doesn't support blit operations"
                )
            }
            Self::DepthStencilNearestMandatory => {
                write!(
                    fmt,
                    "you must use the nearest filter when blitting depth/stencil images"
                )
            }
            Self::DepthStencilFormatMismatch => {
                write!(fmt, "the format of the source and destination must be equal when blitting depth/stencil images")
            }
            Self::IncompatibleFormatTypes { .. } => {
                write!(
                    fmt,
                    "the types of the source format and the destination format aren't compatible"
                )
            }
            Self::UnexpectedMultisampled => {
                write!(fmt, "blitting between multisampled images is forbidden")
            }
            Self::SourceCoordinatesOutOfRange => {
                write!(fmt, "the offsets, array layers and/or mipmap levels are out of range in the source image")
            }
            Self::DestinationCoordinatesOutOfRange => {
                write!(fmt, "the offsets, array layers and/or mipmap levels are out of range in the destination image")
            }
            Self::IncompatibleRangeForImageType => {
                write!(fmt, "the top-left and/or bottom-right coordinates are incompatible with the image type")
            }
            Self::OverlappingRegions => {
                write!(fmt, "the source and destination regions are overlapping")
            }
        }
    }
}
