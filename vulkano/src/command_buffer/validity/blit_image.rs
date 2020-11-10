// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

use device::Device;
use format::FormatTy;
use image::ImageAccess;
use image::ImageDimensions;
use sampler::Filter;
use VulkanObject;

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

    if !source_inner.image.usage_transfer_source() {
        return Err(CheckBlitImageError::MissingTransferSourceUsage);
    }

    if !destination_inner.image.usage_transfer_destination() {
        return Err(CheckBlitImageError::MissingTransferDestinationUsage);
    }

    if !source_inner.image.supports_blit_source() {
        return Err(CheckBlitImageError::SourceFormatNotSupported);
    }

    if !destination_inner.image.supports_blit_destination() {
        return Err(CheckBlitImageError::DestinationFormatNotSupported);
    }

    if source.samples() != 1 || destination.samples() != 1 {
        return Err(CheckBlitImageError::UnexpectedMultisampled);
    }

    let source_format_ty = source.format().ty();
    let destination_format_ty = destination.format().ty();

    if source_format_ty.is_depth_and_or_stencil() {
        if source.format() != destination.format() {
            return Err(CheckBlitImageError::DepthStencilFormatMismatch);
        }

        if filter != Filter::Nearest {
            return Err(CheckBlitImageError::DepthStencilNearestMandatory);
        }
    }

    let types_should_be_same = source_format_ty == FormatTy::Uint
        || destination_format_ty == FormatTy::Uint
        || source_format_ty == FormatTy::Sint
        || destination_format_ty == FormatTy::Sint;
    if types_should_be_same && (source_format_ty != destination_format_ty) {
        return Err(CheckBlitImageError::IncompatibleFormatsTypes {
            source_format_ty: source.format().ty(),
            destination_format_ty: destination.format().ty(),
        });
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

    Ok(())
}

/// Error that can happen from `check_clear_color_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckBlitImageError {
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
    IncompatibleFormatsTypes {
        source_format_ty: FormatTy,
        destination_format_ty: FormatTy,
    },
    /// Blitting between multisampled images is forbidden.
    UnexpectedMultisampled,
    /// The offsets, array layers and/or mipmap levels are out of range in the source image.
    SourceCoordinatesOutOfRange,
    /// The offsets, array layers and/or mipmap levels are out of range in the destination image.
    DestinationCoordinatesOutOfRange,
    /// The top-left and/or bottom-right coordinates are incompatible with the image type.
    IncompatibleRangeForImageType,
}

impl error::Error for CheckBlitImageError {}

impl fmt::Display for CheckBlitImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckBlitImageError::MissingTransferSourceUsage => {
                    "the source is missing the transfer source usage"
                }
                CheckBlitImageError::MissingTransferDestinationUsage => {
                    "the destination is missing the transfer destination usage"
                }
                CheckBlitImageError::SourceFormatNotSupported => {
                    "the format of the source image doesn't support blit operations"
                }
                CheckBlitImageError::DestinationFormatNotSupported => {
                    "the format of the destination image doesn't support blit operations"
                }
                CheckBlitImageError::DepthStencilNearestMandatory => {
                    "you must use the nearest filter when blitting depth/stencil images"
                }
                CheckBlitImageError::DepthStencilFormatMismatch => {
                    "the format of the source and destination must be equal when blitting \
                 depth/stencil images"
                }
                CheckBlitImageError::IncompatibleFormatsTypes { .. } => {
                    "the types of the source format and the destination format aren't compatible"
                }
                CheckBlitImageError::UnexpectedMultisampled => {
                    "blitting between multisampled images is forbidden"
                }
                CheckBlitImageError::SourceCoordinatesOutOfRange => {
                    "the offsets, array layers and/or mipmap levels are out of range in the source \
                 image"
                }
                CheckBlitImageError::DestinationCoordinatesOutOfRange => {
                    "the offsets, array layers and/or mipmap levels are out of range in the \
                 destination image"
                }
                CheckBlitImageError::IncompatibleRangeForImageType => {
                    "the top-left and/or bottom-right coordinates are incompatible with the image type"
                }
            }
        )
    }
}
