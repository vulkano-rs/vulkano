// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::BufferAccess,
    device::{Device, DeviceOwned},
    format::Format,
    image::{ImageAccess, ImageDimensions, SampleCount},
    DeviceSize, VulkanObject,
};
use std::{error, fmt};

/// Type of operation to check.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CheckCopyBufferImageTy {
    BufferToImage,
    ImageToBuffer,
}

/// Checks whether a copy buffer-image command is valid. Can check both buffer-to-image copies and
/// image-to-buffer copies.
///
/// # Panic
///
/// - Panics if the buffer and image were not created with `device`.
///
pub fn check_copy_buffer_image(
    device: &Device,
    buffer: &dyn BufferAccess,
    image: &dyn ImageAccess,
    ty: CheckCopyBufferImageTy,
    image_offset: [u32; 3],
    image_size: [u32; 3],
    image_first_layer: u32,
    image_num_layers: u32,
    image_mipmap: u32,
) -> Result<(), CheckCopyBufferImageError> {
    let buffer_inner = buffer.inner();
    let image_inner = image.inner();

    assert_eq!(
        buffer_inner.buffer.device().internal_object(),
        device.internal_object()
    );
    assert_eq!(
        image_inner.image.device().internal_object(),
        device.internal_object()
    );

    match ty {
        CheckCopyBufferImageTy::BufferToImage => {
            if !buffer_inner.buffer.usage().transfer_source {
                return Err(CheckCopyBufferImageError::SourceMissingTransferUsage);
            }
            if !image_inner.image.usage().transfer_destination {
                return Err(CheckCopyBufferImageError::DestinationMissingTransferUsage);
            }
        }
        CheckCopyBufferImageTy::ImageToBuffer => {
            if !image_inner.image.usage().transfer_source {
                return Err(CheckCopyBufferImageError::SourceMissingTransferUsage);
            }
            if !buffer_inner.buffer.usage().transfer_destination {
                return Err(CheckCopyBufferImageError::DestinationMissingTransferUsage);
            }
        }
    }

    if image.samples() != SampleCount::Sample1 {
        return Err(CheckCopyBufferImageError::UnexpectedMultisampled);
    }

    let image_dimensions = match image.dimensions().mip_level_dimensions(image_mipmap) {
        Some(d) => d,
        None => return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange),
    };

    if image_first_layer + image_num_layers > image_dimensions.array_layers() {
        return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
    }

    if image_offset[0] + image_size[0] > image_dimensions.width() {
        return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
    }

    if image_offset[1] + image_size[1] > image_dimensions.height() {
        return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
    }

    if image_offset[2] + image_size[2] > image_dimensions.depth() {
        return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
    }

    match image.dimensions() {
        ImageDimensions::Dim1d { .. } => {
            // VUID-vkCmdCopyBufferToImage-srcImage-00199
            if image_offset[1] != 0 || image_size[1] != 1 {
                return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
            }

            // VUID-vkCmdCopyBufferToImage-srcImage-00201
            if image_offset[2] != 0 || image_size[2] != 1 {
                return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
            }
        }
        ImageDimensions::Dim2d { .. } => {
            // VUID-vkCmdCopyBufferToImage-srcImage-00201
            if image_offset[2] != 0 || image_size[2] != 1 {
                return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
            }
        }
        ImageDimensions::Dim3d { .. } => {
            // VUID-vkCmdCopyBufferToImage-baseArrayLayer-00213
            if image_first_layer != 0 || image_num_layers != 1 {
                return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
            }
        }
    }

    let required_size = required_size_for_format(image.format(), image_size, image_num_layers);
    if required_size > buffer.size() {
        return Err(CheckCopyBufferImageError::BufferTooSmall {
            required_size,
            actual_size: buffer.size(),
        });
    }

    // TODO: check memory overlap?

    Ok(())
}

/// Computes the minimum required len in elements for buffer with image data in specified
/// format of specified size.
fn required_size_for_format(format: Format, extent: [u32; 3], layer_count: u32) -> DeviceSize {
    let num_blocks = extent
        .into_iter()
        .zip(format.block_extent())
        .map(|(extent, block_extent)| {
            let extent = extent as DeviceSize;
            let block_extent = block_extent as DeviceSize;
            (extent + block_extent - 1) / block_extent
        })
        .product::<DeviceSize>()
        * layer_count as DeviceSize;
    let block_size = format
        .block_size()
        .expect("this format cannot accept pixels");
    num_blocks * block_size
}

#[cfg(test)]
mod tests {
    use crate::command_buffer::validity::copy_image_buffer::required_size_for_format;
    use crate::format::Format;

    #[test]
    fn test_required_len_for_format() {
        // issue #1292
        assert_eq!(
            required_size_for_format(Format::BC1_RGB_UNORM_BLOCK, [2048, 2048, 1], 1),
            2097152
        );
        // other test cases
        assert_eq!(
            required_size_for_format(Format::R8G8B8A8_UNORM, [2048, 2048, 1], 1),
            16777216
        );
        assert_eq!(
            required_size_for_format(Format::R4G4_UNORM_PACK8, [512, 512, 1], 1),
            262144
        );
        assert_eq!(
            required_size_for_format(Format::R8G8B8_USCALED, [512, 512, 1], 1),
            786432
        );
        assert_eq!(
            required_size_for_format(Format::R32G32_UINT, [512, 512, 1], 1),
            2097152
        );
        assert_eq!(
            required_size_for_format(Format::R32G32_UINT, [512, 512, 1], 1),
            2097152
        );
        assert_eq!(
            required_size_for_format(Format::ASTC_8x8_UNORM_BLOCK, [512, 512, 1], 1),
            65536
        );
        assert_eq!(
            required_size_for_format(Format::ASTC_12x12_SRGB_BLOCK, [512, 512, 1], 1),
            29584
        );
    }
}

/// Error that can happen from `check_copy_buffer_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckCopyBufferImageError {
    /// The source buffer or image is missing the transfer source usage.
    SourceMissingTransferUsage,
    /// The destination buffer or image is missing the transfer destination usage.
    DestinationMissingTransferUsage,
    /// The source and destination are overlapping.
    OverlappingRanges,
    /// The image must not be multisampled.
    UnexpectedMultisampled,
    /// The image coordinates are out of range.
    ImageCoordinatesOutOfRange,
    /// The buffer is too small for the copy operation.
    BufferTooSmall {
        /// Required size of the buffer.
        required_size: DeviceSize,
        /// Actual size of the buffer.
        actual_size: DeviceSize,
    },
}

impl error::Error for CheckCopyBufferImageError {}

impl fmt::Display for CheckCopyBufferImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckCopyBufferImageError::SourceMissingTransferUsage => {
                    "the source buffer is missing the transfer source usage"
                }
                CheckCopyBufferImageError::DestinationMissingTransferUsage => {
                    "the destination buffer is missing the transfer destination usage"
                }
                CheckCopyBufferImageError::OverlappingRanges => {
                    "the source and destination are overlapping"
                }
                CheckCopyBufferImageError::UnexpectedMultisampled => {
                    "the image must not be multisampled"
                }
                CheckCopyBufferImageError::ImageCoordinatesOutOfRange => {
                    "the image coordinates are out of range"
                }
                CheckCopyBufferImageError::BufferTooSmall { .. } => {
                    "the buffer is too small for the copy operation"
                }
            }
        )
    }
}
