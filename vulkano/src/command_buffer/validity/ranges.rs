// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::image::ImageDimensions;

/// Checks whether the range `source`..`source + size` is overlapping with the range `destination`..`destination + size`.
/// TODO: add unit tests
pub(super) fn is_overlapping_ranges(
    source: u64,
    source_size: u64,
    destination: u64,
    destination_size: u64,
) -> bool {
    (destination < source + source_size) && (source < destination + destination_size)
}

/// Checks whether there is an overlap between the source and destination regions.
/// The `image_dim` is used to determine the number of dimentions and not the image size.
/// TODO: add unit tests
pub(super) fn is_overlapping_regions(
    source_offset: [i32; 3],
    source_extent: [u32; 3],
    destination_offset: [i32; 3],
    destination_extent: [u32; 3],
    image_dim: ImageDimensions,
) -> bool {
    let dim = match image_dim {
        ImageDimensions::Dim1d { .. } => 1,
        ImageDimensions::Dim2d { .. } => 2,
        ImageDimensions::Dim3d { .. } => 3,
    };
    let mut result = true;
    // for 1d, it will check x only, for 2d x and y, and so on...
    for i in 0..dim {
        result &= is_overlapping_ranges(
            source_offset[i] as u64,
            source_extent[i] as u64,
            destination_offset[i] as u64,
            destination_extent[i] as u64,
        );
    }
    result
}
