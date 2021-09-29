// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

use crate::device::Device;
use crate::image::ImageAccess;
use crate::VulkanObject;

/// Checks whether a clear depth / stencil image command is valid.
///
/// # Panic
///
/// - Panics if the destination was not created with `device`.
///
pub fn check_clear_depth_stencil_image<I>(
    device: &Device,
    image: &I,
    first_layer: u32,
    num_layers: u32,
) -> Result<(), CheckClearDepthStencilImageError>
where
    I: ?Sized + ImageAccess,
{
    assert_eq!(
        image.inner().image.device().internal_object(),
        device.internal_object()
    );

    if !image.inner().image.usage().transfer_destination {
        return Err(CheckClearDepthStencilImageError::MissingTransferUsage);
    }

    if first_layer + num_layers > image.dimensions().array_layers() {
        return Err(CheckClearDepthStencilImageError::OutOfRange);
    }

    Ok(())
}

/// Error that can happen from `check_clear_depth_stencil_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckClearDepthStencilImageError {
    /// The image is missing the transfer destination usage.
    MissingTransferUsage,
    /// The array layers are out of range.
    OutOfRange,
}

impl error::Error for CheckClearDepthStencilImageError {}

impl fmt::Display for CheckClearDepthStencilImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckClearDepthStencilImageError::MissingTransferUsage => {
                    "the image is missing the transfer destination usage"
                }
                CheckClearDepthStencilImageError::OutOfRange => {
                    "the array layers are out of range"
                }
            }
        )
    }
}
