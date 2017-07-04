// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

use VulkanObject;
use device::Device;
use image::ImageAccess;

/// Checks whether a clear color image command is valid.
///
/// # Panic
///
/// - Panics if the destination was not created with `device`.
///
pub fn check_clear_color_image<I>(device: &Device, image: &I)
                                  -> Result<(), CheckClearColorImageError>
    where I: ?Sized + ImageAccess,
{
    assert_eq!(image.inner().image.device().internal_object(),
               device.internal_object());

    if !image.inner().image.usage_transfer_destination() {
        return Err(CheckClearColorImageError::MissingTransferUsage);
    }

    Ok(())
}

/// Error that can happen from `check_clear_color_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckClearColorImageError {
    /// The image is missing the transfer destination usage.
    MissingTransferUsage,
}

impl error::Error for CheckClearColorImageError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CheckClearColorImageError::MissingTransferUsage => {
                "the image is missing the transfer destination usage"
            },
        }
    }
}

impl fmt::Display for CheckClearColorImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
