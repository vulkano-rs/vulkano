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

/// Checks whether the dispatch dimensions are supported by the device.
pub fn check_dispatch(device: &Device, dimensions: [u32; 3]) -> Result<(), CheckDispatchError> {
    let max = device
        .physical_device()
        .limits()
        .max_compute_work_group_count();

    if dimensions[0] > max[0] || dimensions[1] > max[1] || dimensions[2] > max[2] {
        return Err(CheckDispatchError::UnsupportedDimensions {
            requested: dimensions,
            max_supported: max,
        });
    }

    Ok(())
}

/// Error that can happen when checking dispatch command validity.
#[derive(Debug, Copy, Clone)]
pub enum CheckDispatchError {
    /// The dimensions are too large for the device's limits.
    UnsupportedDimensions {
        /// The requested dimensions.
        requested: [u32; 3],
        /// The actual supported dimensions.
        max_supported: [u32; 3],
    },
}

impl error::Error for CheckDispatchError {}

impl fmt::Display for CheckDispatchError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckDispatchError::UnsupportedDimensions { .. } => {
                    "the dimensions are too large for the device's limits"
                }
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use command_buffer::validity;

    #[test]
    fn max_checked() {
        let (device, _) = gfx_dev_and_queue!();

        let attempted = [u32::max_value(), u32::max_value(), u32::max_value()];

        // Just in case the device is some kind of software implementation.
        if device
            .physical_device()
            .limits()
            .max_compute_work_group_count()
            == attempted
        {
            return;
        }

        match validity::check_dispatch(&device, attempted) {
            Err(validity::CheckDispatchError::UnsupportedDimensions { requested, .. }) => {
                assert_eq!(requested, attempted);
            }
            _ => panic!(),
        }
    }
}
