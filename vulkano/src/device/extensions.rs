// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::device::physical::PhysicalDevice;
pub use crate::extensions::{
    ExtensionRestriction, ExtensionRestrictionError, SupportedExtensionsError,
};

pub use crate::autogen::DeviceExtensions;

impl DeviceExtensions {
    /// See the docs of supported_by_device().
    #[deprecated(
        since = "0.25",
        note = "Use PhysicalDevice::supported_extensions instead"
    )]
    pub fn supported_by_device_raw(
        physical_device: PhysicalDevice,
    ) -> Result<Self, SupportedExtensionsError> {
        Ok(*physical_device.supported_extensions())
    }

    /// Returns a `DeviceExtensions` object with extensions supported by the `PhysicalDevice`.
    #[deprecated(
        since = "0.25",
        note = "Use PhysicalDevice::supported_extensions instead"
    )]
    pub fn supported_by_device(physical_device: PhysicalDevice) -> Self {
        *physical_device.supported_extensions()
    }

    /// Returns a `DeviceExtensions` object with extensions required as well as supported by the `PhysicalDevice`.
    /// They are needed to be passed to `Device::new(...)`.
    #[deprecated(
        since = "0.25",
        note = "Use PhysicalDevice::required_extensions instead"
    )]
    pub fn required_extensions(physical_device: PhysicalDevice) -> Self {
        *physical_device.required_extensions()
    }
}

/// This helper type can only be instantiated inside this module.
/// See `*Extensions::_unbuildable`.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Unbuildable(());

#[cfg(test)]
mod tests {
    use crate::device::DeviceExtensions;
    use std::ffi::CString;

    #[test]
    fn empty_extensions() {
        let d: Vec<CString> = (&DeviceExtensions::none()).into();
        assert!(d.iter().next().is_none());
    }

    #[test]
    fn required_if_supported_extensions() {
        assert_eq!(
            DeviceExtensions::required_if_supported_extensions(),
            DeviceExtensions {
                khr_portability_subset: true,
                ..DeviceExtensions::none()
            }
        )
    }
}
