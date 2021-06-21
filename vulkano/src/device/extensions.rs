// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::device::physical::PhysicalDevice;
pub use crate::extensions::{
    ExtensionRestriction, ExtensionRestrictionError, SupportedExtensionsError,
};
use crate::VulkanObject;
use std::ffi::CStr;
use std::ptr;

macro_rules! device_extensions {
    (
        $($member:ident => {
            doc: $doc:expr,
            raw: $raw:expr,
            requires_core: $requires_core:expr,
            requires_device_extensions: [$($requires_device_extension:ident),*],
            requires_instance_extensions: [$($requires_instance_extension:ident),*],
            required_if_supported: $required_if_supported:expr,
            conflicts_device_extensions: [$($conflicts_device_extension:ident),*],
        },)*
    ) => (
        extensions! {
            DeviceExtensions,
            $( $member => {
                doc: $doc,
                raw: $raw,
                requires_core: $requires_core,
                requires_device_extensions: [$($requires_device_extension),*],
                requires_instance_extensions: [$($requires_instance_extension),*],
            },)*
        }

        impl DeviceExtensions {
            /// Checks enabled extensions against the device version, instance extensions and each other.
            pub(super) fn check_requirements(
                &self,
                supported: &DeviceExtensions,
                api_version: crate::Version,
                instance_extensions: &InstanceExtensions,
            ) -> Result<(), crate::extensions::ExtensionRestrictionError> {
                $(
                    if self.$member {
                        if !supported.$member {
                            return Err(crate::extensions::ExtensionRestrictionError {
                                extension: stringify!($member),
                                restriction: crate::extensions::ExtensionRestriction::NotSupported,
                            });
                        }

                        if api_version < $requires_core {
                            return Err(crate::extensions::ExtensionRestrictionError {
                                extension: stringify!($member),
                                restriction: crate::extensions::ExtensionRestriction::RequiresCore($requires_core),
                            });
                        }

                        $(
                            if !self.$requires_device_extension {
                                return Err(crate::extensions::ExtensionRestrictionError {
                                    extension: stringify!($member),
                                    restriction: crate::extensions::ExtensionRestriction::RequiresDeviceExtension(stringify!($requires_device_extension)),
                                });
                            }
                        )*

                        $(
                            if !instance_extensions.$requires_instance_extension {
                                return Err(crate::extensions::ExtensionRestrictionError {
                                    extension: stringify!($member),
                                    restriction: crate::extensions::ExtensionRestriction::RequiresInstanceExtension(stringify!($requires_instance_extension)),
                                });
                            }
                        )*

                        $(
                            if self.$conflicts_device_extension {
                                return Err(crate::extensions::ExtensionRestrictionError {
                                    extension: stringify!($member),
                                    restriction: crate::extensions::ExtensionRestriction::ConflictsDeviceExtension(stringify!($conflicts_device_extension)),
                                });
                            }
                        )*
                    } else {
                        if $required_if_supported && supported.$member {
                            return Err(crate::extensions::ExtensionRestrictionError {
                                extension: stringify!($member),
                                restriction: crate::extensions::ExtensionRestriction::RequiredIfSupported,
                            });
                        }
                    }
                )*
                Ok(())
            }

            pub(crate) fn required_if_supported_extensions() -> Self {
                Self {
                    $(
                        $member: $required_if_supported,
                    )*
                    _unbuildable: crate::extensions::Unbuildable(())
                }
            }
        }
   );
}

pub use crate::autogen::DeviceExtensions;
pub(crate) use device_extensions;

impl DeviceExtensions {
    /// See the docs of supported_by_device().
    pub fn supported_by_device_raw(
        physical_device: PhysicalDevice,
    ) -> Result<Self, SupportedExtensionsError> {
        let fns = physical_device.instance().fns();

        let properties: Vec<ash::vk::ExtensionProperties> = unsafe {
            let mut num = 0;
            check_errors(fns.v1_0.enumerate_device_extension_properties(
                physical_device.internal_object(),
                ptr::null(),
                &mut num,
                ptr::null_mut(),
            ))?;

            let mut properties = Vec::with_capacity(num as usize);
            check_errors(fns.v1_0.enumerate_device_extension_properties(
                physical_device.internal_object(),
                ptr::null(),
                &mut num,
                properties.as_mut_ptr(),
            ))?;
            properties.set_len(num as usize);
            properties
        };

        Ok(Self::from(properties.iter().map(|property| unsafe {
            CStr::from_ptr(property.extension_name.as_ptr())
        })))
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
