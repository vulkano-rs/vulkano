// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::fmt;
use std::iter::FromIterator;
use std::ptr;
use std::str;

use crate::check_errors;
use crate::extensions::SupportedExtensionsError;
use crate::instance::PhysicalDevice;
use crate::vk;
use crate::VulkanObject;

macro_rules! device_extensions {
    ($sname:ident, $rawname:ident, [$($ext_req_if_supported:ident,)*], $($ext:ident => $s:expr,)*) => (
        extensions! {
            $sname, $rawname,
            $( $ext => $s,)*
        }

        impl $rawname {
            /// See the docs of supported_by_device().
            pub fn supported_by_device_raw(physical_device: PhysicalDevice) -> Result<Self, SupportedExtensionsError> {
                let vk = physical_device.instance().pointers();

                let properties: Vec<vk::ExtensionProperties> = unsafe {
                    let mut num = 0;
                    check_errors(vk.EnumerateDeviceExtensionProperties(
                        physical_device.internal_object(), ptr::null(), &mut num, ptr::null_mut()
                    ))?;

                    let mut properties = Vec::with_capacity(num as usize);
                    check_errors(vk.EnumerateDeviceExtensionProperties(
                        physical_device.internal_object(), ptr::null(), &mut num, properties.as_mut_ptr()
                    ))?;
                    properties.set_len(num as usize);
                    properties
                };
                Ok($rawname(properties.iter().map(|x| unsafe { CStr::from_ptr(x.extensionName.as_ptr()) }.to_owned()).collect()))
            }

            /// Returns an `Extensions` object with extensions supported by the `PhysicalDevice`.
            pub fn supported_by_device(physical_device: PhysicalDevice) -> Self {
                match $rawname::supported_by_device_raw(physical_device) {
                    Ok(l) => l,
                    Err(SupportedExtensionsError::LoadingError(_)) => unreachable!(),
                    Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
                }
            }
        }

        impl $sname {
            /// See the docs of supported_by_device().
            pub fn supported_by_device_raw(physical_device: PhysicalDevice) -> Result<Self, SupportedExtensionsError> {
                let vk = physical_device.instance().pointers();

                let properties: Vec<vk::ExtensionProperties> = unsafe {
                    let mut num = 0;
                    check_errors(vk.EnumerateDeviceExtensionProperties(
                        physical_device.internal_object(), ptr::null(), &mut num, ptr::null_mut()
                    ))?;

                    let mut properties = Vec::with_capacity(num as usize);
                    check_errors(vk.EnumerateDeviceExtensionProperties(
                        physical_device.internal_object(), ptr::null(), &mut num, properties.as_mut_ptr()
                    ))?;
                    properties.set_len(num as usize);
                    properties
                };

                let mut extensions = $sname::none();
                for property in properties {
                    let name = unsafe { CStr::from_ptr(property.extensionName.as_ptr()) };
                    $(
                        // TODO: Check specVersion?
                        if name.to_bytes() == &$s[..] {
                            extensions.$ext = true;
                        }
                    )*
                }

                Ok(extensions)
            }

            /// Returns an `Extensions` object with extensions supported by the `PhysicalDevice`.
            pub fn supported_by_device(physical_device: PhysicalDevice) -> Self {
                match $sname::supported_by_device_raw(physical_device) {
                    Ok(l) => l,
                    Err(SupportedExtensionsError::LoadingError(_)) => unreachable!(),
                    Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
                }
            }

            /// Returns an `Extensions` object with extensions required as well as supported by the `PhysicalDevice`.
            /// They are needed to be passed to `Device::new(...)`.
            pub fn required_extensions(physical_device: PhysicalDevice) -> Self {
                let supported = Self::supported_by_device(physical_device);
                let required_if_supported = Self::required_if_supported_extensions();

                required_if_supported.intersection(&supported)
            }

            fn required_if_supported_extensions() -> Self {
                Self {
                    $(
                        $ext_req_if_supported: true,
                    )*
                    ..Self::none()
                }
            }
        }
    );
}

device_extensions! {
    DeviceExtensions,
    RawDeviceExtensions,
    [ // required if supported extensions
        // https://vulkan.lunarg.com/doc/view/1.2.162.1/mac/1.2-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pProperties-04451
        khr_portability_subset,
    ],

    // List in order: khr, ext, then alphabetical
    khr_16bit_storage => b"VK_KHR_16bit_storage",
    khr_8bit_storage => b"VK_KHR_8bit_storage",
    khr_dedicated_allocation => b"VK_KHR_dedicated_allocation",
    khr_display_swapchain => b"VK_KHR_display_swapchain",
    khr_external_memory => b"VK_KHR_external_memory",
    khr_external_memory_fd => b"VK_KHR_external_memory_fd",
    khr_get_memory_requirements2 => b"VK_KHR_get_memory_requirements2",
    khr_incremental_present => b"VK_KHR_incremental_present",
    khr_maintenance1 => b"VK_KHR_maintenance1",
    khr_multiview => b"VK_KHR_multiview",
    khr_portability_subset => b"VK_KHR_portability_subset",
    khr_sampler_mirror_clamp_to_edge => b"VK_KHR_sampler_mirror_clamp_to_edge",
    khr_spirv_1_4 => b"VK_KHR_spirv_1_4",
    khr_storage_buffer_storage_class => b"VK_KHR_storage_buffer_storage_class",
    khr_swapchain => b"VK_KHR_swapchain",
    ext_debug_utils => b"VK_EXT_debug_utils",
    ext_external_memory_dmabuf => b"VK_EXT_external_memory_dma_buf",
    ext_full_screen_exclusive => b"VK_EXT_full_screen_exclusive",
}

/// This helper type can only be instantiated inside this module.
/// See `*Extensions::_unbuildable`.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Unbuildable(());

#[cfg(test)]
mod tests {
    use crate::device::{DeviceExtensions, RawDeviceExtensions};

    #[test]
    fn empty_extensions() {
        let d: RawDeviceExtensions = (&DeviceExtensions::none()).into();
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
