// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::extensions::{
    ExtensionRequirement, ExtensionRequirementError, SupportedExtensionsError,
};
use crate::instance::loader;
use crate::instance::loader::LoadingError;
use crate::Version;
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::fmt;
use std::iter::FromIterator;
use std::ptr;
use std::str;

macro_rules! instance_extensions {
    (
        $($member:ident => {
            raw: $raw:expr,
            requires_core: $requires_core:ident,
            requires_extensions: [$($requires_extension:ident),*]$(,)?
        },)*
    ) => (
        extensions! {
            InstanceExtensions, RawInstanceExtensions,
            $($member => {
                raw: $raw,
                requires_core: $requires_core,
                requires_device_extensions: [],
                requires_instance_extensions: [$($requires_extension),*],
            },)*
        }

        impl InstanceExtensions {
            /// Checks enabled extensions against the instance version and each other.
            pub(super) fn check_requirements(&self, api_version: Version) -> Result<(), ExtensionRequirementError> {
                $(
                    if self.$member {
                        if api_version < Version::$requires_core {
                            return Err(ExtensionRequirementError {
                                extension: stringify!($member),
                                requirement: ExtensionRequirement::Core(Version::$requires_core),
                            });
                        } else {
                            $(
                                if !self.$requires_extension {
                                    return Err(ExtensionRequirementError {
                                        extension: stringify!($member),
                                        requirement: ExtensionRequirement::InstanceExtension(stringify!($requires_extension)),
                                    });
                                }
                            )*
                        }
                    }
                )*
                Ok(())
            }
        }

        impl From<&[ash::vk::ExtensionProperties]> for InstanceExtensions {
            fn from(properties: &[ash::vk::ExtensionProperties]) -> Self {
                let mut extensions = InstanceExtensions::none();
                for property in properties {
                    let name = unsafe { CStr::from_ptr(property.extension_name.as_ptr()) };
                    $(
                        // TODO: Check specVersion?
                        if name.to_bytes() == &$raw[..] {
                            extensions.$member = true;
                        }
                    )*
                }
                extensions
            }
        }
    );
}

impl InstanceExtensions {
    /// See the docs of supported_by_core().
    pub fn supported_by_core_raw() -> Result<Self, SupportedExtensionsError> {
        InstanceExtensions::supported_by_core_raw_with_loader(loader::auto_loader()?)
    }

    /// Returns a `RawExtensions` object with extensions supported by the core driver.
    pub fn supported_by_core() -> Result<Self, LoadingError> {
        match InstanceExtensions::supported_by_core_raw() {
            Ok(l) => Ok(l),
            Err(SupportedExtensionsError::LoadingError(e)) => Err(e),
            Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
        }
    }

    /// Same as `supported_by_core`, but allows specifying a loader.
    pub fn supported_by_core_with_loader<L>(
        ptrs: &loader::FunctionPointers<L>,
    ) -> Result<Self, LoadingError>
    where
        L: loader::Loader,
    {
        match InstanceExtensions::supported_by_core_raw_with_loader(ptrs) {
            Ok(l) => Ok(l),
            Err(SupportedExtensionsError::LoadingError(e)) => Err(e),
            Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
        }
    }

    /// See the docs of supported_by_core().
    pub fn supported_by_core_raw_with_loader<L>(
        ptrs: &loader::FunctionPointers<L>,
    ) -> Result<Self, SupportedExtensionsError>
    where
        L: loader::Loader,
    {
        let fns = ptrs.fns();

        let properties: Vec<ash::vk::ExtensionProperties> = unsafe {
            let mut num = 0;
            check_errors(fns.v1_0.enumerate_instance_extension_properties(
                ptr::null(),
                &mut num,
                ptr::null_mut(),
            ))?;

            let mut properties = Vec::with_capacity(num as usize);
            check_errors(fns.v1_0.enumerate_instance_extension_properties(
                ptr::null(),
                &mut num,
                properties.as_mut_ptr(),
            ))?;
            properties.set_len(num as usize);
            properties
        };

        Ok(Self::from(properties.as_slice()))
    }
}

impl RawInstanceExtensions {
    /// See the docs of supported_by_core().
    pub fn supported_by_core_raw() -> Result<Self, SupportedExtensionsError> {
        RawInstanceExtensions::supported_by_core_raw_with_loader(loader::auto_loader()?)
    }

    /// Same as `supported_by_core_raw()`, but allows specifying a loader.
    pub fn supported_by_core_raw_with_loader<L>(
        ptrs: &loader::FunctionPointers<L>,
    ) -> Result<Self, SupportedExtensionsError>
    where
        L: loader::Loader,
    {
        let fns = ptrs.fns();

        let properties: Vec<ash::vk::ExtensionProperties> = unsafe {
            let mut num = 0;
            check_errors(fns.v1_0.enumerate_instance_extension_properties(
                ptr::null(),
                &mut num,
                ptr::null_mut(),
            ))?;

            let mut properties = Vec::with_capacity(num as usize);
            check_errors(fns.v1_0.enumerate_instance_extension_properties(
                ptr::null(),
                &mut num,
                properties.as_mut_ptr(),
            ))?;
            properties.set_len(num as usize);
            properties
        };
        Ok(RawInstanceExtensions(
            properties
                .iter()
                .map(|x| unsafe { CStr::from_ptr(x.extension_name.as_ptr()) }.to_owned())
                .collect(),
        ))
    }

    /// Returns a `RawExtensions` object with extensions supported by the core driver.
    pub fn supported_by_core() -> Result<Self, LoadingError> {
        match RawInstanceExtensions::supported_by_core_raw() {
            Ok(l) => Ok(l),
            Err(SupportedExtensionsError::LoadingError(e)) => Err(e),
            Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
        }
    }

    /// Same as `supported_by_core`, but allows specifying a loader.
    pub fn supported_by_core_with_loader<L>(
        ptrs: &loader::FunctionPointers<L>,
    ) -> Result<Self, LoadingError>
    where
        L: loader::Loader,
    {
        match RawInstanceExtensions::supported_by_core_raw_with_loader(ptrs) {
            Ok(l) => Ok(l),
            Err(SupportedExtensionsError::LoadingError(e)) => Err(e),
            Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
        }
    }
}

/// This helper type can only be instantiated inside this module.
/// See `*Extensions::_unbuildable`.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Unbuildable(());

#[cfg(test)]
mod tests {
    use crate::instance::{InstanceExtensions, RawInstanceExtensions};

    #[test]
    fn empty_extensions() {
        let i: RawInstanceExtensions = (&InstanceExtensions::none()).into();
        assert!(i.iter().next().is_none());
    }
}

// Auto-generated from vk.xml
instance_extensions! {
    khr_android_surface => {
        raw: b"VK_KHR_android_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    khr_device_group_creation => {
        raw: b"VK_KHR_device_group_creation",
        requires_core: V1_0,
        requires_extensions: [],
    },
    khr_display => {
        raw: b"VK_KHR_display",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    khr_external_fence_capabilities => {
        raw: b"VK_KHR_external_fence_capabilities",
        requires_core: V1_0,
        requires_extensions: [khr_get_physical_device_properties2],
    },
    khr_external_memory_capabilities => {
        raw: b"VK_KHR_external_memory_capabilities",
        requires_core: V1_0,
        requires_extensions: [khr_get_physical_device_properties2],
    },
    khr_external_semaphore_capabilities => {
        raw: b"VK_KHR_external_semaphore_capabilities",
        requires_core: V1_0,
        requires_extensions: [khr_get_physical_device_properties2],
    },
    khr_get_display_properties2 => {
        raw: b"VK_KHR_get_display_properties2",
        requires_core: V1_0,
        requires_extensions: [khr_display],
    },
    khr_get_physical_device_properties2 => {
        raw: b"VK_KHR_get_physical_device_properties2",
        requires_core: V1_0,
        requires_extensions: [],
    },
    khr_get_surface_capabilities2 => {
        raw: b"VK_KHR_get_surface_capabilities2",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    khr_surface => {
        raw: b"VK_KHR_surface",
        requires_core: V1_0,
        requires_extensions: [],
    },
    khr_surface_protected_capabilities => {
        raw: b"VK_KHR_surface_protected_capabilities",
        requires_core: V1_1,
        requires_extensions: [khr_get_surface_capabilities2],
    },
    khr_wayland_surface => {
        raw: b"VK_KHR_wayland_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    khr_win32_surface => {
        raw: b"VK_KHR_win32_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    khr_xcb_surface => {
        raw: b"VK_KHR_xcb_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    khr_xlib_surface => {
        raw: b"VK_KHR_xlib_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    ext_acquire_xlib_display => {
        raw: b"VK_EXT_acquire_xlib_display",
        requires_core: V1_0,
        requires_extensions: [ext_direct_mode_display],
    },
    ext_debug_report => {
        raw: b"VK_EXT_debug_report",
        requires_core: V1_0,
        requires_extensions: [],
    },
    ext_debug_utils => {
        raw: b"VK_EXT_debug_utils",
        requires_core: V1_0,
        requires_extensions: [],
    },
    ext_direct_mode_display => {
        raw: b"VK_EXT_direct_mode_display",
        requires_core: V1_0,
        requires_extensions: [khr_display],
    },
    ext_directfb_surface => {
        raw: b"VK_EXT_directfb_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    ext_display_surface_counter => {
        raw: b"VK_EXT_display_surface_counter",
        requires_core: V1_0,
        requires_extensions: [khr_display],
    },
    ext_headless_surface => {
        raw: b"VK_EXT_headless_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    ext_metal_surface => {
        raw: b"VK_EXT_metal_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    ext_swapchain_colorspace => {
        raw: b"VK_EXT_swapchain_colorspace",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    ext_validation_features => {
        raw: b"VK_EXT_validation_features",
        requires_core: V1_0,
        requires_extensions: [],
    },
    ext_validation_flags => {
        raw: b"VK_EXT_validation_flags",
        requires_core: V1_0,
        requires_extensions: [],
    },
    fuchsia_imagepipe_surface => {
        raw: b"VK_FUCHSIA_imagepipe_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    ggp_stream_descriptor_surface => {
        raw: b"VK_GGP_stream_descriptor_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    mvk_ios_surface => {
        raw: b"VK_MVK_ios_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    mvk_macos_surface => {
        raw: b"VK_MVK_macos_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    nn_vi_surface => {
        raw: b"VK_NN_vi_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
    nv_external_memory_capabilities => {
        raw: b"VK_NV_external_memory_capabilities",
        requires_core: V1_0,
        requires_extensions: [],
    },
    qnx_screen_surface => {
        raw: b"VK_QNX_screen_surface",
        requires_core: V1_0,
        requires_extensions: [khr_surface],
    },
}
