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
    ExtensionRestriction, ExtensionRestrictionError, SupportedExtensionsError,
};
use crate::instance::loader;
use crate::instance::loader::LoadingError;
use crate::Version;
use std::ffi::{CStr, CString};
use std::fmt;
use std::ptr;
use std::str;

macro_rules! instance_extensions {
    (
        $($member:ident => {
            doc: $doc:expr,
            raw: $raw:expr,
            requires_core: $requires_core:expr,
            requires_extensions: [$($requires_extension:ident),*]$(,)?
        },)*
    ) => (
        extensions! {
            InstanceExtensions,
            $($member => {
                doc: $doc,
                raw: $raw,
                requires_core: $requires_core,
                requires_device_extensions: [],
                requires_instance_extensions: [$($requires_extension),*],
            },)*
        }

        impl InstanceExtensions {
            /// Checks enabled extensions against the instance version and each other.
            pub(super) fn check_requirements(&self, api_version: Version) -> Result<(), ExtensionRestrictionError> {
                $(
                    if self.$member {
                        if api_version < $requires_core {
                            return Err(ExtensionRestrictionError {
                                extension: stringify!($member),
                                restriction: ExtensionRestriction::RequiresCore($requires_core),
                            });
                        } else {
                            $(
                                if !self.$requires_extension {
                                    return Err(ExtensionRestrictionError {
                                        extension: stringify!($member),
                                        restriction: ExtensionRestriction::RequiresInstanceExtension(stringify!($requires_extension)),
                                    });
                                }
                            )*
                        }
                    }
                )*
                Ok(())
            }
        }
    );
}

impl InstanceExtensions {
    /// See the docs of supported_by_core().
    pub fn supported_by_core_raw() -> Result<Self, SupportedExtensionsError> {
        InstanceExtensions::supported_by_core_raw_with_loader(loader::auto_loader()?)
    }

    /// Returns an `InstanceExtensions` object with extensions supported by the core driver.
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

        Ok(Self::from(properties.iter().map(|property| unsafe {
            CStr::from_ptr(property.extension_name.as_ptr())
        })))
    }
}

/// This helper type can only be instantiated inside this module.
/// See `*Extensions::_unbuildable`.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Unbuildable(());

#[cfg(test)]
mod tests {
    use crate::instance::InstanceExtensions;
    use std::ffi::CString;

    #[test]
    fn empty_extensions() {
        let i: Vec<CString> = (&InstanceExtensions::none()).into();
        assert!(i.iter().next().is_none());
    }
}

// Auto-generated from vk.xml header version 168
instance_extensions! {
    khr_android_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_android_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_KHR_android_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    khr_device_group_creation => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_device_group_creation.html)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_device_group_creation",
        requires_core: Version::V1_0,
        requires_extensions: [],
    },
    khr_display => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_display.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_KHR_display",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    khr_external_fence_capabilities => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_fence_capabilities.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_external_fence_capabilities",
        requires_core: Version::V1_0,
        requires_extensions: [khr_get_physical_device_properties2],
    },
    khr_external_memory_capabilities => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_memory_capabilities.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_external_memory_capabilities",
        requires_core: Version::V1_0,
        requires_extensions: [khr_get_physical_device_properties2],
    },
    khr_external_semaphore_capabilities => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_semaphore_capabilities.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_external_semaphore_capabilities",
        requires_core: Version::V1_0,
        requires_extensions: [khr_get_physical_device_properties2],
    },
    khr_get_display_properties2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_get_display_properties2.html)
			- Requires instance extensions: [`khr_display`](crate::instance::InstanceExtensions::khr_display)
		",
        raw: b"VK_KHR_get_display_properties2",
        requires_core: Version::V1_0,
        requires_extensions: [khr_display],
    },
    khr_get_physical_device_properties2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_get_physical_device_properties2.html)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_get_physical_device_properties2",
        requires_core: Version::V1_0,
        requires_extensions: [],
    },
    khr_get_surface_capabilities2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_get_surface_capabilities2.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_KHR_get_surface_capabilities2",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    khr_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_surface.html)
		",
        raw: b"VK_KHR_surface",
        requires_core: Version::V1_0,
        requires_extensions: [],
    },
    khr_surface_protected_capabilities => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_surface_protected_capabilities.html)
			- Requires Vulkan 1.1
			- Requires instance extensions: [`khr_get_surface_capabilities2`](crate::instance::InstanceExtensions::khr_get_surface_capabilities2)
		",
        raw: b"VK_KHR_surface_protected_capabilities",
        requires_core: Version::V1_1,
        requires_extensions: [khr_get_surface_capabilities2],
    },
    khr_wayland_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_wayland_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_KHR_wayland_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    khr_win32_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_win32_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_KHR_win32_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    khr_xcb_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_xcb_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_KHR_xcb_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    khr_xlib_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_xlib_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_KHR_xlib_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    ext_acquire_xlib_display => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_acquire_xlib_display.html)
			- Requires instance extensions: [`ext_direct_mode_display`](crate::instance::InstanceExtensions::ext_direct_mode_display)
		",
        raw: b"VK_EXT_acquire_xlib_display",
        requires_core: Version::V1_0,
        requires_extensions: [ext_direct_mode_display],
    },
    ext_debug_report => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_debug_report.html)
			- Deprecated by [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils)
		",
        raw: b"VK_EXT_debug_report",
        requires_core: Version::V1_0,
        requires_extensions: [],
    },
    ext_debug_utils => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_debug_utils.html)
		",
        raw: b"VK_EXT_debug_utils",
        requires_core: Version::V1_0,
        requires_extensions: [],
    },
    ext_direct_mode_display => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_direct_mode_display.html)
			- Requires instance extensions: [`khr_display`](crate::instance::InstanceExtensions::khr_display)
		",
        raw: b"VK_EXT_direct_mode_display",
        requires_core: Version::V1_0,
        requires_extensions: [khr_display],
    },
    ext_directfb_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_directfb_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_EXT_directfb_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    ext_display_surface_counter => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_display_surface_counter.html)
			- Requires instance extensions: [`khr_display`](crate::instance::InstanceExtensions::khr_display)
		",
        raw: b"VK_EXT_display_surface_counter",
        requires_core: Version::V1_0,
        requires_extensions: [khr_display],
    },
    ext_headless_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_headless_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_EXT_headless_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    ext_metal_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_metal_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_EXT_metal_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    ext_swapchain_colorspace => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_swapchain_colorspace.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_EXT_swapchain_colorspace",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    ext_validation_features => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_validation_features.html)
		",
        raw: b"VK_EXT_validation_features",
        requires_core: Version::V1_0,
        requires_extensions: [],
    },
    ext_validation_flags => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_validation_flags.html)
			- Deprecated by [`ext_validation_features`](crate::instance::InstanceExtensions::ext_validation_features)
		",
        raw: b"VK_EXT_validation_flags",
        requires_core: Version::V1_0,
        requires_extensions: [],
    },
    fuchsia_imagepipe_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_FUCHSIA_imagepipe_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_FUCHSIA_imagepipe_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    ggp_stream_descriptor_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_GGP_stream_descriptor_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_GGP_stream_descriptor_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    mvk_ios_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_MVK_ios_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
			- Deprecated by [`ext_metal_surface`](crate::instance::InstanceExtensions::ext_metal_surface)
		",
        raw: b"VK_MVK_ios_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    mvk_macos_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_MVK_macos_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
			- Deprecated by [`ext_metal_surface`](crate::instance::InstanceExtensions::ext_metal_surface)
		",
        raw: b"VK_MVK_macos_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    nn_vi_surface => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NN_vi_surface.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_NN_vi_surface",
        requires_core: Version::V1_0,
        requires_extensions: [khr_surface],
    },
    nv_external_memory_capabilities => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_external_memory_capabilities.html)
			- Deprecated by [`khr_external_memory_capabilities`](crate::instance::InstanceExtensions::khr_external_memory_capabilities)
		",
        raw: b"VK_NV_external_memory_capabilities",
        requires_core: Version::V1_0,
        requires_extensions: [],
    },
}
