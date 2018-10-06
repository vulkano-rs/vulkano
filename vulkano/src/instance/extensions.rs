// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::fmt;
use std::iter::FromIterator;
use std::ptr;
use std::str;

use check_errors;
use instance::loader;
use instance::loader::LoadingError;
use extensions::SupportedExtensionsError;
use vk;

macro_rules! instance_extensions {
    ($sname:ident, $rawname:ident, $($ext:ident => $s:expr,)*) => (
        extensions! {
            $sname, $rawname,
            $( $ext => $s,)*
        }

        impl $rawname {
            /// See the docs of supported_by_core().
            pub fn supported_by_core_raw() -> Result<Self, SupportedExtensionsError> {
                $rawname::supported_by_core_raw_with_loader(loader::auto_loader()?)
            }

            /// Same as `supported_by_core_raw()`, but allows specifying a loader.
            pub fn supported_by_core_raw_with_loader<L>(ptrs: &loader::FunctionPointers<L>)
                        -> Result<Self, SupportedExtensionsError>
                where L: loader::Loader
            {
                let entry_points = ptrs.entry_points();

                let properties: Vec<vk::ExtensionProperties> = unsafe {
                    let mut num = 0;
                    try!(check_errors(entry_points.EnumerateInstanceExtensionProperties(
                        ptr::null(), &mut num, ptr::null_mut())));

                    let mut properties = Vec::with_capacity(num as usize);
                    try!(check_errors(entry_points.EnumerateInstanceExtensionProperties(
                        ptr::null(), &mut num, properties.as_mut_ptr())));
                    properties.set_len(num as usize);
                    properties
                };
                Ok($rawname(properties.iter().map(|x| unsafe { CStr::from_ptr(x.extensionName.as_ptr()) }.to_owned()).collect()))
            }

            /// Returns a `RawExtensions` object with extensions supported by the core driver.
            pub fn supported_by_core() -> Result<Self, LoadingError> {
                match $rawname::supported_by_core_raw() {
                    Ok(l) => Ok(l),
                    Err(SupportedExtensionsError::LoadingError(e)) => Err(e),
                    Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
                }
            }

            /// Same as `supported_by_core`, but allows specifying a loader.
            pub fn supported_by_core_with_loader<L>(ptrs: &loader::FunctionPointers<L>)
                        -> Result<Self, LoadingError>
                where L: loader::Loader
            {
                match $rawname::supported_by_core_raw_with_loader(ptrs) {
                    Ok(l) => Ok(l),
                    Err(SupportedExtensionsError::LoadingError(e)) => Err(e),
                    Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
                }
            }
        }

        impl $sname {
            /// See the docs of supported_by_core().
            pub fn supported_by_core_raw() -> Result<Self, SupportedExtensionsError> {
                $sname::supported_by_core_raw_with_loader(loader::auto_loader()?)
            }

            /// See the docs of supported_by_core().
            pub fn supported_by_core_raw_with_loader<L>(ptrs: &loader::FunctionPointers<L>)
                        -> Result<Self, SupportedExtensionsError>
                where L: loader::Loader
            {
                let entry_points = ptrs.entry_points();

                let properties: Vec<vk::ExtensionProperties> = unsafe {
                    let mut num = 0;
                    try!(check_errors(entry_points.EnumerateInstanceExtensionProperties(
                        ptr::null(), &mut num, ptr::null_mut())));

                    let mut properties = Vec::with_capacity(num as usize);
                    try!(check_errors(entry_points.EnumerateInstanceExtensionProperties(
                        ptr::null(), &mut num, properties.as_mut_ptr())));
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

            /// Returns a `RawExtensions` object with extensions supported by the core driver.
            pub fn supported_by_core() -> Result<Self, LoadingError> {
                match $sname::supported_by_core_raw() {
                    Ok(l) => Ok(l),
                    Err(SupportedExtensionsError::LoadingError(e)) => Err(e),
                    Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
                }
            }

            /// Same as `supported_by_core`, but allows specifying a loader.
            pub fn supported_by_core_with_loader<L>(ptrs: &loader::FunctionPointers<L>)
                        -> Result<Self, LoadingError>
                where L: loader::Loader
            {
                match $sname::supported_by_core_raw_with_loader(ptrs) {
                    Ok(l) => Ok(l),
                    Err(SupportedExtensionsError::LoadingError(e)) => Err(e),
                    Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
                }
            }
        }
    );
}

instance_extensions! {
    InstanceExtensions,
    RawInstanceExtensions,
    khr_surface => b"VK_KHR_surface",
    khr_display => b"VK_KHR_display",
    khr_xlib_surface => b"VK_KHR_xlib_surface",
    khr_xcb_surface => b"VK_KHR_xcb_surface",
    khr_wayland_surface => b"VK_KHR_wayland_surface",
    khr_android_surface => b"VK_KHR_android_surface",
    khr_win32_surface => b"VK_KHR_win32_surface",
    ext_debug_report => b"VK_EXT_debug_report",
    mvk_ios_surface => b"VK_MVK_ios_surface",
    mvk_macos_surface => b"VK_MVK_macos_surface",
    mvk_moltenvk => b"VK_MVK_moltenvk",     // TODO: confirm that it's an instance extension
    nn_vi_surface => b"VK_NN_vi_surface",
    ext_swapchain_colorspace => b"VK_EXT_swapchain_colorspace",
    khr_get_physical_device_properties2 => b"VK_KHR_get_physical_device_properties2",
}

/// This helper type can only be instantiated inside this module.
/// See `*Extensions::_unbuildable`.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Unbuildable(());

#[cfg(test)]
mod tests {
    use instance::{InstanceExtensions, RawInstanceExtensions};

    #[test]
    fn empty_extensions() {
        let i: RawInstanceExtensions = (&InstanceExtensions::none()).into();
        assert!(i.iter().next().is_none());
    }
}
