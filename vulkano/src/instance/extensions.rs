// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::ffi::CString;
use std::fmt;
use std::ptr;
use std::str;

use Error;
use OomError;
use instance::loader;
use instance::loader::LoadingError;
use vk;
use check_errors;

macro_rules! extensions {
    ($sname:ident, $($ext:ident => $s:expr,)*) => (
        /// List of extensions that are enabled or available.
        #[derive(Copy, Clone, PartialEq, Eq)]
        #[allow(missing_docs)]
        pub struct $sname {
            $(
                pub $ext: bool,
            )*

            /// This field ensures that an instance of this `Extensions` struct
            /// can only be created through Vulkano functions and the update
            /// syntax. This way, extensions can be added to Vulkano without
            /// breaking existing code.
            pub _unbuildable: Unbuildable,
        }

        impl $sname {
            /// Returns an `Extensions` object with all members set to `false`.
            #[inline]
            pub fn none() -> $sname {
                $sname {
                    $($ext: false,)*
                    _unbuildable: Unbuildable(())
                }
            }

            /// Builds a Vec containing the list of extensions.
            pub fn build_extensions_list(&self) -> Vec<CString> {
                let mut data = Vec::new();
                $(if self.$ext { data.push(CString::new(&$s[..]).unwrap()); })*
                data
            }

            /// Returns the intersection of this list and another list.
            #[inline]
            pub fn intersection(&self, other: &$sname) -> $sname {
                $sname {
                    $(
                        $ext: self.$ext && other.$ext,
                    )*
                    _unbuildable: Unbuildable(())
                }
            }
        }

        impl fmt::Debug for $sname {
            #[allow(unused_assignments)]
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                try!(write!(f, "["));

                let mut first = true;

                $(
                    if self.$ext {
                        if !first { try!(write!(f, ", ")); }
                        else { first = false; }
                        try!(f.write_str(str::from_utf8($s).unwrap()));
                    }
                )*

                write!(f, "]")
            }
        }
    );
}

macro_rules! instance_extensions {
    ($sname:ident, $($ext:ident => $s:expr,)*) => (
        extensions! {
            $sname,
            $( $ext => $s,)*
        }

        impl $sname {
            /// See the docs of supported_by_core().
            pub fn supported_by_core_raw() -> Result<$sname, SupportedExtensionsError> {
                let entry_points = try!(loader::entry_points());

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
                    let name = property.extensionName;
                    $(
                        // TODO: this is VERY inefficient
                        // TODO: Check specVersion?
                        let same = {
                            let mut i = 0;
                            while name[i] != 0 && $s[i] != 0 && name[i] as u8 == $s[i] && i < $s.len() { i += 1; }
                            name[i] == 0 && (i >= $s.len() || name[i] as u8 == $s[i])
                        };
                        if same {
                            extensions.$ext = true;
                        }
                    )*
                }

                Ok(extensions)
            }

            /// Returns an `Extensions` object with extensions supported by the core driver.
            pub fn supported_by_core() -> Result<$sname, LoadingError> {
                match $sname::supported_by_core_raw() {
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
    khr_surface => b"VK_KHR_surface",
    khr_display => b"VK_KHR_display",
    khr_xlib_surface => b"VK_KHR_xlib_surface",
    khr_xcb_surface => b"VK_KHR_xcb_surface",
    khr_wayland_surface => b"VK_KHR_wayland_surface",
    khr_mir_surface => b"VK_KHR_mir_surface",
    khr_android_surface => b"VK_KHR_android_surface",
    khr_win32_surface => b"VK_KHR_win32_surface",
    ext_debug_report => b"VK_EXT_debug_report",
    nn_vi_surface => b"VK_NN_vi_surface",
    ext_swapchain_colorspace => b"VK_EXT_swapchain_colorspace",
    khr_get_physical_device_properties2 => b"VK_KHR_get_physical_device_properties2",
}

extensions! {
    DeviceExtensions,
    khr_swapchain => b"VK_KHR_swapchain",
    khr_display_swapchain => b"VK_KHR_display_swapchain",
    khr_maintenance1 => b"VK_KHR_maintenance1",
    khr_descriptor_update_template => b"VK_KHR_descriptor_update_template",
    khr_push_descriptor => b"VK_KHR_push_descriptor",
    khr_sampler_mirror_clamp_to_edge => b"VK_KHR_sampler_mirror_clamp_to_edge",
}

/// Error that can happen when loading the list of layers.
#[derive(Clone, Debug)]
pub enum SupportedExtensionsError {
    /// Failed to load the Vulkan shared library.
    LoadingError(LoadingError),
    /// Not enough memory.
    OomError(OomError),
}

impl error::Error for SupportedExtensionsError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            SupportedExtensionsError::LoadingError(_) => "failed to load the Vulkan shared library",
            SupportedExtensionsError::OomError(_) => "not enough memory available",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            SupportedExtensionsError::LoadingError(ref err) => Some(err),
            SupportedExtensionsError::OomError(ref err) => Some(err),
        }
    }
}

impl fmt::Display for SupportedExtensionsError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for SupportedExtensionsError {
    #[inline]
    fn from(err: OomError) -> SupportedExtensionsError {
        SupportedExtensionsError::OomError(err)
    }
}

impl From<LoadingError> for SupportedExtensionsError {
    #[inline]
    fn from(err: LoadingError) -> SupportedExtensionsError {
        SupportedExtensionsError::LoadingError(err)
    }
}

impl From<Error> for SupportedExtensionsError {
    #[inline]
    fn from(err: Error) -> SupportedExtensionsError {
        match err {
            err @ Error::OutOfHostMemory => {
                SupportedExtensionsError::OomError(OomError::from(err))
            },
            err @ Error::OutOfDeviceMemory => {
                SupportedExtensionsError::OomError(OomError::from(err))
            },
            _ => panic!("unexpected error: {:?}", err)
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
    use instance::InstanceExtensions;
    use instance::DeviceExtensions;

    #[test]
    fn empty_extensions() {
        let i = InstanceExtensions::none().build_extensions_list();
        assert!(i.is_empty());

        let d = DeviceExtensions::none().build_extensions_list();
        assert!(d.is_empty());
    }
}
