// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::HashSet;
use std::error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::iter::FromIterator;
use std::ptr;
use std::str;

use Error;
use OomError;
use VulkanObject;
use check_errors;
use instance::PhysicalDevice;
use instance::loader;
use instance::loader::LoadingError;
use vk;

macro_rules! extensions {
    ($sname:ident, $rawname:ident, $($ext:ident => $s:expr,)*) => (
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

            /// Returns the difference of another list from this list.
            #[inline]
            pub fn difference(&self, other: &$sname) -> $sname {
                $sname {
                    $(
                        $ext: self.$ext && !other.$ext,
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

        /// Set of extensions, not restricted to those vulkano knows about.
        ///
        /// This is useful when interacting with external code that has statically-unknown extension
        /// requirements.
        #[derive(Clone, Eq, PartialEq)]
        pub struct $rawname(HashSet<CString>);

        impl $rawname {
            /// Constructs an extension set containing the supplied extensions.
            pub fn new<I>(extensions: I) -> Self
                where I: IntoIterator<Item=CString>
            {
                $rawname(extensions.into_iter().collect())
            }

            /// Constructs an empty extension set.
            pub fn none() -> Self { $rawname(HashSet::new()) }

            /// Adds an extension to the set if it is not already present.
            pub fn insert(&mut self, extension: CString) {
                self.0.insert(extension);
            }

            /// Returns the intersection of this set and another.
            pub fn intersection(&self, other: &Self) -> Self {
                $rawname(self.0.intersection(&other.0).cloned().collect())
            }

            /// Returns the difference of another set from this one.
            pub fn difference(&self, other: &Self) -> Self {
                $rawname(self.0.difference(&other.0).cloned().collect())
            }

            /// Returns the union of both extension sets
            pub fn union(&self, other: &Self) -> Self {
                $rawname(self.0.union(&other.0).cloned().collect())
            }

            // TODO: impl Iterator
            pub fn iter(&self) -> ::std::collections::hash_set::Iter<CString> { self.0.iter() }
        }

        impl fmt::Debug for $rawname {
            #[allow(unused_assignments)]
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl FromIterator<CString> for $rawname {
            fn from_iter<T>(iter: T) -> Self
                where T: IntoIterator<Item = CString>
            {
                $rawname(iter.into_iter().collect())
            }
        }

        impl<'a> From<&'a $sname> for $rawname {
            fn from(x: &'a $sname) -> Self {
                let mut data = HashSet::new();
                $(if x.$ext { data.insert(CString::new(&$s[..]).unwrap()); })*
                $rawname(data)
            }
        }

        impl<'a> From<&'a $rawname> for $sname {
            fn from(x: &'a $rawname) -> Self {
                let mut extensions = $sname::none();
                $(
                    if x.0.iter().any(|x| x.as_bytes() == &$s[..]) {
                        extensions.$ext = true;
                    }
                )*
                extensions
            }
        }
    );
}

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

macro_rules! device_extensions {
    ($sname:ident, $rawname:ident, $($ext:ident => $s:expr,)*) => (
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
                    try!(check_errors(vk.EnumerateDeviceExtensionProperties(
                        physical_device.internal_object(), ptr::null(), &mut num, ptr::null_mut())));

                    let mut properties = Vec::with_capacity(num as usize);
                    try!(check_errors(vk.EnumerateDeviceExtensionProperties(
                        physical_device.internal_object(), ptr::null(), &mut num, properties.as_mut_ptr())));
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
                    try!(check_errors(vk.EnumerateDeviceExtensionProperties(
                        physical_device.internal_object(), ptr::null(), &mut num, ptr::null_mut())));

                    let mut properties = Vec::with_capacity(num as usize);
                    try!(check_errors(vk.EnumerateDeviceExtensionProperties(
                        physical_device.internal_object(), ptr::null(), &mut num, properties.as_mut_ptr())));
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
    khr_mir_surface => b"VK_KHR_mir_surface",
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

device_extensions! {
    DeviceExtensions,
    RawDeviceExtensions,
    khr_swapchain => b"VK_KHR_swapchain",
    khr_display_swapchain => b"VK_KHR_display_swapchain",
    khr_sampler_mirror_clamp_to_edge => b"VK_KHR_sampler_mirror_clamp_to_edge",
    khr_maintenance1 => b"VK_KHR_maintenance1",
    khr_get_memory_requirements2 => b"VK_KHR_get_memory_requirements2",
    khr_dedicated_allocation => b"VK_KHR_dedicated_allocation",
    khr_incremental_present => b"VK_KHR_incremental_present",
    ext_debug_marker => b"VK_EXT_debug_marker",
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
            _ => panic!("unexpected error: {:?}", err),
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
    use instance::{DeviceExtensions, RawDeviceExtensions};
    use instance::{InstanceExtensions, RawInstanceExtensions};

    #[test]
    fn empty_extensions() {
        let i: RawInstanceExtensions = (&InstanceExtensions::none()).into();
        assert!(i.iter().next().is_none());

        let d: RawDeviceExtensions = (&DeviceExtensions::none()).into();
        assert!(d.iter().next().is_none());
    }
}
