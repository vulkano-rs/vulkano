// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ffi::CString;
use std::ptr;

use OomError;
use VK_ENTRY;
use vk;
use check_errors;

macro_rules! extensions {
    ($sname:ident, $($ext:ident => $s:expr,)*) => (
        /// List of extensions that are enabled or available.
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        #[allow(missing_docs)]
        pub struct $sname {
            $(
                pub $ext: bool,
            )*
        }

        impl $sname {
            /// Returns an `Extensions` object with all members set to `false`.
            #[inline]
            pub fn none() -> $sname {
                $sname {
                    $($ext: false,)*
                }
            }

            /// Builds a Vec containing the list of extensions.
            pub fn build_extensions_list(&self) -> Vec<CString> {
                let mut data = Vec::new();
                $(if self.$ext { data.push(CString::new(&$s[..]).unwrap()); })*
                data
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
            pub fn supported_by_core_raw() -> Result<$sname, OomError> {
                let properties: Vec<vk::ExtensionProperties> = unsafe {
                    let mut num = 0;
                    try!(check_errors(VK_ENTRY.EnumerateInstanceExtensionProperties(
                        ptr::null(), &mut num, ptr::null_mut())));
                    
                    let mut properties = Vec::with_capacity(num as usize);
                    try!(check_errors(VK_ENTRY.EnumerateInstanceExtensionProperties(
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
            pub fn supported_by_core() -> $sname {
                $sname::supported_raw().unwrap()
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
}

extensions! {
    DeviceExtensions,
    khr_swapchain => b"VK_KHR_swapchain",
    khr_display_swapchain => b"VK_KHR_display_swapchain",
}

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
