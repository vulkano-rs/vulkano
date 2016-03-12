use std::ffi::CString;

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

extensions! {
    InstanceExtensions,
    khr_surface => b"VK_KHR_surface",
    khr_display => b"VK_KHR_display",
    khr_display_swapchain => b"VK_KHR_display_swapchain",       // FIXME: device extension
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
}

#[cfg(test)]
mod tests {
    use instance::InstanceExtensions;

    #[test]
    fn empty_extensions() {
        let s = InstanceExtensions::none().build_extensions_list();
        assert!(s.is_empty());
    }
}
