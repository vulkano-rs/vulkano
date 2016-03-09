use std::ffi::CString;

macro_rules! extensions {
    ($($ext:ident => $s:expr,)*) => (
        /// List of extensions that are enabled or available.
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        #[allow(missing_docs)]
        pub struct Extensions {
            $(
                pub $ext: bool,
            )*
        }

        impl Extensions {
            /// Returns an `Extensions` object with all members set to `false`.
            #[inline]
            pub fn none() -> Extensions {
                Extensions {
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
    khr_surface => b"VK_KHR_surface",
    khr_swapchain => b"VK_KHR_swapchain",
    khr_display => b"VK_KHR_display",
    khr_display_swapchain => b"VK_KHR_display_swapchain",
    khr_xlib_surface => b"VK_KHR_xlib_surface",
    khr_xcb_surface => b"VK_KHR_xcb_surface",
    khr_wayland_surface => b"VK_KHR_wayland_surface",
    khr_mir_surface => b"VK_KHR_mir_surface",
    khr_android_surface => b"VK_KHR_android_surface",
    khr_win32_surface => b"VK_KHR_win32_surface",
    ext_debug_report => b"VK_EXT_debug_report",
}

#[cfg(test)]
mod tests {
    use instance::Extensions;

    #[test]
    fn empty_extensions() {
        let s = Extensions::none().build_extensions_list();
        assert!(s.is_empty());
    }
}
