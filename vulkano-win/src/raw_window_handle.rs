#[cfg(target_os = "ios")]
use crate::get_metal_layer_ios;
#[cfg(target_os = "macos")]
use crate::get_metal_layer_macos;
use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use std::{any::Any, sync::Arc};
use vulkano::{instance::Instance, swapchain::Surface, Validated, VulkanError};

/// Creates a Vulkan surface from a generic window which implements `HasRawWindowHandle` and thus
/// can reveal the OS-dependent handle.
pub fn create_surface_from_handle(
    window: Arc<impl Any + Send + Sync + HasRawWindowHandle + HasRawDisplayHandle>,
    instance: Arc<Instance>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    unsafe {
        match window.raw_window_handle() {
            RawWindowHandle::AndroidNdk(h) => {
                Surface::from_android(instance, h.a_native_window, Some(window))
            }
            RawWindowHandle::UiKit(_h) => {
                #[cfg(target_os = "ios")]
                {
                    // Ensure the layer is CAMetalLayer
                    let layer = get_metal_layer_ios(_h.ui_view);
                    Surface::from_ios(instance, layer, Some(window))
                }
                #[cfg(not(target_os = "ios"))]
                {
                    panic!("UiKit handle should only be used when target_os == 'ios'");
                }
            }
            RawWindowHandle::AppKit(_h) => {
                #[cfg(target_os = "macos")]
                {
                    // Ensure the layer is CAMetalLayer
                    let layer = get_metal_layer_macos(_h.ns_view);
                    Surface::from_mac_os(instance, layer as *const c_void, Some(window))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    panic!("AppKit handle should only be used when target_os == 'macos'");
                }
            }
            RawWindowHandle::Wayland(h) => {
                let d = match window.raw_display_handle() {
                    RawDisplayHandle::Wayland(d) => d,
                    _ => panic!("Invalid RawDisplayHandle"),
                };
                Surface::from_wayland(instance, d.display, h.surface, Some(window))
            }
            RawWindowHandle::Win32(h) => {
                Surface::from_win32(instance, h.hinstance, h.hwnd, Some(window))
            }
            RawWindowHandle::Xcb(h) => {
                let d = match window.raw_display_handle() {
                    RawDisplayHandle::Xcb(d) => d,
                    _ => panic!("Invalid RawDisplayHandle"),
                };
                Surface::from_xcb(instance, d.connection, h.window, Some(window))
            }
            RawWindowHandle::Xlib(h) => {
                let d = match window.raw_display_handle() {
                    RawDisplayHandle::Xlib(d) => d,
                    _ => panic!("Invalid RawDisplayHandle"),
                };
                Surface::from_xlib(instance, d.display as _, h.window, Some(window))
            }
            RawWindowHandle::Web(_) => unimplemented!(),
            _ => unimplemented!(),
        }
    }
}

/// Creates a Vulkan surface from a generic window which implements `HasRawWindowHandle` and thus
/// can reveal the OS-dependent handle, without ensuring that the window outlives the surface.
///
/// # Safety
///
/// - The passed-in `window` must outlive the created [`Surface`].
pub unsafe fn create_surface_from_handle_ref(
    window: &(impl HasRawWindowHandle + HasRawDisplayHandle),
    instance: Arc<Instance>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    unsafe {
        match window.raw_window_handle() {
            RawWindowHandle::AndroidNdk(h) => {
                Surface::from_android(instance, h.a_native_window, None)
            }
            RawWindowHandle::UiKit(_h) => {
                #[cfg(target_os = "ios")]
                {
                    // Ensure the layer is CAMetalLayer
                    let metal_layer = get_metal_layer_ios(_h.ui_view);
                    Surface::from_ios(instance, metal_layer.render_layer.0 as *const c_void, None)
                }
                #[cfg(not(target_os = "ios"))]
                {
                    panic!("UiKit handle should only be used when target_os == 'ios'");
                }
            }
            RawWindowHandle::AppKit(_h) => {
                #[cfg(target_os = "macos")]
                {
                    // Ensure the layer is CAMetalLayer
                    let metal_layer = get_metal_layer_macos(_h.ns_view);
                    Surface::from_mac_os(instance, layer as *const c_void, None)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    panic!("AppKit handle should only be used when target_os == 'macos'");
                }
            }
            RawWindowHandle::Wayland(h) => {
                let d = match window.raw_display_handle() {
                    RawDisplayHandle::Wayland(d) => d,
                    _ => panic!("Invalid RawDisplayHandle"),
                };
                Surface::from_wayland(instance, d.display, h.surface, None)
            }
            RawWindowHandle::Win32(h) => Surface::from_win32(instance, h.hinstance, h.hwnd, None),
            RawWindowHandle::Xcb(h) => {
                let d = match window.raw_display_handle() {
                    RawDisplayHandle::Xcb(d) => d,
                    _ => panic!("Invalid RawDisplayHandle"),
                };
                Surface::from_xcb(instance, d.connection, h.window, None)
            }
            RawWindowHandle::Xlib(h) => {
                let d = match window.raw_display_handle() {
                    RawDisplayHandle::Xlib(d) => d,
                    _ => panic!("Invalid RawDisplayHandle"),
                };
                Surface::from_xlib(instance, d.display as _, h.window, None)
            }
            RawWindowHandle::Web(_) => unimplemented!(),
            _ => unimplemented!(),
        }
    }
}
