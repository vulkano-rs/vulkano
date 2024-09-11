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
            #[cfg(target_vendor = "apple")]
            RawWindowHandle::AppKit(handle) => {
                let view = std::ptr::NonNull::new(handle.ns_view).unwrap();
                let layer = raw_window_metal::Layer::from_ns_view(view);

                // Vulkan retains the CAMetalLayer, so no need to retain it past this invocation
                Surface::from_metal(instance, layer.as_ptr(), Some(window))
            }
            #[cfg(target_vendor = "apple")]
            RawWindowHandle::UiKit(handle) => {
                let view = std::ptr::NonNull::new(handle.ui_view).unwrap();
                let layer = raw_window_metal::Layer::from_ui_view(view);

                // Vulkan retains the CAMetalLayer, so no need to retain it past this invocation
                Surface::from_metal(instance, layer.as_ptr(), Some(window))
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
            #[cfg(target_vendor = "apple")]
            (RawWindowHandle::AppKit(handle), _) => {
                let view = std::ptr::NonNull::new(handle.ns_view).unwrap();
                let layer = raw_window_metal::Layer::from_ns_view(view);

                // Vulkan retains the CAMetalLayer, so no need to retain it past this invocation
                Surface::from_metal(instance, layer.as_ptr(), None)
            }
            #[cfg(target_vendor = "apple")]
            (RawWindowHandle::UiKit(handle), _) => {
                let view = std::ptr::NonNull::new(handle.ui_view).unwrap();
                let layer = raw_window_metal::Layer::from_ui_view(view);

                // Vulkan retains the CAMetalLayer, so no need to retain it past this invocation
                Surface::from_metal(instance, layer.as_ptr(), None)
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
