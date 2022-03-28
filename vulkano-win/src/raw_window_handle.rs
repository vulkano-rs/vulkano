use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::Arc;
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;
use vulkano::swapchain::SurfaceCreationError;

/// Creates a vulkan surface from a generic window
/// which implements HasRawWindowHandle and thus can reveal the os-dependent handle.
/// - Note that if you wish to use this function with MacOS, you will need to ensure that the
/// `CAMetalLayer` is set to the ns_view. An example of how one might do that can be found in
/// `vulkano_win::set_ca_metal_layer_to_winit`
pub fn create_surface_from_handle<W>(
    window: W,
    instance: Arc<Instance>,
) -> Result<Arc<Surface<W>>, SurfaceCreationError>
where
    W: HasRawWindowHandle,
{
    unsafe {
        match window.raw_window_handle() {
            RawWindowHandle::AndroidNdk(h) => {
                Surface::from_android(instance, h.a_native_window, window)
            }
            RawWindowHandle::UiKit(h) => Surface::from_ios(instance, h.ui_view, window),
            RawWindowHandle::AppKit(h) => Surface::from_mac_os(instance, h.ns_view, window),
            RawWindowHandle::Wayland(h) => {
                Surface::from_wayland(instance, h.display, h.surface, window)
            }
            RawWindowHandle::Win32(h) => Surface::from_win32(instance, h.hinstance, h.hwnd, window),
            RawWindowHandle::Xcb(h) => Surface::from_xcb(instance, h.connection, h.window, window),
            RawWindowHandle::Xlib(h) => Surface::from_xlib(instance, h.display, h.window, window),
            RawWindowHandle::Web(_) => unimplemented!(),
            _ => unimplemented!(),
        }
    }
}
