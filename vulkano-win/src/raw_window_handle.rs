use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::Arc;
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;
use vulkano::swapchain::SurfaceCreationError;

/// Creates a vulkan surface from a generic window
/// which implements HasRawWindowHandle and thus can reveal the os-dependent handle
pub fn create_vk_surface_from_handle<W>(
    window: W,
    instance: Arc<Instance>,
) -> Result<Arc<Surface<W>>, SurfaceCreationError>
where
    W: HasRawWindowHandle,
{
    unsafe {
        match window.raw_window_handle() {
            RawWindowHandle::UiKit(h) => Surface::from_ios_moltenvk(instance, h.ui_view, window),
            RawWindowHandle::AppKit(h) => Surface::from_macos_moltenvk(instance, h.ns_view, window),
            RawWindowHandle::Xlib(h) => Surface::from_xlib(instance, h.display, h.window, window),
            RawWindowHandle::Xcb(h) => Surface::from_xcb(instance, h.connection, h.window, window),
            RawWindowHandle::Wayland(h) => {
                Surface::from_wayland(instance, h.display, h.surface, window)
            }
            RawWindowHandle::AndroidNdk(h) => {
                Surface::from_anativewindow(instance, h.a_native_window, window)
            }
            RawWindowHandle::Win32(h) => Surface::from_hwnd(instance, h.hinstance, h.hwnd, window),
            RawWindowHandle::Web(_) => unimplemented!(),
            _ => unimplemented!(),
        }
    }
}
