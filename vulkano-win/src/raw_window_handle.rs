#[cfg(target_os = "ios")]
use crate::get_metal_layer_ios;
#[cfg(target_os = "macos")]
use crate::get_metal_layer_macos;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::Arc;
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;
use vulkano::swapchain::SurfaceCreationError;

/// Creates a vulkan surface from a generic window
/// which implements HasRawWindowHandle and thus can reveal the os-dependent handle.
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
            RawWindowHandle::UiKit(_h) => {
                #[cfg(target_os = "ios")]
                {
                    // Ensure the layer is CAMetalLayer
                    let layer = get_metal_layer_ios(_h.ui_view);
                    Surface::from_ios(instance, layer, window)
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
                    Surface::from_mac_os(instance, layer as *const (), window)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    panic!("AppKit handle should only be used when target_os == 'ios'");
                }
            }
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
