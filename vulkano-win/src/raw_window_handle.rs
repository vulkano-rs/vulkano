use std::ffi::c_void;
#[cfg(any(
    target_os = "linux",
    target_os = "dragonflybsd",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd"
))]
use std::os::raw::c_ulong;
use std::sync::Arc;

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
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
            #[cfg(target_os = "ios")]
            RawWindowHandle::IOS(h) => handle_to_surface(h.ui_view, instance, window),
            #[cfg(target_os = "macos")]
            RawWindowHandle::MacOS(h) => handle_to_surface(h.ns_view, instance, window),
            #[cfg(any(
                target_os = "linux",
                target_os = "dragonflybsd",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd"
            ))]
            RawWindowHandle::Xlib(h) => {
                handle_to_surface_xlib(h.window, h.display, instance, window)
            }
            #[cfg(any(
                target_os = "linux",
                target_os = "dragonflybsd",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd"
            ))]
            RawWindowHandle::Xcb(h) => {
                handle_to_surface_xcb(h.window, h.connection, instance, window)
            }
            #[cfg(any(
                target_os = "linux",
                target_os = "dragonflybsd",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd"
            ))]
            RawWindowHandle::Wayland(h) => {
                handle_to_surface_wl(h.surface, h.display, instance, window)
            }
            #[cfg(target_os = "android")]
            RawWindowHandle::Android(h) => handle_to_surface(h.a_native_window, instance, window),
            #[cfg(target_os = "windows")]
            RawWindowHandle::Windows(h) => handle_to_surface(h.hinstance, h.hwnd, instance, window),
            #[cfg(target_os = "wasm")]
            RawWindowHandle::Web(_) => unimplemented!(),
            _ => unimplemented!(),
        }
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "dragonflybsd",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd"
))]
unsafe fn handle_to_surface_xlib<W: Sized>(
    window: c_ulong,
    handle: *const c_void,
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    Surface::from_xlib(instance, handle, window, win)
}

#[cfg(any(
    target_os = "linux",
    target_os = "dragonflybsd",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd"
))]
unsafe fn handle_to_surface_xcb<W: Sized>(
    window: u32,
    handle: *mut c_void,
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    Surface::from_xcb(instance, handle as *const _, window, win)
}

#[cfg(any(
    target_os = "linux",
    target_os = "dragonflybsd",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd"
))]
unsafe fn handle_to_surface_wl<W: Sized>(
    surface: *mut c_void,
    display: *mut c_void,
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    Surface::from_wayland(instance, display as *const _, surface as *const _, win)
}

#[cfg(target_os = "macos")]
unsafe fn handle_to_surface<W: Sized>(
    view: *mut c_void,
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    Surface::from_macos_moltenvk(instance, view as *const _, win)
}

#[cfg(target_os = "ios")]
unsafe fn handle_to_surface<W: Sized>(
    view: *mut c_void,
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    Surface::from_ios_moltenvk(instance, view as *const _, win)
}

#[cfg(target_os = "android")]
unsafe fn handle_to_surface<W: Sized>(
    window: *mut c_void,
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    Surface::from_anativewindow(instance, window as *const _, win)
}

#[cfg(target_os = "windows")]
unsafe fn handle_to_surface<W: Sized>(
    hinstance: *mut c_void,
    hwnd: *mut c_void,
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    Surface::from_hwnd(instance, hinstance as *const _, hwnd as *const _, win)
}
