use std::borrow::Borrow;
use std::error;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::VulkanLibrary;
use vulkano::swapchain::Surface;
use vulkano::swapchain::SurfaceCreationError;
use winit::{
    error::OsError as WindowCreationError,
    event_loop::EventLoopWindowTarget,
    window::{Window, WindowBuilder},
};

pub fn required_extensions(lib: &VulkanLibrary) -> InstanceExtensions {
    let ideal = InstanceExtensions {
        khr_surface: true,
        khr_xlib_surface: true,
        khr_xcb_surface: true,
        khr_wayland_surface: true,
        khr_android_surface: true,
        khr_win32_surface: true,
        mvk_ios_surface: true,
        mvk_macos_surface: true,
        khr_get_physical_device_properties2: true,
        khr_get_surface_capabilities2: true,
        ..InstanceExtensions::none()
    };

    match InstanceExtensions::supported_by_core_with_loader(lib) {
        Ok(supported) => supported.intersection(&ideal),
        Err(_) => InstanceExtensions::none(),
    }
}

/// Create a surface from a Winit window or a reference to it. The surface takes `W` to prevent it
/// from being dropped before the surface.
#[inline]
pub fn create_surface_from_winit<W>(
    window: W,
    instance: Arc<Instance>,
) -> Result<Arc<Surface<W>>, SurfaceCreationError>
where
    W: SafeBorrow<Window>,
{
    unsafe { winit_to_surface(instance, window) }
}

pub trait VkSurfaceBuild<E> {
    fn build_vk_surface(
        self,
        event_loop: &EventLoopWindowTarget<E>,
        instance: Arc<Instance>,
    ) -> Result<Arc<Surface<Window>>, CreationError>;
}

impl<E> VkSurfaceBuild<E> for WindowBuilder {
    #[inline]
    fn build_vk_surface(
        self,
        event_loop: &EventLoopWindowTarget<E>,
        instance: Arc<Instance>,
    ) -> Result<Arc<Surface<Window>>, CreationError> {
        let window = self.build(event_loop)?;
        Ok(create_surface_from_winit(window, instance)?)
    }
}

/// Error that can happen when creating a window.
#[derive(Debug)]
pub enum CreationError {
    /// Error when creating the surface.
    SurfaceCreationError(SurfaceCreationError),
    /// Error when creating the window.
    WindowCreationError(WindowCreationError),
}

impl error::Error for CreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            CreationError::SurfaceCreationError(ref err) => Some(err),
            CreationError::WindowCreationError(ref err) => Some(err),
        }
    }
}

impl fmt::Display for CreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CreationError::SurfaceCreationError(_) => "error while creating the surface",
                CreationError::WindowCreationError(_) => "error while creating the window",
            }
        )
    }
}

impl From<SurfaceCreationError> for CreationError {
    #[inline]
    fn from(err: SurfaceCreationError) -> CreationError {
        CreationError::SurfaceCreationError(err)
    }
}

impl From<WindowCreationError> for CreationError {
    #[inline]
    fn from(err: WindowCreationError) -> CreationError {
        CreationError::WindowCreationError(err)
    }
}

#[cfg(target_os = "android")]
#[inline]
unsafe fn winit_to_surface<W: SafeBorrow<Window>>(
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    use raw_window_handle::HasRawWindowHandle;
    use raw_window_handle::RawWindowHandle::AndroidNdk;
    if let AndroidNdk(handle) = win.borrow().raw_window_handle() {
        Surface::from_android(instance, handle.a_native_window, win)
    } else {
        unreachable!("This should be unreachable if the target is android");
    }
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
unsafe fn winit_to_surface<W: SafeBorrow<Window>>(
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    use winit::platform::unix::WindowExtUnix;

    match (
        win.borrow().wayland_display(),
        win.borrow().wayland_surface(),
    ) {
        (Some(display), Some(surface)) => Surface::from_wayland(instance, display, surface, win),
        _ => {
            // No wayland display found, check if we can use xlib.
            // If not, we use xcb.
            if instance.enabled_extensions().khr_xlib_surface {
                Surface::from_xlib(
                    instance,
                    win.borrow().xlib_display().unwrap(),
                    win.borrow().xlib_window().unwrap() as _,
                    win,
                )
            } else {
                Surface::from_xcb(
                    instance,
                    win.borrow().xcb_connection().unwrap(),
                    win.borrow().xlib_window().unwrap() as _,
                    win,
                )
            }
        }
    }
}

#[cfg(target_os = "macos")]
use cocoa::{
    appkit::{NSView, NSWindow},
    base::id as cocoa_id,
};
#[cfg(target_os = "macos")]
use metal::CoreAnimationLayer;
#[cfg(target_os = "macos")]
use objc::runtime::YES;
#[cfg(target_os = "macos")]
use std::mem;

/// Ensure `CAMetalLayer` (native rendering surface on MacOs) is used by the ns_view.
/// This is necessary to be able to render on Mac.
#[cfg(target_os = "macos")]
unsafe fn set_ca_metal_layer_to_winit<W: SafeBorrow<Window>>(win: W) {
    use winit::platform::macos::WindowExtMacOS;

    let wnd: cocoa_id = mem::transmute(win.borrow().ns_window());
    let layer = CoreAnimationLayer::new();

    layer.set_edge_antialiasing_mask(0);
    layer.set_presents_with_transaction(false);
    layer.remove_all_animations();

    let view = wnd.contentView();

    layer.set_contents_scale(view.backingScaleFactor());
    view.setLayer(mem::transmute(layer.as_ref())); // Bombs here with out of memory
    view.setWantsLayer(YES);
}

#[cfg(target_os = "macos")]
unsafe fn winit_to_surface<W: SafeBorrow<Window>>(
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    use winit::platform::macos::WindowExtMacOS;

    set_ca_metal_layer_to_winit(win.borrow());
    Surface::from_mac_os(instance, win.borrow().ns_view() as *const (), win)
}

#[cfg(target_os = "windows")]
#[inline]
unsafe fn winit_to_surface<W: SafeBorrow<Window>>(
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    use winit::platform::windows::WindowExtWindows;

    Surface::from_win32(instance, win.borrow().hinstance(), win.borrow().hwnd(), win)
}

#[cfg(target_os = "windows")]
use vulkano::swapchain::Win32Monitor;
#[cfg(target_os = "windows")]
use winit::{monitor::MonitorHandle, platform::windows::MonitorHandleExtWindows};

#[cfg(target_os = "windows")]
/// Creates a `Win32Monitor` from a Winit monitor handle.
#[inline]
pub fn create_win32_monitor_from_winit(monitor_handle: &MonitorHandle) -> Win32Monitor {
    unsafe { Win32Monitor::new(monitor_handle.hmonitor()) }
}

/// An alternative to `Borrow<T>` with the requirement that all calls to
/// `borrow` return the same object.
pub unsafe trait SafeBorrow<T>: Borrow<T> {}

unsafe impl<T> SafeBorrow<T> for T {}
unsafe impl<'a, T> SafeBorrow<T> for &'a T {}
unsafe impl<'a, T> SafeBorrow<T> for &'a mut T {}
unsafe impl<T> SafeBorrow<T> for Rc<T> {}
unsafe impl<T> SafeBorrow<T> for Arc<T> {}
unsafe impl<T> SafeBorrow<T> for Box<T> {}
