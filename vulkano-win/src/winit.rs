use std::{borrow::Borrow, error, fmt, rc::Rc, sync::Arc};
use vulkano::{
    instance::{Instance, InstanceExtensions},
    swapchain::{Surface, SurfaceCreationError},
    VulkanLibrary,
};
use winit::{
    error::OsError as WindowCreationError,
    event_loop::EventLoopWindowTarget,
    window::{Window, WindowBuilder},
};

pub fn required_extensions(library: &VulkanLibrary) -> InstanceExtensions {
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

    library.supported_extensions().intersection(&ideal)
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

#[cfg(all(
    unix,
    not(target_os = "android"),
    not(target_os = "macos"),
    not(target_os = "ios")
))]
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

#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::{class, msg_send, runtime::Object, sel, sel_impl};

/// Get (and set) `CAMetalLayer` to ns_view.
/// This is necessary to be able to render on Mac.
#[cfg(target_os = "macos")]
pub(crate) unsafe fn get_metal_layer_macos(view: *mut std::ffi::c_void) -> *mut Object {
    use core_graphics_types::base::CGFloat;
    use objc::runtime::YES;
    use objc::runtime::{BOOL, NO};

    let view: *mut Object = std::mem::transmute(view);
    let main_layer: *mut Object = msg_send![view, layer];
    let class = class!(CAMetalLayer);
    let is_valid_layer: BOOL = msg_send![main_layer, isKindOfClass: class];
    if is_valid_layer == NO {
        let new_layer: *mut Object = msg_send![class, new];
        let () = msg_send![new_layer, setEdgeAntialiasingMask: 0];
        let () = msg_send![new_layer, setPresentsWithTransaction: false];
        let () = msg_send![new_layer, removeAllAnimations];
        let () = msg_send![view, setLayer: new_layer];
        let () = msg_send![view, setWantsLayer: YES];
        let window: *mut Object = msg_send![view, window];
        if !window.is_null() {
            let scale_factor: CGFloat = msg_send![window, backingScaleFactor];
            let () = msg_send![new_layer, setContentsScale: scale_factor];
        }
        new_layer
    } else {
        main_layer
    }
}

#[cfg(target_os = "macos")]
unsafe fn winit_to_surface<W: SafeBorrow<Window>>(
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    use winit::platform::macos::WindowExtMacOS;
    let layer = get_metal_layer_macos(win.borrow().ns_view());
    Surface::from_mac_os(instance, layer as *const (), win)
}

#[cfg(target_os = "ios")]
use vulkano::swapchain::IOSMetalLayer;

/// Get sublayer from iOS main view (ui_view). The sublayer is created as CAMetalLayer
#[cfg(target_os = "ios")]
pub(crate) unsafe fn get_metal_layer_ios(view: *mut std::ffi::c_void) -> IOSMetalLayer {
    use core_graphics_types::{base::CGFloat, geometry::CGRect};

    let view: *mut Object = std::mem::transmute(view);
    let main_layer: *mut Object = msg_send![view, layer];
    let class = class!(CAMetalLayer);
    let new_layer: *mut Object = msg_send![class, new];
    let frame: CGRect = msg_send![main_layer, bounds];
    let () = msg_send![new_layer, setFrame: frame];
    let () = msg_send![main_layer, addSublayer: new_layer];
    let screen: *mut Object = msg_send![class!(UIScreen), mainScreen];
    let scale_factor: CGFloat = msg_send![screen, nativeScale];
    let () = msg_send![view, setContentScaleFactor: scale_factor];
    IOSMetalLayer::new(view, new_layer)
}

#[cfg(target_os = "ios")]
unsafe fn winit_to_surface<W: SafeBorrow<Window>>(
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    use winit::platform::ios::WindowExtIOS;
    let layer = get_metal_layer_ios(win.borrow().ui_view());
    Surface::from_ios(instance, layer, win)
}

#[cfg(target_os = "windows")]
#[inline]
unsafe fn winit_to_surface<W: SafeBorrow<Window>>(
    instance: Arc<Instance>,
    win: W,
) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
    use winit::platform::windows::WindowExtWindows;

    Surface::from_win32(
        instance,
        win.borrow().hinstance() as *const (),
        win.borrow().hwnd() as *const (),
        win,
    )
}

#[cfg(target_os = "windows")]
use vulkano::swapchain::Win32Monitor;
#[cfg(target_os = "windows")]
use winit::{monitor::MonitorHandle, platform::windows::MonitorHandleExtWindows};

#[cfg(target_os = "windows")]
/// Creates a `Win32Monitor` from a Winit monitor handle.
#[inline]
pub fn create_win32_monitor_from_winit(monitor_handle: &MonitorHandle) -> Win32Monitor {
    unsafe { Win32Monitor::new(monitor_handle.hmonitor() as *const ()) }
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
