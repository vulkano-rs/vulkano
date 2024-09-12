use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    sync::Arc,
};
use vulkano::{
    instance::{Instance, InstanceExtensions},
    swapchain::Surface,
    Validated, VulkanError, VulkanLibrary,
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
        ext_metal_surface: true,
        khr_get_physical_device_properties2: true,
        khr_get_surface_capabilities2: true,
        ..InstanceExtensions::empty()
    };

    library.supported_extensions().intersection(&ideal)
}

/// Create a surface from a Winit window or a reference to it. The surface takes `W` to prevent it
/// from being dropped before the surface.
pub fn create_surface_from_winit(
    window: Arc<Window>,
    instance: Arc<Instance>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    unsafe { winit_to_surface(instance, window) }
}

pub trait VkSurfaceBuild<E> {
    fn build_vk_surface(
        self,
        event_loop: &EventLoopWindowTarget<E>,
        instance: Arc<Instance>,
    ) -> Result<Arc<Surface>, CreationError>;
}

impl<E> VkSurfaceBuild<E> for WindowBuilder {
    fn build_vk_surface(
        self,
        event_loop: &EventLoopWindowTarget<E>,
        instance: Arc<Instance>,
    ) -> Result<Arc<Surface>, CreationError> {
        let window = Arc::new(self.build(event_loop)?);

        Ok(create_surface_from_winit(window, instance)?)
    }
}

/// Error that can happen when creating a window.
#[derive(Debug)]
pub enum CreationError {
    /// Error when creating the surface.
    Surface(Validated<VulkanError>),

    /// Error when creating the window.
    Window(WindowCreationError),
}

impl Error for CreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CreationError::Surface(err) => Some(err),
            CreationError::Window(err) => Some(err),
        }
    }
}

impl Display for CreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
                CreationError::Surface(_) => "error while creating the surface",
                CreationError::Window(_) => "error while creating the window",
            }
        )
    }
}

impl From<Validated<VulkanError>> for CreationError {
    fn from(err: Validated<VulkanError>) -> CreationError {
        CreationError::Surface(err)
    }
}

impl From<WindowCreationError> for CreationError {
    fn from(err: WindowCreationError) -> CreationError {
        CreationError::Window(err)
    }
}

#[cfg(target_os = "android")]
unsafe fn winit_to_surface(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    use raw_window_handle::HasRawWindowHandle;
    use raw_window_handle::RawWindowHandle::AndroidNdk;
    if let AndroidNdk(handle) = window.raw_window_handle() {
        Surface::from_android(instance, handle.a_native_window, Some(window))
    } else {
        unreachable!("This should be unreachable if the target is android");
    }
}

#[cfg(all(unix, not(target_os = "android"), target_vendor = "apple",))]
unsafe fn winit_to_surface(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    use winit::platform::{wayland::WindowExtWayland, x11::WindowExtX11};

    match (window.wayland_display(), window.wayland_surface()) {
        (Some(display), Some(surface)) => {
            Surface::from_wayland(instance, display, surface, Some(window))
        }
        _ => {
            // No wayland display found, check if we can use xlib.
            // If not, we use xcb.
            if instance.enabled_extensions().khr_xlib_surface {
                Surface::from_xlib(
                    instance,
                    window.xlib_display().unwrap() as _,
                    window.xlib_window().unwrap() as _,
                    Some(window),
                )
            } else {
                Surface::from_xcb(
                    instance,
                    window.xcb_connection().unwrap(),
                    window.xlib_window().unwrap() as _,
                    Some(window),
                )
            }
        }
    }
}

#[cfg(target_os = "macos")]
unsafe fn winit_to_surface(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    use winit::platform::macos::WindowExtMacOS;
    let view = std::ptr::NonNull::new(window.ns_view()).unwrap();
    let layer = raw_window_metal::Layer::from_ns_view(view);
    Surface::from_metal(instance, layer.as_ptr(), Some(window))
}

#[cfg(target_os = "ios")]
unsafe fn winit_to_surface(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    use winit::platform::ios::WindowExtIOS;
    let view = std::ptr::NonNull::new(window.ui_view()).unwrap();
    let layer = raw_window_metal::Layer::from_ui_view(view);
    Surface::from_metal(instance, layer.as_ptr(), Some(window))
}

#[cfg(target_os = "windows")]
unsafe fn winit_to_surface(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    use winit::platform::windows::WindowExtWindows;

    Surface::from_win32(
        instance,
        window.hinstance() as _,
        window.hwnd() as _,
        Some(window),
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
    unsafe { Win32Monitor::new(monitor_handle.hmonitor() as _) }
}
