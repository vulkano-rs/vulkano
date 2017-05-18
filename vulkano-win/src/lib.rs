extern crate vulkano;
extern crate winit;

use std::error;
use std::fmt;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::swapchain::Surface;
use vulkano::swapchain::SurfaceCreationError;
use winit::{EventsLoop, WindowBuilder};
use winit::CreationError as WindowCreationError;

pub fn required_extensions() -> InstanceExtensions {
    let ideal = InstanceExtensions {
        khr_surface: true,
        khr_xlib_surface: true,
        khr_xcb_surface: true,
        khr_wayland_surface: true,
        khr_mir_surface: true,
        khr_android_surface: true,
        khr_win32_surface: true,
        ..InstanceExtensions::none()
    };

    match InstanceExtensions::supported_by_core() {
        Ok(supported) => supported.intersection(&ideal),
        Err(_) => InstanceExtensions::none()
    }
}

pub trait VkSurfaceBuild {
    fn build_vk_surface(self, events_loop: &EventsLoop, instance: &Arc<Instance>) -> Result<Window, CreationError>;
}

impl VkSurfaceBuild for WindowBuilder {
    fn build_vk_surface(self, events_loop: &EventsLoop, instance: &Arc<Instance>) -> Result<Window, CreationError> {
        let window = try!(self.build(events_loop));
        let surface = try!(unsafe { winit_to_surface(instance, &window) });

        Ok(Window {
            window: window,
            surface: surface,
        })
    }
}

pub trait IntoVkWindow<W> {
    fn into_vk_win(self, instance: &Arc<Instance>) -> Result<Window<W>, CreationError>;
}

impl IntoVkWindow<winit::Window> for winit::Window {
    fn into_vk_win(self, instance: &Arc<Instance>) -> Result<Window<winit::Window>, CreationError> {
        let surface = try!(unsafe { winit_to_surface(instance, &self) });
        Ok(Window {
            window: self,
            surface: surface,
        })
    }
}

impl IntoVkWindow<Arc<Mutex<winit::Window>>> for Arc<Mutex<winit::Window>> {
    fn into_vk_win(self, instance: &Arc<Instance>) -> Result<Window<Arc<Mutex<winit::Window>>>, CreationError> {
        let surface = try!(unsafe { winit_to_surface(instance, &self.lock().unwrap()) });
        Ok(Window {
            window: self,
            surface: surface,
        })
    }
}

pub struct Window<W = winit::Window> {
    window: W,
    surface: Arc<Surface>,
}

impl <W> Window <W> {
    #[inline]
    pub fn window(&self) -> &W {
        &self.window
    }

    #[inline]
    pub fn surface(&self) -> &Arc<Surface> {
        &self.surface
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
    fn description(&self) -> &str {
        match *self {
            CreationError::SurfaceCreationError(_) => "error while creating the surface",
            CreationError::WindowCreationError(_) => "error while creating the window",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            CreationError::SurfaceCreationError(ref err) => Some(err),
            CreationError::WindowCreationError(ref err) => Some(err),
        }
    }
}

impl fmt::Display for CreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
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
unsafe fn winit_to_surface(instance: &Arc<Instance>,
                           win: &winit::Window)
                           -> Result<Arc<Surface>, SurfaceCreationError> {
    use winit::os::android::WindowExt;
    Surface::from_anativewindow(instance, win.get_native_window())
}

#[cfg(all(unix, not(target_os = "android")))]
unsafe fn winit_to_surface(instance: &Arc<Instance>,
                           win: &winit::Window)
                           -> Result<Arc<Surface>, SurfaceCreationError> {
    use winit::os::unix::WindowExt;
    match (win.get_wayland_display(), win.get_wayland_surface()) {
        (Some(display), Some(surface)) => Surface::from_wayland(instance, display, surface),
        _ => {
            // No wayland display found, check if we can use xlib.
            // If not, we use xcb.
            if instance.loaded_extensions().khr_xlib_surface {
                Surface::from_xlib(instance,
                                   win.get_xlib_display().unwrap(),
                                   win.get_xlib_window().unwrap() as _)
            } else {
                Surface::from_xcb(instance,
                                   win.get_xcb_connection().unwrap(),
                                   win.get_xlib_window().unwrap() as _)
            }
        }
    }
}

#[cfg(windows)]
unsafe fn winit_to_surface(instance: &Arc<Instance>,
                           win: &winit::Window)
                           -> Result<Arc<Surface>, SurfaceCreationError> {
    use winit::os::windows::WindowExt;
    Surface::from_hwnd(instance,
                       ptr::null() as *const (), // FIXME
                       win.get_hwnd())
}
