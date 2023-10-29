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

use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

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
    
    let window_handle = {
        match window.window_handle() {
            Ok(window_handle) => window_handle,
            Err(e) => unreachable!("Window handle unobtainable : {}",e)
        }
    };
    
    if let RawWindowHandle::AndroidNdk(surface) = window_handle.as_raw()
    {
        return Surface::from_android(instance,
            surface.a_native_window.as_ptr(),
            Some(window));
    }
    
    unreachable!("This should be unreachable if the target is android");
}

#[cfg(all(
    unix,
    not(target_os = "android"),
    not(target_os = "macos"),
    not(target_os = "ios")
))]
unsafe fn winit_to_surface(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    
    let window_handle = {
        match window.window_handle() {
            Ok(window_handle) => window_handle,
            Err(e) => unreachable!("Window handle unobtainable : {}",e)
        }
    };
    let display_handle = {
        match window.display_handle() {
            Ok(display_handle) => display_handle,
            Err(e) => unreachable!("Display handle unobtainable : {}",e)
        }
    };
    
    if let RawWindowHandle::Wayland(window_handle_raw) = window_handle.as_raw()
    {
        if let RawDisplayHandle::Wayland(display_raw) = display_handle.as_raw()
        {
            return Surface::from_wayland(instance,
                display_raw.display.as_ptr(),
                window_handle_raw.surface.as_ptr(),
                Some(window)
            );
        }
    }
    
    if instance.enabled_extensions().khr_xlib_surface
    {
        if let RawWindowHandle::Xlib(window_handle_raw) = window_handle.as_raw()
        {
            if let RawDisplayHandle::Xlib(display_raw) = display_handle.as_raw()
            {
                return Surface::from_xlib(
                    instance,
                    display_raw.display.unwrap().as_ptr(),
                    window_handle_raw.window,
                    Some(window),
                );
            }
        }
    }
    else if let RawWindowHandle::Xcb(window_handle_raw) = window_handle.as_raw()
    {
        if let RawDisplayHandle::Xcb(display_raw) = display_handle.as_raw()
        {
            return Surface::from_xcb(
                instance,
                display_raw.connection.unwrap().as_ptr(),
                window_handle_raw.window.get(),
                Some(window),
            );
        }
    }
    
    unreachable!("This should be unreachable.");
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
unsafe fn winit_to_surface(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    use winit::platform::macos::WindowExtMacOS;
    let layer = get_metal_layer_macos(window.ns_screen().unwrap());
    Surface::from_mac_os(instance, layer as *const (), Some(window))
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
unsafe fn winit_to_surface(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    use winit::platform::ios::WindowExtIOS;
    let layer = get_metal_layer_ios(window.ui_screen());
    Surface::from_ios(instance, layer, Some(window))
}

#[cfg(target_os = "windows")]
unsafe fn winit_to_surface(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> Result<Arc<Surface>, Validated<VulkanError>> {
    use winit::platform::windows::WindowExtWindows;
    
    let window_handle = {
        match window.window_handle() {
            Ok(window_handle) => window_handle,
            Err(e) => unreachable!("Window handle unobtainable : {}",e)
        }
    };
    
    if let RawWindowHandle::Win32(window_handle_raw) = window_handle.as_raw()
    {
        return Surface::from_win32(
            instance,
            window_handle_raw.hinstance.unwrap().get() as *const (),
            window_handle_raw.hwnd.get() as *const (),
            Some(Arc::new(window)),
        );
    }
    
    unreachable!("This should be unreachable.");
}

#[cfg(target_os = "windows")]
use vulkano::swapchain::Win32Monitor;
#[cfg(target_os = "windows")]
use winit::{monitor::MonitorHandle, platform::windows::MonitorHandleExtWindows};
#[cfg(target_os = "windows")]
use winit::raw_window_handle::{HandleError, WindowHandle};

#[cfg(target_os = "windows")]
/// Creates a `Win32Monitor` from a Winit monitor handle.
#[inline]
pub fn create_win32_monitor_from_winit(monitor_handle: &MonitorHandle) -> Win32Monitor {
    unsafe { Win32Monitor::new(monitor_handle.hmonitor() as *const ()) }
}
