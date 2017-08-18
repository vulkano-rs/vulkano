// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::os::raw::c_ulong;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use format::Format;
use image::ImageUsage;
use instance::Instance;
use instance::PhysicalDevice;
use instance::QueueFamily;
use swapchain::Capabilities;
use swapchain::SurfaceSwapchainLock;
use swapchain::capabilities;
use swapchain::display::DisplayMode;
use swapchain::display::DisplayPlane;

use Error;
use OomError;
use VulkanObject;
use check_errors;
use vk;

/// Represents a surface on the screen.
///
/// Creating a `Surface` is platform-specific.
pub struct Surface {
    instance: Arc<Instance>,
    surface: vk::SurfaceKHR,

    // If true, a swapchain has been associated to this surface, and that any new swapchain
    // creation should be forbidden.
    has_swapchain: AtomicBool,
}

impl Surface {
    /// Creates a `Surface` given the raw handler.
    ///
    /// Be careful when using it
    ///
    pub unsafe fn from_raw_surface(instance: Arc<Instance>, surface: vk::SurfaceKHR) -> Surface {
        Surface {
            instance: instance,
            surface: surface,
            has_swapchain: AtomicBool::new(false),
        }
    }

    /// Creates a `Surface` that covers a display mode.
    ///
    /// # Panic
    ///
    /// - Panics if `display_mode` and `plane` don't belong to the same physical device.
    /// - Panics if `plane` doesn't support the display of `display_mode`.
    ///
    pub fn from_display_mode(display_mode: &DisplayMode, plane: &DisplayPlane)
                             -> Result<Arc<Surface>, SurfaceCreationError> {
        if !display_mode
            .display()
            .physical_device()
            .instance()
            .loaded_extensions()
            .khr_display
        {
            return Err(SurfaceCreationError::MissingExtension { name: "VK_KHR_display" });
        }

        assert_eq!(display_mode.display().physical_device().internal_object(),
                   plane.physical_device().internal_object());
        assert!(plane.supports(display_mode.display()));

        let instance = display_mode.display().physical_device().instance();
        let vk = instance.pointers();

        let surface = unsafe {
            let infos = vk::DisplaySurfaceCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0, // reserved
                displayMode: display_mode.internal_object(),
                planeIndex: plane.index(),
                planeStackIndex: 0, // FIXME: plane.properties.currentStackIndex,
                transform: vk::SURFACE_TRANSFORM_IDENTITY_BIT_KHR, // TODO: let user choose
                globalAlpha: 0.0, // TODO: let user choose
                alphaMode: vk::DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR, // TODO: let user choose
                imageExtent: vk::Extent2D {
                    // TODO: let user choose
                    width: display_mode.visible_region()[0],
                    height: display_mode.visible_region()[1],
                },
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateDisplayPlaneSurfaceKHR(instance.internal_object(),
                                                         &infos,
                                                         ptr::null(),
                                                         &mut output))?;
            output
        };

        Ok(Arc::new(Surface {
                        instance: instance.clone(),
                        surface: surface,
                        has_swapchain: AtomicBool::new(false),
                    }))
    }

    /// Creates a `Surface` from a Win32 window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `hinstance` and the `hwnd` are both correct and stay
    /// alive for the entire lifetime of the surface.
    pub unsafe fn from_hwnd<T, U>(instance: Arc<Instance>, hinstance: *const T, hwnd: *const U)
                                  -> Result<Arc<Surface>, SurfaceCreationError> {
        let vk = instance.pointers();

        if !instance.loaded_extensions().khr_win32_surface {
            return Err(SurfaceCreationError::MissingExtension { name: "VK_KHR_win32_surface" });
        }

        let surface = {
            let infos = vk::Win32SurfaceCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0, // reserved
                hinstance: hinstance as *mut _,
                hwnd: hwnd as *mut _,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateWin32SurfaceKHR(instance.internal_object(),
                                                  &infos,
                                                  ptr::null(),
                                                  &mut output))?;
            output
        };

        Ok(Arc::new(Surface {
                        instance: instance.clone(),
                        surface: surface,
                        has_swapchain: AtomicBool::new(false),
                    }))
    }

    /// Creates a `Surface` from an XCB window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `connection` and the `window` are both correct and stay
    /// alive for the entire lifetime of the surface.
    pub unsafe fn from_xcb<C>(instance: Arc<Instance>, connection: *const C, window: u32)
                              -> Result<Arc<Surface>, SurfaceCreationError> {
        let vk = instance.pointers();

        if !instance.loaded_extensions().khr_xcb_surface {
            return Err(SurfaceCreationError::MissingExtension { name: "VK_KHR_xcb_surface" });
        }

        let surface = {
            let infos = vk::XcbSurfaceCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0, // reserved
                connection: connection as *mut _,
                window: window,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateXcbSurfaceKHR(instance.internal_object(),
                                                &infos,
                                                ptr::null(),
                                                &mut output))?;
            output
        };

        Ok(Arc::new(Surface {
                        instance: instance.clone(),
                        surface: surface,
                        has_swapchain: AtomicBool::new(false),
                    }))
    }

    /// Creates a `Surface` from an Xlib window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `display` and the `window` are both correct and stay
    /// alive for the entire lifetime of the surface.
    pub unsafe fn from_xlib<D>(instance: Arc<Instance>, display: *const D, window: c_ulong)
                               -> Result<Arc<Surface>, SurfaceCreationError> {
        let vk = instance.pointers();

        if !instance.loaded_extensions().khr_xlib_surface {
            return Err(SurfaceCreationError::MissingExtension { name: "VK_KHR_xlib_surface" });
        }

        let surface = {
            let infos = vk::XlibSurfaceCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0, // reserved
                dpy: display as *mut _,
                window: window,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateXlibSurfaceKHR(instance.internal_object(),
                                                 &infos,
                                                 ptr::null(),
                                                 &mut output))?;
            output
        };

        Ok(Arc::new(Surface {
                        instance: instance.clone(),
                        surface: surface,
                        has_swapchain: AtomicBool::new(false),
                    }))
    }

    /// Creates a `Surface` from a Wayland window.
    ///
    /// The window's dimensions will be set to the size of the swapchain.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `display` and the `surface` are both correct and stay
    /// alive for the entire lifetime of the surface.
    pub unsafe fn from_wayland<D, S>(instance: Arc<Instance>, display: *const D,
                                     surface: *const S)
                                     -> Result<Arc<Surface>, SurfaceCreationError> {
        let vk = instance.pointers();

        if !instance.loaded_extensions().khr_wayland_surface {
            return Err(SurfaceCreationError::MissingExtension { name: "VK_KHR_wayland_surface" });
        }

        let surface = {
            let infos = vk::WaylandSurfaceCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0, // reserved
                display: display as *mut _,
                surface: surface as *mut _,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateWaylandSurfaceKHR(instance.internal_object(),
                                                    &infos,
                                                    ptr::null(),
                                                    &mut output))?;
            output
        };

        Ok(Arc::new(Surface {
                        instance: instance.clone(),
                        surface: surface,
                        has_swapchain: AtomicBool::new(false),
                    }))
    }

    /// Creates a `Surface` from a MIR window.
    ///
    /// If the swapchain's dimensions does not match the window's dimensions, the image will
    /// automatically be scaled during presentation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `connection` and the `surface` are both correct and stay
    /// alive for the entire lifetime of the surface.
    pub unsafe fn from_mir<C, S>(instance: Arc<Instance>, connection: *const C, surface: *const S)
                                 -> Result<Arc<Surface>, SurfaceCreationError> {
        let vk = instance.pointers();

        if !instance.loaded_extensions().khr_mir_surface {
            return Err(SurfaceCreationError::MissingExtension { name: "VK_KHR_mir_surface" });
        }

        let surface = {
            let infos = vk::MirSurfaceCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_MIR_SURFACE_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0, // reserved
                connection: connection as *mut _,
                mirSurface: surface as *mut _,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateMirSurfaceKHR(instance.internal_object(),
                                                &infos,
                                                ptr::null(),
                                                &mut output))?;
            output
        };

        Ok(Arc::new(Surface {
                        instance: instance.clone(),
                        surface: surface,
                        has_swapchain: AtomicBool::new(false),
                    }))
    }

    /// Creates a `Surface` from an Android window.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `window` is correct and stays alive for the entire
    /// lifetime of the surface.
    pub unsafe fn from_anativewindow<T>(instance: Arc<Instance>, window: *const T)
                                        -> Result<Arc<Surface>, SurfaceCreationError> {
        let vk = instance.pointers();

        if !instance.loaded_extensions().khr_android_surface {
            return Err(SurfaceCreationError::MissingExtension { name: "VK_KHR_android_surface" });
        }

        let surface = {
            let infos = vk::AndroidSurfaceCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0, // reserved
                window: window as *mut _,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateAndroidSurfaceKHR(instance.internal_object(),
                                                    &infos,
                                                    ptr::null(),
                                                    &mut output))?;
            output
        };

        Ok(Arc::new(Surface {
                        instance: instance.clone(),
                        surface: surface,
                        has_swapchain: AtomicBool::new(false),
                    }))
    }

    /// Creates a `Surface` from an iOS `UIView`.
    ///
    /// # Safety
    ///
    /// - The caller must ensure that the `view` is correct and stays alive for the entire
    ///   lifetime of the surface.
    /// - The `UIView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    pub unsafe fn from_ios_moltenvk<T>(instance: Arc<Instance>, view: *const T)
                                       -> Result<Arc<Surface>, SurfaceCreationError> {
        let vk = instance.pointers();

        if !instance.loaded_extensions().mvk_ios_surface {
            return Err(SurfaceCreationError::MissingExtension { name: "VK_MVK_ios_surface" });
        }

        let surface = {
            let infos = vk::IOSSurfaceCreateInfoMVK {
                sType: vk::STRUCTURE_TYPE_IOS_SURFACE_CREATE_INFO_MVK,
                pNext: ptr::null(),
                flags: 0, // reserved
                pView: view as *const _,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateIOSSurfaceMVK(instance.internal_object(),
                                                &infos,
                                                ptr::null(),
                                                &mut output))?;
            output
        };

        Ok(Arc::new(Surface {
                        instance: instance.clone(),
                        surface: surface,
                        has_swapchain: AtomicBool::new(false),
                    }))
    }

    /// Creates a `Surface` from a MacOS `NSView`.
    ///
    /// # Safety
    ///
    /// - The caller must ensure that the `view` is correct and stays alive for the entire
    ///   lifetime of the surface.
    /// - The `NSView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    pub unsafe fn from_macos_moltenvk<T>(instance: Arc<Instance>, view: *const T)
                                         -> Result<Arc<Surface>, SurfaceCreationError> {
        let vk = instance.pointers();

        if !instance.loaded_extensions().mvk_macos_surface {
            return Err(SurfaceCreationError::MissingExtension { name: "VK_MVK_macos_surface" });
        }

        let surface = {
            let infos = vk::MacOSSurfaceCreateInfoMVK {
                sType: vk::STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK,
                pNext: ptr::null(),
                flags: 0, // reserved
                pView: view as *const _,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateMacOSSurfaceMVK(instance.internal_object(),
                                                  &infos,
                                                  ptr::null(),
                                                  &mut output))?;
            output
        };

        Ok(Arc::new(Surface {
                        instance: instance.clone(),
                        surface: surface,
                        has_swapchain: AtomicBool::new(false),
                    }))
    }

    /// Creates a `Surface` from a `code:nn::code:vi::code:Layer`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `window` is correct and stays alive for the entire
    /// lifetime of the surface.
    pub unsafe fn from_vi_surface<T>(instance: Arc<Instance>, window: *const T)
                                     -> Result<Arc<Surface>, SurfaceCreationError> {
        let vk = instance.pointers();

        if !instance.loaded_extensions().nn_vi_surface {
            return Err(SurfaceCreationError::MissingExtension { name: "VK_NN_vi_surface" });
        }

        let surface = {
            let infos = vk::ViSurfaceCreateInfoNN {
                sType: vk::STRUCTURE_TYPE_VI_SURFACE_CREATE_INFO_NN,
                pNext: ptr::null(),
                flags: 0, // reserved
                window: window as *mut _,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateViSurfaceNN(instance.internal_object(),
                                              &infos,
                                              ptr::null(),
                                              &mut output))?;
            output
        };

        Ok(Arc::new(Surface {
                        instance: instance.clone(),
                        surface: surface,
                        has_swapchain: AtomicBool::new(false),
                    }))
    }

    /// Returns true if the given queue family can draw on this surface.
    // FIXME: vulkano doesn't check this for the moment!
    pub fn is_supported(&self, queue: QueueFamily) -> Result<bool, CapabilitiesError> {
        unsafe {
            let vk = self.instance.pointers();

            let mut output = mem::uninitialized();
            check_errors(vk.GetPhysicalDeviceSurfaceSupportKHR(queue
                                                                   .physical_device()
                                                                   .internal_object(),
                                                               queue.id(),
                                                               self.surface,
                                                               &mut output))?;
            Ok(output != 0)
        }
    }

    /// Retreives the capabilities of a surface when used by a certain device.
    ///
    /// # Panic
    ///
    /// - Panics if the device and the surface don't belong to the same instance.
    ///
    pub fn capabilities(&self, device: PhysicalDevice) -> Result<Capabilities, CapabilitiesError> {
        unsafe {
            assert_eq!(&*self.instance as *const _,
                       &**device.instance() as *const _,
                       "Instance mismatch in Surface::capabilities");

            let vk = self.instance.pointers();

            let caps = {
                let mut out: vk::SurfaceCapabilitiesKHR = mem::uninitialized();
                check_errors(vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(device.internal_object(),
                                                                        self.surface,
                                                                        &mut out))?;
                out
            };

            let formats = {
                let mut num = 0;
                check_errors(vk.GetPhysicalDeviceSurfaceFormatsKHR(device.internal_object(),
                                                                   self.surface,
                                                                   &mut num,
                                                                   ptr::null_mut()))?;

                let mut formats = Vec::with_capacity(num as usize);
                check_errors(vk.GetPhysicalDeviceSurfaceFormatsKHR(device.internal_object(),
                                                                   self.surface,
                                                                   &mut num,
                                                                   formats.as_mut_ptr()))?;
                formats.set_len(num as usize);
                formats
            };

            let modes = {
                let mut num = 0;
                check_errors(vk.GetPhysicalDeviceSurfacePresentModesKHR(device.internal_object(),
                                                                        self.surface,
                                                                        &mut num,
                                                                        ptr::null_mut()))?;

                let mut modes = Vec::with_capacity(num as usize);
                check_errors(vk.GetPhysicalDeviceSurfacePresentModesKHR(device.internal_object(),
                                                                        self.surface,
                                                                        &mut num,
                                                                        modes.as_mut_ptr()))?;
                modes.set_len(num as usize);
                debug_assert!(modes
                                  .iter()
                                  .find(|&&m| m == vk::PRESENT_MODE_FIFO_KHR)
                                  .is_some());
                debug_assert!(modes.iter().count() > 0);
                capabilities::supported_present_modes_from_list(modes.into_iter())
            };

            Ok(Capabilities {
                min_image_count: caps.minImageCount,
                max_image_count: if caps.maxImageCount == 0 { None }
                                 else { Some(caps.maxImageCount) },
                current_extent: if caps.currentExtent.width == 0xffffffff &&
                                   caps.currentExtent.height == 0xffffffff
                {
                    None
                } else {
                    Some([caps.currentExtent.width, caps.currentExtent.height])
                },
                min_image_extent: [caps.minImageExtent.width, caps.minImageExtent.height],
                max_image_extent: [caps.maxImageExtent.width, caps.maxImageExtent.height],
                max_image_array_layers: caps.maxImageArrayLayers,
                supported_transforms: capabilities::surface_transforms_from_bits(caps.supportedTransforms),
                current_transform: capabilities::surface_transforms_from_bits(caps.currentTransform).iter().next().unwrap(),        // TODO:
                supported_composite_alpha: capabilities::supported_composite_alpha_from_bits(caps.supportedCompositeAlpha),
                supported_usage_flags: {
                    let usage = ImageUsage::from_bits(caps.supportedUsageFlags);
                    debug_assert!(usage.color_attachment);  // specs say that this must be true
                    usage
                },
                supported_formats: formats.into_iter().map(|f| {
                    (Format::from_vulkan_num(f.format).unwrap(), capabilities::color_space_from_num(f.colorSpace))
                }).collect(),
                present_modes: modes,
            })
        }
    }

    /// Returns the instance this surface was created with.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
}

unsafe impl SurfaceSwapchainLock for Surface {
    #[inline]
    fn flag(&self) -> &AtomicBool {
        &self.has_swapchain
    }
}

unsafe impl VulkanObject for Surface {
    type Object = vk::SurfaceKHR;

    #[inline]
    fn internal_object(&self) -> vk::SurfaceKHR {
        self.surface
    }
}

impl fmt::Debug for Surface {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan surface {:?}>", self.surface)
    }
}

impl Drop for Surface {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.instance.pointers();
            vk.DestroySurfaceKHR(self.instance.internal_object(), self.surface, ptr::null());
        }
    }
}

/// Error that can happen when creating a debug callback.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SurfaceCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// The extension required for this function was not enabled.
    MissingExtension {
        /// Name of the missing extension.
        name: &'static str,
    },
}

impl error::Error for SurfaceCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            SurfaceCreationError::OomError(_) => "not enough memory available",
            SurfaceCreationError::MissingExtension { .. } =>
                "the extension required for this function was not enabled",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            SurfaceCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SurfaceCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for SurfaceCreationError {
    #[inline]
    fn from(err: OomError) -> SurfaceCreationError {
        SurfaceCreationError::OomError(err)
    }
}

impl From<Error> for SurfaceCreationError {
    #[inline]
    fn from(err: Error) -> SurfaceCreationError {
        match err {
            err @ Error::OutOfHostMemory => SurfaceCreationError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => SurfaceCreationError::OomError(OomError::from(err)),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// Error that can happen when retreiving a surface's capabilities.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CapabilitiesError {
    /// Not enough memory.
    OomError(OomError),

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,
}

impl error::Error for CapabilitiesError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CapabilitiesError::OomError(_) => "not enough memory",
            CapabilitiesError::SurfaceLost => "the surface is no longer valid",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            CapabilitiesError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for CapabilitiesError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for CapabilitiesError {
    #[inline]
    fn from(err: OomError) -> CapabilitiesError {
        CapabilitiesError::OomError(err)
    }
}

impl From<Error> for CapabilitiesError {
    #[inline]
    fn from(err: Error) -> CapabilitiesError {
        match err {
            err @ Error::OutOfHostMemory => CapabilitiesError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => CapabilitiesError::OomError(OomError::from(err)),
            Error::SurfaceLost => CapabilitiesError::SurfaceLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use swapchain::Surface;
    use swapchain::SurfaceCreationError;

    #[test]
    fn khr_win32_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_hwnd(instance, ptr::null::<u8>(), ptr::null::<u8>()) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn khr_xcb_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_xcb(instance, ptr::null::<u8>(), 0) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn khr_xlib_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_xlib(instance, ptr::null::<u8>(), 0) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn khr_wayland_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_wayland(instance, ptr::null::<u8>(), ptr::null::<u8>()) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn khr_mir_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_mir(instance, ptr::null::<u8>(), ptr::null::<u8>()) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn khr_android_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_anativewindow(instance, ptr::null::<u8>()) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }
}
