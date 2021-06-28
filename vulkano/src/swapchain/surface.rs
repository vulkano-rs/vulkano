// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::device::physical::PhysicalDevice;
use crate::device::physical::QueueFamily;
use crate::format::Format;
use crate::image::ImageUsage;
use crate::instance::Instance;
use crate::swapchain::capabilities::SupportedSurfaceTransforms;
use crate::swapchain::display::DisplayMode;
use crate::swapchain::display::DisplayPlane;
use crate::swapchain::Capabilities;
use crate::swapchain::SurfaceSwapchainLock;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use std::convert::TryFrom;
use std::error;
use std::fmt;
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Represents a surface on the screen.
///
/// Creating a `Surface` is platform-specific.
pub struct Surface<W> {
    window: W,
    instance: Arc<Instance>,
    surface: ash::vk::SurfaceKHR,

    // If true, a swapchain has been associated to this surface, and that any new swapchain
    // creation should be forbidden.
    has_swapchain: AtomicBool,
}

impl<W> Surface<W> {
    /// Creates a `Surface` given the raw handler.
    ///
    /// Be careful when using it
    ///
    pub unsafe fn from_raw_surface(
        instance: Arc<Instance>,
        surface: ash::vk::SurfaceKHR,
        win: W,
    ) -> Surface<W> {
        Surface {
            window: win,
            instance,
            surface,
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
    pub fn from_display_mode(
        display_mode: &DisplayMode,
        plane: &DisplayPlane,
    ) -> Result<Arc<Surface<()>>, SurfaceCreationError> {
        if !display_mode
            .display()
            .physical_device()
            .instance()
            .enabled_extensions()
            .khr_display
        {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_display",
            });
        }

        assert_eq!(
            display_mode.display().physical_device().internal_object(),
            plane.physical_device().internal_object()
        );
        assert!(plane.supports(display_mode.display()));

        let instance = display_mode.display().physical_device().instance();
        let fns = instance.fns();

        let surface = unsafe {
            let infos = ash::vk::DisplaySurfaceCreateInfoKHR {
                flags: ash::vk::DisplaySurfaceCreateFlagsKHR::empty(),
                display_mode: display_mode.internal_object(),
                plane_index: plane.index(),
                plane_stack_index: 0, // FIXME: plane.properties.currentStackIndex,
                transform: ash::vk::SurfaceTransformFlagsKHR::IDENTITY, // TODO: let user choose
                global_alpha: 0.0,    // TODO: let user choose
                alpha_mode: ash::vk::DisplayPlaneAlphaFlagsKHR::OPAQUE, // TODO: let user choose
                image_extent: ash::vk::Extent2D {
                    // TODO: let user choose
                    width: display_mode.visible_region()[0],
                    height: display_mode.visible_region()[1],
                },
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_display.create_display_plane_surface_khr(
                instance.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            window: (),
            instance: instance.clone(),
            surface,
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
    /// alive for the entire lifetime of the surface. The `win` parameter can be used to ensure this.

    pub unsafe fn from_hwnd<T, U>(
        instance: Arc<Instance>,
        hinstance: *const T,
        hwnd: *const U,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        let fns = instance.fns();

        if !instance.enabled_extensions().khr_win32_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_win32_surface",
            });
        }

        let surface = {
            let infos = ash::vk::Win32SurfaceCreateInfoKHR {
                flags: ash::vk::Win32SurfaceCreateFlagsKHR::empty(),
                hinstance: hinstance as *mut _,
                hwnd: hwnd as *mut _,
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_win32_surface.create_win32_surface_khr(
                instance.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            window: win,
            instance: instance.clone(),
            surface,
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
    /// alive for the entire lifetime of the surface. The `win` parameter can be used to ensure this.
    pub unsafe fn from_xcb<C>(
        instance: Arc<Instance>,
        connection: *const C,
        window: u32,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        let fns = instance.fns();

        if !instance.enabled_extensions().khr_xcb_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_xcb_surface",
            });
        }

        let surface = {
            let infos = ash::vk::XcbSurfaceCreateInfoKHR {
                flags: ash::vk::XcbSurfaceCreateFlagsKHR::empty(),
                connection: connection as *mut _,
                window,
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_xcb_surface.create_xcb_surface_khr(
                instance.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            window: win,
            instance: instance.clone(),
            surface,
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
    /// alive for the entire lifetime of the surface. The `win` parameter can be used to ensure this.
    pub unsafe fn from_xlib<D>(
        instance: Arc<Instance>,
        display: *const D,
        window: c_ulong,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        let fns = instance.fns();

        if !instance.enabled_extensions().khr_xlib_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_xlib_surface",
            });
        }

        let surface = {
            let infos = ash::vk::XlibSurfaceCreateInfoKHR {
                flags: ash::vk::XlibSurfaceCreateFlagsKHR::empty(),
                dpy: display as *mut _,
                window,
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_xlib_surface.create_xlib_surface_khr(
                instance.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            window: win,
            instance: instance.clone(),
            surface,
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
    /// alive for the entire lifetime of the surface. The `win` parameter can be used to ensure this.
    pub unsafe fn from_wayland<D, S>(
        instance: Arc<Instance>,
        display: *const D,
        surface: *const S,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        let fns = instance.fns();

        if !instance.enabled_extensions().khr_wayland_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_wayland_surface",
            });
        }

        let surface = {
            let infos = ash::vk::WaylandSurfaceCreateInfoKHR {
                flags: ash::vk::WaylandSurfaceCreateFlagsKHR::empty(),
                display: display as *mut _,
                surface: surface as *mut _,
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_wayland_surface.create_wayland_surface_khr(
                instance.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            window: win,
            instance: instance.clone(),
            surface,
            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from an Android window.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `window` is correct and stays alive for the entire
    /// lifetime of the surface. The `win` parameter can be used to ensure this.
    pub unsafe fn from_anativewindow<T>(
        instance: Arc<Instance>,
        window: *const T,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        let fns = instance.fns();

        if !instance.enabled_extensions().khr_android_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_android_surface",
            });
        }

        let surface = {
            let infos = ash::vk::AndroidSurfaceCreateInfoKHR {
                flags: ash::vk::AndroidSurfaceCreateFlagsKHR::empty(),
                window: window as *mut _,
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_android_surface.create_android_surface_khr(
                instance.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            window: win,
            instance: instance.clone(),
            surface,
            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from an iOS `UIView`.
    ///
    /// # Safety
    ///
    /// - The caller must ensure that the `view` is correct and stays alive for the entire
    ///   lifetime of the surface. The win parameter can be used to ensure this.
    /// - The `UIView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    pub unsafe fn from_ios_moltenvk<T>(
        instance: Arc<Instance>,
        view: *const T,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        let fns = instance.fns();

        if !instance.enabled_extensions().mvk_ios_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_MVK_ios_surface",
            });
        }

        let surface = {
            let infos = ash::vk::IOSSurfaceCreateInfoMVK {
                flags: ash::vk::IOSSurfaceCreateFlagsMVK::empty(),
                p_view: view as *const _,
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.mvk_ios_surface.create_ios_surface_mvk(
                instance.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            window: win,
            instance: instance.clone(),
            surface,
            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from a MacOS `NSView`.
    ///
    /// # Safety
    ///
    /// - The caller must ensure that the `view` is correct and stays alive for the entire
    ///   lifetime of the surface. The `win` parameter can be used to ensure this.
    /// - The `NSView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    pub unsafe fn from_macos_moltenvk<T>(
        instance: Arc<Instance>,
        view: *const T,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        let fns = instance.fns();

        if !instance.enabled_extensions().mvk_macos_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_MVK_macos_surface",
            });
        }

        let surface = {
            let infos = ash::vk::MacOSSurfaceCreateInfoMVK {
                flags: ash::vk::MacOSSurfaceCreateFlagsMVK::empty(),
                p_view: view as *const _,
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.mvk_macos_surface.create_mac_os_surface_mvk(
                instance.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            window: win,
            instance: instance.clone(),
            surface,
            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from a `code:nn::code:vi::code:Layer`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `window` is correct and stays alive for the entire
    /// lifetime of the surface. The `win` parameter can be used to ensure this.
    pub unsafe fn from_vi_surface<T>(
        instance: Arc<Instance>,
        window: *const T,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        let fns = instance.fns();

        if !instance.enabled_extensions().nn_vi_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_NN_vi_surface",
            });
        }

        let surface = {
            let infos = ash::vk::ViSurfaceCreateInfoNN {
                flags: ash::vk::ViSurfaceCreateFlagsNN::empty(),
                window: window as *mut _,
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.nn_vi_surface.create_vi_surface_nn(
                instance.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            window: win,
            instance: instance.clone(),
            surface,
            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Returns true if the given queue family can draw on this surface.
    // FIXME: vulkano doesn't check this for the moment!
    pub fn is_supported(&self, queue: QueueFamily) -> Result<bool, CapabilitiesError> {
        unsafe {
            let fns = self.instance.fns();

            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_surface.get_physical_device_surface_support_khr(
                queue.physical_device().internal_object(),
                queue.id(),
                self.surface,
                output.as_mut_ptr(),
            ))?;
            Ok(output.assume_init() != 0)
        }
    }

    /// Retrieves the capabilities of a surface when used by a certain device.
    ///
    /// # Notes
    ///
    /// - Capabilities that are not supported in `vk-sys` are silently dropped
    ///
    /// # Panic
    ///
    /// - Panics if the device and the surface don't belong to the same instance.
    ///
    pub fn capabilities(&self, device: PhysicalDevice) -> Result<Capabilities, CapabilitiesError> {
        unsafe {
            assert_eq!(
                &*self.instance as *const _,
                &**device.instance() as *const _,
                "Instance mismatch in Surface::capabilities"
            );

            let fns = self.instance.fns();

            let caps = {
                let mut out: MaybeUninit<ash::vk::SurfaceCapabilitiesKHR> = MaybeUninit::uninit();
                check_errors(
                    fns.khr_surface
                        .get_physical_device_surface_capabilities_khr(
                            device.internal_object(),
                            self.surface,
                            out.as_mut_ptr(),
                        ),
                )?;
                out.assume_init()
            };

            let formats = {
                let mut num = 0;
                check_errors(fns.khr_surface.get_physical_device_surface_formats_khr(
                    device.internal_object(),
                    self.surface,
                    &mut num,
                    ptr::null_mut(),
                ))?;

                let mut formats = Vec::with_capacity(num as usize);
                check_errors(fns.khr_surface.get_physical_device_surface_formats_khr(
                    device.internal_object(),
                    self.surface,
                    &mut num,
                    formats.as_mut_ptr(),
                ))?;
                formats.set_len(num as usize);
                formats
            };

            let modes = {
                let mut num = 0;
                check_errors(
                    fns.khr_surface
                        .get_physical_device_surface_present_modes_khr(
                            device.internal_object(),
                            self.surface,
                            &mut num,
                            ptr::null_mut(),
                        ),
                )?;

                let mut modes = Vec::with_capacity(num as usize);
                check_errors(
                    fns.khr_surface
                        .get_physical_device_surface_present_modes_khr(
                            device.internal_object(),
                            self.surface,
                            &mut num,
                            modes.as_mut_ptr(),
                        ),
                )?;
                modes.set_len(num as usize);
                debug_assert!(modes
                    .iter()
                    .find(|&&m| m == ash::vk::PresentModeKHR::FIFO)
                    .is_some());
                debug_assert!(modes.iter().count() > 0);
                modes.into_iter().collect()
            };

            Ok(Capabilities {
                min_image_count: caps.min_image_count,
                max_image_count: if caps.max_image_count == 0 {
                    None
                } else {
                    Some(caps.max_image_count)
                },
                current_extent: if caps.current_extent.width == 0xffffffff
                    && caps.current_extent.height == 0xffffffff
                {
                    None
                } else {
                    Some([caps.current_extent.width, caps.current_extent.height])
                },
                min_image_extent: [caps.min_image_extent.width, caps.min_image_extent.height],
                max_image_extent: [caps.max_image_extent.width, caps.max_image_extent.height],
                max_image_array_layers: caps.max_image_array_layers,
                supported_transforms: caps.supported_transforms.into(),

                current_transform: SupportedSurfaceTransforms::from(caps.current_transform)
                    .iter()
                    .next()
                    .unwrap(), // TODO:
                supported_composite_alpha: caps.supported_composite_alpha.into(),
                supported_usage_flags: {
                    let usage = ImageUsage::from(caps.supported_usage_flags);
                    debug_assert!(usage.color_attachment); // specs say that this must be true
                    usage
                },
                supported_formats: formats
                    .into_iter()
                    .filter_map(|f| {
                        // TODO: Change the way capabilities not supported in vk-sys are handled
                        Format::try_from(f.format)
                            .ok()
                            .map(|format| (format, f.color_space.into()))
                    })
                    .collect(),
                present_modes: modes,
            })
        }
    }

    #[inline]
    pub fn window(&self) -> &W {
        &self.window
    }

    /// Returns the instance this surface was created with.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
}

unsafe impl<W> SurfaceSwapchainLock for Surface<W> {
    #[inline]
    fn flag(&self) -> &AtomicBool {
        &self.has_swapchain
    }
}

unsafe impl<W> VulkanObject for Surface<W> {
    type Object = ash::vk::SurfaceKHR;

    #[inline]
    fn internal_object(&self) -> ash::vk::SurfaceKHR {
        self.surface
    }
}

impl<W> fmt::Debug for Surface<W> {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan surface {:?}>", self.surface)
    }
}

impl<W> Drop for Surface<W> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.instance.fns();
            fns.khr_surface.destroy_surface_khr(
                self.instance.internal_object(),
                self.surface,
                ptr::null(),
            );
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
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            SurfaceCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SurfaceCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                SurfaceCreationError::OomError(_) => "not enough memory available",
                SurfaceCreationError::MissingExtension { .. } => {
                    "the extension required for this function was not enabled"
                }
            }
        )
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

/// Error that can happen when retrieving a surface's capabilities.
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
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            CapabilitiesError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for CapabilitiesError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CapabilitiesError::OomError(_) => "not enough memory",
                CapabilitiesError::SurfaceLost => "the surface is no longer valid",
            }
        )
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
    use crate::swapchain::Surface;
    use crate::swapchain::SurfaceCreationError;
    use std::ptr;

    #[test]
    fn khr_win32_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_hwnd(instance, ptr::null::<u8>(), ptr::null::<u8>(), ()) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn khr_xcb_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_xcb(instance, ptr::null::<u8>(), 0, ()) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn khr_xlib_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_xlib(instance, ptr::null::<u8>(), 0, ()) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn khr_wayland_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_wayland(instance, ptr::null::<u8>(), ptr::null::<u8>(), ()) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn khr_android_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_anativewindow(instance, ptr::null::<u8>(), ()) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }
}
