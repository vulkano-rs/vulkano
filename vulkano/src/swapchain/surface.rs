// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::image::ImageUsage;
use crate::instance::Instance;
use crate::swapchain::display::DisplayMode;
use crate::swapchain::display::DisplayPlane;
use crate::swapchain::SurfaceSwapchainLock;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use std::error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Represents a surface on the screen.
///
/// Creating a `Surface` is platform-specific.
pub struct Surface<W> {
    handle: ash::vk::SurfaceKHR,
    instance: Arc<Instance>,
    api: SurfaceApi,
    window: W,

    // If true, a swapchain has been associated to this surface, and that any new swapchain
    // creation should be forbidden.
    has_swapchain: AtomicBool,
}

impl<W> Surface<W> {
    /// Creates a `Surface` given the raw handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan surface handle owned by `instance`.
    /// - `handle` must have been created from `api`.
    /// - The window object that `handle` was created from must outlive the created `Surface`.
    ///   The `win` parameter can be used to ensure this.
    pub unsafe fn from_raw_surface(
        instance: Arc<Instance>,
        handle: ash::vk::SurfaceKHR,
        api: SurfaceApi,
        win: W,
    ) -> Surface<W> {
        Surface {
            handle,
            instance,
            api,
            window: win,

            has_swapchain: AtomicBool::new(false),
        }
    }

    /// Creates a `Surface` from a `DisplayPlane`.
    ///
    /// # Panic
    ///
    /// - Panics if `display_mode` and `plane` don't belong to the same physical device.
    /// - Panics if `plane` doesn't support the display of `display_mode`.
    pub fn from_display_plane(
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

        let create_info = ash::vk::DisplaySurfaceCreateInfoKHR {
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

        let handle = unsafe {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_display.create_display_plane_surface_khr(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            handle,
            instance: instance.clone(),
            api: SurfaceApi::DisplayPlane,
            window: (),

            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from an Android window.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid handle.
    /// - The object referred to by `window` must outlive the created `Surface`.
    ///   The `win` parameter can be used to ensure this.
    pub unsafe fn from_android<T>(
        instance: Arc<Instance>,
        window: *const T,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        if !instance.enabled_extensions().khr_android_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_android_surface",
            });
        }

        let create_info = ash::vk::AndroidSurfaceCreateInfoKHR {
            flags: ash::vk::AndroidSurfaceCreateFlagsKHR::empty(),
            window: window as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_android_surface.create_android_surface_khr(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            handle,
            instance,
            api: SurfaceApi::Android,
            window: win,

            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from an iOS `UIView`.
    ///
    /// # Safety
    ///
    /// - `view` must be a valid handle.
    /// - The object referred to by `view` must outlive the created `Surface`.
    ///   The `win` parameter can be used to ensure this.
    /// - The `UIView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    pub unsafe fn from_ios<T>(
        instance: Arc<Instance>,
        view: *const T,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        if !instance.enabled_extensions().mvk_ios_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_MVK_ios_surface",
            });
        }

        let create_info = ash::vk::IOSSurfaceCreateInfoMVK {
            flags: ash::vk::IOSSurfaceCreateFlagsMVK::empty(),
            p_view: view as *const _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.mvk_ios_surface.create_ios_surface_mvk(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            handle,
            instance,
            api: SurfaceApi::Ios,
            window: win,

            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from a MacOS `NSView`.
    ///
    /// # Safety
    ///
    /// - `view` must be a valid handle.
    /// - The object referred to by `view` must outlive the created `Surface`.
    ///   The `win` parameter can be used to ensure this.
    /// - The `NSView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    pub unsafe fn from_mac_os<T>(
        instance: Arc<Instance>,
        view: *const T,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        if !instance.enabled_extensions().mvk_macos_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_MVK_macos_surface",
            });
        }

        let create_info = ash::vk::MacOSSurfaceCreateInfoMVK {
            flags: ash::vk::MacOSSurfaceCreateFlagsMVK::empty(),
            p_view: view as *const _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.mvk_macos_surface.create_mac_os_surface_mvk(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            handle,
            instance,
            api: SurfaceApi::MacOs,
            window: win,

            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from a Metal `CAMetalLayer`.
    ///
    /// # Safety
    ///
    /// - `layer` must be a valid handle.
    /// - The object referred to by `layer` must outlive the created `Surface`.
    ///   The `win` parameter can be used to ensure this.
    pub unsafe fn from_metal<T>(
        instance: Arc<Instance>,
        layer: *const T,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        if !instance.enabled_extensions().ext_metal_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_EXT_metal_surface",
            });
        }

        let create_info = ash::vk::MetalSurfaceCreateInfoEXT {
            flags: ash::vk::MetalSurfaceCreateFlagsEXT::empty(),
            p_layer: layer as *const _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.ext_metal_surface.create_metal_surface_ext(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            handle,
            instance,
            api: SurfaceApi::Metal,
            window: win,

            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from a `code:nn::code:vi::code:Layer`.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid handle.
    /// - The object referred to by `window` must outlive the created `Surface`.
    ///   The `win` parameter can be used to ensure this.
    pub unsafe fn from_vi<T>(
        instance: Arc<Instance>,
        window: *const T,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        if !instance.enabled_extensions().nn_vi_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_NN_vi_surface",
            });
        }

        let create_info = ash::vk::ViSurfaceCreateInfoNN {
            flags: ash::vk::ViSurfaceCreateFlagsNN::empty(),
            window: window as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.nn_vi_surface.create_vi_surface_nn(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            handle,
            instance,
            api: SurfaceApi::Vi,
            window: win,

            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from a Wayland window.
    ///
    /// The window's dimensions will be set to the size of the swapchain.
    ///
    /// # Safety
    ///
    /// - `display` and `surface` must be valid handles.
    /// - The objects referred to by `display` and `surface` must outlive the created `Surface`.
    ///   The `win` parameter can be used to ensure this.
    pub unsafe fn from_wayland<D, S>(
        instance: Arc<Instance>,
        display: *const D,
        surface: *const S,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        if !instance.enabled_extensions().khr_wayland_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_wayland_surface",
            });
        }

        let create_info = ash::vk::WaylandSurfaceCreateInfoKHR {
            flags: ash::vk::WaylandSurfaceCreateFlagsKHR::empty(),
            display: display as *mut _,
            surface: surface as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_wayland_surface.create_wayland_surface_khr(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            handle,
            instance,
            api: SurfaceApi::Wayland,
            window: win,

            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from a Win32 window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// - `hinstance` and `hwnd` must be valid handles.
    /// - The objects referred to by `hwnd` and `hinstance` must outlive the created `Surface`.
    ///   The `win` parameter can be used to ensure this.
    pub unsafe fn from_win32<T, U>(
        instance: Arc<Instance>,
        hinstance: *const T,
        hwnd: *const U,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        if !instance.enabled_extensions().khr_win32_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_win32_surface",
            });
        }

        let create_info = ash::vk::Win32SurfaceCreateInfoKHR {
            flags: ash::vk::Win32SurfaceCreateFlagsKHR::empty(),
            hinstance: hinstance as *mut _,
            hwnd: hwnd as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_win32_surface.create_win32_surface_khr(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            handle,
            instance,
            api: SurfaceApi::Win32,
            window: win,

            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from an XCB window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// - `connection` and `window` must be valid handles.
    /// - The objects referred to by `connection` and `window` must outlive the created `Surface`.
    ///   The `win` parameter can be used to ensure this.
    pub unsafe fn from_xcb<C>(
        instance: Arc<Instance>,
        connection: *const C,
        window: u32,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        if !instance.enabled_extensions().khr_xcb_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_xcb_surface",
            });
        }

        let create_info = ash::vk::XcbSurfaceCreateInfoKHR {
            flags: ash::vk::XcbSurfaceCreateFlagsKHR::empty(),
            connection: connection as *mut _,
            window,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_xcb_surface.create_xcb_surface_khr(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            handle,
            instance,
            api: SurfaceApi::Xcb,
            window: win,

            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Creates a `Surface` from an Xlib window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// - `display` and `window` must be valid handles.
    /// - The objects referred to by `display` and `window` must outlive the created `Surface`.
    ///   The `win` parameter can be used to ensure this.
    pub unsafe fn from_xlib<D>(
        instance: Arc<Instance>,
        display: *const D,
        window: c_ulong,
        win: W,
    ) -> Result<Arc<Surface<W>>, SurfaceCreationError> {
        if !instance.enabled_extensions().khr_xlib_surface {
            return Err(SurfaceCreationError::MissingExtension {
                name: "VK_KHR_xlib_surface",
            });
        }

        let create_info = ash::vk::XlibSurfaceCreateInfoKHR {
            flags: ash::vk::XlibSurfaceCreateFlagsKHR::empty(),
            dpy: display as *mut _,
            window,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_xlib_surface.create_xlib_surface_khr(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(Surface {
            handle,
            instance,
            api: SurfaceApi::Xlib,
            window: win,

            has_swapchain: AtomicBool::new(false),
        }))
    }

    /// Returns the instance this surface was created with.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Returns the windowing API that was used to construct the surface.
    #[inline]
    pub fn api(&self) -> SurfaceApi {
        self.api
    }

    /// Returns a reference to the `W` type parameter that was passed when creating the surface.
    #[inline]
    pub fn window(&self) -> &W {
        &self.window
    }
}

impl<W> Drop for Surface<W> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.instance.fns();
            fns.khr_surface.destroy_surface_khr(
                self.instance.internal_object(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

unsafe impl<W> VulkanObject for Surface<W> {
    type Object = ash::vk::SurfaceKHR;

    #[inline]
    fn internal_object(&self) -> ash::vk::SurfaceKHR {
        self.handle
    }
}

impl<W> fmt::Debug for Surface<W> {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let Self {
            handle,
            instance,
            api,
            window: _,

            has_swapchain,
        } = self;

        fmt.debug_struct("Surface")
            .field("handle", handle)
            .field("instance", instance)
            .field("api", api)
            .field("window", &())
            .field("has_swapchain", &has_swapchain)
            .finish()
    }
}

impl<W> PartialEq for Surface<W> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.instance() == other.instance()
    }
}

impl<W> Eq for Surface<W> {}

impl<W> Hash for Surface<W> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.instance().hash(state);
    }
}

unsafe impl<W> SurfaceSwapchainLock for Surface<W> {
    #[inline]
    fn flag(&self) -> &AtomicBool {
        &self.has_swapchain
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

/// The windowing API that was used to construct a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SurfaceApi {
    DisplayPlane,

    // Alphabetical order
    Android,
    Ios,
    MacOs,
    Metal,
    Vi,
    Wayland,
    Win32,
    Xcb,
    Xlib,
}

/// The way presenting a swapchain is accomplished.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
#[non_exhaustive]
pub enum PresentMode {
    /// Immediately shows the image to the user. May result in visible tearing.
    Immediate = ash::vk::PresentModeKHR::IMMEDIATE.as_raw(),

    /// The action of presenting an image puts it in wait. When the next vertical blanking period
    /// happens, the waiting image is effectively shown to the user. If an image is presented while
    /// another one is waiting, it is replaced.
    Mailbox = ash::vk::PresentModeKHR::MAILBOX.as_raw(),

    /// The action of presenting an image adds it to a queue of images. At each vertical blanking
    /// period, the queue is popped and an image is presented.
    ///
    /// Guaranteed to be always supported.
    ///
    /// This is the equivalent of OpenGL's `SwapInterval` with a value of 1.
    Fifo = ash::vk::PresentModeKHR::FIFO.as_raw(),

    /// Same as `Fifo`, except that if the queue was empty during the previous vertical blanking
    /// period then it is equivalent to `Immediate`.
    ///
    /// This is the equivalent of OpenGL's `SwapInterval` with a value of -1.
    FifoRelaxed = ash::vk::PresentModeKHR::FIFO_RELAXED.as_raw(),
    // TODO: These can't be enabled yet because they have to be used with shared present surfaces
    // which vulkano doesnt support yet.
    //SharedDemand = ash::vk::PresentModeKHR::SHARED_DEMAND_REFRESH,
    //SharedContinuous = ash::vk::PresentModeKHR::SHARED_CONTINUOUS_REFRESH,
}

impl From<PresentMode> for ash::vk::PresentModeKHR {
    #[inline]
    fn from(val: PresentMode) -> Self {
        Self::from_raw(val as i32)
    }
}

impl TryFrom<ash::vk::PresentModeKHR> for PresentMode {
    type Error = ();

    fn try_from(value: ash::vk::PresentModeKHR) -> Result<Self, Self::Error> {
        Ok(match value {
            ash::vk::PresentModeKHR::IMMEDIATE => Self::Immediate,
            ash::vk::PresentModeKHR::MAILBOX => Self::Mailbox,
            ash::vk::PresentModeKHR::FIFO => Self::Fifo,
            ash::vk::PresentModeKHR::FIFO_RELAXED => Self::FifoRelaxed,
            //ash::vk::PresentModeKHR::SHARED_DEMAND_REFRESH => Self::SharedDemandRefresh,
            //ash::vk::PresentModeKHR::SHARED_CONTINUOUS_REFRESH => Self::SharedContinuousRefresh,
            _ => return Err(()),
        })
    }
}

/// A transformation to apply to the image before showing it on the screen.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
pub enum SurfaceTransform {
    /// Don't transform the image.
    Identity = ash::vk::SurfaceTransformFlagsKHR::IDENTITY.as_raw(),
    /// Rotate 90 degrees.
    Rotate90 = ash::vk::SurfaceTransformFlagsKHR::ROTATE_90.as_raw(),
    /// Rotate 180 degrees.
    Rotate180 = ash::vk::SurfaceTransformFlagsKHR::ROTATE_180.as_raw(),
    /// Rotate 270 degrees.
    Rotate270 = ash::vk::SurfaceTransformFlagsKHR::ROTATE_270.as_raw(),
    /// Mirror the image horizontally.
    HorizontalMirror = ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR.as_raw(),
    /// Mirror the image horizontally and rotate 90 degrees.
    HorizontalMirrorRotate90 =
        ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_90.as_raw(),
    /// Mirror the image horizontally and rotate 180 degrees.
    HorizontalMirrorRotate180 =
        ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_180.as_raw(),
    /// Mirror the image horizontally and rotate 270 degrees.
    HorizontalMirrorRotate270 =
        ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_270.as_raw(),
    /// Let the operating system or driver implementation choose.
    Inherit = ash::vk::SurfaceTransformFlagsKHR::INHERIT.as_raw(),
}

impl From<SurfaceTransform> for ash::vk::SurfaceTransformFlagsKHR {
    #[inline]
    fn from(val: SurfaceTransform) -> Self {
        Self::from_raw(val as u32)
    }
}

/// List of supported composite alpha modes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SupportedSurfaceTransforms {
    pub identity: bool,
    pub rotate90: bool,
    pub rotate180: bool,
    pub rotate270: bool,
    pub horizontal_mirror: bool,
    pub horizontal_mirror_rotate90: bool,
    pub horizontal_mirror_rotate180: bool,
    pub horizontal_mirror_rotate270: bool,
    pub inherit: bool,
}

impl From<ash::vk::SurfaceTransformFlagsKHR> for SupportedSurfaceTransforms {
    fn from(val: ash::vk::SurfaceTransformFlagsKHR) -> Self {
        macro_rules! v {
            ($val:expr, $out:ident, $e:expr, $f:ident) => {
                if !($val & $e).is_empty() {
                    $out.$f = true;
                }
            };
        }

        let mut result = SupportedSurfaceTransforms::none();
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::IDENTITY,
            identity
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::ROTATE_90,
            rotate90
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::ROTATE_180,
            rotate180
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::ROTATE_270,
            rotate270
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR,
            horizontal_mirror
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_90,
            horizontal_mirror_rotate90
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_180,
            horizontal_mirror_rotate180
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_270,
            horizontal_mirror_rotate270
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::INHERIT,
            inherit
        );
        result
    }
}

impl SupportedSurfaceTransforms {
    /// Builds a `SupportedSurfaceTransforms` with all fields set to false.
    #[inline]
    pub fn none() -> SupportedSurfaceTransforms {
        SupportedSurfaceTransforms {
            identity: false,
            rotate90: false,
            rotate180: false,
            rotate270: false,
            horizontal_mirror: false,
            horizontal_mirror_rotate90: false,
            horizontal_mirror_rotate180: false,
            horizontal_mirror_rotate270: false,
            inherit: false,
        }
    }

    /// Returns true if the given `SurfaceTransform` is in this list.
    #[inline]
    pub fn supports(&self, value: SurfaceTransform) -> bool {
        match value {
            SurfaceTransform::Identity => self.identity,
            SurfaceTransform::Rotate90 => self.rotate90,
            SurfaceTransform::Rotate180 => self.rotate180,
            SurfaceTransform::Rotate270 => self.rotate270,
            SurfaceTransform::HorizontalMirror => self.horizontal_mirror,
            SurfaceTransform::HorizontalMirrorRotate90 => self.horizontal_mirror_rotate90,
            SurfaceTransform::HorizontalMirrorRotate180 => self.horizontal_mirror_rotate180,
            SurfaceTransform::HorizontalMirrorRotate270 => self.horizontal_mirror_rotate270,
            SurfaceTransform::Inherit => self.inherit,
        }
    }

    /// Returns an iterator to the list of supported composite alpha.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = SurfaceTransform> {
        let moved = *self;
        [
            SurfaceTransform::Identity,
            SurfaceTransform::Rotate90,
            SurfaceTransform::Rotate180,
            SurfaceTransform::Rotate270,
            SurfaceTransform::HorizontalMirror,
            SurfaceTransform::HorizontalMirrorRotate90,
            SurfaceTransform::HorizontalMirrorRotate180,
            SurfaceTransform::HorizontalMirrorRotate270,
            SurfaceTransform::Inherit,
        ]
        .into_iter()
        .filter(move |&mode| moved.supports(mode))
    }
}

impl Default for SurfaceTransform {
    #[inline]
    fn default() -> SurfaceTransform {
        SurfaceTransform::Identity
    }
}

/// How the alpha values of the pixels of the window are treated.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
pub enum CompositeAlpha {
    /// The alpha channel of the image is ignored. All the pixels are considered as if they have a
    /// value of 1.0.
    Opaque = ash::vk::CompositeAlphaFlagsKHR::OPAQUE.as_raw(),

    /// The alpha channel of the image is respected. The color channels are expected to have
    /// already been multiplied by the alpha value.
    PreMultiplied = ash::vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED.as_raw(),

    /// The alpha channel of the image is respected. The color channels will be multiplied by the
    /// alpha value by the compositor before being added to what is behind.
    PostMultiplied = ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED.as_raw(),

    /// Let the operating system or driver implementation choose.
    Inherit = ash::vk::CompositeAlphaFlagsKHR::INHERIT.as_raw(),
}

impl From<CompositeAlpha> for ash::vk::CompositeAlphaFlagsKHR {
    #[inline]
    fn from(val: CompositeAlpha) -> Self {
        Self::from_raw(val as u32)
    }
}

/// List of supported composite alpha modes.
///
/// See the docs of `CompositeAlpha`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SupportedCompositeAlpha {
    pub opaque: bool,
    pub pre_multiplied: bool,
    pub post_multiplied: bool,
    pub inherit: bool,
}

impl From<ash::vk::CompositeAlphaFlagsKHR> for SupportedCompositeAlpha {
    #[inline]
    fn from(val: ash::vk::CompositeAlphaFlagsKHR) -> SupportedCompositeAlpha {
        let mut result = SupportedCompositeAlpha::none();
        if !(val & ash::vk::CompositeAlphaFlagsKHR::OPAQUE).is_empty() {
            result.opaque = true;
        }
        if !(val & ash::vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED).is_empty() {
            result.pre_multiplied = true;
        }
        if !(val & ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED).is_empty() {
            result.post_multiplied = true;
        }
        if !(val & ash::vk::CompositeAlphaFlagsKHR::INHERIT).is_empty() {
            result.inherit = true;
        }
        result
    }
}

impl SupportedCompositeAlpha {
    /// Builds a `SupportedCompositeAlpha` with all fields set to false.
    #[inline]
    pub fn none() -> SupportedCompositeAlpha {
        SupportedCompositeAlpha {
            opaque: false,
            pre_multiplied: false,
            post_multiplied: false,
            inherit: false,
        }
    }

    /// Returns true if the given `CompositeAlpha` is in this list.
    #[inline]
    pub fn supports(&self, value: CompositeAlpha) -> bool {
        match value {
            CompositeAlpha::Opaque => self.opaque,
            CompositeAlpha::PreMultiplied => self.pre_multiplied,
            CompositeAlpha::PostMultiplied => self.post_multiplied,
            CompositeAlpha::Inherit => self.inherit,
        }
    }

    /// Returns an iterator to the list of supported composite alpha.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = CompositeAlpha> {
        let moved = *self;
        [
            CompositeAlpha::Opaque,
            CompositeAlpha::PreMultiplied,
            CompositeAlpha::PostMultiplied,
            CompositeAlpha::Inherit,
        ]
        .into_iter()
        .filter(move |&mode| moved.supports(mode))
    }
}

/// How the presentation engine should interpret the data.
///
/// # A quick lesson about color spaces
///
/// ## What is a color space?
///
/// Each pixel of a monitor is made of three components: one red, one green, and one blue. In the
/// past, computers would simply send to the monitor the intensity of each of the three components.
///
/// This proved to be problematic, because depending on the brand of the monitor the colors would
/// not exactly be the same. For example on some monitors, a value of `[1.0, 0.0, 0.0]` would be a
/// bit more orange than on others.
///
/// In order to standardize this, there exist what are called *color spaces*: sRGB, AdobeRGB,
/// DCI-P3, scRGB, etc. When you manipulate RGB values in a specific color space, these values have
/// a precise absolute meaning in terms of color, that is the same across all systems and monitors.
///
/// > **Note**: Color spaces are orthogonal to concept of RGB. *RGB* only indicates what is the
/// > representation of the data, but not how it is interpreted. You can think of this a bit like
/// > text encoding. An *RGB* value is a like a byte, in other words it is the medium by which
/// > values are communicated, and a *color space* is like a text encoding (eg. UTF-8), in other
/// > words it is the way the value should be interpreted.
///
/// The most commonly used color space today is sRGB. Most monitors today use this color space,
/// and most images files are encoded in this color space.
///
/// ## Pixel formats and linear vs non-linear
///
/// In Vulkan all images have a specific format in which the data is stored. The data of an image
/// consists of pixels in RGB but contains no information about the color space (or lack thereof)
/// of these pixels. You are free to store them in whatever color space you want.
///
/// But one big practical problem with color spaces is that they are sometimes not linear, and in
/// particular the popular sRGB color space is not linear. In a non-linear color space, a value of
/// `[0.6, 0.6, 0.6]` for example is **not** twice as bright as a value of `[0.3, 0.3, 0.3]`. This
/// is problematic, because operations such as taking the average of two colors or calculating the
/// lighting of a texture with a dot product are mathematically incorrect and will produce
/// incorrect colors.
///
/// > **Note**: If the texture format has an alpha component, it is not affected by the color space
/// > and always behaves linearly.
///
/// In order to solve this Vulkan also provides image formats with the `Srgb` suffix, which are
/// expected to contain RGB data in the sRGB color space. When you sample an image with such a
/// format from a shader, the implementation will automatically turn the pixel values into a linear
/// color space that is suitable for linear operations (such as additions or multiplications).
/// When you write to a framebuffer attachment with such a format, the implementation will
/// automatically perform the opposite conversion. These conversions are most of the time performed
/// by the hardware and incur no additional cost.
///
/// ## Color space of the swapchain
///
/// The color space that you specify when you create a swapchain is how the implementation will
/// interpret the raw data inside of the image.
///
/// > **Note**: The implementation can choose to send the data in the swapchain image directly to
/// > the monitor, but it can also choose to write it in an intermediary buffer that is then read
/// > by the operating system or windowing system. Therefore the color space that the
/// > implementation supports is not necessarily the same as the one supported by the monitor.
///
/// It is *your* job to ensure that the data in the swapchain image is in the color space
/// that is specified here, otherwise colors will be incorrect.
/// The implementation will never perform any additional automatic conversion after the colors have
/// been written to the swapchain image.
///
/// # How do I handle this correctly?
///
/// The easiest way to handle color spaces in a cross-platform program is:
///
/// - Always request the `SrgbNonLinear` color space when creating the swapchain.
/// - Make sure that all your image files use the sRGB color space, and load them in images whose
///   format has the `Srgb` suffix. Only use non-sRGB image formats for intermediary computations
///   or to store non-color data.
/// - Swapchain images should have a format with the `Srgb` suffix.
///
/// > **Note**: It is unclear whether the `SrgbNonLinear` color space is always supported by the
/// > the implementation or not. See <https://github.com/KhronosGroup/Vulkan-Docs/issues/442>.
///
/// > **Note**: Lots of developers are confused by color spaces. You can sometimes find articles
/// > talking about gamma correction and suggestion to put your colors to the power 2.2 for
/// > example. These are all hacks and you should use the sRGB pixel formats instead.
///
/// If you follow these three rules, then everything should render the same way on all platforms.
///
/// Additionally you can try detect whether the implementation supports any additional color space
/// and perform a manual conversion to that color space from inside your shader.
///
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
#[non_exhaustive]
pub enum ColorSpace {
    SrgbNonLinear = ash::vk::ColorSpaceKHR::SRGB_NONLINEAR.as_raw(),
    DisplayP3NonLinear = ash::vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT.as_raw(),
    ExtendedSrgbLinear = ash::vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT.as_raw(),
    ExtendedSrgbNonLinear = ash::vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT.as_raw(),
    DisplayP3Linear = ash::vk::ColorSpaceKHR::DISPLAY_P3_LINEAR_EXT.as_raw(),
    DciP3NonLinear = ash::vk::ColorSpaceKHR::DCI_P3_NONLINEAR_EXT.as_raw(),
    Bt709Linear = ash::vk::ColorSpaceKHR::BT709_LINEAR_EXT.as_raw(),
    Bt709NonLinear = ash::vk::ColorSpaceKHR::BT709_NONLINEAR_EXT.as_raw(),
    Bt2020Linear = ash::vk::ColorSpaceKHR::BT2020_LINEAR_EXT.as_raw(),
    Hdr10St2084 = ash::vk::ColorSpaceKHR::HDR10_ST2084_EXT.as_raw(),
    DolbyVision = ash::vk::ColorSpaceKHR::DOLBYVISION_EXT.as_raw(),
    Hdr10Hlg = ash::vk::ColorSpaceKHR::HDR10_HLG_EXT.as_raw(),
    AdobeRgbLinear = ash::vk::ColorSpaceKHR::ADOBERGB_LINEAR_EXT.as_raw(),
    AdobeRgbNonLinear = ash::vk::ColorSpaceKHR::ADOBERGB_NONLINEAR_EXT.as_raw(),
    PassThrough = ash::vk::ColorSpaceKHR::PASS_THROUGH_EXT.as_raw(),
    DisplayNative = ash::vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD.as_raw(),
}

impl From<ColorSpace> for ash::vk::ColorSpaceKHR {
    #[inline]
    fn from(val: ColorSpace) -> Self {
        Self::from_raw(val as i32)
    }
}

impl From<ash::vk::ColorSpaceKHR> for ColorSpace {
    #[inline]
    fn from(val: ash::vk::ColorSpaceKHR) -> Self {
        match val {
            ash::vk::ColorSpaceKHR::SRGB_NONLINEAR => ColorSpace::SrgbNonLinear,
            ash::vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT => ColorSpace::DisplayP3NonLinear,
            ash::vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT => ColorSpace::ExtendedSrgbLinear,
            ash::vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT => {
                ColorSpace::ExtendedSrgbNonLinear
            }
            ash::vk::ColorSpaceKHR::DISPLAY_P3_LINEAR_EXT => ColorSpace::DisplayP3Linear,
            ash::vk::ColorSpaceKHR::DCI_P3_NONLINEAR_EXT => ColorSpace::DciP3NonLinear,
            ash::vk::ColorSpaceKHR::BT709_LINEAR_EXT => ColorSpace::Bt709Linear,
            ash::vk::ColorSpaceKHR::BT709_NONLINEAR_EXT => ColorSpace::Bt709NonLinear,
            ash::vk::ColorSpaceKHR::BT2020_LINEAR_EXT => ColorSpace::Bt2020Linear,
            ash::vk::ColorSpaceKHR::HDR10_ST2084_EXT => ColorSpace::Hdr10St2084,
            ash::vk::ColorSpaceKHR::DOLBYVISION_EXT => ColorSpace::DolbyVision,
            ash::vk::ColorSpaceKHR::HDR10_HLG_EXT => ColorSpace::Hdr10Hlg,
            ash::vk::ColorSpaceKHR::ADOBERGB_LINEAR_EXT => ColorSpace::AdobeRgbLinear,
            ash::vk::ColorSpaceKHR::ADOBERGB_NONLINEAR_EXT => ColorSpace::AdobeRgbNonLinear,
            ash::vk::ColorSpaceKHR::PASS_THROUGH_EXT => ColorSpace::PassThrough,
            ash::vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD => ColorSpace::DisplayNative,
            _ => panic!("Wrong value for color space enum {:?}", val),
        }
    }
}

/// The capabilities of a surface when used by a physical device.
///
/// You have to match these capabilities when you create a swapchain.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SurfaceCapabilities {
    /// Minimum number of images that must be present in the swapchain.
    pub min_image_count: u32,

    /// Maximum number of images that must be present in the swapchain, or `None` if there is no
    /// maximum value. Note that "no maximum" doesn't mean that you can set a very high value, as
    /// you may still get out of memory errors.
    pub max_image_count: Option<u32>,

    /// The current dimensions of the surface. `None` means that the surface's dimensions will
    /// depend on the dimensions of the swapchain that you are going to create.
    pub current_extent: Option<[u32; 2]>,

    /// Minimum width and height of a swapchain that uses this surface.
    pub min_image_extent: [u32; 2],

    /// Maximum width and height of a swapchain that uses this surface.
    pub max_image_extent: [u32; 2],

    /// Maximum number of image layers if you create an image array. The minimum is 1.
    pub max_image_array_layers: u32,

    /// List of transforms supported for the swapchain.
    pub supported_transforms: SupportedSurfaceTransforms,

    /// Current transform used by the surface.
    pub current_transform: SurfaceTransform,

    /// List of composite alpha modes supports for the swapchain.
    pub supported_composite_alpha: SupportedCompositeAlpha,

    /// List of image usages that are supported for images of the swapchain. Only
    /// the `color_attachment` usage is guaranteed to be supported.
    pub supported_usage_flags: ImageUsage,

    /// Whether full-screen exclusivity is supported.
    pub full_screen_exclusive_supported: bool,
}

#[cfg(test)]
mod tests {
    use crate::swapchain::Surface;
    use crate::swapchain::SurfaceCreationError;
    use std::ptr;

    #[test]
    fn khr_win32_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_win32(instance, ptr::null::<u8>(), ptr::null::<u8>(), ()) } {
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
        match unsafe { Surface::from_android(instance, ptr::null::<u8>(), ()) } {
            Err(SurfaceCreationError::MissingExtension { .. }) => (),
            _ => panic!(),
        }
    }
}
