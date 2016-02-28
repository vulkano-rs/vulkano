//! Link between Vulkan and a window and/or the screen.
//! 
//! In order to draw on the screen or a window, you have to use two steps:
//! 
//! - Create a `Surface` object that represents the location where the image will show up.
//! - Create a `Swapchain` using that `Surface`.
//! 
//! Creating a surface can be done with only an `Instance` object. However creating a swapchain
//! requires a `Device` object.
//!
//! Once you have a swapchain, you can retreive `Image` objects from it and draw to them. However
//! due to double-buffering or other caching mechanism, the rendering will not automatically be
//! shown on screen. In order to show the output on screen, you have to *present* the swapchain
//! by using the method with the same name.
//!
//! # Extensions
//! 
//! Theses capabilities depend on some extensions:
//! 
//! - `VK_KHR_surface`
//! - `VK_KHR_swapchain`
//! - `VK_KHR_display`
//! - `VK_KHR_display_swapchain`
//! - `VK_KHR_xlib_surface`
//! - `VK_KHR_xcb_surface`
//! - `VK_KHR_wayland_surface`
//! - `VK_KHR_mir_surface`
//! - `VK_KHR_android_surface`
//! - `VK_KHR_win32_surface`
//!
use std::ffi::CStr;
use std::ptr;
use std::sync::Arc;
use std::vec::IntoIter;

use instance::PhysicalDevice;
use memory::MemorySourceChunk;

use check_errors;
use OomError;
use VulkanObject;
use VulkanPointers;
use vk;

pub use self::surface::Capabilities;
pub use self::surface::Surface;
pub use self::surface::PresentMode;
pub use self::surface::SurfaceTransform;
pub use self::surface::CompositeAlpha;
pub use self::surface::ColorSpace;
pub use self::swapchain::Swapchain;
pub use self::swapchain::SwapchainAllocatedChunk;
pub use self::swapchain::AcquireError;

mod surface;
mod swapchain;

// TODO: extract this to a `display` module and solve the visibility problems

/// ?
// TODO: plane capabilities
pub struct DisplayPlane {
    device: PhysicalDevice,
    index: u32,
    properties: vk::DisplayPlanePropertiesKHR,
    supported_displays: Vec<vk::DisplayKHR>,
}

impl DisplayPlane {
    /// Enumerates all the display planes that are available on a given physical device.
    pub fn enumerate(device: &PhysicalDevice) -> Result<IntoIter<DisplayPlane>, OomError> {
        let vk = device.instance().pointers();

        let num = unsafe {
            let mut num: u32 = 0;
            try!(check_errors(vk.GetPhysicalDeviceDisplayPlanePropertiesKHR(device.internal_object(),
                                                                            &mut num, ptr::null_mut())));
            num
        };

        let planes: Vec<vk::DisplayPlanePropertiesKHR> = unsafe {
            let mut planes = Vec::with_capacity(num as usize);
            let mut num = num;
            try!(check_errors(vk.GetPhysicalDeviceDisplayPlanePropertiesKHR(device.internal_object(),
                                                                            &mut num,
                                                                            planes.as_mut_ptr())));
            planes.set_len(num as usize);
            planes
        };

        Ok(planes.into_iter().enumerate().map(|(index, prop)| {
            let num = unsafe {
                let mut num: u32 = 0;
                check_errors(vk.GetDisplayPlaneSupportedDisplaysKHR(device.internal_object(), index as u32,
                                                                    &mut num, ptr::null_mut())).unwrap();       // TODO: shouldn't unwrap
                num
            };

            let supported_displays: Vec<vk::DisplayKHR> = unsafe {
                let mut displays = Vec::with_capacity(num as usize);
                let mut num = num;
                check_errors(vk.GetDisplayPlaneSupportedDisplaysKHR(device.internal_object(),
                                                                    index as u32, &mut num,
                                                                    displays.as_mut_ptr())).unwrap();       // TODO: shouldn't unwrap
                displays.set_len(num as usize);
                displays
            };

            DisplayPlane {
                device: device.clone(),
                index: index as u32,
                properties: prop,
                supported_displays: supported_displays,
            }
        }).collect::<Vec<_>>().into_iter())
    }

    /// Returns true if this plane supports the given display.
    #[inline]
    pub fn supports(&self, display: &Display) -> bool {
        // making sure that the physical device is the same
        if self.device.internal_object() != display.device.internal_object() {
            return false;
        }

        self.supported_displays.iter().find(|&&d| d == display.internal_object()).is_some()
    }
}

/// Represents a monitor connected to a physical device.
#[derive(Clone)]
pub struct Display {
    device: PhysicalDevice,
    properties: Arc<vk::DisplayPropertiesKHR>,      // TODO: Arc because struct isn't clone
}

impl Display {
    /// Enumerates all the displays that are available on a given physical device.
    pub fn enumerate(device: &PhysicalDevice) -> Result<IntoIter<Display>, OomError> {
        let vk = device.instance().pointers();

        let num = unsafe {
            let mut num = 0;
            try!(check_errors(vk.GetPhysicalDeviceDisplayPropertiesKHR(device.internal_object(),
                                                                       &mut num, ptr::null_mut())));
            num
        };

        let displays: Vec<vk::DisplayPropertiesKHR> = unsafe {
            let mut displays = Vec::with_capacity(num as usize);
            let mut num = num;
            try!(check_errors(vk.GetPhysicalDeviceDisplayPropertiesKHR(device.internal_object(),
                                                                       &mut num,
                                                                       displays.as_mut_ptr())));
            displays.set_len(num as usize);
            displays
        };

        Ok(displays.into_iter().map(|prop| {
            Display {
                device: device.clone(),
                properties: Arc::new(prop),
            }
        }).collect::<Vec<_>>().into_iter())
    }

    /// Returns the name of the display.
    #[inline]
    pub fn name(&self) -> &str {
        unsafe {
            CStr::from_ptr(self.properties.displayName).to_str()
                                                    .expect("non UTF-8 characters in display name")
        }
    }

    /// Returns the physical resolution of the display.
    #[inline]
    pub fn physical_resolution(&self) -> [u32; 2] {
        let ref r = self.properties.physicalResolution;
        [r.width, r.height]
    }

    /// Returns a list of all modes available on this display.
    pub fn display_modes(&self) -> Result<IntoIter<DisplayMode>, OomError> {
        let vk = self.device.instance().pointers();

        let num = unsafe {
            let mut num = 0;
            try!(check_errors(vk.GetDisplayModePropertiesKHR(self.device.internal_object(),
                                                             self.properties.display, 
                                                             &mut num, ptr::null_mut())));
            num
        };

        let modes: Vec<vk::DisplayModePropertiesKHR> = unsafe {
            let mut modes = Vec::with_capacity(num as usize);
            let mut num = num;
            try!(check_errors(vk.GetDisplayModePropertiesKHR(self.device.internal_object(),
                                                             self.properties.display, &mut num,
                                                             modes.as_mut_ptr())));
            modes.set_len(num as usize);
            modes
        };

        Ok(modes.into_iter().map(|mode| {
            DisplayMode {
                display: self.clone(),
                display_mode: mode.displayMode,
                parameters: mode.parameters,
            }
        }).collect::<Vec<_>>().into_iter())
    }
}

unsafe impl VulkanObject for Display {
    type Object = vk::DisplayKHR;

    #[inline]
    fn internal_object(&self) -> vk::DisplayKHR {
        self.properties.display
    }
}

/// Represents a mode on a specific display.
pub struct DisplayMode {
    display: Display,
    display_mode: vk::DisplayModeKHR,
    parameters: vk::DisplayModeParametersKHR,
}

impl DisplayMode {
    /*pub fn new(display: &Display) -> Result<Arc<DisplayMode>, OomError> {
        let vk = instance.pointers();

        let parameters = vk::DisplayModeParametersKHR {
            visibleRegion: vk::Extent2D { width: , height:  },
            refreshRate: ,
        };

        let display_mode = {
            let infos = vk::DisplayModeCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_DISPLAY_MODE_CREATE_INFO_KHR,
                pNext: ptr::null(),
                flags: 0,   // reserved
                parameters: parameters,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateDisplayModeKHR(display.device.internal_object(),
                                                      display.display, &infos, ptr::null(),
                                                      &mut output)));
            output
        };

        Ok(Arc::new(DisplayMode {
            instance: display.device.instance().clone(),
            display_mode: display_mode,
            parameters: ,
        }))
    }*/

    /// Returns the display corresponding to this mode.
    #[inline]
    pub fn display(&self) -> &Display {
        &self.display
    }

    /// Returns the dimensions of the region that is visible on the monitor.
    #[inline]
    pub fn visible_region(&self) -> [u32; 2] {
        let ref d = self.parameters.visibleRegion;
        [d.width, d.height]
    }

    /// Returns the refresh rate of this mode.
    #[inline]
    pub fn refresh_rate(&self) -> u32 {
        self.parameters.refreshRate
    }
}
