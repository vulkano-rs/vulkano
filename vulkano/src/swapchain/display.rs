// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Allows you to create surfaces that fill a whole display, outside of the windowing system.
//!
//! **As far as the author knows, no existing device supports these features. Therefore the code
//! here is mostly a draft and needs rework in both the API and the implementation.**
//!
//! The purpose of the objects in this module is to let you create a `Surface` object that
//! represents a location on the screen. This is done in four steps:
//!
//! - Choose a `Display` where the surface will be located. A `Display` represents a display
//!   display, usually a monitor. The available displays can be enumerated with
//!   `Display::enumerate`.
//! - Choose a `DisplayMode`, which is the combination of a display, a resolution and a refresh
//!   rate. You can enumerate the modes available on a display with `Display::display_modes`, or
//!   attempt to create your own mode with `TODO`.
//! - Choose a `DisplayPlane`. A display can show multiple planes in a stacking fashion.
//! - Create a `Surface` object with `Surface::from_display_mode` and pass the chosen `DisplayMode`
//!   and `DisplayPlane`.

#![allow(dead_code)] // TODO: this module isn't finished
#![allow(unused_variables)] // TODO: this module isn't finished

use std::ffi::CStr;
use std::ptr;
use std::sync::Arc;
use std::vec::IntoIter;

use instance::Instance;
use instance::PhysicalDevice;
use swapchain::SupportedSurfaceTransforms;
use swapchain::capabilities;

use OomError;
use VulkanObject;
use check_errors;
use vk;

// TODO: extract this to a `display` module and solve the visibility problems

/// ?
// TODO: plane capabilities
// TODO: store properties in the instance?
pub struct DisplayPlane {
    instance: Arc<Instance>,
    physical_device: usize,
    index: u32,
    properties: vk::DisplayPlanePropertiesKHR,
    supported_displays: Vec<vk::DisplayKHR>,
}

impl DisplayPlane {
    /// See the docs of enumerate().
    pub fn enumerate_raw(device: PhysicalDevice) -> Result<IntoIter<DisplayPlane>, OomError> {
        let vk = device.instance().pointers();

        assert!(device.instance().loaded_extensions().khr_display); // TODO: return error instead

        let num = unsafe {
            let mut num: u32 = 0;
            check_errors(vk.GetPhysicalDeviceDisplayPlanePropertiesKHR(device.internal_object(),
                                                                       &mut num,
                                                                       ptr::null_mut()))?;
            num
        };

        let planes: Vec<vk::DisplayPlanePropertiesKHR> = unsafe {
            let mut planes = Vec::with_capacity(num as usize);
            let mut num = num;
            check_errors(vk.GetPhysicalDeviceDisplayPlanePropertiesKHR(device.internal_object(),
                                                                       &mut num,
                                                                       planes.as_mut_ptr()))?;
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
                instance: device.instance().clone(),
                physical_device: device.index(),
                index: index as u32,
                properties: prop,
                supported_displays: supported_displays,
            }
        }).collect::<Vec<_>>().into_iter())
    }

    /// Enumerates all the display planes that are available on a given physical device.
    ///
    /// # Panic
    ///
    /// - Panics if the device or host ran out of memory.
    ///
    // TODO: move iterator creation here from raw constructor?
    #[inline]
    pub fn enumerate(device: PhysicalDevice) -> IntoIter<DisplayPlane> {
        DisplayPlane::enumerate_raw(device).unwrap()
    }

    /// Returns the physical device that was used to create this display.
    #[inline]
    pub fn physical_device(&self) -> PhysicalDevice {
        PhysicalDevice::from_index(&self.instance, self.physical_device).unwrap()
    }

    /// Returns the index of the plane.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Returns true if this plane supports the given display.
    #[inline]
    pub fn supports(&self, display: &Display) -> bool {
        // making sure that the physical device is the same
        if self.physical_device().internal_object() != display.physical_device().internal_object() {
            return false;
        }

        self.supported_displays
            .iter()
            .find(|&&d| d == display.internal_object())
            .is_some()
    }
}

/// Represents a monitor connected to a physical device.
// TODO: store properties in the instance?
#[derive(Clone)]
pub struct Display {
    instance: Arc<Instance>,
    physical_device: usize,
    properties: Arc<vk::DisplayPropertiesKHR>, // TODO: Arc because struct isn't clone
}

impl Display {
    /// See the docs of enumerate().
    pub fn enumerate_raw(device: PhysicalDevice) -> Result<IntoIter<Display>, OomError> {
        let vk = device.instance().pointers();
        assert!(device.instance().loaded_extensions().khr_display); // TODO: return error instead

        let num = unsafe {
            let mut num = 0;
            check_errors(vk.GetPhysicalDeviceDisplayPropertiesKHR(device.internal_object(),
                                                                  &mut num,
                                                                  ptr::null_mut()))?;
            num
        };

        let displays: Vec<vk::DisplayPropertiesKHR> = unsafe {
            let mut displays = Vec::with_capacity(num as usize);
            let mut num = num;
            check_errors(vk.GetPhysicalDeviceDisplayPropertiesKHR(device.internal_object(),
                                                                  &mut num,
                                                                  displays.as_mut_ptr()))?;
            displays.set_len(num as usize);
            displays
        };

        Ok(displays
               .into_iter()
               .map(|prop| {
                        Display {
                            instance: device.instance().clone(),
                            physical_device: device.index(),
                            properties: Arc::new(prop),
                        }
                    })
               .collect::<Vec<_>>()
               .into_iter())
    }

    /// Enumerates all the displays that are available on a given physical device.
    ///
    /// # Panic
    ///
    /// - Panics if the device or host ran out of memory.
    ///
    // TODO: move iterator creation here from raw constructor?
    #[inline]
    pub fn enumerate(device: PhysicalDevice) -> IntoIter<Display> {
        Display::enumerate_raw(device).unwrap()
    }

    /// Returns the name of the display.
    #[inline]
    pub fn name(&self) -> &str {
        unsafe {
            CStr::from_ptr(self.properties.displayName)
                .to_str()
                .expect("non UTF-8 characters in display name")
        }
    }

    /// Returns the physical device that was used to create this display.
    #[inline]
    pub fn physical_device(&self) -> PhysicalDevice {
        PhysicalDevice::from_index(&self.instance, self.physical_device).unwrap()
    }

    /// Returns the physical dimensions of the display in millimeters.
    #[inline]
    pub fn physical_dimensions(&self) -> [u32; 2] {
        let ref r = self.properties.physicalDimensions;
        [r.width, r.height]
    }

    /// Returns the physical, native, or preferred resolution of the display.
    ///
    /// > **Note**: The display is usually still capable of displaying other resolutions. This is
    /// > only the "best" resolution.
    #[inline]
    pub fn physical_resolution(&self) -> [u32; 2] {
        let ref r = self.properties.physicalResolution;
        [r.width, r.height]
    }

    /// Returns the transforms supported by this display.
    #[inline]
    pub fn supported_transforms(&self) -> SupportedSurfaceTransforms {
        capabilities::surface_transforms_from_bits(self.properties.supportedTransforms)
    }

    /// Returns true if TODO.
    #[inline]
    pub fn plane_reorder_possible(&self) -> bool {
        self.properties.planeReorderPossible != 0
    }

    /// Returns true if TODO.
    #[inline]
    pub fn persistent_content(&self) -> bool {
        self.properties.persistentContent != 0
    }

    /// See the docs of display_modes().
    pub fn display_modes_raw(&self) -> Result<IntoIter<DisplayMode>, OomError> {
        let vk = self.instance.pointers();

        let num = unsafe {
            let mut num = 0;
            check_errors(vk.GetDisplayModePropertiesKHR(self.physical_device().internal_object(),
                                                        self.properties.display,
                                                        &mut num,
                                                        ptr::null_mut()))?;
            num
        };

        let modes: Vec<vk::DisplayModePropertiesKHR> = unsafe {
            let mut modes = Vec::with_capacity(num as usize);
            let mut num = num;
            check_errors(vk.GetDisplayModePropertiesKHR(self.physical_device().internal_object(),
                                                        self.properties.display,
                                                        &mut num,
                                                        modes.as_mut_ptr()))?;
            modes.set_len(num as usize);
            modes
        };

        Ok(modes
               .into_iter()
               .map(|mode| {
                        DisplayMode {
                            display: self.clone(),
                            display_mode: mode.displayMode,
                            parameters: mode.parameters,
                        }
                    })
               .collect::<Vec<_>>()
               .into_iter())
    }

    /// Returns a list of all modes available on this display.
    ///
    /// # Panic
    ///
    /// - Panics if the device or host ran out of memory.
    ///
    // TODO: move iterator creation here from display_modes_raw?
    #[inline]
    pub fn display_modes(&self) -> IntoIter<DisplayMode> {
        self.display_modes_raw().unwrap()
    }
}

unsafe impl VulkanObject for Display {
    type Object = vk::DisplayKHR;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_DISPLAY_KHR;

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
        assert!(device.instance().loaded_extensions().khr_display);     // TODO: return error instead

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
            check_errors(vk.CreateDisplayModeKHR(display.device.internal_object(),
                                                      display.display, &infos, ptr::null(),
                                                      &mut output))?;
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

unsafe impl VulkanObject for DisplayMode {
    type Object = vk::DisplayModeKHR;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_DISPLAY_MODE_KHR;

    #[inline]
    fn internal_object(&self) -> vk::DisplayModeKHR {
        self.display_mode
    }
}
