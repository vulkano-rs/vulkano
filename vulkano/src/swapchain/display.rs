// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
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

use crate::check_errors;
use crate::device::physical::PhysicalDevice;
use crate::instance::Instance;
use crate::swapchain::SupportedSurfaceTransforms;
use crate::OomError;
use crate::VulkanObject;
use std::ffi::CStr;
use std::fmt::Formatter;
use std::sync::Arc;
use std::vec::IntoIter;
use std::{fmt, ptr};

// TODO: extract this to a `display` module and solve the visibility problems

/// ?
// TODO: plane capabilities
// TODO: store properties in the instance?
pub struct DisplayPlane {
    instance: Arc<Instance>,
    physical_device: usize,
    index: u32,
    properties: ash::vk::DisplayPlanePropertiesKHR,
    supported_displays: Vec<ash::vk::DisplayKHR>,
}

impl DisplayPlane {
    /// See the docs of enumerate().
    pub fn enumerate_raw(device: PhysicalDevice) -> Result<IntoIter<DisplayPlane>, OomError> {
        let fns = device.instance().fns();

        assert!(device.instance().enabled_extensions().khr_display); // TODO: return error instead

        let num = unsafe {
            let mut num: u32 = 0;
            check_errors(
                fns.khr_display
                    .get_physical_device_display_plane_properties_khr(
                        device.internal_object(),
                        &mut num,
                        ptr::null_mut(),
                    ),
            )?;
            num
        };

        let planes: Vec<ash::vk::DisplayPlanePropertiesKHR> = unsafe {
            let mut planes = Vec::with_capacity(num as usize);
            let mut num = num;
            check_errors(
                fns.khr_display
                    .get_physical_device_display_plane_properties_khr(
                        device.internal_object(),
                        &mut num,
                        planes.as_mut_ptr(),
                    ),
            )?;
            planes.set_len(num as usize);
            planes
        };

        Ok(planes
            .into_iter()
            .enumerate()
            .map(|(index, prop)| {
                let num = unsafe {
                    let mut num: u32 = 0;
                    check_errors(fns.khr_display.get_display_plane_supported_displays_khr(
                        device.internal_object(),
                        index as u32,
                        &mut num,
                        ptr::null_mut(),
                    ))
                    .unwrap(); // TODO: shouldn't unwrap
                    num
                };

                let supported_displays: Vec<ash::vk::DisplayKHR> = unsafe {
                    let mut displays = Vec::with_capacity(num as usize);
                    let mut num = num;
                    check_errors(fns.khr_display.get_display_plane_supported_displays_khr(
                        device.internal_object(),
                        index as u32,
                        &mut num,
                        displays.as_mut_ptr(),
                    ))
                    .unwrap(); // TODO: shouldn't unwrap
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
            })
            .collect::<Vec<_>>()
            .into_iter())
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
    properties: Arc<ash::vk::DisplayPropertiesKHR>, // TODO: Arc because struct isn't clone
}

impl Display {
    /// See the docs of enumerate().
    pub fn enumerate_raw(device: PhysicalDevice) -> Result<IntoIter<Display>, OomError> {
        let fns = device.instance().fns();
        assert!(device.instance().enabled_extensions().khr_display); // TODO: return error instead

        let num = unsafe {
            let mut num = 0;
            check_errors(fns.khr_display.get_physical_device_display_properties_khr(
                device.internal_object(),
                &mut num,
                ptr::null_mut(),
            ))?;
            num
        };

        let displays: Vec<ash::vk::DisplayPropertiesKHR> = unsafe {
            let mut displays = Vec::with_capacity(num as usize);
            let mut num = num;
            check_errors(fns.khr_display.get_physical_device_display_properties_khr(
                device.internal_object(),
                &mut num,
                displays.as_mut_ptr(),
            ))?;
            displays.set_len(num as usize);
            displays
        };

        Ok(displays
            .into_iter()
            .map(|prop| Display {
                instance: device.instance().clone(),
                physical_device: device.index(),
                properties: Arc::new(prop),
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
            CStr::from_ptr(self.properties.display_name)
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
        let ref r = self.properties.physical_dimensions;
        [r.width, r.height]
    }

    /// Returns the physical, native, or preferred resolution of the display.
    ///
    /// > **Note**: The display is usually still capable of displaying other resolutions. This is
    /// > only the "best" resolution.
    #[inline]
    pub fn physical_resolution(&self) -> [u32; 2] {
        let ref r = self.properties.physical_resolution;
        [r.width, r.height]
    }

    /// Returns the transforms supported by this display.
    #[inline]
    pub fn supported_transforms(&self) -> SupportedSurfaceTransforms {
        self.properties.supported_transforms.into()
    }

    /// Returns true if TODO.
    #[inline]
    pub fn plane_reorder_possible(&self) -> bool {
        self.properties.plane_reorder_possible != 0
    }

    /// Returns true if TODO.
    #[inline]
    pub fn persistent_content(&self) -> bool {
        self.properties.persistent_content != 0
    }

    /// See the docs of display_modes().
    pub fn display_modes_raw(&self) -> Result<IntoIter<DisplayMode>, OomError> {
        let fns = self.instance.fns();

        let num = unsafe {
            let mut num = 0;
            check_errors(fns.khr_display.get_display_mode_properties_khr(
                self.physical_device().internal_object(),
                self.properties.display,
                &mut num,
                ptr::null_mut(),
            ))?;
            num
        };

        let modes: Vec<ash::vk::DisplayModePropertiesKHR> = unsafe {
            let mut modes = Vec::with_capacity(num as usize);
            let mut num = num;
            check_errors(fns.khr_display.get_display_mode_properties_khr(
                self.physical_device().internal_object(),
                self.properties.display,
                &mut num,
                modes.as_mut_ptr(),
            ))?;
            modes.set_len(num as usize);
            modes
        };

        Ok(modes
            .into_iter()
            .map(|mode| DisplayMode {
                display: self.clone(),
                display_mode: mode.display_mode,
                parameters: mode.parameters,
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
    type Object = ash::vk::DisplayKHR;

    #[inline]
    fn internal_object(&self) -> ash::vk::DisplayKHR {
        self.properties.display
    }
}

/// Represents a mode on a specific display.
///
/// A display mode describes a supported display resolution and refresh rate.
pub struct DisplayMode {
    display: Display,
    display_mode: ash::vk::DisplayModeKHR,
    parameters: ash::vk::DisplayModeParametersKHR,
}

impl DisplayMode {
    /*pub fn new(display: &Display) -> Result<Arc<DisplayMode>, OomError> {
        let fns = instance.fns();
        assert!(device.instance().enabled_extensions().khr_display);     // TODO: return error instead

        let parameters = ash::vk::DisplayModeParametersKHR {
            visibleRegion: ash::vk::Extent2D { width: , height:  },
            refreshRate: ,
        };

        let display_mode = {
            let infos = ash::vk::DisplayModeCreateInfoKHR {
                flags: ash::vk::DisplayModeCreateFlags::empty(),
                parameters: parameters,
                ..Default::default()
            };

            let mut output = mem::uninitialized();
            check_errors(fns.v1_0.CreateDisplayModeKHR(display.device.internal_object(),
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
        let ref d = self.parameters.visible_region;
        [d.width, d.height]
    }

    /// Returns the refresh rate of this mode.
    ///
    /// The returned value is multiplied by 1000. As such the value is in terms of millihertz (mHz).
    /// For example, a 60Hz display mode would have a refresh rate of 60,000 mHz.
    #[inline]
    pub fn refresh_rate(&self) -> u32 {
        self.parameters.refresh_rate
    }
}

impl fmt::Display for DisplayMode {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let visible_region = self.visible_region();

        write!(
            f,
            "{}×{}px @ {}.{:03} Hz",
            visible_region[0],
            visible_region[1],
            self.refresh_rate() / 1000,
            self.refresh_rate() % 1000
        )
    }
}

unsafe impl VulkanObject for DisplayMode {
    type Object = ash::vk::DisplayModeKHR;

    #[inline]
    fn internal_object(&self) -> ash::vk::DisplayModeKHR {
        self.display_mode
    }
}
