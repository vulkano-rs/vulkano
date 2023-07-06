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
//! - Create a `Surface` object with `Surface::from_display_plane` and pass the chosen `DisplayMode`
//!   and `DisplayPlane`.

#![allow(dead_code)] // TODO: this module isn't finished
#![allow(unused_variables)] // TODO: this module isn't finished

use crate::{
    device::physical::PhysicalDevice, swapchain::SurfaceTransforms, OomError, VulkanError,
    VulkanObject,
};
use std::{
    ffi::CStr,
    fmt::{Display as FmtDisplay, Error as FmtError, Formatter},
    ptr,
    sync::Arc,
    vec::IntoIter,
};

// TODO: extract this to a `display` module and solve the visibility problems

/// ?
// TODO: plane capabilities
// TODO: store properties in the instance?
pub struct DisplayPlane {
    physical_device: Arc<PhysicalDevice>,
    index: u32,
    properties: ash::vk::DisplayPlanePropertiesKHR,
    supported_displays: Vec<ash::vk::DisplayKHR>,
}

impl DisplayPlane {
    /// See the docs of enumerate().
    pub fn enumerate_raw(
        physical_device: Arc<PhysicalDevice>,
    ) -> Result<IntoIter<DisplayPlane>, OomError> {
        let fns = physical_device.instance().fns();

        assert!(physical_device.instance().enabled_extensions().khr_display); // TODO: return error instead

        let display_plane_properties = unsafe {
            loop {
                let mut count = 0;
                (fns.khr_display
                    .get_physical_device_display_plane_properties_khr)(
                    physical_device.handle(),
                    &mut count,
                    ptr::null_mut(),
                )
                .result()
                .map_err(VulkanError::from)?;

                let mut properties = Vec::with_capacity(count as usize);
                let result = (fns
                    .khr_display
                    .get_physical_device_display_plane_properties_khr)(
                    physical_device.handle(),
                    &mut count,
                    properties.as_mut_ptr(),
                );

                match result {
                    ash::vk::Result::SUCCESS => {
                        properties.set_len(count as usize);
                        break properties;
                    }
                    ash::vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err).into()),
                }
            }
        };

        Ok(display_plane_properties
            .into_iter()
            .enumerate()
            .map(|(index, prop)| {
                let supported_displays = unsafe {
                    loop {
                        let mut count = 0;
                        (fns.khr_display.get_display_plane_supported_displays_khr)(
                            physical_device.handle(),
                            index as u32,
                            &mut count,
                            ptr::null_mut(),
                        )
                        .result()
                        .map_err(VulkanError::from)
                        .unwrap(); // TODO: shouldn't unwrap

                        let mut displays = Vec::with_capacity(count as usize);
                        let result = (fns.khr_display.get_display_plane_supported_displays_khr)(
                            physical_device.handle(),
                            index as u32,
                            &mut count,
                            displays.as_mut_ptr(),
                        );

                        match result {
                            ash::vk::Result::SUCCESS => {
                                displays.set_len(count as usize);
                                break displays;
                            }
                            ash::vk::Result::INCOMPLETE => (),
                            err => todo!(), // TODO: shouldn't panic
                        }
                    }
                };

                DisplayPlane {
                    physical_device: physical_device.clone(),
                    index: index as u32,
                    properties: prop,
                    supported_displays,
                }
            })
            .collect::<Vec<_>>()
            .into_iter())
    }

    /// Enumerates all the display planes that are available on a given physical device.
    ///
    /// # Panics
    ///
    /// - Panics if the device or host ran out of memory.
    // TODO: move iterator creation here from raw constructor?
    #[inline]
    pub fn enumerate(physical_device: Arc<PhysicalDevice>) -> IntoIter<DisplayPlane> {
        DisplayPlane::enumerate_raw(physical_device).unwrap()
    }

    /// Returns the physical device that was used to create this display.
    #[inline]
    pub fn physical_device(&self) -> &Arc<PhysicalDevice> {
        &self.physical_device
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
        if self.physical_device().handle() != display.physical_device().handle() {
            return false;
        }

        self.supported_displays
            .iter()
            .any(|&d| d == display.handle())
    }
}

/// Represents a monitor connected to a physical device.
// TODO: store properties in the instance?
#[derive(Clone)]
pub struct Display {
    physical_device: Arc<PhysicalDevice>,
    properties: Arc<ash::vk::DisplayPropertiesKHR>, // TODO: Arc because struct isn't clone
}

impl Display {
    /// See the docs of enumerate().
    pub fn enumerate_raw(
        physical_device: Arc<PhysicalDevice>,
    ) -> Result<IntoIter<Display>, OomError> {
        let fns = physical_device.instance().fns();
        assert!(physical_device.instance().enabled_extensions().khr_display); // TODO: return error instead

        let display_properties = unsafe {
            loop {
                let mut count = 0;
                (fns.khr_display.get_physical_device_display_properties_khr)(
                    physical_device.handle(),
                    &mut count,
                    ptr::null_mut(),
                )
                .result()
                .map_err(VulkanError::from)?;

                let mut properties = Vec::with_capacity(count as usize);
                let result = (fns.khr_display.get_physical_device_display_properties_khr)(
                    physical_device.handle(),
                    &mut count,
                    properties.as_mut_ptr(),
                );

                match result {
                    ash::vk::Result::SUCCESS => {
                        properties.set_len(count as usize);
                        break properties;
                    }
                    ash::vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err).into()),
                }
            }
        };

        Ok(display_properties
            .into_iter()
            .map(|prop| Display {
                physical_device: physical_device.clone(),
                properties: Arc::new(prop),
            })
            .collect::<Vec<_>>()
            .into_iter())
    }

    /// Enumerates all the displays that are available on a given physical device.
    ///
    /// # Panics
    ///
    /// - Panics if the device or host ran out of memory.
    // TODO: move iterator creation here from raw constructor?
    #[inline]
    pub fn enumerate(physical_device: Arc<PhysicalDevice>) -> IntoIter<Display> {
        Display::enumerate_raw(physical_device).unwrap()
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
    pub fn physical_device(&self) -> &Arc<PhysicalDevice> {
        &self.physical_device
    }

    /// Returns the physical dimensions of the display in millimeters.
    #[inline]
    pub fn physical_dimensions(&self) -> [u32; 2] {
        let r = &self.properties.physical_dimensions;
        [r.width, r.height]
    }

    /// Returns the physical, native, or preferred resolution of the display.
    ///
    /// > **Note**: The display is usually still capable of displaying other resolutions. This is
    /// > only the "best" resolution.
    #[inline]
    pub fn physical_resolution(&self) -> [u32; 2] {
        let r = &self.properties.physical_resolution;
        [r.width, r.height]
    }

    /// Returns the transforms supported by this display.
    #[inline]
    pub fn supported_transforms(&self) -> SurfaceTransforms {
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
        let fns = self.physical_device.instance().fns();

        let mode_properties = unsafe {
            loop {
                let mut count = 0;
                (fns.khr_display.get_display_mode_properties_khr)(
                    self.physical_device().handle(),
                    self.properties.display,
                    &mut count,
                    ptr::null_mut(),
                )
                .result()
                .map_err(VulkanError::from)?;

                let mut properties = Vec::with_capacity(count as usize);
                let result = (fns.khr_display.get_display_mode_properties_khr)(
                    self.physical_device().handle(),
                    self.properties.display,
                    &mut count,
                    properties.as_mut_ptr(),
                );

                match result {
                    ash::vk::Result::SUCCESS => {
                        properties.set_len(count as usize);
                        break properties;
                    }
                    ash::vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err).into()),
                }
            }
        };

        Ok(mode_properties
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
    /// # Panics
    ///
    /// - Panics if the device or host ran out of memory.
    // TODO: move iterator creation here from display_modes_raw?
    #[inline]
    pub fn display_modes(&self) -> IntoIter<DisplayMode> {
        self.display_modes_raw().unwrap()
    }
}

unsafe impl VulkanObject for Display {
    type Handle = ash::vk::DisplayKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
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
            (fns.v1_0.CreateDisplayModeKHR)(display.device.handle(),
                                                      display.display, &infos, ptr::null(),
                                                      &mut output).result().map_err(VulkanError::from)?;
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
        let d = &self.parameters.visible_region;
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

impl FmtDisplay for DisplayMode {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
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
    type Handle = ash::vk::DisplayModeKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.display_mode
    }
}
