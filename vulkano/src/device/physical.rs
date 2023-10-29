// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::QueueFamilyProperties;
use crate::{
    buffer::{ExternalBufferInfo, ExternalBufferProperties},
    cache::{OnceCache, WeakArcOnceCache},
    device::{properties::Properties, DeviceExtensions, Features, FeaturesFfi, PropertiesFfi},
    display::{Display, DisplayPlaneProperties, DisplayPlanePropertiesRaw, DisplayProperties},
    format::{DrmFormatModifierProperties, Format, FormatProperties},
    image::{
        ImageDrmFormatModifierInfo, ImageFormatInfo, ImageFormatProperties, ImageUsage,
        SparseImageFormatInfo, SparseImageFormatProperties,
    },
    instance::{Instance, InstanceOwned},
    macros::{impl_id_counter, vulkan_bitflags, vulkan_enum},
    memory::{ExternalMemoryHandleType, MemoryProperties},
    swapchain::{
        ColorSpace, FullScreenExclusive, PresentMode, Surface, SurfaceApi, SurfaceCapabilities,
        SurfaceInfo, SurfaceTransforms,
    },
    sync::{
        fence::{ExternalFenceInfo, ExternalFenceProperties},
        semaphore::{ExternalSemaphoreInfo, ExternalSemaphoreProperties, SemaphoreType},
        Sharing,
    },
    DebugWrapper, ExtensionProperties, Requires, RequiresAllOf, RequiresOneOf, Validated,
    ValidationError, Version, VulkanError, VulkanObject,
};
use bytemuck::cast_slice;
use parking_lot::RwLock;
use std::{
    ffi::CStr,
    fmt::{Debug, Error as FmtError, Formatter},
    mem::MaybeUninit,
    num::NonZeroU64,
    ptr,
    sync::Arc,
};

/// Represents one of the available physical devices on this machine.
///
/// # Examples
///
/// ```no_run
/// # use vulkano::{
/// #     instance::{Instance, InstanceExtensions},
/// #     Version, VulkanLibrary,
/// # };
/// use vulkano::device::physical::PhysicalDevice;
///
/// # let library = VulkanLibrary::new().unwrap();
/// # let instance = Instance::new(library, Default::default()).unwrap();
/// for physical_device in instance.enumerate_physical_devices().unwrap() {
///     print_infos(&physical_device);
/// }
///
/// fn print_infos(dev: &PhysicalDevice) {
///     println!("Name: {}", dev.properties().device_name);
/// }
/// ```
pub struct PhysicalDevice {
    handle: ash::vk::PhysicalDevice,
    instance: DebugWrapper<Arc<Instance>>,
    id: NonZeroU64,

    // Data queried at `PhysicalDevice` creation.
    api_version: Version,
    supported_extensions: DeviceExtensions,
    supported_features: Features,
    properties: Properties,
    extension_properties: Vec<ExtensionProperties>,
    memory_properties: MemoryProperties,
    queue_family_properties: Vec<QueueFamilyProperties>,

    // Data queried by the user at runtime, cached for faster lookups.
    display_properties: WeakArcOnceCache<ash::vk::DisplayKHR, Display>,
    display_plane_properties: RwLock<Vec<DisplayPlanePropertiesRaw>>,
    external_buffer_properties: OnceCache<ExternalBufferInfo, ExternalBufferProperties>,
    external_fence_properties: OnceCache<ExternalFenceInfo, ExternalFenceProperties>,
    external_semaphore_properties: OnceCache<ExternalSemaphoreInfo, ExternalSemaphoreProperties>,
    format_properties: OnceCache<Format, FormatProperties>,
    image_format_properties: OnceCache<ImageFormatInfo, Option<ImageFormatProperties>>,
    sparse_image_format_properties:
        OnceCache<SparseImageFormatInfo, Vec<SparseImageFormatProperties>>,
}

impl PhysicalDevice {
    /// Creates a new `PhysicalDevice` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `instance`.
    pub unsafe fn from_handle(
        instance: Arc<Instance>,
        handle: ash::vk::PhysicalDevice,
    ) -> Result<Arc<Self>, VulkanError> {
        let api_version = Self::get_api_version(handle, &instance);
        let extension_properties = Self::get_extension_properties(handle, &instance)?;
        let supported_extensions: DeviceExtensions = extension_properties
            .iter()
            .map(|property| property.extension_name.as_str())
            .collect();

        let supported_features;
        let properties;
        let memory_properties;
        let queue_family_properties;

        // Get the remaining infos.
        // If possible, we use VK_KHR_get_physical_device_properties2.
        if api_version >= Version::V1_1
            || instance
                .enabled_extensions()
                .khr_get_physical_device_properties2
        {
            supported_features =
                Self::get_features2(handle, &instance, api_version, &supported_extensions);
            properties =
                Self::get_properties2(handle, &instance, api_version, &supported_extensions);
            memory_properties = Self::get_memory_properties2(handle, &instance);
            queue_family_properties = Self::get_queue_family_properties2(handle, &instance);
        } else {
            supported_features = Self::get_features(handle, &instance);
            properties =
                Self::get_properties(handle, &instance, api_version, &supported_extensions);
            memory_properties = Self::get_memory_properties(handle, &instance);
            queue_family_properties = Self::get_queue_family_properties(handle, &instance);
        };

        Ok(Arc::new(PhysicalDevice {
            handle,
            instance: DebugWrapper(instance),
            id: Self::next_id(),

            api_version,
            supported_extensions,
            supported_features,
            properties,
            extension_properties,
            memory_properties,
            queue_family_properties,

            display_properties: WeakArcOnceCache::new(),
            display_plane_properties: RwLock::new(Vec::new()),
            external_buffer_properties: OnceCache::new(),
            external_fence_properties: OnceCache::new(),
            external_semaphore_properties: OnceCache::new(),
            format_properties: OnceCache::new(),
            image_format_properties: OnceCache::new(),
            sparse_image_format_properties: OnceCache::new(),
        }))
    }

    unsafe fn get_api_version(handle: ash::vk::PhysicalDevice, instance: &Instance) -> Version {
        let fns = instance.fns();
        let mut output = MaybeUninit::uninit();
        (fns.v1_0.get_physical_device_properties)(handle, output.as_mut_ptr());
        let api_version = Version::try_from(output.assume_init().api_version).unwrap();
        std::cmp::min(instance.max_api_version(), api_version)
    }

    unsafe fn get_extension_properties(
        handle: ash::vk::PhysicalDevice,
        instance: &Instance,
    ) -> Result<Vec<ExtensionProperties>, VulkanError> {
        let fns = instance.fns();

        loop {
            let mut count = 0;
            (fns.v1_0.enumerate_device_extension_properties)(
                handle,
                ptr::null(),
                &mut count,
                ptr::null_mut(),
            )
            .result()
            .map_err(VulkanError::from)?;

            let mut output = Vec::with_capacity(count as usize);
            let result = (fns.v1_0.enumerate_device_extension_properties)(
                handle,
                ptr::null(),
                &mut count,
                output.as_mut_ptr(),
            );

            match result {
                ash::vk::Result::SUCCESS => {
                    output.set_len(count as usize);
                    return Ok(output.into_iter().map(Into::into).collect());
                }
                ash::vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        }
    }

    unsafe fn get_features(handle: ash::vk::PhysicalDevice, instance: &Instance) -> Features {
        let mut output = FeaturesFfi::default();

        let fns = instance.fns();
        (fns.v1_0.get_physical_device_features)(handle, &mut output.head_as_mut().features);

        Features::from(&output)
    }

    unsafe fn get_features2(
        handle: ash::vk::PhysicalDevice,
        instance: &Instance,
        api_version: Version,
        supported_extensions: &DeviceExtensions,
    ) -> Features {
        let mut output = FeaturesFfi::default();
        output.make_chain(
            api_version,
            supported_extensions,
            instance.enabled_extensions(),
        );

        let fns = instance.fns();

        if instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_features2)(handle, output.head_as_mut());
        } else {
            (fns.khr_get_physical_device_properties2
                .get_physical_device_features2_khr)(handle, output.head_as_mut());
        }

        Features::from(&output)
    }

    unsafe fn get_properties(
        handle: ash::vk::PhysicalDevice,
        instance: &Instance,
        api_version: Version,
        supported_extensions: &DeviceExtensions,
    ) -> Properties {
        let mut output = PropertiesFfi::default();
        output.make_chain(
            api_version,
            supported_extensions,
            instance.enabled_extensions(),
        );

        let fns = instance.fns();
        (fns.v1_0.get_physical_device_properties)(handle, &mut output.head_as_mut().properties);

        Properties::from(&output)
    }

    unsafe fn get_properties2(
        handle: ash::vk::PhysicalDevice,
        instance: &Instance,
        api_version: Version,
        supported_extensions: &DeviceExtensions,
    ) -> Properties {
        let mut output = PropertiesFfi::default();
        output.make_chain(
            api_version,
            supported_extensions,
            instance.enabled_extensions(),
        );

        let fns = instance.fns();

        if instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_properties2)(handle, output.head_as_mut());
        } else {
            (fns.khr_get_physical_device_properties2
                .get_physical_device_properties2_khr)(handle, output.head_as_mut());
        }

        Properties::from(&output)
    }

    unsafe fn get_memory_properties(
        handle: ash::vk::PhysicalDevice,
        instance: &Instance,
    ) -> MemoryProperties {
        let mut output = MaybeUninit::uninit();

        let fns = instance.fns();
        (fns.v1_0.get_physical_device_memory_properties)(handle, output.as_mut_ptr());

        output.assume_init().into()
    }

    unsafe fn get_memory_properties2(
        handle: ash::vk::PhysicalDevice,
        instance: &Instance,
    ) -> MemoryProperties {
        let mut output = ash::vk::PhysicalDeviceMemoryProperties2KHR::default();

        let fns = instance.fns();

        if instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_memory_properties2)(handle, &mut output);
        } else {
            (fns.khr_get_physical_device_properties2
                .get_physical_device_memory_properties2_khr)(handle, &mut output);
        }

        output.memory_properties.into()
    }

    unsafe fn get_queue_family_properties(
        handle: ash::vk::PhysicalDevice,
        instance: &Instance,
    ) -> Vec<QueueFamilyProperties> {
        let fns = instance.fns();

        let mut num = 0;
        (fns.v1_0.get_physical_device_queue_family_properties)(handle, &mut num, ptr::null_mut());

        let mut output = Vec::with_capacity(num as usize);
        (fns.v1_0.get_physical_device_queue_family_properties)(
            handle,
            &mut num,
            output.as_mut_ptr(),
        );
        output.set_len(num as usize);

        output.into_iter().map(Into::into).collect()
    }

    unsafe fn get_queue_family_properties2(
        handle: ash::vk::PhysicalDevice,
        instance: &Instance,
    ) -> Vec<QueueFamilyProperties> {
        let mut num = 0;
        let fns = instance.fns();

        if instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_queue_family_properties2)(
                handle,
                &mut num,
                ptr::null_mut(),
            );
        } else {
            (fns.khr_get_physical_device_properties2
                .get_physical_device_queue_family_properties2_khr)(
                handle,
                &mut num,
                ptr::null_mut(),
            );
        }

        let mut output = vec![ash::vk::QueueFamilyProperties2::default(); num as usize];

        if instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_queue_family_properties2)(
                handle,
                &mut num,
                output.as_mut_ptr(),
            );
        } else {
            (fns.khr_get_physical_device_properties2
                .get_physical_device_queue_family_properties2_khr)(
                handle,
                &mut num,
                output.as_mut_ptr(),
            );
        }

        output
            .into_iter()
            .map(|family| family.queue_family_properties.into())
            .collect()
    }

    /// Returns the instance that owns the physical device.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Returns the version of Vulkan supported by the physical device.
    ///
    /// Unlike the `api_version` property, which is the version reported by the device directly,
    /// this function returns the version the device can actually support, based on the instance's
    /// `max_api_version`.
    #[inline]
    pub fn api_version(&self) -> Version {
        self.api_version
    }

    /// Returns the properties reported by the physical device.
    #[inline]
    pub fn properties(&self) -> &Properties {
        &self.properties
    }

    /// Returns the extension properties reported by the physical device.
    #[inline]
    pub fn extension_properties(&self) -> &[ExtensionProperties] {
        &self.extension_properties
    }

    /// Returns the extensions that are supported by the physical device.
    #[inline]
    pub fn supported_extensions(&self) -> &DeviceExtensions {
        &self.supported_extensions
    }

    /// Returns the features that are supported by the physical device.
    #[inline]
    pub fn supported_features(&self) -> &Features {
        &self.supported_features
    }

    /// Returns the memory properties reported by the physical device.
    #[inline]
    pub fn memory_properties(&self) -> &MemoryProperties {
        &self.memory_properties
    }

    /// Returns the queue family properties reported by the physical device.
    #[inline]
    pub fn queue_family_properties(&self) -> &[QueueFamilyProperties] {
        &self.queue_family_properties
    }

    /// Queries whether the physical device supports presenting to DirectFB surfaces from queues of
    /// the given queue family.
    ///
    /// # Safety
    ///
    /// - `dfb` must be a valid DirectFB `IDirectFB` handle.
    #[inline]
    pub unsafe fn directfb_presentation_support<D>(
        &self,
        queue_family_index: u32,
        dfb: *const D,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_directfb_presentation_support(queue_family_index, dfb)?;

        Ok(self.directfb_presentation_support_unchecked(queue_family_index, dfb))
    }

    fn validate_directfb_presentation_support<D>(
        &self,
        queue_family_index: u32,
        _dfb: *const D,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().ext_directfb_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_directfb_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceDirectFBPresentationSupportEXT-queueFamilyIndex-04119",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceDirectFBPresentationSupportEXT-dfb-parameter
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn directfb_presentation_support_unchecked<D>(
        &self,
        queue_family_index: u32,
        dfb: *const D,
    ) -> bool {
        let fns = self.instance.fns();
        (fns.ext_directfb_surface
            .get_physical_device_direct_fb_presentation_support_ext)(
            self.handle,
            queue_family_index,
            dfb as *mut _,
        ) != 0
    }

    /// Returns the properties of displays attached to the physical device.
    #[inline]
    pub fn display_properties<'a>(
        self: &'a Arc<Self>,
    ) -> Result<Vec<Arc<Display>>, Validated<VulkanError>> {
        self.validate_display_properties()?;

        unsafe { Ok(self.display_properties_unchecked()?) }
    }

    fn validate_display_properties(&self) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_display {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_display",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn display_properties_unchecked<'a>(
        self: &'a Arc<Self>,
    ) -> Result<Vec<Arc<Display>>, VulkanError> {
        let fns = self.instance.fns();

        if self
            .instance
            .enabled_extensions()
            .khr_get_display_properties2
        {
            let properties_vk = unsafe {
                loop {
                    let mut count = 0;
                    (fns.khr_get_display_properties2
                        .get_physical_device_display_properties2_khr)(
                        self.handle,
                        &mut count,
                        ptr::null_mut(),
                    )
                    .result()
                    .map_err(VulkanError::from)?;

                    let mut properties =
                        vec![ash::vk::DisplayProperties2KHR::default(); count as usize];
                    let result = (fns
                        .khr_get_display_properties2
                        .get_physical_device_display_properties2_khr)(
                        self.handle,
                        &mut count,
                        properties.as_mut_ptr(),
                    );

                    match result {
                        ash::vk::Result::SUCCESS => {
                            properties.set_len(count as usize);
                            break properties;
                        }
                        ash::vk::Result::INCOMPLETE => (),
                        err => return Err(VulkanError::from(err)),
                    }
                }
            };

            Ok(properties_vk
                .into_iter()
                .map(|properties_vk| {
                    let properties_vk = &properties_vk.display_properties;
                    self.display_properties
                        .get_or_insert(properties_vk.display, |&handle| {
                            let properties = DisplayProperties {
                                name: properties_vk.display_name.as_ref().map(|name| {
                                    CStr::from_ptr(name)
                                        .to_str()
                                        .expect("non UTF-8 characters in display name")
                                        .to_owned()
                                }),
                                physical_dimensions: [
                                    properties_vk.physical_dimensions.width,
                                    properties_vk.physical_dimensions.height,
                                ],
                                physical_resolution: [
                                    properties_vk.physical_resolution.width,
                                    properties_vk.physical_resolution.height,
                                ],
                                supported_transforms: properties_vk.supported_transforms.into(),
                                plane_reorder_possible: properties_vk.plane_reorder_possible
                                    != ash::vk::FALSE,
                                persistent_content: properties_vk.persistent_content
                                    != ash::vk::FALSE,
                            };

                            Display::from_handle(self.clone(), handle, properties)
                        })
                })
                .collect())
        } else {
            let properties_vk = unsafe {
                loop {
                    let mut count = 0;
                    (fns.khr_display.get_physical_device_display_properties_khr)(
                        self.handle,
                        &mut count,
                        ptr::null_mut(),
                    )
                    .result()
                    .map_err(VulkanError::from)?;

                    let mut properties = Vec::with_capacity(count as usize);
                    let result = (fns.khr_display.get_physical_device_display_properties_khr)(
                        self.handle,
                        &mut count,
                        properties.as_mut_ptr(),
                    );

                    match result {
                        ash::vk::Result::SUCCESS => {
                            properties.set_len(count as usize);
                            break properties;
                        }
                        ash::vk::Result::INCOMPLETE => (),
                        err => return Err(VulkanError::from(err)),
                    }
                }
            };

            Ok(properties_vk
                .into_iter()
                .map(|properties_vk| {
                    self.display_properties
                        .get_or_insert(properties_vk.display, |&handle| {
                            let properties = DisplayProperties {
                                name: properties_vk.display_name.as_ref().map(|name| {
                                    CStr::from_ptr(name)
                                        .to_str()
                                        .expect("non UTF-8 characters in display name")
                                        .to_owned()
                                }),
                                physical_dimensions: [
                                    properties_vk.physical_dimensions.width,
                                    properties_vk.physical_dimensions.height,
                                ],
                                physical_resolution: [
                                    properties_vk.physical_resolution.width,
                                    properties_vk.physical_resolution.height,
                                ],
                                supported_transforms: properties_vk.supported_transforms.into(),
                                plane_reorder_possible: properties_vk.plane_reorder_possible
                                    != ash::vk::FALSE,
                                persistent_content: properties_vk.persistent_content
                                    != ash::vk::FALSE,
                            };

                            Display::from_handle(self.clone(), handle, properties)
                        })
                })
                .collect())
        }
    }

    /// Returns the properties of the display planes of the physical device.
    #[inline]
    pub fn display_plane_properties(
        self: &Arc<Self>,
    ) -> Result<Vec<DisplayPlaneProperties>, Validated<VulkanError>> {
        self.validate_display_plane_properties()?;

        unsafe { Ok(self.display_plane_properties_unchecked()?) }
    }

    fn validate_display_plane_properties(&self) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_display {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_display",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn display_plane_properties_unchecked(
        self: &Arc<Self>,
    ) -> Result<Vec<DisplayPlaneProperties>, VulkanError> {
        self.get_display_plane_properties_raw()?
            .iter()
            .map(|properties_raw| -> Result<_, VulkanError> {
                let &DisplayPlanePropertiesRaw {
                    current_display,
                    current_stack_index,
                } = properties_raw;

                let current_display = current_display
                    .map(|display_handle| {
                        self.display_properties
                            .get(&display_handle)
                            .map(Ok)
                            .unwrap_or_else(|| -> Result<_, VulkanError> {
                                self.display_properties_unchecked()?;
                                Ok(self.display_properties.get(&display_handle).unwrap())
                            })
                    })
                    .transpose()?;

                Ok(DisplayPlaneProperties {
                    current_display,
                    current_stack_index,
                })
            })
            .collect()
    }

    pub(crate) unsafe fn display_plane_properties_raw(
        &self,
    ) -> Result<Vec<DisplayPlanePropertiesRaw>, VulkanError> {
        {
            let read = self.display_plane_properties.read();

            if !read.is_empty() {
                return Ok(read.clone());
            }
        }

        self.get_display_plane_properties_raw()
    }

    unsafe fn get_display_plane_properties_raw(
        &self,
    ) -> Result<Vec<DisplayPlanePropertiesRaw>, VulkanError> {
        let fns = self.instance.fns();

        let properties_raw: Vec<_> = if self
            .instance
            .enabled_extensions()
            .khr_get_display_properties2
        {
            let properties_vk = unsafe {
                loop {
                    let mut count = 0;
                    (fns.khr_get_display_properties2
                        .get_physical_device_display_plane_properties2_khr)(
                        self.handle,
                        &mut count,
                        ptr::null_mut(),
                    )
                    .result()
                    .map_err(VulkanError::from)?;

                    let mut properties =
                        vec![ash::vk::DisplayPlaneProperties2KHR::default(); count as usize];
                    let result = (fns
                        .khr_get_display_properties2
                        .get_physical_device_display_plane_properties2_khr)(
                        self.handle,
                        &mut count,
                        properties.as_mut_ptr(),
                    );

                    match result {
                        ash::vk::Result::SUCCESS => {
                            properties.set_len(count as usize);
                            break properties;
                        }
                        ash::vk::Result::INCOMPLETE => (),
                        err => return Err(VulkanError::from(err)),
                    }
                }
            };

            properties_vk
                .into_iter()
                .map(|properties_vk| {
                    let properties_vk = &properties_vk.display_plane_properties;
                    DisplayPlanePropertiesRaw {
                        current_display: Some(properties_vk.current_display)
                            .filter(|&x| x != ash::vk::DisplayKHR::null()),
                        current_stack_index: properties_vk.current_stack_index,
                    }
                })
                .collect()
        } else {
            let properties_vk = unsafe {
                loop {
                    let mut count = 0;
                    (fns.khr_display
                        .get_physical_device_display_plane_properties_khr)(
                        self.handle,
                        &mut count,
                        ptr::null_mut(),
                    )
                    .result()
                    .map_err(VulkanError::from)?;

                    let mut properties = Vec::with_capacity(count as usize);
                    let result = (fns
                        .khr_display
                        .get_physical_device_display_plane_properties_khr)(
                        self.handle,
                        &mut count,
                        properties.as_mut_ptr(),
                    );

                    match result {
                        ash::vk::Result::SUCCESS => {
                            properties.set_len(count as usize);
                            break properties;
                        }
                        ash::vk::Result::INCOMPLETE => (),
                        err => return Err(VulkanError::from(err)),
                    }
                }
            };

            properties_vk
                .into_iter()
                .map(|properties_vk| DisplayPlanePropertiesRaw {
                    current_display: Some(properties_vk.current_display)
                        .filter(|&x| x != ash::vk::DisplayKHR::null()),
                    current_stack_index: properties_vk.current_stack_index,
                })
                .collect()
        };

        *self.display_plane_properties.write() = properties_raw.clone();
        Ok(properties_raw)
    }

    /// Returns the displays that are supported for the given plane index.
    ///
    /// The index must be less than the number of elements returned by
    /// [`display_plane_properties`](Self::display_plane_properties).
    #[inline]
    pub fn display_plane_supported_displays(
        self: &Arc<Self>,
        plane_index: u32,
    ) -> Result<Vec<Arc<Display>>, Validated<VulkanError>> {
        self.validate_display_plane_supported_displays(plane_index)?;

        unsafe { Ok(self.display_plane_supported_displays_unchecked(plane_index)?) }
    }

    fn validate_display_plane_supported_displays(
        &self,
        plane_index: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_display {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_display",
                )])]),
                ..Default::default()
            }));
        }

        let display_plane_properties_raw = unsafe {
            self.display_plane_properties_raw().map_err(|_err| {
                Box::new(ValidationError {
                    problem: "`PhysicalDevice::display_plane_properties` \
                        returned an error"
                        .into(),
                    ..Default::default()
                })
            })?
        };

        if plane_index as usize >= display_plane_properties_raw.len() {
            return Err(Box::new(ValidationError {
                problem: "`plane_index` is not less than the number of display planes on the \
                    physical device"
                    .into(),
                vuids: &["VUID-vkGetDisplayPlaneSupportedDisplaysKHR-planeIndex-01249"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn display_plane_supported_displays_unchecked(
        self: &Arc<Self>,
        plane_index: u32,
    ) -> Result<Vec<Arc<Display>>, VulkanError> {
        let fns = self.instance.fns();

        let displays_vk = unsafe {
            loop {
                let mut count = 0;
                (fns.khr_display.get_display_plane_supported_displays_khr)(
                    self.handle,
                    plane_index,
                    &mut count,
                    ptr::null_mut(),
                )
                .result()
                .map_err(VulkanError::from)?;

                let mut displays = Vec::with_capacity(count as usize);
                let result = (fns.khr_display.get_display_plane_supported_displays_khr)(
                    self.handle,
                    plane_index,
                    &mut count,
                    displays.as_mut_ptr(),
                );

                match result {
                    ash::vk::Result::SUCCESS => {
                        displays.set_len(count as usize);
                        break displays;
                    }
                    ash::vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err)),
                }
            }
        };

        let displays: Vec<_> = displays_vk
            .into_iter()
            .map(|display_vk| -> Result<_, VulkanError> {
                Ok(
                    if let Some(display) = self.display_properties.get(&display_vk) {
                        display
                    } else {
                        self.display_properties_unchecked()?;
                        self.display_properties.get(&display_vk).unwrap()
                    },
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(displays)
    }

    /// Retrieves the external memory properties supported for buffers with a given configuration.
    ///
    /// Instance API version must be at least 1.1, or the [`khr_external_memory_capabilities`]
    /// extension must be enabled on the instance.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// [`khr_external_memory_capabilities`]: crate::instance::InstanceExtensions::khr_external_memory_capabilities
    #[inline]
    pub fn external_buffer_properties(
        &self,
        info: ExternalBufferInfo,
    ) -> Result<ExternalBufferProperties, Box<ValidationError>> {
        self.validate_external_buffer_properties(&info)?;

        unsafe { Ok(self.external_buffer_properties_unchecked(info)) }
    }

    fn validate_external_buffer_properties(
        &self,
        info: &ExternalBufferInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_memory_capabilities)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::InstanceExtension(
                        "khr_external_memory_capabilities",
                    )]),
                ]),
                ..Default::default()
            }));
        }

        info.validate(self).map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn external_buffer_properties_unchecked(
        &self,
        info: ExternalBufferInfo,
    ) -> ExternalBufferProperties {
        self.external_buffer_properties.get_or_insert(info, |info| {
            /* Input */

            let &ExternalBufferInfo {
                flags,
                usage,
                handle_type,
                _ne: _,
            } = info;

            let external_buffer_info = ash::vk::PhysicalDeviceExternalBufferInfo {
                flags: flags.into(),
                usage: usage.into(),
                handle_type: handle_type.into(),
                ..Default::default()
            };

            /* Output */

            let mut external_buffer_properties = ash::vk::ExternalBufferProperties::default();

            /* Call */

            let fns = self.instance.fns();

            if self.instance.api_version() >= Version::V1_1 {
                (fns.v1_1.get_physical_device_external_buffer_properties)(
                    self.handle,
                    &external_buffer_info,
                    &mut external_buffer_properties,
                )
            } else {
                (fns.khr_external_memory_capabilities
                    .get_physical_device_external_buffer_properties_khr)(
                    self.handle,
                    &external_buffer_info,
                    &mut external_buffer_properties,
                );
            }

            ExternalBufferProperties {
                external_memory_properties: external_buffer_properties
                    .external_memory_properties
                    .into(),
            }
        })
    }

    /// Retrieves the external handle properties supported for fences with a given
    /// configuration.
    ///
    /// The instance API version must be at least 1.1, or the [`khr_external_fence_capabilities`]
    /// extension must be enabled on the instance.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// [`khr_external_fence_capabilities`]: crate::instance::InstanceExtensions::khr_external_fence_capabilities
    #[inline]
    pub fn external_fence_properties(
        &self,
        info: ExternalFenceInfo,
    ) -> Result<ExternalFenceProperties, Box<ValidationError>> {
        self.validate_external_fence_properties(&info)?;

        unsafe { Ok(self.external_fence_properties_unchecked(info)) }
    }

    fn validate_external_fence_properties(
        &self,
        info: &ExternalFenceInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_fence_capabilities)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::InstanceExtension(
                        "khr_external_fence_capabilities",
                    )]),
                ]),
                ..Default::default()
            }));
        }

        info.validate(self).map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn external_fence_properties_unchecked(
        &self,
        info: ExternalFenceInfo,
    ) -> ExternalFenceProperties {
        self.external_fence_properties.get_or_insert(info, |info| {
            /* Input */

            let &ExternalFenceInfo {
                handle_type,
                _ne: _,
            } = info;

            let external_fence_info = ash::vk::PhysicalDeviceExternalFenceInfo {
                handle_type: handle_type.into(),
                ..Default::default()
            };

            /* Output */

            let mut external_fence_properties = ash::vk::ExternalFenceProperties::default();

            /* Call */

            let fns = self.instance.fns();

            if self.instance.api_version() >= Version::V1_1 {
                (fns.v1_1.get_physical_device_external_fence_properties)(
                    self.handle,
                    &external_fence_info,
                    &mut external_fence_properties,
                )
            } else {
                (fns.khr_external_fence_capabilities
                    .get_physical_device_external_fence_properties_khr)(
                    self.handle,
                    &external_fence_info,
                    &mut external_fence_properties,
                );
            }

            ExternalFenceProperties {
                exportable: external_fence_properties
                    .external_fence_features
                    .intersects(ash::vk::ExternalFenceFeatureFlags::EXPORTABLE),
                importable: external_fence_properties
                    .external_fence_features
                    .intersects(ash::vk::ExternalFenceFeatureFlags::IMPORTABLE),
                export_from_imported_handle_types: external_fence_properties
                    .export_from_imported_handle_types
                    .into(),
                compatible_handle_types: external_fence_properties.compatible_handle_types.into(),
            }
        })
    }

    /// Retrieves the external handle properties supported for semaphores with a given
    /// configuration.
    ///
    /// The instance API version must be at least 1.1, or the
    /// [`khr_external_semaphore_capabilities`] extension must be enabled on the instance.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// [`khr_external_semaphore_capabilities`]: crate::instance::InstanceExtensions::khr_external_semaphore_capabilities
    #[inline]
    pub fn external_semaphore_properties(
        &self,
        info: ExternalSemaphoreInfo,
    ) -> Result<ExternalSemaphoreProperties, Box<ValidationError>> {
        self.validate_external_semaphore_properties(&info)?;

        unsafe { Ok(self.external_semaphore_properties_unchecked(info)) }
    }

    fn validate_external_semaphore_properties(
        &self,
        info: &ExternalSemaphoreInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_semaphore_capabilities)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::InstanceExtension(
                        "khr_external_semaphore_capabilities",
                    )]),
                ]),
                ..Default::default()
            }));
        }

        info.validate(self).map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn external_semaphore_properties_unchecked(
        &self,
        info: ExternalSemaphoreInfo,
    ) -> ExternalSemaphoreProperties {
        self.external_semaphore_properties
            .get_or_insert(info, |info| {
                /* Input */

                let &ExternalSemaphoreInfo {
                    handle_type,
                    semaphore_type,
                    initial_value,
                    _ne: _,
                } = info;

                let mut external_semaphore_info_vk = ash::vk::PhysicalDeviceExternalSemaphoreInfo {
                    handle_type: handle_type.into(),
                    ..Default::default()
                };
                let mut semaphore_type_create_info_vk = None;

                if semaphore_type != SemaphoreType::Binary {
                    let next =
                        semaphore_type_create_info_vk.insert(ash::vk::SemaphoreTypeCreateInfo {
                            semaphore_type: semaphore_type.into(),
                            initial_value,
                            ..Default::default()
                        });

                    next.p_next = external_semaphore_info_vk.p_next;
                    external_semaphore_info_vk.p_next = next as *const _ as *const _;
                }

                /* Output */

                let mut external_semaphore_properties =
                    ash::vk::ExternalSemaphoreProperties::default();

                /* Call */

                let fns = self.instance.fns();

                if self.instance.api_version() >= Version::V1_1 {
                    (fns.v1_1.get_physical_device_external_semaphore_properties)(
                        self.handle,
                        &external_semaphore_info_vk,
                        &mut external_semaphore_properties,
                    )
                } else {
                    (fns.khr_external_semaphore_capabilities
                        .get_physical_device_external_semaphore_properties_khr)(
                        self.handle,
                        &external_semaphore_info_vk,
                        &mut external_semaphore_properties,
                    );
                }

                ExternalSemaphoreProperties {
                    exportable: external_semaphore_properties
                        .external_semaphore_features
                        .intersects(ash::vk::ExternalSemaphoreFeatureFlags::EXPORTABLE),
                    importable: external_semaphore_properties
                        .external_semaphore_features
                        .intersects(ash::vk::ExternalSemaphoreFeatureFlags::IMPORTABLE),
                    export_from_imported_handle_types: external_semaphore_properties
                        .export_from_imported_handle_types
                        .into(),
                    compatible_handle_types: external_semaphore_properties
                        .compatible_handle_types
                        .into(),
                }
            })
    }

    /// Retrieves the properties of a format when used by this physical device.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    #[inline]
    pub fn format_properties(
        &self,
        format: Format,
    ) -> Result<FormatProperties, Box<ValidationError>> {
        self.validate_format_properties(format)?;

        unsafe { Ok(self.format_properties_unchecked(format)) }
    }

    fn validate_format_properties(&self, format: Format) -> Result<(), Box<ValidationError>> {
        format.validate_physical_device(self).map_err(|err| {
            err.add_context("format")
                .set_vuids(&["VUID-vkGetPhysicalDeviceFormatProperties2-format-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn format_properties_unchecked(&self, format: Format) -> FormatProperties {
        self.format_properties.get_or_insert(format, |&format| {
            let mut format_properties2_vk = ash::vk::FormatProperties2::default();
            let mut format_properties3_vk = None;
            let mut drm_format_modifier_properties_list_vk = None;
            let mut drm_format_modifier_properties_vk = Vec::new();
            let mut drm_format_modifier_properties_list2_vk = None;
            let mut drm_format_modifier_properties2_vk = Vec::new();

            if self.api_version() >= Version::V1_3
                || self.supported_extensions().khr_format_feature_flags2
            {
                let next = format_properties3_vk.insert(ash::vk::FormatProperties3KHR::default());
                next.p_next = format_properties2_vk.p_next;
                format_properties2_vk.p_next = next as *mut _ as *mut _;
            }

            if self.supported_extensions().ext_image_drm_format_modifier {
                let next = drm_format_modifier_properties_list_vk
                    .insert(ash::vk::DrmFormatModifierPropertiesListEXT::default());
                next.p_next = format_properties2_vk.p_next;
                format_properties2_vk.p_next = next as *mut _ as *mut _;

                if self.api_version() >= Version::V1_3
                    || self.supported_extensions().khr_format_feature_flags2
                {
                    let next = drm_format_modifier_properties_list2_vk
                        .insert(ash::vk::DrmFormatModifierPropertiesList2EXT::default());
                    next.p_next = format_properties2_vk.p_next;
                    format_properties2_vk.p_next = next as *mut _ as *mut _;
                }
            }

            let fns = self.instance.fns();

            // Get the number of DRM format modifier properties first.
            if let Some(drm_format_modifier_properties_list_vk) =
                &mut drm_format_modifier_properties_list_vk
            {
                if self.api_version() >= Version::V1_1 {
                    (fns.v1_1.get_physical_device_format_properties2)(
                        self.handle,
                        format.into(),
                        &mut format_properties2_vk,
                    );
                } else if self
                    .instance
                    .enabled_extensions()
                    .khr_get_physical_device_properties2
                {
                    (fns.khr_get_physical_device_properties2
                        .get_physical_device_format_properties2_khr)(
                        self.handle,
                        format.into(),
                        &mut format_properties2_vk,
                    );
                }

                drm_format_modifier_properties_vk = vec![
                        ash::vk::DrmFormatModifierPropertiesEXT::default();
                        drm_format_modifier_properties_list_vk.drm_format_modifier_count as usize
                    ];
                drm_format_modifier_properties_list_vk.p_drm_format_modifier_properties =
                    drm_format_modifier_properties_vk.as_mut_ptr();

                if let Some(drm_format_modifier_properties_list2_vk) =
                    &mut drm_format_modifier_properties_list2_vk
                {
                    drm_format_modifier_properties2_vk = vec![
                        ash::vk::DrmFormatModifierProperties2EXT::default();
                        drm_format_modifier_properties_list2_vk.drm_format_modifier_count as usize
                    ];
                    drm_format_modifier_properties_list2_vk.p_drm_format_modifier_properties =
                        drm_format_modifier_properties2_vk.as_mut_ptr();
                }
            }

            if self.api_version() >= Version::V1_1 {
                (fns.v1_1.get_physical_device_format_properties2)(
                    self.handle,
                    format.into(),
                    &mut format_properties2_vk,
                );
            } else if self
                .instance
                .enabled_extensions()
                .khr_get_physical_device_properties2
            {
                (fns.khr_get_physical_device_properties2
                    .get_physical_device_format_properties2_khr)(
                    self.handle,
                    format.into(),
                    &mut format_properties2_vk,
                );
            } else {
                (fns.v1_0.get_physical_device_format_properties)(
                    self.handle(),
                    format.into(),
                    &mut format_properties2_vk.format_properties,
                );
            }

            match format_properties3_vk {
                Some(format_properties3) => {
                    FormatProperties {
                        linear_tiling_features: format_properties3.linear_tiling_features.into(),
                        optimal_tiling_features: format_properties3.optimal_tiling_features.into(),
                        buffer_features: format_properties3.buffer_features.into(),
                        drm_format_modifier_properties: drm_format_modifier_properties_list2_vk
                            .map_or(Vec::new(), |list2_vk| {
                                drm_format_modifier_properties2_vk
                                    [..list2_vk.drm_format_modifier_count as usize]
                                    .iter()
                                    .map(|properties2_vk| DrmFormatModifierProperties {
                                        drm_format_modifier: properties2_vk.drm_format_modifier,
                                        drm_format_modifier_plane_count: properties2_vk
                                            .drm_format_modifier_plane_count,
                                        drm_format_modifier_tiling_features: properties2_vk
                                            .drm_format_modifier_tiling_features
                                            .into(),
                                    })
                                    .collect()
                            }),
                        _ne: crate::NonExhaustive(()),
                    }
                }
                None => {
                    FormatProperties {
                        linear_tiling_features: format_properties2_vk
                            .format_properties
                            .linear_tiling_features
                            .into(),
                        optimal_tiling_features: format_properties2_vk
                            .format_properties
                            .optimal_tiling_features
                            .into(),
                        buffer_features: format_properties2_vk
                            .format_properties
                            .buffer_features
                            .into(),
                        drm_format_modifier_properties: drm_format_modifier_properties_list_vk
                            .map_or(Vec::new(), |list_vk| {
                                drm_format_modifier_properties_vk
                                    [..list_vk.drm_format_modifier_count as usize]
                                    .iter()
                                    .map(|properties_vk| DrmFormatModifierProperties {
                                        drm_format_modifier: properties_vk.drm_format_modifier,
                                        drm_format_modifier_plane_count: properties_vk
                                            .drm_format_modifier_plane_count,
                                        drm_format_modifier_tiling_features: properties_vk
                                            .drm_format_modifier_tiling_features
                                            .into(),
                                    })
                                    .collect()
                            }),
                        _ne: crate::NonExhaustive(()),
                    }
                }
            }
        })
    }

    /// Returns the properties supported for images with a given image configuration.
    ///
    /// `Some` is returned if the configuration is supported, `None` if it is not.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// # Panics
    ///
    /// - Panics if `image_format_info.format` is `None`.
    #[inline]
    pub fn image_format_properties(
        &self,
        image_format_info: ImageFormatInfo,
    ) -> Result<Option<ImageFormatProperties>, Validated<VulkanError>> {
        self.validate_image_format_properties(&image_format_info)?;

        unsafe { Ok(self.image_format_properties_unchecked(image_format_info)?) }
    }

    fn validate_image_format_properties(
        &self,
        image_format_info: &ImageFormatInfo,
    ) -> Result<(), Box<ValidationError>> {
        image_format_info
            .validate(self)
            .map_err(|err| err.add_context("image_format_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn image_format_properties_unchecked(
        &self,
        image_format_info: ImageFormatInfo,
    ) -> Result<Option<ImageFormatProperties>, VulkanError> {
        self.image_format_properties
            .get_or_try_insert(image_format_info, |image_format_info| {
                /* Input */
                let &ImageFormatInfo {
                    flags,
                    format,
                    image_type,
                    tiling,
                    usage,
                    stencil_usage,
                    external_memory_handle_type,
                    image_view_type,
                    ref drm_format_modifier_info,
                    ref view_formats,
                    _ne: _,
                } = image_format_info;

                let mut info2_vk = ash::vk::PhysicalDeviceImageFormatInfo2 {
                    format: format.into(),
                    ty: image_type.into(),
                    tiling: tiling.into(),
                    usage: usage.into(),
                    flags: flags.into(),
                    ..Default::default()
                };
                let mut drm_format_modifier_info_vk = None;
                let mut external_info_vk = None;
                let mut format_list_info_vk = None;
                let format_list_view_formats_vk: Vec<_>;
                let mut image_view_info_vk = None;
                let mut stencil_usage_info_vk = None;

                if let Some(drm_format_modifier_info) = drm_format_modifier_info {
                    let &ImageDrmFormatModifierInfo {
                        drm_format_modifier,
                        ref sharing,
                        _ne: _,
                    } = drm_format_modifier_info;

                    let (sharing_mode, queue_family_index_count, p_queue_family_indices) =
                        match sharing {
                            Sharing::Exclusive => (ash::vk::SharingMode::EXCLUSIVE, 0, &[] as _),
                            Sharing::Concurrent(queue_family_indices) => (
                                ash::vk::SharingMode::CONCURRENT,
                                queue_family_indices.len() as u32,
                                queue_family_indices.as_ptr(),
                            ),
                        };

                    let next = drm_format_modifier_info_vk.insert(
                        ash::vk::PhysicalDeviceImageDrmFormatModifierInfoEXT {
                            drm_format_modifier,
                            sharing_mode,
                            queue_family_index_count,
                            p_queue_family_indices,
                            ..Default::default()
                        },
                    );

                    next.p_next = info2_vk.p_next;
                    info2_vk.p_next = next as *const _ as *const _;
                }

                if let Some(handle_type) = external_memory_handle_type {
                    let next =
                        external_info_vk.insert(ash::vk::PhysicalDeviceExternalImageFormatInfo {
                            handle_type: handle_type.into(),
                            ..Default::default()
                        });

                    next.p_next = info2_vk.p_next;
                    info2_vk.p_next = next as *const _ as *const _;
                }

                if !view_formats.is_empty() {
                    format_list_view_formats_vk = view_formats
                        .iter()
                        .copied()
                        .map(ash::vk::Format::from)
                        .collect();

                    let next = format_list_info_vk.insert(ash::vk::ImageFormatListCreateInfo {
                        view_format_count: format_list_view_formats_vk.len() as u32,
                        p_view_formats: format_list_view_formats_vk.as_ptr(),
                        ..Default::default()
                    });

                    next.p_next = info2_vk.p_next;
                    info2_vk.p_next = next as *const _ as *const _;
                }

                if let Some(image_view_type) = image_view_type {
                    let next = image_view_info_vk.insert(
                        ash::vk::PhysicalDeviceImageViewImageFormatInfoEXT {
                            image_view_type: image_view_type.into(),
                            ..Default::default()
                        },
                    );

                    next.p_next = info2_vk.p_next as *mut _;
                    info2_vk.p_next = next as *const _ as *const _;
                }

                if let Some(stencil_usage) = stencil_usage {
                    let next = stencil_usage_info_vk.insert(ash::vk::ImageStencilUsageCreateInfo {
                        stencil_usage: stencil_usage.into(),
                        ..Default::default()
                    });

                    next.p_next = info2_vk.p_next as *mut _;
                    info2_vk.p_next = next as *const _ as *const _;
                }

                /* Output */

                let mut properties2_vk = ash::vk::ImageFormatProperties2::default();
                let mut external_properties_vk = None;
                let mut filter_cubic_image_view_properties_vk = None;

                if external_info_vk.is_some() {
                    let next = external_properties_vk
                        .insert(ash::vk::ExternalImageFormatProperties::default());

                    next.p_next = properties2_vk.p_next;
                    properties2_vk.p_next = next as *mut _ as *mut _;
                }

                if image_view_info_vk.is_some() {
                    let next = filter_cubic_image_view_properties_vk
                        .insert(ash::vk::FilterCubicImageViewImageFormatPropertiesEXT::default());

                    next.p_next = properties2_vk.p_next;
                    properties2_vk.p_next = next as *mut _ as *mut _;
                }

                let result = {
                    let fns = self.instance.fns();

                    if self.api_version() >= Version::V1_1 {
                        (fns.v1_1.get_physical_device_image_format_properties2)(
                            self.handle,
                            &info2_vk,
                            &mut properties2_vk,
                        )
                    } else if self
                        .instance
                        .enabled_extensions()
                        .khr_get_physical_device_properties2
                    {
                        (fns.khr_get_physical_device_properties2
                            .get_physical_device_image_format_properties2_khr)(
                            self.handle,
                            &info2_vk,
                            &mut properties2_vk,
                        )
                    } else {
                        // Can't query this, return unsupported
                        if !info2_vk.p_next.is_null() {
                            return Ok(None);
                        }
                        if let Some(ExternalMemoryHandleType::DmaBuf) = external_memory_handle_type
                        {
                            // VUID-vkGetPhysicalDeviceImageFormatProperties-tiling-02248
                            // VUID-VkPhysicalDeviceImageFormatInfo2-tiling-02249
                            return Ok(None);
                        }

                        (fns.v1_0.get_physical_device_image_format_properties)(
                            self.handle,
                            info2_vk.format,
                            info2_vk.ty,
                            info2_vk.tiling,
                            info2_vk.usage,
                            info2_vk.flags,
                            &mut properties2_vk.image_format_properties,
                        )
                    }
                    .result()
                    .map_err(VulkanError::from)
                };

                Ok(match result {
                    Ok(_) => Some(ImageFormatProperties {
                        external_memory_properties: external_properties_vk
                            .map(|properties| properties.external_memory_properties.into())
                            .unwrap_or_default(),
                        filter_cubic: filter_cubic_image_view_properties_vk
                            .map_or(false, |properties| {
                                properties.filter_cubic != ash::vk::FALSE
                            }),
                        filter_cubic_minmax: filter_cubic_image_view_properties_vk
                            .map_or(false, |properties| {
                                properties.filter_cubic_minmax != ash::vk::FALSE
                            }),
                        ..properties2_vk.image_format_properties.into()
                    }),
                    Err(VulkanError::FormatNotSupported) => None,
                    Err(err) => return Err(err),
                })
            })
    }

    /// Queries whether the physical device supports presenting to QNX Screen surfaces from queues
    /// of the given queue family.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid QNX Screen `_screen_window` handle.
    pub unsafe fn qnx_screen_presentation_support<W>(
        &self,
        queue_family_index: u32,
        window: *const W,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_qnx_screen_presentation_support(queue_family_index, window)?;

        Ok(self.qnx_screen_presentation_support_unchecked(queue_family_index, window))
    }

    fn validate_qnx_screen_presentation_support<W>(
        &self,
        queue_family_index: u32,
        _window: *const W,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().qnx_screen_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "qnx_screen_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceScreenPresentationSupportQNX-queueFamilyIndex-04743",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceScreenPresentationSupportQNX-window-parameter
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn qnx_screen_presentation_support_unchecked<W>(
        &self,
        queue_family_index: u32,
        window: *const W,
    ) -> bool {
        let fns = self.instance.fns();
        (fns.qnx_screen_surface
            .get_physical_device_screen_presentation_support_qnx)(
            self.handle,
            queue_family_index,
            window as *mut _,
        ) != 0
    }

    /// Returns the properties of sparse images with a given image configuration.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// # Panics
    ///
    /// - Panics if `format_info.format` is `None`.
    #[inline]
    pub fn sparse_image_format_properties(
        &self,
        format_info: SparseImageFormatInfo,
    ) -> Result<Vec<SparseImageFormatProperties>, Box<ValidationError>> {
        self.validate_sparse_image_format_properties(&format_info)?;

        unsafe { Ok(self.sparse_image_format_properties_unchecked(format_info)) }
    }

    fn validate_sparse_image_format_properties(
        &self,
        format_info: &SparseImageFormatInfo,
    ) -> Result<(), Box<ValidationError>> {
        format_info
            .validate(self)
            .map_err(|err| err.add_context("format_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn sparse_image_format_properties_unchecked(
        &self,
        format_info: SparseImageFormatInfo,
    ) -> Vec<SparseImageFormatProperties> {
        self.sparse_image_format_properties
            .get_or_insert(format_info, |format_info| {
                let &SparseImageFormatInfo {
                    format,
                    image_type,
                    samples,
                    usage,
                    tiling,
                    _ne: _,
                } = format_info;

                let format_info2 = ash::vk::PhysicalDeviceSparseImageFormatInfo2 {
                    format: format.into(),
                    ty: image_type.into(),
                    samples: samples.into(),
                    usage: usage.into(),
                    tiling: tiling.into(),
                    ..Default::default()
                };

                let fns = self.instance.fns();

                if self.api_version() >= Version::V1_1
                    || self
                        .instance
                        .enabled_extensions()
                        .khr_get_physical_device_properties2
                {
                    let mut count = 0;

                    if self.api_version() >= Version::V1_1 {
                        (fns.v1_1.get_physical_device_sparse_image_format_properties2)(
                            self.handle,
                            &format_info2,
                            &mut count,
                            ptr::null_mut(),
                        );
                    } else {
                        (fns.khr_get_physical_device_properties2
                            .get_physical_device_sparse_image_format_properties2_khr)(
                            self.handle,
                            &format_info2,
                            &mut count,
                            ptr::null_mut(),
                        );
                    }

                    let mut sparse_image_format_properties2 =
                        vec![ash::vk::SparseImageFormatProperties2::default(); count as usize];

                    if self.api_version() >= Version::V1_1 {
                        (fns.v1_1.get_physical_device_sparse_image_format_properties2)(
                            self.handle,
                            &format_info2,
                            &mut count,
                            sparse_image_format_properties2.as_mut_ptr(),
                        );
                    } else {
                        (fns.khr_get_physical_device_properties2
                            .get_physical_device_sparse_image_format_properties2_khr)(
                            self.handle,
                            &format_info2,
                            &mut count,
                            sparse_image_format_properties2.as_mut_ptr(),
                        );
                    }

                    sparse_image_format_properties2.set_len(count as usize);

                    sparse_image_format_properties2
                        .into_iter()
                        .map(
                            |sparse_image_format_properties2| SparseImageFormatProperties {
                                aspects: sparse_image_format_properties2
                                    .properties
                                    .aspect_mask
                                    .into(),
                                image_granularity: [
                                    sparse_image_format_properties2
                                        .properties
                                        .image_granularity
                                        .width,
                                    sparse_image_format_properties2
                                        .properties
                                        .image_granularity
                                        .height,
                                    sparse_image_format_properties2
                                        .properties
                                        .image_granularity
                                        .depth,
                                ],
                                flags: sparse_image_format_properties2.properties.flags.into(),
                            },
                        )
                        .collect()
                } else {
                    let mut count = 0;

                    (fns.v1_0.get_physical_device_sparse_image_format_properties)(
                        self.handle,
                        format_info2.format,
                        format_info2.ty,
                        format_info2.samples,
                        format_info2.usage,
                        format_info2.tiling,
                        &mut count,
                        ptr::null_mut(),
                    );

                    let mut sparse_image_format_properties =
                        vec![ash::vk::SparseImageFormatProperties::default(); count as usize];

                    (fns.v1_0.get_physical_device_sparse_image_format_properties)(
                        self.handle,
                        format_info2.format,
                        format_info2.ty,
                        format_info2.samples,
                        format_info2.usage,
                        format_info2.tiling,
                        &mut count,
                        sparse_image_format_properties.as_mut_ptr(),
                    );

                    sparse_image_format_properties.set_len(count as usize);

                    sparse_image_format_properties
                        .into_iter()
                        .map(
                            |sparse_image_format_properties| SparseImageFormatProperties {
                                aspects: sparse_image_format_properties.aspect_mask.into(),
                                image_granularity: [
                                    sparse_image_format_properties.image_granularity.width,
                                    sparse_image_format_properties.image_granularity.height,
                                    sparse_image_format_properties.image_granularity.depth,
                                ],
                                flags: sparse_image_format_properties.flags.into(),
                            },
                        )
                        .collect()
                }
            })
    }

    /// Returns the capabilities that are supported by the physical device for the given surface.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// # Panics
    ///
    /// - Panics if the physical device and the surface don't belong to the same instance.
    pub fn surface_capabilities(
        &self,
        surface: &Surface,
        surface_info: SurfaceInfo,
    ) -> Result<SurfaceCapabilities, Validated<VulkanError>> {
        self.validate_surface_capabilities(surface, &surface_info)?;

        unsafe { Ok(self.surface_capabilities_unchecked(surface, surface_info)?) }
    }

    fn validate_surface_capabilities(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !(self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
            || self.instance.enabled_extensions().khr_surface)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::InstanceExtension("khr_get_surface_capabilities2")]),
                    RequiresAllOf(&[Requires::InstanceExtension("khr_surface")]),
                ]),
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceSurfaceCapabilities2KHR-commonparent
        assert_eq!(self.instance(), surface.instance());

        if !(0..self.queue_family_properties.len() as u32).any(|index| unsafe {
            self.surface_support_unchecked(index, surface)
                .unwrap_or_default()
        }) {
            return Err(Box::new(ValidationError {
                context: "surface".into(),
                problem: "is not supported by the physical device".into(),
                vuids: &["VUID-vkGetPhysicalDeviceSurfaceCapabilities2KHR-pSurfaceInfo-06210"],
                ..Default::default()
            }));
        }

        surface_info
            .validate(self)
            .map_err(|err| err.add_context("surface_info"))?;

        let &SurfaceInfo {
            present_mode,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        if let Some(present_mode) = present_mode {
            let mut present_modes = unsafe {
                self.surface_present_modes_unchecked(
                    surface,
                    SurfaceInfo {
                        present_mode: None,
                        ..surface_info.clone()
                    },
                )
                .map_err(|_err| {
                    Box::new(ValidationError {
                        problem: "`PhysicalDevice::surface_present_modes` \
                                returned an error"
                            .into(),
                        ..Default::default()
                    })
                })?
            };

            if !present_modes.any(|mode| mode == present_mode) {
                return Err(Box::new(ValidationError {
                    problem: "`surface_info.present_mode` is not supported for `surface`".into(),
                    vuids: &["VUID-VkSurfacePresentModeEXT-presentMode-07780"],
                    ..Default::default()
                }));
            }
        }

        match (
            surface.api() == SurfaceApi::Win32
                && full_screen_exclusive == FullScreenExclusive::ApplicationControlled,
            win32_monitor.is_some(),
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "`surface` is a Win32 surface, and \
                        `surface_info.full_screen_exclusive` is \
                        `FullScreenExclusive::ApplicationControlled`, but \
                        `surface_info.win32_monitor` is `None`"
                        .into(),
                    vuids: &["VUID-VkPhysicalDeviceSurfaceInfo2KHR-pNext-02672"],
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "`surface` is not a Win32 surface, or \
                        `surface_info.full_screen_exclusive` is not \
                        `FullScreenExclusive::ApplicationControlled`, but \
                        `surface_info.win32_monitor` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (true, true) | (false, false) => (),
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_capabilities_unchecked(
        &self,
        surface: &Surface,
        surface_info: SurfaceInfo,
    ) -> Result<SurfaceCapabilities, VulkanError> {
        /* Input */

        let SurfaceInfo {
            present_mode,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        let mut info_vk = ash::vk::PhysicalDeviceSurfaceInfo2KHR {
            surface: surface.handle(),
            ..Default::default()
        };
        let mut present_mode_vk = None;
        let mut full_screen_exclusive_info_vk = None;
        let mut full_screen_exclusive_win32_info_vk = None;

        if let Some(present_mode) = present_mode {
            let next = present_mode_vk.insert(ash::vk::SurfacePresentModeEXT {
                present_mode: present_mode.into(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next as *mut _;
            info_vk.p_next = next as *const _ as *const _;
        }

        if full_screen_exclusive != FullScreenExclusive::Default {
            let next =
                full_screen_exclusive_info_vk.insert(ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                    full_screen_exclusive: full_screen_exclusive.into(),
                    ..Default::default()
                });

            next.p_next = info_vk.p_next as *mut _;
            info_vk.p_next = next as *const _ as *const _;
        }

        if let Some(win32_monitor) = win32_monitor {
            let next = full_screen_exclusive_win32_info_vk.insert(
                ash::vk::SurfaceFullScreenExclusiveWin32InfoEXT {
                    hmonitor: win32_monitor.0,
                    ..Default::default()
                },
            );

            next.p_next = info_vk.p_next as *mut _;
            info_vk.p_next = next as *const _ as *const _;
        }

        /* Output */

        let mut capabilities_vk = ash::vk::SurfaceCapabilities2KHR::default();
        let mut capabilities_full_screen_exclusive_vk = None;
        let mut capabilities_present_modes_vk =
            [ash::vk::PresentModeKHR::default(); PresentMode::COUNT];
        let mut capabilities_present_mode_compatibility_vk = None;
        let mut capabilities_present_scaling_vk = None;
        let mut capabilities_protected_vk = None;

        if full_screen_exclusive_info_vk.is_some() {
            let next = capabilities_full_screen_exclusive_vk
                .insert(ash::vk::SurfaceCapabilitiesFullScreenExclusiveEXT::default());

            next.p_next = capabilities_vk.p_next as *mut _;
            capabilities_vk.p_next = next as *mut _ as *mut _;
        }

        if present_mode.is_some() {
            {
                let next = capabilities_present_mode_compatibility_vk.insert(
                    ash::vk::SurfacePresentModeCompatibilityEXT {
                        present_mode_count: capabilities_present_modes_vk.len() as u32,
                        p_present_modes: capabilities_present_modes_vk.as_mut_ptr(),
                        ..Default::default()
                    },
                );

                next.p_next = capabilities_vk.p_next as *mut _;
                capabilities_vk.p_next = next as *mut _ as *mut _;
            }

            {
                let next = capabilities_present_scaling_vk
                    .insert(ash::vk::SurfacePresentScalingCapabilitiesEXT::default());

                next.p_next = capabilities_vk.p_next as *mut _;
                capabilities_vk.p_next = next as *mut _ as *mut _;
            }
        }

        if self
            .instance
            .enabled_extensions()
            .khr_surface_protected_capabilities
        {
            let next = capabilities_protected_vk
                .insert(ash::vk::SurfaceProtectedCapabilitiesKHR::default());

            next.p_next = capabilities_vk.p_next as *mut _;
            capabilities_vk.p_next = next as *mut _ as *mut _;
        }

        let fns = self.instance.fns();

        if self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
        {
            (fns.khr_get_surface_capabilities2
                .get_physical_device_surface_capabilities2_khr)(
                self.handle(),
                &info_vk,
                &mut capabilities_vk,
            )
            .result()
            .map_err(VulkanError::from)?;
        } else {
            (fns.khr_surface.get_physical_device_surface_capabilities_khr)(
                self.handle(),
                info_vk.surface,
                &mut capabilities_vk.surface_capabilities,
            )
            .result()
            .map_err(VulkanError::from)?;
        };

        Ok(SurfaceCapabilities {
            min_image_count: capabilities_vk.surface_capabilities.min_image_count,
            max_image_count: (capabilities_vk.surface_capabilities.max_image_count != 0)
                .then_some(capabilities_vk.surface_capabilities.max_image_count),
            current_extent: (!matches!(
                capabilities_vk.surface_capabilities.current_extent,
                ash::vk::Extent2D {
                    width: u32::MAX,
                    height: u32::MAX
                }
            ))
            .then_some([
                capabilities_vk.surface_capabilities.current_extent.width,
                capabilities_vk.surface_capabilities.current_extent.height,
            ]),
            min_image_extent: [
                capabilities_vk.surface_capabilities.min_image_extent.width,
                capabilities_vk.surface_capabilities.min_image_extent.height,
            ],
            max_image_extent: [
                capabilities_vk.surface_capabilities.max_image_extent.width,
                capabilities_vk.surface_capabilities.max_image_extent.height,
            ],
            max_image_array_layers: capabilities_vk.surface_capabilities.max_image_array_layers,
            supported_transforms: capabilities_vk
                .surface_capabilities
                .supported_transforms
                .into(),

            current_transform: SurfaceTransforms::from(
                capabilities_vk.surface_capabilities.current_transform,
            )
            .into_iter()
            .next()
            .unwrap(), // TODO:
            supported_composite_alpha: capabilities_vk
                .surface_capabilities
                .supported_composite_alpha
                .into(),
            supported_usage_flags: ImageUsage::from(
                capabilities_vk.surface_capabilities.supported_usage_flags,
            ),

            compatible_present_modes: capabilities_present_mode_compatibility_vk.map_or_else(
                Default::default,
                |capabilities_present_mode_compatibility_vk| {
                    capabilities_present_modes_vk
                        [..capabilities_present_mode_compatibility_vk.present_mode_count as usize]
                        .iter()
                        .copied()
                        .map(PresentMode::try_from)
                        .filter_map(Result::ok)
                        .collect()
                },
            ),

            supported_present_scaling: capabilities_present_scaling_vk
                .as_ref()
                .map_or_else(Default::default, |c| c.supported_present_scaling.into()),
            supported_present_gravity: capabilities_present_scaling_vk.as_ref().map_or_else(
                Default::default,
                |c| {
                    [
                        c.supported_present_gravity_x.into(),
                        c.supported_present_gravity_y.into(),
                    ]
                },
            ),
            min_scaled_image_extent: capabilities_present_scaling_vk.as_ref().map_or(
                Some([
                    capabilities_vk.surface_capabilities.min_image_extent.width,
                    capabilities_vk.surface_capabilities.min_image_extent.height,
                ]),
                |c| {
                    (!matches!(
                        c.min_scaled_image_extent,
                        ash::vk::Extent2D {
                            width: u32::MAX,
                            height: u32::MAX,
                        }
                    ))
                    .then_some([
                        c.min_scaled_image_extent.width,
                        c.min_scaled_image_extent.height,
                    ])
                },
            ),
            max_scaled_image_extent: capabilities_present_scaling_vk.as_ref().map_or(
                Some([
                    capabilities_vk.surface_capabilities.max_image_extent.width,
                    capabilities_vk.surface_capabilities.max_image_extent.height,
                ]),
                |c| {
                    (!matches!(
                        c.max_scaled_image_extent,
                        ash::vk::Extent2D {
                            width: u32::MAX,
                            height: u32::MAX,
                        }
                    ))
                    .then_some([
                        c.max_scaled_image_extent.width,
                        c.max_scaled_image_extent.height,
                    ])
                },
            ),

            supports_protected: capabilities_protected_vk
                .map_or(false, |c| c.supports_protected != 0),

            full_screen_exclusive_supported: capabilities_full_screen_exclusive_vk
                .map_or(false, |c| c.full_screen_exclusive_supported != 0),
        })
    }

    /// Returns the combinations of format and color space that are supported by the physical device
    /// for the given surface.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// # Panics
    ///
    /// - Panics if the physical device and the surface don't belong to the same instance.
    pub fn surface_formats(
        &self,
        surface: &Surface,
        surface_info: SurfaceInfo,
    ) -> Result<Vec<(Format, ColorSpace)>, Validated<VulkanError>> {
        self.validate_surface_formats(surface, &surface_info)?;

        unsafe { Ok(self.surface_formats_unchecked(surface, surface_info)?) }
    }

    fn validate_surface_formats(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !(self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
            || self.instance.enabled_extensions().khr_surface)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::InstanceExtension("khr_get_surface_capabilities2")]),
                    RequiresAllOf(&[Requires::InstanceExtension("khr_surface")]),
                ]),
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceSurfaceFormats2KHR-commonparent
        assert_eq!(self.instance(), surface.instance());

        if !(0..self.queue_family_properties.len() as u32).any(|index| unsafe {
            self.surface_support_unchecked(index, surface)
                .unwrap_or_default()
        }) {
            return Err(Box::new(ValidationError {
                context: "surface".into(),
                problem: "is not supported by the physical device".into(),
                vuids: &["VUID-vkGetPhysicalDeviceSurfaceFormats2KHR-pSurfaceInfo-06522"],
                ..Default::default()
            }));
        }

        surface_info
            .validate(self)
            .map_err(|err| err.add_context("surface_info"))?;

        let &SurfaceInfo {
            present_mode,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        if let Some(present_mode) = present_mode {
            let mut present_modes = unsafe {
                self.surface_present_modes_unchecked(
                    surface,
                    SurfaceInfo {
                        present_mode: None,
                        ..surface_info.clone()
                    },
                )
                .map_err(|_err| {
                    Box::new(ValidationError {
                        problem: "`PhysicalDevice::surface_present_modes` \
                                returned an error"
                            .into(),
                        ..Default::default()
                    })
                })?
            };

            if !present_modes.any(|mode| mode == present_mode) {
                return Err(Box::new(ValidationError {
                    problem: "`surface_info.present_mode` is not supported for `surface`".into(),
                    vuids: &["VUID-VkSurfacePresentModeEXT-presentMode-07780"],
                    ..Default::default()
                }));
            }
        }

        if !self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
        {
            if full_screen_exclusive != FullScreenExclusive::Default {
                return Err(Box::new(ValidationError {
                    context: "surface_info.full_screen_exclusive".into(),
                    problem: "is not `FullScreenExclusive::Default`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                        Requires::InstanceExtension("khr_get_surface_capabilities2"),
                    ])]),
                    ..Default::default()
                }));
            }

            if win32_monitor.is_some() {
                return Err(Box::new(ValidationError {
                    context: "surface_info.win32_monitor".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                        Requires::InstanceExtension("khr_get_surface_capabilities2"),
                    ])]),
                    ..Default::default()
                }));
            }
        }

        match (
            surface.api() == SurfaceApi::Win32
                && full_screen_exclusive == FullScreenExclusive::ApplicationControlled,
            win32_monitor.is_some(),
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "`surface` is a Win32 surface, and \
                        `surface_info.full_screen_exclusive` is \
                        `FullScreenExclusive::ApplicationControlled`, but \
                        `surface_info.win32_monitor` is `None`"
                        .into(),
                    vuids: &["VUID-VkPhysicalDeviceSurfaceInfo2KHR-pNext-02672"],
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "`surface` is not a Win32 surface, or \
                        `surface_info.full_screen_exclusive` is not \
                        `FullScreenExclusive::ApplicationControlled`, but \
                        `surface_info.win32_monitor` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (true, true) | (false, false) => (),
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_formats_unchecked(
        &self,
        surface: &Surface,
        surface_info: SurfaceInfo,
    ) -> Result<Vec<(Format, ColorSpace)>, VulkanError> {
        surface.surface_formats.get_or_try_insert(
            (self.handle, surface_info),
            |(_, surface_info)| {
                let &SurfaceInfo {
                    present_mode,
                    full_screen_exclusive,
                    win32_monitor,
                    _ne: _,
                } = surface_info;

                let mut info_vk = ash::vk::PhysicalDeviceSurfaceInfo2KHR {
                    surface: surface.handle(),
                    ..Default::default()
                };
                let mut present_mode_vk = None;
                let mut full_screen_exclusive_info_vk = None;
                let mut full_screen_exclusive_win32_info_vk = None;

                if let Some(present_mode) = present_mode {
                    let next = present_mode_vk.insert(ash::vk::SurfacePresentModeEXT {
                        present_mode: present_mode.into(),
                        ..Default::default()
                    });

                    next.p_next = info_vk.p_next as *mut _;
                    info_vk.p_next = next as *const _ as *const _;
                }

                if full_screen_exclusive != FullScreenExclusive::Default {
                    let next = full_screen_exclusive_info_vk.insert(
                        ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                            full_screen_exclusive: full_screen_exclusive.into(),
                            ..Default::default()
                        },
                    );

                    next.p_next = info_vk.p_next as *mut _;
                    info_vk.p_next = next as *const _ as *const _;
                }

                if let Some(win32_monitor) = win32_monitor {
                    let next = full_screen_exclusive_win32_info_vk.insert(
                        ash::vk::SurfaceFullScreenExclusiveWin32InfoEXT {
                            hmonitor: win32_monitor.0,
                            ..Default::default()
                        },
                    );

                    next.p_next = info_vk.p_next as *mut _;
                    info_vk.p_next = next as *const _ as *const _;
                }

                let fns = self.instance.fns();

                if self
                    .instance
                    .enabled_extensions()
                    .khr_get_surface_capabilities2
                {
                    let surface_format2s_vk = loop {
                        let mut count = 0;
                        (fns.khr_get_surface_capabilities2
                            .get_physical_device_surface_formats2_khr)(
                            self.handle(),
                            &info_vk,
                            &mut count,
                            ptr::null_mut(),
                        )
                        .result()
                        .map_err(VulkanError::from)?;

                        let mut surface_format2s_vk =
                            vec![ash::vk::SurfaceFormat2KHR::default(); count as usize];
                        let result = (fns
                            .khr_get_surface_capabilities2
                            .get_physical_device_surface_formats2_khr)(
                            self.handle(),
                            &info_vk,
                            &mut count,
                            surface_format2s_vk.as_mut_ptr(),
                        );

                        match result {
                            ash::vk::Result::SUCCESS => {
                                surface_format2s_vk.set_len(count as usize);
                                break surface_format2s_vk;
                            }
                            ash::vk::Result::INCOMPLETE => (),
                            err => return Err(VulkanError::from(err)),
                        }
                    };

                    Ok(surface_format2s_vk
                        .into_iter()
                        .filter_map(|surface_format2| {
                            (surface_format2.surface_format.format.try_into().ok())
                                .zip(surface_format2.surface_format.color_space.try_into().ok())
                        })
                        .collect())
                } else {
                    let surface_formats = loop {
                        let mut count = 0;
                        (fns.khr_surface.get_physical_device_surface_formats_khr)(
                            self.handle(),
                            surface.handle(),
                            &mut count,
                            ptr::null_mut(),
                        )
                        .result()
                        .map_err(VulkanError::from)?;

                        let mut surface_formats = Vec::with_capacity(count as usize);
                        let result = (fns.khr_surface.get_physical_device_surface_formats_khr)(
                            self.handle(),
                            surface.handle(),
                            &mut count,
                            surface_formats.as_mut_ptr(),
                        );

                        match result {
                            ash::vk::Result::SUCCESS => {
                                surface_formats.set_len(count as usize);
                                break surface_formats;
                            }
                            ash::vk::Result::INCOMPLETE => (),
                            err => return Err(VulkanError::from(err)),
                        }
                    };

                    Ok(surface_formats
                        .into_iter()
                        .filter_map(|surface_format| {
                            (surface_format.format.try_into().ok())
                                .zip(surface_format.color_space.try_into().ok())
                        })
                        .collect())
                }
            },
        )
    }

    /// Returns the present modes that are supported by the physical device for the given surface.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    ///
    /// # Panics
    ///
    /// - Panics if the physical device and the surface don't belong to the same instance.
    pub fn surface_present_modes(
        &self,
        surface: &Surface,
        surface_info: SurfaceInfo,
    ) -> Result<impl Iterator<Item = PresentMode>, Validated<VulkanError>> {
        self.validate_surface_present_modes(surface, &surface_info)?;

        unsafe { Ok(self.surface_present_modes_unchecked(surface, surface_info)?) }
    }

    fn validate_surface_present_modes(
        &self,
        surface: &Surface,
        surface_info: &SurfaceInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceSurfacePresentModesKHR-commonparent
        assert_eq!(self.instance(), surface.instance());

        if !(0..self.queue_family_properties.len() as u32).any(|index| unsafe {
            self.surface_support_unchecked(index, surface)
                .unwrap_or_default()
        }) {
            return Err(Box::new(ValidationError {
                context: "surface".into(),
                problem: "is not supported by the physical device".into(),
                vuids: &["VUID-vkGetPhysicalDeviceSurfacePresentModes2EXT-pSurfaceInfo-06522"],
                ..Default::default()
            }));
        }

        surface_info
            .validate(self)
            .map_err(|err| err.add_context("surface_info"))?;

        let &SurfaceInfo {
            present_mode,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        // We can't validate supported present modes while querying for supported present modes,
        // so just demand that it's `None` here.
        if present_mode.is_some() {
            return Err(Box::new(ValidationError {
                context: "surface_info.present_mode".into(),
                problem: "is `Some`".into(),
                ..Default::default()
            }));
        }

        if !self.supported_extensions().ext_full_screen_exclusive {
            if full_screen_exclusive != FullScreenExclusive::Default {
                return Err(Box::new(ValidationError {
                    context: "surface_info.full_screen_exclusive".into(),
                    problem: "is not `FullScreenExclusive::Default`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_full_screen_exclusive",
                    )])]),
                    ..Default::default()
                }));
            }

            if win32_monitor.is_some() {
                return Err(Box::new(ValidationError {
                    context: "surface_info.win32_monitor".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_full_screen_exclusive",
                    )])]),
                    ..Default::default()
                }));
            }
        }

        match (
            surface.api() == SurfaceApi::Win32
                && full_screen_exclusive == FullScreenExclusive::ApplicationControlled,
            win32_monitor.is_some(),
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "`surface` is a Win32 surface, and \
                        `surface_info.full_screen_exclusive` is \
                        `FullScreenExclusive::ApplicationControlled`, but \
                        `surface_info.win32_monitor` is `None`"
                        .into(),
                    vuids: &["VUID-VkPhysicalDeviceSurfaceInfo2KHR-pNext-02672"],
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "`surface` is not a Win32 surface, or \
                        `surface_info.full_screen_exclusive` is not \
                        `FullScreenExclusive::ApplicationControlled`, but \
                        `surface_info.win32_monitor` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (true, true) | (false, false) => (),
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_present_modes_unchecked(
        &self,
        surface: &Surface,
        surface_info: SurfaceInfo,
    ) -> Result<impl Iterator<Item = PresentMode>, VulkanError> {
        surface
            .surface_present_modes
            .get_or_try_insert((self.handle, surface_info), |(_, surface_info)| {
                let &SurfaceInfo {
                    present_mode: _,
                    full_screen_exclusive,
                    win32_monitor,
                    _ne: _,
                } = surface_info;

                let mut info_vk = ash::vk::PhysicalDeviceSurfaceInfo2KHR {
                    surface: surface.handle(),
                    ..Default::default()
                };
                let mut full_screen_exclusive_info_vk = None;
                let mut full_screen_exclusive_win32_info_vk = None;

                if full_screen_exclusive != FullScreenExclusive::Default {
                    let next = full_screen_exclusive_info_vk.insert(
                        ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                            full_screen_exclusive: full_screen_exclusive.into(),
                            ..Default::default()
                        },
                    );

                    next.p_next = info_vk.p_next as *mut _;
                    info_vk.p_next = next as *const _ as *const _;
                }

                if let Some(win32_monitor) = win32_monitor {
                    let next = full_screen_exclusive_win32_info_vk.insert(
                        ash::vk::SurfaceFullScreenExclusiveWin32InfoEXT {
                            hmonitor: win32_monitor.0,
                            ..Default::default()
                        },
                    );

                    next.p_next = info_vk.p_next as *mut _;
                    info_vk.p_next = next as *const _ as *const _;
                }

                let fns = self.instance.fns();

                if self.supported_extensions().ext_full_screen_exclusive {
                    let modes = loop {
                        let mut count = 0;
                        (fns.ext_full_screen_exclusive
                            .get_physical_device_surface_present_modes2_ext)(
                            self.handle(),
                            &info_vk,
                            &mut count,
                            ptr::null_mut(),
                        )
                        .result()
                        .map_err(VulkanError::from)?;

                        let mut modes = Vec::with_capacity(count as usize);
                        let result = (fns
                            .ext_full_screen_exclusive
                            .get_physical_device_surface_present_modes2_ext)(
                            self.handle(),
                            &info_vk,
                            &mut count,
                            modes.as_mut_ptr(),
                        );

                        match result {
                            ash::vk::Result::SUCCESS => {
                                modes.set_len(count as usize);
                                break modes;
                            }
                            ash::vk::Result::INCOMPLETE => (),
                            err => return Err(VulkanError::from(err)),
                        }
                    };

                    Ok(modes
                        .into_iter()
                        .filter_map(|mode_vk| mode_vk.try_into().ok())
                        .collect())
                } else {
                    let modes = loop {
                        let mut count = 0;
                        (fns.khr_surface
                            .get_physical_device_surface_present_modes_khr)(
                            self.handle(),
                            surface.handle(),
                            &mut count,
                            ptr::null_mut(),
                        )
                        .result()
                        .map_err(VulkanError::from)?;

                        let mut modes = Vec::with_capacity(count as usize);
                        let result = (fns
                            .khr_surface
                            .get_physical_device_surface_present_modes_khr)(
                            self.handle(),
                            surface.handle(),
                            &mut count,
                            modes.as_mut_ptr(),
                        );

                        match result {
                            ash::vk::Result::SUCCESS => {
                                modes.set_len(count as usize);
                                break modes;
                            }
                            ash::vk::Result::INCOMPLETE => (),
                            err => return Err(VulkanError::from(err)),
                        }
                    };

                    Ok(modes
                        .into_iter()
                        .filter_map(|mode_vk| mode_vk.try_into().ok())
                        .collect())
                }
            })
            .map(IntoIterator::into_iter)
    }

    /// Returns whether queues of the given queue family can draw on the given surface.
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    #[inline]
    pub fn surface_support(
        &self,
        queue_family_index: u32,
        surface: &Surface,
    ) -> Result<bool, Validated<VulkanError>> {
        self.validate_surface_support(queue_family_index, surface)?;

        unsafe { Ok(self.surface_support_unchecked(queue_family_index, surface)?) }
    }

    fn validate_surface_support(
        &self,
        queue_family_index: u32,
        _surface: &Surface,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &["VUID-vkGetPhysicalDeviceSurfaceSupportKHR-queueFamilyIndex-01269"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_support_unchecked(
        &self,
        queue_family_index: u32,
        surface: &Surface,
    ) -> Result<bool, VulkanError> {
        surface
            .surface_support
            .get_or_try_insert((self.handle, queue_family_index), |_| {
                let fns = self.instance.fns();

                let mut output = MaybeUninit::uninit();
                (fns.khr_surface.get_physical_device_surface_support_khr)(
                    self.handle,
                    queue_family_index,
                    surface.handle(),
                    output.as_mut_ptr(),
                )
                .result()
                .map_err(VulkanError::from)?;

                Ok(output.assume_init() != 0)
            })
    }

    /// Retrieves the properties of tools that are currently active on the physical device.
    ///
    /// These properties may change during runtime, so the result only reflects the current
    /// situation and is not cached.
    ///
    /// The physical device API version must be at least 1.3, or the
    /// [`ext_tooling_info`](crate::device::DeviceExtensions::ext_tooling_info)
    /// extension must be supported by the physical device.
    #[inline]
    pub fn tool_properties(&self) -> Result<Vec<ToolProperties>, Validated<VulkanError>> {
        self.validate_tool_properties()?;

        unsafe { Ok(self.tool_properties_unchecked()?) }
    }

    fn validate_tool_properties(&self) -> Result<(), Box<ValidationError>> {
        if !(self.api_version() >= Version::V1_3 || self.supported_extensions().ext_tooling_info) {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::DeviceExtension("ext_tooling_info")]),
                ]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn tool_properties_unchecked(&self) -> Result<Vec<ToolProperties>, VulkanError> {
        let fns = self.instance.fns();

        loop {
            let mut count = 0;

            if self.api_version() >= Version::V1_3 {
                (fns.v1_3.get_physical_device_tool_properties)(
                    self.handle(),
                    &mut count,
                    ptr::null_mut(),
                )
            } else {
                (fns.ext_tooling_info.get_physical_device_tool_properties_ext)(
                    self.handle(),
                    &mut count,
                    ptr::null_mut(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut tool_properties = Vec::with_capacity(count as usize);
            let result = if self.api_version() >= Version::V1_3 {
                (fns.v1_3.get_physical_device_tool_properties)(
                    self.handle(),
                    &mut count,
                    tool_properties.as_mut_ptr(),
                )
            } else {
                (fns.ext_tooling_info.get_physical_device_tool_properties_ext)(
                    self.handle(),
                    &mut count,
                    tool_properties.as_mut_ptr(),
                )
            };

            match result {
                ash::vk::Result::INCOMPLETE => (),
                ash::vk::Result::SUCCESS => {
                    tool_properties.set_len(count as usize);

                    return Ok(tool_properties
                        .into_iter()
                        .map(|tool_properties| ToolProperties {
                            name: {
                                let bytes = cast_slice(tool_properties.name.as_slice());
                                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                                String::from_utf8_lossy(&bytes[0..end]).into()
                            },
                            version: {
                                let bytes = cast_slice(tool_properties.version.as_slice());
                                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                                String::from_utf8_lossy(&bytes[0..end]).into()
                            },
                            purposes: tool_properties.purposes.into(),
                            description: {
                                let bytes = cast_slice(tool_properties.description.as_slice());
                                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                                String::from_utf8_lossy(&bytes[0..end]).into()
                            },
                            layer: {
                                let bytes = cast_slice(tool_properties.layer.as_slice());
                                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                                String::from_utf8_lossy(&bytes[0..end]).into()
                            },
                        })
                        .collect());
                }
                err => return Err(VulkanError::from(err)),
            }
        }
    }

    /// Queries whether the physical device supports presenting to Wayland surfaces from queues of
    /// the given queue family.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Wayland `wl_display` handle.
    pub unsafe fn wayland_presentation_support<D>(
        &self,
        queue_family_index: u32,
        display: *const D,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_wayland_presentation_support(queue_family_index, display)?;

        Ok(self.wayland_presentation_support_unchecked(queue_family_index, display))
    }

    fn validate_wayland_presentation_support<D>(
        &self,
        queue_family_index: u32,
        _display: *const D,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_wayland_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_wayland_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceWaylandPresentationSupportKHR-queueFamilyIndex-01306",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceWaylandPresentationSupportKHR-display-parameter
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wayland_presentation_support_unchecked<D>(
        &self,
        queue_family_index: u32,
        display: *const D,
    ) -> bool {
        let fns = self.instance.fns();
        (fns.khr_wayland_surface
            .get_physical_device_wayland_presentation_support_khr)(
            self.handle,
            queue_family_index,
            display as *mut _,
        ) != 0
    }

    /// Queries whether the physical device supports presenting to Win32 surfaces from queues of the
    /// given queue family.
    #[inline]
    pub fn win32_presentation_support(
        &self,
        queue_family_index: u32,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_win32_presentation_support(queue_family_index)?;

        unsafe { Ok(self.win32_presentation_support_unchecked(queue_family_index)) }
    }

    fn validate_win32_presentation_support(
        &self,
        queue_family_index: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_win32_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_win32_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceWin32PresentationSupportKHR-queueFamilyIndex-01309",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn win32_presentation_support_unchecked(&self, queue_family_index: u32) -> bool {
        let fns = self.instance.fns();
        (fns.khr_win32_surface
            .get_physical_device_win32_presentation_support_khr)(
            self.handle, queue_family_index
        ) != 0
    }

    /// Queries whether the physical device supports presenting to XCB surfaces from queues of the
    /// given queue family.
    ///
    /// # Safety
    ///
    /// - `connection` must be a valid X11 `xcb_connection_t` handle.
    pub unsafe fn xcb_presentation_support<C>(
        &self,
        queue_family_index: u32,
        connection: *const C,
        visual_id: ash::vk::xcb_visualid_t,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_xcb_presentation_support(queue_family_index, connection, visual_id)?;

        Ok(self.xcb_presentation_support_unchecked(queue_family_index, connection, visual_id))
    }

    fn validate_xcb_presentation_support<C>(
        &self,
        queue_family_index: u32,
        _connection: *const C,
        _visual_id: ash::vk::xcb_visualid_t,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_xcb_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_xcb_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceXcbPresentationSupportKHR-queueFamilyIndex-01312",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceXcbPresentationSupportKHR-connection-parameter
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn xcb_presentation_support_unchecked<C>(
        &self,
        queue_family_index: u32,
        connection: *const C,
        visual_id: ash::vk::VisualID,
    ) -> bool {
        let fns = self.instance.fns();
        (fns.khr_xcb_surface
            .get_physical_device_xcb_presentation_support_khr)(
            self.handle,
            queue_family_index,
            connection as *mut _,
            visual_id,
        ) != 0
    }

    /// Queries whether the physical device supports presenting to Xlib surfaces from queues of the
    /// given queue family.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Xlib `Display` handle.
    pub unsafe fn xlib_presentation_support<D>(
        &self,
        queue_family_index: u32,
        display: *const D,
        visual_id: ash::vk::VisualID,
    ) -> Result<bool, Box<ValidationError>> {
        self.validate_xlib_presentation_support(queue_family_index, display, visual_id)?;

        Ok(self.xlib_presentation_support_unchecked(queue_family_index, display, visual_id))
    }

    fn validate_xlib_presentation_support<D>(
        &self,
        queue_family_index: u32,
        _display: *const D,
        _visual_id: ash::vk::VisualID,
    ) -> Result<(), Box<ValidationError>> {
        if !self.instance.enabled_extensions().khr_xlib_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_xlib_surface",
                )])]),
                ..Default::default()
            }));
        }

        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(Box::new(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &[
                    "VUID-vkGetPhysicalDeviceXlibPresentationSupportKHR-queueFamilyIndex-01315",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetPhysicalDeviceXlibPresentationSupportKHR-dpy-parameter
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn xlib_presentation_support_unchecked<D>(
        &self,
        queue_family_index: u32,
        display: *const D,
        visual_id: ash::vk::VisualID,
    ) -> bool {
        let fns = self.instance.fns();
        (fns.khr_xlib_surface
            .get_physical_device_xlib_presentation_support_khr)(
            self.handle,
            queue_family_index,
            display as *mut _,
            visual_id,
        ) != 0
    }
}

impl Debug for PhysicalDevice {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let Self {
            handle,
            instance,
            id,

            api_version,
            supported_extensions,
            supported_features,
            properties,
            extension_properties,
            memory_properties,
            queue_family_properties,

            display_properties: _,
            display_plane_properties: _,
            external_buffer_properties: _,
            external_fence_properties: _,
            external_semaphore_properties: _,
            format_properties: _,
            image_format_properties: _,
            sparse_image_format_properties: _,
        } = self;

        f.debug_struct("PhysicalDevice")
            .field("handle", handle)
            .field("instance", instance)
            .field("id", id)
            .field("api_version", api_version)
            .field("supported_extensions", supported_extensions)
            .field("supported_features", supported_features)
            .field("properties", properties)
            .field("extension_properties", extension_properties)
            .field("memory_properties", memory_properties)
            .field("queue_family_properties", queue_family_properties)
            .finish_non_exhaustive()
    }
}

unsafe impl VulkanObject for PhysicalDevice {
    type Handle = ash::vk::PhysicalDevice;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl InstanceOwned for PhysicalDevice {
    #[inline]
    fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
}

impl_id_counter!(PhysicalDevice);

/// Properties of a group of physical devices that can be used to create a single logical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PhysicalDeviceGroupProperties {
    /// The physical devices that make up the group.
    ///
    /// A physical device can belong to at most one group, but a group can consist of a single
    /// physical device.
    pub physical_devices: Vec<Arc<PhysicalDevice>>,

    /// Whether memory can be allocated from a subset of the devices in the group,
    /// rather than only from the group as a whole.
    ///
    /// If the length of `physical_devices` is 1, then this is always `false`.
    pub subset_allocation: bool,
}

#[repr(C)]
pub(crate) struct PhysicalDeviceGroupPropertiesRaw {
    pub(crate) physical_device_count: u32,
    pub(crate) physical_devices: [ash::vk::PhysicalDevice; ash::vk::MAX_DEVICE_GROUP_SIZE],
    pub(crate) subset_allocation: ash::vk::Bool32,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Type of a physical device.
    PhysicalDeviceType = PhysicalDeviceType(i32);

    /// The device is an integrated GPU.
    IntegratedGpu = INTEGRATED_GPU,

    /// The device is a discrete GPU.
    DiscreteGpu = DISCRETE_GPU,

    /// The device is a virtual GPU.
    VirtualGpu = VIRTUAL_GPU,

    /// The device is a CPU.
    Cpu = CPU,

    /// The device is something else.
    Other = OTHER,
}

impl Default for PhysicalDeviceType {
    #[inline]
    fn default() -> Self {
        PhysicalDeviceType::Other
    }
}

/// The version of the Vulkan conformance test that a driver is conformant against.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConformanceVersion {
    pub major: u8,
    pub minor: u8,
    pub subminor: u8,
    pub patch: u8,
}

impl From<ash::vk::ConformanceVersion> for ConformanceVersion {
    #[inline]
    fn from(val: ash::vk::ConformanceVersion) -> Self {
        ConformanceVersion {
            major: val.major,
            minor: val.minor,
            subminor: val.subminor,
            patch: val.patch,
        }
    }
}

impl Debug for ConformanceVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::fmt::Display for ConformanceVersion {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        Debug::fmt(self, f)
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// An identifier for the driver of a physical device.
    DriverId = DriverId(i32);

    // TODO: document
    AMDProprietary = AMD_PROPRIETARY,

    // TODO: document
    AMDOpenSource = AMD_OPEN_SOURCE,

    // TODO: document
    MesaRADV = MESA_RADV,

    // TODO: document
    NvidiaProprietary = NVIDIA_PROPRIETARY,

    // TODO: document
    IntelProprietaryWindows = INTEL_PROPRIETARY_WINDOWS,

    // TODO: document
    IntelOpenSourceMesa = INTEL_OPEN_SOURCE_MESA,

    // TODO: document
    ImaginationProprietary = IMAGINATION_PROPRIETARY,

    // TODO: document
    QualcommProprietary = QUALCOMM_PROPRIETARY,

    // TODO: document
    ARMProprietary = ARM_PROPRIETARY,

    // TODO: document
    GoogleSwiftshader = GOOGLE_SWIFTSHADER,

    // TODO: document
    GGPProprietary = GGP_PROPRIETARY,

    // TODO: document
    BroadcomProprietary = BROADCOM_PROPRIETARY,

    // TODO: document
    MesaLLVMpipe = MESA_LLVMPIPE,

    // TODO: document
    MoltenVK = MOLTENVK,

    // TODO: document
    CoreAVIProprietary = COREAVI_PROPRIETARY,

    // TODO: document
    JuiceProprietary = JUICE_PROPRIETARY,

    // TODO: document
    VeriSiliconPropertary = VERISILICON_PROPRIETARY,

    // TODO: document
    MesaTurnip = MESA_TURNIP,

    // TODO: document
    MesaV3DV = MESA_V3DV,

    // TODO: document
    MesaPanVK = MESA_PANVK,

    // TODO: document
    SamsungProprietary = SAMSUNG_PROPRIETARY,

    // TODO: document
    MesaVenus = MESA_VENUS,

    // TODO: document
    MesaDozen = MESA_DOZEN,
}

/// Information provided about an active tool.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ToolProperties {
    /// The name of the tool.
    pub name: String,

    /// The version of the tool.
    pub version: String,

    /// The purposes supported by the tool.
    pub purposes: ToolPurposes,

    /// A description of the tool.
    pub description: String,

    /// The layer implementing the tool, or empty if it is not implemented by a layer.
    pub layer: String,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// The purpose of an active tool.
    ToolPurposes = ToolPurposeFlags(u32);

    /// The tool provides validation of API usage.
    VALIDATION = VALIDATION,

    /// The tool provides profiling of API usage.
    PROFILING = PROFILING,

    /// The tool is capturing data about the application's API usage.
    TRACING = TRACING,

    /// The tool provides additional API features or extensions on top of the underlying
    /// implementation.
    ADDITIONAL_FEATURES = ADDITIONAL_FEATURES,

    /// The tool modifies the API features, limits or extensions presented to the application.
    MODIFYING_FEATURES = MODIFYING_FEATURES,

    /// The tool reports information to the user via a
    /// [`DebugUtilsMessenger`](crate::instance::debug::DebugUtilsMessenger).
    DEBUG_REPORTING = DEBUG_REPORTING_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_debug_utils)]),
        RequiresAllOf([InstanceExtension(ext_debug_report)]),
    ]),

    /// The tool consumes debug markers or object debug annotation, queue labels or command buffer
    /// labels.
    DEBUG_MARKERS = DEBUG_MARKERS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_debug_marker)]),
        RequiresAllOf([InstanceExtension(ext_debug_utils)]),
    ]),
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Specifies which subgroup operations are supported.
    SubgroupFeatures = SubgroupFeatureFlags(u32);

    // TODO: document
    BASIC = BASIC,

    // TODO: document
    VOTE = VOTE,

    // TODO: document
    ARITHMETIC = ARITHMETIC,

    // TODO: document
    BALLOT = BALLOT,

    // TODO: document
    SHUFFLE = SHUFFLE,

    // TODO: document
    SHUFFLE_RELATIVE = SHUFFLE_RELATIVE,

    // TODO: document
    CLUSTERED = CLUSTERED,

    // TODO: document
    QUAD = QUAD,

    // TODO: document
    PARTITIONED = PARTITIONED_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_shader_subgroup_partitioned)]),
    ]),
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies how the device clips single point primitives.
    PointClippingBehavior = PointClippingBehavior(i32);

    /// Points are clipped if they lie outside any clip plane, both those bounding the view volume
    /// and user-defined clip planes.
    AllClipPlanes = ALL_CLIP_PLANES,

    /// Points are clipped only if they lie outside a user-defined clip plane.
    UserClipPlanesOnly = USER_CLIP_PLANES_ONLY,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Specifies whether, and how, shader float controls can be set independently.
    ShaderFloatControlsIndependence = ShaderFloatControlsIndependence(i32);

    // TODO: document
    Float32Only = TYPE_32_ONLY,

    // TODO: document
    All = ALL,

    // TODO: document
    None = NONE,
}

/// Specifies shader core properties.
#[derive(Clone, Copy, Debug)]
pub struct ShaderCoreProperties {}

impl From<ash::vk::ShaderCorePropertiesFlagsAMD> for ShaderCoreProperties {
    #[inline]
    fn from(_val: ash::vk::ShaderCorePropertiesFlagsAMD) -> Self {
        Self {}
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    // TODO: document
    MemoryDecompressionMethods = MemoryDecompressionMethodFlagsNV(u64);

    // TODO: document
    GDEFLATE_1_0 = GDEFLATE_1_0,
}

vulkan_bitflags! {
    #[non_exhaustive]

    // TODO: document
    OpticalFlowGridSizes = OpticalFlowGridSizeFlagsNV(u32);

    // TODO: document
    SIZE_1X1 = TYPE_1X1,

    // TODO: document
    SIZE_2X2 = TYPE_2X2,

    // TODO: document
    SIZE_4X4 = TYPE_4X4,

    // TODO: document
    SIZE_8X8 = TYPE_8X8,
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    PipelineRobustnessBufferBehavior = PipelineRobustnessBufferBehaviorEXT(i32);

    // TODO: document
    DeviceDefault = DEVICE_DEFAULT,

    // TODO: document
    Disabled = DISABLED,

    // TODO: document
    RobustBufferAccess = ROBUST_BUFFER_ACCESS,

    // TODO: document
    RobustBufferAccess2 = ROBUST_BUFFER_ACCESS_2,
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    PipelineRobustnessImageBehavior = PipelineRobustnessImageBehaviorEXT(i32);

    // TODO: document
    DeviceDefault = DEVICE_DEFAULT,

    // TODO: document
    Disabled = DISABLED,

    // TODO: document
    RobustImageAccess = ROBUST_IMAGE_ACCESS,

    // TODO: document
    RobustImageAccess2 = ROBUST_IMAGE_ACCESS_2,
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    RayTracingInvocationReorderMode = RayTracingInvocationReorderModeNV(i32);

    // TODO: document
    None = NONE,

    // TODO: document
    Reorder = REORDER,
}
