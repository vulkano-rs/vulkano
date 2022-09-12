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
    device::{DeviceExtensions, Features, FeaturesFfi, Properties, PropertiesFfi},
    format::{Format, FormatProperties},
    image::{
        ImageCreateFlags, ImageFormatInfo, ImageFormatProperties, ImageUsage,
        SparseImageFormatInfo, SparseImageFormatProperties,
    },
    instance::Instance,
    macros::{vulkan_bitflags, vulkan_enum},
    memory::MemoryProperties,
    swapchain::{
        ColorSpace, FullScreenExclusive, PresentMode, SupportedSurfaceTransforms, Surface,
        SurfaceApi, SurfaceCapabilities, SurfaceInfo,
    },
    sync::{
        ExternalFenceInfo, ExternalFenceProperties, ExternalSemaphoreInfo,
        ExternalSemaphoreProperties,
    },
    OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use bytemuck::cast_slice;
use std::{
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ptr,
    sync::Arc,
};

/// Represents one of the available physical devices on this machine.
///
/// # Example
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
#[derive(Clone, Debug)]
pub struct PhysicalDevice {
    handle: ash::vk::PhysicalDevice,
    instance: Arc<Instance>,

    properties: Properties,
    extension_properties: Vec<ExtensionProperties>,
    memory_properties: MemoryProperties,
    queue_family_properties: Vec<QueueFamilyProperties>,

    api_version: Version,
    supported_extensions: DeviceExtensions,
    supported_features: Features,
}

impl PhysicalDevice {
    pub unsafe fn from_handle(
        handle: ash::vk::PhysicalDevice,
        instance: Arc<Instance>,
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
            instance,

            properties,
            extension_properties,
            memory_properties,
            queue_family_properties,

            api_version,
            supported_extensions,
            supported_features,
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

    /// Retrieves the external memory properties supported for buffers with a given configuration.
    ///
    /// Instance API version must be at least 1.1, or the
    /// [`khr_external_memory_capabilities`](crate::instance::InstanceExtensions::khr_external_memory_capabilities)
    /// extension must be enabled on the instance.
    #[inline]
    pub fn external_buffer_properties(
        &self,
        info: ExternalBufferInfo,
    ) -> Result<ExternalBufferProperties, ExternalBufferPropertiesError> {
        self.validate_external_buffer_properties(&info)?;

        unsafe { Ok(self.external_buffer_properties_unchecked(info)) }
    }

    fn validate_external_buffer_properties(
        &self,
        info: &ExternalBufferInfo,
    ) -> Result<(), ExternalBufferPropertiesError> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_memory_capabilities)
        {
            return Err(ExternalBufferPropertiesError::RequirementNotMet {
                required_for: "`external_buffer_properties`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_1),
                    instance_extensions: &["khr_external_memory_capabilities"],
                    ..Default::default()
                },
            });
        }

        let &ExternalBufferInfo {
            handle_type,
            usage,
            sparse: _,
            _ne: _,
        } = info;

        // VUID-VkPhysicalDeviceExternalBufferInfo-usage-parameter
        usage.validate_physical_device(self)?;

        // VUID-VkPhysicalDeviceExternalBufferInfo-usage-requiredbitmask
        assert!(!usage.is_empty());

        // VUID-VkPhysicalDeviceExternalBufferInfo-handleType-parameter
        handle_type.validate_physical_device(self)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn external_buffer_properties_unchecked(
        &self,
        info: ExternalBufferInfo,
    ) -> ExternalBufferProperties {
        /* Input */

        let ExternalBufferInfo {
            handle_type,
            usage,
            sparse,
            _ne: _,
        } = info;

        let external_buffer_info = ash::vk::PhysicalDeviceExternalBufferInfo {
            flags: sparse.map(Into::into).unwrap_or_default(),
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
    }

    /// Retrieves the external handle properties supported for fences with a given
    /// configuration.
    ///
    /// The instance API version must be at least 1.1, or the
    /// [`khr_external_fence_capabilities`](crate::instance::InstanceExtensions::khr_external_fence_capabilities)
    /// extension must be enabled on the instance.
    #[inline]
    pub fn external_fence_properties(
        &self,
        info: ExternalFenceInfo,
    ) -> Result<ExternalFenceProperties, ExternalFenceSemaphorePropertiesError> {
        self.validate_external_fence_properties(&info)?;

        unsafe { Ok(self.external_fence_properties_unchecked(info)) }
    }

    fn validate_external_fence_properties(
        &self,
        info: &ExternalFenceInfo,
    ) -> Result<(), ExternalFenceSemaphorePropertiesError> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_fence_capabilities)
        {
            return Err(ExternalFenceSemaphorePropertiesError::RequirementNotMet {
                required_for: "`external_fence_properties`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_1),
                    instance_extensions: &["khr_external_fence_capabilities"],
                    ..Default::default()
                },
            });
        }

        let &ExternalFenceInfo {
            handle_type,
            _ne: _,
        } = info;

        // VUID-VkPhysicalDeviceExternalFenceInfo-handleType-parameter
        handle_type.validate_physical_device(self)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn external_fence_properties_unchecked(
        &self,
        info: ExternalFenceInfo,
    ) -> ExternalFenceProperties {
        /* Input */

        let ExternalFenceInfo {
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
    }

    /// Retrieves the external handle properties supported for semaphores with a given
    /// configuration.
    ///
    /// The instance API version must be at least 1.1, or the
    /// [`khr_external_semaphore_capabilities`](crate::instance::InstanceExtensions::khr_external_semaphore_capabilities)
    /// extension must be enabled on the instance.
    #[inline]
    pub fn external_semaphore_properties(
        &self,
        info: ExternalSemaphoreInfo,
    ) -> Result<ExternalSemaphoreProperties, ExternalFenceSemaphorePropertiesError> {
        self.validate_external_semaphore_properties(&info)?;

        unsafe { Ok(self.external_semaphore_properties_unchecked(info)) }
    }

    fn validate_external_semaphore_properties(
        &self,
        info: &ExternalSemaphoreInfo,
    ) -> Result<(), ExternalFenceSemaphorePropertiesError> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_semaphore_capabilities)
        {
            return Err(ExternalFenceSemaphorePropertiesError::RequirementNotMet {
                required_for: "`external_semaphore_properties`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_1),
                    instance_extensions: &["khr_external_semaphore_capabilities"],
                    ..Default::default()
                },
            });
        }

        let &ExternalSemaphoreInfo {
            handle_type,
            _ne: _,
        } = info;

        // VUID-VkPhysicalDeviceExternalSemaphoreInfo-handleType-parameter
        handle_type.validate_physical_device(self)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn external_semaphore_properties_unchecked(
        &self,
        info: ExternalSemaphoreInfo,
    ) -> ExternalSemaphoreProperties {
        /* Input */

        let ExternalSemaphoreInfo {
            handle_type,
            _ne: _,
        } = info;

        let external_semaphore_info = ash::vk::PhysicalDeviceExternalSemaphoreInfo {
            handle_type: handle_type.into(),
            ..Default::default()
        };

        /* Output */

        let mut external_semaphore_properties = ash::vk::ExternalSemaphoreProperties::default();

        /* Call */

        let fns = self.instance.fns();

        if self.instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_external_semaphore_properties)(
                self.handle,
                &external_semaphore_info,
                &mut external_semaphore_properties,
            )
        } else {
            (fns.khr_external_semaphore_capabilities
                .get_physical_device_external_semaphore_properties_khr)(
                self.handle,
                &external_semaphore_info,
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
            compatible_handle_types: external_semaphore_properties.compatible_handle_types.into(),
        }
    }

    /// Retrieves the properties of a format when used by this physical device.
    #[inline]
    pub fn format_properties(&self, format: Format) -> FormatProperties {
        // TODO: self.validate_format_properties(format_properties)?;

        unsafe { self.format_properties_unchecked(format) }
    }

    /*
    TODO:
    fn validate_format_properties(&self, format: Format) -> Result<(), ()> {
        format.validate_physical_device(self)?;
    }
     */

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn format_properties_unchecked(&self, format: Format) -> FormatProperties {
        let mut format_properties2 = ash::vk::FormatProperties2::default();
        let mut format_properties3 = if self.api_version() >= Version::V1_3
            || self.supported_extensions().khr_format_feature_flags2
        {
            Some(ash::vk::FormatProperties3KHR::default())
        } else {
            None
        };

        if let Some(next) = format_properties3.as_mut() {
            next.p_next = format_properties2.p_next;
            format_properties2.p_next = next as *mut _ as *mut _;
        }

        let fns = self.instance.fns();

        if self.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_format_properties2)(
                self.handle,
                format.into(),
                &mut format_properties2,
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
                &mut format_properties2,
            );
        } else {
            (fns.v1_0.get_physical_device_format_properties)(
                self.internal_object(),
                format.into(),
                &mut format_properties2.format_properties,
            );
        }

        match format_properties3 {
            Some(format_properties3) => FormatProperties {
                linear_tiling_features: format_properties3.linear_tiling_features.into(),
                optimal_tiling_features: format_properties3.optimal_tiling_features.into(),
                buffer_features: format_properties3.buffer_features.into(),
                _ne: crate::NonExhaustive(()),
            },
            None => FormatProperties {
                linear_tiling_features: format_properties2
                    .format_properties
                    .linear_tiling_features
                    .into(),
                optimal_tiling_features: format_properties2
                    .format_properties
                    .optimal_tiling_features
                    .into(),
                buffer_features: format_properties2.format_properties.buffer_features.into(),
                _ne: crate::NonExhaustive(()),
            },
        }
    }

    /// Returns the properties supported for images with a given image configuration.
    ///
    /// `Some` is returned if the configuration is supported, `None` if it is not.
    ///
    /// # Panics
    ///
    /// - Panics if `image_format_info.format` is `None`.
    #[inline]
    pub fn image_format_properties(
        &self,
        image_format_info: ImageFormatInfo,
    ) -> Result<Option<ImageFormatProperties>, ImageFormatPropertiesError> {
        self.validate_image_format_properties(&image_format_info)?;

        unsafe { Ok(self.image_format_properties_unchecked(image_format_info)?) }
    }

    pub fn validate_image_format_properties(
        &self,
        image_format_info: &ImageFormatInfo,
    ) -> Result<(), ImageFormatPropertiesError> {
        let &ImageFormatInfo {
            format: _,
            image_type,
            tiling,
            usage,
            external_memory_handle_type,
            image_view_type,
            mutable_format: _,
            cube_compatible: _,
            array_2d_compatible: _,
            block_texel_view_compatible: _,
            _ne: _,
        } = image_format_info;

        // VUID-VkPhysicalDeviceImageFormatInfo2-format-parameter
        // TODO: format.validate_physical_device(self)?;

        // VUID-VkPhysicalDeviceImageFormatInfo2-imageType-parameter
        image_type.validate_physical_device(self)?;

        // VUID-VkPhysicalDeviceImageFormatInfo2-tiling-parameter
        tiling.validate_physical_device(self)?;

        // VUID-VkPhysicalDeviceImageFormatInfo2-usage-parameter
        usage.validate_physical_device(self)?;

        // VUID-VkPhysicalDeviceImageFormatInfo2-usage-requiredbitmask
        assert!(!usage.is_empty());

        if let Some(handle_type) = external_memory_handle_type {
            if !(self.api_version() >= Version::V1_1
                || self
                    .instance()
                    .enabled_extensions()
                    .khr_external_memory_capabilities)
            {
                return Err(ImageFormatPropertiesError::RequirementNotMet {
                    required_for: "`image_format_info.external_memory_handle_type` is `Some`",
                    requires_one_of: RequiresOneOf {
                        api_version: Some(Version::V1_1),
                        instance_extensions: &["khr_external_memory_capabilities"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkPhysicalDeviceExternalImageFormatInfo-handleType-parameter
            handle_type.validate_physical_device(self)?;
        }

        if let Some(image_view_type) = image_view_type {
            if !self.supported_extensions().ext_filter_cubic {
                return Err(ImageFormatPropertiesError::RequirementNotMet {
                    required_for: "`image_format_info.image_view_type` is `Some`",
                    requires_one_of: RequiresOneOf {
                        device_extensions: &["ext_filter_cubic"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkPhysicalDeviceImageViewImageFormatInfoEXT-imageViewType-parameter
            image_view_type.validate_physical_device(self)?;
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn image_format_properties_unchecked(
        &self,
        image_format_info: ImageFormatInfo,
    ) -> Result<Option<ImageFormatProperties>, VulkanError> {
        /* Input */

        let ImageFormatInfo {
            format,
            image_type,
            tiling,
            usage,
            external_memory_handle_type,
            image_view_type,
            mutable_format,
            cube_compatible,
            array_2d_compatible,
            block_texel_view_compatible,
            _ne: _,
        } = image_format_info;

        let flags = ImageCreateFlags {
            mutable_format,
            cube_compatible,
            array_2d_compatible,
            block_texel_view_compatible,
            ..ImageCreateFlags::empty()
        };

        let mut format_info2 = ash::vk::PhysicalDeviceImageFormatInfo2::builder()
            .format(format.unwrap().into())
            .ty(image_type.into())
            .tiling(tiling.into())
            .usage(usage.into())
            .flags(flags.into());

        let mut external_image_format_info = external_memory_handle_type.map(|handle_type| {
            ash::vk::PhysicalDeviceExternalImageFormatInfo::builder()
                .handle_type(handle_type.into())
        });

        if let Some(next) = external_image_format_info.as_mut() {
            format_info2 = format_info2.push_next(next);
        }

        let mut image_view_image_format_info = image_view_type.map(|image_view_type| {
            ash::vk::PhysicalDeviceImageViewImageFormatInfoEXT::builder()
                .image_view_type(image_view_type.into())
        });

        if let Some(next) = image_view_image_format_info.as_mut() {
            format_info2 = format_info2.push_next(next);
        }

        /* Output */

        let mut image_format_properties2 = ash::vk::ImageFormatProperties2::default();

        let mut external_image_format_properties = if external_memory_handle_type.is_some() {
            Some(ash::vk::ExternalImageFormatProperties::default())
        } else {
            None
        };

        if let Some(next) = external_image_format_properties.as_mut() {
            next.p_next = image_format_properties2.p_next;
            image_format_properties2.p_next = next as *mut _ as *mut _;
        }

        let mut filter_cubic_image_view_image_format_properties = if image_view_type.is_some() {
            Some(ash::vk::FilterCubicImageViewImageFormatPropertiesEXT::default())
        } else {
            None
        };

        if let Some(next) = filter_cubic_image_view_image_format_properties.as_mut() {
            next.p_next = image_format_properties2.p_next;
            image_format_properties2.p_next = next as *mut _ as *mut _;
        }

        let result = {
            let fns = self.instance.fns();

            if self.api_version() >= Version::V1_1 {
                (fns.v1_1.get_physical_device_image_format_properties2)(
                    self.handle,
                    &format_info2.build(),
                    &mut image_format_properties2,
                )
            } else if self
                .instance
                .enabled_extensions()
                .khr_get_physical_device_properties2
            {
                (fns.khr_get_physical_device_properties2
                    .get_physical_device_image_format_properties2_khr)(
                    self.handle,
                    &format_info2.build(),
                    &mut image_format_properties2,
                )
            } else {
                // Can't query this, return unsupported
                if !format_info2.p_next.is_null() {
                    return Ok(None);
                }

                (fns.v1_0.get_physical_device_image_format_properties)(
                    self.handle,
                    format_info2.format,
                    format_info2.ty,
                    format_info2.tiling,
                    format_info2.usage,
                    format_info2.flags,
                    &mut image_format_properties2.image_format_properties,
                )
            }
            .result()
            .map_err(VulkanError::from)
        };

        match result {
            Ok(_) => Ok(Some(ImageFormatProperties {
                external_memory_properties: external_image_format_properties
                    .map(|properties| properties.external_memory_properties.into())
                    .unwrap_or_default(),
                filter_cubic: filter_cubic_image_view_image_format_properties
                    .map_or(false, |properties| {
                        properties.filter_cubic != ash::vk::FALSE
                    }),
                filter_cubic_minmax: filter_cubic_image_view_image_format_properties
                    .map_or(false, |properties| {
                        properties.filter_cubic_minmax != ash::vk::FALSE
                    }),
                ..image_format_properties2.image_format_properties.into()
            })),
            Err(VulkanError::FormatNotSupported) => Ok(None),
            Err(err) => Err(err),
        }
    }

    /// Returns the properties of sparse images with a given image configuration.
    ///
    /// # Panics
    ///
    /// - Panics if `format_info.format` is `None`.
    #[inline]
    pub fn sparse_image_format_properties(
        &self,
        format_info: SparseImageFormatInfo,
    ) -> Result<Vec<SparseImageFormatProperties>, ImageFormatPropertiesError> {
        self.validate_sparse_image_format_properties(&format_info)?;

        unsafe { Ok(self.sparse_image_format_properties_unchecked(format_info)) }
    }

    fn validate_sparse_image_format_properties(
        &self,
        format_info: &SparseImageFormatInfo,
    ) -> Result<(), ImageFormatPropertiesError> {
        let &SparseImageFormatInfo {
            format: _,
            image_type,
            samples,
            usage,
            tiling,
            _ne: _,
        } = format_info;

        // VUID-VkPhysicalDeviceSparseImageFormatInfo2-format-parameter
        // TODO: format.validate_physical_device(self)?;

        // VUID-VkPhysicalDeviceSparseImageFormatInfo2-type-parameter
        image_type.validate_physical_device(self)?;

        // VUID-VkPhysicalDeviceSparseImageFormatInfo2-samples-parameter
        samples.validate_physical_device(self)?;

        // VUID-VkPhysicalDeviceSparseImageFormatInfo2-usage-parameter
        usage.validate_physical_device(self)?;

        // VUID-VkPhysicalDeviceSparseImageFormatInfo2-usage-requiredbitmask
        assert!(!usage.is_empty());

        // VUID-VkPhysicalDeviceSparseImageFormatInfo2-tiling-parameter
        tiling.validate_physical_device(self)?;

        // VUID-VkPhysicalDeviceSparseImageFormatInfo2-samples-01095
        // TODO:

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn sparse_image_format_properties_unchecked(
        &self,
        format_info: SparseImageFormatInfo,
    ) -> Vec<SparseImageFormatProperties> {
        let SparseImageFormatInfo {
            format,
            image_type,
            samples,
            usage,
            tiling,
            _ne: _,
        } = format_info;

        let format_info2 = ash::vk::PhysicalDeviceSparseImageFormatInfo2 {
            format: format.unwrap().into(),
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
    }

    /// Returns the capabilities that are supported by the physical device for the given surface.
    ///
    /// # Panic
    ///
    /// - Panics if the physical device and the surface don't belong to the same instance.
    #[inline]
    pub fn surface_capabilities<W>(
        &self,
        surface: &Surface<W>,
        surface_info: SurfaceInfo,
    ) -> Result<SurfaceCapabilities, SurfacePropertiesError> {
        self.validate_surface_capabilities(surface, &surface_info)?;

        unsafe { Ok(self.surface_capabilities_unchecked(surface, surface_info)?) }
    }

    fn validate_surface_capabilities<W>(
        &self,
        surface: &Surface<W>,
        surface_info: &SurfaceInfo,
    ) -> Result<(), SurfacePropertiesError> {
        if !(self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
            || self.instance.enabled_extensions().khr_surface)
        {
            return Err(SurfacePropertiesError::RequirementNotMet {
                required_for: "`surface_capabilities`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["khr_get_surface_capabilities2", "khr_surface"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkGetPhysicalDeviceSurfaceCapabilities2KHR-commonparent
        assert_eq!(self.instance(), surface.instance());

        let &SurfaceInfo {
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        // VUID-vkGetPhysicalDeviceSurfaceCapabilities2KHR-pSurfaceInfo-06210
        // TODO:

        if !self.supported_extensions().ext_full_screen_exclusive
            && full_screen_exclusive != FullScreenExclusive::Default
        {
            return Err(SurfacePropertiesError::NotSupported);
        }

        // VUID-VkPhysicalDeviceSurfaceInfo2KHR-pNext-02672
        if (surface.api() == SurfaceApi::Win32
            && full_screen_exclusive == FullScreenExclusive::ApplicationControlled)
            != win32_monitor.is_some()
        {
            return Err(SurfacePropertiesError::NotSupported);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_capabilities_unchecked<W>(
        &self,
        surface: &Surface<W>,
        surface_info: SurfaceInfo,
    ) -> Result<SurfaceCapabilities, VulkanError> {
        /* Input */

        let SurfaceInfo {
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        let mut surface_full_screen_exclusive_info = self
            .supported_extensions()
            .ext_full_screen_exclusive
            .then(|| ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                full_screen_exclusive: full_screen_exclusive.into(),
                ..Default::default()
            });

        let mut surface_full_screen_exclusive_win32_info =
            win32_monitor.map(
                |win32_monitor| ash::vk::SurfaceFullScreenExclusiveWin32InfoEXT {
                    hmonitor: win32_monitor.0,
                    ..Default::default()
                },
            );

        let mut surface_info2 = ash::vk::PhysicalDeviceSurfaceInfo2KHR {
            surface: surface.internal_object(),
            ..Default::default()
        };

        if let Some(surface_full_screen_exclusive_info) =
            surface_full_screen_exclusive_info.as_mut()
        {
            surface_full_screen_exclusive_info.p_next = surface_info2.p_next as *mut _;
            surface_info2.p_next = surface_full_screen_exclusive_info as *const _ as *const _;
        }

        if let Some(surface_full_screen_exclusive_win32_info) =
            surface_full_screen_exclusive_win32_info.as_mut()
        {
            surface_full_screen_exclusive_win32_info.p_next = surface_info2.p_next as *mut _;
            surface_info2.p_next = surface_full_screen_exclusive_win32_info as *const _ as *const _;
        }

        /* Output */

        let mut surface_capabilities2 = ash::vk::SurfaceCapabilities2KHR::default();

        let mut surface_capabilities_full_screen_exclusive =
            if surface_full_screen_exclusive_info.is_some() {
                Some(ash::vk::SurfaceCapabilitiesFullScreenExclusiveEXT::default())
            } else {
                None
            };

        if let Some(surface_capabilities_full_screen_exclusive) =
            surface_capabilities_full_screen_exclusive.as_mut()
        {
            surface_capabilities_full_screen_exclusive.p_next =
                surface_capabilities2.p_next as *mut _;
            surface_capabilities2.p_next =
                surface_capabilities_full_screen_exclusive as *mut _ as *mut _;
        }

        let fns = self.instance.fns();

        if self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
        {
            (fns.khr_get_surface_capabilities2
                .get_physical_device_surface_capabilities2_khr)(
                self.internal_object(),
                &surface_info2,
                &mut surface_capabilities2,
            )
            .result()
            .map_err(VulkanError::from)?;
        } else {
            (fns.khr_surface.get_physical_device_surface_capabilities_khr)(
                self.internal_object(),
                surface_info2.surface,
                &mut surface_capabilities2.surface_capabilities,
            )
            .result()
            .map_err(VulkanError::from)?;
        };

        Ok(SurfaceCapabilities {
            min_image_count: surface_capabilities2.surface_capabilities.min_image_count,
            max_image_count: if surface_capabilities2.surface_capabilities.max_image_count == 0 {
                None
            } else {
                Some(surface_capabilities2.surface_capabilities.max_image_count)
            },
            current_extent: if surface_capabilities2
                .surface_capabilities
                .current_extent
                .width
                == 0xffffffff
                && surface_capabilities2
                    .surface_capabilities
                    .current_extent
                    .height
                    == 0xffffffff
            {
                None
            } else {
                Some([
                    surface_capabilities2
                        .surface_capabilities
                        .current_extent
                        .width,
                    surface_capabilities2
                        .surface_capabilities
                        .current_extent
                        .height,
                ])
            },
            min_image_extent: [
                surface_capabilities2
                    .surface_capabilities
                    .min_image_extent
                    .width,
                surface_capabilities2
                    .surface_capabilities
                    .min_image_extent
                    .height,
            ],
            max_image_extent: [
                surface_capabilities2
                    .surface_capabilities
                    .max_image_extent
                    .width,
                surface_capabilities2
                    .surface_capabilities
                    .max_image_extent
                    .height,
            ],
            max_image_array_layers: surface_capabilities2
                .surface_capabilities
                .max_image_array_layers,
            supported_transforms: surface_capabilities2
                .surface_capabilities
                .supported_transforms
                .into(),

            current_transform: SupportedSurfaceTransforms::from(
                surface_capabilities2.surface_capabilities.current_transform,
            )
            .iter()
            .next()
            .unwrap(), // TODO:
            supported_composite_alpha: surface_capabilities2
                .surface_capabilities
                .supported_composite_alpha
                .into(),
            supported_usage_flags: {
                let usage = ImageUsage::from(
                    surface_capabilities2
                        .surface_capabilities
                        .supported_usage_flags,
                );
                debug_assert!(usage.color_attachment); // specs say that this must be true
                usage
            },

            full_screen_exclusive_supported: surface_capabilities_full_screen_exclusive
                .map_or(false, |c| c.full_screen_exclusive_supported != 0),
        })
    }

    /// Returns the combinations of format and color space that are supported by the physical device
    /// for the given surface.
    ///
    /// # Panic
    ///
    /// - Panics if the physical device and the surface don't belong to the same instance.
    #[inline]
    pub fn surface_formats<W>(
        &self,
        surface: &Surface<W>,
        surface_info: SurfaceInfo,
    ) -> Result<Vec<(Format, ColorSpace)>, SurfacePropertiesError> {
        self.validate_surface_formats(surface, &surface_info)?;

        unsafe { Ok(self.surface_formats_unchecked(surface, surface_info)?) }
    }

    fn validate_surface_formats<W>(
        &self,
        surface: &Surface<W>,
        surface_info: &SurfaceInfo,
    ) -> Result<(), SurfacePropertiesError> {
        if !(self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
            || self.instance.enabled_extensions().khr_surface)
        {
            return Err(SurfacePropertiesError::RequirementNotMet {
                required_for: "`surface_formats`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["khr_get_surface_capabilities2", "khr_surface"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkGetPhysicalDeviceSurfaceFormats2KHR-commonparent
        assert_eq!(self.instance(), surface.instance());

        // VUID-vkGetPhysicalDeviceSurfaceFormats2KHR-pSurfaceInfo-06522
        // TODO:

        let &SurfaceInfo {
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        if self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
        {
            if !self.supported_extensions().ext_full_screen_exclusive
                && full_screen_exclusive != FullScreenExclusive::Default
            {
                return Err(SurfacePropertiesError::NotSupported);
            }

            // VUID-VkPhysicalDeviceSurfaceInfo2KHR-pNext-02672
            if (surface.api() == SurfaceApi::Win32
                && full_screen_exclusive == FullScreenExclusive::ApplicationControlled)
                != win32_monitor.is_some()
            {
                return Err(SurfacePropertiesError::NotSupported);
            }
        } else {
            if full_screen_exclusive != FullScreenExclusive::Default {
                return Err(SurfacePropertiesError::NotSupported);
            }

            if win32_monitor.is_some() {
                return Err(SurfacePropertiesError::NotSupported);
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_formats_unchecked<W>(
        &self,
        surface: &Surface<W>,
        surface_info: SurfaceInfo,
    ) -> Result<Vec<(Format, ColorSpace)>, VulkanError> {
        let SurfaceInfo {
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        let mut surface_full_screen_exclusive_info = (full_screen_exclusive
            != FullScreenExclusive::Default)
            .then(|| ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                full_screen_exclusive: full_screen_exclusive.into(),
                ..Default::default()
            });

        let mut surface_full_screen_exclusive_win32_info =
            win32_monitor.map(
                |win32_monitor| ash::vk::SurfaceFullScreenExclusiveWin32InfoEXT {
                    hmonitor: win32_monitor.0,
                    ..Default::default()
                },
            );

        let mut surface_info2 = ash::vk::PhysicalDeviceSurfaceInfo2KHR {
            surface: surface.internal_object(),
            ..Default::default()
        };

        if let Some(surface_full_screen_exclusive_info) =
            surface_full_screen_exclusive_info.as_mut()
        {
            surface_full_screen_exclusive_info.p_next = surface_info2.p_next as *mut _;
            surface_info2.p_next = surface_full_screen_exclusive_info as *const _ as *const _;
        }

        if let Some(surface_full_screen_exclusive_win32_info) =
            surface_full_screen_exclusive_win32_info.as_mut()
        {
            surface_full_screen_exclusive_win32_info.p_next = surface_info2.p_next as *mut _;
            surface_info2.p_next = surface_full_screen_exclusive_win32_info as *const _ as *const _;
        }

        let fns = self.instance.fns();

        if self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
        {
            let surface_format2s = loop {
                let mut count = 0;
                (fns.khr_get_surface_capabilities2
                    .get_physical_device_surface_formats2_khr)(
                    self.internal_object(),
                    &surface_info2,
                    &mut count,
                    ptr::null_mut(),
                )
                .result()
                .map_err(VulkanError::from)?;

                let mut surface_format2s =
                    vec![ash::vk::SurfaceFormat2KHR::default(); count as usize];
                let result = (fns
                    .khr_get_surface_capabilities2
                    .get_physical_device_surface_formats2_khr)(
                    self.internal_object(),
                    &surface_info2,
                    &mut count,
                    surface_format2s.as_mut_ptr(),
                );

                match result {
                    ash::vk::Result::SUCCESS => {
                        surface_format2s.set_len(count as usize);
                        break surface_format2s;
                    }
                    ash::vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err)),
                }
            };

            Ok(surface_format2s
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
                    self.internal_object(),
                    surface.internal_object(),
                    &mut count,
                    ptr::null_mut(),
                )
                .result()
                .map_err(VulkanError::from)?;

                let mut surface_formats = Vec::with_capacity(count as usize);
                let result = (fns.khr_surface.get_physical_device_surface_formats_khr)(
                    self.internal_object(),
                    surface.internal_object(),
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
    }

    /// Returns the present modes that are supported by the physical device for the given surface.
    ///
    /// # Panic
    ///
    /// - Panics if the physical device and the surface don't belong to the same instance.
    #[inline]
    pub fn surface_present_modes<W>(
        &self,
        surface: &Surface<W>,
    ) -> Result<impl Iterator<Item = PresentMode>, SurfacePropertiesError> {
        self.validate_surface_present_modes(surface)?;

        unsafe { Ok(self.surface_present_modes_unchecked(surface)?) }
    }

    fn validate_surface_present_modes<W>(
        &self,
        surface: &Surface<W>,
    ) -> Result<(), SurfacePropertiesError> {
        if !self.instance.enabled_extensions().khr_surface {
            return Err(SurfacePropertiesError::RequirementNotMet {
                required_for: "`surface_present_modes`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["khr_surface"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkGetPhysicalDeviceSurfacePresentModesKHR-commonparent
        assert_eq!(self.instance(), surface.instance());

        // VUID-vkGetPhysicalDeviceSurfacePresentModesKHR-surface-06525
        // TODO:

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_present_modes_unchecked<W>(
        &self,
        surface: &Surface<W>,
    ) -> Result<impl Iterator<Item = PresentMode>, VulkanError> {
        let fns = self.instance.fns();

        let modes = loop {
            let mut count = 0;
            (fns.khr_surface
                .get_physical_device_surface_present_modes_khr)(
                self.internal_object(),
                surface.internal_object(),
                &mut count,
                ptr::null_mut(),
            )
            .result()
            .map_err(VulkanError::from)?;

            let mut modes = Vec::with_capacity(count as usize);
            let result = (fns
                .khr_surface
                .get_physical_device_surface_present_modes_khr)(
                self.internal_object(),
                surface.internal_object(),
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

        debug_assert!(!modes.is_empty());
        debug_assert!(modes.iter().any(|&m| m == ash::vk::PresentModeKHR::FIFO));

        Ok(modes
            .into_iter()
            .filter_map(|mode_vk| mode_vk.try_into().ok()))
    }

    /// Returns whether queues of the given queue family can draw on the given surface.
    #[inline]
    pub fn surface_support<W>(
        &self,
        queue_family_index: u32,
        surface: &Surface<W>,
    ) -> Result<bool, SurfacePropertiesError> {
        self.validate_surface_support(queue_family_index, surface)?;

        unsafe { Ok(self.surface_support_unchecked(queue_family_index, surface)?) }
    }

    fn validate_surface_support<W>(
        &self,
        queue_family_index: u32,
        _surface: &Surface<W>,
    ) -> Result<(), SurfacePropertiesError> {
        if !self.instance.enabled_extensions().khr_surface {
            return Err(SurfacePropertiesError::RequirementNotMet {
                required_for: "`surface_support`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["khr_surface"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkGetPhysicalDeviceSurfaceSupportKHR-queueFamilyIndex-01269
        if queue_family_index >= self.queue_family_properties.len() as u32 {
            return Err(SurfacePropertiesError::QueueFamilyIndexOutOfRange {
                queue_family_index,
                queue_family_count: self.queue_family_properties.len() as u32,
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn surface_support_unchecked<W>(
        &self,
        queue_family_index: u32,
        surface: &Surface<W>,
    ) -> Result<bool, VulkanError> {
        let fns = self.instance.fns();

        let mut output = MaybeUninit::uninit();
        (fns.khr_surface.get_physical_device_surface_support_khr)(
            self.handle,
            queue_family_index,
            surface.internal_object(),
            output.as_mut_ptr(),
        )
        .result()
        .map_err(VulkanError::from)?;

        Ok(output.assume_init() != 0)
    }
}

unsafe impl VulkanObject for PhysicalDevice {
    type Object = ash::vk::PhysicalDevice;

    #[inline]
    fn internal_object(&self) -> ash::vk::PhysicalDevice {
        self.handle
    }
}

impl PartialEq for PhysicalDevice {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.instance == other.instance
    }
}

impl Eq for PhysicalDevice {}

impl Hash for PhysicalDevice {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.instance.hash(state);
    }
}

/// Properties of an extension in the loader or a physical device.
#[derive(Clone, Debug)]
pub struct ExtensionProperties {
    /// The name of the extension.
    pub extension_name: String,

    /// The version of the extension.
    pub spec_version: u32,
}

impl From<ash::vk::ExtensionProperties> for ExtensionProperties {
    #[inline]
    fn from(val: ash::vk::ExtensionProperties) -> Self {
        Self {
            extension_name: {
                let bytes = cast_slice(val.extension_name.as_slice());
                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                String::from_utf8_lossy(&bytes[0..end]).into()
            },
            spec_version: val.spec_version,
        }
    }
}

vulkan_enum! {
    /// Type of a physical device.
    #[non_exhaustive]
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
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), FmtError> {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Display for ConformanceVersion {
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), FmtError> {
        Debug::fmt(self, formatter)
    }
}

vulkan_enum! {
    /// An identifier for the driver of a physical device.
    #[non_exhaustive]
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
}

vulkan_bitflags! {
    /// Specifies which subgroup operations are supported.
    #[non_exhaustive]
    SubgroupFeatures = SubgroupFeatureFlags(u32);

    // TODO: document
    basic = BASIC,

    // TODO: document
    vote = VOTE,

    // TODO: document
    arithmetic = ARITHMETIC,

    // TODO: document
    ballot = BALLOT,

    // TODO: document
    shuffle = SHUFFLE,

    // TODO: document
    shuffle_relative = SHUFFLE_RELATIVE,

    // TODO: document
    clustered = CLUSTERED,

    // TODO: document
    quad = QUAD,

    // TODO: document
    partitioned = PARTITIONED_NV {
        device_extensions: [nv_shader_subgroup_partitioned],
    },
}

vulkan_enum! {
    /// Specifies how the device clips single point primitives.
    #[non_exhaustive]
    PointClippingBehavior = PointClippingBehavior(i32);

    /// Points are clipped if they lie outside any clip plane, both those bounding the view volume
    /// and user-defined clip planes.
    AllClipPlanes = ALL_CLIP_PLANES,

    /// Points are clipped only if they lie outside a user-defined clip plane.
    UserClipPlanesOnly = USER_CLIP_PLANES_ONLY,
}

vulkan_enum! {
    /// Specifies whether, and how, shader float controls can be set independently.
    #[non_exhaustive]
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

/// Error that can happen when retrieving properties of an external buffer.
#[derive(Clone, Debug)]
pub enum ExternalBufferPropertiesError {
    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },
}

impl Error for ExternalBufferPropertiesError {}

impl Display for ExternalBufferPropertiesError {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
        }
    }
}

impl From<RequirementNotMet> for ExternalBufferPropertiesError {
    #[inline]
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

/// Error that can happen when retrieving properties of an external fence or semaphore.
#[derive(Clone, Debug)]
pub enum ExternalFenceSemaphorePropertiesError {
    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },
}

impl Error for ExternalFenceSemaphorePropertiesError {}

impl Display for ExternalFenceSemaphorePropertiesError {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
        }
    }
}

impl From<RequirementNotMet> for ExternalFenceSemaphorePropertiesError {
    #[inline]
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

/// Error that can happen when retrieving format properties of an image.
#[derive(Clone, Debug)]
pub enum ImageFormatPropertiesError {
    /// Not enough memory.
    OomError(OomError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },
}

impl Error for ImageFormatPropertiesError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for ImageFormatPropertiesError {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory"),

            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
        }
    }
}

impl From<VulkanError> for ImageFormatPropertiesError {
    #[inline]
    fn from(err: VulkanError) -> Self {
        match err {
            err @ VulkanError::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ VulkanError::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<RequirementNotMet> for ImageFormatPropertiesError {
    #[inline]
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

/// Error that can happen when retrieving properties of a surface.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SurfacePropertiesError {
    /// Not enough memory.
    OomError(OomError),

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,

    // The given `SurfaceInfo` values are not supported for the surface by the physical device.
    NotSupported,

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The provided `queue_family_index` was not less than the number of queue families in the
    /// physical device.
    QueueFamilyIndexOutOfRange {
        queue_family_index: u32,
        queue_family_count: u32,
    },
}

impl Error for SurfacePropertiesError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl Display for SurfacePropertiesError {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(
                f,
                "not enough memory",
            ),
            Self::SurfaceLost => write!(
                f,
                "the surface is no longer valid",
            ),
            Self::NotSupported => write!(
                f,
                "the given `SurfaceInfo` values are not supported for the surface by the physical device",
            ),

            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),

            Self::QueueFamilyIndexOutOfRange {
                queue_family_index,
                queue_family_count,
            } => write!(
                f,
                "the provided `queue_family_index` ({}) was not less than the number of queue families in the physical device ({})",
                queue_family_index, queue_family_count,
            ),
        }
    }
}

impl From<OomError> for SurfacePropertiesError {
    #[inline]
    fn from(err: OomError) -> SurfacePropertiesError {
        Self::OomError(err)
    }
}

impl From<VulkanError> for SurfacePropertiesError {
    #[inline]
    fn from(err: VulkanError) -> SurfacePropertiesError {
        match err {
            err @ VulkanError::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ VulkanError::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            VulkanError::SurfaceLost => Self::SurfaceLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
