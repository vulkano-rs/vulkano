// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::{ExternalBufferInfo, ExternalBufferProperties},
    device::{DeviceExtensions, Features, FeaturesFfi, Properties, PropertiesFfi},
    format::{Format, FormatProperties},
    image::{ImageCreateFlags, ImageFormatInfo, ImageFormatProperties, ImageUsage},
    instance::{Instance, InstanceCreationError},
    macros::{vulkan_bitflags, vulkan_enum},
    swapchain::{
        ColorSpace, FullScreenExclusive, PresentMode, SupportedSurfaceTransforms, Surface,
        SurfaceApi, SurfaceCapabilities, SurfaceInfo,
    },
    sync::{ExternalSemaphoreInfo, ExternalSemaphoreProperties, PipelineStage},
    DeviceSize, OomError, Version, VulkanError, VulkanObject,
};
use std::{error::Error, ffi::CStr, fmt, hash::Hash, mem::MaybeUninit, ptr, sync::Arc};

#[derive(Clone, Debug)]
pub(crate) struct PhysicalDeviceInfo {
    handle: ash::vk::PhysicalDevice,
    api_version: Version,
    supported_extensions: DeviceExtensions,
    supported_features: Features,
    properties: Properties,
    memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
    queue_families: Vec<ash::vk::QueueFamilyProperties>,
}

pub(crate) fn init_physical_devices(
    instance: &Instance,
) -> Result<Vec<PhysicalDeviceInfo>, InstanceCreationError> {
    let fns = instance.fns();
    let instance_extensions = instance.enabled_extensions();

    let handles = unsafe {
        loop {
            let mut count = 0;
            (fns.v1_0.enumerate_physical_devices)(
                instance.internal_object(),
                &mut count,
                ptr::null_mut(),
            )
            .result()
            .map_err(VulkanError::from)?;

            let mut handles = Vec::with_capacity(count as usize);
            let result = (fns.v1_0.enumerate_physical_devices)(
                instance.internal_object(),
                &mut count,
                handles.as_mut_ptr(),
            );

            match result {
                ash::vk::Result::SUCCESS => {
                    handles.set_len(count as usize);
                    break handles;
                }
                ash::vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err).into()),
            }
        }
    };

    handles
        .into_iter()
        .enumerate()
        .map(|(_index, handle)| -> Result<_, InstanceCreationError> {
            let api_version = unsafe {
                let mut output = MaybeUninit::uninit();
                (fns.v1_0.get_physical_device_properties)(handle, output.as_mut_ptr());
                let api_version = Version::try_from(output.assume_init().api_version).unwrap();
                std::cmp::min(instance.max_api_version(), api_version)
            };

            let extension_properties = unsafe {
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

                    let mut properties = Vec::with_capacity(count as usize);
                    let result = (fns.v1_0.enumerate_device_extension_properties)(
                        handle,
                        ptr::null(),
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

            let supported_extensions = DeviceExtensions::from(
                extension_properties
                    .iter()
                    .map(|property| unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }),
            );

            let mut info = PhysicalDeviceInfo {
                handle,
                api_version,
                supported_extensions,
                supported_features: Default::default(),
                properties: Default::default(),
                memory_properties: Default::default(),
                queue_families: Default::default(),
            };

            // Get the remaining infos.
            // If possible, we use VK_KHR_get_physical_device_properties2.
            if api_version >= Version::V1_1
                || instance_extensions.khr_get_physical_device_properties2
            {
                init_info2(instance, &mut info)
            } else {
                init_info(instance, &mut info)
            };

            Ok(info)
        })
        .collect::<Result<_, _>>()
}

fn init_info(instance: &Instance, info: &mut PhysicalDeviceInfo) {
    let fns = instance.fns();

    info.supported_features = unsafe {
        let mut output = FeaturesFfi::default();
        (fns.v1_0.get_physical_device_features)(info.handle, &mut output.head_as_mut().features);
        Features::from(&output)
    };

    info.properties = unsafe {
        let mut output = PropertiesFfi::default();
        output.make_chain(
            info.api_version,
            &info.supported_extensions,
            instance.enabled_extensions(),
        );
        (fns.v1_0.get_physical_device_properties)(
            info.handle,
            &mut output.head_as_mut().properties,
        );
        Properties::from(&output)
    };

    info.memory_properties = unsafe {
        let mut output = MaybeUninit::uninit();
        (fns.v1_0.get_physical_device_memory_properties)(info.handle, output.as_mut_ptr());
        output.assume_init()
    };

    info.queue_families = unsafe {
        let mut num = 0;
        (fns.v1_0.get_physical_device_queue_family_properties)(
            info.handle,
            &mut num,
            ptr::null_mut(),
        );

        let mut families = Vec::with_capacity(num as usize);
        (fns.v1_0.get_physical_device_queue_family_properties)(
            info.handle,
            &mut num,
            families.as_mut_ptr(),
        );
        families.set_len(num as usize);
        families
    };
}

// TODO: Query extension-specific physical device properties, once a new instance extension is supported.
fn init_info2(instance: &Instance, info: &mut PhysicalDeviceInfo) {
    let fns = instance.fns();

    info.supported_features = unsafe {
        let mut output = FeaturesFfi::default();
        output.make_chain(
            info.api_version,
            &info.supported_extensions,
            instance.enabled_extensions(),
        );

        if instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_features2)(info.handle, output.head_as_mut());
        } else {
            (fns.khr_get_physical_device_properties2
                .get_physical_device_features2_khr)(info.handle, output.head_as_mut());
        }

        Features::from(&output)
    };

    info.properties = unsafe {
        let mut output = PropertiesFfi::default();
        output.make_chain(
            info.api_version,
            &info.supported_extensions,
            instance.enabled_extensions(),
        );

        if instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_properties2)(info.handle, output.head_as_mut());
        } else {
            (fns.khr_get_physical_device_properties2
                .get_physical_device_properties2_khr)(info.handle, output.head_as_mut());
        }

        Properties::from(&output)
    };

    info.memory_properties = unsafe {
        let mut output = ash::vk::PhysicalDeviceMemoryProperties2KHR::default();

        if instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_memory_properties2)(info.handle, &mut output);
        } else {
            (fns.khr_get_physical_device_properties2
                .get_physical_device_memory_properties2_khr)(info.handle, &mut output);
        }

        output.memory_properties
    };

    info.queue_families = unsafe {
        let mut num = 0;

        if instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_queue_family_properties2)(
                info.handle,
                &mut num,
                ptr::null_mut(),
            );
        } else {
            (fns.khr_get_physical_device_properties2
                .get_physical_device_queue_family_properties2_khr)(
                info.handle,
                &mut num,
                ptr::null_mut(),
            );
        }

        let mut families = vec![ash::vk::QueueFamilyProperties2::default(); num as usize];

        if instance.api_version() >= Version::V1_1 {
            (fns.v1_1.get_physical_device_queue_family_properties2)(
                info.handle,
                &mut num,
                families.as_mut_ptr(),
            );
        } else {
            (fns.khr_get_physical_device_properties2
                .get_physical_device_queue_family_properties2_khr)(
                info.handle,
                &mut num,
                families.as_mut_ptr(),
            );
        }

        families
            .into_iter()
            .map(|family| family.queue_family_properties)
            .collect()
    };
}

/// Represents one of the available devices on this machine.
///
/// This struct simply contains a pointer to an instance and a number representing the physical
/// device. You are therefore encouraged to pass this around by value instead of by reference.
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
/// for physical_device in PhysicalDevice::enumerate(&instance) {
///     print_infos(physical_device);
/// }
///
/// fn print_infos(dev: PhysicalDevice) {
///     println!("Name: {}", dev.properties().device_name);
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PhysicalDevice<'a> {
    instance: &'a Arc<Instance>,
    index: usize,
    info: &'a PhysicalDeviceInfo,
}

impl<'a> PhysicalDevice<'a> {
    /// Returns an iterator that enumerates the physical devices available.
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
    /// for physical_device in PhysicalDevice::enumerate(&instance) {
    ///     println!("Available device: {}", physical_device.properties().device_name);
    /// }
    /// ```
    #[inline]
    pub fn enumerate(
        instance: &'a Arc<Instance>,
    ) -> impl ExactSizeIterator<Item = PhysicalDevice<'a>> {
        instance
            .physical_device_infos
            .iter()
            .enumerate()
            .map(move |(index, info)| PhysicalDevice {
                instance,
                index,
                info,
            })
    }

    /// Returns a physical device from its index. Returns `None` if out of range.
    ///
    /// Indices range from 0 to the number of devices.
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
    /// let first_physical_device = PhysicalDevice::from_index(&instance, 0).unwrap();
    /// ```
    #[inline]
    pub fn from_index(instance: &'a Arc<Instance>, index: usize) -> Option<PhysicalDevice<'a>> {
        instance
            .physical_device_infos
            .get(index)
            .map(|info| PhysicalDevice {
                instance,
                index,
                info,
            })
    }

    /// Returns the instance corresponding to this physical device.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::device::physical::PhysicalDevice;
    ///
    /// fn do_something(physical_device: PhysicalDevice) {
    ///     let _loaded_extensions = physical_device.instance().enabled_extensions();
    ///     // ...
    /// }
    /// ```
    #[inline]
    pub fn instance(&self) -> &'a Arc<Instance> {
        self.instance
    }

    /// Returns the index of the physical device in the physical devices list.
    ///
    /// This index never changes and can be used later to retrieve a `PhysicalDevice` from an
    /// instance and an index.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the version of Vulkan supported by this device.
    ///
    /// Unlike the `api_version` property, which is the version reported by the device directly,
    /// this function returns the version the device can actually support, based on the instance's
    /// `max_api_version`.
    #[inline]
    pub fn api_version(&self) -> Version {
        self.info.api_version
    }

    /// Returns the extensions that are supported by this physical device.
    #[inline]
    pub fn supported_extensions(&self) -> &'a DeviceExtensions {
        &self.info.supported_extensions
    }

    /// Returns the properties reported by the device.
    #[inline]
    pub fn properties(&self) -> &'a Properties {
        &self.info.properties
    }

    /// Returns the features that are supported by this physical device.
    #[inline]
    pub fn supported_features(&self) -> &'a Features {
        &self.info.supported_features
    }

    /// Retrieves the external memory properties supported for buffers with a given configuration.
    ///
    /// Returns `None` if the instance API version is less than 1.1 and the
    /// [`khr_external_memory_capabilities`](crate::instance::InstanceExtensions::khr_external_memory_capabilities)
    /// extension is not enabled on the instance.
    pub fn external_buffer_properties(
        &self,
        info: ExternalBufferInfo,
    ) -> Option<ExternalBufferProperties> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_memory_capabilities)
        {
            return None;
        }

        /* Input */

        let ExternalBufferInfo {
            handle_type,
            usage,
            sparse,
            _ne: _,
        } = info;

        // VUID-VkPhysicalDeviceExternalBufferInfo-usage-parameter
        // TODO: usage.validate()?;

        assert!(!usage.is_empty());

        // VUID-VkPhysicalDeviceExternalBufferInfo-handleType-parameter
        // TODO: handle_type.validate()?;

        let external_buffer_info = ash::vk::PhysicalDeviceExternalBufferInfo {
            flags: sparse.map(Into::into).unwrap_or_default(),
            usage: usage.into(),
            handle_type: handle_type.into(),
            ..Default::default()
        };

        /* Output */

        let mut external_buffer_properties = ash::vk::ExternalBufferProperties::default();

        /* Call */

        unsafe {
            let fns = self.instance.fns();

            if self.instance.api_version() >= Version::V1_1 {
                (fns.v1_1.get_physical_device_external_buffer_properties)(
                    self.info.handle,
                    &external_buffer_info,
                    &mut external_buffer_properties,
                )
            } else {
                (fns.khr_external_memory_capabilities
                    .get_physical_device_external_buffer_properties_khr)(
                    self.info.handle,
                    &external_buffer_info,
                    &mut external_buffer_properties,
                );
            }
        }

        Some(ExternalBufferProperties {
            external_memory_properties: external_buffer_properties
                .external_memory_properties
                .into(),
        })
    }

    /// Retrieves the properties of a format when used by this physical device.
    pub fn format_properties(&self, format: Format) -> FormatProperties {
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

        unsafe {
            let fns = self.instance.fns();

            if self.api_version() >= Version::V1_1 {
                (fns.v1_1.get_physical_device_format_properties2)(
                    self.info.handle,
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
                    self.info.handle,
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

    /// Retrieves the external handle properties supported for semaphores with a given
    /// configuration.
    ///
    /// Returns `None` if the instance API version is less than 1.1 and the
    /// [`khr_external_semaphore_capabilities`](crate::instance::InstanceExtensions::khr_external_semaphore_capabilities)
    /// extension is not enabled on the instance.
    pub fn external_semaphore_properties(
        &self,
        info: ExternalSemaphoreInfo,
    ) -> Option<ExternalSemaphoreProperties> {
        if !(self.instance.api_version() >= Version::V1_1
            || self
                .instance
                .enabled_extensions()
                .khr_external_semaphore_capabilities)
        {
            return None;
        }

        /* Input */

        let ExternalSemaphoreInfo {
            handle_type,
            _ne: _,
        } = info;

        // VUID-VkPhysicalDeviceExternalSemaphoreInfo-handleType-parameter
        // TODO: handle_type.validate()?;

        let external_semaphore_info = ash::vk::PhysicalDeviceExternalSemaphoreInfo {
            handle_type: handle_type.into(),
            ..Default::default()
        };

        /* Output */

        let mut external_semaphore_properties = ash::vk::ExternalSemaphoreProperties::default();

        /* Call */

        unsafe {
            let fns = self.instance.fns();

            if self.instance.api_version() >= Version::V1_1 {
                (fns.v1_1.get_physical_device_external_semaphore_properties)(
                    self.info.handle,
                    &external_semaphore_info,
                    &mut external_semaphore_properties,
                )
            } else {
                (fns.khr_external_semaphore_capabilities
                    .get_physical_device_external_semaphore_properties_khr)(
                    self.info.handle,
                    &external_semaphore_info,
                    &mut external_semaphore_properties,
                );
            }
        }

        Some(ExternalSemaphoreProperties {
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
        })
    }

    /// Returns the properties supported for images with a given image configuration.
    ///
    /// `Some` is returned if the configuration is supported, `None` if it is not.
    ///
    /// # Panics
    ///
    /// - Panics if `image_format_info.format` is `None`.
    pub fn image_format_properties(
        &self,
        image_format_info: ImageFormatInfo,
    ) -> Result<Option<ImageFormatProperties>, OomError> {
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

        // VUID-VkPhysicalDeviceImageFormatInfo2-format-parameter
        // TODO: format.validate()?;

        // VUID-VkPhysicalDeviceImageFormatInfo2-imageType-parameter
        // TODO: image_type.validate()?;

        // VUID-VkPhysicalDeviceImageFormatInfo2-tiling-parameter
        // TODO: tiling.validate()?;

        // VUID-VkPhysicalDeviceImageFormatInfo2-usage-parameter
        // TODO: usage.validate()?;

        // VUID-VkPhysicalDeviceExternalImageFormatInfo-handleType-parameter
        // TODO: external_memory_handle_type.validate()?;

        // VUID-VkPhysicalDeviceImageViewImageFormatInfoEXT-imageViewType-parameter
        // TODO: image_view_type.validate()?;

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

        let mut external_image_format_info = if let Some(handle_type) = external_memory_handle_type
        {
            if !(self.api_version() >= Version::V1_1
                || self
                    .instance()
                    .enabled_extensions()
                    .khr_external_memory_capabilities)
            {
                // Can't query this, return unsupported
                return Ok(None);
            }

            Some(
                ash::vk::PhysicalDeviceExternalImageFormatInfo::builder()
                    .handle_type(handle_type.into()),
            )
        } else {
            None
        };

        if let Some(next) = external_image_format_info.as_mut() {
            format_info2 = format_info2.push_next(next);
        }

        let mut image_view_image_format_info = if let Some(image_view_type) = image_view_type {
            if !(self.supported_extensions().ext_filter_cubic
                || self.supported_extensions().img_filter_cubic)
            {
                // Can't query this, return unsupported
                return Ok(None);
            }

            if !image_view_type.is_compatible_with(image_type) {
                return Ok(None);
            }

            Some(
                ash::vk::PhysicalDeviceImageViewImageFormatInfoEXT::builder()
                    .image_view_type(image_view_type.into()),
            )
        } else {
            None
        };

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

        let result = unsafe {
            let fns = self.instance.fns();

            if self.api_version() >= Version::V1_1 {
                (fns.v1_1.get_physical_device_image_format_properties2)(
                    self.info.handle,
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
                    self.info.handle,
                    &format_info2.build(),
                    &mut image_format_properties2,
                )
            } else {
                // Can't query this, return unsupported
                if !format_info2.p_next.is_null() {
                    return Ok(None);
                }

                (fns.v1_0.get_physical_device_image_format_properties)(
                    self.info.handle,
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
            Err(err) => Err(err.into()),
        }
    }

    /// Builds an iterator that enumerates all the memory types on this physical device.
    #[inline]
    pub fn memory_types(&self) -> impl ExactSizeIterator<Item = MemoryType<'a>> {
        let physical_device = *self;
        self.info.memory_properties.memory_types
            [0..self.info.memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .map(move |(id, info)| MemoryType {
                physical_device,
                id: id as u32,
                info,
            })
    }

    /// Returns the memory type with the given index, or `None` if out of range.
    #[inline]
    pub fn memory_type_by_id(&self, id: u32) -> Option<MemoryType<'a>> {
        if id < self.info.memory_properties.memory_type_count {
            Some(MemoryType {
                physical_device: *self,
                id,
                info: &self.info.memory_properties.memory_types[id as usize],
            })
        } else {
            None
        }
    }

    /// Builds an iterator that enumerates all the memory heaps on this physical device.
    #[inline]
    pub fn memory_heaps(&self) -> impl ExactSizeIterator<Item = MemoryHeap<'a>> {
        let physical_device = *self;
        self.info.memory_properties.memory_heaps
            [0..self.info.memory_properties.memory_heap_count as usize]
            .iter()
            .enumerate()
            .map(move |(id, info)| MemoryHeap {
                physical_device,
                id: id as u32,
                info,
            })
    }

    /// Returns the memory heap with the given index, or `None` if out of range.
    #[inline]
    pub fn memory_heap_by_id(&self, id: u32) -> Option<MemoryHeap<'a>> {
        if id < self.info.memory_properties.memory_heap_count {
            Some(MemoryHeap {
                physical_device: *self,
                id,
                info: &self.info.memory_properties.memory_heaps[id as usize],
            })
        } else {
            None
        }
    }

    /// Builds an iterator that enumerates all the queue families on this physical device.
    #[inline]
    pub fn queue_families(&self) -> impl ExactSizeIterator<Item = QueueFamily<'a>> {
        let physical_device = *self;
        self.info
            .queue_families
            .iter()
            .enumerate()
            .map(move |(id, properties)| QueueFamily {
                physical_device,
                id: id as u32,
                properties,
            })
    }

    /// Returns the queue family with the given index, or `None` if out of range.
    #[inline]
    pub fn queue_family_by_id(&self, id: u32) -> Option<QueueFamily<'a>> {
        if (id as usize) < self.info.queue_families.len() {
            Some(QueueFamily {
                physical_device: *self,
                id,
                properties: &self.info.queue_families[id as usize],
            })
        } else {
            None
        }
    }

    /// Returns the capabilities that are supported by the physical device for the given surface.
    ///
    /// # Panic
    ///
    /// - Panics if the physical device and the surface don't belong to the same instance.
    pub fn surface_capabilities<W>(
        &self,
        surface: &Surface<W>,
        surface_info: SurfaceInfo,
    ) -> Result<SurfaceCapabilities, SurfacePropertiesError> {
        assert_eq!(
            self.instance.internal_object(),
            surface.instance().internal_object(),
        );

        /* Input */

        let SurfaceInfo {
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = surface_info;

        let mut surface_full_screen_exclusive_info =
            if self.supported_extensions().ext_full_screen_exclusive {
                Some(ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                    full_screen_exclusive: full_screen_exclusive.into(),
                    ..Default::default()
                })
            } else {
                if full_screen_exclusive != FullScreenExclusive::Default {
                    return Err(SurfacePropertiesError::NotSupported);
                }

                None
            };

        let mut surface_full_screen_exclusive_win32_info = if surface.api() == SurfaceApi::Win32
            && full_screen_exclusive == FullScreenExclusive::ApplicationControlled
        {
            if let Some(win32_monitor) = win32_monitor {
                Some(ash::vk::SurfaceFullScreenExclusiveWin32InfoEXT {
                    hmonitor: win32_monitor.0,
                    ..Default::default()
                })
            } else {
                return Err(SurfacePropertiesError::NotSupported);
            }
        } else {
            if win32_monitor.is_some() {
                return Err(SurfacePropertiesError::NotSupported);
            } else {
                None
            }
        };

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

        unsafe {
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
        }

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
    pub fn surface_formats<W>(
        &self,
        surface: &Surface<W>,
        surface_info: SurfaceInfo,
    ) -> Result<Vec<(Format, ColorSpace)>, SurfacePropertiesError> {
        assert_eq!(
            self.instance.internal_object(),
            surface.instance().internal_object(),
        );

        if self
            .instance
            .enabled_extensions()
            .khr_get_surface_capabilities2
        {
            let SurfaceInfo {
                full_screen_exclusive,
                win32_monitor,
                _ne: _,
            } = surface_info;

            let mut surface_full_screen_exclusive_info =
                if full_screen_exclusive != FullScreenExclusive::Default {
                    if !self.supported_extensions().ext_full_screen_exclusive {
                        return Err(SurfacePropertiesError::NotSupported);
                    }

                    Some(ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                        full_screen_exclusive: full_screen_exclusive.into(),
                        ..Default::default()
                    })
                } else {
                    None
                };

            let mut surface_full_screen_exclusive_win32_info = if surface.api() == SurfaceApi::Win32
                && full_screen_exclusive == FullScreenExclusive::ApplicationControlled
            {
                if let Some(win32_monitor) = win32_monitor {
                    Some(ash::vk::SurfaceFullScreenExclusiveWin32InfoEXT {
                        hmonitor: win32_monitor.0,
                        ..Default::default()
                    })
                } else {
                    return Err(SurfacePropertiesError::NotSupported);
                }
            } else {
                if win32_monitor.is_some() {
                    return Err(SurfacePropertiesError::NotSupported);
                } else {
                    None
                }
            };

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
                surface_info2.p_next =
                    surface_full_screen_exclusive_win32_info as *const _ as *const _;
            }

            let fns = self.instance.fns();

            let surface_format2s = unsafe {
                loop {
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
                        err => return Err(VulkanError::from(err).into()),
                    }
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
            if surface_info != SurfaceInfo::default() {
                return Ok(Vec::new());
            }

            let fns = self.instance.fns();

            let surface_formats = unsafe {
                loop {
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
                        err => return Err(VulkanError::from(err).into()),
                    }
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
    pub fn surface_present_modes<W>(
        &self,
        surface: &Surface<W>,
    ) -> Result<impl Iterator<Item = PresentMode>, SurfacePropertiesError> {
        assert_eq!(
            self.instance.internal_object(),
            surface.instance().internal_object(),
        );

        let fns = self.instance.fns();

        let modes = unsafe {
            loop {
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
                    err => return Err(VulkanError::from(err).into()),
                }
            }
        };

        debug_assert!(!modes.is_empty());
        debug_assert!(modes.iter().any(|&m| m == ash::vk::PresentModeKHR::FIFO));

        Ok(modes
            .into_iter()
            .filter_map(|mode_vk| mode_vk.try_into().ok()))
    }
}

unsafe impl<'a> VulkanObject for PhysicalDevice<'a> {
    type Object = ash::vk::PhysicalDevice;

    #[inline]
    fn internal_object(&self) -> ash::vk::PhysicalDevice {
        self.info.handle
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

/// Represents a memory type in a physical device.
#[derive(Debug, Copy, Clone)]
pub struct MemoryType<'a> {
    physical_device: PhysicalDevice<'a>,
    id: u32,
    info: &'a ash::vk::MemoryType,
}

impl<'a> MemoryType<'a> {
    /// Returns the physical device associated to this memory type.
    #[inline]
    pub fn physical_device(&self) -> PhysicalDevice<'a> {
        self.physical_device
    }

    /// Returns the identifier of this memory type within the physical device.
    #[inline]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the heap that corresponds to this memory type.
    #[inline]
    pub fn heap(&self) -> MemoryHeap<'a> {
        self.physical_device
            .memory_heap_by_id(self.info.heap_index)
            .unwrap()
    }

    /// Returns true if the memory type is located on the device, which means that it's the most
    /// efficient for GPU accesses.
    #[inline]
    pub fn is_device_local(&self) -> bool {
        self.info
            .property_flags
            .intersects(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    /// Returns true if the memory type can be accessed by the host.
    #[inline]
    pub fn is_host_visible(&self) -> bool {
        self.info
            .property_flags
            .intersects(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
    }

    /// Returns true if modifications made by the host or the GPU on this memory type are
    /// instantaneously visible to the other party. False means that changes have to be flushed.
    ///
    /// You don't need to worry about this, as this library handles that for you.
    #[inline]
    pub fn is_host_coherent(&self) -> bool {
        self.info
            .property_flags
            .intersects(ash::vk::MemoryPropertyFlags::HOST_COHERENT)
    }

    /// Returns true if memory of this memory type is cached by the host. Host memory accesses to
    /// cached memory is faster than for uncached memory. However you are not guaranteed that it
    /// is coherent.
    #[inline]
    pub fn is_host_cached(&self) -> bool {
        self.info
            .property_flags
            .intersects(ash::vk::MemoryPropertyFlags::HOST_CACHED)
    }

    /// Returns true if allocations made to this memory type is lazy.
    ///
    /// This means that no actual allocation is performed. Instead memory is automatically
    /// allocated by the Vulkan implementation.
    ///
    /// Memory of this type can only be used on images created with a certain flag. Memory of this
    /// type is never host-visible.
    #[inline]
    pub fn is_lazily_allocated(&self) -> bool {
        self.info
            .property_flags
            .intersects(ash::vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
    }

    /// Returns whether the memory type is protected.
    #[inline]
    pub fn is_protected(&self) -> bool {
        self.info
            .property_flags
            .intersects(ash::vk::MemoryPropertyFlags::PROTECTED)
    }
}

/// Represents a memory heap in a physical device.
#[derive(Debug, Copy, Clone)]
pub struct MemoryHeap<'a> {
    physical_device: PhysicalDevice<'a>,
    id: u32,
    info: &'a ash::vk::MemoryHeap,
}

impl<'a> MemoryHeap<'a> {
    /// Returns the physical device associated to this memory heap.
    #[inline]
    pub fn physical_device(&self) -> PhysicalDevice<'a> {
        self.physical_device
    }

    /// Returns the identifier of this memory heap within the physical device.
    #[inline]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the size in bytes on this heap.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.info.size
    }

    /// Returns true if the heap is local to the GPU.
    #[inline]
    pub fn is_device_local(&self) -> bool {
        !(self.info.flags & ash::vk::MemoryHeapFlags::DEVICE_LOCAL).is_empty()
    }

    /// Returns true if the heap is multi-instance enabled, that is allocation from such
    /// heap will replicate to each physical-device's instance of heap.
    #[inline]
    pub fn is_multi_instance(&self) -> bool {
        !(self.info.flags & ash::vk::MemoryHeapFlags::MULTI_INSTANCE).is_empty()
    }
}

/// Represents a queue family in a physical device.
///
/// A queue family is group of one or multiple queues. All queues of one family have the same
/// characteristics.
#[derive(Debug, Copy, Clone)]
pub struct QueueFamily<'a> {
    physical_device: PhysicalDevice<'a>,
    id: u32,
    properties: &'a ash::vk::QueueFamilyProperties,
}

impl<'a> QueueFamily<'a> {
    /// Returns the physical device associated to this queue family.
    #[inline]
    pub fn physical_device(&self) -> PhysicalDevice<'a> {
        self.physical_device
    }

    /// Returns the identifier of this queue family within the physical device.
    #[inline]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the number of queues that belong to this family.
    ///
    /// Guaranteed to be at least 1 (or else that family wouldn't exist).
    #[inline]
    pub fn queues_count(&self) -> usize {
        self.properties.queue_count as usize
    }

    /// If timestamps are supported, returns the number of bits supported by timestamp operations.
    /// The returned value will be in the range 36..64.
    /// If timestamps are not supported, returns None.
    #[inline]
    pub fn timestamp_valid_bits(&self) -> Option<u32> {
        let value = self.properties.timestamp_valid_bits;
        if value == 0 {
            None
        } else {
            Some(value)
        }
    }

    /// Returns the minimum granularity supported for image transfers in terms
    /// of `[width, height, depth]`
    #[inline]
    pub fn min_image_transfer_granularity(&self) -> [u32; 3] {
        let granularity = &self.properties.min_image_transfer_granularity;
        [granularity.width, granularity.height, granularity.depth]
    }

    /// Returns `true` if queues of this family can execute graphics operations.
    #[inline]
    pub fn supports_graphics(&self) -> bool {
        self.properties
            .queue_flags
            .contains(ash::vk::QueueFlags::GRAPHICS)
    }

    /// Returns `true` if queues of this family can execute compute operations.
    #[inline]
    pub fn supports_compute(&self) -> bool {
        self.properties
            .queue_flags
            .contains(ash::vk::QueueFlags::COMPUTE)
    }

    /// Returns `true` if queues of this family can execute transfer operations.
    /// > **Note**: While all queues that can perform graphics or compute operations can implicitly perform
    /// > transfer operations, graphics & compute queues only optionally indicate support for tranfers.
    /// > Many discrete cards will have one queue family that exclusively sets the VK_QUEUE_TRANSFER_BIT
    /// > to indicate a special relationship with the DMA module and more efficient transfers.
    #[inline]
    pub fn explicitly_supports_transfers(&self) -> bool {
        self.properties
            .queue_flags
            .contains(ash::vk::QueueFlags::TRANSFER)
    }

    /// Returns `true` if queues of this family can execute sparse resources binding operations.
    #[inline]
    pub fn supports_sparse_binding(&self) -> bool {
        self.properties
            .queue_flags
            .contains(ash::vk::QueueFlags::SPARSE_BINDING)
    }

    /// Returns `true` if the queues of this family support a particular pipeline stage.
    #[inline]
    pub fn supports_stage(&self, stage: PipelineStage) -> bool {
        self.properties
            .queue_flags
            .contains(stage.required_queue_flags())
    }

    /// Returns whether queues of this family can draw on the given surface.
    pub fn supports_surface<W>(
        &self,
        surface: &Surface<W>,
    ) -> Result<bool, SurfacePropertiesError> {
        unsafe {
            let fns = self.physical_device.instance.fns();

            let mut output = MaybeUninit::uninit();
            (fns.khr_surface.get_physical_device_surface_support_khr)(
                self.physical_device.internal_object(),
                self.id,
                surface.internal_object(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            Ok(output.assume_init() != 0)
        }
    }
}

impl<'a> PartialEq for QueueFamily<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.physical_device.internal_object() == other.physical_device.internal_object()
    }
}

impl<'a> Eq for QueueFamily<'a> {}

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

impl fmt::Debug for ConformanceVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl fmt::Display for ConformanceVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self, formatter)
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
        extensions: [nv_shader_subgroup_partitioned],
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

/// Error that can happen when retrieving properties of a surface.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SurfacePropertiesError {
    /// Not enough memory.
    OomError(OomError),

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,

    // The given `SurfaceInfo` values are not supported for the surface by the physical device.
    NotSupported,
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

impl fmt::Display for SurfacePropertiesError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::OomError(_) => "not enough memory",
                Self::SurfaceLost => "the surface is no longer valid",
                Self::NotSupported => "the given `SurfaceInfo` values are not supported for the surface by the physical device",
            }
        )
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
