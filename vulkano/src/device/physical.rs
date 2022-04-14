// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::{BufferUsage, ExternalBufferInfo, ExternalBufferProperties},
    check_errors,
    device::{DeviceExtensions, Features, FeaturesFfi, Properties, PropertiesFfi},
    format::{Format, FormatProperties},
    image::{ImageCreateFlags, ImageFormatInfo, ImageFormatProperties, ImageUsage},
    instance::{Instance, InstanceCreationError},
    swapchain::{
        ColorSpace, FullScreenExclusive, PresentMode, SupportedSurfaceTransforms, Surface,
        SurfaceApi, SurfaceCapabilities, SurfaceInfo,
    },
    sync::{ExternalSemaphoreInfo, ExternalSemaphoreProperties, PipelineStage},
    DeviceSize, Error, OomError, Version, VulkanObject,
};
use std::{error, ffi::CStr, fmt, hash::Hash, mem::MaybeUninit, ptr, sync::Arc};

#[derive(Clone, Debug)]
pub(crate) struct PhysicalDeviceInfo {
    handle: ash::vk::PhysicalDevice,
    api_version: Version,
    supported_extensions: DeviceExtensions,
    required_extensions: DeviceExtensions,
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

    let handles: Vec<ash::vk::PhysicalDevice> = unsafe {
        let mut num = 0;
        check_errors(fns.v1_0.enumerate_physical_devices(
            instance.internal_object(),
            &mut num,
            ptr::null_mut(),
        ))?;

        let mut handles = Vec::with_capacity(num as usize);
        check_errors(fns.v1_0.enumerate_physical_devices(
            instance.internal_object(),
            &mut num,
            handles.as_mut_ptr(),
        ))?;
        handles.set_len(num as usize);
        handles
    };

    Ok(handles
        .into_iter()
        .enumerate()
        .map(|(index, handle)| -> Result<_, InstanceCreationError> {
            let api_version = unsafe {
                let mut output = MaybeUninit::uninit();
                fns.v1_0
                    .get_physical_device_properties(handle, output.as_mut_ptr());
                let api_version = Version::try_from(output.assume_init().api_version).unwrap();
                std::cmp::min(instance.max_api_version(), api_version)
            };

            let extension_properties: Vec<ash::vk::ExtensionProperties> = unsafe {
                let mut num = 0;
                check_errors(fns.v1_0.enumerate_device_extension_properties(
                    handle,
                    ptr::null(),
                    &mut num,
                    ptr::null_mut(),
                ))?;

                let mut properties = Vec::with_capacity(num as usize);
                check_errors(fns.v1_0.enumerate_device_extension_properties(
                    handle,
                    ptr::null(),
                    &mut num,
                    properties.as_mut_ptr(),
                ))?;
                properties.set_len(num as usize);
                properties
            };

            let supported_extensions = DeviceExtensions::from(
                extension_properties
                    .iter()
                    .map(|property| unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }),
            );

            let required_extensions = supported_extensions
                .intersection(&DeviceExtensions::required_if_supported_extensions());

            let mut info = PhysicalDeviceInfo {
                handle,
                api_version,
                supported_extensions,
                required_extensions,
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
        .collect::<Result<_, _>>()?)
}

fn init_info(instance: &Instance, info: &mut PhysicalDeviceInfo) {
    let fns = instance.fns();

    info.supported_features = unsafe {
        let mut output = FeaturesFfi::default();
        fns.v1_0
            .get_physical_device_features(info.handle, &mut output.head_as_mut().features);
        Features::from(&output)
    };

    info.properties = unsafe {
        let mut output = PropertiesFfi::default();
        output.make_chain(
            info.api_version,
            &info.supported_extensions,
            instance.enabled_extensions(),
        );
        fns.v1_0
            .get_physical_device_properties(info.handle, &mut output.head_as_mut().properties);
        Properties::from(&output)
    };

    info.memory_properties = unsafe {
        let mut output = MaybeUninit::uninit();
        fns.v1_0
            .get_physical_device_memory_properties(info.handle, output.as_mut_ptr());
        output.assume_init()
    };

    info.queue_families = unsafe {
        let mut num = 0;
        fns.v1_0.get_physical_device_queue_family_properties(
            info.handle,
            &mut num,
            ptr::null_mut(),
        );

        let mut families = Vec::with_capacity(num as usize);
        fns.v1_0.get_physical_device_queue_family_properties(
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
            fns.v1_1
                .get_physical_device_features2(info.handle, output.head_as_mut());
        } else {
            fns.khr_get_physical_device_properties2
                .get_physical_device_features2_khr(info.handle, output.head_as_mut());
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
            fns.v1_1
                .get_physical_device_properties2(info.handle, output.head_as_mut());
        } else {
            fns.khr_get_physical_device_properties2
                .get_physical_device_properties2_khr(info.handle, output.head_as_mut());
        }

        Properties::from(&output)
    };

    info.memory_properties = unsafe {
        let mut output = ash::vk::PhysicalDeviceMemoryProperties2KHR::default();

        if instance.api_version() >= Version::V1_1 {
            fns.v1_1
                .get_physical_device_memory_properties2(info.handle, &mut output);
        } else {
            fns.khr_get_physical_device_properties2
                .get_physical_device_memory_properties2_khr(info.handle, &mut output);
        }

        output.memory_properties
    };

    info.queue_families = unsafe {
        let mut num = 0;

        if instance.api_version() >= Version::V1_1 {
            fns.v1_1.get_physical_device_queue_family_properties2(
                info.handle,
                &mut num,
                ptr::null_mut(),
            );
        } else {
            fns.khr_get_physical_device_properties2
                .get_physical_device_queue_family_properties2_khr(
                    info.handle,
                    &mut num,
                    ptr::null_mut(),
                );
        }

        let mut families = vec![ash::vk::QueueFamilyProperties2::default(); num as usize];

        if instance.api_version() >= Version::V1_1 {
            fns.v1_1.get_physical_device_queue_family_properties2(
                info.handle,
                &mut num,
                families.as_mut_ptr(),
            );
        } else {
            fns.khr_get_physical_device_properties2
                .get_physical_device_queue_family_properties2_khr(
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
/// # use vulkano::instance::Instance;
/// # use vulkano::instance::InstanceExtensions;
/// # use vulkano::instance::VulkanLibrary;
/// # use vulkano::Version;
/// use vulkano::device::physical::PhysicalDevice;
///
/// let lib = VulkanLibrary::default();
/// let instance = Instance::new(lib, Default::default()).unwrap();
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
    /// # use vulkano::instance::Instance;
    /// # use vulkano::instance::InstanceExtensions;
    /// # use vulkano::instance::VulkanLibrary;
    /// # use vulkano::Version;
    /// use vulkano::device::physical::PhysicalDevice;
    ///
    /// let lib = VulkanLibrary::default();
    /// let instance = Instance::new(lib, Default::default()).unwrap();
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
    /// use vulkano::instance::Instance;
    /// use vulkano::instance::InstanceExtensions;
    /// use vulkano::instance::VulkanLibrary;
    /// use vulkano::device::physical::PhysicalDevice;
    /// use vulkano::Version;
    ///
    /// let lib = VulkanLibrary::default();
    /// let instance = Instance::new(lib, Default::default()).unwrap();
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
        &self.instance
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
    /// this function returns the version the device can actually support, based on the instance's,
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

    /// Returns the extensions that must be enabled as a minimum when creating a `Device` from this
    /// physical device.
    pub fn required_extensions(&self) -> &'a DeviceExtensions {
        &self.info.required_extensions
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

        assert!(usage != BufferUsage::none());

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
                fns.v1_1.get_physical_device_external_buffer_properties(
                    self.info.handle,
                    &external_buffer_info,
                    &mut external_buffer_properties,
                )
            } else {
                fns.khr_external_memory_capabilities
                    .get_physical_device_external_buffer_properties_khr(
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
                fns.v1_1.get_physical_device_format_properties2(
                    self.info.handle,
                    format.into(),
                    &mut format_properties2,
                );
            } else if self
                .instance
                .enabled_extensions()
                .khr_get_physical_device_properties2
            {
                fns.khr_get_physical_device_properties2
                    .get_physical_device_format_properties2_khr(
                        self.info.handle,
                        format.into(),
                        &mut format_properties2,
                    );
            } else {
                fns.v1_0.get_physical_device_format_properties(
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
                fns.v1_1.get_physical_device_external_semaphore_properties(
                    self.info.handle,
                    &external_semaphore_info,
                    &mut external_semaphore_properties,
                )
            } else {
                fns.khr_external_semaphore_capabilities
                    .get_physical_device_external_semaphore_properties_khr(
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

        let flags = ImageCreateFlags {
            mutable_format,
            cube_compatible,
            array_2d_compatible,
            block_texel_view_compatible,
            ..ImageCreateFlags::none()
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
            if !self.supported_extensions().ext_filter_cubic {
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

            check_errors(if self.api_version() >= Version::V1_1 {
                fns.v1_1.get_physical_device_image_format_properties2(
                    self.info.handle,
                    &format_info2.build(),
                    &mut image_format_properties2,
                )
            } else if self
                .instance
                .enabled_extensions()
                .khr_get_physical_device_properties2
            {
                fns.khr_get_physical_device_properties2
                    .get_physical_device_image_format_properties2_khr(
                        self.info.handle,
                        &format_info2.build(),
                        &mut image_format_properties2,
                    )
            } else {
                // Can't query this, return unsupported
                if !format_info2.p_next.is_null() {
                    return Ok(None);
                }

                fns.v1_0.get_physical_device_image_format_properties(
                    self.info.handle,
                    format_info2.format,
                    format_info2.ty,
                    format_info2.tiling,
                    format_info2.usage,
                    format_info2.flags,
                    &mut image_format_properties2.image_format_properties,
                )
            })
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
            Err(Error::FormatNotSupported) => Ok(None),
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
                check_errors(
                    fns.khr_get_surface_capabilities2
                        .get_physical_device_surface_capabilities2_khr(
                            self.internal_object(),
                            &surface_info2,
                            &mut surface_capabilities2,
                        ),
                )?;
            } else {
                check_errors(
                    fns.khr_surface
                        .get_physical_device_surface_capabilities_khr(
                            self.internal_object(),
                            surface_info2.surface,
                            &mut surface_capabilities2.surface_capabilities,
                        ),
                )?;
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

            let mut surface_format2s;

            unsafe {
                let fns = self.instance.fns();

                let mut num = 0;
                check_errors(
                    fns.khr_get_surface_capabilities2
                        .get_physical_device_surface_formats2_khr(
                            self.internal_object(),
                            &surface_info2,
                            &mut num,
                            ptr::null_mut(),
                        ),
                )?;

                surface_format2s = vec![ash::vk::SurfaceFormat2KHR::default(); num as usize];
                check_errors(
                    fns.khr_get_surface_capabilities2
                        .get_physical_device_surface_formats2_khr(
                            self.internal_object(),
                            &surface_info2,
                            &mut num,
                            surface_format2s.as_mut_ptr(),
                        ),
                )?;
                surface_format2s.set_len(num as usize);
            }

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

            let mut surface_formats;

            unsafe {
                let fns = self.instance.fns();

                let mut num = 0;
                check_errors(fns.khr_surface.get_physical_device_surface_formats_khr(
                    self.internal_object(),
                    surface.internal_object(),
                    &mut num,
                    ptr::null_mut(),
                ))?;

                surface_formats = Vec::with_capacity(num as usize);
                check_errors(fns.khr_surface.get_physical_device_surface_formats_khr(
                    self.internal_object(),
                    surface.internal_object(),
                    &mut num,
                    surface_formats.as_mut_ptr(),
                ))?;
                surface_formats.set_len(num as usize);
            }

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

        let modes = unsafe {
            let fns = self.instance.fns();

            let mut num = 0;
            check_errors(
                fns.khr_surface
                    .get_physical_device_surface_present_modes_khr(
                        self.internal_object(),
                        surface.internal_object(),
                        &mut num,
                        ptr::null_mut(),
                    ),
            )?;

            let mut modes = Vec::with_capacity(num as usize);
            check_errors(
                fns.khr_surface
                    .get_physical_device_surface_present_modes_khr(
                        self.internal_object(),
                        surface.internal_object(),
                        &mut num,
                        modes.as_mut_ptr(),
                    ),
            )?;
            modes.set_len(num as usize);
            modes
        };

        debug_assert!(modes.len() > 0);
        debug_assert!(modes
            .iter()
            .find(|&&m| m == ash::vk::PresentModeKHR::FIFO)
            .is_some());

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

/// Type of a physical device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(i32)]
pub enum PhysicalDeviceType {
    /// The device is an integrated GPU.
    IntegratedGpu = ash::vk::PhysicalDeviceType::INTEGRATED_GPU.as_raw(),
    /// The device is a discrete GPU.
    DiscreteGpu = ash::vk::PhysicalDeviceType::DISCRETE_GPU.as_raw(),
    /// The device is a virtual GPU.
    VirtualGpu = ash::vk::PhysicalDeviceType::VIRTUAL_GPU.as_raw(),
    /// The device is a CPU.
    Cpu = ash::vk::PhysicalDeviceType::CPU.as_raw(),
    /// The device is something else.
    Other = ash::vk::PhysicalDeviceType::OTHER.as_raw(),
}

/// VkPhysicalDeviceType::Other is represented as 0
impl Default for PhysicalDeviceType {
    fn default() -> Self {
        PhysicalDeviceType::Other
    }
}

impl TryFrom<ash::vk::PhysicalDeviceType> for PhysicalDeviceType {
    type Error = ();

    #[inline]
    fn try_from(val: ash::vk::PhysicalDeviceType) -> Result<Self, Self::Error> {
        match val {
            ash::vk::PhysicalDeviceType::INTEGRATED_GPU => Ok(Self::IntegratedGpu),
            ash::vk::PhysicalDeviceType::DISCRETE_GPU => Ok(Self::DiscreteGpu),
            ash::vk::PhysicalDeviceType::VIRTUAL_GPU => Ok(Self::VirtualGpu),
            ash::vk::PhysicalDeviceType::CPU => Ok(Self::Cpu),
            ash::vk::PhysicalDeviceType::OTHER => Ok(Self::Other),
            _ => Err(()),
        }
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
        let ref granularity = self.properties.min_image_transfer_granularity;
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
            check_errors(fns.khr_surface.get_physical_device_surface_support_khr(
                self.physical_device.internal_object(),
                self.id,
                surface.internal_object(),
                output.as_mut_ptr(),
            ))?;
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

/// An identifier for the driver of a physical device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum DriverId {
    AMDProprietary = ash::vk::DriverId::AMD_PROPRIETARY.as_raw(),
    AMDOpenSource = ash::vk::DriverId::AMD_OPEN_SOURCE.as_raw(),
    MesaRADV = ash::vk::DriverId::MESA_RADV.as_raw(),
    NvidiaProprietary = ash::vk::DriverId::NVIDIA_PROPRIETARY.as_raw(),
    IntelProprietaryWindows = ash::vk::DriverId::INTEL_PROPRIETARY_WINDOWS.as_raw(),
    IntelOpenSourceMesa = ash::vk::DriverId::INTEL_OPEN_SOURCE_MESA.as_raw(),
    ImaginationProprietary = ash::vk::DriverId::IMAGINATION_PROPRIETARY.as_raw(),
    QualcommProprietary = ash::vk::DriverId::QUALCOMM_PROPRIETARY.as_raw(),
    ARMProprietary = ash::vk::DriverId::ARM_PROPRIETARY.as_raw(),
    GoogleSwiftshader = ash::vk::DriverId::GOOGLE_SWIFTSHADER.as_raw(),
    GGPProprietary = ash::vk::DriverId::GGP_PROPRIETARY.as_raw(),
    BroadcomProprietary = ash::vk::DriverId::BROADCOM_PROPRIETARY.as_raw(),
    MesaLLVMpipe = ash::vk::DriverId::MESA_LLVMPIPE.as_raw(),
    MoltenVK = ash::vk::DriverId::MOLTENVK.as_raw(),
}

impl TryFrom<ash::vk::DriverId> for DriverId {
    type Error = ();

    #[inline]
    fn try_from(val: ash::vk::DriverId) -> Result<Self, Self::Error> {
        match val {
            ash::vk::DriverId::AMD_PROPRIETARY => Ok(Self::AMDProprietary),
            ash::vk::DriverId::AMD_OPEN_SOURCE => Ok(Self::AMDOpenSource),
            ash::vk::DriverId::MESA_RADV => Ok(Self::MesaRADV),
            ash::vk::DriverId::NVIDIA_PROPRIETARY => Ok(Self::NvidiaProprietary),
            ash::vk::DriverId::INTEL_PROPRIETARY_WINDOWS => Ok(Self::IntelProprietaryWindows),
            ash::vk::DriverId::INTEL_OPEN_SOURCE_MESA => Ok(Self::IntelOpenSourceMesa),
            ash::vk::DriverId::IMAGINATION_PROPRIETARY => Ok(Self::ImaginationProprietary),
            ash::vk::DriverId::QUALCOMM_PROPRIETARY => Ok(Self::QualcommProprietary),
            ash::vk::DriverId::ARM_PROPRIETARY => Ok(Self::ARMProprietary),
            ash::vk::DriverId::GOOGLE_SWIFTSHADER => Ok(Self::GoogleSwiftshader),
            ash::vk::DriverId::GGP_PROPRIETARY => Ok(Self::GGPProprietary),
            ash::vk::DriverId::BROADCOM_PROPRIETARY => Ok(Self::BroadcomProprietary),
            ash::vk::DriverId::MESA_LLVMPIPE => Ok(Self::MesaLLVMpipe),
            ash::vk::DriverId::MOLTENVK => Ok(Self::MoltenVK),
            _ => Err(()),
        }
    }
}

/// Specifies which subgroup operations are supported.
#[derive(Clone, Copy, Debug)]
pub struct SubgroupFeatures {
    pub basic: bool,
    pub vote: bool,
    pub arithmetic: bool,
    pub ballot: bool,
    pub shuffle: bool,
    pub shuffle_relative: bool,
    pub clustered: bool,
    pub quad: bool,
    pub partitioned: bool,

    pub _ne: crate::NonExhaustive,
}

impl From<ash::vk::SubgroupFeatureFlags> for SubgroupFeatures {
    #[inline]
    fn from(val: ash::vk::SubgroupFeatureFlags) -> Self {
        Self {
            basic: val.intersects(ash::vk::SubgroupFeatureFlags::BASIC),
            vote: val.intersects(ash::vk::SubgroupFeatureFlags::VOTE),
            arithmetic: val.intersects(ash::vk::SubgroupFeatureFlags::ARITHMETIC),
            ballot: val.intersects(ash::vk::SubgroupFeatureFlags::BALLOT),
            shuffle: val.intersects(ash::vk::SubgroupFeatureFlags::SHUFFLE),
            shuffle_relative: val.intersects(ash::vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE),
            clustered: val.intersects(ash::vk::SubgroupFeatureFlags::CLUSTERED),
            quad: val.intersects(ash::vk::SubgroupFeatureFlags::QUAD),
            partitioned: val.intersects(ash::vk::SubgroupFeatureFlags::PARTITIONED_NV),

            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Specifies how the device clips single point primitives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum PointClippingBehavior {
    /// Points are clipped if they lie outside any clip plane, both those bounding the view volume
    /// and user-defined clip planes.
    AllClipPlanes = ash::vk::PointClippingBehavior::ALL_CLIP_PLANES.as_raw(),
    /// Points are clipped only if they lie outside a user-defined clip plane.
    UserClipPlanesOnly = ash::vk::PointClippingBehavior::USER_CLIP_PLANES_ONLY.as_raw(),
}

impl TryFrom<ash::vk::PointClippingBehavior> for PointClippingBehavior {
    type Error = ();

    #[inline]
    fn try_from(val: ash::vk::PointClippingBehavior) -> Result<Self, Self::Error> {
        match val {
            ash::vk::PointClippingBehavior::ALL_CLIP_PLANES => Ok(Self::AllClipPlanes),
            ash::vk::PointClippingBehavior::USER_CLIP_PLANES_ONLY => Ok(Self::UserClipPlanesOnly),
            _ => Err(()),
        }
    }
}

/// Specifies whether, and how, shader float controls can be set independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ShaderFloatControlsIndependence {
    Float32Only = ash::vk::ShaderFloatControlsIndependence::TYPE_32_ONLY.as_raw(),
    All = ash::vk::ShaderFloatControlsIndependence::ALL.as_raw(),
    None = ash::vk::ShaderFloatControlsIndependence::NONE.as_raw(),
}

impl TryFrom<ash::vk::ShaderFloatControlsIndependence> for ShaderFloatControlsIndependence {
    type Error = ();

    #[inline]
    fn try_from(val: ash::vk::ShaderFloatControlsIndependence) -> Result<Self, Self::Error> {
        match val {
            ash::vk::ShaderFloatControlsIndependence::TYPE_32_ONLY => Ok(Self::Float32Only),
            ash::vk::ShaderFloatControlsIndependence::ALL => Ok(Self::All),
            ash::vk::ShaderFloatControlsIndependence::NONE => Ok(Self::None),
            _ => Err(()),
        }
    }
}

/// Specifies shader core properties.
#[derive(Clone, Copy, Debug)]
pub struct ShaderCoreProperties {}

impl From<ash::vk::ShaderCorePropertiesFlagsAMD> for ShaderCoreProperties {
    #[inline]
    fn from(val: ash::vk::ShaderCorePropertiesFlagsAMD) -> Self {
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

impl error::Error for SurfacePropertiesError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
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

impl From<Error> for SurfacePropertiesError {
    #[inline]
    fn from(err: Error) -> SurfacePropertiesError {
        match err {
            err @ Error::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            Error::SurfaceLost => Self::SurfaceLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
