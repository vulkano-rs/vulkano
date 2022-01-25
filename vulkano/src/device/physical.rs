// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::device::{DeviceExtensions, Features, FeaturesFfi, Properties, PropertiesFfi};
use crate::format::Format;
use crate::image::view::ImageViewType;
use crate::image::{ImageCreateFlags, ImageTiling, ImageType, ImageUsage, SampleCounts};
use crate::instance::{Instance, InstanceCreationError};
use crate::memory::ExternalMemoryHandleType;
use crate::sync::PipelineStage;
use crate::Version;
use crate::VulkanObject;
use crate::{check_errors, OomError};
use crate::{DeviceSize, Error};
use std::ffi::CStr;
use std::fmt;
use std::hash::Hash;
use std::mem::MaybeUninit;
use std::ops::BitOr;
use std::ptr;
use std::sync::Arc;

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
/// # use vulkano::Version;
/// use vulkano::device::physical::PhysicalDevice;
///
/// # let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None).unwrap();
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
    /// # use vulkano::Version;
    /// use vulkano::device::physical::PhysicalDevice;
    ///
    /// # let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None).unwrap();
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
    /// use vulkano::device::physical::PhysicalDevice;
    /// use vulkano::Version;
    ///
    /// let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None).unwrap();
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

    /// Retrieves the properties of a format when used by this physical device.
    pub fn format_properties(&self, format: Format) -> FormatProperties {
        let mut format_properties2 = ash::vk::FormatProperties2::default();
        let mut format_properties3 = if self.supported_extensions().khr_format_feature_flags2 {
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
            },
        }
    }

    /// Returns the properties supported for images with a given image configuration.
    ///
    /// `Some` is returned if the configuration is supported, `None` if it is not.
    pub fn image_format_properties(
        &self,
        format: Format,
        ty: ImageType,
        tiling: ImageTiling,
        usage: ImageUsage,
        flags: ImageCreateFlags,
        external_memory_handle_type: Option<ExternalMemoryHandleType>,
        image_view_type: Option<ImageViewType>,
    ) -> Result<Option<ImageFormatProperties>, OomError> {
        /* Input */

        let mut format_info2 = ash::vk::PhysicalDeviceImageFormatInfo2::builder()
            .format(format.into())
            .ty(ty.into())
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

            if !image_view_type.is_compatible_with(ty) {
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
        let mut filter_cubic_image_view_image_format_properties =
            ash::vk::FilterCubicImageViewImageFormatPropertiesEXT::default();

        if image_view_type.is_some() {
            filter_cubic_image_view_image_format_properties.p_next =
                image_format_properties2.p_next;
            image_format_properties2.p_next =
                &mut filter_cubic_image_view_image_format_properties as *mut _ as *mut _;
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
                filter_cubic: filter_cubic_image_view_image_format_properties.filter_cubic
                    != ash::vk::FALSE,
                filter_cubic_minmax: filter_cubic_image_view_image_format_properties
                    .filter_cubic_minmax
                    != ash::vk::FALSE,
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
        !(self.info.property_flags & ash::vk::MemoryPropertyFlags::DEVICE_LOCAL).is_empty()
    }

    /// Returns true if the memory type can be accessed by the host.
    #[inline]
    pub fn is_host_visible(&self) -> bool {
        !(self.info.property_flags & ash::vk::MemoryPropertyFlags::HOST_VISIBLE).is_empty()
    }

    /// Returns true if modifications made by the host or the GPU on this memory type are
    /// instantaneously visible to the other party. False means that changes have to be flushed.
    ///
    /// You don't need to worry about this, as this library handles that for you.
    #[inline]
    pub fn is_host_coherent(&self) -> bool {
        !(self.info.property_flags & ash::vk::MemoryPropertyFlags::HOST_COHERENT).is_empty()
    }

    /// Returns true if memory of this memory type is cached by the host. Host memory accesses to
    /// cached memory is faster than for uncached memory. However you are not guaranteed that it
    /// is coherent.
    #[inline]
    pub fn is_host_cached(&self) -> bool {
        !(self.info.property_flags & ash::vk::MemoryPropertyFlags::HOST_CACHED).is_empty()
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
        !(self.info.property_flags & ash::vk::MemoryPropertyFlags::LAZILY_ALLOCATED).is_empty()
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
        !(self.properties.queue_flags & ash::vk::QueueFlags::GRAPHICS).is_empty()
    }

    /// Returns `true` if queues of this family can execute compute operations.
    #[inline]
    pub fn supports_compute(&self) -> bool {
        !(self.properties.queue_flags & ash::vk::QueueFlags::COMPUTE).is_empty()
    }

    /// Returns `true` if queues of this family can execute transfer operations.
    /// > **Note**: While all queues that can perform graphics or compute operations can implicitly perform
    /// > transfer operations, graphics & compute queues only optionally indicate support for tranfers.
    /// > Many discrete cards will have one queue family that exclusively sets the VK_QUEUE_TRANSFER_BIT
    /// > to indicate a special relationship with the DMA module and more efficient transfers.
    #[inline]
    pub fn explicitly_supports_transfers(&self) -> bool {
        !(self.properties.queue_flags & ash::vk::QueueFlags::TRANSFER).is_empty()
    }

    /// Returns `true` if queues of this family can execute sparse resources binding operations.
    #[inline]
    pub fn supports_sparse_binding(&self) -> bool {
        !(self.properties.queue_flags & ash::vk::QueueFlags::SPARSE_BINDING).is_empty()
    }

    /// Returns `true` if the queues of this family support a particular pipeline stage.
    #[inline]
    pub fn supports_stage(&self, stage: PipelineStage) -> bool {
        !(self.properties.queue_flags & stage.required_queue_flags()).is_empty()
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

/// The properties of a format that are supported by a physical device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FormatProperties {
    /// Features available for images with linear tiling.
    pub linear_tiling_features: FormatFeatures,

    /// Features available for images with optimal tiling.
    pub optimal_tiling_features: FormatFeatures,

    /// Features available for buffers.
    pub buffer_features: FormatFeatures,
}

/// The features supported by a device for an image or buffer with a particular format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(missing_docs)]
pub struct FormatFeatures {
    // Image usage
    /// Can be used with a sampled image descriptor.
    pub sampled_image: bool,
    /// Can be used with a storage image descriptor.
    pub storage_image: bool,
    /// Can be used with a storage image descriptor with atomic operations in a shader.
    pub storage_image_atomic: bool,
    /// Can be used with a storage image descriptor for reading, without specifying a format on the
    /// image view.
    pub storage_read_without_format: bool,
    /// Can be used with a storage image descriptor for writing, without specifying a format on the
    /// image view.
    pub storage_write_without_format: bool,
    /// Can be used with a color attachment in a framebuffer, or with an input attachment
    /// descriptor.
    pub color_attachment: bool,
    /// Can be used with a color attachment in a framebuffer with blending, or with an input
    /// attachment descriptor.
    pub color_attachment_blend: bool,
    /// Can be used with a depth/stencil attachment in a framebuffer, or with an input attachment
    /// descriptor.
    pub depth_stencil_attachment: bool,
    /// Can be used with a fragment density map attachment in a framebuffer.
    pub fragment_density_map: bool,
    /// Can be used with a fragment shading rate attachment in a framebuffer.
    pub fragment_shading_rate_attachment: bool,
    /// Can be used with the source image in a transfer (copy) operation.
    pub transfer_src: bool,
    /// Can be used with the destination image in a transfer (copy) operation.
    pub transfer_dst: bool,
    /// Can be used with the source image in a blit operation.
    pub blit_src: bool,
    /// Can be used with the destination image in a blit operation.
    pub blit_dst: bool,

    // Sampling
    /// Can be used with samplers or as a blit source, using the
    /// [`Linear`](crate::sampler::Filter::Linear) filter.
    pub sampled_image_filter_linear: bool,
    /// Can be used with samplers or as a blit source, using the
    /// [`Cubic`](crate::sampler::Filter::Cubic) filter.
    pub sampled_image_filter_cubic: bool,
    /// Can be used with samplers using a reduction mode of
    /// [`Min`](crate::sampler::SamplerReductionMode::Min) or
    /// [`Max`](crate::sampler::SamplerReductionMode::Max).
    pub sampled_image_filter_minmax: bool,
    /// Can be used with sampler YCbCr conversions using a chroma offset of
    /// [`Midpoint`](crate::sampler::ycbcr::ChromaLocation::Midpoint).
    pub midpoint_chroma_samples: bool,
    /// Can be used with sampler YCbCr conversions using a chroma offset of
    /// [`CositedEven`](crate::sampler::ycbcr::ChromaLocation::CositedEven).
    pub cosited_chroma_samples: bool,
    /// Can be used with sampler YCbCr conversions using the
    /// [`Linear`](crate::sampler::Filter::Linear) chroma filter.
    pub sampled_image_ycbcr_conversion_linear_filter: bool,
    /// Can be used with sampler YCbCr conversions whose chroma filter differs from the filters of
    /// the base sampler.
    pub sampled_image_ycbcr_conversion_separate_reconstruction_filter: bool,
    /// When used with a sampler YCbCr conversion, the implementation will always perform
    /// explicit chroma reconstruction.
    pub sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: bool,
    /// Can be used with sampler YCbCr conversions with forced explicit reconstruction.
    pub sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: bool,
    /// Can be used with samplers using depth comparison.
    pub sampled_image_depth_comparison: bool,

    // Video
    /// Can be used with the output image of a video decode operation.
    pub video_decode_output: bool,
    /// Can be used with the DPB image of a video decode operation.
    pub video_decode_dpb: bool,
    /// Can be used with the input image of a video encode operation.
    pub video_encode_input: bool,
    /// Can be used with the DPB image of a video encode operation.
    pub video_encode_dpb: bool,

    // Misc image features
    /// For multi-planar formats, can be used with images created with the `disjoint` flag.
    pub disjoint: bool,

    // Buffer usage
    /// Can be used with a uniform texel buffer descriptor.
    pub uniform_texel_buffer: bool,
    /// Can be used with a storage texel buffer descriptor.
    pub storage_texel_buffer: bool,
    /// Can be used with a storage texel buffer descriptor with atomic operations in a shader.
    pub storage_texel_buffer_atomic: bool,
    /// Can be used as the format of a vertex attribute in the vertex input state of a graphics
    /// pipeline.
    pub vertex_buffer: bool,
    /// Can be used with the vertex buffer of an acceleration structure.
    pub acceleration_structure_vertex_buffer: bool,
}

impl BitOr for &FormatFeatures {
    type Output = FormatFeatures;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::Output {
            sampled_image: self.sampled_image || rhs.sampled_image,
            storage_image: self.storage_image || rhs.storage_image,
            storage_image_atomic: self.storage_image_atomic || rhs.storage_image_atomic,
            storage_read_without_format: self.storage_read_without_format
                || rhs.storage_read_without_format,
            storage_write_without_format: self.storage_write_without_format
                || rhs.storage_write_without_format,
            color_attachment: self.color_attachment || rhs.color_attachment,
            color_attachment_blend: self.color_attachment_blend || rhs.color_attachment_blend,
            depth_stencil_attachment: self.depth_stencil_attachment || rhs.depth_stencil_attachment,
            fragment_density_map: self.fragment_density_map || rhs.fragment_density_map,
            fragment_shading_rate_attachment: self.fragment_shading_rate_attachment
                || rhs.fragment_shading_rate_attachment,
            transfer_src: self.transfer_src || rhs.transfer_src,
            transfer_dst: self.transfer_dst || rhs.transfer_dst,
            blit_src: self.blit_src || rhs.blit_src,
            blit_dst: self.blit_dst || rhs.blit_dst,

            sampled_image_filter_linear: self.sampled_image_filter_linear
                || rhs.sampled_image_filter_linear,
            sampled_image_filter_cubic: self.sampled_image_filter_cubic
                || rhs.sampled_image_filter_cubic,
            sampled_image_filter_minmax: self.sampled_image_filter_minmax
                || rhs.sampled_image_filter_minmax,
            midpoint_chroma_samples: self.midpoint_chroma_samples || rhs.midpoint_chroma_samples,
            cosited_chroma_samples: self.cosited_chroma_samples || rhs.cosited_chroma_samples,
            sampled_image_ycbcr_conversion_linear_filter: self
                .sampled_image_ycbcr_conversion_linear_filter
                || rhs.sampled_image_ycbcr_conversion_linear_filter,
            sampled_image_ycbcr_conversion_separate_reconstruction_filter: self
                .sampled_image_ycbcr_conversion_separate_reconstruction_filter
                || rhs.sampled_image_ycbcr_conversion_separate_reconstruction_filter,
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: self
                .sampled_image_ycbcr_conversion_chroma_reconstruction_explicit
                || rhs.sampled_image_ycbcr_conversion_chroma_reconstruction_explicit,
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: self
                .sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable
                || rhs.sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable,
            sampled_image_depth_comparison: self.sampled_image_depth_comparison
                || rhs.sampled_image_depth_comparison,

            video_decode_output: self.video_decode_output || rhs.video_decode_output,
            video_decode_dpb: self.video_decode_dpb || rhs.video_decode_dpb,
            video_encode_input: self.video_encode_input || rhs.video_encode_input,
            video_encode_dpb: self.video_encode_dpb || rhs.video_encode_dpb,

            disjoint: self.disjoint || rhs.disjoint,

            uniform_texel_buffer: self.uniform_texel_buffer || rhs.uniform_texel_buffer,
            storage_texel_buffer: self.storage_texel_buffer || rhs.storage_texel_buffer,
            storage_texel_buffer_atomic: self.storage_texel_buffer_atomic
                || rhs.storage_texel_buffer_atomic,
            vertex_buffer: self.vertex_buffer || rhs.vertex_buffer,
            acceleration_structure_vertex_buffer: self.acceleration_structure_vertex_buffer
                || rhs.acceleration_structure_vertex_buffer,
        }
    }
}

impl From<ash::vk::FormatFeatureFlags> for FormatFeatures {
    #[inline]
    #[rustfmt::skip]
    fn from(val: ash::vk::FormatFeatureFlags) -> FormatFeatures {
        FormatFeatures {
            sampled_image: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE),
            storage_image: val.intersects(ash::vk::FormatFeatureFlags::STORAGE_IMAGE),
            storage_image_atomic: val.intersects(ash::vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC),
            storage_read_without_format: false, // FormatFeatureFlags2KHR only
            storage_write_without_format: false, // FormatFeatureFlags2KHR only
            color_attachment: val.intersects(ash::vk::FormatFeatureFlags::COLOR_ATTACHMENT),
            color_attachment_blend: val.intersects(ash::vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
            depth_stencil_attachment: val.intersects(ash::vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
            fragment_density_map: val.intersects(ash::vk::FormatFeatureFlags::FRAGMENT_DENSITY_MAP_EXT),
            fragment_shading_rate_attachment: val.intersects(ash::vk::FormatFeatureFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR),
            transfer_src: val.intersects(ash::vk::FormatFeatureFlags::TRANSFER_SRC),
            transfer_dst: val.intersects(ash::vk::FormatFeatureFlags::TRANSFER_DST),
            blit_src: val.intersects(ash::vk::FormatFeatureFlags::BLIT_SRC),
            blit_dst: val.intersects(ash::vk::FormatFeatureFlags::BLIT_DST),

            sampled_image_filter_linear: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
            sampled_image_filter_cubic: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_CUBIC_EXT),
            sampled_image_filter_minmax: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_MINMAX),
            midpoint_chroma_samples: val.intersects(ash::vk::FormatFeatureFlags::MIDPOINT_CHROMA_SAMPLES),
            cosited_chroma_samples: val.intersects(ash::vk::FormatFeatureFlags::COSITED_CHROMA_SAMPLES),
            sampled_image_ycbcr_conversion_linear_filter: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER),
            sampled_image_ycbcr_conversion_separate_reconstruction_filter: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER),
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT),
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: val.intersects(ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE),
            sampled_image_depth_comparison: false, // FormatFeatureFlags2KHR only

            video_decode_output: val.intersects(ash::vk::FormatFeatureFlags::VIDEO_DECODE_OUTPUT_KHR),
            video_decode_dpb: val.intersects(ash::vk::FormatFeatureFlags::VIDEO_DECODE_DPB_KHR),
            video_encode_input: val.intersects(ash::vk::FormatFeatureFlags::VIDEO_ENCODE_INPUT_KHR),
            video_encode_dpb: val.intersects(ash::vk::FormatFeatureFlags::VIDEO_ENCODE_DPB_KHR),

            disjoint: val.intersects(ash::vk::FormatFeatureFlags::DISJOINT),

            uniform_texel_buffer: val.intersects(ash::vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER),
            storage_texel_buffer: val.intersects(ash::vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER),
            storage_texel_buffer_atomic: val.intersects(ash::vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER_ATOMIC),
            vertex_buffer: val.intersects(ash::vk::FormatFeatureFlags::VERTEX_BUFFER),
            acceleration_structure_vertex_buffer: val.intersects(ash::vk::FormatFeatureFlags::ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR),
        }
    }
}

impl From<ash::vk::FormatFeatureFlags2KHR> for FormatFeatures {
    #[inline]
    #[rustfmt::skip]
    fn from(val: ash::vk::FormatFeatureFlags2KHR) -> FormatFeatures {
        FormatFeatures {
            sampled_image: val.intersects(ash::vk::FormatFeatureFlags2KHR::SAMPLED_IMAGE),
            storage_image: val.intersects(ash::vk::FormatFeatureFlags2KHR::STORAGE_IMAGE),
            storage_image_atomic: val.intersects(ash::vk::FormatFeatureFlags2KHR::STORAGE_IMAGE_ATOMIC),
            storage_read_without_format: val.intersects(ash::vk::FormatFeatureFlags2KHR::STORAGE_READ_WITHOUT_FORMAT),
            storage_write_without_format: val.intersects(ash::vk::FormatFeatureFlags2KHR::STORAGE_WRITE_WITHOUT_FORMAT),
            color_attachment: val.intersects(ash::vk::FormatFeatureFlags2KHR::COLOR_ATTACHMENT),
            color_attachment_blend: val.intersects(ash::vk::FormatFeatureFlags2KHR::COLOR_ATTACHMENT_BLEND),
            depth_stencil_attachment: val.intersects(ash::vk::FormatFeatureFlags2KHR::DEPTH_STENCIL_ATTACHMENT),
            fragment_density_map: val.intersects(ash::vk::FormatFeatureFlags2KHR::FRAGMENT_DENSITY_MAP_EXT),
            fragment_shading_rate_attachment: val.intersects(ash::vk::FormatFeatureFlags2KHR::FRAGMENT_SHADING_RATE_ATTACHMENT),
            transfer_src: val.intersects(ash::vk::FormatFeatureFlags2KHR::TRANSFER_SRC),
            transfer_dst: val.intersects(ash::vk::FormatFeatureFlags2KHR::TRANSFER_DST),
            blit_src: val.intersects(ash::vk::FormatFeatureFlags2KHR::BLIT_SRC),
            blit_dst: val.intersects(ash::vk::FormatFeatureFlags2KHR::BLIT_DST),

            sampled_image_filter_linear: val.intersects(ash::vk::FormatFeatureFlags2KHR::SAMPLED_IMAGE_FILTER_LINEAR),
            sampled_image_filter_cubic: val.intersects(ash::vk::FormatFeatureFlags2KHR::SAMPLED_IMAGE_FILTER_CUBIC_EXT),
            sampled_image_filter_minmax: val.intersects(ash::vk::FormatFeatureFlags2KHR::SAMPLED_IMAGE_FILTER_MINMAX),
            midpoint_chroma_samples: val.intersects(ash::vk::FormatFeatureFlags2KHR::MIDPOINT_CHROMA_SAMPLES),
            cosited_chroma_samples: val.intersects(ash::vk::FormatFeatureFlags2KHR::COSITED_CHROMA_SAMPLES),
            sampled_image_ycbcr_conversion_linear_filter: val.intersects(ash::vk::FormatFeatureFlags2KHR::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER),
            sampled_image_ycbcr_conversion_separate_reconstruction_filter: val.intersects(ash::vk::FormatFeatureFlags2KHR::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER),
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit: val.intersects(ash::vk::FormatFeatureFlags2KHR::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT),
            sampled_image_ycbcr_conversion_chroma_reconstruction_explicit_forceable: val.intersects(ash::vk::FormatFeatureFlags2KHR::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE),
            sampled_image_depth_comparison: val.intersects(ash::vk::FormatFeatureFlags2KHR::SAMPLED_IMAGE_DEPTH_COMPARISON),

            video_decode_output: val.intersects(ash::vk::FormatFeatureFlags2KHR::VIDEO_DECODE_OUTPUT),
            video_decode_dpb: val.intersects(ash::vk::FormatFeatureFlags2KHR::VIDEO_DECODE_DPB),
            video_encode_input: val.intersects(ash::vk::FormatFeatureFlags2KHR::VIDEO_ENCODE_INPUT),
            video_encode_dpb: val.intersects(ash::vk::FormatFeatureFlags2KHR::VIDEO_ENCODE_DPB),

            disjoint: val.intersects(ash::vk::FormatFeatureFlags2KHR::DISJOINT),

            uniform_texel_buffer: val.intersects(ash::vk::FormatFeatureFlags2KHR::UNIFORM_TEXEL_BUFFER),
            storage_texel_buffer: val.intersects(ash::vk::FormatFeatureFlags2KHR::STORAGE_TEXEL_BUFFER),
            storage_texel_buffer_atomic: val.intersects(ash::vk::FormatFeatureFlags2KHR::STORAGE_TEXEL_BUFFER_ATOMIC),
            vertex_buffer: val.intersects(ash::vk::FormatFeatureFlags2KHR::VERTEX_BUFFER),
            acceleration_structure_vertex_buffer: val.intersects(ash::vk::FormatFeatureFlags2KHR::ACCELERATION_STRUCTURE_VERTEX_BUFFER),
        }
    }
}

/// The properties that are supported by a physical device for images of a certain type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageFormatProperties {
    /// The maximum dimensions.
    pub max_extent: [u32; 3],
    /// The maximum number of mipmap levels.
    pub max_mip_levels: u32,
    /// The maximum number of array layers.
    pub max_array_layers: u32,
    /// The supported sample counts.
    pub sample_counts: SampleCounts,
    /// The maximum total size of an image, in bytes. This is guaranteed to be at least
    /// 0x80000000.
    pub max_resource_size: DeviceSize,
    /// When querying with an image view type, whether such image views support sampling with
    /// a [`Cubic`](crate::sampler::Filter::Cubic) `mag_filter` or `min_filter`.
    pub filter_cubic: bool,
    /// When querying with an image view type, whether such image views support sampling with
    /// a [`Cubic`](crate::sampler::Filter::Cubic) `mag_filter` or `min_filter`, and with a
    /// [`Min`](crate::sampler::SamplerReductionMode::Min) or
    /// [`Max`](crate::sampler::SamplerReductionMode::Max) `reduction_mode`.
    pub filter_cubic_minmax: bool,
}

impl From<ash::vk::ImageFormatProperties> for ImageFormatProperties {
    fn from(props: ash::vk::ImageFormatProperties) -> Self {
        Self {
            max_extent: [
                props.max_extent.width,
                props.max_extent.height,
                props.max_extent.depth,
            ],
            max_mip_levels: props.max_mip_levels,
            max_array_layers: props.max_array_layers,
            sample_counts: props.sample_counts.into(),
            max_resource_size: props.max_resource_size,
            filter_cubic: false,
            filter_cubic_minmax: false,
        }
    }
}
