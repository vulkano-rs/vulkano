// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::device::{DeviceExtensions, Features, FeaturesFfi, Properties, PropertiesFfi};
use crate::instance::{Instance, InstanceCreationError};
use crate::sync::PipelineStage;
use crate::Version;
use crate::VulkanObject;
use std::convert::TryFrom;
use std::ffi::CStr;
use std::fmt;
use std::hash::Hash;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

pub(super) fn init_physical_devices(
    instance: &Instance,
) -> Result<Vec<PhysicalDeviceInfos>, InstanceCreationError> {
    let fns = instance.fns();
    let instance_extensions = instance.loaded_extensions();

    let physical_devices: Vec<ash::vk::PhysicalDevice> = unsafe {
        let mut num = 0;
        check_errors(fns.v1_0.enumerate_physical_devices(
            instance.internal_object(),
            &mut num,
            ptr::null_mut(),
        ))?;

        let mut devices = Vec::with_capacity(num as usize);
        check_errors(fns.v1_0.enumerate_physical_devices(
            instance.internal_object(),
            &mut num,
            devices.as_mut_ptr(),
        ))?;
        devices.set_len(num as usize);
        devices
    };

    let info: Vec<_> = physical_devices
        .into_iter()
        .map(|physical_device| -> Result<_, InstanceCreationError> {
            let api_version = unsafe {
                let mut output = MaybeUninit::uninit();
                fns.v1_0
                    .get_physical_device_properties(physical_device, output.as_mut_ptr());
                let api_version = Version::try_from(output.assume_init().api_version).unwrap();
                std::cmp::min(instance.max_api_version(), api_version)
            };

            let extension_properties: Vec<ash::vk::ExtensionProperties> = unsafe {
                let mut num = 0;
                check_errors(fns.v1_0.enumerate_device_extension_properties(
                    physical_device,
                    ptr::null(),
                    &mut num,
                    ptr::null_mut(),
                ))?;

                let mut properties = Vec::with_capacity(num as usize);
                check_errors(fns.v1_0.enumerate_device_extension_properties(
                    physical_device,
                    ptr::null(),
                    &mut num,
                    properties.as_mut_ptr(),
                ))?;
                properties.set_len(num as usize);
                properties
            };

            let extensions = DeviceExtensions::from(
                extension_properties
                    .iter()
                    .map(|property| unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }),
            );

            Ok((physical_device, api_version, extensions))
        })
        .collect::<Result<_, _>>()?;

    // Getting the properties of all physical devices.
    // If possible, we use VK_KHR_get_physical_device_properties2.
    let physical_devices = if instance.api_version() >= Version::V1_1
        || instance_extensions.khr_get_physical_device_properties2
    {
        init_physical_devices_inner2(instance, info)
    } else {
        init_physical_devices_inner(instance, info)
    };

    Ok(physical_devices)
}

/// Initialize all physical devices
fn init_physical_devices_inner<I>(instance: &Instance, info: I) -> Vec<PhysicalDeviceInfos>
where
    I: IntoIterator<Item = (ash::vk::PhysicalDevice, Version, DeviceExtensions)>,
{
    let fns = instance.fns();

    info.into_iter()
        .map(|(physical_device, api_version, supported_extensions)| {
            let properties: Properties = unsafe {
                let mut output = PropertiesFfi::default();
                output.make_chain(
                    api_version,
                    &supported_extensions,
                    instance.loaded_extensions(),
                );
                fns.v1_0.get_physical_device_properties(
                    physical_device,
                    &mut output.head_as_mut().properties,
                );
                Properties::from(&output)
            };

            let queue_families = unsafe {
                let mut num = 0;
                fns.v1_0.get_physical_device_queue_family_properties(
                    physical_device,
                    &mut num,
                    ptr::null_mut(),
                );

                let mut families = Vec::with_capacity(num as usize);
                fns.v1_0.get_physical_device_queue_family_properties(
                    physical_device,
                    &mut num,
                    families.as_mut_ptr(),
                );
                families.set_len(num as usize);
                families
            };

            let memory_properties: ash::vk::PhysicalDeviceMemoryProperties = unsafe {
                let mut output = MaybeUninit::uninit();
                fns.v1_0
                    .get_physical_device_memory_properties(physical_device, output.as_mut_ptr());
                output.assume_init()
            };

            let available_features: Features = unsafe {
                let mut output = FeaturesFfi::default();
                fns.v1_0.get_physical_device_features(
                    physical_device,
                    &mut output.head_as_mut().features,
                );
                Features::from(&output)
            };

            PhysicalDeviceInfos {
                physical_device,
                api_version,
                properties,
                memory_properties,
                queue_families,
                available_features,
            }
        })
        .collect()
}

/// Initialize all physical devices, but use VK_KHR_get_physical_device_properties2
/// TODO: Query extension-specific physical device properties, once a new instance extension is supported.
fn init_physical_devices_inner2<I>(instance: &Instance, info: I) -> Vec<PhysicalDeviceInfos>
where
    I: IntoIterator<Item = (ash::vk::PhysicalDevice, Version, DeviceExtensions)>,
{
    let fns = instance.fns();

    info.into_iter()
        .map(|(physical_device, api_version, supported_extensions)| {
            let properties: Properties = unsafe {
                let mut output = PropertiesFfi::default();
                output.make_chain(
                    api_version,
                    &supported_extensions,
                    instance.loaded_extensions(),
                );

                if instance.api_version() >= Version::V1_1 {
                    fns.v1_1
                        .get_physical_device_properties2(physical_device, output.head_as_mut());
                } else {
                    fns.khr_get_physical_device_properties2
                        .get_physical_device_properties2_khr(physical_device, output.head_as_mut());
                }

                Properties::from(&output)
            };

            let queue_families = unsafe {
                let mut num = 0;

                if instance.api_version() >= Version::V1_1 {
                    fns.v1_1.get_physical_device_queue_family_properties2(
                        physical_device,
                        &mut num,
                        ptr::null_mut(),
                    );
                } else {
                    fns.khr_get_physical_device_properties2
                        .get_physical_device_queue_family_properties2_khr(
                            physical_device,
                            &mut num,
                            ptr::null_mut(),
                        );
                }

                let mut families = vec![ash::vk::QueueFamilyProperties2::default(); num as usize];

                if instance.api_version() >= Version::V1_1 {
                    fns.v1_1.get_physical_device_queue_family_properties2(
                        physical_device,
                        &mut num,
                        families.as_mut_ptr(),
                    );
                } else {
                    fns.khr_get_physical_device_properties2
                        .get_physical_device_queue_family_properties2_khr(
                            physical_device,
                            &mut num,
                            families.as_mut_ptr(),
                        );
                }

                families
                    .into_iter()
                    .map(|family| family.queue_family_properties)
                    .collect()
            };

            let memory_properties: ash::vk::PhysicalDeviceMemoryProperties = unsafe {
                let mut output = ash::vk::PhysicalDeviceMemoryProperties2KHR::default();

                if instance.api_version() >= Version::V1_1 {
                    fns.v1_1
                        .get_physical_device_memory_properties2(physical_device, &mut output);
                } else {
                    fns.khr_get_physical_device_properties2
                        .get_physical_device_memory_properties2_khr(physical_device, &mut output);
                }

                output.memory_properties
            };

            let available_features: Features = unsafe {
                let mut output = FeaturesFfi::default();
                output.make_chain(
                    api_version,
                    &supported_extensions,
                    instance.loaded_extensions(),
                );

                if instance.api_version() >= Version::V1_1 {
                    fns.v1_1
                        .get_physical_device_features2(physical_device, output.head_as_mut());
                } else {
                    fns.khr_get_physical_device_properties2
                        .get_physical_device_features2_khr(physical_device, output.head_as_mut());
                }

                Features::from(&output)
            };

            PhysicalDeviceInfos {
                physical_device,
                api_version,
                properties,
                memory_properties,
                queue_families,
                available_features,
            }
        })
        .collect()
}

pub(super) struct PhysicalDeviceInfos {
    physical_device: ash::vk::PhysicalDevice,
    api_version: Version,
    properties: Properties,
    queue_families: Vec<ash::vk::QueueFamilyProperties>,
    memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
    available_features: Features,
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
/// use vulkano::instance::PhysicalDevice;
///
/// # let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None).unwrap();
/// for physical_device in PhysicalDevice::enumerate(&instance) {
///     print_infos(physical_device);
/// }
///
/// fn print_infos(dev: PhysicalDevice) {
///     println!("Name: {}", dev.properties().device_name.as_ref().unwrap());
/// }
/// ```
#[derive(Debug, Copy, Clone)]
pub struct PhysicalDevice<'a> {
    instance: &'a Arc<Instance>,
    device: usize,
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
    /// use vulkano::instance::PhysicalDevice;
    ///
    /// # let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None).unwrap();
    /// for physical_device in PhysicalDevice::enumerate(&instance) {
    ///     println!("Available device: {}", physical_device.properties().device_name.as_ref().unwrap());
    /// }
    /// ```
    #[inline]
    pub fn enumerate(instance: &'a Arc<Instance>) -> PhysicalDevicesIter<'a> {
        PhysicalDevicesIter {
            instance,
            current_id: 0,
        }
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
    /// use vulkano::instance::PhysicalDevice;
    /// use vulkano::Version;
    ///
    /// let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None).unwrap();
    /// let first_physical_device = PhysicalDevice::from_index(&instance, 0).unwrap();
    /// ```
    #[inline]
    pub fn from_index(instance: &'a Arc<Instance>, index: usize) -> Option<PhysicalDevice<'a>> {
        if instance.physical_devices.len() > index {
            Some(PhysicalDevice {
                instance,
                device: index,
            })
        } else {
            None
        }
    }

    /// Returns the instance corresponding to this physical device.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::instance::PhysicalDevice;
    ///
    /// fn do_something(physical_device: PhysicalDevice) {
    ///     let _loaded_extensions = physical_device.instance().loaded_extensions();
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
        self.device
    }

    /// Returns the version of Vulkan supported by this device.
    ///
    /// Unlike the `api_version` property, which is the version reported by the device directly,
    /// this function returns the version the device can actually support, based on the instance's,
    /// `max_api_version`.
    #[inline]
    pub fn api_version(&self) -> Version {
        self.infos().api_version
    }

    /// Returns the Vulkan properties reported by the device.
    #[inline]
    pub fn properties(&self) -> &'a Properties {
        &self.infos().properties
    }

    /// Returns the Vulkan features that are supported by this physical device.
    #[inline]
    pub fn supported_features(&self) -> &'a Features {
        &self.infos().available_features
    }

    /// Builds an iterator that enumerates all the queue families on this physical device.
    #[inline]
    pub fn queue_families(&self) -> QueueFamiliesIter<'a> {
        QueueFamiliesIter {
            physical_device: *self,
            current_id: 0,
        }
    }

    /// Returns the queue family with the given index, or `None` if out of range.
    #[inline]
    pub fn queue_family_by_id(&self, id: u32) -> Option<QueueFamily<'a>> {
        if (id as usize) < self.infos().queue_families.len() {
            Some(QueueFamily {
                physical_device: *self,
                id,
            })
        } else {
            None
        }
    }

    /// Builds an iterator that enumerates all the memory types on this physical device.
    #[inline]
    pub fn memory_types(&self) -> MemoryTypesIter<'a> {
        MemoryTypesIter {
            physical_device: *self,
            current_id: 0,
        }
    }

    /// Returns the memory type with the given index, or `None` if out of range.
    #[inline]
    pub fn memory_type_by_id(&self, id: u32) -> Option<MemoryType<'a>> {
        if id < self.infos().memory_properties.memory_type_count {
            Some(MemoryType {
                physical_device: *self,
                id,
            })
        } else {
            None
        }
    }

    /// Builds an iterator that enumerates all the memory heaps on this physical device.
    #[inline]
    pub fn memory_heaps(&self) -> MemoryHeapsIter<'a> {
        MemoryHeapsIter {
            physical_device: *self,
            current_id: 0,
        }
    }

    /// Returns the memory heap with the given index, or `None` if out of range.
    #[inline]
    pub fn memory_heap_by_id(&self, id: u32) -> Option<MemoryHeap<'a>> {
        if id < self.infos().memory_properties.memory_heap_count {
            Some(MemoryHeap {
                physical_device: *self,
                id,
            })
        } else {
            None
        }
    }

    // Internal function to make it easier to get the infos of this device.
    #[inline]
    fn infos(&self) -> &'a PhysicalDeviceInfos {
        &self.instance.physical_devices[self.device]
    }
}

unsafe impl<'a> VulkanObject for PhysicalDevice<'a> {
    type Object = ash::vk::PhysicalDevice;

    #[inline]
    fn internal_object(&self) -> ash::vk::PhysicalDevice {
        self.infos().physical_device
    }
}

/// Iterator for all the physical devices available on hardware.
#[derive(Debug, Clone)]
pub struct PhysicalDevicesIter<'a> {
    instance: &'a Arc<Instance>,
    current_id: usize,
}

impl<'a> Iterator for PhysicalDevicesIter<'a> {
    type Item = PhysicalDevice<'a>;

    #[inline]
    fn next(&mut self) -> Option<PhysicalDevice<'a>> {
        if self.current_id >= self.instance.physical_devices.len() {
            return None;
        }

        let dev = PhysicalDevice {
            instance: self.instance,
            device: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.instance.physical_devices.len() - self.current_id;
        (len, Some(len))
    }
}

impl<'a> ExactSizeIterator for PhysicalDevicesIter<'a> {}

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

/// Represents a queue family in a physical device.
///
/// A queue family is group of one or multiple queues. All queues of one family have the same
/// characteristics.
#[derive(Debug, Copy, Clone)]
pub struct QueueFamily<'a> {
    physical_device: PhysicalDevice<'a>,
    id: u32,
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
        self.physical_device.infos().queue_families[self.id as usize].queue_count as usize
    }

    /// If timestamps are supported, returns the number of bits supported by timestamp operations.
    /// The returned value will be in the range 36..64.
    /// If timestamps are not supported, returns None.
    #[inline]
    pub fn timestamp_valid_bits(&self) -> Option<u32> {
        let value =
            self.physical_device.infos().queue_families[self.id as usize].timestamp_valid_bits;
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
        let ref granularity = self.physical_device.infos().queue_families[self.id as usize]
            .min_image_transfer_granularity;
        [granularity.width, granularity.height, granularity.depth]
    }

    /// Returns `true` if queues of this family can execute graphics operations.
    #[inline]
    pub fn supports_graphics(&self) -> bool {
        !(self.flags() & ash::vk::QueueFlags::GRAPHICS).is_empty()
    }

    /// Returns `true` if queues of this family can execute compute operations.
    #[inline]
    pub fn supports_compute(&self) -> bool {
        !(self.flags() & ash::vk::QueueFlags::COMPUTE).is_empty()
    }

    /// Returns `true` if queues of this family can execute transfer operations.
    /// > **Note**: While all queues that can perform graphics or compute operations can implicitly perform
    /// > transfer operations, graphics & compute queues only optionally indicate support for tranfers.
    /// > Many discrete cards will have one queue family that exclusively sets the VK_QUEUE_TRANSFER_BIT
    /// > to indicate a special relationship with the DMA module and more efficient transfers.
    #[inline]
    pub fn explicitly_supports_transfers(&self) -> bool {
        !(self.flags() & ash::vk::QueueFlags::TRANSFER).is_empty()
    }

    /// Returns `true` if queues of this family can execute sparse resources binding operations.
    #[inline]
    pub fn supports_sparse_binding(&self) -> bool {
        !(self.flags() & ash::vk::QueueFlags::SPARSE_BINDING).is_empty()
    }

    /// Returns `true` if the queues of this family support a particular pipeline stage.
    #[inline]
    pub fn supports_stage(&self, stage: PipelineStage) -> bool {
        !(self.flags() & stage.required_queue_flags()).is_empty()
    }

    /// Internal utility function that returns the flags of this queue family.
    #[inline]
    fn flags(&self) -> ash::vk::QueueFlags {
        self.physical_device.infos().queue_families[self.id as usize].queue_flags
    }
}

impl<'a> PartialEq for QueueFamily<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.physical_device.internal_object() == other.physical_device.internal_object()
    }
}

impl<'a> Eq for QueueFamily<'a> {}

/// Iterator for all the queue families available on a physical device.
#[derive(Debug, Clone)]
pub struct QueueFamiliesIter<'a> {
    physical_device: PhysicalDevice<'a>,
    current_id: u32,
}

impl<'a> Iterator for QueueFamiliesIter<'a> {
    type Item = QueueFamily<'a>;

    #[inline]
    fn next(&mut self) -> Option<QueueFamily<'a>> {
        if self.current_id as usize >= self.physical_device.infos().queue_families.len() {
            return None;
        }

        let dev = QueueFamily {
            physical_device: self.physical_device,
            id: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.physical_device.infos().queue_families.len();
        let remain = len - self.current_id as usize;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for QueueFamiliesIter<'a> {}

/// Represents a memory type in a physical device.
#[derive(Debug, Copy, Clone)]
pub struct MemoryType<'a> {
    physical_device: PhysicalDevice<'a>,
    id: u32,
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
        let heap_id = self.physical_device.infos().memory_properties.memory_types[self.id as usize]
            .heap_index;
        MemoryHeap {
            physical_device: self.physical_device,
            id: heap_id,
        }
    }

    /// Returns true if the memory type is located on the device, which means that it's the most
    /// efficient for GPU accesses.
    #[inline]
    pub fn is_device_local(&self) -> bool {
        !(self.flags() & ash::vk::MemoryPropertyFlags::DEVICE_LOCAL).is_empty()
    }

    /// Returns true if the memory type can be accessed by the host.
    #[inline]
    pub fn is_host_visible(&self) -> bool {
        !(self.flags() & ash::vk::MemoryPropertyFlags::HOST_VISIBLE).is_empty()
    }

    /// Returns true if modifications made by the host or the GPU on this memory type are
    /// instantaneously visible to the other party. False means that changes have to be flushed.
    ///
    /// You don't need to worry about this, as this library handles that for you.
    #[inline]
    pub fn is_host_coherent(&self) -> bool {
        !(self.flags() & ash::vk::MemoryPropertyFlags::HOST_COHERENT).is_empty()
    }

    /// Returns true if memory of this memory type is cached by the host. Host memory accesses to
    /// cached memory is faster than for uncached memory. However you are not guaranteed that it
    /// is coherent.
    #[inline]
    pub fn is_host_cached(&self) -> bool {
        !(self.flags() & ash::vk::MemoryPropertyFlags::HOST_CACHED).is_empty()
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
        !(self.flags() & ash::vk::MemoryPropertyFlags::LAZILY_ALLOCATED).is_empty()
    }

    /// Internal utility function that returns the flags of this queue family.
    #[inline]
    fn flags(&self) -> ash::vk::MemoryPropertyFlags {
        self.physical_device.infos().memory_properties.memory_types[self.id as usize].property_flags
    }
}

/// Iterator for all the memory types available on a physical device.
#[derive(Debug, Clone)]
pub struct MemoryTypesIter<'a> {
    physical_device: PhysicalDevice<'a>,
    current_id: u32,
}

impl<'a> Iterator for MemoryTypesIter<'a> {
    type Item = MemoryType<'a>;

    #[inline]
    fn next(&mut self) -> Option<MemoryType<'a>> {
        if self.current_id
            >= self
                .physical_device
                .infos()
                .memory_properties
                .memory_type_count
        {
            return None;
        }

        let dev = MemoryType {
            physical_device: self.physical_device,
            id: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self
            .physical_device
            .infos()
            .memory_properties
            .memory_type_count;
        let remain = (len - self.current_id) as usize;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for MemoryTypesIter<'a> {}

/// Represents a memory heap in a physical device.
#[derive(Debug, Copy, Clone)]
pub struct MemoryHeap<'a> {
    physical_device: PhysicalDevice<'a>,
    id: u32,
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
    pub fn size(&self) -> usize {
        self.physical_device.infos().memory_properties.memory_heaps[self.id as usize].size as usize
    }

    /// Returns true if the heap is local to the GPU.
    #[inline]
    pub fn is_device_local(&self) -> bool {
        let flags =
            self.physical_device.infos().memory_properties.memory_heaps[self.id as usize].flags;
        !(flags & ash::vk::MemoryHeapFlags::DEVICE_LOCAL).is_empty()
    }

    /// Returns true if the heap is multi-instance enabled, that is allocation from such
    /// heap will replicate to each physical-device's instance of heap.
    #[inline]
    pub fn is_multi_instance(&self) -> bool {
        let flags =
            self.physical_device.infos().memory_properties.memory_heaps[self.id as usize].flags;
        !(flags & ash::vk::MemoryHeapFlags::MULTI_INSTANCE).is_empty()
    }
}

/// Iterator for all the memory heaps available on a physical device.
#[derive(Debug, Clone)]
pub struct MemoryHeapsIter<'a> {
    physical_device: PhysicalDevice<'a>,
    current_id: u32,
}

impl<'a> Iterator for MemoryHeapsIter<'a> {
    type Item = MemoryHeap<'a>;

    #[inline]
    fn next(&mut self) -> Option<MemoryHeap<'a>> {
        if self.current_id
            >= self
                .physical_device
                .infos()
                .memory_properties
                .memory_heap_count
        {
            return None;
        }

        let dev = MemoryHeap {
            physical_device: self.physical_device,
            id: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self
            .physical_device
            .infos()
            .memory_properties
            .memory_heap_count;
        let remain = (len - self.current_id) as usize;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for MemoryHeapsIter<'a> {}

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
