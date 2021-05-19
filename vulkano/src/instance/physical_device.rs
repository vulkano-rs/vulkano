// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::device::{Features, FeaturesFfi};
use crate::instance::limits::Limits;
use crate::instance::{Instance, InstanceCreationError};
use crate::sync::PipelineStage;
use crate::vk;
use crate::Version;
use crate::VulkanObject;
use std::ffi::CStr;
use std::hash::Hash;
use std::mem;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

pub(super) fn init_physical_devices(
    instance: &Instance,
) -> Result<Vec<PhysicalDeviceInfos>, InstanceCreationError> {
    let vk = instance.pointers();
    let extensions = instance.loaded_extensions();

    let physical_devices: Vec<vk::PhysicalDevice> = unsafe {
        let mut num = 0;
        check_errors(vk.EnumeratePhysicalDevices(
            instance.internal_object(),
            &mut num,
            ptr::null_mut(),
        ))?;

        let mut devices = Vec::with_capacity(num as usize);
        check_errors(vk.EnumeratePhysicalDevices(
            instance.internal_object(),
            &mut num,
            devices.as_mut_ptr(),
        ))?;
        devices.set_len(num as usize);
        devices
    };

    // Getting the properties of all physical devices.
    // If possible, we use VK_KHR_get_physical_device_properties2.
    let physical_devices = if extensions.khr_get_physical_device_properties2 {
        init_physical_devices_inner2(instance, physical_devices)
    } else {
        init_physical_devices_inner(instance, physical_devices)
    };

    Ok(physical_devices)
}

/// Initialize all physical devices
fn init_physical_devices_inner(
    instance: &Instance,
    physical_devices: Vec<vk::PhysicalDevice>,
) -> Vec<PhysicalDeviceInfos> {
    let vk = instance.pointers();
    let mut output = Vec::with_capacity(physical_devices.len());

    for device in physical_devices.into_iter() {
        let properties: vk::PhysicalDeviceProperties = unsafe {
            let mut output = MaybeUninit::uninit();
            vk.GetPhysicalDeviceProperties(device, output.as_mut_ptr());
            output.assume_init()
        };

        let queue_families = unsafe {
            let mut num = 0;
            vk.GetPhysicalDeviceQueueFamilyProperties(device, &mut num, ptr::null_mut());

            let mut families = Vec::with_capacity(num as usize);
            vk.GetPhysicalDeviceQueueFamilyProperties(device, &mut num, families.as_mut_ptr());
            families.set_len(num as usize);
            families
        };

        let memory: vk::PhysicalDeviceMemoryProperties = unsafe {
            let mut output = MaybeUninit::uninit();
            vk.GetPhysicalDeviceMemoryProperties(device, output.as_mut_ptr());
            output.assume_init()
        };

        let available_features: Features = unsafe {
            let mut output = FeaturesFfi::default();
            vk.GetPhysicalDeviceFeatures(device, &mut output.head_as_mut().features);
            Features::from(&output)
        };

        output.push(PhysicalDeviceInfos {
            device,
            properties,
            extended_properties: PhysicalDeviceExtendedProperties::empty(),
            memory,
            queue_families,
            available_features: Features::from(available_features),
        });
    }
    output
}

/// Initialize all physical devices, but use VK_KHR_get_physical_device_properties2
/// TODO: Query extension-specific physical device properties, once a new instance extension is supported.
fn init_physical_devices_inner2(
    instance: &Instance,
    physical_devices: Vec<vk::PhysicalDevice>,
) -> Vec<PhysicalDeviceInfos> {
    let vk = instance.pointers();
    let mut output = Vec::with_capacity(physical_devices.len());

    for device in physical_devices.into_iter() {
        let mut extended_properties = PhysicalDeviceExtendedProperties::empty();

        let properties: vk::PhysicalDeviceProperties = unsafe {
            let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties {
                sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                pNext: ptr::null_mut(),
                subgroupSize: 0,
                supportedStages: 0,
                supportedOperations: 0,
                quadOperationsInAllStages: 0,
            };

            let mut output = vk::PhysicalDeviceProperties2KHR {
                sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR,
                pNext: if instance.api_version() >= Version::major_minor(1, 1) {
                    &mut subgroup_properties
                } else {
                    ptr::null_mut()
                },
                properties: mem::zeroed(),
            };

            vk.GetPhysicalDeviceProperties2KHR(device, &mut output);

            extended_properties = PhysicalDeviceExtendedProperties {
                subgroup_size: if instance.api_version() >= Version::major_minor(1, 1) {
                    Some(subgroup_properties.subgroupSize)
                } else {
                    None
                },

                ..extended_properties
            };

            output.properties
        };

        let queue_families = unsafe {
            let mut num = 0;
            vk.GetPhysicalDeviceQueueFamilyProperties2KHR(device, &mut num, ptr::null_mut());

            let mut families = (0..num)
                .map(|_| vk::QueueFamilyProperties2KHR {
                    sType: vk::STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR,
                    pNext: ptr::null_mut(),
                    queueFamilyProperties: mem::zeroed(),
                })
                .collect::<Vec<_>>();

            vk.GetPhysicalDeviceQueueFamilyProperties2KHR(device, &mut num, families.as_mut_ptr());
            families
                .into_iter()
                .map(|family| family.queueFamilyProperties)
                .collect()
        };

        let memory: vk::PhysicalDeviceMemoryProperties = unsafe {
            let mut output = vk::PhysicalDeviceMemoryProperties2KHR {
                sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR,
                pNext: ptr::null_mut(),
                memoryProperties: mem::zeroed(),
            };
            vk.GetPhysicalDeviceMemoryProperties2KHR(device, &mut output);
            output.memoryProperties
        };

        let available_features: Features = unsafe {
            let max_api_version = instance.max_api_version();
            let api_version = std::cmp::min(
                max_api_version,
                Version::from_vulkan_version(properties.apiVersion),
            );

            let mut output = FeaturesFfi::default();
            output.make_chain(api_version);
            vk.GetPhysicalDeviceFeatures2KHR(device, output.head_as_mut());
            Features::from(&output)
        };

        output.push(PhysicalDeviceInfos {
            device,
            properties,
            extended_properties,
            memory,
            queue_families,
            available_features,
        });
    }
    output
}

pub(super) struct PhysicalDeviceInfos {
    device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    extended_properties: PhysicalDeviceExtendedProperties,
    queue_families: Vec<vk::QueueFamilyProperties>,
    memory: vk::PhysicalDeviceMemoryProperties,
    available_features: Features,
}

/// Represents additional information related to Physical Devices fetched from
/// `vkGetPhysicalDeviceProperties` call. Certain features available only when
/// appropriate `Instance` extensions enabled. The core extension required
/// for this features is `InstanceExtensions::khr_get_physical_device_properties2`
///
/// TODO: Only a small subset of available properties(https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceProperties2.html) is implemented at this moment.
pub struct PhysicalDeviceExtendedProperties {
    subgroup_size: Option<u32>,
}

impl PhysicalDeviceExtendedProperties {
    pub(super) fn empty() -> Self {
        Self {
            subgroup_size: None,
        }
    }

    /// The default number of invocations in each subgroup
    ///
    /// See https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceSubgroupProperties.html for details
    #[inline]
    pub fn subgroup_size(&self) -> &Option<u32> {
        &self.subgroup_size
    }
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
/// # let instance = Instance::new(None, Version::major_minor(1, 1), &InstanceExtensions::none(), None).unwrap();
/// for physical_device in PhysicalDevice::enumerate(&instance) {
///     print_infos(physical_device);
/// }
///
/// fn print_infos(dev: PhysicalDevice) {
///     println!("Name: {}", dev.name());
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
    /// # let instance = Instance::new(None, Version::major_minor(1, 1), &InstanceExtensions::none(), None).unwrap();
    /// for physical_device in PhysicalDevice::enumerate(&instance) {
    ///     println!("Available device: {}", physical_device.name());
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
    /// let instance = Instance::new(None, Version::major_minor(1, 1), &InstanceExtensions::none(), None).unwrap();
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

    /// Returns the human-readable name of the device.
    #[inline]
    pub fn name(&self) -> &str {
        unsafe {
            let val = &self.infos().properties.deviceName;
            let val = CStr::from_ptr(val.as_ptr());
            val.to_str()
                .expect("physical device name contained non-UTF8 characters")
        }
    }

    /// Returns the type of the device.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vulkano::instance::Instance;
    /// # use vulkano::instance::InstanceExtensions;
    /// # use vulkano::Version;
    /// use vulkano::instance::PhysicalDevice;
    ///
    /// # let instance = Instance::new(None, Version::major_minor(1, 1), &InstanceExtensions::none(), None).unwrap();
    /// for physical_device in PhysicalDevice::enumerate(&instance) {
    ///     println!("Available device: {} (type: {:?})",
    ///               physical_device.name(), physical_device.ty());
    /// }
    /// ```
    #[inline]
    pub fn ty(&self) -> PhysicalDeviceType {
        match self.instance.physical_devices[self.device]
            .properties
            .deviceType
        {
            vk::PHYSICAL_DEVICE_TYPE_OTHER => PhysicalDeviceType::Other,
            vk::PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => PhysicalDeviceType::IntegratedGpu,
            vk::PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => PhysicalDeviceType::DiscreteGpu,
            vk::PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => PhysicalDeviceType::VirtualGpu,
            vk::PHYSICAL_DEVICE_TYPE_CPU => PhysicalDeviceType::Cpu,
            _ => panic!("Unrecognized Vulkan device type"),
        }
    }

    /// Returns the version of Vulkan supported by this device.
    #[inline]
    pub fn api_version(&self) -> Version {
        let val = self.infos().properties.apiVersion;
        Version::from_vulkan_version(val)
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
        if id < self.infos().memory.memoryTypeCount {
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
        if id < self.infos().memory.memoryHeapCount {
            Some(MemoryHeap {
                physical_device: *self,
                id,
            })
        } else {
            None
        }
    }

    /// Gives access to the limits of the physical device.
    ///
    /// This function should be zero-cost in release mode. It only exists to not pollute the
    /// namespace of `PhysicalDevice` with all the limits-related getters.
    #[inline]
    pub fn limits(&self) -> Limits<'a> {
        Limits::from_vk_limits(&self.infos().properties.limits)
    }

    /// Returns an opaque number representing the version of the driver of this device.
    ///
    /// The meaning of this number is implementation-specific. It can be used in bug reports, for
    /// example.
    #[inline]
    pub fn driver_version(&self) -> u32 {
        self.infos().properties.driverVersion
    }

    /// Returns the PCI ID of the device.
    #[inline]
    pub fn pci_device_id(&self) -> u32 {
        self.infos().properties.deviceID
    }

    /// Returns the PCI ID of the vendor.
    #[inline]
    pub fn pci_vendor_id(&self) -> u32 {
        self.infos().properties.vendorID
    }

    /// Returns a unique identifier for the device.
    ///
    /// Can be stored in a configuration file, so that you can retrieve the device again the next
    /// time the program is run.
    #[inline]
    pub fn uuid(&self) -> &[u8; 16] {
        // must be equal to vk::UUID_SIZE
        &self.infos().properties.pipelineCacheUUID
    }

    #[inline]
    pub fn extended_properties(&self) -> &PhysicalDeviceExtendedProperties {
        &self.infos().extended_properties
    }

    // Internal function to make it easier to get the infos of this device.
    #[inline]
    fn infos(&self) -> &'a PhysicalDeviceInfos {
        &self.instance.physical_devices[self.device]
    }
}

unsafe impl<'a> VulkanObject for PhysicalDevice<'a> {
    type Object = vk::PhysicalDevice;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_PHYSICAL_DEVICE;

    #[inline]
    fn internal_object(&self) -> vk::PhysicalDevice {
        self.infos().device
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
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(u32)]
pub enum PhysicalDeviceType {
    /// The device is an integrated GPU.
    IntegratedGpu = 1,
    /// The device is a discrete GPU.
    DiscreteGpu = 2,
    /// The device is a virtual GPU.
    VirtualGpu = 3,
    /// The device is a CPU.
    Cpu = 4,
    /// The device is something else.
    Other = 0,
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
        self.physical_device.infos().queue_families[self.id as usize].queueCount as usize
    }

    /// If timestamps are supported, returns the number of bits supported by timestamp operations.
    /// The returned value will be in the range 36..64.
    /// If timestamps are not supported, returns None.
    #[inline]
    pub fn timestamp_valid_bits(&self) -> Option<u32> {
        let value =
            self.physical_device.infos().queue_families[self.id as usize].timestampValidBits;
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
            .minImageTransferGranularity;
        [granularity.width, granularity.height, granularity.depth]
    }

    /// Returns `true` if queues of this family can execute graphics operations.
    #[inline]
    pub fn supports_graphics(&self) -> bool {
        (self.flags() & vk::QUEUE_GRAPHICS_BIT) != 0
    }

    /// Returns `true` if queues of this family can execute compute operations.
    #[inline]
    pub fn supports_compute(&self) -> bool {
        (self.flags() & vk::QUEUE_COMPUTE_BIT) != 0
    }

    /// Returns `true` if queues of this family can execute transfer operations.
    /// > **Note**: While all queues that can perform graphics or compute operations can implicitly perform
    /// > transfer operations, graphics & compute queues only optionally indicate support for tranfers.
    /// > Many discrete cards will have one queue family that exclusively sets the VK_QUEUE_TRANSFER_BIT
    /// > to indicate a special relationship with the DMA module and more efficient transfers.
    #[inline]
    pub fn explicitly_supports_transfers(&self) -> bool {
        (self.flags() & vk::QUEUE_TRANSFER_BIT) != 0
    }

    /// Returns `true` if queues of this family can execute sparse resources binding operations.
    #[inline]
    pub fn supports_sparse_binding(&self) -> bool {
        (self.flags() & vk::QUEUE_SPARSE_BINDING_BIT) != 0
    }

    /// Returns `true` if the queues of this family support a particular pipeline stage.
    #[inline]
    pub fn supports_stage(&self, stage: PipelineStage) -> bool {
        (self.flags() & stage.required_queue_flags()) != 0
    }

    /// Internal utility function that returns the flags of this queue family.
    #[inline]
    fn flags(&self) -> u32 {
        self.physical_device.infos().queue_families[self.id as usize].queueFlags
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
        let heap_id = self.physical_device.infos().memory.memoryTypes[self.id as usize].heapIndex;
        MemoryHeap {
            physical_device: self.physical_device,
            id: heap_id,
        }
    }

    /// Returns true if the memory type is located on the device, which means that it's the most
    /// efficient for GPU accesses.
    #[inline]
    pub fn is_device_local(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0
    }

    /// Returns true if the memory type can be accessed by the host.
    #[inline]
    pub fn is_host_visible(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0
    }

    /// Returns true if modifications made by the host or the GPU on this memory type are
    /// instantaneously visible to the other party. False means that changes have to be flushed.
    ///
    /// You don't need to worry about this, as this library handles that for you.
    #[inline]
    pub fn is_host_coherent(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0
    }

    /// Returns true if memory of this memory type is cached by the host. Host memory accesses to
    /// cached memory is faster than for uncached memory. However you are not guaranteed that it
    /// is coherent.
    #[inline]
    pub fn is_host_cached(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_HOST_CACHED_BIT) != 0
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
        (self.flags() & vk::MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0
    }

    /// Internal utility function that returns the flags of this queue family.
    #[inline]
    fn flags(&self) -> u32 {
        self.physical_device.infos().memory.memoryTypes[self.id as usize].propertyFlags
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
        if self.current_id >= self.physical_device.infos().memory.memoryTypeCount {
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
        let len = self.physical_device.infos().memory.memoryTypeCount;
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
        self.physical_device.infos().memory.memoryHeaps[self.id as usize].size as usize
    }

    /// Returns true if the heap is local to the GPU.
    #[inline]
    pub fn is_device_local(&self) -> bool {
        let flags = self.physical_device.infos().memory.memoryHeaps[self.id as usize].flags;
        (flags & vk::MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0
    }

    /// Returns true if the heap is multi-instance enabled, that is allocation from such
    /// heap will replicate to each physical-device's instance of heap.
    #[inline]
    pub fn is_multi_instance(&self) -> bool {
        let flags = self.physical_device.infos().memory.memoryHeaps[self.id as usize].flags;
        (flags & vk::MEMORY_HEAP_MULTI_INSTANCE_BIT) != 0
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
        if self.current_id >= self.physical_device.infos().memory.memoryHeapCount {
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
        let len = self.physical_device.infos().memory.memoryHeapCount;
        let remain = (len - self.current_id) as usize;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for MemoryHeapsIter<'a> {}
