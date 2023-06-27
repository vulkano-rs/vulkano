// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Communication channel with a physical device.
//!
//! The `Device` is one of the most important objects of Vulkan. Creating a `Device` is required
//! before you can create buffers, textures, shaders, etc.
//!
//! Basic example:
//!
//! ```no_run
//! use vulkano::{
//!     device::{physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo},
//!     instance::{Instance, InstanceExtensions},
//!     Version, VulkanLibrary,
//! };
//!
//! // Creating the instance. See the documentation of the `instance` module.
//! let library = VulkanLibrary::new()
//!     .unwrap_or_else(|err| panic!("Couldn't load Vulkan library: {:?}", err));
//! let instance = Instance::new(library, Default::default())
//!     .unwrap_or_else(|err| panic!("Couldn't create instance: {:?}", err));
//!
//! // We just choose the first physical device. In a real application you would choose depending
//! // on the capabilities of the physical device and the user's preferences.
//! let physical_device = instance
//!     .enumerate_physical_devices()
//!     .unwrap_or_else(|err| panic!("Couldn't enumerate physical devices: {:?}", err))
//!     .next().expect("No physical device");
//!
//! // Here is the device-creating code.
//! let device = {
//!     let features = Features::empty();
//!     let extensions = DeviceExtensions::empty();
//!
//!     match Device::new(
//!         physical_device,
//!         DeviceCreateInfo {
//!             queue_create_infos: vec![QueueCreateInfo {
//!                 queue_family_index: 0,
//!                 ..Default::default()
//!             }],
//!             enabled_extensions: extensions,
//!             enabled_features: features,
//!             ..Default::default()
//!         },
//!     ) {
//!         Ok(d) => d,
//!         Err(err) => panic!("Couldn't build device: {:?}", err)
//!     }
//! };
//! ```
//!
//! # Features and extensions
//!
//! Two of the parameters that you pass to `Device::new` are the list of the features and the list
//! of extensions to enable on the newly-created device.
//!
//! > **Note**: Device extensions are the same as instance extensions, except for the device.
//! > Features are similar to extensions, except that they are part of the core Vulkan
//! > specifications instead of being separate documents.
//!
//! Some Vulkan capabilities, such as swapchains (that allow you to render on the screen) or
//! geometry shaders for example, require that you enable a certain feature or extension when you
//! create the device. Contrary to OpenGL, you can't use the functions provided by a feature or an
//! extension if you didn't explicitly enable it when creating the device.
//!
//! Not all physical devices support all possible features and extensions. For example mobile
//! devices tend to not support geometry shaders, because their hardware is not capable of it. You
//! can query what is supported with respectively `PhysicalDevice::supported_features` and
//! `DeviceExtensions::supported_by_device`.
//!
//! > **Note**: The fact that you need to manually enable features at initialization also means
//! > that you don't need to worry about a capability not being supported later on in your code.
//!
//! # Queues
//!
//! Each physical device proposes one or more *queues* that are divided in *queue families*. A
//! queue is a thread of execution to which you can submit commands that the GPU will execute.
//!
//! > **Note**: You can think of a queue like a CPU thread. Each queue executes its commands one
//! > after the other, and queues run concurrently. A GPU behaves similarly to the hyper-threading
//! > technology, in the sense that queues will only run partially in parallel.
//!
//! The Vulkan API requires that you specify the list of queues that you are going to use at the
//! same time as when you create the device. This is done in vulkano by passing an iterator where
//! each element is a tuple containing a queue family and a number between 0.0 and 1.0 indicating
//! the priority of execution of the queue relative to the others.
//!
//! TODO: write better doc here
//!
//! The `Device::new` function returns the newly-created device, but also the list of queues.
//!
//! # Extended example
//!
//! TODO: write

use self::physical::PhysicalDevice;
pub(crate) use self::properties::PropertiesFfi;
pub use self::{
    properties::Properties,
    queue::{Queue, QueueFamilyProperties, QueueFlags, QueueGuard},
};
pub use crate::fns::DeviceFunctions;
use crate::{
    acceleration_structure::{
        AccelerationStructureBuildGeometryInfo, AccelerationStructureBuildSizesInfo,
        AccelerationStructureBuildType, AccelerationStructureGeometries,
    },
    descriptor_set::layout::{
        DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorSetLayoutSupport,
    },
    instance::Instance,
    macros::{impl_id_counter, vulkan_bitflags},
    memory::ExternalMemoryHandleType,
    OomError, Requires, RequiresAllOf, RequiresOneOf, RuntimeError, ValidationError, Version,
    VulkanError, VulkanObject,
};
use ash::vk::Handle;
use parking_lot::Mutex;
use smallvec::{smallvec, SmallVec};
use std::{
    ffi::CString,
    fmt::{Debug, Error as FmtError, Formatter},
    fs::File,
    mem::MaybeUninit,
    num::NonZeroU64,
    ops::Deref,
    ptr,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

pub mod physical;
pub(crate) mod properties;
mod queue;

// Generated by build.rs
include!(concat!(env!("OUT_DIR"), "/device_extensions.rs"));
include!(concat!(env!("OUT_DIR"), "/features.rs"));

/// Represents a Vulkan context.
pub struct Device {
    handle: ash::vk::Device,
    physical_device: Arc<PhysicalDevice>, // NOTE: `physical_devices` always contains this
    id: NonZeroU64,

    enabled_extensions: DeviceExtensions,
    enabled_features: Features,
    physical_devices: SmallVec<[Arc<PhysicalDevice>; 2]>,

    // The highest version that is supported for this device.
    // This is the minimum of Instance::max_api_version and PhysicalDevice::api_version.
    api_version: Version,
    fns: DeviceFunctions,
    active_queue_family_indices: SmallVec<[u32; 2]>,

    // This is required for validation in `memory::device_memory`, the count must only be modified
    // in that module.
    pub(crate) allocation_count: AtomicU32,
    fence_pool: Mutex<Vec<ash::vk::Fence>>,
    semaphore_pool: Mutex<Vec<ash::vk::Semaphore>>,
    event_pool: Mutex<Vec<ash::vk::Event>>,
}

impl Device {
    /// Creates a new `Device`.
    #[inline]
    pub fn new(
        physical_device: Arc<PhysicalDevice>,
        create_info: DeviceCreateInfo,
    ) -> Result<(Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>), VulkanError> {
        Self::validate_new(&physical_device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(physical_device, create_info)?) }
    }

    fn validate_new(
        physical_device: &PhysicalDevice,
        create_info: &DeviceCreateInfo,
    ) -> Result<(), ValidationError> {
        create_info
            .validate(physical_device)
            .map_err(|err| err.add_context("create_info"))?;

        let &DeviceCreateInfo {
            queue_create_infos: _,
            enabled_extensions: _,
            enabled_features: _,
            ref physical_devices,
            _ne: _,
        } = create_info;

        if !physical_devices.is_empty()
            && !physical_devices
                .iter()
                .any(|p| p.as_ref() == physical_device)
        {
            return Err(ValidationError {
                problem: "`create_info.physical_devices` is not empty, but does not contain \
                    `physical_device`"
                    .into(),
                vuids: &["VUID-VkDeviceGroupDeviceCreateInfo-physicalDeviceCount-00377"],
                ..Default::default()
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        physical_device: Arc<PhysicalDevice>,
        mut create_info: DeviceCreateInfo,
    ) -> Result<(Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>), RuntimeError> {
        // VUID-vkCreateDevice-ppEnabledExtensionNames-01387
        create_info.enabled_extensions.enable_dependencies(
            physical_device.api_version(),
            physical_device.supported_extensions(),
        );

        // VUID-VkDeviceCreateInfo-pProperties-04451
        if physical_device
            .supported_extensions()
            .khr_portability_subset
        {
            create_info.enabled_extensions.khr_portability_subset = true;
        }

        if physical_device.api_version() >= Version::V1_1 {
            // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-04476
            if create_info.enabled_extensions.khr_shader_draw_parameters {
                create_info.enabled_features.shader_draw_parameters = true;
            }
        }

        if physical_device.api_version() >= Version::V1_2 {
            // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02831
            if create_info.enabled_extensions.khr_draw_indirect_count {
                create_info.enabled_features.draw_indirect_count = true;
            }

            // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02832
            if create_info
                .enabled_extensions
                .khr_sampler_mirror_clamp_to_edge
            {
                create_info.enabled_features.sampler_mirror_clamp_to_edge = true;
            }

            // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02833
            if create_info.enabled_extensions.ext_descriptor_indexing {
                create_info.enabled_features.descriptor_indexing = true;
            }

            // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02834
            if create_info.enabled_extensions.ext_sampler_filter_minmax {
                create_info.enabled_features.sampler_filter_minmax = true;
            }

            // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02835
            if create_info
                .enabled_extensions
                .ext_shader_viewport_index_layer
            {
                create_info.enabled_features.shader_output_viewport_index = true;
                create_info.enabled_features.shader_output_layer = true;
            }
        }

        let &DeviceCreateInfo {
            ref queue_create_infos,
            ref enabled_extensions,
            ref enabled_features,
            ref physical_devices,
            _ne: _,
        } = &create_info;

        let queue_create_infos_vk: SmallVec<[_; 2]> = queue_create_infos
            .iter()
            .map(|queue_create_info| {
                let &QueueCreateInfo {
                    flags,
                    queue_family_index,
                    ref queues,
                    _ne: _,
                } = queue_create_info;

                ash::vk::DeviceQueueCreateInfo {
                    flags: flags.into(),
                    queue_family_index,
                    queue_count: queues.len() as u32,
                    p_queue_priorities: queues.as_ptr(),
                    ..Default::default()
                }
            })
            .collect();

        // Device layers were deprecated in Vulkan 1.0.13, and device layer requests should be
        // ignored by the driver. For backwards compatibility, the spec recommends passing the
        // exact instance layers to the device as well. There's no need to support separate
        // requests at device creation time for legacy drivers: the spec claims that "[at] the
        // time of deprecation there were no known device-only layers."
        //
        // Because there's no way to query the list of layers enabled for an instance, we need
        // to save it alongside the instance. (`vkEnumerateDeviceLayerProperties` should get
        // the right list post-1.0.13, but not pre-1.0.13, so we can't use it here.)
        let enabled_layers_vk: Vec<CString> = physical_device
            .instance()
            .enabled_layers()
            .iter()
            .map(|name| CString::new(name.clone()).unwrap())
            .collect();
        let enabled_layers_ptrs_vk = enabled_layers_vk
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<SmallVec<[_; 2]>>();

        let enabled_extensions_strings_vk = Vec::<CString>::from(enabled_extensions);
        let enabled_extensions_ptrs_vk = enabled_extensions_strings_vk
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<SmallVec<[_; 16]>>();

        let mut features_ffi = FeaturesFfi::default();
        features_ffi.make_chain(
            physical_device.api_version(),
            enabled_extensions,
            physical_device.instance().enabled_extensions(),
        );
        features_ffi.write(enabled_features);

        let has_khr_get_physical_device_properties2 = physical_device.instance().api_version()
            >= Version::V1_1
            || physical_device
                .instance()
                .enabled_extensions()
                .khr_get_physical_device_properties2;

        let mut create_info_vk = ash::vk::DeviceCreateInfo {
            flags: ash::vk::DeviceCreateFlags::empty(),
            queue_create_info_count: queue_create_infos_vk.len() as u32,
            p_queue_create_infos: queue_create_infos_vk.as_ptr(),
            enabled_layer_count: enabled_layers_ptrs_vk.len() as u32,
            pp_enabled_layer_names: enabled_layers_ptrs_vk.as_ptr(),
            enabled_extension_count: enabled_extensions_ptrs_vk.len() as u32,
            pp_enabled_extension_names: enabled_extensions_ptrs_vk.as_ptr(),
            p_enabled_features: ptr::null(),
            ..Default::default()
        };
        let mut device_group_create_info_vk = None;
        let device_group_physical_devices_vk: SmallVec<[_; 2]>;

        // Length of zero and length of one are completely equivalent,
        // so only do anything special here if more than one physical device was given.
        // Spec:
        // A logical device created without using VkDeviceGroupDeviceCreateInfo,
        // or with physicalDeviceCount equal to zero, is equivalent to a physicalDeviceCount of one
        // and pPhysicalDevices pointing to the physicalDevice parameter to vkCreateDevice.
        if physical_devices.len() > 1 {
            device_group_physical_devices_vk = physical_devices
                .iter()
                .map(|physical_device| physical_device.handle())
                .collect();

            let next = device_group_create_info_vk.insert(ash::vk::DeviceGroupDeviceCreateInfo {
                physical_device_count: device_group_physical_devices_vk.len() as u32,
                p_physical_devices: device_group_physical_devices_vk.as_ptr(),
                ..Default::default()
            });

            next.p_next = create_info_vk.p_next;
            create_info_vk.p_next = next as *mut _ as *mut _;
        }

        // VUID-VkDeviceCreateInfo-pNext-00373
        if has_khr_get_physical_device_properties2 {
            create_info_vk.p_next = features_ffi.head_as_ref() as *const _ as _;
        } else {
            create_info_vk.p_enabled_features = &features_ffi.head_as_ref().features;
        }

        let handle = unsafe {
            let fns = physical_device.instance().fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_device)(
                physical_device.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(physical_device, handle, create_info))
    }

    /// Creates a new `Device` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `physical_device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        physical_device: Arc<PhysicalDevice>,
        handle: ash::vk::Device,
        create_info: DeviceCreateInfo,
    ) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
        let DeviceCreateInfo {
            queue_create_infos,
            enabled_features,
            enabled_extensions,
            physical_devices,
            _ne: _,
        } = create_info;

        let api_version = physical_device.api_version();
        let fns = DeviceFunctions::load(|name| unsafe {
            (physical_device.instance().fns().v1_0.get_device_proc_addr)(handle, name.as_ptr())
                .map_or(ptr::null(), |func| func as _)
        });

        struct QueueToGet {
            flags: QueueCreateFlags,
            queue_family_index: u32,
            id: u32,
        }

        let mut active_queue_family_indices: SmallVec<[_; 2]> =
            SmallVec::with_capacity(queue_create_infos.len());
        let mut queues_to_get: SmallVec<[_; 2]> = SmallVec::with_capacity(queue_create_infos.len());

        for queue_create_info in &queue_create_infos {
            let &QueueCreateInfo {
                flags,
                queue_family_index,
                ref queues,
                _ne: _,
            } = queue_create_info;

            active_queue_family_indices.push(queue_family_index);
            queues_to_get.extend((0..queues.len() as u32).map(move |id| QueueToGet {
                flags,
                queue_family_index,
                id,
            }));
        }

        active_queue_family_indices.sort_unstable();
        active_queue_family_indices.dedup();

        let physical_devices = if physical_devices.is_empty() {
            smallvec![physical_device.clone()]
        } else {
            physical_devices
        };

        let device = Arc::new(Device {
            handle,
            physical_device,
            id: Self::next_id(),

            enabled_extensions,
            enabled_features,
            physical_devices,

            api_version,
            fns,
            active_queue_family_indices,

            allocation_count: AtomicU32::new(0),
            fence_pool: Mutex::new(Vec::new()),
            semaphore_pool: Mutex::new(Vec::new()),
            event_pool: Mutex::new(Vec::new()),
        });

        let queues_iter = {
            let device = device.clone();
            queues_to_get.into_iter().map(move |queue_to_get| unsafe {
                let QueueToGet {
                    flags,
                    queue_family_index,
                    id,
                } = queue_to_get;

                Queue::new(device.clone(), flags, queue_family_index, id)
            })
        };

        (device, queues_iter)
    }

    /// Returns the Vulkan version supported by the device.
    ///
    /// This is the lower of the
    /// [physical device's supported version](crate::device::physical::PhysicalDevice::api_version)
    /// and the instance's [`max_api_version`](crate::instance::Instance::max_api_version).
    #[inline]
    pub fn api_version(&self) -> Version {
        self.api_version
    }

    /// Returns pointers to the raw Vulkan functions of the device.
    #[inline]
    pub fn fns(&self) -> &DeviceFunctions {
        &self.fns
    }

    /// Returns the physical device that was used to create this device.
    #[inline]
    pub fn physical_device(&self) -> &Arc<PhysicalDevice> {
        &self.physical_device
    }

    /// Returns the list of physical devices that was used to create this device. The index of
    /// each physical device in this list is its *device index*.
    ///
    /// This always contains the physical device returned by [`physical_device`].
    ///
    /// [`physical_device`]: Self::physical_device
    #[inline]
    pub fn physical_devices(&self) -> &[Arc<PhysicalDevice>] {
        &self.physical_devices
    }

    /// Returns the instance used to create this device.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        self.physical_device.instance()
    }

    /// Returns the queue family indices that this device uses.
    #[inline]
    pub fn active_queue_family_indices(&self) -> &[u32] {
        &self.active_queue_family_indices
    }

    /// Returns the extensions that have been enabled on the device.
    ///
    /// This includes both the extensions specified in [`DeviceCreateInfo::enabled_extensions`],
    /// and any extensions that are required by those extensions.
    #[inline]
    pub fn enabled_extensions(&self) -> &DeviceExtensions {
        &self.enabled_extensions
    }

    /// Returns the features that have been enabled on the device.
    ///
    /// This includes both the features specified in [`DeviceCreateInfo::enabled_features`],
    /// and any features that are required by the enabled extensions.
    #[inline]
    pub fn enabled_features(&self) -> &Features {
        &self.enabled_features
    }

    /// Returns the current number of active [`DeviceMemory`] allocations the device has.
    ///
    /// [`DeviceMemory`]: crate::memory::DeviceMemory
    #[inline]
    pub fn allocation_count(&self) -> u32 {
        self.allocation_count.load(Ordering::Acquire)
    }

    pub(crate) fn fence_pool(&self) -> &Mutex<Vec<ash::vk::Fence>> {
        &self.fence_pool
    }

    pub(crate) fn semaphore_pool(&self) -> &Mutex<Vec<ash::vk::Semaphore>> {
        &self.semaphore_pool
    }

    pub(crate) fn event_pool(&self) -> &Mutex<Vec<ash::vk::Event>> {
        &self.event_pool
    }

    /// For the given acceleration structure build info and primitive counts, returns the
    /// minimum size required to build the acceleration structure, and the minimum size of the
    /// scratch buffer used during the build operation.
    #[inline]
    pub fn acceleration_structure_build_sizes(
        &self,
        build_type: AccelerationStructureBuildType,
        build_info: &AccelerationStructureBuildGeometryInfo,
        max_primitive_counts: &[u32],
    ) -> Result<AccelerationStructureBuildSizesInfo, ValidationError> {
        self.validate_acceleration_structure_build_sizes(
            build_type,
            build_info,
            max_primitive_counts,
        )?;

        unsafe {
            Ok(self.acceleration_structure_build_sizes_unchecked(
                build_type,
                build_info,
                max_primitive_counts,
            ))
        }
    }

    fn validate_acceleration_structure_build_sizes(
        &self,
        build_type: AccelerationStructureBuildType,
        build_info: &AccelerationStructureBuildGeometryInfo,
        max_primitive_counts: &[u32],
    ) -> Result<(), ValidationError> {
        if !self.enabled_extensions().khr_acceleration_structure {
            return Err(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_acceleration_structure",
                )])]),
                ..Default::default()
            });
        }

        if !self.enabled_features().acceleration_structure {
            return Err(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "acceleration_structure",
                )])]),
                vuids: &[
                    "VUID-vkGetAccelerationStructureBuildSizesKHR-accelerationStructure-08933",
                ],
                ..Default::default()
            });
        }

        build_type
            .validate_device(self)
            .map_err(|err| ValidationError {
                context: "build_type".into(),
                vuids: &["VUID-vkGetAccelerationStructureBuildSizesKHR-buildType-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        // VUID-vkGetAccelerationStructureBuildSizesKHR-pBuildInfo-parameter
        build_info
            .validate(self)
            .map_err(|err| err.add_context("build_info"))?;

        let max_primitive_count = self
            .physical_device()
            .properties()
            .max_primitive_count
            .unwrap();
        let max_instance_count = self
            .physical_device()
            .properties()
            .max_instance_count
            .unwrap();

        let geometry_count = match &build_info.geometries {
            AccelerationStructureGeometries::Triangles(geometries) => {
                for (index, &primitive_count) in max_primitive_counts.iter().enumerate() {
                    if primitive_count as u64 > max_primitive_count {
                        return Err(ValidationError {
                            context: format!("max_primitive_counts[{}]", index).into(),
                            problem: "exceeds the `max_primitive_count` limit".into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03795"],
                            ..Default::default()
                        });
                    }
                }

                geometries.len()
            }
            AccelerationStructureGeometries::Aabbs(geometries) => {
                for (index, &primitive_count) in max_primitive_counts.iter().enumerate() {
                    if primitive_count as u64 > max_primitive_count {
                        return Err(ValidationError {
                            context: format!("max_primitive_counts[{}]", index).into(),
                            problem: "exceeds the `max_primitive_count` limit".into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03794"],
                            ..Default::default()
                        });
                    }
                }

                geometries.len()
            }
            AccelerationStructureGeometries::Instances(_) => {
                for (index, &instance_count) in max_primitive_counts.iter().enumerate() {
                    if instance_count as u64 > max_instance_count {
                        return Err(ValidationError {
                            context: format!("max_primitive_counts[{}]", index).into(),
                            problem: "exceeds the `max_instance_count` limit".into(),
                            vuids: &[
                                "VUID-vkGetAccelerationStructureBuildSizesKHR-pBuildInfo-03785",
                            ],
                            ..Default::default()
                        });
                    }
                }

                1
            }
        };

        if max_primitive_counts.len() != geometry_count {
            return Err(ValidationError {
                problem: "`build_info.geometries` and `max_primitive_counts` \
                    do not have the same length"
                    .into(),
                vuids: &["VUID-vkGetAccelerationStructureBuildSizesKHR-pBuildInfo-03619"],
                ..Default::default()
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn acceleration_structure_build_sizes_unchecked(
        &self,
        build_type: AccelerationStructureBuildType,
        build_info: &AccelerationStructureBuildGeometryInfo,
        max_primitive_counts: &[u32],
    ) -> AccelerationStructureBuildSizesInfo {
        let (mut build_info_vk, geometries_vk) = build_info.to_vulkan();
        build_info_vk = ash::vk::AccelerationStructureBuildGeometryInfoKHR {
            geometry_count: geometries_vk.len() as u32,
            p_geometries: geometries_vk.as_ptr(),
            ..build_info_vk
        };

        let mut build_sizes_info_vk = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();

        let fns = self.fns();
        (fns.khr_acceleration_structure
            .get_acceleration_structure_build_sizes_khr)(
            self.handle,
            build_type.into(),
            &build_info_vk,
            max_primitive_counts.as_ptr(),
            &mut build_sizes_info_vk,
        );

        AccelerationStructureBuildSizesInfo {
            acceleration_structure_size: build_sizes_info_vk.acceleration_structure_size,
            update_scratch_size: build_sizes_info_vk.update_scratch_size,
            build_scratch_size: build_sizes_info_vk.build_scratch_size,
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Returns whether a serialized acceleration structure with the specified version data
    /// is compatible with this device.
    #[inline]
    pub fn acceleration_structure_is_compatible(
        &self,
        version_data: &[u8; 2 * ash::vk::UUID_SIZE],
    ) -> Result<bool, ValidationError> {
        self.validate_acceleration_structure_is_compatible(version_data)?;

        unsafe { Ok(self.acceleration_structure_is_compatible_unchecked(version_data)) }
    }

    fn validate_acceleration_structure_is_compatible(
        &self,
        _version_data: &[u8; 2 * ash::vk::UUID_SIZE],
    ) -> Result<(), ValidationError> {
        if !self.enabled_extensions().khr_acceleration_structure {
            return Err(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_acceleration_structure",
                )])]),
                ..Default::default()
            });
        }

        if !self.enabled_features().acceleration_structure {
            return Err(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::Feature("acceleration_structure")]),
                ]),
                vuids: &["VUID-vkGetDeviceAccelerationStructureCompatibilityKHR-accelerationStructure-08928"],
                ..Default::default()
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn acceleration_structure_is_compatible_unchecked(
        &self,
        version_data: &[u8; 2 * ash::vk::UUID_SIZE],
    ) -> bool {
        let version_info_vk = ash::vk::AccelerationStructureVersionInfoKHR {
            p_version_data: version_data,
            ..Default::default()
        };
        let mut compatibility_vk = ash::vk::AccelerationStructureCompatibilityKHR::default();

        let fns = self.fns();
        (fns.khr_acceleration_structure
            .get_device_acceleration_structure_compatibility_khr)(
            self.handle,
            &version_info_vk,
            &mut compatibility_vk,
        );

        compatibility_vk == ash::vk::AccelerationStructureCompatibilityKHR::COMPATIBLE
    }

    /// Returns whether a descriptor set layout with the given `create_info` could be created
    /// on the device, and additional supported properties where relevant. `Some` is returned if
    /// the descriptor set layout is supported, `None` if it is not.
    ///
    /// This is primarily useful for checking whether the device supports a descriptor set layout
    /// that goes beyond the [`max_per_set_descriptors`] limit. A layout that does not exceed
    /// that limit is guaranteed to be supported, otherwise this function can be called.
    ///
    /// The device API version must be at least 1.1, or the [`khr_maintenance3`] extension must
    /// be enabled on the device.
    ///
    /// [`max_per_set_descriptors`]: crate::device::Properties::max_per_set_descriptors
    /// [`khr_maintenance3`]: crate::device::DeviceExtensions::khr_maintenance3
    #[inline]
    pub fn descriptor_set_layout_support(
        &self,
        create_info: &DescriptorSetLayoutCreateInfo,
    ) -> Result<Option<DescriptorSetLayoutSupport>, ValidationError> {
        self.validate_descriptor_set_layout_support(create_info)?;

        unsafe { Ok(self.descriptor_set_layout_support_unchecked(create_info)) }
    }

    fn validate_descriptor_set_layout_support(
        &self,
        create_info: &DescriptorSetLayoutCreateInfo,
    ) -> Result<(), ValidationError> {
        if !(self.api_version() >= Version::V1_1 || self.enabled_extensions().khr_maintenance3) {
            return Err(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance3")]),
                ]),
                ..Default::default()
            });
        }

        // VUID-vkGetDescriptorSetLayoutSupport-pCreateInfo-parameter
        create_info
            .validate(self)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn descriptor_set_layout_support_unchecked(
        &self,
        create_info: &DescriptorSetLayoutCreateInfo,
    ) -> Option<DescriptorSetLayoutSupport> {
        let &DescriptorSetLayoutCreateInfo {
            flags,
            ref bindings,
            _ne: _,
        } = create_info;

        struct PerBinding {
            immutable_samplers_vk: Vec<ash::vk::Sampler>,
        }

        let mut bindings_vk = Vec::with_capacity(bindings.len());
        let mut per_binding_vk = Vec::with_capacity(bindings.len());
        let mut binding_flags_info_vk = None;
        let mut binding_flags_vk = Vec::with_capacity(bindings.len());

        let mut support_vk = ash::vk::DescriptorSetLayoutSupport::default();
        let mut variable_descriptor_count_support_vk = None;

        for (&binding_num, binding) in bindings.iter() {
            let &DescriptorSetLayoutBinding {
                binding_flags,
                descriptor_type,
                descriptor_count,
                stages,
                ref immutable_samplers,
                _ne: _,
            } = binding;

            bindings_vk.push(ash::vk::DescriptorSetLayoutBinding {
                binding: binding_num,
                descriptor_type: descriptor_type.into(),
                descriptor_count,
                stage_flags: stages.into(),
                p_immutable_samplers: ptr::null(),
            });
            per_binding_vk.push(PerBinding {
                immutable_samplers_vk: immutable_samplers
                    .iter()
                    .map(VulkanObject::handle)
                    .collect(),
            });
            binding_flags_vk.push(binding_flags.into());
        }

        for (binding_vk, per_binding_vk) in bindings_vk.iter_mut().zip(per_binding_vk.iter_mut()) {
            binding_vk.p_immutable_samplers = per_binding_vk.immutable_samplers_vk.as_ptr();
        }

        let mut create_info_vk = ash::vk::DescriptorSetLayoutCreateInfo {
            flags: flags.into(),
            binding_count: bindings_vk.len() as u32,
            p_bindings: bindings_vk.as_ptr(),
            ..Default::default()
        };

        if self.api_version() >= Version::V1_2 || self.enabled_extensions().ext_descriptor_indexing
        {
            let next =
                binding_flags_info_vk.insert(ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo {
                    binding_count: binding_flags_vk.len() as u32,
                    p_binding_flags: binding_flags_vk.as_ptr(),
                    ..Default::default()
                });

            next.p_next = create_info_vk.p_next;
            create_info_vk.p_next = next as *const _ as *const _;

            let next = variable_descriptor_count_support_vk
                .insert(ash::vk::DescriptorSetVariableDescriptorCountLayoutSupport::default());

            next.p_next = support_vk.p_next;
            support_vk.p_next = next as *mut _ as *mut _;
        }

        let fns = self.fns();

        if self.api_version() >= Version::V1_1 {
            (fns.v1_1.get_descriptor_set_layout_support)(
                self.handle(),
                &create_info_vk,
                &mut support_vk,
            )
        } else {
            (fns.khr_maintenance3.get_descriptor_set_layout_support_khr)(
                self.handle(),
                &create_info_vk,
                &mut support_vk,
            )
        }

        (support_vk.supported != ash::vk::FALSE).then(|| DescriptorSetLayoutSupport {
            max_variable_descriptor_count: variable_descriptor_count_support_vk
                .map_or(0, |s| s.max_variable_descriptor_count),
        })
    }

    /// Retrieves the properties of an external file descriptor when imported as a given external
    /// handle type.
    ///
    /// An error will be returned if the
    /// [`khr_external_memory_fd`](DeviceExtensions::khr_external_memory_fd) extension was not
    /// enabled on the device, or if `handle_type` is [`ExternalMemoryHandleType::OpaqueFd`].
    ///
    /// # Safety
    ///
    /// - `file` must be a handle to external memory that was created outside the Vulkan API.
    #[cfg_attr(not(unix), allow(unused_variables))]
    #[inline]
    pub unsafe fn memory_fd_properties(
        &self,
        handle_type: ExternalMemoryHandleType,
        file: File,
    ) -> Result<MemoryFdProperties, VulkanError> {
        self.validate_memory_fd_properties(handle_type, &file)?;

        Ok(self.memory_fd_properties_unchecked(handle_type, file)?)
    }

    fn validate_memory_fd_properties(
        &self,
        handle_type: ExternalMemoryHandleType,
        _file: &File,
    ) -> Result<(), ValidationError> {
        if !self.enabled_extensions().khr_external_memory_fd {
            return Err(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_memory_fd",
                )])]),
                ..Default::default()
            });
        }

        handle_type
            .validate_device(self)
            .map_err(|err| ValidationError {
                context: "handle_type".into(),
                vuids: &["VUID-vkGetMemoryFdPropertiesKHR-handleType-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if handle_type == ExternalMemoryHandleType::OpaqueFd {
            return Err(ValidationError {
                context: "handle_type".into(),
                problem: "is `ExternalMemoryHandleType::OpaqueFd`".into(),
                vuids: &["VUID-vkGetMemoryFdPropertiesKHR-handleType-00674"],
                ..Default::default()
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[cfg_attr(not(unix), allow(unused_variables))]
    pub unsafe fn memory_fd_properties_unchecked(
        &self,
        handle_type: ExternalMemoryHandleType,
        file: File,
    ) -> Result<MemoryFdProperties, RuntimeError> {
        #[cfg(not(unix))]
        unreachable!("`khr_external_memory_fd` was somehow enabled on a non-Unix system");

        #[cfg(unix)]
        {
            use std::os::unix::io::IntoRawFd;

            let mut memory_fd_properties = ash::vk::MemoryFdPropertiesKHR::default();

            let fns = self.fns();
            (fns.khr_external_memory_fd.get_memory_fd_properties_khr)(
                self.handle,
                handle_type.into(),
                file.into_raw_fd(),
                &mut memory_fd_properties,
            )
            .result()
            .map_err(RuntimeError::from)?;

            Ok(MemoryFdProperties {
                memory_type_bits: memory_fd_properties.memory_type_bits,
            })
        }
    }

    /// Assigns a human-readable name to `object` for debugging purposes.
    ///
    /// If `object_name` is `None`, a previously set object name is removed.
    ///
    /// # Panics
    /// - If `object` is not owned by this device.
    pub fn set_debug_utils_object_name<T: VulkanObject + DeviceOwned>(
        &self,
        object: &T,
        object_name: Option<&str>,
    ) -> Result<(), OomError> {
        assert!(object.device().handle() == self.handle());

        let object_name_vk = object_name.map(|object_name| CString::new(object_name).unwrap());
        let info = ash::vk::DebugUtilsObjectNameInfoEXT {
            object_type: T::Handle::TYPE,
            object_handle: object.handle().as_raw(),
            p_object_name: object_name_vk.map_or(ptr::null(), |object_name| object_name.as_ptr()),
            ..Default::default()
        };

        unsafe {
            let fns = self.instance().fns();
            (fns.ext_debug_utils.set_debug_utils_object_name_ext)(self.handle, &info)
                .result()
                .map_err(RuntimeError::from)?;
        }

        Ok(())
    }

    /// Waits until all work on this device has finished. You should never need to call
    /// this function, but it can be useful for debugging or benchmarking purposes.
    ///
    /// > **Note**: This is the Vulkan equivalent of OpenGL's `glFinish`.
    ///
    /// # Safety
    ///
    /// This function is not thread-safe. You must not submit anything to any of the queue
    /// of the device (either explicitly or implicitly, for example with a future's destructor)
    /// while this function is waiting.
    #[inline]
    pub unsafe fn wait_idle(&self) -> Result<(), OomError> {
        let fns = self.fns();
        (fns.v1_0.device_wait_idle)(self.handle)
            .result()
            .map_err(RuntimeError::from)?;

        Ok(())
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let Self {
            handle,
            physical_device,
            id,
            enabled_extensions,
            enabled_features,
            physical_devices,
            api_version,
            fns,
            active_queue_family_indices,
            allocation_count,
            fence_pool: _,
            semaphore_pool: _,
            event_pool: _,
        } = self;

        struct DebugPhysicalDevice(NonZeroU64);

        impl Debug for DebugPhysicalDevice {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
                write!(f, "<physical device #{}>", self.0)
            }
        }

        // A physical device prints pages of poetry when debug-printed. We don't want to spam the
        // user's console/logs with all that multiple times for each logical device. The user can
        // still debug-print the physical device(s) manually if they need to.
        let physical_device = DebugPhysicalDevice(physical_device.id());
        let physical_devices = physical_devices
            .iter()
            .map(|p| DebugPhysicalDevice(p.id()))
            .collect::<SmallVec<[_; 2]>>();

        f.debug_struct("Device")
            .field("handle", handle)
            .field("physical_device", &physical_device)
            .field("id", id)
            .field("enabled_extensions", enabled_extensions)
            .field("enabled_features", enabled_features)
            .field("physical_devices", &physical_devices)
            .field("api_version", api_version)
            .field("fns", fns)
            .field("active_queue_family_indices", active_queue_family_indices)
            .field("allocation_count", allocation_count)
            .finish_non_exhaustive()
    }
}

impl Drop for Device {
    #[inline]
    fn drop(&mut self) {
        let fns = self.fns();

        unsafe {
            for &raw_fence in self.fence_pool.lock().iter() {
                (fns.v1_0.destroy_fence)(self.handle, raw_fence, ptr::null());
            }
            for &raw_sem in self.semaphore_pool.lock().iter() {
                (fns.v1_0.destroy_semaphore)(self.handle, raw_sem, ptr::null());
            }
            for &raw_event in self.event_pool.lock().iter() {
                (fns.v1_0.destroy_event)(self.handle, raw_event, ptr::null());
            }
            (fns.v1_0.destroy_device)(self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for Device {
    type Handle = ash::vk::Device;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

impl_id_counter!(Device);

/// Parameters to create a new `Device`.
#[derive(Clone, Debug)]
pub struct DeviceCreateInfo {
    /// The queues to create for the device.
    ///
    /// The default value is empty, which must be overridden.
    pub queue_create_infos: Vec<QueueCreateInfo>,

    /// The extensions to enable on the device.
    ///
    /// You only need to enable the extensions that you need. If the extensions you specified
    /// require additional extensions to be enabled, they will be automatically enabled as well.
    ///
    /// If the [`khr_portability_subset`](DeviceExtensions::khr_portability_subset) extension is
    /// available, it will be enabled automatically, so you do not have to do this yourself.
    /// You are responsible for ensuring that your program can work correctly on such devices.
    /// See [the documentation of the `instance` module](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// for more information.
    ///
    /// The default value is [`DeviceExtensions::empty()`].
    pub enabled_extensions: DeviceExtensions,

    /// The features to enable on the device.
    ///
    /// You only need to enable the features that you need. If the extensions you specified
    /// require certain features to be enabled, they will be automatically enabled as well.
    ///
    /// The default value is [`Features::empty()`].
    pub enabled_features: Features,

    /// A list of physical devices to create this device from, to act together as a single
    /// logical device. The physical devices must all belong to the same device group, as returned
    /// by [`Instance::enumerate_physical_device_groups`], and a physical device must not appear
    /// in the list more than once.
    ///
    /// The index of each physical device in this list becomes that physical
    /// device's *device index*, which can be used in other Vulkan functions to specify
    /// a particular physical device within the group.
    /// If the list is left empty, then it behaves as if it contained the physical device,
    /// that was passed to the `physical_device` parameter of [`Device::new`], as its only element.
    /// Otherwise, that physical device must always be part of the list, but it does not need to
    /// be the first element.
    ///
    /// If the list contains more than one physical device, the instance API version must be at
    /// least 1.1, or the [`khr_device_group_creation`] extension must be enabled on the instance.
    /// In order to use any device-level functionality for dealing with device groups,
    /// the physical device API version should also be at least 1.1,
    /// or `enabled_extensions` should contain [`khr_device_group`].
    ///
    /// The default value is empty.
    ///
    /// [`Instance::enumerate_physical_device_groups`]: Instance::enumerate_physical_device_groups
    /// [`khr_device_group_creation`]: crate::instance::InstanceExtensions::khr_device_group_creation
    /// [`khr_device_group`]: crate::device::DeviceExtensions::khr_device_group
    pub physical_devices: SmallVec<[Arc<PhysicalDevice>; 2]>,

    pub _ne: crate::NonExhaustive,
}

impl Default for DeviceCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            queue_create_infos: Vec::new(),
            enabled_extensions: DeviceExtensions::empty(),
            enabled_features: Features::empty(),
            physical_devices: SmallVec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl DeviceCreateInfo {
    pub(crate) fn validate(&self, physical_device: &PhysicalDevice) -> Result<(), ValidationError> {
        let &Self {
            ref queue_create_infos,
            ref enabled_extensions,
            ref enabled_features,
            ref physical_devices,
            _ne: _,
        } = self;

        if queue_create_infos.is_empty() {
            return Err(ValidationError {
                context: "queue_create_infos".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkDeviceCreateInfo-queueCreateInfoCount-arraylength"],
                ..Default::default()
            });
        }

        for (index, queue_create_info) in queue_create_infos.iter().enumerate() {
            queue_create_info
                .validate(physical_device, enabled_extensions, enabled_features)
                .map_err(|err| err.add_context(format!("queue_create_infos[{}]", index)))?;

            let &QueueCreateInfo {
                flags: _,
                queue_family_index,
                queues: _,
                _ne: _,
            } = queue_create_info;

            if queue_create_infos
                .iter()
                .filter(|qc2| qc2.queue_family_index == queue_family_index)
                .count()
                != 1
            {
                return Err(ValidationError {
                    problem: format!(
                        "`queue_create_infos[{}].queue_family_index` occurs more than once in \
                        `queue_create_infos`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-queueFamilyIndex-02802"],
                    ..Default::default()
                });
            }
        }

        enabled_extensions
            .check_requirements(
                physical_device.supported_extensions(),
                physical_device.api_version(),
                physical_device.instance().enabled_extensions(),
            )
            .map_err(|err| ValidationError {
                context: "enabled_extensions".into(),
                problem: err.to_string().into(),
                vuids: &["VUID-vkCreateDevice-ppEnabledExtensionNames-01387"],
                ..Default::default()
            })?;

        enabled_features
            .check_requirements(physical_device.supported_features())
            .map_err(|err| ValidationError {
                context: "enabled_features".into(),
                problem: err.to_string().into(),
                ..Default::default()
            })?;

        let mut dependency_extensions = *enabled_extensions;
        dependency_extensions.enable_dependencies(
            physical_device.api_version(),
            physical_device.supported_extensions(),
        );

        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-01840
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-00374
        // Ensured because `DeviceExtensions` doesn't contain obsoleted extensions.

        if enabled_extensions.ext_buffer_device_address {
            if enabled_extensions.khr_buffer_device_address {
                return Err(ValidationError {
                    context: "enabled_extensions".into(),
                    problem: "contains `khr_buffer_device_address`, \
                        but also contains `ext_buffer_device_address`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-03328"],
                    ..Default::default()
                });
            } else if dependency_extensions.khr_buffer_device_address {
                return Err(ValidationError {
                    context: "enabled_extensions".into(),
                    problem: "contains an extension that requires `khr_buffer_device_address`, \
                        but also contains `ext_buffer_device_address`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-03328"],
                    ..Default::default()
                });
            }

            if physical_device.api_version() >= Version::V1_2
                && enabled_features.buffer_device_address
            {
                return Err(ValidationError {
                    problem: "the physical device API version is at least 1.2, \
                    `enabled_features` contains `buffer_device_address`, and \
                    `enabled_extensions` contains `ext_buffer_device_address`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-pNext-04748"],
                    ..Default::default()
                });
            }
        }

        if enabled_features.shading_rate_image {
            if enabled_features.pipeline_fragment_shading_rate {
                return Err(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `shading_rate_image` and \
                        `pipeline_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04478"],
                    ..Default::default()
                });
            }

            if enabled_features.primitive_fragment_shading_rate {
                return Err(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `shading_rate_image` and \
                        `primitive_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04479"],
                    ..Default::default()
                });
            }

            if enabled_features.attachment_fragment_shading_rate {
                return Err(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `shading_rate_image` and \
                        `attachment_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04480"],
                    ..Default::default()
                });
            }
        }

        if enabled_features.fragment_density_map {
            if enabled_features.pipeline_fragment_shading_rate {
                return Err(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `fragment_density_map` and \
                        `pipeline_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04481"],
                    ..Default::default()
                });
            }

            if enabled_features.primitive_fragment_shading_rate {
                return Err(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `fragment_density_map` and \
                        `primitive_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04482"],
                    ..Default::default()
                });
            }

            if enabled_features.attachment_fragment_shading_rate {
                return Err(ValidationError {
                    context: "enabled_features".into(),
                    problem: "contains both `fragment_density_map` and \
                        `attachment_fragment_shading_rate`"
                        .into(),
                    vuids: &["VUID-VkDeviceCreateInfo-shadingRateImage-04483"],
                    ..Default::default()
                });
            }
        }

        if enabled_features.sparse_image_int64_atomics
            && !enabled_features.shader_image_int64_atomics
        {
            return Err(ValidationError {
                context: "enabled_features".into(),
                problem: "contains `sparse_image_int64_atomics`, but does not contain \
                    `shader_image_int64_atomics`"
                    .into(),
                vuids: &["VUID-VkDeviceCreateInfo-None-04896"],
                ..Default::default()
            });
        }

        if enabled_features.sparse_image_float32_atomics
            && !enabled_features.shader_image_float32_atomics
        {
            return Err(ValidationError {
                context: "enabled_features".into(),
                problem: "contains `sparse_image_float32_atomics`, but does not contain \
                    `shader_image_float32_atomics`"
                    .into(),
                vuids: &["VUID-VkDeviceCreateInfo-None-04897"],
                ..Default::default()
            });
        }

        if enabled_features.sparse_image_float32_atomic_add
            && !enabled_features.shader_image_float32_atomic_add
        {
            return Err(ValidationError {
                context: "enabled_features".into(),
                problem: "contains `sparse_image_float32_atomic_add`, but does not contain \
                    `shader_image_float32_atomic_add`"
                    .into(),
                vuids: &["VUID-VkDeviceCreateInfo-None-04898"],
                ..Default::default()
            });
        }

        if enabled_features.sparse_image_float32_atomic_min_max
            && !enabled_features.shader_image_float32_atomic_min_max
        {
            return Err(ValidationError {
                context: "enabled_features".into(),
                problem: "contains `sparse_image_float32_atomic_min_max`, but does not contain \
                    `shader_image_float32_atomic_min_max`"
                    .into(),
                vuids: &["VUID-VkDeviceCreateInfo-sparseImageFloat32AtomicMinMax-04975"],
                ..Default::default()
            });
        }

        if enabled_features.descriptor_buffer && enabled_extensions.amd_shader_fragment_mask {
            return Err(ValidationError {
                problem: "`enabled_features` contains `descriptor_buffer`, and \
                    `enabled_extensions` contains `amd_shader_fragment_mask`"
                    .into(),
                vuids: &["VUID-VkDeviceCreateInfo-None-08095"],
                ..Default::default()
            });
        }

        if physical_devices.len() > 1 {
            for (index, group_physical_device) in physical_devices.iter().enumerate() {
                if physical_devices[..index].contains(group_physical_device) {
                    return Err(ValidationError {
                        context: "physical_devices".into(),
                        problem: format!(
                            "the physical device at index {} is contained in the list more than \
                            once",
                            index,
                        )
                        .into(),
                        vuids: &["VUID-VkDeviceGroupDeviceCreateInfo-pPhysicalDevices-00375"],
                        ..Default::default()
                    });
                }
            }

            if !(physical_device.instance().api_version() >= Version::V1_1
                || physical_device
                    .instance()
                    .enabled_extensions()
                    .khr_device_group_creation)
            {
                return Err(ValidationError {
                    context: "physical_devices".into(),
                    problem: "the length is greater than 1".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::InstanceExtension("khr_device_group_creation")]),
                    ]),
                    ..Default::default()
                });
            }

            if !physical_device
                .instance()
                .is_same_device_group(physical_devices.iter().map(AsRef::as_ref))
            {
                return Err(ValidationError {
                    context: "physical_devices".into(),
                    problem: "the physical devices do not all belong to the same device group"
                        .into(),
                    vuids: &["VUID-VkDeviceGroupDeviceCreateInfo-pPhysicalDevices-00376"],
                    ..Default::default()
                });
            }
        }

        Ok(())
    }
}

/// Parameters to create queues in a new `Device`.
#[derive(Clone, Debug)]
pub struct QueueCreateInfo {
    /// Additional properties of the queue.
    ///
    /// The default value is empty.
    pub flags: QueueCreateFlags,

    /// The index of the queue family to create queues for.
    ///
    /// The default value is `0`.
    pub queue_family_index: u32,

    /// The queues to create for the given queue family, each with a relative priority.
    ///
    /// The relative priority value is an arbitrary number between 0.0 and 1.0. Giving a queue a
    /// higher priority is a hint to the driver that the queue should be given more processing time.
    /// As this is only a hint, different drivers may handle this value differently and there are no
    /// guarantees about its behavior.
    ///
    /// The default value is a single queue with a priority of 0.5.
    pub queues: Vec<f32>,

    pub _ne: crate::NonExhaustive,
}

impl Default for QueueCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: QueueCreateFlags::empty(),
            queue_family_index: 0,
            queues: vec![0.5],
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl QueueCreateInfo {
    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
        device_extensions: &DeviceExtensions,
        device_features: &Features,
    ) -> Result<(), ValidationError> {
        let &Self {
            flags,
            queue_family_index,
            ref queues,
            _ne: _,
        } = self;

        flags
            .validate_device_raw(
                physical_device.api_version(),
                device_features,
                device_extensions,
                physical_device.instance().enabled_extensions(),
            )
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkDeviceQueueCreateInfo-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        let queue_family_properties = physical_device
            .queue_family_properties()
            .get(queue_family_index as usize)
            .ok_or(ValidationError {
                context: "queue_family_index".into(),
                problem: "is not less than the number of queue families in the physical device"
                    .into(),
                vuids: &["VUID-VkDeviceQueueCreateInfo-queueFamilyIndex-00381"],
                ..Default::default()
            })?;

        if queues.is_empty() {
            return Err(ValidationError {
                context: "queues".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkDeviceQueueCreateInfo-queueCount-arraylength"],
                ..Default::default()
            });
        }

        if queues.len() > queue_family_properties.queue_count as usize {
            return Err(ValidationError {
                problem: "the length of `queues` is greater than the number of queues in the
                    queue family indicated by `queue_family_index`"
                    .into(),
                vuids: &["VUID-VkDeviceQueueCreateInfo-queueCount-00382"],
                ..Default::default()
            });
        }

        for (index, &priority) in queues.iter().enumerate() {
            if !(0.0..=1.0).contains(&priority) {
                return Err(ValidationError {
                    context: format!("queues[{}]", index).into(),
                    problem: "is not between 0.0 and 1.0 inclusive".into(),
                    vuids: &["VUID-VkDeviceQueueCreateInfo-pQueuePriorities-00383"],
                    ..Default::default()
                });
            }
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a queue.
    QueueCreateFlags = DeviceQueueCreateFlags(u32);

    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),
}

/// Implemented on objects that belong to a Vulkan device.
///
/// # Safety
///
/// - `device()` must return the correct device.
pub unsafe trait DeviceOwned {
    /// Returns the device that owns `Self`.
    fn device(&self) -> &Arc<Device>;
}

unsafe impl<T> DeviceOwned for T
where
    T: Deref,
    T::Target: DeviceOwned,
{
    fn device(&self) -> &Arc<Device> {
        (**self).device()
    }
}

/// The properties of a Unix file descriptor when it is imported.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MemoryFdProperties {
    /// A bitmask of the indices of memory types that can be used with the file.
    pub memory_type_bits: u32,
}

#[cfg(test)]
mod tests {
    use crate::device::{Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo};
    use std::{ffi::CString, sync::Arc};

    #[test]
    fn empty_extensions() {
        let d: Vec<CString> = (&DeviceExtensions::empty()).into();
        assert!(d.get(0).is_none());
    }

    #[test]
    fn extensions_into_iter() {
        let extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        for (name, enabled) in extensions {
            if name == "VK_KHR_swapchain" {
                assert!(enabled);
            } else {
                assert!(!enabled);
            }
        }
    }

    #[test]
    fn features_into_iter() {
        let features = Features {
            tessellation_shader: true,
            ..Features::empty()
        };
        for (name, enabled) in features {
            if name == "tessellationShader" {
                assert!(enabled);
            } else {
                assert!(!enabled);
            }
        }
    }

    #[test]
    fn one_ref() {
        let (mut device, _) = gfx_dev_and_queue!();
        assert!(Arc::get_mut(&mut device).is_some());
    }

    #[test]
    fn too_many_queues() {
        let instance = instance!();
        let physical_device = match instance.enumerate_physical_devices().unwrap().next() {
            Some(p) => p,
            None => return,
        };

        let queue_family_index = 0;
        let queue_family_properties =
            &physical_device.queue_family_properties()[queue_family_index as usize];
        let queues = (0..queue_family_properties.queue_count + 1)
            .map(|_| (0.5))
            .collect();

        if Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    queues,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .is_ok()
        {
            panic!()
        }
    }

    #[test]
    fn unsupported_features() {
        let instance = instance!();
        let physical_device = match instance.enumerate_physical_devices().unwrap().next() {
            Some(p) => p,
            None => return,
        };

        let features = Features::all();
        // In the unlikely situation where the device supports everything, we ignore the test.
        if physical_device.supported_features().contains(&features) {
            return;
        }

        if Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                enabled_features: features,
                ..Default::default()
            },
        )
        .is_ok()
        {
            panic!()
        }
    }

    #[test]
    fn priority_out_of_range() {
        let instance = instance!();
        let physical_device = match instance.enumerate_physical_devices().unwrap().next() {
            Some(p) => p,
            None => return,
        };

        if Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    queues: vec![1.4],
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .is_ok()
        {
            panic!();
        }

        if Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    queues: vec![-0.2],
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .is_ok()
        {
            panic!();
        }
    }
}
