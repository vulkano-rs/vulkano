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
//! use vulkano::device::physical::PhysicalDevice;
//! use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo};
//! use vulkano::instance::{Instance, InstanceExtensions};
//! use vulkano::Version;
//!
//! // Creating the instance. See the documentation of the `instance` module.
//! let instance = match Instance::new(Default::default()) {
//!     Ok(i) => i,
//!     Err(err) => panic!("Couldn't build instance: {:?}", err)
//! };
//!
//! // We just choose the first physical device. In a real application you would choose depending
//! // on the capabilities of the physical device and the user's preferences.
//! let physical_device = PhysicalDevice::enumerate(&instance).next().expect("No physical device");
//!
//! // Here is the device-creating code.
//! let device = {
//!     let queue_family = physical_device.queue_families().next().unwrap();
//!     let features = Features::none();
//!     let extensions = DeviceExtensions::none();
//!
//!     match Device::new(
//!         physical_device,
//!         DeviceCreateInfo {
//!             enabled_extensions: extensions,
//!             enabled_features: features,
//!             queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
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

pub(crate) use self::features::FeaturesFfi;
pub use self::features::{FeatureRestriction, FeatureRestrictionError, Features};
pub use self::properties::Properties;
pub(crate) use self::properties::PropertiesFfi;
use crate::check_errors;
use crate::command_buffer::pool::StandardCommandPool;
use crate::descriptor_set::pool::StdDescriptorPool;
pub use crate::device::extensions::DeviceExtensions;
use crate::device::physical::PhysicalDevice;
use crate::device::physical::QueueFamily;
pub use crate::extensions::{
    ExtensionRestriction, ExtensionRestrictionError, SupportedExtensionsError,
};
use crate::fns::DeviceFunctions;
use crate::instance::Instance;
use crate::memory::pool::StdMemoryPool;
use crate::Error;
use crate::OomError;
use crate::SynchronizedVulkanObject;
use crate::Version;
use crate::VulkanObject;
use ash::vk::Handle;
use fnv::FnvHasher;
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::error;
use std::ffi::CStr;
use std::ffi::CString;
use std::fmt;
use std::hash::BuildHasherDefault;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::Weak;

pub(crate) mod extensions;
pub(crate) mod features;
pub mod physical;
pub(crate) mod properties;

/// Represents a Vulkan context.
#[derive(Debug)]
pub struct Device {
    handle: ash::vk::Device,
    instance: Arc<Instance>,
    physical_device: usize,

    // The highest version that is supported for this device.
    // This is the minimum of Instance::max_api_version and PhysicalDevice::api_version.
    api_version: Version,

    fns: DeviceFunctions,
    standard_pool: Mutex<Weak<StdMemoryPool>>,
    standard_descriptor_pool: Mutex<Weak<StdDescriptorPool>>,
    standard_command_pools:
        Mutex<HashMap<u32, Weak<StandardCommandPool>, BuildHasherDefault<FnvHasher>>>,
    enabled_extensions: DeviceExtensions,
    enabled_features: Features,
    active_queue_families: SmallVec<[u32; 2]>,
    allocation_count: Mutex<u32>,
    fence_pool: Mutex<Vec<ash::vk::Fence>>,
    semaphore_pool: Mutex<Vec<ash::vk::Semaphore>>,
    event_pool: Mutex<Vec<ash::vk::Event>>,
}

// The `StandardCommandPool` type doesn't implement Send/Sync, so we have to manually reimplement
// them for the device itself.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Creates a new `Device`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.queues` is empty.
    /// - Panics if one of the queue families in `create_info.queues` doesn't belong to the given
    ///   physical device.
    /// - Panics if `create_info.queues` contains multiple elements for the same queue family.
    /// - Panics if `create_info.queues` contains an element where `queues` is empty.
    /// - Panics if `create_info.queues` contains an element where `queues` contains a value that is
    ///   not between 0.0 and 1.0 inclusive.
    pub fn new(
        physical_device: PhysicalDevice,
        create_info: DeviceCreateInfo,
    ) -> Result<(Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>), DeviceCreationError> {
        let DeviceCreateInfo {
            enabled_extensions,
            mut enabled_features,
            queue_create_infos,
            _ne: _,
        } = create_info;

        let instance = physical_device.instance();
        let fns_i = instance.fns();
        let api_version = physical_device.api_version();

        /*
            Queues
        */

        struct QueueToGet {
            family: u32,
            id: u32,
        }

        // VUID-VkDeviceCreateInfo-queueCreateInfoCount-arraylength
        assert!(!queue_create_infos.is_empty());

        let mut queue_create_infos_vk: SmallVec<[_; 2]> =
            SmallVec::with_capacity(queue_create_infos.len());
        let mut active_queue_families: SmallVec<[_; 2]> =
            SmallVec::with_capacity(queue_create_infos.len());
        let mut queues_to_get: SmallVec<[_; 2]> = SmallVec::with_capacity(queue_create_infos.len());

        for QueueCreateInfo {
            family,
            queues,
            _ne: _,
        } in &queue_create_infos
        {
            assert_eq!(
                family.physical_device().internal_object(),
                physical_device.internal_object()
            );

            // VUID-VkDeviceCreateInfo-queueFamilyIndex-02802
            assert!(
                queue_create_infos
                    .iter()
                    .filter(|qc2| qc2.family == *family)
                    .count()
                    == 1
            );

            // VUID-VkDeviceQueueCreateInfo-queueCount-arraylength
            assert!(!queues.is_empty());

            // VUID-VkDeviceQueueCreateInfo-pQueuePriorities-00383
            assert!(queues
                .iter()
                .all(|&priority| priority >= 0.0 && priority <= 1.0));

            if queues.len() > family.queues_count() {
                return Err(DeviceCreationError::TooManyQueuesForFamily);
            }

            let family = family.id();
            queue_create_infos_vk.push(ash::vk::DeviceQueueCreateInfo {
                flags: ash::vk::DeviceQueueCreateFlags::empty(),
                queue_family_index: family,
                queue_count: queues.len() as u32,
                p_queue_priorities: queues.as_ptr(), // borrows from queue_create
                ..Default::default()
            });
            active_queue_families.push(family);
            queues_to_get.extend((0..queues.len() as u32).map(move |id| QueueToGet { family, id }));
        }

        active_queue_families.sort_unstable();
        active_queue_families.dedup();

        /*
            Extensions
        */

        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-01840
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-03328
        // VUID-VkDeviceCreateInfo-pProperties-04451
        enabled_extensions.check_requirements(
            physical_device.supported_extensions(),
            api_version,
            instance.enabled_extensions(),
        )?;

        let enabled_extensions_strings = Vec::<CString>::from(&enabled_extensions);
        let enabled_extensions_ptrs = enabled_extensions_strings
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<SmallVec<[_; 16]>>();

        /*
            Features
        */

        // TODO: The plan regarding `robust_buffer_access` is to check the shaders' code to see
        //       if they can possibly perform out-of-bounds reads and writes. If the user tries
        //       to use a shader that can perform out-of-bounds operations without having
        //       `robust_buffer_access` enabled, an error is returned.
        //
        //       However for the moment this verification isn't performed. In order to be safe,
        //       we always enable the `robust_buffer_access` feature as it is guaranteed to be
        //       supported everywhere.
        //
        //       The only alternative (while waiting for shaders introspection to work) is to
        //       make all shaders depend on `robust_buffer_access`. But since usually the
        //       majority of shaders don't need this feature, it would be very annoying to have
        //       to enable it manually when you don't need it.
        //
        //       Note that if we ever remove this, don't forget to adjust the change in
        //       `Device`'s construction below.
        enabled_features.robust_buffer_access = true;

        // VUID-VkDeviceCreateInfo-pNext-04748
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-04476
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02831
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02832
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02833
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02834
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02835
        // VUID-VkDeviceCreateInfo-shadingRateImage-04478
        // VUID-VkDeviceCreateInfo-shadingRateImage-04479
        // VUID-VkDeviceCreateInfo-shadingRateImage-04480
        // VUID-VkDeviceCreateInfo-fragmentDensityMap-04481
        // VUID-VkDeviceCreateInfo-fragmentDensityMap-04482
        // VUID-VkDeviceCreateInfo-fragmentDensityMap-04483
        // VUID-VkDeviceCreateInfo-None-04896
        // VUID-VkDeviceCreateInfo-None-04897
        // VUID-VkDeviceCreateInfo-None-04898
        // VUID-VkDeviceCreateInfo-sparseImageFloat32AtomicMinMax-04975
        enabled_features.check_requirements(
            physical_device.supported_features(),
            api_version,
            &enabled_extensions,
        )?;

        // VUID-VkDeviceCreateInfo-pNext-02829
        // VUID-VkDeviceCreateInfo-pNext-02830
        // VUID-VkDeviceCreateInfo-pNext-06532
        let mut features_ffi = FeaturesFfi::default();
        features_ffi.make_chain(
            api_version,
            &enabled_extensions,
            instance.enabled_extensions(),
        );
        features_ffi.write(&enabled_features);

        // Device layers were deprecated in Vulkan 1.0.13, and device layer requests should be
        // ignored by the driver. For backwards compatibility, the spec recommends passing the
        // exact instance layers to the device as well. There's no need to support separate
        // requests at device creation time for legacy drivers: the spec claims that "[at] the
        // time of deprecation there were no known device-only layers."
        //
        // Because there's no way to query the list of layers enabled for an instance, we need
        // to save it alongside the instance. (`vkEnumerateDeviceLayerProperties` should get
        // the right list post-1.0.13, but not pre-1.0.13, so we can't use it here.)
        let enabled_layers_cstr: Vec<CString> = instance
            .enabled_layers()
            .iter()
            .map(|name| CString::new(name.clone()).unwrap())
            .collect();
        let enabled_layers_ptrs = enabled_layers_cstr
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<SmallVec<[_; 2]>>();

        /*
            Create the device
        */

        let has_khr_get_physical_device_properties2 = instance
            .enabled_extensions()
            .khr_get_physical_device_properties2;

        let mut create_info = ash::vk::DeviceCreateInfo {
            flags: ash::vk::DeviceCreateFlags::empty(),
            queue_create_info_count: queue_create_infos_vk.len() as u32,
            p_queue_create_infos: queue_create_infos_vk.as_ptr(),
            enabled_layer_count: enabled_layers_ptrs.len() as u32,
            pp_enabled_layer_names: enabled_layers_ptrs.as_ptr(),
            enabled_extension_count: enabled_extensions_ptrs.len() as u32,
            pp_enabled_extension_names: enabled_extensions_ptrs.as_ptr(),
            p_enabled_features: ptr::null(),
            ..Default::default()
        };

        // VUID-VkDeviceCreateInfo-pNext-00373
        if has_khr_get_physical_device_properties2 {
            create_info.p_next = features_ffi.head_as_ref() as *const _ as _;
        } else {
            create_info.p_enabled_features = &features_ffi.head_as_ref().features;
        }

        let handle = unsafe {
            let mut output = MaybeUninit::uninit();
            check_errors(fns_i.v1_0.create_device(
                physical_device.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        // loading the function pointers of the newly-created device
        let fns = DeviceFunctions::load(|name| unsafe {
            mem::transmute(fns_i.v1_0.get_device_proc_addr(handle, name.as_ptr()))
        });

        let device = Arc::new(Device {
            handle,
            instance: physical_device.instance().clone(),
            physical_device: physical_device.index(),
            api_version,
            fns,
            standard_pool: Mutex::new(Weak::new()),
            standard_descriptor_pool: Mutex::new(Weak::new()),
            standard_command_pools: Mutex::new(Default::default()),
            enabled_extensions,
            enabled_features,
            active_queue_families,
            allocation_count: Mutex::new(0),
            fence_pool: Mutex::new(Vec::new()),
            semaphore_pool: Mutex::new(Vec::new()),
            event_pool: Mutex::new(Vec::new()),
        });

        // Iterator to return the queues
        let queues_iter = {
            let device = device.clone();
            queues_to_get
                .into_iter()
                .map(move |QueueToGet { family, id }| unsafe {
                    let mut output = MaybeUninit::uninit();
                    device
                        .fns()
                        .v1_0
                        .get_device_queue(handle, family, id, output.as_mut_ptr());

                    Arc::new(Queue {
                        handle: Mutex::new(output.assume_init()),
                        device: device.clone(),
                        family,
                        id,
                    })
                })
        };

        Ok((device, queues_iter))
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

    /// Grants access to the Vulkan functions of the device.
    #[inline]
    pub fn fns(&self) -> &DeviceFunctions {
        &self.fns
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
    ///
    pub unsafe fn wait(&self) -> Result<(), OomError> {
        check_errors(self.fns.v1_0.device_wait_idle(self.handle))?;
        Ok(())
    }

    /// Returns the instance used to create this device.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Returns the physical device that was used to create this device.
    #[inline]
    pub fn physical_device(&self) -> PhysicalDevice {
        PhysicalDevice::from_index(&self.instance, self.physical_device).unwrap()
    }

    /// Returns an iterator to the list of queues families that this device uses.
    ///
    /// > **Note**: Will return `-> impl ExactSizeIterator<Item = QueueFamily>` in the future.
    // TODO: ^
    #[inline]
    pub fn active_queue_families<'a>(&'a self) -> impl ExactSizeIterator<Item = QueueFamily<'a>> {
        let physical_device = self.physical_device();
        self.active_queue_families
            .iter()
            .map(move |&id| physical_device.queue_family_by_id(id).unwrap())
    }

    /// Returns the extensions that have been enabled on the device.
    #[inline]
    pub fn enabled_extensions(&self) -> &DeviceExtensions {
        &self.enabled_extensions
    }

    /// Returns the features that have been enabled on the device.
    #[inline]
    pub fn enabled_features(&self) -> &Features {
        &self.enabled_features
    }

    /// Returns the standard memory pool used by default if you don't provide any other pool.
    pub fn standard_pool(me: &Arc<Self>) -> Arc<StdMemoryPool> {
        let mut pool = me.standard_pool.lock().unwrap();

        if let Some(p) = pool.upgrade() {
            return p;
        }

        // The weak pointer is empty, so we create the pool.
        let new_pool = StdMemoryPool::new(me.clone());
        *pool = Arc::downgrade(&new_pool);
        new_pool
    }

    /// Returns the standard descriptor pool used by default if you don't provide any other pool.
    pub fn standard_descriptor_pool(me: &Arc<Self>) -> Arc<StdDescriptorPool> {
        let mut pool = me.standard_descriptor_pool.lock().unwrap();

        if let Some(p) = pool.upgrade() {
            return p;
        }

        // The weak pointer is empty, so we create the pool.
        let new_pool = Arc::new(StdDescriptorPool::new(me.clone()));
        *pool = Arc::downgrade(&new_pool);
        new_pool
    }

    /// Returns the standard command buffer pool used by default if you don't provide any other
    /// pool.
    ///
    /// # Panic
    ///
    /// - Panics if the device and the queue family don't belong to the same physical device.
    ///
    pub fn standard_command_pool(me: &Arc<Self>, queue: QueueFamily) -> Arc<StandardCommandPool> {
        let mut standard_command_pools = me.standard_command_pools.lock().unwrap();

        match standard_command_pools.entry(queue.id()) {
            Entry::Occupied(mut entry) => {
                if let Some(pool) = entry.get().upgrade() {
                    return pool;
                }

                let new_pool = Arc::new(StandardCommandPool::new(me.clone(), queue));
                *entry.get_mut() = Arc::downgrade(&new_pool);
                new_pool
            }
            Entry::Vacant(entry) => {
                let new_pool = Arc::new(StandardCommandPool::new(me.clone(), queue));
                entry.insert(Arc::downgrade(&new_pool));
                new_pool
            }
        }
    }

    /// Used to track the number of allocations on this device.
    ///
    /// To ensure valid usage of the Vulkan API, we cannot call `vkAllocateMemory` when
    /// `maxMemoryAllocationCount` has been exceeded. See the Vulkan specs:
    /// https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#vkAllocateMemory
    ///
    /// Warning: You should never modify this value, except in `device_memory` module
    pub(crate) fn allocation_count(&self) -> &Mutex<u32> {
        &self.allocation_count
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

    /// Assigns a human-readable name to `object` for debugging purposes.
    ///
    /// # Panics
    /// * If `object` is not owned by this device.
    pub fn set_object_name<T: VulkanObject + DeviceOwned>(
        &self,
        object: &T,
        name: &CStr,
    ) -> Result<(), OomError> {
        assert!(object.device().internal_object() == self.internal_object());
        unsafe {
            self.set_object_name_raw(T::Object::TYPE, object.internal_object().as_raw(), name)
        }
    }

    /// Assigns a human-readable name to `object` for debugging purposes.
    ///
    /// # Safety
    /// `object` must be a Vulkan handle owned by this device, and its type must be accurately described by `ty`.
    pub unsafe fn set_object_name_raw(
        &self,
        ty: ash::vk::ObjectType,
        object: u64,
        name: &CStr,
    ) -> Result<(), OomError> {
        let info = ash::vk::DebugUtilsObjectNameInfoEXT {
            object_type: ty,
            object_handle: object,
            p_object_name: name.as_ptr(),
            ..Default::default()
        };
        check_errors(
            self.instance
                .fns()
                .ext_debug_utils
                .set_debug_utils_object_name_ext(self.handle, &info),
        )?;
        Ok(())
    }
}

impl Drop for Device {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            for &raw_fence in self.fence_pool.lock().unwrap().iter() {
                self.fns
                    .v1_0
                    .destroy_fence(self.handle, raw_fence, ptr::null());
            }
            for &raw_sem in self.semaphore_pool.lock().unwrap().iter() {
                self.fns
                    .v1_0
                    .destroy_semaphore(self.handle, raw_sem, ptr::null());
            }
            for &raw_event in self.event_pool.lock().unwrap().iter() {
                self.fns
                    .v1_0
                    .destroy_event(self.handle, raw_event, ptr::null());
            }
            self.fns.v1_0.destroy_device(self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for Device {
    type Object = ash::vk::Device;

    #[inline]
    fn internal_object(&self) -> ash::vk::Device {
        self.handle
    }
}

impl PartialEq for Device {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.instance == other.instance
    }
}

impl Eq for Device {}

impl Hash for Device {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.instance.hash(state);
    }
}

/// Parameters to create a new `Device`.
#[derive(Clone, Debug)]
pub struct DeviceCreateInfo<'qf> {
    /// The extensions to enable on the device.
    ///
    /// The default value is [`DeviceExtensions::none()`].
    pub enabled_extensions: DeviceExtensions,

    /// The features to enable on the device.
    ///
    /// The default value is [`Features::none()`].
    pub enabled_features: Features,

    /// The queues to create for the device.
    pub queue_create_infos: Vec<QueueCreateInfo<'qf>>,

    pub _ne: crate::NonExhaustive,
}

impl Default for DeviceCreateInfo<'static> {
    #[inline]
    fn default() -> Self {
        Self {
            enabled_extensions: DeviceExtensions::none(),
            enabled_features: Features::none(),
            queue_create_infos: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to create queues in a new `Device`.
#[derive(Clone, Debug)]
pub struct QueueCreateInfo<'qf> {
    /// The queue family to create queues for.
    pub family: QueueFamily<'qf>,

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

impl<'qf> QueueCreateInfo<'qf> {
    /// Returns a `QueueCreateInfo` with the given queue family.
    #[inline]
    pub fn family<'qf2>(family: QueueFamily<'qf2>) -> QueueCreateInfo<'qf2> {
        QueueCreateInfo {
            family,
            queues: vec![0.5],
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Implemented on objects that belong to a Vulkan device.
///
/// # Safety
///
/// - `device()` must return the correct device.
///
pub unsafe trait DeviceOwned {
    /// Returns the device that owns `Self`.
    fn device(&self) -> &Arc<Device>;
}

unsafe impl<T> DeviceOwned for T
where
    T: Deref,
    T::Target: DeviceOwned,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        (**self).device()
    }
}

/// Error that can be returned when creating a device.
#[derive(Copy, Clone, Debug)]
pub enum DeviceCreationError {
    /// Failed to create the device for an implementation-specific reason.
    InitializationFailed,
    /// You have reached the limit to the number of devices that can be created from the same
    /// physical device.
    TooManyObjects,
    /// Failed to connect to the device.
    DeviceLost,
    /// Some of the requested features are unsupported by the physical device.
    FeatureNotPresent,
    /// Some of the requested device extensions are not supported by the physical device.
    ExtensionNotPresent,
    /// Tried to create too many queues for a given family.
    TooManyQueuesForFamily,
    /// The priority of one of the queues is out of the [0.0; 1.0] range.
    PriorityOutOfRange,
    /// There is no memory available on the host (ie. the CPU, RAM, etc.).
    OutOfHostMemory,
    /// There is no memory available on the device (ie. video memory).
    OutOfDeviceMemory,
    /// A restriction for an extension was not met.
    ExtensionRestrictionNotMet(ExtensionRestrictionError),
    /// A restriction for a feature was not met.
    FeatureRestrictionNotMet(FeatureRestrictionError),
}

impl error::Error for DeviceCreationError {}

impl fmt::Display for DeviceCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            DeviceCreationError::InitializationFailed => {
                write!(
                    fmt,
                    "failed to create the device for an implementation-specific reason"
                )
            }
            DeviceCreationError::OutOfHostMemory => write!(fmt, "no memory available on the host"),
            DeviceCreationError::OutOfDeviceMemory => {
                write!(fmt, "no memory available on the graphical device")
            }
            DeviceCreationError::DeviceLost => write!(fmt, "failed to connect to the device"),
            DeviceCreationError::TooManyQueuesForFamily => {
                write!(fmt, "tried to create too many queues for a given family")
            }
            DeviceCreationError::FeatureNotPresent => {
                write!(
                    fmt,
                    "some of the requested features are unsupported by the physical device"
                )
            }
            DeviceCreationError::PriorityOutOfRange => {
                write!(
                    fmt,
                    "the priority of one of the queues is out of the [0.0; 1.0] range"
                )
            }
            DeviceCreationError::ExtensionNotPresent => {
                write!(fmt,"some of the requested device extensions are not supported by the physical device")
            }
            DeviceCreationError::TooManyObjects => {
                write!(fmt,"you have reached the limit to the number of devices that can be created from the same physical device")
            }
            DeviceCreationError::ExtensionRestrictionNotMet(err) => err.fmt(fmt),
            DeviceCreationError::FeatureRestrictionNotMet(err) => err.fmt(fmt),
        }
    }
}

impl From<Error> for DeviceCreationError {
    #[inline]
    fn from(err: Error) -> DeviceCreationError {
        match err {
            Error::InitializationFailed => DeviceCreationError::InitializationFailed,
            Error::OutOfHostMemory => DeviceCreationError::OutOfHostMemory,
            Error::OutOfDeviceMemory => DeviceCreationError::OutOfDeviceMemory,
            Error::DeviceLost => DeviceCreationError::DeviceLost,
            Error::ExtensionNotPresent => DeviceCreationError::ExtensionNotPresent,
            Error::FeatureNotPresent => DeviceCreationError::FeatureNotPresent,
            Error::TooManyObjects => DeviceCreationError::TooManyObjects,
            _ => panic!("Unexpected error value: {}", err as i32),
        }
    }
}

impl From<ExtensionRestrictionError> for DeviceCreationError {
    #[inline]
    fn from(err: ExtensionRestrictionError) -> Self {
        Self::ExtensionRestrictionNotMet(err)
    }
}

impl From<FeatureRestrictionError> for DeviceCreationError {
    #[inline]
    fn from(err: FeatureRestrictionError) -> Self {
        Self::FeatureRestrictionNotMet(err)
    }
}

/// Represents a queue where commands can be submitted.
// TODO: should use internal synchronization?
#[derive(Debug)]
pub struct Queue {
    handle: Mutex<ash::vk::Queue>,
    device: Arc<Device>,
    family: u32,
    id: u32, // id within family
}

impl Queue {
    /// Returns the device this queue belongs to.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns true if this is the same queue as another one.
    #[inline]
    pub fn is_same(&self, other: &Queue) -> bool {
        self.id == other.id
            && self.family == other.family
            && self.device.internal_object() == other.device.internal_object()
    }

    /// Returns the family this queue belongs to.
    #[inline]
    pub fn family(&self) -> QueueFamily {
        self.device
            .physical_device()
            .queue_family_by_id(self.family)
            .unwrap()
    }

    /// Returns the index of this queue within its family.
    #[inline]
    pub fn id_within_family(&self) -> u32 {
        self.id
    }

    /// Waits until all work on this queue has finished.
    ///
    /// Just like `Device::wait()`, you shouldn't have to call this function in a typical program.
    #[inline]
    pub fn wait(&self) -> Result<(), OomError> {
        unsafe {
            let fns = self.device.fns();
            let handle = self.handle.lock().unwrap();
            check_errors(fns.v1_0.queue_wait_idle(*handle))?;
            Ok(())
        }
    }
}

impl PartialEq for Queue {
    fn eq(&self, other: &Self) -> bool {
        self.is_same(other)
    }
}

impl Eq for Queue {}

unsafe impl DeviceOwned for Queue {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl SynchronizedVulkanObject for Queue {
    type Object = ash::vk::Queue;

    #[inline]
    fn internal_object_guard(&self) -> MutexGuard<Self::Object> {
        self.handle.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::device::physical::PhysicalDevice;
    use crate::device::{Device, DeviceCreateInfo, DeviceCreationError, QueueCreateInfo};
    use crate::device::{FeatureRestriction, FeatureRestrictionError, Features};
    use std::sync::Arc;

    #[test]
    fn one_ref() {
        let (mut device, _) = gfx_dev_and_queue!();
        assert!(Arc::get_mut(&mut device).is_some());
    }

    #[test]
    fn too_many_queues() {
        let instance = instance!();
        let physical = match PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return,
        };

        let family = physical.queue_families().next().unwrap();
        let queues = (0..family.queues_count() + 1).map(|_| (family, 1.0));

        match Device::new(
            physical,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queues: (0..family.queues_count() + 1).map(|_| (0.5)).collect(),
                    ..QueueCreateInfo::family(family)
                }],
                ..Default::default()
            },
        ) {
            Err(DeviceCreationError::TooManyQueuesForFamily) => return, // Success
            _ => panic!(),
        };
    }

    #[test]
    fn unsupposed_features() {
        let instance = instance!();
        let physical = match PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return,
        };

        let family = physical.queue_families().next().unwrap();

        let features = Features::all();
        // In the unlikely situation where the device supports everything, we ignore the test.
        if physical.supported_features().is_superset_of(&features) {
            return;
        }

        match Device::new(
            physical,
            DeviceCreateInfo {
                enabled_features: features,
                queue_create_infos: vec![QueueCreateInfo::family(family)],
                ..Default::default()
            },
        ) {
            Err(DeviceCreationError::FeatureRestrictionNotMet(FeatureRestrictionError {
                restriction: FeatureRestriction::NotSupported,
                ..
            })) => return, // Success
            _ => panic!(),
        };
    }

    #[test]
    fn priority_out_of_range() {
        let instance = instance!();
        let physical = match PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return,
        };

        let family = physical.queue_families().next().unwrap();

        assert_should_panic!({
            Device::new(
                physical,
                DeviceCreateInfo {
                    queue_create_infos: vec![QueueCreateInfo {
                        queues: vec![1.4],
                        ..QueueCreateInfo::family(family)
                    }],
                    ..Default::default()
                },
            )
        });

        assert_should_panic!({
            Device::new(
                physical,
                DeviceCreateInfo {
                    queue_create_infos: vec![QueueCreateInfo {
                        queues: vec![-0.2],
                        ..QueueCreateInfo::family(family)
                    }],
                    ..Default::default()
                },
            )
        });
    }
}
