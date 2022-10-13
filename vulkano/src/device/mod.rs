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
//!             enabled_extensions: extensions,
//!             enabled_features: features,
//!             queue_create_infos: vec![QueueCreateInfo {
//!                 queue_family_index: 0,
//!                 ..Default::default()
//!             }],
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
pub(crate) use self::{features::FeaturesFfi, properties::PropertiesFfi};
pub use self::{
    features::{FeatureRestriction, FeatureRestrictionError, Features},
    properties::Properties,
    queue::{Queue, QueueError, QueueFamilyProperties, QueueFlags, QueueGuard},
};
pub use crate::{
    device::extensions::DeviceExtensions,
    extensions::{ExtensionRestriction, ExtensionRestrictionError},
    fns::DeviceFunctions,
};
use crate::{
    instance::Instance,
    memory::{pool::StandardMemoryPool, ExternalMemoryHandleType},
    OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use ash::vk::Handle;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    error::Error,
    ffi::CString,
    fmt::{Display, Error as FmtError, Formatter},
    fs::File,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops::Deref,
    ptr,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Weak,
    },
};

pub(crate) mod extensions;
pub(crate) mod features;
pub mod physical;
pub(crate) mod properties;
mod queue;

/// Represents a Vulkan context.
#[derive(Debug)]
pub struct Device {
    handle: ash::vk::Device,
    physical_device: Arc<PhysicalDevice>,

    // The highest version that is supported for this device.
    // This is the minimum of Instance::max_api_version and PhysicalDevice::api_version.
    api_version: Version,

    fns: DeviceFunctions,
    standard_memory_pool: Mutex<Weak<StandardMemoryPool>>,
    enabled_extensions: DeviceExtensions,
    enabled_features: Features,
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
        physical_device: Arc<PhysicalDevice>,
        create_info: DeviceCreateInfo,
    ) -> Result<(Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>), DeviceCreationError> {
        let DeviceCreateInfo {
            mut enabled_extensions,
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
            queue_family_index: u32,
            id: u32,
        }

        // VUID-VkDeviceCreateInfo-queueCreateInfoCount-arraylength
        assert!(!queue_create_infos.is_empty());

        let mut queue_create_infos_vk: SmallVec<[_; 2]> =
            SmallVec::with_capacity(queue_create_infos.len());
        let mut active_queue_family_indices: SmallVec<[_; 2]> =
            SmallVec::with_capacity(queue_create_infos.len());
        let mut queues_to_get: SmallVec<[_; 2]> = SmallVec::with_capacity(queue_create_infos.len());

        for queue_create_info in &queue_create_infos {
            let &QueueCreateInfo {
                queue_family_index,
                ref queues,
                _ne: _,
            } = queue_create_info;

            // VUID-VkDeviceQueueCreateInfo-queueFamilyIndex-00381
            // TODO: return error instead of panicking?
            let queue_family_properties =
                &physical_device.queue_family_properties()[queue_family_index as usize];

            // VUID-VkDeviceCreateInfo-queueFamilyIndex-02802
            assert!(
                queue_create_infos
                    .iter()
                    .filter(|qc2| qc2.queue_family_index == queue_family_index)
                    .count()
                    == 1
            );

            // VUID-VkDeviceQueueCreateInfo-queueCount-arraylength
            assert!(!queues.is_empty());

            // VUID-VkDeviceQueueCreateInfo-pQueuePriorities-00383
            assert!(queues
                .iter()
                .all(|&priority| (0.0..=1.0).contains(&priority)));

            if queues.len() > queue_family_properties.queue_count as usize {
                return Err(DeviceCreationError::TooManyQueuesForFamily);
            }

            queue_create_infos_vk.push(ash::vk::DeviceQueueCreateInfo {
                flags: ash::vk::DeviceQueueCreateFlags::empty(),
                queue_family_index,
                queue_count: queues.len() as u32,
                p_queue_priorities: queues.as_ptr(), // borrows from queue_create
                ..Default::default()
            });
            active_queue_family_indices.push(queue_family_index);
            queues_to_get.extend((0..queues.len() as u32).map(move |id| QueueToGet {
                queue_family_index,
                id,
            }));
        }

        active_queue_family_indices.sort_unstable();
        active_queue_family_indices.dedup();
        let supported_extensions = physical_device.supported_extensions();

        if supported_extensions.khr_portability_subset {
            enabled_extensions.khr_portability_subset = true;
        }

        /*
            Extensions
        */

        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-01840
        // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-03328
        // VUID-VkDeviceCreateInfo-pProperties-04451
        enabled_extensions.check_requirements(
            supported_extensions,
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
            (fns_i.v1_0.create_device)(
                physical_device.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        // loading the function pointers of the newly-created device
        let fns = DeviceFunctions::load(|name| unsafe {
            (fns_i.v1_0.get_device_proc_addr)(handle, name.as_ptr())
                .map_or(ptr::null(), |func| func as _)
        });

        let device = Arc::new(Device {
            handle,
            physical_device,
            api_version,
            fns,
            standard_memory_pool: Mutex::new(Weak::new()),
            enabled_extensions,
            enabled_features,
            active_queue_family_indices,
            allocation_count: AtomicU32::new(0),
            fence_pool: Mutex::new(Vec::new()),
            semaphore_pool: Mutex::new(Vec::new()),
            event_pool: Mutex::new(Vec::new()),
        });

        // Iterator to return the queues
        let queues_iter = {
            let device = device.clone();
            queues_to_get.into_iter().map(
                move |QueueToGet {
                          queue_family_index,
                          id,
                      }| unsafe {
                    let fns = device.fns();
                    let mut output = MaybeUninit::uninit();
                    (fns.v1_0.get_device_queue)(
                        handle,
                        queue_family_index,
                        id,
                        output.as_mut_ptr(),
                    );

                    Queue::from_handle(device.clone(), output.assume_init(), queue_family_index, id)
                },
            )
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
    pub fn standard_memory_pool(self: &Arc<Self>) -> Arc<StandardMemoryPool> {
        let mut pool = self.standard_memory_pool.lock();

        if let Some(p) = pool.upgrade() {
            return p;
        }

        // The weak pointer is empty, so we create the pool.
        let new_pool = StandardMemoryPool::new(self.clone());
        *pool = Arc::downgrade(&new_pool);

        new_pool
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
    ) -> Result<MemoryFdProperties, MemoryFdPropertiesError> {
        if !self.enabled_extensions().khr_external_memory_fd {
            return Err(MemoryFdPropertiesError::NotSupported);
        }

        #[cfg(not(unix))]
        unreachable!("`khr_external_memory_fd` was somehow enabled on a non-Unix system");

        #[cfg(unix)]
        {
            use std::os::unix::io::IntoRawFd;

            // VUID-vkGetMemoryFdPropertiesKHR-handleType-parameter
            handle_type.validate_device(self)?;

            // VUID-vkGetMemoryFdPropertiesKHR-handleType-00674
            if handle_type == ExternalMemoryHandleType::OpaqueFd {
                return Err(MemoryFdPropertiesError::InvalidExternalHandleType);
            }

            let mut memory_fd_properties = ash::vk::MemoryFdPropertiesKHR::default();

            let fns = self.fns();
            (fns.khr_external_memory_fd.get_memory_fd_properties_khr)(
                self.handle,
                handle_type.into(),
                file.into_raw_fd(),
                &mut memory_fd_properties,
            )
            .result()
            .map_err(VulkanError::from)?;

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
                .map_err(VulkanError::from)?;
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
            .map_err(VulkanError::from)?;

        Ok(())
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
    fn handle(&self) -> ash::vk::Device {
        self.handle
    }
}

impl PartialEq for Device {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.physical_device == other.physical_device
    }
}

impl Eq for Device {}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.physical_device.hash(state);
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

impl Error for DeviceCreationError {}

impl Display for DeviceCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::InitializationFailed => write!(
                f,
                "failed to create the device for an implementation-specific reason",
            ),
            Self::OutOfHostMemory => write!(f, "no memory available on the host"),
            Self::OutOfDeviceMemory => {
                write!(f, "no memory available on the graphical device")
            }
            Self::DeviceLost => write!(f, "failed to connect to the device"),
            Self::TooManyQueuesForFamily => {
                write!(f, "tried to create too many queues for a given family")
            }
            Self::FeatureNotPresent => write!(
                f,
                "some of the requested features are unsupported by the physical device",
            ),
            Self::PriorityOutOfRange => write!(
                f,
                "the priority of one of the queues is out of the [0.0; 1.0] range",
            ),
            Self::ExtensionNotPresent => write!(
                f,
                "some of the requested device extensions are not supported by the physical device",
            ),
            Self::TooManyObjects => write!(
                f,
                "you have reached the limit to the number of devices that can be created from the \
                same physical device",
            ),
            Self::ExtensionRestrictionNotMet(err) => err.fmt(f),
            Self::FeatureRestrictionNotMet(err) => err.fmt(f),
        }
    }
}

impl From<VulkanError> for DeviceCreationError {
    fn from(err: VulkanError) -> Self {
        match err {
            VulkanError::InitializationFailed => Self::InitializationFailed,
            VulkanError::OutOfHostMemory => Self::OutOfHostMemory,
            VulkanError::OutOfDeviceMemory => Self::OutOfDeviceMemory,
            VulkanError::DeviceLost => Self::DeviceLost,
            VulkanError::ExtensionNotPresent => Self::ExtensionNotPresent,
            VulkanError::FeatureNotPresent => Self::FeatureNotPresent,
            VulkanError::TooManyObjects => Self::TooManyObjects,
            _ => panic!("Unexpected error value"),
        }
    }
}

impl From<ExtensionRestrictionError> for DeviceCreationError {
    fn from(err: ExtensionRestrictionError) -> Self {
        Self::ExtensionRestrictionNotMet(err)
    }
}

impl From<FeatureRestrictionError> for DeviceCreationError {
    fn from(err: FeatureRestrictionError) -> Self {
        Self::FeatureRestrictionNotMet(err)
    }
}

/// Parameters to create a new `Device`.
#[derive(Clone, Debug)]
pub struct DeviceCreateInfo {
    /// The extensions to enable on the device.
    ///
    /// The default value is [`DeviceExtensions::empty()`].
    pub enabled_extensions: DeviceExtensions,

    /// The features to enable on the device.
    ///
    /// The default value is [`Features::empty()`].
    pub enabled_features: Features,

    /// The queues to create for the device.
    ///
    /// The default value is empty, which must be overridden.
    pub queue_create_infos: Vec<QueueCreateInfo>,

    pub _ne: crate::NonExhaustive,
}

impl Default for DeviceCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            enabled_extensions: DeviceExtensions::empty(),
            enabled_features: Features::empty(),
            queue_create_infos: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to create queues in a new `Device`.
#[derive(Clone, Debug)]
pub struct QueueCreateInfo {
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
            queue_family_index: 0,
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

/// Error that can happen when calling `memory_fd_properties`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFdPropertiesError {
    /// No memory available on the host.
    OutOfHostMemory,

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The provided external handle was not valid.
    InvalidExternalHandle,

    /// The provided external handle type was not valid.
    InvalidExternalHandleType,

    /// The `khr_external_memory_fd` extension was not enabled on the device.
    NotSupported,
}

impl Error for MemoryFdPropertiesError {}

impl Display for MemoryFdPropertiesError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OutOfHostMemory => write!(f, "no memory available on the host"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::InvalidExternalHandle => {
                write!(f, "the provided external handle was not valid")
            }
            Self::InvalidExternalHandleType => {
                write!(f, "the provided external handle type was not valid")
            }
            Self::NotSupported => write!(
                f,
                "the `khr_external_memory_fd` extension was not enabled on the device",
            ),
        }
    }
}

impl From<VulkanError> for MemoryFdPropertiesError {
    fn from(err: VulkanError) -> Self {
        match err {
            VulkanError::OutOfHostMemory => Self::OutOfHostMemory,
            VulkanError::InvalidExternalHandle => Self::InvalidExternalHandle,
            _ => panic!("Unexpected error value"),
        }
    }
}

impl From<RequirementNotMet> for MemoryFdPropertiesError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::device::{
        Device, DeviceCreateInfo, DeviceCreationError, FeatureRestriction, FeatureRestrictionError,
        Features, QueueCreateInfo,
    };
    use std::sync::Arc;

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

        match Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    queues,
                    ..Default::default()
                }],
                ..Default::default()
            },
        ) {
            Err(DeviceCreationError::TooManyQueuesForFamily) => (), // Success
            _ => panic!(),
        };
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

        match Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_features: features,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                ..Default::default()
            },
        ) {
            Err(DeviceCreationError::FeatureRestrictionNotMet(FeatureRestrictionError {
                restriction: FeatureRestriction::NotSupported,
                ..
            })) => (), // Success
            _ => panic!(),
        };
    }

    #[test]
    fn priority_out_of_range() {
        let instance = instance!();
        let physical_device = match instance.enumerate_physical_devices().unwrap().next() {
            Some(p) => p,
            None => return,
        };

        assert_should_panic!({
            Device::new(
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
        });

        assert_should_panic!({
            Device::new(
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
        });
    }
}
