// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
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
//! use vulkano::device::Device;
//! use vulkano::instance::DeviceExtensions;
//! use vulkano::instance::Features;
//! use vulkano::instance::Instance;
//! use vulkano::instance::InstanceExtensions;
//! use vulkano::instance::PhysicalDevice;
//!
//! // Creating the instance. See the documentation of the `instance` module.
//! let instance = match Instance::new(None, &InstanceExtensions::none(), None) {
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
//!     let ext = DeviceExtensions::none();
//!
//!     match Device::new(physical_device, &features, &ext, Some((queue_family, 1.0))) {
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
//! TODO: oops, there's no method for querying supported extensions in vulkan yet.
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

use fnv::FnvHasher;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::error;
use std::fmt;
use std::hash::BuildHasherDefault;
use std::mem;
use std::ops::Deref;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::Weak;

use command_buffer::pool::StandardCommandPool;
use descriptor::descriptor_set::StdDescriptorPool;
use instance::Features;
use instance::Instance;
use instance::PhysicalDevice;
use instance::QueueFamily;
use memory::pool::StdMemoryPool;

use Error;
use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use check_errors;
use vk;

pub use instance::{DeviceExtensions, RawDeviceExtensions};

/// Represents a Vulkan context.
pub struct Device {
    instance: Arc<Instance>,
    physical_device: usize,
    device: vk::Device,
    vk: vk::DevicePointers,
    standard_pool: Mutex<Weak<StdMemoryPool>>,
    standard_descriptor_pool: Mutex<Weak<StdDescriptorPool>>,
    standard_command_pools:
        Mutex<HashMap<u32, Weak<StandardCommandPool>, BuildHasherDefault<FnvHasher>>>,
    features: Features,
    extensions: DeviceExtensions,
    active_queue_families: SmallVec<[u32; 8]>,
    allocation_count: Mutex<u32>,
    fence_pool: Mutex<Vec<vk::Fence>>,
    semaphore_pool: Mutex<Vec<vk::Semaphore>>,
    event_pool: Mutex<Vec<vk::Event>>,
}

// The `StandardCommandPool` type doesn't implement Send/Sync, so we have to manually reimplement
// them for the device itself.
unsafe impl Send for Device {
}
unsafe impl Sync for Device {
}

impl Device {
    /// Builds a new Vulkan device for the given physical device.
    ///
    /// You must pass two things when creating a logical device:
    ///
    /// - A list of optional Vulkan features that must be enabled on the device. Note that if a
    ///   feature is not enabled at device creation, you can't use it later even it it's supported
    ///   by the physical device.
    ///
    /// - An iterator to a list of queues to create. Each element of the iterator must indicate
    ///   the family whose queue belongs to and a priority between 0.0 and 1.0 to assign to it.
    ///   A queue with a higher value indicates that the commands will execute faster than on a
    ///   queue with a lower value. Note however that no guarantee can be made on the way the
    ///   priority value is handled by the implementation.
    ///
    /// # Panic
    ///
    /// - Panics if one of the queue families doesn't belong to the given device.
    ///
    // TODO: return Arc<Queue> and handle synchronization in the Queue
    // TODO: should take the PhysicalDevice by value
    pub fn new<'a, I, Ext>(phys: PhysicalDevice, requested_features: &Features, extensions: Ext,
                           queue_families: I)
                           -> Result<(Arc<Device>, QueuesIter), DeviceCreationError>
        where I: IntoIterator<Item = (QueueFamily<'a>, f32)>,
              Ext: Into<RawDeviceExtensions>
    {
        let queue_families = queue_families.into_iter();

        if !phys.supported_features().superset_of(&requested_features) {
            return Err(DeviceCreationError::FeatureNotPresent);
        }

        let vk_i = phys.instance().pointers();

        // this variable will contain the queue family ID and queue ID of each requested queue
        let mut output_queues: SmallVec<[(u32, u32); 8]> = SmallVec::new();

        // Device layers were deprecated in Vulkan 1.0.13, and device layer requests should be
        // ignored by the driver. For backwards compatibility, the spec recommends passing the
        // exact instance layers to the device as well. There's no need to support separate
        // requests at device creation time for legacy drivers: the spec claims that "[at] the
        // time of deprecation there were no known device-only layers."
        //
        // Because there's no way to query the list of layers enabled for an instance, we need
        // to save it alongside the instance. (`vkEnumerateDeviceLayerProperties` should get
        // the right list post-1.0.13, but not pre-1.0.13, so we can't use it here.)
        let layers_ptr = phys.instance()
            .loaded_layers()
            .map(|layer| layer.as_ptr())
            .collect::<SmallVec<[_; 16]>>();

        let extensions = extensions.into();
        let extensions_list = extensions
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<SmallVec<[_; 16]>>();

        // device creation
        let device = unsafe {
            // each element of `queues` is a `(queue_family, priorities)`
            // each queue family must only have one entry in `queues`
            let mut queues: Vec<(u32, Vec<f32>)> = Vec::with_capacity(phys.queue_families().len());

            for (queue_family, priority) in queue_families {
                // checking the parameters
                assert_eq!(queue_family.physical_device().internal_object(),
                           phys.internal_object());
                if priority < 0.0 || priority > 1.0 {
                    return Err(DeviceCreationError::PriorityOutOfRange);
                }

                // adding to `queues` and `output_queues`
                if let Some(q) = queues.iter_mut().find(|q| q.0 == queue_family.id()) {
                    output_queues.push((queue_family.id(), q.1.len() as u32));
                    q.1.push(priority);
                    if q.1.len() > queue_family.queues_count() {
                        return Err(DeviceCreationError::TooManyQueuesForFamily);
                    }
                    continue;
                }
                queues.push((queue_family.id(), vec![priority]));
                output_queues.push((queue_family.id(), 0));
            }

            // turning `queues` into an array of `vkDeviceQueueCreateInfo` suitable for Vulkan
            let queues = queues
                .iter()
                .map(|&(queue_id, ref priorities)| {
                    vk::DeviceQueueCreateInfo {
                        sType: vk::STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                        pNext: ptr::null(),
                        flags: 0, // reserved
                        queueFamilyIndex: queue_id,
                        queueCount: priorities.len() as u32,
                        pQueuePriorities: priorities.as_ptr(),
                    }
                })
                .collect::<SmallVec<[_; 16]>>();

            // TODO: The plan regarding `robustBufferAccess` is to check the shaders' code to see
            //       if they can possibly perform out-of-bounds reads and writes. If the user tries
            //       to use a shader that can perform out-of-bounds operations without having
            //       `robustBufferAccess` enabled, an error is returned.
            //
            //       However for the moment this verification isn't performed. In order to be safe,
            //       we always enable the `robustBufferAccess` feature as it is guaranteed to be
            //       supported everywhere.
            //
            //       The only alternative (while waiting for shaders introspection to work) is to
            //       make all shaders depend on `robustBufferAccess`. But since usually the
            //       majority of shaders don't need this feature, it would be very annoying to have
            //       to enable it manually when you don't need it.
            //
            //       Note that if we ever remove this, don't forget to adjust the change in
            //       `Device`'s construction below.
            let features = {
                let mut features = requested_features.clone().into_vulkan_features();
                features.robustBufferAccess = vk::TRUE;
                features
            };

            let infos = vk::DeviceCreateInfo {
                sType: vk::STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                queueCreateInfoCount: queues.len() as u32,
                pQueueCreateInfos: queues.as_ptr(),
                enabledLayerCount: layers_ptr.len() as u32,
                ppEnabledLayerNames: layers_ptr.as_ptr(),
                enabledExtensionCount: extensions_list.len() as u32,
                ppEnabledExtensionNames: extensions_list.as_ptr(),
                pEnabledFeatures: &features,
            };

            let mut output = mem::uninitialized();
            check_errors(vk_i.CreateDevice(phys.internal_object(),
                                           &infos,
                                           ptr::null(),
                                           &mut output))?;
            output
        };

        // loading the function pointers of the newly-created device
        let vk = vk::DevicePointers::load(|name| unsafe {
                                              vk_i.GetDeviceProcAddr(device, name.as_ptr()) as
                                                  *const _
                                          });

        let device =
            Arc::new(Device {
                         instance: phys.instance().clone(),
                         physical_device: phys.index(),
                         device: device,
                         vk: vk,
                         standard_pool: Mutex::new(Weak::new()),
                         standard_descriptor_pool: Mutex::new(Weak::new()),
                         standard_command_pools: Mutex::new(Default::default()),
                         features: Features {
                             // Always enabled ; see above
                             robust_buffer_access: true,
                             ..requested_features.clone()
                         },
                         extensions: (&extensions).into(),
                         active_queue_families: output_queues.iter().map(|&(q, _)| q).collect(),
                         allocation_count: Mutex::new(0),
                         fence_pool: Mutex::new(Vec::new()),
                         semaphore_pool: Mutex::new(Vec::new()),
                         event_pool: Mutex::new(Vec::new()),
                     });

        // Iterator for the produced queues.
        let output_queues = QueuesIter {
            next_queue: 0,
            device: device.clone(),
            families_and_ids: output_queues,
        };

        Ok((device, output_queues))
    }

    /// Grants access to the pointers to the Vulkan functions of the device.
    #[inline]
    pub(crate) fn pointers(&self) -> &vk::DevicePointers {
        &self.vk
    }

    /// Waits until all work on this device has finished. You should never need to call
    /// this function, but it can be useful for debugging or benchmarking purposes.
    ///
    /// > **Note**: This is the Vulkan equivalent of OpenGL's `glFinish`.
    ///
    /// # Safety
    ///
    /// This function is not thread-safe. You must not submit anything to any of the queue
    /// of the device (either explicitely or implicitely, for example with a future's destructor)
    /// while this function is waiting.
    ///
    pub unsafe fn wait(&self) -> Result<(), OomError> {
        check_errors(self.vk.DeviceWaitIdle(self.device))?;
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
    pub fn active_queue_families<'a>(&'a self)
                                     -> Box<ExactSizeIterator<Item = QueueFamily<'a>> + 'a> {
        let physical_device = self.physical_device();
        Box::new(self.active_queue_families
                     .iter()
                     .map(move |&id| physical_device.queue_family_by_id(id).unwrap()))
    }

    /// Returns the features that are enabled in the device.
    #[inline]
    pub fn enabled_features(&self) -> &Features {
        &self.features
    }

    /// Returns the list of extensions that have been loaded.
    #[inline]
    pub fn loaded_extensions(&self) -> &DeviceExtensions {
        &self.extensions
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
            },
            Entry::Vacant(entry) => {
                let new_pool = Arc::new(StandardCommandPool::new(me.clone(), queue));
                entry.insert(Arc::downgrade(&new_pool));
                new_pool
            },
        }
    }

    /// Used to track the number of allocations on this device.
    ///
    /// To ensure valid usage of the vulkan API, we cannot call `vkAllocateMemory` when
    /// `maxMemoryAllocationCount` has been exceeded. See the vulkan specs:
    /// https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#vkAllocateMemory
    ///
    /// Warning: You should never modify this value, except in `device_memory` module
    pub(crate) fn allocation_count(&self) -> &Mutex<u32> {
        &self.allocation_count
    }

    pub(crate) fn fence_pool(&self) -> &Mutex<Vec<vk::Fence>> {
        &self.fence_pool
    }

    pub(crate) fn semaphore_pool(&self) -> &Mutex<Vec<vk::Semaphore>> {
        &self.semaphore_pool
    }

    pub(crate) fn event_pool(&self) -> &Mutex<Vec<vk::Event>> {
        &self.event_pool
    }
}

impl fmt::Debug for Device {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan device {:?}>", self.device)
    }
}

unsafe impl VulkanObject for Device {
    type Object = vk::Device;

    #[inline]
    fn internal_object(&self) -> vk::Device {
        self.device
    }
}

impl Drop for Device {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            for &raw_fence in self.fence_pool.lock().unwrap().iter() {
                self.vk.DestroyFence(self.device, raw_fence, ptr::null());
            }
            for &raw_sem in self.semaphore_pool.lock().unwrap().iter() {
                self.vk.DestroySemaphore(self.device, raw_sem, ptr::null());
            }
            for &raw_event in self.event_pool.lock().unwrap().iter() {
                self.vk.DestroyEvent(self.device, raw_event, ptr::null());
            }
            self.vk.DestroyDevice(self.device, ptr::null());
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
    where T: Deref,
          T::Target: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        (**self).device()
    }
}

/// Iterator that returns the queues produced when creating a device.
pub struct QueuesIter {
    next_queue: usize,
    device: Arc<Device>,
    families_and_ids: SmallVec<[(u32, u32); 8]>,
}

unsafe impl DeviceOwned for QueuesIter {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Iterator for QueuesIter {
    type Item = Arc<Queue>;

    fn next(&mut self) -> Option<Arc<Queue>> {
        unsafe {
            let &(family, id) = match self.families_and_ids.get(self.next_queue) {
                Some(a) => a,
                None => return None,
            };

            self.next_queue += 1;

            let mut output = mem::uninitialized();
            self.device
                .vk
                .GetDeviceQueue(self.device.device, family, id, &mut output);

            Some(Arc::new(Queue {
                              queue: Mutex::new(output),
                              device: self.device.clone(),
                              family: family,
                              id: id,
                          }))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.families_and_ids.len().saturating_sub(self.next_queue);
        (len, Some(len))
    }
}

impl ExactSizeIterator for QueuesIter {
}

/// Error that can be returned when creating a device.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
}

impl error::Error for DeviceCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            DeviceCreationError::InitializationFailed => {
                "failed to create the device for an implementation-specific reason"
            },
            DeviceCreationError::OutOfHostMemory => "no memory available on the host",
            DeviceCreationError::OutOfDeviceMemory => {
                "no memory available on the graphical device"
            },
            DeviceCreationError::DeviceLost => {
                "failed to connecgt to the device"
            },
            DeviceCreationError::TooManyQueuesForFamily => {
                "tried to create too many queues for a given family"
            },
            DeviceCreationError::FeatureNotPresent => {
                "some of the requested features are unsupported by the physical device"
            },
            DeviceCreationError::PriorityOutOfRange => {
                "the priority of one of the queues is out of the [0.0; 1.0] range"
            },
            DeviceCreationError::ExtensionNotPresent => {
                "some of the requested device extensions are not supported by the physical device"
            },
            DeviceCreationError::TooManyObjects => {
                "you have reached the limit to the number of devices that can be created from the
                 same physical device"
            },
        }
    }
}

impl fmt::Display for DeviceCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
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

/// Represents a queue where commands can be submitted.
// TODO: should use internal synchronization?
#[derive(Debug)]
pub struct Queue {
    queue: Mutex<vk::Queue>,
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
        self.id == other.id && self.family == other.family &&
            self.device.internal_object() == other.device.internal_object()
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
            let vk = self.device.pointers();
            let queue = self.queue.lock().unwrap();
            check_errors(vk.QueueWaitIdle(*queue))?;
            Ok(())
        }
    }
}


unsafe impl DeviceOwned for Queue {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl SynchronizedVulkanObject for Queue {
    type Object = vk::Queue;

    #[inline]
    fn internal_object_guard(&self) -> MutexGuard<vk::Queue> {
        self.queue.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use device::Device;
    use device::DeviceCreationError;
    use device::DeviceExtensions;
    use features::Features;
    use instance;
    use std::sync::Arc;

    #[test]
    fn one_ref() {
        let (mut device, _) = gfx_dev_and_queue!();
        assert!(Arc::get_mut(&mut device).is_some());
    }

    #[test]
    fn too_many_queues() {
        let instance = instance!();
        let physical = match instance::PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return,
        };

        let family = physical.queue_families().next().unwrap();
        let queues = (0 .. family.queues_count() + 1).map(|_| (family, 1.0));

        match Device::new(physical,
                            &Features::none(),
                            &DeviceExtensions::none(),
                            queues) {
            Err(DeviceCreationError::TooManyQueuesForFamily) => return,     // Success
            _ => panic!(),
        };
    }

    #[test]
    fn unsupposed_features() {
        let instance = instance!();
        let physical = match instance::PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return,
        };

        let family = physical.queue_families().next().unwrap();

        let features = Features::all();
        // In the unlikely situation where the device supports everything, we ignore the test.
        if physical.supported_features().superset_of(&features) {
            return;
        }

        match Device::new(physical,
                            &features,
                            &DeviceExtensions::none(),
                            Some((family, 1.0))) {
            Err(DeviceCreationError::FeatureNotPresent) => return,     // Success
            _ => panic!(),
        };
    }

    #[test]
    fn priority_out_of_range() {
        let instance = instance!();
        let physical = match instance::PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return,
        };

        let family = physical.queue_families().next().unwrap();

        match Device::new(physical,
                            &Features::none(),
                            &DeviceExtensions::none(),
                            Some((family, 1.4))) {
            Err(DeviceCreationError::PriorityOutOfRange) => (),     // Success
            _ => panic!(),
        };

        match Device::new(physical,
                            &Features::none(),
                            &DeviceExtensions::none(),
                            Some((family, -0.2))) {
            Err(DeviceCreationError::PriorityOutOfRange) => (),     // Success
            _ => panic!(),
        };
    }
}
