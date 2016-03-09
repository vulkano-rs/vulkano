//! Communication channel with a physical device.
//!
//! The `Device` is one of the most important objects of Vulkan. Creating a `Device` is required
//! before you can create buffers, textures, shaders, etc.
//!
use std::ffi::CString;
use std::fmt;
use std::error;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;

use instance::Features;
use instance::Instance;
use instance::PhysicalDevice;
use instance::QueueFamily;

use Error;
use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Represents a Vulkan context.
pub struct Device {
    instance: Arc<Instance>,
    physical_device: usize,
    device: vk::Device,
    vk: vk::DevicePointers,
    features: Features,
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
    /// - Panicks if one of the requested features is not supported by the physical device.
    /// - Panicks if one of the queue families doesn't belong to the given device.
    /// - Panicks if you request more queues from a family than available.
    /// - Panicks if one of the priorities is outside of the `[0.0 ; 1.0]` range.
    ///
    // TODO: return Arc<Queue> and handle synchronization in the Queue
    pub fn new<'a, I, L>(phys: &'a PhysicalDevice, requested_features: &Features, queue_families: I,
                         layers: L)
                         -> Result<(Arc<Device>, Vec<Arc<Queue>>), DeviceCreationError>
        where I: IntoIterator<Item = (QueueFamily<'a>, f32)>,
              L: IntoIterator<Item = &'a &'a str>
    {
        let queue_families = queue_families.into_iter();

        assert!(phys.supported_features().superset_of(&requested_features));

        let vk_i = phys.instance().pointers();

        // this variable will contain the queue family ID and queue ID of each requested queue
        let mut output_queues: Vec<(u32, u32)> = Vec::with_capacity(queue_families.size_hint().0);

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let layers = layers.into_iter().map(|&layer| {
            // FIXME: check whether each layer is supported
            CString::new(layer).unwrap()
        }).collect::<Vec<_>>();
        let layers = layers.iter().map(|layer| {
            layer.as_ptr()
        }).collect::<Vec<_>>();

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let extensions = ["VK_KHR_swapchain"].iter().map(|&ext| {
            // FIXME: check whether each extension is supported
            CString::new(ext).unwrap()
        }).collect::<Vec<_>>();
        let extensions = extensions.iter().map(|extension| {
            extension.as_ptr()
        }).collect::<Vec<_>>();

        // device creation
        let device = unsafe {
            // each element of `queues` is a `(queue_family, priorities)`
            // each queue family must only have one entry in `queues`
            let mut queues: Vec<(u32, Vec<f32>)> = Vec::with_capacity(phys.queue_families().len());

            for (queue_family, priority) in queue_families {
                // checking the parameters
                assert_eq!(queue_family.physical_device().internal_object(),
                           phys.internal_object());
                assert!(priority >= 0.0 && priority <= 1.0);

                // adding to `queues` and `output_queues`
                if let Some(q) = queues.iter_mut().find(|q| q.0 == queue_family.id()) {
                    output_queues.push((queue_family.id(), q.1.len() as u32));
                    q.1.push(priority);
                    assert!(q.1.len() < queue_family.queues_count());
                    continue;
                }
                queues.push((queue_family.id(), vec![priority]));
                output_queues.push((queue_family.id(), 0));
            }

            // turning `queues` into an array of `vkDeviceQueueCreateInfo` suitable for Vulkan
            // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
            let queues = queues.iter().map(|&(queue_id, ref priorities)| {
                vk::DeviceQueueCreateInfo {
                    sType: vk::STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,   // reserved
                    queueFamilyIndex: queue_id,
                    queueCount: priorities.len() as u32,
                    pQueuePriorities: priorities.as_ptr()
                }
            }).collect::<Vec<_>>();

            let features: vk::PhysicalDeviceFeatures = requested_features.clone().into();

            let infos = vk::DeviceCreateInfo {
                sType: vk::STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                queueCreateInfoCount: queues.len() as u32,
                pQueueCreateInfos: queues.as_ptr(),
                enabledLayerCount: layers.len() as u32,
                ppEnabledLayerNames: layers.as_ptr(),
                enabledExtensionCount: extensions.len() as u32,
                ppEnabledExtensionNames: extensions.as_ptr(),
                pEnabledFeatures: &features,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk_i.CreateDevice(phys.internal_object(), &infos,
                                                ptr::null(), &mut output)));
            output
        };

        // loading the function pointers of the newly-created device
        let vk = vk::DevicePointers::load(|name| {
            unsafe { vk_i.GetDeviceProcAddr(device, name.as_ptr()) as *const _ }
        });

        let device = Arc::new(Device {
            instance: phys.instance().clone(),
            physical_device: phys.index(),
            device: device,
            vk: vk,
            features: requested_features.clone(),
        });

        // querying the queues
        let output_queues = output_queues.into_iter().map(|(family, id)| {
            unsafe {
                let mut output = mem::uninitialized();
                device.vk.GetDeviceQueue(device.device, family, id, &mut output);
                Arc::new(Queue {
                    queue: Mutex::new(output),
                    device: device.clone(),
                    family: family,
                    id: id,
                })
            }
        }).collect();

        Ok((device, output_queues))
    }

    /// Waits until all work on this device has finished. You should never need to call
    /// this function, but it can be useful for debugging or benchmarking purposes.
    ///
    /// This is the Vulkan equivalent of `glFinish`.
    // FIXME: must synchronize all queuees
    #[inline]
    pub fn wait(&self) -> Result<(), OomError> {
        unsafe {
            try!(check_errors(self.vk.DeviceWaitIdle(self.device)));
            Ok(())
        }
    }

    /// Returns the physical device that was used to create this device.
    #[inline]
    pub fn physical_device(&self) -> PhysicalDevice {
        PhysicalDevice::from_index(&self.instance, self.physical_device).unwrap()
    }

    /// Returns the features that are enabled in the device.
    #[inline]
    pub fn enabled_features(&self) -> &Features {
        &self.features
    }
}

impl fmt::Debug for Device {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan device>")
    }
}

unsafe impl VulkanObject for Device {
    type Object = vk::Device;

    #[inline]
    fn internal_object(&self) -> vk::Device {
        self.device
    }
}

impl VulkanPointers for Device {
    type Pointers = vk::DevicePointers;

    #[inline]
    fn pointers(&self) -> &vk::DevicePointers {
        &self.vk
    }
}

impl Drop for Device {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.vk.DeviceWaitIdle(self.device);
            self.vk.DestroyDevice(self.device, ptr::null());
        }
    }
}

/// Error that can be returned when creating a device.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeviceCreationError {
    /// There is no memory available on the host (ie. the CPU, RAM, etc.).
    OutOfHostMemory,
    /// There is no memory available on the device (ie. video memory).
    OutOfDeviceMemory,
    // FIXME: other values
}

impl error::Error for DeviceCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            DeviceCreationError::OutOfHostMemory => "no memory available on the host",
            DeviceCreationError::OutOfDeviceMemory => "no memory available on the graphical device",
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
            Error::OutOfHostMemory => DeviceCreationError::OutOfHostMemory,
            Error::OutOfDeviceMemory => DeviceCreationError::OutOfDeviceMemory,
            _ => panic!("Unexpected error value: {}", err as i32)
        }
    }
}

/// Represents a queue where commands can be submitted.
// TODO: should use internal synchronization
pub struct Queue {
    queue: Mutex<vk::Queue>,
    device: Arc<Device>,
    family: u32,
    id: u32,    // id within family
}

impl Queue {
    /// Returns the device this queue belongs to.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the family this queue belongs to.
    #[inline]
    pub fn family(&self) -> QueueFamily {
        self.device.physical_device().queue_family_by_id(self.family).unwrap()
    }

    /// Waits until all work on this queue has finished.
    ///
    /// Just like `Device::wait()`, you shouldn't have to call this function.
    #[inline]
    pub fn wait(&self) -> Result<(), OomError> {
        unsafe {
            let vk = self.device.pointers();
            let queue = self.queue.lock().unwrap();
            try!(check_errors(vk.QueueWaitIdle(*queue)));
            Ok(())
        }
    }
}

unsafe impl SynchronizedVulkanObject for Queue {
    type Object = vk::Queue;

    #[inline]
    fn internal_object_guard(&self) -> MutexGuard<vk::Queue> {
        self.queue.lock().unwrap()
    }
}
