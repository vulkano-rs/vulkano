// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo,
};
use vulkano::instance::debug::{DebugUtilsMessenger, DebugUtilsMessengerCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::Version;

/// A configuration struct to pass various creation options to create [`VulkanoContext`].
pub struct VulkanoConfig {
    instance_create_info: InstanceCreateInfo,
    /// Pass the `DebugUtilsMessengerCreateInfo` to create the debug callback
    /// for printing debug information at runtime.
    debug_create_info: Option<DebugUtilsMessengerCreateInfo>,
    /// Pass filter function for your physical device selection. See default for example.
    device_filter_fn: Arc<dyn Fn(&PhysicalDevice) -> bool>,
    /// Pass priority order function for your physical device selection. See default for example.
    device_priority_fn: Arc<dyn Fn(&PhysicalDevice) -> u32>,
    device_extensions: DeviceExtensions,
    device_features: Features,
    /// Print your selected device name at start.
    print_device_name: bool,
}

impl Default for VulkanoConfig {
    fn default() -> Self {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        VulkanoConfig {
            instance_create_info: InstanceCreateInfo {
                application_version: Version::V1_2,
                enabled_extensions: vulkano_win::required_extensions(),
                ..Default::default()
            },
            debug_create_info: None,
            device_filter_fn: Arc::new(move |p| {
                p.supported_extensions().is_superset_of(&device_extensions)
            }),
            device_priority_fn: Arc::new(|p| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 1,
                PhysicalDeviceType::IntegratedGpu => 2,
                PhysicalDeviceType::VirtualGpu => 3,
                PhysicalDeviceType::Cpu => 4,
                PhysicalDeviceType::Other => 5,
            }),
            print_device_name: true,
            device_extensions,
            device_features: Features::none(),
        }
    }
}

/// A utility struct to create, access and hold alive Vulkano device, instance and queues.
///
/// Vulkano context is used in the creation of your graphics or compute pipelines, images and
/// in the creation of [`VulkanoWindowRenderer`] through [`VulkanoWindows`].
///
/// ## Example
///
/// ```
/// use vulkano_util::context::{VulkanoConfig, VulkanoContext};
///
/// #[test]
/// fn test() {
///     let context = VulkanoContext::new(VulkanoConfig::default());
///     // Then create event loop, windows, pipelines, etc.
/// }
/// ```
pub struct VulkanoContext {
    instance: Arc<Instance>,
    _debug_utils_messenger: Option<DebugUtilsMessenger>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    compute_queue: Arc<Queue>,
}

unsafe impl Sync for VulkanoContext {}

unsafe impl Send for VulkanoContext {}

impl Default for VulkanoContext {
    fn default() -> Self {
        VulkanoContext::new(VulkanoConfig::default())
    }
}

impl VulkanoContext {
    /// Creates a new [`VulkanoContext`].
    ///
    /// # Panics
    ///
    /// - Panics where the underlying Vulkano struct creations fail
    pub fn new(mut config: VulkanoConfig) -> Self {
        // Create instance
        let instance = create_instance(config.instance_create_info);
        // Create debug callback
        let _debug_utils_messenger = if let Some(dbg_create_info) = config.debug_create_info.take()
        {
            Some(unsafe {
                DebugUtilsMessenger::new(instance.clone(), dbg_create_info)
                    .expect("Failed to create debug callback")
            })
        } else {
            None
        };
        // Get prioritized device
        let physical_device = PhysicalDevice::enumerate(&instance)
            .filter(|p| (config.device_filter_fn)(p))
            .min_by_key(|p| (config.device_priority_fn)(p))
            .expect("Failed to create physical device");
        // Print used device
        if config.print_device_name {
            println!(
                "Using device {}, type: {:?}",
                physical_device.properties().device_name,
                physical_device.properties().device_type,
            );
        }

        // Create device
        let (device, graphics_queue, compute_queue) = Self::create_device(
            physical_device,
            config.device_extensions,
            config.device_features,
        );

        Self {
            instance,
            _debug_utils_messenger,
            device,
            graphics_queue,
            compute_queue,
        }
    }

    /// Creates vulkano device with required queue families and required extensions. Creates a separate queue for compute
    /// if possible. If not, same queue as graphics is used.
    fn create_device(
        physical: PhysicalDevice,
        device_extensions: DeviceExtensions,
        features: Features,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let (gfx_index, queue_family_graphics) = physical
            .queue_families()
            .enumerate()
            .find(|&(_i, q)| q.supports_graphics())
            .expect("Could not find a queue that supports graphics");
        // Try finding a separate queue for compute
        let compute_family_data = physical
            .queue_families()
            .enumerate()
            .find(|&(i, q)| q.supports_compute() && i != gfx_index);

        let is_separate_compute_queue = compute_family_data.is_some();
        let queue_create_infos = if is_separate_compute_queue {
            let (_i, queue_family_compute) = compute_family_data.unwrap();
            vec![
                QueueCreateInfo::family(queue_family_graphics),
                QueueCreateInfo::family(queue_family_compute),
            ]
        } else {
            vec![QueueCreateInfo::family(queue_family_graphics)]
        };
        let (device, mut queues) = {
            Device::new(
                physical,
                DeviceCreateInfo {
                    enabled_extensions: physical.required_extensions().union(&device_extensions),
                    enabled_features: features,
                    queue_create_infos,
                    ..Default::default()
                },
            )
            .expect("Failed to create device")
        };
        let gfx_queue = queues.next().unwrap();
        let compute_queue = if is_separate_compute_queue {
            queues.next().unwrap()
        } else {
            gfx_queue.clone()
        };
        (device, gfx_queue, compute_queue)
    }

    /// Check device name
    pub fn device_name(&self) -> &str {
        &self.device.physical_device().properties().device_name
    }

    /// Check device type
    pub fn device_type(&self) -> PhysicalDeviceType {
        self.device.physical_device().properties().device_type
    }

    /// Check device memory count
    pub fn max_memory(&self) -> u32 {
        self.device
            .physical_device()
            .properties()
            .max_memory_allocation_count as u32
    }

    /// Access instance
    pub fn instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }

    /// Access device
    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// Access rendering queue
    pub fn graphics_queue(&self) -> Arc<Queue> {
        self.graphics_queue.clone()
    }

    /// Access compute queue. Depending on your device, this might be the same as graphics queue.
    pub fn compute_queue(&self) -> Arc<Queue> {
        self.compute_queue.clone()
    }
}

/// Create instance, but remind user to install vulkan SDK on mac os if loading error is received on that platform.
fn create_instance(instance_create_info: InstanceCreateInfo) -> Arc<Instance> {
    #[cfg(target_os = "macos")]
    {
        match Instance::new(instance_create_info) {
            Err(e) => match e {
                InstanceCreationError::LoadingError(le) => {
                     Err(le).expect("Failed to create instance. Did you install vulkanSDK from https://vulkan.lunarg.com/sdk/home ?")
                }
                _ => Err(e).expect("Failed to create instance"),
            },
            Ok(i) => i,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        Instance::new(instance_create_info).expect("Failed to create instance")
    }
}

#[cfg(test)]
mod tests {
    use crate::context::{VulkanoConfig, VulkanoContext};

    // Simply test creation of the context...
    #[test]
    fn test_creation() {
        let context = VulkanoContext::new(VulkanoConfig::default());
        assert_ne!(context.max_memory(), 0);
    }
}
