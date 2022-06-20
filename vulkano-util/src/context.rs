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

pub struct VulkanoConfig {
    instance_create_info: InstanceCreateInfo,
    debug_create_info: Option<DebugUtilsMessengerCreateInfo>,
    device_priority_fn: fn(device_type: PhysicalDeviceType) -> u32,
    device_extensions: DeviceExtensions,
    device_features: Features,
    print_device_name: bool,
}

impl Default for VulkanoConfig {
    fn default() -> Self {
        VulkanoConfig {
            instance_create_info: InstanceCreateInfo {
                application_version: Version::V1_2,
                enabled_extensions: vulkano_win::required_extensions(),
                ..Default::default()
            },
            debug_create_info: None,
            device_priority_fn: |p| match p {
                PhysicalDeviceType::DiscreteGpu => 1,
                PhysicalDeviceType::IntegratedGpu => 2,
                PhysicalDeviceType::VirtualGpu => 3,
                PhysicalDeviceType::Cpu => 4,
                PhysicalDeviceType::Other => 5,
            },
            print_device_name: true,
            device_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::none()
            },
            device_features: Features::none(),
        }
    }
}

/// VulkanoContext is a utility struct to create and access Vulkano device(s), queues and so on.
/// ## Example
///
/// ```
/// use vulkano_util::context::{VulkanoConfig, VulkanoContext};
///
/// #[test]
/// fn test() {
///     let context = VulkanoContext::new(VulkanoConfig::default());
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
    /// Creates a new `VulkanoContext`.
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
            .min_by_key(|p| (config.device_priority_fn)(p.properties().device_type))
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

    /// Creates vulkan device with required queue families and required extensions. Returns an optional secondary queue for compute.
    /// However, typically you can just use the gfx queue for that.
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
        let compute_family_data = physical
            .queue_families()
            .enumerate()
            .find(|&(i, q)| i != gfx_index && q.supports_compute());

        // If we have an extra compute queue:
        if let Some((_compute_index, queue_family_compute)) = compute_family_data {
            let (device, mut queues) = {
                Device::new(
                    physical,
                    DeviceCreateInfo {
                        enabled_extensions: physical
                            .required_extensions()
                            .union(&device_extensions),
                        enabled_features: features,
                        queue_create_infos: vec![
                            QueueCreateInfo::family(queue_family_graphics),
                            QueueCreateInfo::family(queue_family_compute),
                        ],
                        ..Default::default()
                    },
                )
                .expect("Failed to create device")
            };
            let gfx_queue = queues.next().unwrap();
            let compute_queue = queues.next().unwrap();
            (device, gfx_queue, compute_queue)
        }
        // And if we do not have an extra compute queue, just use the same queue for gfx and compute
        else {
            let (device, mut queues) = {
                Device::new(
                    physical,
                    DeviceCreateInfo {
                        enabled_extensions: physical
                            .required_extensions()
                            .union(&device_extensions),
                        enabled_features: features,
                        queue_create_infos: vec![QueueCreateInfo::family(queue_family_graphics)],
                        ..Default::default()
                    },
                )
                .expect("Failed to create device")
            };
            let gfx_queue = queues.next().unwrap();
            let compute_queue = gfx_queue.clone();
            (device, gfx_queue, compute_queue)
        }
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
