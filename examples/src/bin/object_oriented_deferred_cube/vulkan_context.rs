use std::sync::Arc;

use anyhow::anyhow;
use log::info;

use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};

use vulkano::instance::Instance;
use vulkano::swapchain::Surface;

/// A convenient wrapper around the contexts bound to a particular window.
pub struct VulkanContext<TWindow> {
    /// The logical device is the actual Vulkan context,
    /// bound to a specific GPU (or other Vulkan-compatible hardware, even software)
    /// with fixed functionality specified at creation.
    /// If any of these need to be changed, we need to create a new one
    /// (and re-create everything else except `Instance` and `Surface`).
    /// That's why most games require a restart upon changing some graphics settings -
    /// writing code for re-creating everything is too much bother.
    device: Arc<Device>,
    /// The surface is the handle needed to paint to a specific window,
    /// but it's not the actual "canvas". (The actual "canvas" is the swapchain images.)
    /// It holds a reference to `TWindow` to enforce correct dropping order for winit.
    surface: Arc<Surface<TWindow>>,
    /// Queues are what we use to submit commands to the GPU.
    /// They might only support a subset of all the possible operations in order to be fast at that.
    /// For simplicity, we'll only use graphics-capable queues,
    /// which supports all types of operations, is guaranteed to be redundant on actual GPUs
    /// and almost as fast as other dedicated queues for simple use cases AFAIK.
    queues: Arc<Vec<Arc<Queue>>>,
}

impl<TWindow> VulkanContext<TWindow> {
    /// The `instance` and `surface` are passed in as dependencies
    /// to be agnostic of windowing libraries.
    /// (I know the pain when I found out I hate winit and want to swap to SDL)
    /// Plus, if you actually want to implement on-the-fly rebuilding of the entire graphics part,
    /// you should probably not put instance and surface creation in here.
    pub fn new(
        instance: Arc<Instance>,
        surface: Arc<Surface<TWindow>>,
        extra_device_extensions: DeviceExtensions,
        extra_features: Features,
    ) -> anyhow::Result<Self> {
        // Device extensions are extensions that introduce shiny new functionalities
        // but some devices may not support.
        // You need to explicitly enable extensions before using them.
        // Here's a list of extension support coverage: https://vulkan.gpuinfo.org/listextensions.php
        let device_extensions = DeviceExtensions {
            // The swapchain extension is needed for rendering to a window.
            // It sounds like a basic functionality that every Vulkan implementation supports,
            // but it is somehow an extension.
            khr_swapchain: true,
            ..extra_device_extensions
        };

        // Choose which physical device to use.
        // First, we enumerate all the available physical devices,
        // then apply filters to narrow them down to those that can support our needs.
        let (physical_device, queue_family): (PhysicalDevice, QueueFamily) =
            PhysicalDevice::enumerate(&instance)
                .filter(|&p| {
                    // Some devices may not support the extensions or features that your application, or
                    // report properties and limits that are not sufficient for your application. These
                    // should be filtered out here.
                    p.supported_extensions().is_superset_of(&device_extensions)
                })
                .filter_map(|p| {
                    // For each physical device, we try to find a suitable queue family that will execute
                    // our draw commands.
                    //
                    // Devices can provide multiple queues to run commands in parallel (for example a draw
                    // queue and a compute queue), similar to CPU threads. This is something you have to
                    // have to manage manually in Vulkan. Queues of the same type belong to the same
                    // queue family.
                    //
                    // Here, we look for a single queue family that is suitable for our purposes. In a
                    // real-life application, you may want to use a separate dedicated transfer queue to
                    // handle data transfers in parallel with graphics operations. You may also need a
                    // separate queue for compute operations, if your application uses those.
                    p.queue_families()
                        .find(|&q| {
                            // We select a queue family that supports graphics operations. When drawing to
                            // a window surface, as we do in this example, we also need to check that queues
                            // in this queue family are capable of presenting images to the surface.
                            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
                        })
                        // The code here searches for the first queue family that is suitable. If none is
                        // found, `None` is returned to `filter_map`, which disqualifies this physical
                        // device.
                        .map(|q| (p, q))
                })
                // All the physical devices that pass the filters above are suitable for the application.
                // However, not every device is equal, some are preferred over others. Now, we assign
                // each physical device a score, and pick the device with the
                // lowest ("best") score.
                //
                // In this example, we simply select the best-scoring device to use in the application.
                // In a real-life setting, you may want to use the best-scoring device only as a
                // "default" or "recommended" device, and let the user choose the device themselves.
                .max_by_key(|(p, _)| {
                    // We assign a better score to device types that are likely to be faster/better.
                    match p.properties().device_type {
                        PhysicalDeviceType::DiscreteGpu => 5,
                        PhysicalDeviceType::IntegratedGpu => 4,
                        PhysicalDeviceType::VirtualGpu => 3,
                        PhysicalDeviceType::Cpu => 2,
                        PhysicalDeviceType::Other => 1,
                    }
                })
                .ok_or_else(|| anyhow!("No Vulkan physical device found"))?;
        info!(
            "Using physical device {:}",
            physical_device.properties().device_name
        );

        // Create the device.
        let (device, queues) = {
            // Physical device might require additional extensions.
            let device_extensions = device_extensions.union(physical_device.required_extensions());
            // Features are basically extensions that have been promoted into the core Vulkan specs,
            // but they still might not be supported by all GPUs.
            // List of features for Vulkan 1.2: https://vulkan.gpuinfo.org/listfeaturescore12.php
            let features = extra_features;

            Device::new(
                physical_device,
                &features,
                &device_extensions,
                [(queue_family, 0.5)].iter().cloned(),
            )?
        };
        let queues: Arc<Vec<_>> = Arc::new(queues.collect());

        // return
        Ok(Self {
            device,
            surface,
            queues,
        })
    }

    //#region Getters
    #[inline]
    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    #[inline]
    pub fn surface(&self) -> Arc<Surface<TWindow>> {
        self.surface.clone()
    }

    /// We'll just use the first queue for rendering.
    /// You might want to balance works between queues (like putting shadow mapping on another one),
    /// because commands submitted to the same queue are executed sequentially.
    /// But, you should put sequential work on the same queue,
    /// because syncing between queues on the GPU (semaphores) is as costly as syncing on the CPU.
    #[inline]
    pub fn main_queue(&self) -> Arc<Queue> {
        self.queues[0].clone()
    }
}

/// https://stackoverflow.com/questions/39415052/deriving-a-trait-results-in-unexpected-compiler-error-but-the-manual-implementa
impl<TWindow> Clone for VulkanContext<TWindow> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            surface: self.surface.clone(),
            queues: self.queues.clone(),
        }
    }
}
