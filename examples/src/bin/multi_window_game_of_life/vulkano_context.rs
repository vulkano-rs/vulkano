// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
use std::sync::Arc;

#[cfg(target_os = "macos")]
use vulkano::instance::InstanceCreationError;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceExtensions, Features, Queue, QueueCreate,
    },
    image::{view::ImageView, ImageUsage},
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        Instance, InstanceExtensions,
    },
    swapchain::{
        ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform, Swapchain,
    },
    Version,
};
use winit::window::Window;

use crate::vulkano_config::VulkanoConfig;
use vulkano::image::{StorageImage, SwapchainImage};

/// Final render target onto which whole app is rendered (per window)
pub type FinalImageView = Arc<ImageView<SwapchainImage<Window>>>;
/// Multipurpose image view
pub type DeviceImageView = Arc<ImageView<StorageImage>>;

/// Vulkano context provides access to device, graphics queues and instance allowing you to separate
/// window handling and the context initiation.
/// The purpose of this struct is to allow you to focus on the graphics keep your code much less verbose
pub struct VulkanoContext {
    _debug_callback: DebugCallback,
    instance: Arc<Instance>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    compute_queue: Arc<Queue>,
    device_name: String,
    device_type: PhysicalDeviceType,
    max_mem_gb: f32,
}

unsafe impl Sync for VulkanoContext {}

unsafe impl Send for VulkanoContext {}

impl VulkanoContext {
    pub fn new(config: &VulkanoConfig) -> Self {
        let instance = create_vk_instance(config.instance_extensions, &config.layers);
        let is_debug = config
            .layers
            .contains(&"VK_LAYER_LUNARG_standard_validation")
            || config.layers.contains(&"VK_LAYER_KHRONOS_validation");
        let debug_callback = create_vk_debug_callback(&instance, is_debug);
        // Get desired device
        let physical_device = PhysicalDevice::enumerate(&instance)
            .min_by_key(|p| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 1,
                PhysicalDeviceType::IntegratedGpu => 2,
                PhysicalDeviceType::VirtualGpu => 3,
                PhysicalDeviceType::Cpu => 4,
                PhysicalDeviceType::Other => 5,
            })
            .unwrap();
        let device_name = physical_device.properties().device_name.to_string();
        #[cfg(target_os = "windows")]
        let max_mem_gb = physical_device.properties().max_memory_allocation_count as f32 * 9.31e-4;
        #[cfg(not(target_os = "windows"))]
        let max_mem_gb = physical_device.properties().max_memory_allocation_count as f32 * 9.31e-10;
        println!(
            "Using device {}, type: {:?}, mem: {:.2} gb",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
            max_mem_gb,
        );
        let device_type = physical_device.properties().device_type;

        // Create device
        let (device, graphics_queue, compute_queue) = Self::create_device(
            physical_device,
            config.device_extensions,
            config.features.clone(),
        );

        Self {
            instance,
            _debug_callback: debug_callback,
            device,
            graphics_queue,
            compute_queue,
            device_name,
            device_type,
            max_mem_gb,
        }
    }

    /// Creates vulkan device with required queue families and required extensions
    fn create_device(
        physical: PhysicalDevice,
        device_extensions: DeviceExtensions,
        features: Features,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        // Choose a graphics queue family
        let (gfx_index, queue_family_graphics) = physical
            .queue_families()
            .enumerate()
            .find(|&(_i, q)| q.supports_graphics())
            .unwrap();
        // Choose compute queue family (separate from gfx)
        let compute_family_data = physical
            .queue_families()
            .enumerate()
            .find(|&(i, q)| i != gfx_index && q.supports_compute());

        // If we can create a compute queue, do so. Else use same queue as graphics
        if let Some((_compute_index, queue_family_compute)) = compute_family_data {
            let (device, mut queues) = Device::start()
                .queues([
                    QueueCreate::family(queue_family_graphics).queues([1.0]),
                    QueueCreate::family(queue_family_compute).queues([0.5]),
                ])
                .enabled_extensions(physical.required_extensions().union(&device_extensions))
                .enabled_features(features)
                .build(physical)
                .unwrap();
            let gfx_queue = queues.next().unwrap();
            let compute_queue = queues.next().unwrap();
            (device, gfx_queue, compute_queue)
        } else {
            let (device, mut queues) = Device::start()
                .queues([QueueCreate::family(queue_family_graphics)])
                .enabled_extensions(physical.required_extensions().union(&device_extensions))
                .enabled_features(features)
                .build(physical)
                .unwrap();
            let gfx_queue = queues.next().unwrap();
            let compute_queue = gfx_queue.clone();
            (device, gfx_queue, compute_queue)
        }
    }

    /// Creates swapchain and swapchain images
    pub(crate) fn create_swap_chain(
        &self,
        surface: Arc<Surface<Window>>,
        queue: Arc<Queue>,
        present_mode: PresentMode,
    ) -> (Arc<Swapchain<Window>>, Vec<FinalImageView>) {
        let caps = surface.capabilities(self.device.physical_device()).unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        let (swap_chain, images) = Swapchain::start(self.device.clone(), surface)
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(alpha)
            .transform(SurfaceTransform::Identity)
            .present_mode(present_mode)
            .fullscreen_exclusive(FullscreenExclusive::Default)
            .clipped(true)
            .color_space(ColorSpace::SrgbNonLinear)
            .layers(1)
            .build()
            .unwrap();
        let images = images
            .into_iter()
            .map(|image| ImageView::new(image).unwrap())
            .collect::<Vec<_>>();
        (swap_chain, images)
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn device_type(&self) -> PhysicalDeviceType {
        self.device_type
    }

    pub fn max_mem_gb(&self) -> f32 {
        self.max_mem_gb
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

    /// Access compute queue
    pub fn compute_queue(&self) -> Arc<Queue> {
        self.compute_queue.clone()
    }
}

// Create vk instance with given layers
pub fn create_vk_instance(
    instance_extensions: InstanceExtensions,
    layers: &[&str],
) -> Arc<Instance> {
    // Create instance.
    let result = Instance::start()
        .enabled_extensions(instance_extensions)
        .enabled_layers(layers.to_vec())
        .build();

    // Handle errors. On mac os, it will ask you to install vulkan sdk if you have not done so.
    #[cfg(target_os = "macos")]
    let instance = match result {
        Err(e) => match e {
            InstanceCreationError::LoadingError(le) => {
                println!(
                    "{:?}, Did you install vulkanSDK from https://vulkan.lunarg.com/sdk/home ?",
                    le
                );
                Err(le).expect("")
            }
            _ => Err(e).expect("Failed to create instance"),
        },
        Ok(i) => i,
    };
    #[cfg(not(target_os = "macos"))]
    let instance = result.expect("Failed to create instance");

    instance
}

// Create vk debug call back (to exists outside renderer)
pub fn create_vk_debug_callback(instance: &Arc<Instance>, is_debug: bool) -> DebugCallback {
    // Create debug callback for printing vulkan errors and warnings. This will do nothing unless the layers are enabled
    let severity = if is_debug {
        MessageSeverity {
            error: true,
            warning: true,
            information: true,
            verbose: true,
        }
    } else {
        MessageSeverity::none()
    };

    let ty = MessageType::all();
    DebugCallback::new(instance, severity, ty, |msg| {
        let severity = if msg.severity.error {
            "error"
        } else if msg.severity.warning {
            "warning"
        } else if msg.severity.information {
            "information"
        } else if msg.severity.verbose {
            "verbose"
        } else {
            panic!("no-impl");
        };

        let ty = if msg.ty.general {
            "general"
        } else if msg.ty.validation {
            "validation"
        } else if msg.ty.performance {
            "performance"
        } else {
            panic!("no-impl");
        };

        println!(
            "{} {} {}: {}",
            msg.layer_prefix.unwrap_or("unknown"),
            ty,
            severity,
            msg.description
        );
    })
    .unwrap()
}
