use scene::Scene;
use std::{error::Error, sync::Arc};
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
        Queue, QueueCreateInfo, QueueFlags,
    },
    image::{ImageFormatInfo, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{layout::PipelineLayoutCreateInfo, PipelineLayout},
    shader::ShaderStages,
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Version, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod scene;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    rcx: Option<RenderContext>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

pub struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    recreate_swapchain: bool,
    scene: Scene,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: InstanceExtensions {
                    ext_swapchain_colorspace: true,
                    ..required_extensions
                },
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_ray_tracing_pipeline: true,
            khr_ray_tracing_maintenance1: true,
            khr_synchronization2: true,
            khr_deferred_host_operations: true,
            khr_acceleration_structure: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.api_version() >= Version::V1_3)
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags
                            .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_features: DeviceFeatures {
                    acceleration_structure: true,
                    ray_tracing_pipeline: true,
                    buffer_device_address: true,
                    synchronization2: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        App {
            instance,
            device,
            queue,
            rcx: None,
            command_buffer_allocator,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();

        let physical_device = self.device.physical_device();
        let supported_surface_formats = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap();

        // For each supported format, check if it is supported for storage images
        let supported_storage_formats = supported_surface_formats
            .into_iter()
            .filter(|(format, _)| {
                physical_device
                    .image_format_properties(ImageFormatInfo {
                        format: *format,
                        usage: ImageUsage::STORAGE,
                        ..Default::default()
                    })
                    .unwrap()
                    .is_some()
            })
            .collect::<Vec<_>>();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let (swapchain_format, swapchain_color_space) = supported_storage_formats
                .first()
                .map(|(format, color_space)| (*format, *color_space))
                .unwrap();
            Swapchain::new(
                self.device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format: swapchain_format,
                    image_color_space: swapchain_color_space,
                    image_extent: window.inner_size().into(),
                    // To simplify the example, we will directly write to the swapchain images
                    // from the ray tracing shader. This requires the images to support storage
                    // usage.
                    image_usage: ImageUsage::STORAGE,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let pipeline_layout = PipelineLayout::new(
            self.device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![
                    DescriptorSetLayout::new(
                        self.device.clone(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: [
                                (
                                    0,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::RAYGEN,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::AccelerationStructure,
                                        )
                                    },
                                ),
                                (
                                    1,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::RAYGEN,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::UniformBuffer,
                                        )
                                    },
                                ),
                            ]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                    DescriptorSetLayout::new(
                        self.device.clone(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: [(
                                0,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::RAYGEN,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::StorageImage,
                                    )
                                },
                            )]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                ],
                ..Default::default()
            },
        )
        .unwrap();

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            self.device.clone(),
            Default::default(),
        ));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));

        let scene = Scene::new(
            self,
            &images,
            pipeline_layout,
            descriptor_set_allocator.clone(),
            memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
        );
        self.rcx = Some(RenderContext {
            window,
            swapchain,
            recreate_swapchain: false,
            previous_frame_end: None,
            scene,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // Cleanup previous frame
                if let Some(previous_frame_end) = rcx.previous_frame_end.as_mut() {
                    previous_frame_end.cleanup_finished();
                }

                // Recreate swapchain if needed
                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) =
                        match rcx.swapchain.recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(e) => panic!("Failed to recreate swapchain: {e:?}"),
                        };

                    rcx.swapchain = new_swapchain;
                    rcx.scene.handle_resize(&new_images);
                    rcx.recreate_swapchain = false;
                }

                // Acquire next image
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(rcx.swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("Failed to acquire next image: {e:?}");
                            rcx.recreate_swapchain = true;
                            return;
                        }
                    };

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                rcx.scene.record_commands(image_index, &mut builder);

                let command_buffer = builder.build().unwrap();

                let future = rcx
                    .previous_frame_end
                    .take()
                    .unwrap_or_else(|| {
                        Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>
                    })
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        rcx.previous_frame_end = Some(Box::new(future) as Box<dyn GpuFuture>);
                    }
                    Err(e) => {
                        println!("Failed to flush future: {e:?}");
                        rcx.previous_frame_end = Some(Box::new(sync::now(self.device.clone())));
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}
