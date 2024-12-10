// TODO: document

use scene::SceneTask;
use std::{error::Error, sync::Arc};
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
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
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    Validated, Version, VulkanError, VulkanLibrary,
};
use vulkano_taskgraph::{
    graph::{CompileInfo, ExecutableTaskGraph, ExecuteError, NodeId, TaskGraph},
    resource::{AccessType, Flight, ImageLayoutType, Resources},
    resource_map, Id, QueueFamilyType,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod scene;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
    rcx: Option<RenderContext>,
}

pub struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    recreate_swapchain: bool,
    task_graph: ExecutableTaskGraph<Self>,
    scene_node_id: NodeId,
    virtual_swapchain_id: Id<Swapchain>,
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
                    ext_debug_utils: true,
                    ext_swapchain_colorspace: true,
                    ..required_extensions
                },
                enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_owned()],
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

        let resources = Resources::new(&device, &Default::default());

        let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();

        App {
            instance,
            device,
            queue,
            resources,
            flight_id,
            rcx: None,
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
        println!("Supported storage formats: {:?}", supported_storage_formats);

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let swapchain_id = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let (swapchain_format, swapchain_color_space) = supported_storage_formats
                .first()
                .map(|(format, color_space)| (*format, *color_space))
                .unwrap();
            println!("Using swapchain format: {:?}", swapchain_format);

            self.resources
                .create_swapchain(
                    self.flight_id,
                    surface,
                    SwapchainCreateInfo {
                        min_image_count: surface_capabilities.min_image_count.max(3),
                        image_format: swapchain_format,
                        image_extent: window.inner_size().into(),
                        image_usage: ImageUsage::STORAGE | ImageUsage::COLOR_ATTACHMENT,
                        image_color_space: swapchain_color_space,
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
                push_constant_ranges: vec![],
                ..Default::default()
            },
        )
        .unwrap();

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            self.device.clone(),
            Default::default(),
        ));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            self.device.clone(),
            Default::default(),
        ));

        let mut task_graph = TaskGraph::new(&self.resources, 3, 2);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());

        let scene_node_id = task_graph
            .create_task_node(
                "Scene",
                QueueFamilyType::Graphics,
                SceneTask::new(
                    self,
                    pipeline_layout.clone(),
                    swapchain_id,
                    virtual_swapchain_id,
                    descriptor_set_allocator,
                    memory_allocator,
                    command_buffer_allocator,
                ),
            )
            .image_access(
                virtual_swapchain_id.current_image_id(),
                AccessType::RayTracingShaderStorageWrite,
                ImageLayoutType::General,
            )
            .build();

        let task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.queue],
                present_queue: Some(&self.queue),
                flight_id: self.flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            virtual_swapchain_id,
            recreate_swapchain: false,
            task_graph,
            scene_node_id,
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

                let flight = self.resources.flight(self.flight_id).unwrap();

                if rcx.recreate_swapchain {
                    rcx.swapchain_id = self
                        .resources
                        .recreate_swapchain(rcx.swapchain_id, |create_info| SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..create_info
                        })
                        .expect("failed to recreate swapchain");

                    rcx.task_graph
                        .task_node_mut(rcx.scene_node_id)
                        .unwrap()
                        .task_mut()
                        .downcast_mut::<SceneTask>()
                        .unwrap()
                        .handle_resize(&self.resources, rcx.swapchain_id);

                    rcx.recreate_swapchain = false;
                }

                flight.wait(None).unwrap();

                let resource_map = resource_map!(
                    &rcx.task_graph,
                    rcx.virtual_swapchain_id => rcx.swapchain_id,
                )
                .unwrap();

                match unsafe {
                    rcx.task_graph
                        .execute(resource_map, rcx, || rcx.window.pre_present_notify())
                } {
                    Ok(()) => {}
                    Err(ExecuteError::Swapchain {
                        error: Validated::Error(VulkanError::OutOfDate),
                        ..
                    }) => {
                        rcx.recreate_swapchain = true;
                    }
                    Err(e) => {
                        panic!("failed to execute next frame: {e:?}");
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
