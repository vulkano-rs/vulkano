// TODO: document

use bloom::BloomTask;
use scene::SceneTask;
use std::{cmp, error::Error, sync::Arc};
use tonemap::TonemapTask;
use vulkano::{
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        DescriptorImageViewInfo, DescriptorSet, DescriptorSetWithOffsets, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned,
        Queue, QueueCreateInfo, QueueFlags,
    },
    format::{Format, NumericFormat},
    image::{
        max_mip_levels,
        sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode, LOD_CLAMP_NONE},
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageSubresourceRange,
        ImageType, ImageUsage,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::AllocationCreateInfo,
    pipeline::{
        graphics::viewport::Viewport,
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        PipelineLayout,
    },
    shader::ShaderStages,
    swapchain::{ColorSpace, Surface, Swapchain, SwapchainCreateInfo},
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

mod bloom;
mod scene;
mod tonemap;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const MAX_BLOOM_MIP_LEVELS: u32 = 6;

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
    bloom_image_id: Id<Image>,
    viewport: Viewport,
    pipeline_layout: Arc<PipelineLayout>,
    recreate_swapchain: bool,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    sampler: Arc<Sampler>,
    descriptor_set: DescriptorSetWithOffsets,
    task_graph: ExecutableTaskGraph<Self>,
    scene_node_id: NodeId,
    tonemap_node_id: NodeId,
    virtual_swapchain_id: Id<Swapchain>,
    virtual_bloom_image_id: Id<Image>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.api_version() >= Version::V1_1 || p.supported_extensions().khr_maintenance2
            })
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

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_1 {
            device_extensions.khr_maintenance2 = true;
        }

        if physical_device.api_version() < Version::V1_2
            && physical_device.supported_extensions().khr_image_format_list
        {
            device_extensions.khr_image_format_list = true;
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
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
        let window_size = window.inner_size();

        let swapchain_format;
        let swapchain_id = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            (swapchain_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()
                .into_iter()
                .find(|&(format, color_space)| {
                    format.numeric_format_color() == Some(NumericFormat::SRGB)
                        && color_space == ColorSpace::SrgbNonLinear
                })
                .unwrap();

            self.resources
                .create_swapchain(
                    self.flight_id,
                    surface,
                    SwapchainCreateInfo {
                        min_image_count: surface_capabilities.min_image_count.max(3),
                        image_format: swapchain_format,
                        image_extent: window.inner_size().into(),
                        image_usage: ImageUsage::COLOR_ATTACHMENT,
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

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let pipeline_layout = PipelineLayout::new(
            self.device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![DescriptorSetLayout::new(
                    self.device.clone(),
                    DescriptorSetLayoutCreateInfo {
                        bindings: [
                            (
                                0,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::Sampler,
                                    )
                                },
                            ),
                            (
                                1,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::SampledImage,
                                    )
                                },
                            ),
                            (
                                2,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::COMPUTE,
                                    descriptor_count: MAX_BLOOM_MIP_LEVELS,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::StorageImage,
                                    )
                                },
                            ),
                        ]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                )
                .unwrap()],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    offset: 0,
                    size: 12,
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            self.device.clone(),
            Default::default(),
        ));

        let sampler = Sampler::new(
            self.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Nearest,
                lod: 0.0..=LOD_CLAMP_NONE,
                ..Default::default()
            },
        )
        .unwrap();

        let (bloom_image_id, descriptor_set) = window_size_dependent_setup(
            &self.resources,
            swapchain_id,
            &pipeline_layout,
            &sampler,
            &descriptor_set_allocator,
        );

        let mut task_graph = TaskGraph::new(&self.resources, 3, 2);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());
        let virtual_bloom_image_id = task_graph.add_image(&ImageCreateInfo::default());

        let scene_node_id = task_graph
            .create_task_node(
                "Scene",
                QueueFamilyType::Graphics,
                SceneTask::new(self, &pipeline_layout, bloom_image_id),
            )
            .image_access(
                virtual_bloom_image_id,
                AccessType::ColorAttachmentWrite,
                ImageLayoutType::Optimal,
            )
            .build();
        let bloom_node_id = task_graph
            .create_task_node(
                "Bloom",
                QueueFamilyType::Compute,
                BloomTask::new(self, &pipeline_layout, virtual_bloom_image_id),
            )
            .image_access(
                virtual_bloom_image_id,
                AccessType::ComputeShaderSampledRead,
                ImageLayoutType::General,
            )
            .image_access(
                virtual_bloom_image_id,
                AccessType::ComputeShaderStorageWrite,
                ImageLayoutType::General,
            )
            .build();
        let tonemap_node_id = task_graph
            .create_task_node(
                "Tonemap",
                QueueFamilyType::Graphics,
                TonemapTask::new(self, &pipeline_layout, swapchain_id, virtual_swapchain_id),
            )
            .image_access(
                virtual_swapchain_id.current_image_id(),
                AccessType::ColorAttachmentWrite,
                ImageLayoutType::Optimal,
            )
            .image_access(
                virtual_bloom_image_id,
                AccessType::FragmentShaderSampledRead,
                ImageLayoutType::General,
            )
            .build();

        task_graph.add_edge(scene_node_id, bloom_node_id).unwrap();
        task_graph.add_edge(bloom_node_id, tonemap_node_id).unwrap();

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
            bloom_image_id,
            viewport,
            pipeline_layout,
            recreate_swapchain: false,
            sampler,
            descriptor_set_allocator,
            descriptor_set,
            task_graph,
            scene_node_id,
            tonemap_node_id,
            virtual_swapchain_id,
            virtual_bloom_image_id,
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

                    rcx.viewport.extent = window_size.into();

                    unsafe { self.resources.remove_image(rcx.bloom_image_id) }.unwrap();

                    (rcx.bloom_image_id, rcx.descriptor_set) = window_size_dependent_setup(
                        &self.resources,
                        rcx.swapchain_id,
                        &rcx.pipeline_layout,
                        &rcx.sampler,
                        &rcx.descriptor_set_allocator,
                    );

                    rcx.task_graph
                        .task_node_mut(rcx.scene_node_id)
                        .unwrap()
                        .task_mut()
                        .downcast_mut::<SceneTask>()
                        .unwrap()
                        .handle_resize(&self.resources, rcx.bloom_image_id);
                    rcx.task_graph
                        .task_node_mut(rcx.tonemap_node_id)
                        .unwrap()
                        .task_mut()
                        .downcast_mut::<TonemapTask>()
                        .unwrap()
                        .handle_resize(&self.resources, rcx.swapchain_id);

                    rcx.recreate_swapchain = false;
                }

                flight.wait(None).unwrap();

                let resource_map = resource_map!(
                    &rcx.task_graph,
                    rcx.virtual_swapchain_id => rcx.swapchain_id,
                    rcx.virtual_bloom_image_id => rcx.bloom_image_id,
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

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    resources: &Resources,
    swapchain_id: Id<Swapchain>,
    pipeline_layout: &Arc<PipelineLayout>,
    sampler: &Arc<Sampler>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
) -> (Id<Image>, DescriptorSetWithOffsets) {
    let device = resources.device();
    let swapchain_state = resources.swapchain(swapchain_id).unwrap();
    let images = swapchain_state.images();
    let extent = images[0].extent();

    let bloom_image_mip_levels = cmp::min(MAX_BLOOM_MIP_LEVELS, max_mip_levels(extent));

    let bloom_image_id = {
        let view_formats = if device.api_version() >= Version::V1_2
            || device.enabled_extensions().khr_image_format_list
        {
            vec![Format::R32_UINT, Format::E5B9G9R9_UFLOAT_PACK32]
        } else {
            Vec::new()
        };

        resources
            .create_image(
                ImageCreateInfo {
                    flags: ImageCreateFlags::MUTABLE_FORMAT,
                    image_type: ImageType::Dim2d,
                    format: Format::R32_UINT,
                    view_formats,
                    extent,
                    mip_levels: bloom_image_mip_levels,
                    usage: ImageUsage::TRANSFER_DST
                        | ImageUsage::SAMPLED
                        | ImageUsage::STORAGE
                        | ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap()
    };

    let bloom_image_state = resources.image(bloom_image_id).unwrap();
    let bloom_image = bloom_image_state.image();

    let bloom_texture_view = ImageView::new(
        bloom_image.clone(),
        ImageViewCreateInfo {
            format: Format::E5B9G9R9_UFLOAT_PACK32,
            subresource_range: bloom_image.subresource_range(),
            usage: ImageUsage::SAMPLED,
            ..Default::default()
        },
    )
    .unwrap();

    let bloom_mip_chain_views = (0..MAX_BLOOM_MIP_LEVELS).map(|mip_level| {
        let mip_level = cmp::min(mip_level, max_mip_levels(extent) - 1);

        ImageView::new(
            bloom_image.clone(),
            ImageViewCreateInfo {
                format: Format::R32_UINT,
                subresource_range: ImageSubresourceRange {
                    aspects: ImageAspects::COLOR,
                    mip_levels: mip_level..mip_level + 1,
                    array_layers: 0..1,
                },
                usage: ImageUsage::STORAGE,
                ..Default::default()
            },
        )
        .unwrap()
    });

    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        pipeline_layout.set_layouts()[0].clone(),
        [
            WriteDescriptorSet::sampler(0, sampler.clone()),
            WriteDescriptorSet::image_view_with_layout(
                1,
                DescriptorImageViewInfo {
                    image_view: bloom_texture_view,
                    image_layout: ImageLayout::General,
                },
            ),
            WriteDescriptorSet::image_view_with_layout_array(
                2,
                0,
                bloom_mip_chain_views.map(|image_view| DescriptorImageViewInfo {
                    image_view,
                    image_layout: ImageLayout::General,
                }),
            ),
        ],
        [],
    )
    .unwrap();

    (bloom_image_id, descriptor_set.into())
}
