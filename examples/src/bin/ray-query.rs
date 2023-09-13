// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use vulkano::acceleration_structure::{
    AccelerationStructureBuildSizesInfo, AccelerationStructureBuildType,
};
use vulkano::{
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureCreateInfo,
        AccelerationStructureGeometries, AccelerationStructureGeometryInstancesData,
        AccelerationStructureGeometryInstancesDataType, AccelerationStructureGeometryTrianglesData,
        AccelerationStructureInstance, AccelerationStructureType, BuildAccelerationStructureFlags,
        BuildAccelerationStructureMode, GeometryFlags, GeometryInstanceFlags,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract, RenderingAttachmentInfo, RenderingInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::AttachmentStoreOp,
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    DeviceSize, Packed24_8, Validated, Version, VulkanError, VulkanLibrary,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    let event_loop = EventLoop::new();

    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Surface::required_extensions(&event_loop);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let mut device_extensions = DeviceExtensions {
        khr_acceleration_structure: true,
        khr_ray_query: true,
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let features = Features {
        acceleration_structure: true,
        buffer_device_address: true,
        dynamic_rendering: true,
        ray_query: true,
        ..Features::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
        })
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
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
        .expect("no suitable physical device found");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    if physical_device.api_version() < Version::V1_3 {
        device_extensions.khr_dynamic_rendering = true;
    }

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            enabled_features: features,
            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
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

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

    // The quad buffer that covers the entire surface
    let quad = [
        Vertex {
            position: [-1.0, -1.0],
        },
        Vertex {
            position: [-1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0],
        },
        Vertex {
            position: [1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0],
        },
        Vertex {
            position: [-1.0, 1.0],
        },
    ];
    let quad_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        quad,
    )
    .unwrap();

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450

                layout(location = 0) in vec2 position;
                layout(location = 0) out vec2 out_uv;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    out_uv = position;
                }
            ",
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 460
                #extension GL_EXT_ray_query : enable

                layout(location = 0) in vec2 in_uv;
                layout(location = 0) out vec4 f_color;

                layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_acceleration_structure;

                void main() {
	                float t_min = 0.01;
	                float t_max = 1000.0;
	                vec3 origin = vec3(0.0, 0.0, 0.0);
	                vec3 direction = normalize(vec3(in_uv * 1.0, 1.0));

                    rayQueryEXT ray_query;
                    rayQueryInitializeEXT(
                        ray_query,
                        top_level_acceleration_structure,
                        gl_RayFlagsTerminateOnFirstHitEXT,
                        0xFF,
                        origin,
                        t_min,
                        direction,
                        t_max
                    );
                    rayQueryProceedEXT(ray_query);

                    if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
                        // miss
                        f_color = vec4(0.0, 0.0, 0.0, 1.0);
                    } else {
                        // hit
                        f_color = vec4(1.0, 0.0, 0.0, 1.0);
                    }
                }
            ",
        }
    }

    let pipeline = {
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let vertex_input_state = Vertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(swapchain.image_format())],
            ..Default::default()
        };
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::viewport_dynamic_scissor_irrelevant()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::new(
                    subpass.color_attachment_formats.len() as u32,
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0,
    };

    let mut attachment_image_views = window_size_dependent_setup(&images, &mut viewport);
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let (top_level_acceleration_structure, bottom_level_acceleration_structure) = {
        #[derive(BufferContents, Vertex)]
        #[repr(C)]
        struct Vertex {
            #[format(R32G32B32_SFLOAT)]
            position: [f32; 3],
        }

        let vertices = [
            Vertex {
                position: [-0.5, -0.25, 1.0],
            },
            Vertex {
                position: [0.0, 0.5, 1.0],
            },
            Vertex {
                position: [0.25, -0.1, 1.0],
            },
        ];

        let vertex_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let bottom_level_acceleration_structure = create_bottom_level_acceleration_structure(
            &memory_allocator,
            &command_buffer_allocator,
            queue.clone(),
            &[&vertex_buffer],
        );
        let top_level_acceleration_structure = create_top_level_acceleration_structure(
            &memory_allocator,
            &command_buffer_allocator,
            queue.clone(),
            &[&bottom_level_acceleration_structure],
        );

        (
            top_level_acceleration_structure,
            bottom_level_acceleration_structure,
        )
    };

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        pipeline.layout().set_layouts().get(0).unwrap().clone(),
        [WriteDescriptorSet::acceleration_structure(
            0,
            top_level_acceleration_structure,
        )],
        [],
    )
    .unwrap();

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            let image_extent: [u32; 2] = window.inner_size().into();

            if image_extent.contains(&0) {
                return;
            }

            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if recreate_swapchain {
                let (new_swapchain, new_images) = swapchain
                    .recreate(SwapchainCreateInfo {
                        image_extent,
                        ..swapchain.create_info()
                    })
                    .expect("failed to recreate swapchain");

                swapchain = new_swapchain;
                attachment_image_views = window_size_dependent_setup(&new_images, &mut viewport);
                recreate_swapchain = false;
            }

            let (image_index, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let mut builder = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .begin_rendering(RenderingInfo {
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        store_op: AttachmentStoreOp::Store,
                        ..RenderingAttachmentInfo::image_view(
                            attachment_image_views[image_index as usize].clone(),
                        )
                    })],
                    ..Default::default()
                })
                .unwrap()
                .set_viewport(0, [viewport.clone()].into_iter().collect())
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .unwrap()
                .bind_vertex_buffers(0, quad_buffer.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    descriptor_set.clone(),
                )
                .unwrap()
                .draw(quad_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_rendering()
                .unwrap();
            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                )
                .then_signal_fence_and_flush();

            match future.map_err(Validated::unwrap) {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(VulkanError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => (),
    });
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    viewport: &mut Viewport,
) -> Vec<Arc<ImageView>> {
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>()
}

fn create_top_level_acceleration_structure(
    memory_allocator: &impl MemoryAllocator,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    bottom_level_acceleration_structures: &[&AccelerationStructure],
) -> Arc<AccelerationStructure> {
    let instances = bottom_level_acceleration_structures
        .iter()
        .map(
            |&bottom_level_acceleration_structure| AccelerationStructureInstance {
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                    0,
                    GeometryInstanceFlags::TRIANGLE_FACING_CULL_DISABLE.into(),
                ),
                acceleration_structure_reference: bottom_level_acceleration_structure
                    .device_address()
                    .get(),
                ..Default::default()
            },
        )
        .collect::<Vec<_>>();

    let values = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        instances,
    )
    .unwrap();

    let geometries =
        AccelerationStructureGeometries::Instances(AccelerationStructureGeometryInstancesData {
            flags: GeometryFlags::OPAQUE,
            ..AccelerationStructureGeometryInstancesData::new(
                AccelerationStructureGeometryInstancesDataType::Values(Some(values)),
            )
        });

    let build_info = AccelerationStructureBuildGeometryInfo {
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        mode: BuildAccelerationStructureMode::Build,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let build_range_infos = [AccelerationStructureBuildRangeInfo {
        primitive_count: bottom_level_acceleration_structures.len() as _,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
    }];

    build_acceleration_structure(
        memory_allocator,
        command_buffer_allocator,
        queue,
        AccelerationStructureType::TopLevel,
        build_info,
        &[bottom_level_acceleration_structures.len() as u32],
        build_range_infos,
    )
}

fn create_bottom_level_acceleration_structure<T: BufferContents + Vertex>(
    memory_allocator: &impl MemoryAllocator,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    vertex_buffers: &[&Subbuffer<[T]>],
) -> Arc<AccelerationStructure> {
    let description = T::per_vertex();

    assert_eq!(description.stride, std::mem::size_of::<T>() as u32);

    let mut triangles = vec![];
    let mut max_primitive_counts = vec![];
    let mut build_range_infos = vec![];

    for &vertex_buffer in vertex_buffers {
        let primitive_count = vertex_buffer.len() as u32 / 3;
        triangles.push(AccelerationStructureGeometryTrianglesData {
            flags: GeometryFlags::OPAQUE,
            vertex_data: Some(vertex_buffer.clone().into_bytes()),
            vertex_stride: description.stride,
            max_vertex: vertex_buffer.len() as _,
            index_data: None,
            transform_data: None,
            ..AccelerationStructureGeometryTrianglesData::new(
                description.members.get("position").unwrap().format,
            )
        });
        max_primitive_counts.push(primitive_count);
        build_range_infos.push(AccelerationStructureBuildRangeInfo {
            primitive_count,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        })
    }

    let geometries = AccelerationStructureGeometries::Triangles(triangles);
    let build_info = AccelerationStructureBuildGeometryInfo {
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        mode: BuildAccelerationStructureMode::Build,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    build_acceleration_structure(
        memory_allocator,
        command_buffer_allocator,
        queue,
        AccelerationStructureType::BottomLevel,
        build_info,
        &max_primitive_counts,
        build_range_infos,
    )
}

fn create_acceleration_structure(
    memory_allocator: &impl MemoryAllocator,
    ty: AccelerationStructureType,
    size: DeviceSize,
) -> Arc<AccelerationStructure> {
    let buffer = Buffer::new_slice::<u8>(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size,
    )
    .unwrap();

    unsafe {
        AccelerationStructure::new(
            memory_allocator.device().clone(),
            AccelerationStructureCreateInfo {
                ty,
                ..AccelerationStructureCreateInfo::new(buffer)
            },
        )
        .unwrap()
    }
}

fn create_scratch_buffer(
    memory_allocator: &impl MemoryAllocator,
    size: DeviceSize,
) -> Subbuffer<[u8]> {
    Buffer::new_slice::<u8>(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size,
    )
    .unwrap()
}

fn build_acceleration_structure(
    memory_allocator: &impl MemoryAllocator,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    ty: AccelerationStructureType,
    mut build_info: AccelerationStructureBuildGeometryInfo,
    max_primitive_counts: &[u32],
    build_range_infos: impl IntoIterator<Item = AccelerationStructureBuildRangeInfo>,
) -> Arc<AccelerationStructure> {
    let device = memory_allocator.device();

    let AccelerationStructureBuildSizesInfo {
        acceleration_structure_size,
        build_scratch_size,
        ..
    } = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &build_info,
            max_primitive_counts,
        )
        .unwrap();

    let acceleration_structure =
        create_acceleration_structure(memory_allocator, ty, acceleration_structure_size);
    let scratch_buffer = create_scratch_buffer(memory_allocator, build_scratch_size);

    build_info.dst_acceleration_structure = Some(acceleration_structure.clone());
    build_info.scratch_data = Some(scratch_buffer);

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    unsafe {
        builder
            .build_acceleration_structure(build_info, build_range_infos.into_iter().collect())
            .unwrap();
    }

    let command_buffer = builder.build().unwrap();
    command_buffer
        .execute(queue)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    acceleration_structure
}
