// Welcome to the triangle-util example!
//
// This is almost exactly the same as the triangle example, except that it uses utility functions
// to make life easier.
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.

use glam::{Mat4, Vec3};
use std::{
    error::Error,
    mem::size_of,
    sync::Arc,
    time::{Duration, Instant},
};
use vulkano::{
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureCreateInfo, AccelerationStructureGeometries,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureFlags, BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferUsage, RecordingCommandBuffer,
    },
    descriptor_set::{
        allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator},
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, DeviceExtensions, DeviceFeatures, DeviceOwnedVulkanObject},
    format::Format,
    image::{view::ImageView, ImageUsage},
    instance::{InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::{
        graphics::{vertex_input::Vertex, viewport::Viewport},
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
        Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::ShaderStages,
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::VulkanoWindows,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

mod raygen {
    vulkano_shaders::shader! {
        ty: "raygen",
        path: "raytrace.rgen",
        vulkan_version: "1.2"
    }
}

mod closest_hit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        path: "raytrace.rchit",
        vulkan_version: "1.2"
    }
}

mod miss {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "raytrace.rmiss",
        vulkan_version: "1.2"
    }
}

fn main() -> Result<(), impl Error> {
    let context = VulkanoContext::new(VulkanoConfig {
        device_extensions: DeviceExtensions {
            khr_swapchain: true,
            khr_acceleration_structure: true,
            khr_ray_tracing_pipeline: true,
            khr_deferred_host_operations: true,
            ..Default::default()
        },
        device_features: DeviceFeatures {
            acceleration_structure: true,
            ray_tracing_pipeline: true,
            buffer_device_address: true,
            synchronization2: true,
            ..Default::default()
        },
        instance_create_info: InstanceCreateInfo {
            enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_owned()],
            enabled_extensions: InstanceExtensions {
                ext_debug_utils: true,
                ..InstanceExtensions::empty()
            },
            ..Default::default()
        },
        ..Default::default()
    });
    let event_loop = EventLoop::new().unwrap();
    // Manages any windows and their rendering.
    let mut windows_manager = VulkanoWindows::default();
    windows_manager.create_window(
        &event_loop,
        &context,
        &Default::default(),
        |swapchain_create_info| {
            swapchain_create_info.image_usage |= ImageUsage::STORAGE;
        },
    );
    let window_renderer = windows_manager.get_primary_renderer_mut().unwrap();

    // Some little debug infos.
    println!(
        "Using device: {} (type: {:?})",
        context.device().physical_device().properties().device_name,
        context.device().physical_device().properties().device_type,
    );

    // We now create a buffer that will store the shape of our triangle. We use `#[repr(C)]` here
    // to force rustc to use a defined layout for our data, as the default representation has *no
    // guarantees*.

    let vertices = [
        MyVertex {
            position: [-0.5, -0.25, 0.0],
        },
        MyVertex {
            position: [0.0, 0.5, 0.0],
        },
        MyVertex {
            position: [0.25, -0.1, 0.0],
        },
    ];
    let vertex_buffer = Buffer::from_iter(
        context.memory_allocator().clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
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

    let uniform_buffer = Buffer::from_data(
        context.memory_allocator().clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        raygen::Camera {
            projInverse: Default::default(),
            viewInverse: Default::default(),
            viewProj: Default::default(),
        },
    )
    .unwrap();

    // Before we can start creating and recording command buffers, we need a way of allocating
    // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command pools
    // underneath and provides a safe interface for them.
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        context.device().clone(),
        Default::default(),
    ));

    let tlas = unsafe {
        let mut builder = RecordingCommandBuffer::new(
            command_buffer_allocator.clone(),
            context.graphics_queue().queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .unwrap();

        let blas = build_acceleration_structure_triangles(
            vertex_buffer.clone(),
            context.memory_allocator().clone(),
            context.device().clone(),
            &mut builder,
        );
        blas.set_debug_utils_object_name("Triangle BLAS".into())
            .unwrap();
        let tlas = build_top_level_acceleration_structure(
            blas,
            context.memory_allocator().clone(),
            context.device().clone(),
            &mut builder,
        );
        tlas.set_debug_utils_object_name("Triangle TLAS".into())
            .unwrap();

        builder
            .end()
            .unwrap()
            .execute(context.graphics_queue().clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        tlas
    };

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        context.device().clone(),
        Default::default(),
    ));

    let descriptor_set_layout_0 = DescriptorSetLayout::new(
        context.device().clone(),
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
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                    },
                ),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap();

    let descriptor_set_0 = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        descriptor_set_layout_0,
        [
            WriteDescriptorSet::acceleration_structure(0, tlas),
            WriteDescriptorSet::buffer(1, uniform_buffer.clone()),
        ],
        [],
    )
    .unwrap();
    descriptor_set_0
        .set_debug_utils_object_name("Descriptor Set 0".into())
        .unwrap();

    // Before we draw, we have to create what is called a **pipeline**. A pipeline describes how
    // a GPU operation is to be performed. It is similar to an OpenGL program, but it also contains
    // many settings for customization, all baked into a single object. For drawing, we create
    // a **graphics** pipeline, but there are also other types of pipeline.
    let pipeline = {
        // First, we load the shaders that the pipeline will use:
        // the vertex shader and the fragment shader.
        //
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify which
        // one.
        let raygen = raygen::load(context.device().clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let closest_hit = closest_hit::load(context.device().clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let miss = miss::load(context.device().clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        // Make a list of the shader stages that the pipeline will have.
        let stages = [
            PipelineShaderStageCreateInfo::new(raygen),
            PipelineShaderStageCreateInfo::new(miss),
            PipelineShaderStageCreateInfo::new(closest_hit),
        ];

        // We must now create a **pipeline layout** object, which describes the locations and types
        // of descriptor sets and push constants used by the shaders in the pipeline.
        //
        // Multiple pipelines can share a common layout object, which is more efficient.
        // The shaders in a pipeline must use a subset of the resources described in its pipeline
        // layout, but the pipeline layout is allowed to contain resources that are not present in
        // the shaders; they can be used by shaders in other pipelines that share the same
        // layout. Thus, it is a good idea to design shaders so that many pipelines have
        // common resource locations, which allows them to share pipeline layouts.
        let layout = PipelineLayout::new(
            context.device().clone(),
            // Since we only have one pipeline in this example, and thus one pipeline layout,
            // we automatically generate the creation info for it from the resources used in the
            // shaders. In a real application, you would specify this information manually so that
            // you can re-use one layout in multiple pipelines.
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(context.device().clone())
                .unwrap(),
        )
        .unwrap();

        let groups = [
            RayTracingShaderGroupCreateInfo {
                // Raygen
                general_shader: Some(0),
                ..Default::default()
            },
            RayTracingShaderGroupCreateInfo {
                // Miss
                general_shader: Some(1),
                ..Default::default()
            },
            RayTracingShaderGroupCreateInfo {
                // Closest Hit
                group_type: ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
                closest_hit_shader: Some(2),
                ..Default::default()
            },
        ];

        RayTracingPipeline::new(
            context.device().clone(),
            None,
            RayTracingPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                groups: groups.into_iter().collect(),
                max_pipeline_ray_recursion_depth: 1,

                ..RayTracingPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };
    pipeline
        .set_debug_utils_object_name("Ray Tracing Pipeline".into())
        .unwrap();

    let shader_binding_table =
        ShaderBindingTable::new(context.memory_allocator().clone(), &pipeline, 1, 1, 0).unwrap();

    // Dynamic viewports allow us to recreate just the viewport when the window is resized.
    // Otherwise we would have to recreate the whole pipeline.
    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0,
    };

    // The render pass we created above only describes the layout of our framebuffers. Before we
    // can draw we also need to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.

    // Initialization is finally finished!

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.

    let rotation_start = Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                elwt.exit();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                window_renderer.resize();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let window_size_dependent_items = window_size_dependent_setup(
                    window_renderer.swapchain_image_views()[window_renderer.image_index() as usize]
                        .clone(),
                    descriptor_set_allocator.clone(),
                    context.device().clone(),
                );

                // Do not draw the frame when the screen size is zero. On Windows, this can
                // occur when minimizing the application.
                if window_size_dependent_items.extent.contains(&0) {
                    return;
                }

                let elapsed = rotation_start.elapsed();

                // NOTE: This teapot was meant for OpenGL where the origin is at the lower left
                // instead the origin is at the upper left in Vulkan, so we reverse the Y axis.
                let aspect_ratio = window_size_dependent_items.extent[0] as f32
                    / window_size_dependent_items.extent[1] as f32;

                let proj =
                    Mat4::perspective_rh_gl(std::f32::consts::FRAC_PI_2, aspect_ratio, 0.01, 100.0);
                let view = Mat4::look_at_rh(
                    Vec3::new(0.3, 0.3, 1.0),
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, -1.0, 0.0),
                );

                let uniform_data = raygen::Camera {
                    projInverse: proj.inverse().to_cols_array_2d(),
                    viewInverse: view.inverse().to_cols_array_2d(),
                    viewProj: (proj * view).to_cols_array_2d(),
                };

                *uniform_buffer.write().unwrap() = uniform_data;

                // Begin rendering by acquiring the gpu future from the window renderer.
                let previous_frame_end = window_renderer
                    .acquire(Some(Duration::from_millis(1)), |_| {})
                    .unwrap();

                // In order to draw, we have to record a *command buffer*. The command buffer object
                // holds the list of commands that are going to be executed.
                //
                // Recording a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to
                // be optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The
                // command buffer will only be executable on that given queue family.
                let mut builder = RecordingCommandBuffer::new(
                    command_buffer_allocator.clone(),
                    context.graphics_queue().queue_family_index(),
                    CommandBufferLevel::Primary,
                    CommandBufferBeginInfo {
                        usage: CommandBufferUsage::OneTimeSubmit,
                        ..Default::default()
                    },
                )
                .unwrap();

                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::RayTracing,
                        pipeline.layout().clone(),
                        0,
                        vec![
                            descriptor_set_0.clone(),
                            window_size_dependent_items.image_descriptor_set,
                        ],
                    )
                    .unwrap()
                    .bind_pipeline_ray_tracing(pipeline.clone())
                    .unwrap();

                unsafe {
                    builder
                        .trace_rays(
                            shader_binding_table.clone(),
                            window_size_dependent_items.extent[0],
                            window_size_dependent_items.extent[1],
                            1,
                        )
                        .unwrap();
                }

                // Finish recording the command buffer by calling `end`.
                let command_buffer = builder.end().unwrap();

                let future = previous_frame_end
                    .then_execute(context.graphics_queue().clone(), command_buffer)
                    .unwrap()
                    .boxed();

                // The color output is now expected to contain our triangle. But in order to
                // show it on the screen, we have to *present* the image by calling
                // `present` on the window renderer.
                //
                // This function does not actually present the image immediately. Instead it
                // submits a present command at the end of the queue. This means that it will
                // only be presented once the GPU has finished executing the command buffer
                // that draws the triangle.
                window_renderer.present(future, false);

                panic!("Done");
            }
            Event::AboutToWait => window_renderer.window().request_redraw(),
            _ => (),
        }
    })
}

struct WindowSizeDependentItems {
    extent: [u32; 3],
    image_descriptor_set: Arc<DescriptorSet>,
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    target_image_view: Arc<ImageView>,
    descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
    device: Arc<Device>,
) -> WindowSizeDependentItems {
    let extent = target_image_view.image().extent();

    let descriptor_set_layout = DescriptorSetLayout::new(
        device,
        DescriptorSetLayoutCreateInfo {
            bindings: [(
                0,
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::RAYGEN,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
                },
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap();

    let image_descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::image_view(0, target_image_view.clone())],
        [],
    )
    .unwrap();
    image_descriptor_set
        .set_debug_utils_object_name("Image Descriptor Set".into())
        .unwrap();

    WindowSizeDependentItems {
        extent,
        image_descriptor_set,
    }
}

unsafe fn build_acceleration_structure_triangles(
    vertex_buffer: Subbuffer<[MyVertex]>,
    allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
    command_buffer: &mut RecordingCommandBuffer,
) -> Arc<AccelerationStructure> {
    let primitive_count = (vertex_buffer.len() / 3) as u32;
    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        // TODO: Modify constructor?
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.into_bytes()),
        vertex_stride: size_of::<MyVertex>() as _,
        ..AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT)
    };

    let as_geometries =
        AccelerationStructureGeometries::Triangles(vec![as_geometry_triangles_data]);

    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Build,
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        ..AccelerationStructureBuildGeometryInfo::new(as_geometries)
    };

    let as_build_sizes_info = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &as_build_geometry_info,
            &[primitive_count],
        )
        .unwrap();

    let scratch_buffer = Buffer::new_slice::<u8>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
        as_build_sizes_info.build_scratch_size,
    )
    .unwrap();

    let as_create_info = AccelerationStructureCreateInfo {
        ty: AccelerationStructureType::BottomLevel,
        ..AccelerationStructureCreateInfo::new(
            Buffer::new_slice::<u8>(
                allocator,
                BufferCreateInfo {
                    usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                        | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
                as_build_sizes_info.acceleration_structure_size,
            )
            .unwrap(),
        )
    };
    let acceleration = unsafe { AccelerationStructure::new(device, as_create_info).unwrap() };

    as_build_geometry_info.dst_acceleration_structure = Some(acceleration.clone());
    as_build_geometry_info.scratch_data = Some(scratch_buffer);

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        ..Default::default()
    };

    command_buffer
        .build_acceleration_structure(
            as_build_geometry_info,
            Some(as_build_range_info).into_iter().collect(),
        )
        .unwrap();

    acceleration
}

unsafe fn build_top_level_acceleration_structure(
    acceleration_structure: Arc<AccelerationStructure>,
    allocator: Arc<dyn MemoryAllocator>,
    device: Arc<Device>,
    command_buffer: &mut RecordingCommandBuffer,
) -> Arc<AccelerationStructure> {
    let primitive_count = 1;
    let as_instance = AccelerationStructureInstance {
        acceleration_structure_reference: acceleration_structure.device_address().into(), // TODO: Need to hold AS
        ..Default::default()
    };

    let instance_buffer = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        [as_instance],
    )
    .unwrap();

    let as_geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
        AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer)),
    );

    let as_geometries = AccelerationStructureGeometries::Instances(as_geometry_instances_data);

    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Build,
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        ..AccelerationStructureBuildGeometryInfo::new(as_geometries)
    };

    let as_build_sizes_info = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &as_build_geometry_info,
            &[primitive_count],
        )
        .unwrap();

    let scratch_buffer = Buffer::new_slice::<u8>(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
        as_build_sizes_info.build_scratch_size,
    )
    .unwrap();

    let as_create_info = AccelerationStructureCreateInfo {
        ty: AccelerationStructureType::TopLevel,
        ..AccelerationStructureCreateInfo::new(
            Buffer::new_slice::<u8>(
                allocator,
                BufferCreateInfo {
                    usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                        | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
                as_build_sizes_info.acceleration_structure_size,
            )
            .unwrap(),
        )
    };
    let acceleration = unsafe { AccelerationStructure::new(device, as_create_info).unwrap() };

    as_build_geometry_info.dst_acceleration_structure = Some(acceleration.clone());
    as_build_geometry_info.scratch_data = Some(scratch_buffer);

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        ..Default::default()
    };

    command_buffer
        .build_acceleration_structure(
            as_build_geometry_info,
            Some(as_build_range_info).into_iter().collect(),
        )
        .unwrap();

    acceleration
}
