// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// A minimal particle-sandbox to demonstrate a reasonable use-case for a device-local buffer. We
// gain significant runtime performance by writing the inital vertex values to the GPU using a
// staging buffer and then copying the data to a device-local buffer to be accessed solely by the
// GPU through the compute shader and as a vertex array.

use std::{error::Error, sync::Arc, time::SystemTime};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    image::{view::ImageView, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo,
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    swapchain::{
        acquire_next_image, PresentMode, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainPresentInfo,
    },
    sync::{self, future::FenceSignalFuture, GpuFuture},
    Validated, VulkanLibrary,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

const PARTICLE_COUNT: usize = 100_000;

fn main() -> Result<(), impl Error> {
    // The usual Vulkan initialization. Largely the same as example `triangle.rs` until further
    // commentation is provided.
    let event_loop = EventLoop::new().unwrap();

    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Surface::required_extensions(&event_loop).unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    let window = Arc::new(
        WindowBuilder::new()
            // For simplicity, we are going to assert that the window size is static.
            .with_resizable(false)
            .with_title("simple particles")
            .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(&event_loop)
            .unwrap(),
    );
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
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
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

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

    let (swapchain, images) = {
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
                image_extent: [WINDOW_WIDTH, WINDOW_HEIGHT],
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                present_mode: PresentMode::Fifo,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap();

    let framebuffers: Vec<Arc<Framebuffer>> = images
        .into_iter()
        .map(|img| {
            let view = ImageView::new_default(img).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect();

    // Compute shader for updating the position and velocity of each particle every frame.
    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: r"
                #version 450

                layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

                struct VertexData {
                    vec2 pos;
                    vec2 vel;
                };

                // Storage buffer binding, which we optimize by using a DeviceLocalBuffer.
                layout (binding = 0) buffer VertexBuffer {
                    VertexData verticies[];
                };

                // Allow push constants to define a parameters of compute.
                layout (push_constant) uniform PushConstants {
                    vec2 attractor;
                    float attractor_strength;
                    float delta_time;
                } push;

                // Keep this value in sync with the `maxSpeed` const in the vertex shader.
                const float maxSpeed = 10.0; 

                const float minLength = 0.02;
                const float friction = -2.0;

                void main() {
                    const uint index = gl_GlobalInvocationID.x;

                    vec2 vel = verticies[index].vel;

                    // Update particle position according to velocity.
                    vec2 pos = verticies[index].pos + push.delta_time * vel;

                    // Bounce particle off screen-border.
                    if (abs(pos.x) > 1.0) {
                        vel.x = sign(pos.x) * (-0.95 * abs(vel.x) - 0.0001);
                        if (abs(pos.x) >= 1.05) {
                            pos.x = sign(pos.x);
                        }
                    }
                    if (abs(pos.y) > 1.0) {
                        vel.y = sign(pos.y) * (-0.95 * abs(vel.y) - 0.0001);
                        if (abs(pos.y) >= 1.05) {
                            pos.y = sign(pos.y);
                        }
                    }

                    // Simple inverse-square force.
                    vec2 t = push.attractor - pos;
                    float r = max(length(t), minLength);
                    vec2 force = push.attractor_strength * (t / r) / (r * r);

                    // Update velocity, enforcing a maximum speed.
                    vel += push.delta_time * force;
                    if (length(vel) > maxSpeed) {
                        vel = maxSpeed*normalize(vel);
                    }

                    // Set new values back into buffer.
                    verticies[index].pos = pos;
	                verticies[index].vel = vel * exp(friction * push.delta_time);
                }
            ",
        }
    }

    // The vertex shader determines color and is run once per particle. The vertices will be
    // updated by the compute shader each frame.
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450

                layout(location = 0) in vec2 pos;
                layout(location = 1) in vec2 vel;

                layout(location = 0) out vec4 outColor;

                // Keep this value in sync with the `maxSpeed` const in the compute shader.
                const float maxSpeed = 10.0; 

                void main() {
                    gl_Position = vec4(pos, 0.0, 1.0);
	                gl_PointSize = 1.0;

                    // Mix colors based on position and velocity.
                    outColor = mix(
                        0.2 * vec4(pos, abs(vel.x) + abs(vel.y), 1.0),
                        vec4(1.0, 0.5, 0.8, 1.0),
                        sqrt(length(vel) / maxSpeed)
                    );
                }
            ",
        }
    }

    // The fragment shader will only need to apply the color forwarded by the vertex shader,
    // because the color of a particle should be identical over all pixels.
    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450

                layout(location = 0) in vec4 outColor;

                layout(location = 0) out vec4 fragColor;

                void main() {
                    fragColor = outColor;
                }
            ",
        }
    }

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        pos: [f32; 2],
        #[format(R32G32_SFLOAT)]
        vel: [f32; 2],
    }

    // Apply scoped logic to create `DeviceLocalBuffer` initialized with vertex data.
    let vertex_buffer = {
        // Initialize vertex data as an iterator.
        let vertices = (0..PARTICLE_COUNT).map(|i| {
            let f = i as f32 / (PARTICLE_COUNT / 10) as f32;
            Vertex {
                pos: [2. * f.fract() - 1., 0.2 * f.floor() - 1.],
                vel: [0.; 2],
            }
        });

        // Create a CPU-accessible buffer initialized with the vertex data.
        let temporary_accessible_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                // Specify this buffer will be used as a transfer source.
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                // Specify this buffer will be used for uploading to the GPU.
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        // Create a buffer in device-local memory with enough space for `PARTICLE_COUNT` number of
        // `Vertex`.
        let device_local_buffer = Buffer::new_slice::<Vertex>(
            memory_allocator,
            BufferCreateInfo {
                // Specify use as a storage buffer, vertex buffer, and transfer destination.
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                // Specify this buffer will only be used by the device.
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            PARTICLE_COUNT as vulkano::DeviceSize,
        )
        .unwrap();

        // Create one-time command to copy between the buffers.
        let mut cbb = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        cbb.copy_buffer(CopyBufferInfo::buffers(
            temporary_accessible_buffer,
            device_local_buffer.clone(),
        ))
        .unwrap();
        let cb = cbb.build().unwrap();

        // Execute copy and wait for copy to complete before proceeding.
        cb.execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        device_local_buffer
    };

    // Create a compute-pipeline for applying the compute shader to vertices.
    let compute_pipeline = {
        let cs = cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .expect("failed to create descriptor set layouts"),
        )
        .expect("failed to create pipeline layout");
        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute shader")
    };

    // Create a new descriptor set for binding vertices as a storage buffer.
    use vulkano::pipeline::Pipeline; // Required to access the `layout` method of pipeline.
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        compute_pipeline
            .layout()
            .set_layouts()
            // 0 is the index of the descriptor set.
            .get(0)
            .unwrap()
            .clone(),
        [
            // 0 is the binding of the data in this set. We bind the `Buffer` of vertices here.
            WriteDescriptorSet::buffer(0, vertex_buffer.clone()),
        ],
        [],
    )
    .unwrap();

    // Create a basic graphics pipeline for rendering particles.
    let graphics_pipeline = {
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
        let subpass = Subpass::from(render_pass, 0).unwrap();
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                // Vertices will be rendered as a list of points.
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::PointList,
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState {
                    viewports: [Viewport {
                        offset: [0.0, 0.0],
                        extent: [WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let mut fences: Vec<Option<FenceSignalFuture<_>>> =
        (0..framebuffers.len()).map(|_| None).collect();
    let mut previous_fence_index = 0u32;

    let start_time = SystemTime::now();
    let mut last_frame_time = start_time;
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
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let image_extent: [u32; 2] = window.inner_size().into();

                if image_extent.contains(&0) {
                    return;
                }

                // Update per-frame variables.
                let now = SystemTime::now();
                let time = now.duration_since(start_time).unwrap().as_secs_f32();
                let delta_time = now.duration_since(last_frame_time).unwrap().as_secs_f32();
                last_frame_time = now;

                // Create push contants to be passed to compute shader.
                let push_constants = cs::PushConstants {
                    attractor: [0.75 * (3. * time).cos(), 0.6 * (0.75 * time).sin()],
                    attractor_strength: 1.2 * (2. * time).cos(),
                    delta_time,
                };

                // Acquire information on the next swapchain target.
                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    swapchain.clone(),
                    None, // timeout
                ) {
                    Ok(tuple) => tuple,
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                // Since we disallow resizing, assert that the swapchain and surface are optimally
                // configured.
                assert!(
                    !suboptimal,
                    "not handling sub-optimal swapchains in this sample code",
                );

                // If this image buffer already has a future then attempt to cleanup fence
                // resources. Usually the future for this index will have completed by the time we
                // are rendering it again.
                if let Some(image_fence) = &mut fences[image_index as usize] {
                    image_fence.cleanup_finished()
                }

                // If the previous image has a fence then use it for synchronization, else create
                // a new one.
                let previous_future = match fences[previous_fence_index as usize].take() {
                    // Ensure current frame is synchronized with previous.
                    Some(fence) => fence.boxed(),
                    // Create new future to guarentee synchronization with (fake) previous frame.
                    None => sync::now(device.clone()).boxed(),
                };

                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                builder
                    // Push constants for compute shader.
                    .push_constants(compute_pipeline.layout().clone(), 0, push_constants)
                    .unwrap()
                    // Perform compute operation to update particle positions.
                    .bind_pipeline_compute(compute_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        compute_pipeline.layout().clone(),
                        0, // Bind this descriptor set to index 0.
                        descriptor_set.clone(),
                    )
                    .unwrap()
                    .dispatch([PARTICLE_COUNT as u32 / 128, 1, 1])
                    .unwrap()
                    // Use render-pass to draw particles to swapchain.
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0., 0., 0., 1.].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        Default::default(),
                    )
                    .unwrap()
                    .bind_pipeline_graphics(graphics_pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    .draw(PARTICLE_COUNT as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(Default::default())
                    .unwrap();
                let command_buffer = builder.build().unwrap();

                let future = previous_future
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                // Update this frame's future with current fence.
                fences[image_index as usize] = match future.map_err(Validated::unwrap) {
                    // Success, store result into vector.
                    Ok(future) => Some(future),

                    // Unknown failure.
                    Err(e) => panic!("failed to flush future: {e}"),
                };
                previous_fence_index = image_index;
            }
            Event::AboutToWait => window.request_redraw(),
            _ => (),
        }
    })
}
