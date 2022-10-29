// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A minimal particle-sandbox to demonstrate a reasonable use-case for a `DeviceLocalBuffer`.
//! We gain significant runtime performance by copying the inital vertex values to the GPU using
//! a `CpuAccessibleBuffer` and then moving the data to a `DeviceLocalBuffer` to be accessed soley
//! by the GPU through the compute shader and as a vertex array.
//!

use bytemuck::{Pod, Zeroable};
use std::{sync::Arc, time::SystemTime};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    image::{view::ImageView, ImageUsage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    swapchain::{PresentMode, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{FenceSignalFuture, GpuFuture},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

const PARTICLE_COUNT: usize = 100_000;

fn main() {
    // The usual Vulkan initialization.
    // Largely the same as example `triangle.rs` until further commentation is provided.
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_resizable(false) // For simplicity, we are going to assert that the window size is static
        .with_title("simple particles")
        .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

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
                    q.queue_flags.graphics && p.surface_support(i as u32, &surface).unwrap_or(false)
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
        physical_device.properties().device_type
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

        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent: [WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32],
                image_usage: ImageUsage {
                    color_attachment: true,
                    ..ImageUsage::empty()
                },
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
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
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
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
            src: "
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
                layout (push_constant) uniform PushConstants
                {
                    vec2 attractor;
                    float attractor_strength;
                    float delta_time;
                } push;

                const float maxSpeed = 10.0; // Keep this value in sync with the `maxSpeed` const in the vertex shader.
                const float minLength = 0.02;
                const float friction = -2.0;

                void main() {
                    const uint index = gl_GlobalInvocationID.x;

                    vec2 vel = verticies[index].vel;

                    // Update particle position according to velocity.
                    vec2 pos = verticies[index].pos + push.delta_time * vel;

                    // Bounce particle off screen-border.
                    if(abs(pos.x) > 1.0) {
                        vel.x = sign(pos.x) * (-0.95 * abs(vel.x) - 0.0001);
                        if(abs(pos.x) >= 1.05) {
                            pos.x = sign(pos.x);
                        }
                    }
                    if(abs(pos.y) > 1.0) {
                        vel.y = sign(pos.y) * (-0.95 * abs(vel.y) - 0.0001);
                        if(abs(pos.y) >= 1.05) {
                            pos.y = sign(pos.y);
                        }
                    }

                    // Simple inverse-square force.
                    vec2 t = push.attractor - pos;
                    float r = max(length(t), minLength);
                    vec2 force = push.attractor_strength * (t/r) / (r*r);

                    // Update velocity, enforcing a maximum speed.
                    vel += push.delta_time * force;
                    if(length(vel) > maxSpeed) {
                        vel = maxSpeed*normalize(vel);
                    }

                    // Set new values back into buffer.
                    verticies[index].pos = pos;
	                verticies[index].vel = vel * exp(friction * push.delta_time);
                }
            ",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            },
        }
    }
    // Vertex shader determines color and is run once per particle.
    // The vertices will be updates by the compute shader each frame.
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
                #version 450

                layout(location = 0) in vec2 pos;
                layout(location = 1) in vec2 vel;

                layout(location = 0) out vec4 outColor;

                const float maxSpeed = 10.0; // Keep this value in sync with the `maxSpeed` const in the compute shader.

                void main() {
                    gl_Position = vec4(pos, 0.0, 1.0);
	                gl_PointSize = 1.0;

                    // Mix colors based on position and velocity.
                    outColor = mix(0.2*vec4(pos, abs(vel.x)+abs(vel.y), 1.0), vec4(1.0, 0.5, 0.8, 1.0), sqrt(length(vel)/maxSpeed));
                }"
        }
    }
    // Fragment shader will only need to apply the color forwarded by the vertex shader.
    // This is because the color of a particle should be identical over all pixels.
    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
                #version 450

                layout(location = 0) in vec4 outColor;

                layout(location = 0) out vec4 fragColor;

                void main() {
                    fragColor = outColor;
                }
            "
        }
    }

    let cs = cs::load(device.clone()).unwrap();
    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
    struct Vertex {
        pos: [f32; 2],
        vel: [f32; 2],
    }
    impl_vertex!(Vertex, pos, vel);

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

        // Create a CPU accessible buffer initialized with the vertex data.
        let temporary_accessible_buffer = CpuAccessibleBuffer::from_iter(
            &memory_allocator,
            BufferUsage {
                transfer_src: true,
                ..BufferUsage::empty()
            }, // Specify this buffer will be used as a transfer source.
            false,
            vertices,
        )
        .unwrap();

        // Create a buffer array on the GPU with enough space for `PARTICLE_COUNT` number of `Vertex`.
        let device_local_buffer = DeviceLocalBuffer::<[Vertex]>::array(
            &memory_allocator,
            PARTICLE_COUNT as vulkano::DeviceSize,
            BufferUsage {
                storage_buffer: true,
                ..BufferUsage::empty()
            } | BufferUsage {
                transfer_dst: true,
                vertex_buffer: true,
                ..BufferUsage::empty()
            }, // Specify use as a storage buffer, vertex buffer, and transfer destination.
            device.active_queue_family_indices().iter().copied(),
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

    // Create compute-pipeline for applying compute shader to vertices.
    let compute_pipeline = vulkano::pipeline::ComputePipeline::new(
        device.clone(),
        cs.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("Failed to create compute shader");

    // Create a new descriptor set for binding vertices as a Storage Buffer.
    use vulkano::pipeline::Pipeline; // Required to access layout() method of pipeline.
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        compute_pipeline
            .layout()
            .set_layouts()
            .get(0) // 0 is the index of the descriptor set.
            .unwrap()
            .clone(),
        [
            // 0 is the binding of the data in this set. We bind the `DeviceLocalBuffer` of vertices here.
            WriteDescriptorSet::buffer(0, vertex_buffer.clone()),
        ],
    )
    .unwrap();

    // Fixed viewport dimensions.
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32],
        depth_range: 0.0..1.0,
    };

    // Create a basic graphics pipeline for rendering particles.
    let graphics_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::PointList)) // Vertices will be rendered as a list of points.
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device.clone())
        .unwrap();

    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; framebuffers.len()];
    let mut previous_fence_index = 0u32;

    let start_time = SystemTime::now();
    let mut last_frame_time = start_time;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::RedrawEventsCleared => {
                let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
                let dimensions = window.inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                // Update per-frame variables.
                let now = SystemTime::now();
                let time = now.duration_since(start_time).unwrap().as_secs_f32();
                let delta_time = now.duration_since(last_frame_time).unwrap().as_secs_f32();
                last_frame_time = now;

                // Create push contants to be passed to compute shader.
                let push_constants = cs::ty::PushConstants {
                    attractor: [0.75 * (3. * time).cos(), 0.6 * (0.75 * time).sin()],
                    attractor_strength: 1.2 * (2. * time).cos(),
                    delta_time,
                };

                // Acquire information on the next swapchain target.
                let (image_index, suboptimal, acquire_future) =
                    match vulkano::swapchain::acquire_next_image(
                        swapchain.clone(),
                        None, /*timeout*/
                    ) {
                        Ok(tuple) => tuple,
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                // Since we disallow resizing, assert the swapchain and surface are optimally configured.
                assert!(
                    !suboptimal,
                    "Not handling sub-optimal swapchains in this sample code"
                );

                // If this image buffer already has a future then attempt to cleanup fence resources.
                // Usually the future for this index will have completed by the time we are rendering it again.
                if let Some(image_fence) = &mut fences[image_index as usize] {
                    image_fence.cleanup_finished()
                }

                // If the previous image has a fence then use it for synchronization, else create a new one.
                let previous_future = match fences[previous_fence_index as usize].clone() {
                    // Ensure current frame is synchronized with previous.
                    Some(fence) => fence.boxed(),

                    // Create new future to guarentee synchronization with (fake) previous frame.
                    None => vulkano::sync::now(device.clone()).boxed(),
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
                    // Perform compute operation to update particle positions.
                    .bind_pipeline_compute(compute_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        compute_pipeline.layout().clone(),
                        0, // Bind this descriptor set to index 0.
                        descriptor_set.clone(),
                    )
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
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    .bind_pipeline_graphics(graphics_pipeline.clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(PARTICLE_COUNT as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass()
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
                fences[image_index as usize] = match future {
                    // Success, store result into vector.
                    Ok(future) => Some(Arc::new(future)),

                    // Unknown failure.
                    Err(e) => panic!("Failed to flush future: {:?}", e),
                };
                previous_fence_index = image_index;
            }
            _ => (),
        }
    });
}
