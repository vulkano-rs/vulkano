// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Welcome to the instancing example!
//
// This is a simple, modified version of the `triangle.rs` example that demonstrates how we can use
// the "instancing" technique with vulkano to draw many instances of the triangle.

use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::PipelineShaderStageCreateInfo,
    single_pass_renderpass,
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

/// The vertex type that we will be used to describe the triangle's geometry.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct TriangleVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

/// The vertex type that describes the unique data per instance.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct InstanceData {
    #[format(R32G32_SFLOAT)]
    position_offset: [f32; 2],
    #[format(R32_SFLOAT)]
    scale: f32,
}

fn main() {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
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

    let (mut swapchain, images) = {
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
        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
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

    // We now create a buffer that will store the shape of our triangle. This triangle is identical
    // to the one in the `triangle.rs` example.
    let vertices = [
        TriangleVertex {
            position: [-0.5, -0.25],
        },
        TriangleVertex {
            position: [0.0, 0.5],
        },
        TriangleVertex {
            position: [0.25, -0.1],
        },
    ];
    let vertex_buffer = {
        Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vertices,
        )
        .unwrap()
    };

    // Now we create another buffer that will store the unique data per instance. For this example,
    // we'll have the instances form a 10x10 grid that slowly gets larger.
    let instances = {
        let rows = 10;
        let cols = 10;
        let n_instances = rows * cols;
        let mut data = Vec::new();
        for c in 0..cols {
            for r in 0..rows {
                let half_cell_w = 0.5 / cols as f32;
                let half_cell_h = 0.5 / rows as f32;
                let x = half_cell_w + (c as f32 / cols as f32) * 2.0 - 1.0;
                let y = half_cell_h + (r as f32 / rows as f32) * 2.0 - 1.0;
                let position_offset = [x, y];
                let scale = (2.0 / rows as f32) * (c * rows + r) as f32 / n_instances as f32;
                data.push(InstanceData {
                    position_offset,
                    scale,
                });
            }
        }
        data
    };
    let instance_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        instances,
    )
    .unwrap();

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450

                // The triangle vertex positions.
                layout(location = 0) in vec2 position;

                // The per-instance data.
                layout(location = 1) in vec2 position_offset;
                layout(location = 2) in float scale;

                void main() {
                    // Apply the scale and offset for the instance.
                    gl_Position = vec4(position * scale + position_offset, 0.0, 1.0);
                }
            ",
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            ",
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let render_pass = single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap();

    let pipeline = GraphicsPipeline::start()
        .stages([
            PipelineShaderStageCreateInfo::entry_point(vs.entry_point("main").unwrap()),
            PipelineShaderStageCreateInfo::entry_point(fs.entry_point("main").unwrap()),
        ])
        // Use the implementations of the `Vertex` trait to describe to vulkano how the two vertex
        // types are expected to be used.
        .vertex_input_state([TriangleVertex::per_vertex(), InstanceData::per_instance()])
        .input_assembly_state(InputAssemblyState::default())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .rasterization_state(RasterizationState::default())
        .multisample_state(MultisampleState::default())
        .color_blend_state(ColorBlendState::default())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    event_loop.run(move |event, _, control_flow| {
        match event {
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
                let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
                let dimensions = window.inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("failed to recreate swapchain: {e}"),
                        };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                    recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
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
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    // We pass both our lists of vertices here.
                    .bind_vertex_buffers(0, (vertex_buffer.clone(), instance_buffer.clone()))
                    .draw(
                        vertex_buffer.len() as u32,
                        instance_buffer.len() as u32,
                        0,
                        0,
                    )
                    .unwrap()
                    .end_render_pass()
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

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
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
        }
    });
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
