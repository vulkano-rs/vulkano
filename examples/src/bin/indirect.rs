// Copyright (c) 2019 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Indirect draw example
//
// Indirect draw calls allow us to issue a draw without needing to know the number of vertices
// until later when the draw is executed by the GPU.
//
// This is used in situations where vertices are being generated on the GPU, such as a GPU
// particle simulation, and the exact number of output vertices cannot be known until
// the compute shader has run.
//
// In this example the compute shader is trivial and the number of vertices does not change.
// However is does demonstrate that each compute instance atomically updates the vertex
// counter before filling the vertex buffer.
//
// For an explanation of how the rendering of the triangles takes place see the `triangle.rs`
// example.
//

#[macro_use]
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use std::iter;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuBufferPool};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, DrawIndirectCommand, SubpassContents,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, QueueCreate};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageAccess, ImageUsage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::swapchain::{self, AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

// # Vertex Types
// `Vertex` is the vertex type that will be output from the compute shader and be input to the vertex shader.
#[repr(C)]
#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
impl_vertex!(Vertex, position);

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::start()
        .enabled_extensions(required_extensions)
        .build()
        .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::start()
        .queues([QueueCreate::family(queue_family)])
        .enabled_extensions(
            physical_device
                .required_extensions()
                .union(&device_extensions),
        )
        .build(physical_device)
        .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical_device).unwrap();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(composite_alpha)
            .build()
            .unwrap()
    };

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
                #version 450

                // The triangle vertex positions.
                layout(location = 0) in vec2 position;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            "
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
                #version 450

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            "
        }
    }

    // A simple compute shader that generates vertices. It has two buffers bound: the first is where we output the vertices, the second
    // is the IndirectDrawArgs struct we passed the draw_indirect so we can set the number to vertices to draw
    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
                #version 450

                layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

                layout(set = 0, binding = 0) buffer Output {
                    vec2 pos[];
                } triangles;

                layout(set = 0, binding = 1) buffer IndirectDrawArgs {
                    uint vertices;
                    uint unused0;
                    uint unused1;
                    uint unused2;
                };

                void main() {
                    uint idx = gl_GlobalInvocationID.x;

                    // each thread of compute shader is going to increment the counter, so we need to use atomic
                    // operations for safety. The previous value of the counter is returned so that gives us
                    // the offset into the vertex buffer this thread can write it's vertices into.
                    uint offset = atomicAdd(vertices, 6);

                    vec2 center = vec2(-0.8, -0.8) + idx * vec2(0.1, 0.1);
                    triangles.pos[offset + 0] = center + vec2(0.0, 0.0375);
                    triangles.pos[offset + 1] = center + vec2(0.025, -0.01725);
                    triangles.pos[offset + 2] = center + vec2(-0.025, -0.01725);
                    triangles.pos[offset + 3] = center + vec2(0.0, -0.0375);
                    triangles.pos[offset + 4] = center + vec2(0.025, 0.01725);
                    triangles.pos[offset + 5] = center + vec2(-0.025, 0.01725);
                }
            "
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();
    let cs = cs::load(device.clone()).unwrap();

    // Each frame we generate a new set of vertices and each frame we need a new DrawIndirectCommand struct to
    // set the number of vertices to draw
    let indirect_args_pool: CpuBufferPool<DrawIndirectCommand> =
        CpuBufferPool::new(device.clone(), BufferUsage::all());
    let vertex_pool: CpuBufferPool<Vertex> = CpuBufferPool::new(device.clone(), BufferUsage::all());

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        cs.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .unwrap();

    let render_pass = single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let render_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
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
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

                // Allocate a GPU buffer to hold the arguments for this frames draw call. The compute
                // shader will only update vertex_count, so set the other parameters correctly here.
                let indirect_args = indirect_args_pool
                    .chunk(iter::once(DrawIndirectCommand {
                        vertex_count: 0,
                        instance_count: 1,
                        first_vertex: 0,
                        first_instance: 0,
                    }))
                    .unwrap();

                // Allocate a GPU buffer to hold this frames vertices. This needs to be large enough to hold
                // the worst case number of vertices generated by the compute shader
                let vertices = vertex_pool
                    .chunk((0..(6 * 16)).map(|_| Vertex { position: [0.0; 2] }))
                    .unwrap();

                // Pass the two buffers to the compute shader
                let layout = compute_pipeline
                    .layout()
                    .descriptor_set_layouts()
                    .get(0)
                    .unwrap();
                let cs_desciptor_set = PersistentDescriptorSet::new(
                    layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, vertices.clone()),
                        WriteDescriptorSet::buffer(1, indirect_args.clone()),
                    ],
                )
                .unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // First in the command buffer we dispatch the compute shader to generate the vertices and fill out the draw
                // call arguments
                builder
                    .bind_pipeline_compute(compute_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        compute_pipeline.layout().clone(),
                        0,
                        cs_desciptor_set.clone(),
                    )
                    .dispatch([1, 1, 1])
                    .unwrap()
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values,
                    )
                    .unwrap()
                    // The indirect draw call is placed in the command buffer with a reference to the GPU buffer that will
                    // contain the arguments when the draw is executed on the GPU
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(render_pipeline.clone())
                    .bind_vertex_buffers(0, vertices.clone())
                    .draw_indirect(indirect_args.clone())
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
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
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
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Framebuffer::start(render_pass.clone())
                .add(view)
                .unwrap()
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>()
}
