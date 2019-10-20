// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Welcome to the instancing example!
//
// This is a simple, modified version of the `triangle.rs` example that demonstrates how we can use
// the "instancing" technique with vulkano to draw many instances of the triangle.

#[macro_use]
extern crate vulkano;
extern crate vulkano_shaders;
extern crate winit;
extern crate vulkano_win;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::vertex::OneVertexOneInstanceDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;

use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{Window, WindowBuilder};
use winit::event::{Event, WindowEvent};

use std::sync::Arc;

// # Vertex Types
//
// Seeing as we are going to use the `OneVertexOneInstanceDefinition` vertex definition for our
// graphics pipeline, we need to define two vertex types:
//
// 1. `Vertex` is the vertex type that we will use to describe the triangle's geometry.
#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
impl_vertex!(Vertex, position);

// 2. `InstanceData` is the vertex type that describes the unique data per instance.
#[derive(Default, Debug, Clone)]
struct InstanceData {
    position_offset: [f32; 2],
    scale: f32,
}
impl_vertex!(InstanceData, position_offset, scale);

fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).unwrap()
    };

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());


    let events_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window();

    let queue_family = physical.queue_families().find(|&q| {
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };
    let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
        [(queue_family, 0.5)].iter().cloned()).unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let initial_dimensions = {
            let dimensions: (u32, u32) = window.inner_size().to_physical(window.hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        };

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
            initial_dimensions, 1, usage, &queue, SurfaceTransform::Identity, alpha,
            PresentMode::Fifo, true, None).unwrap()
    };

    // We now create a buffer that will store the shape of our triangle.
    // This triangle is identical to the one in the `triangle.rs` example.
    let triangle_vertex_buffer = {
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
            Vertex { position: [-0.5, -0.25] },
            Vertex { position: [0.0, 0.5] },
            Vertex { position: [0.25, -0.1] }
        ].iter().cloned()).unwrap()
    };

    // Now we create another buffer that will store the unique data per instance.
    // For this example, we'll have the instances form a 10x10 grid that slowly gets larger.
    let instance_data_buffer = {
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
                data.push(InstanceData { position_offset, scale });
            }
        }
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), data.iter().cloned())
            .unwrap()
    };

    mod vs {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: "
#version 450

// The triangle vertex positions.
layout(location = 0) in vec2 position;

// The per-instance data.
layout(location = 1) in vec2 position_offset;
layout(location = 2) in float scale;

void main() {
    // Apply the scale and offset for the instance.
    gl_Position = vec4(position * scale + position_offset, 0.0, 1.0);
}"
        }
    }

    mod fs {
        vulkano_shaders::shader!{
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

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(single_pass_renderpass!(
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
    ).unwrap());

    let pipeline = Arc::new(GraphicsPipeline::start()
        // Use the `OneVertexOneInstanceDefinition` to describe to vulkano how the two vertex types
        // are expected to be used.
        .vertex_input(OneVertexOneInstanceDefinition::<Vertex, InstanceData>::new())
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    events_loop.run(move |ev, _, cf| {
        *cf = ControlFlow::Poll;
        let window = surface.window();

        previous_frame_end.as_mut().unwrap().cleanup_finished();

        if recreate_swapchain {
            let dimensions = {
                let dimensions: (u32, u32) = window.inner_size().to_physical(window.hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            };
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                Err(err) => panic!("{:?}", err)
            };
            swapchain = new_swapchain;
            framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state);
            recreate_swapchain = false;
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                return;
            },
            Err(err) => panic!("{:?}", err)
        };

        let clear_values = vec!([0.0, 0.0, 1.0, 1.0].into());

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            .draw(
                pipeline.clone(),
                &dynamic_state,
                // We pass both our lists of vertices here.
                (triangle_vertex_buffer.clone(), instance_data_buffer.clone()),
                (),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            .build().unwrap();

		let prev = previous_frame_end.take();
        let future = prev.unwrap().join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                future.wait(None).unwrap();
                previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
            }
        }

        match ev {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => *cf = ControlFlow::Exit,
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
            _ => ()
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}
