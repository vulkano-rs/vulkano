// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Welcome to the triangle example!
//
// This is the only example that is entirely detailed. All the other examples avoid code
// duplication by using helper functions.
//
// This example assumes that you are already more or less familiar with graphics programming
// and that you want to learn Vulkan. This means that for example it won't go into details about
// what a vertex or a shader is.

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::swapchain;
use vulkano::swapchain::Surface;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;
use winit::event::ElementState;
use winit::event::KeyboardInput;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use std::collections::HashMap;
use std::sync::Arc;

// A struct to contain resources related to a window
struct WindowSurface {
    surface: Arc<Surface<Window>>,
    swapchain: Arc<Swapchain<Window>>,
    framebuffers: Vec<Arc<(dyn FramebufferAbstract + Send + Sync + 'static)>>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &required_extensions, None).unwrap();
    let event_loop = EventLoop::new();

    // A hashmap that contains all of our created windows and their resources
    let mut window_surfaces = HashMap::new();

    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    // Use the window's id as a means to access it from the hashmap
    let window_id = surface.window().id();

    // Find the device and a queue.
    // TODO: it is assumed the device, queue, and surface caps are the same for all windows

    let (device, queue, surface_caps) = {
        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
            .unwrap();

        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        let (device, mut queues) = Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .unwrap();
        (
            device,
            queues.next().unwrap(),
            surface.capabilities(physical).unwrap(),
        )
    };

    // The swapchain and framebuffer images for this perticular window

    let (swapchain, images) = {
        let alpha = surface_caps
            .supported_composite_alpha
            .iter()
            .next()
            .unwrap();
        let format = surface_caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            surface_caps.min_image_count,
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        )
        .unwrap()
    };

    let vertex_buffer = {
        #[derive(Default, Debug, Clone)]
        struct Vertex {
            position: [f32; 2],
        }
        vulkano::impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            [
                Vertex {
                    position: [-0.5, -0.25],
                },
                Vertex {
                    position: [0.0, 0.5],
                },
                Vertex {
                    position: [0.25, -0.1],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap()
    };

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
                #version 450

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

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
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
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None,
    };

    window_surfaces.insert(
        window_id,
        WindowSurface {
            surface,
            swapchain,
            recreate_swapchain: false,
            framebuffers: window_size_dependent_setup(
                &images,
                render_pass.clone(),
                &mut dynamic_state,
            ),
            previous_frame_end: Some(sync::now(device.clone()).boxed()),
        },
    );

    event_loop.run(move |event, event_loop, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            window_id,
            event: WindowEvent::Resized(_),
            ..
        } => {
            window_surfaces
                .get_mut(&window_id)
                .unwrap()
                .recreate_swapchain = true;
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                },
            ..
        } => {
            let surface = WindowBuilder::new()
                .build_vk_surface(&event_loop, instance.clone())
                .unwrap();
            let window_id = surface.window().id();
            let (swapchain, images) = {
                let alpha = surface_caps
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .unwrap();
                let format = surface_caps.supported_formats[0].0;
                let dimensions: [u32; 2] = surface.window().inner_size().into();

                Swapchain::new(
                    device.clone(),
                    surface.clone(),
                    surface_caps.min_image_count,
                    format,
                    dimensions,
                    1,
                    ImageUsage::color_attachment(),
                    &queue,
                    SurfaceTransform::Identity,
                    alpha,
                    PresentMode::Fifo,
                    FullscreenExclusive::Default,
                    true,
                    ColorSpace::SrgbNonLinear,
                )
                .unwrap()
            };

            window_surfaces.insert(
                window_id,
                WindowSurface {
                    surface,
                    swapchain,
                    recreate_swapchain: false,
                    framebuffers: window_size_dependent_setup(
                        &images,
                        render_pass.clone(),
                        &mut dynamic_state,
                    ),
                    previous_frame_end: Some(sync::now(device.clone()).boxed()),
                },
            );
        }
        Event::RedrawEventsCleared => {
            window_surfaces
                .values()
                .for_each(|s| s.surface.window().request_redraw());
        }
        Event::RedrawRequested(window_id) => {
            let WindowSurface {
                ref surface,
                ref mut swapchain,
                ref mut recreate_swapchain,
                ref mut framebuffers,
                ref mut previous_frame_end,
            } = window_surfaces.get_mut(&window_id).unwrap();

            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if *recreate_swapchain {
                let dimensions: [u32; 2] = surface.window().inner_size().into();
                let (new_swapchain, new_images) =
                    match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };

                *swapchain = new_swapchain;
                *framebuffers = window_size_dependent_setup(
                    &new_images,
                    render_pass.clone(),
                    &mut dynamic_state,
                );
                *recreate_swapchain = false;
            }

            let (image_num, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        *recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                *recreate_swapchain = true;
            }

            let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

            let mut builder =
                AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                    .unwrap();

            builder
                .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
                .unwrap()
                .draw(
                    pipeline.clone(),
                    &dynamic_state,
                    vertex_buffer.clone(),
                    (),
                    (),
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
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    *previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    *recreate_swapchain = true;
                    *previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    *previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => (),
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}
