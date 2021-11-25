// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Some relevant documentation:
// *    Tessellation overview           https://www.khronos.org/opengl/wiki/Tessellation
// *    Tessellation Control Shader     https://www.khronos.org/opengl/wiki/Tessellation_Control_Shader
// *    Tessellation Evaluation Shader  https://www.khronos.org/opengl/wiki/Tessellation_Evaluation_Shader
// *    Tessellation real-world usage 1 http://ogldev.atspace.co.uk/www/tutorial30/tutorial30.html
// *    Tessellation real-world usage 2 https://prideout.net/blog/?p=48

// Notable elements of this example:
// *    tessellation control shader and a tessellation evaluation shader
// *    tessellation_shaders(..), patch_list(3) and polygon_mode_line() are called on the pipeline builder

use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageAccess, ImageUsage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::rasterization::{PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::tessellation::TessellationState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::swapchain::{self, AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

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

mod tcs {
    vulkano_shaders::shader! {
        ty: "tess_ctrl",
        src: "
            #version 450

            layout (vertices = 3) out; // a value of 3 means a patch consists of a single triangle

            void main(void)
            {
                // save the position of the patch, so the tes can access it
                // We could define our own output variables for this,
                // but gl_out is handily provided.
                gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

                gl_TessLevelInner[0] = 10; // many triangles are generated in the center
                gl_TessLevelOuter[0] = 1;  // no triangles are generated for this edge
                gl_TessLevelOuter[1] = 10; // many triangles are generated for this edge
                gl_TessLevelOuter[2] = 10; // many triangles are generated for this edge
                // gl_TessLevelInner[1] = only used when tes uses layout(quads)
                // gl_TessLevelOuter[3] = only used when tes uses layout(quads)
            }
        "
    }
}

// PG
// There is a stage in between tcs and tes called Primitive Generation (PG)
// Shaders cannot be defined for it.
// It takes gl_TessLevelInner and gl_TessLevelOuter and uses them to generate positions within
// the patch and pass them to tes via gl_TessCoord.
//
// When tes uses layout(triangles) then gl_TessCoord is in barrycentric coordinates.
// if layout(quads) is used then gl_TessCoord is in cartesian coordinates.
// Barrycentric coordinates are of the form (x, y, z) where x + y + z = 1
// and the values x, y and z represent the distance from a vertex of the triangle.
// https://mathworld.wolfram.com/BarycentricCoordinates.html

mod tes {
    vulkano_shaders::shader! {
        ty: "tess_eval",
        src: "
            #version 450

            layout(triangles, equal_spacing, cw) in;

            void main(void)
            {
                // retrieve the vertex positions set by the tcs
                vec4 vert_x = gl_in[0].gl_Position;
                vec4 vert_y = gl_in[1].gl_Position;
                vec4 vert_z = gl_in[2].gl_Position;

                // convert gl_TessCoord from barycentric coordinates to cartesian coordinates
                gl_Position = vec4(
                    gl_TessCoord.x * vert_x.x + gl_TessCoord.y * vert_y.x + gl_TessCoord.z * vert_z.x,
                    gl_TessCoord.x * vert_x.y + gl_TessCoord.y * vert_y.y + gl_TessCoord.z * vert_z.y,
                    gl_TessCoord.x * vert_x.z + gl_TessCoord.y * vert_y.z + gl_TessCoord.z * vert_z.z,
                    1.0
                );
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
                f_color = vec4(1.0, 1.0, 1.0, 1.0);
            }
        "
    }
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let features = Features {
        tessellation_shader: true,
        fill_mode_non_solid: true,
        ..Features::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter(|&p| p.supported_features().is_superset_of(&features))
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
        physical_device.properties().device_type
    );

    let (device, mut queues) = Device::new(
        physical_device,
        &features,
        &physical_device
            .required_extensions()
            .union(&device_extensions),
        [(queue_family, 0.5)].iter().cloned(),
    )
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

    #[derive(Default, Debug, Clone)]
    #[repr(C)]
    struct Vertex {
        position: [f32; 2],
    }
    vulkano::impl_vertex!(Vertex, position);

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
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
            Vertex {
                position: [0.9, 0.9],
            },
            Vertex {
                position: [0.9, 0.8],
            },
            Vertex {
                position: [0.8, 0.8],
            },
            Vertex {
                position: [-0.9, 0.9],
            },
            Vertex {
                position: [-0.7, 0.6],
            },
            Vertex {
                position: [-0.5, 0.9],
            },
        ]
        .iter()
        .cloned(),
    )
    .unwrap();

    let vs = vs::load(device.clone()).unwrap();
    let tcs = tcs::load(device.clone()).unwrap();
    let tes = tes::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!(
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

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        // Actually use the tessellation shaders.
        .tessellation_shaders(
            tcs.entry_point("main").unwrap(),
            (),
            tes.entry_point("main").unwrap(),
            (),
        )
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::PatchList))
        .rasterization_state(RasterizationState::new().polygon_mode(PolygonMode::Line))
        .tessellation_state(
            TessellationState::new()
                // Use a patch_control_points of 3, because we want to convert one triangle into
                // lots of little ones. A value of 4 would convert a rectangle into lots of little
                // triangles.
                .patch_control_points(3),
        )
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

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
                framebuffers =
                    window_size_dependent_setup(&new_images, render_pass.clone(), &mut viewport);
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

            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    SubpassContents::Inline,
                    vec![[0.0, 0.0, 0.0, 1.0].into()],
                )
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
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
