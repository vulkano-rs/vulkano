// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Some relevant documentation:
// *    Tessellation overview           https://www.khronos.org/opengl/wiki/Tessellation
// *    Tessellation Control Shader     https://www.khronos.org/opengl/wiki/Tessellation_Control_Shader
// *    Tessellation Evaluation Shader  https://www.khronos.org/opengl/wiki/Tessellation_Evaluation_Shader
// *    Tessellation real-world usage 1 http://ogldev.atspace.co.uk/www/tutorial30/tutorial30.html
// *    Tessellation real-world usage 2 http://prideout.net/blog/?p=48

// Notable elements of this example:
// *    tessellation control shader and a tessellation evaluation shader
// *    tessellation_shaders(..), patch_list(3) and polygon_mode_line() are called on the pipeline builder

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
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

mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}"
    }
}

mod tcs {
    vulkano_shaders::shader!{
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
}"
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
// http://mathworld.wolfram.com/BarycentricCoordinates.html

mod tes {
    vulkano_shaders::shader!{
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
    f_color = vec4(1.0, 1.0, 1.0, 1.0);
}"
    }
}


fn main() {
    let extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &extensions, None).unwrap();

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

    let initial_dimensions = {
        let dimensions: (u32, u32) = window.inner_size().to_physical(window.hidpi_factor()).into();
        [dimensions.0, dimensions.1]
    };

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format, initial_dimensions,
            1, usage, &queue, SurfaceTransform::Identity, alpha, PresentMode::Fifo, true, None).unwrap()
    };

    let vertex_buffer = {
        #[derive(Default, Debug, Clone)]
        struct Vertex { position: [f32; 2] }
        vulkano::impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
            Vertex { position: [-0.5,  -0.25] },
            Vertex { position: [ 0.0,   0.5] },
            Vertex { position: [ 0.25, -0.1] },
            Vertex { position: [ 0.9,   0.9] },
            Vertex { position: [ 0.9,   0.8] },
            Vertex { position: [ 0.8,   0.8] },
            Vertex { position: [-0.9,   0.9] },
            Vertex { position: [-0.7,   0.6] },
            Vertex { position: [-0.5,   0.9] },
        ].iter().cloned()).unwrap()
    };

    let vs = vs::Shader::load(device.clone()).unwrap();
    let tcs = tcs::Shader::load(device.clone()).unwrap();
    let tes = tes::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(vulkano::single_pass_renderpass!(
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
        .vertex_input_single_buffer()
        .vertex_shader(vs.main_entry_point(), ())
        // Actually use the tessellation shaders.
        .tessellation_shaders(tcs.main_entry_point(), (), tes.main_entry_point(), ())
        // use PrimitiveTopology::PathList(3)
        // Use a vertices_per_patch of 3, because we want to convert one triangle into lots of
        // little ones. A value of 4 would convert a rectangle into lots of little triangles.
        .patch_list(3)
        // Enable line mode so we can see the generated vertices.
        .polygon_mode_line()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

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
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    return;
                },
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

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers[image_num].clone(), false, vec![[0.0, 0.0, 0.0, 1.0].into()])
            .unwrap()
            .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
            .unwrap()
            .end_render_pass()
            .unwrap()
            .build().unwrap();

        let future = previous_frame_end.take().unwrap().join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                // This wait is required when using NVIDIA or running on macOS. See https://github.com/vulkano-rs/vulkano/issues/1247
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
            _ => (),
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
