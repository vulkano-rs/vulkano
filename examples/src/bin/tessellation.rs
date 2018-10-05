#![feature(proc_macro_non_items)]

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

#[macro_use]
extern crate vulkano;
extern crate vulkano_shader_derive;
extern crate winit;
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::device::Device;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::Subpass;
use vulkano::instance::Instance;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::AcquireError;
use vulkano::swapchain::SwapchainCreationError;
use vulkano::sync::now;
use vulkano::sync::GpuFuture;
use vulkano_shader_derive::vulkano_shader;

use std::sync::Arc;

fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).expect("failed to create Vulkan instance")
    };

    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let mut events_loop = winit::EventsLoop::new();
    let surface = winit::WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();

    let queue = physical.queue_families().find(|&q| {
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).expect("couldn't find a graphical queue family");
    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            .. vulkano::device::DeviceExtensions::none()
        };

        Device::new(physical, physical.supported_features(), &device_ext,
                    [(queue, 0.5)].iter().cloned()).expect("failed to create device")
    };
    let queue = queues.next().unwrap();

    let mut dimensions;

    let (mut swapchain, mut images) = {
        let caps = surface.capabilities(physical)
                         .expect("failed to get surface capabilities");

        dimensions = caps.current_extent.unwrap_or([1024, 768]);
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
                       dimensions, 1, caps.supported_usage_flags, &queue,
                       SurfaceTransform::Identity, alpha, PresentMode::Fifo, true,
                       None).expect("failed to create swapchain")
    };

    let vertex_buffer = {
        #[derive(Debug, Clone)]
        struct Vertex { position: [f32; 2] }
        impl_vertex!(Vertex, position);

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
        ].iter().cloned()).expect("failed to create buffer")
    };

    vulkano_shader!{
        mod_name: vs,
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}"
    }

    vulkano_shader!{
        mod_name: tcs,
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

    vulkano_shader!{
        mod_name: tes,
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

    vulkano_shader!{
        mod_name: fs,
        ty: "fragment",
        src: "
#version 450

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 1.0, 1.0, 1.0);
}"
    }

    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let tcs = tcs::Shader::load(device.clone()).expect("failed to create shader module");
    let tes = tes::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");

    let render_pass = Arc::new(single_pass_renderpass!(device.clone(),
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

    let mut framebuffers: Option<Vec<Arc<vulkano::framebuffer::Framebuffer<_,_>>>> = None;
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(now(device.clone())) as Box<GpuFuture>;
    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0 .. 1.0,
        }]),
        scissors: None,
    };

    loop {
        previous_frame_end.cleanup_finished();
        if recreate_swapchain {
            dimensions = surface.capabilities(physical)
                        .expect("failed to get surface capabilities")
                        .current_extent.unwrap();

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                },
                Err(err) => panic!("{:?}", err)
            };

            swapchain = new_swapchain;
            images = new_images;
            framebuffers = None;
            dynamic_state.viewports = Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0 .. 1.0,
            }]);

            recreate_swapchain = false;
        }

        if framebuffers.is_none() {
            framebuffers = Some(images.iter().map(|image| {
                Arc::new(Framebuffer::start(render_pass.clone())
                         .add(image.clone()).unwrap()
                         .build().unwrap())
            }).collect::<Vec<_>>());
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(),
                                                                              None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers.as_ref().unwrap()[image_num].clone(), false,
                               vec![[0.0, 0.0, 0.0, 1.0].into()])
            .unwrap()
            .draw(pipeline.clone(),
                  &dynamic_state,
                  vertex_buffer.clone(), (), ())
            .unwrap()
            .end_render_pass()
            .unwrap()
            .build().unwrap();

        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::CloseRequested, .. } => done = true,
                _ => ()
            }
        });
        if done { return }
    }
}
