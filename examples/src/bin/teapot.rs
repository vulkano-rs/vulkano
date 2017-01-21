// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

extern crate examples;
extern crate cgmath;
extern crate winit;
extern crate time;

#[macro_use]
extern crate vulkano;
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;
use vulkano::command_buffer::CommandsList;
use vulkano::command_buffer::Submit;

use std::sync::Arc;
use std::time::Duration;

mod vs { include!{concat!(env!("OUT_DIR"), "/shaders/src/bin/teapot_vs.glsl")} }
mod fs { include!{concat!(env!("OUT_DIR"), "/shaders/src/bin/teapot_fs.glsl")} }

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    let extensions = vulkano_win::required_extensions();
    let instance = vulkano::instance::Instance::new(None, &extensions, None).expect("failed to create instance");

    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let window = winit::WindowBuilder::new().build_vk_surface(&instance).unwrap();

    let queue = physical.queue_families().find(|q| q.supports_graphics() &&
                                                   window.surface().is_supported(q).unwrap_or(false))
                                                .expect("couldn't find a graphical queue family");

    let device_ext = vulkano::device::DeviceExtensions {
        khr_swapchain: true,
        .. vulkano::device::DeviceExtensions::none()
    };

    let (device, mut queues) = vulkano::device::Device::new(&physical, physical.supported_features(),
                                                            &device_ext, [(queue, 0.5)].iter().cloned())
                               .expect("failed to create device");
    let queue = queues.next().unwrap();

    let (swapchain, images) = {
        let caps = window.surface().get_capabilities(&physical).expect("failed to get surface capabilities");

        let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
        let present = caps.present_modes.iter().next().unwrap();
        let usage = caps.supported_usage_flags;
        let format = caps.supported_formats[0].0;

        vulkano::swapchain::Swapchain::new(&device, &window.surface(), caps.min_image_count, format, dimensions, 1,
                                           &usage, &queue, vulkano::swapchain::SurfaceTransform::Identity,
                                           vulkano::swapchain::CompositeAlpha::Opaque,
                                           present, true, None).expect("failed to create swapchain")
    };


    let depth_buffer = vulkano::image::attachment::AttachmentImage::transient(&device, images[0].dimensions(), vulkano::format::D16Unorm).unwrap();

    let vertex_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::from_iter(&device, &vulkano::buffer::BufferUsage::all(), Some(queue.family()), examples::VERTICES.iter().cloned())
                                .expect("failed to create buffer");

    let normals_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::from_iter(&device, &vulkano::buffer::BufferUsage::all(), Some(queue.family()), examples::NORMALS.iter().cloned())
                                .expect("failed to create buffer");

    let index_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::from_iter(&device, &vulkano::buffer::BufferUsage::all(), Some(queue.family()), examples::INDICES.iter().cloned())
                                .expect("failed to create buffer");

    // note: this teapot was meant for OpenGL where the origin is at the lower left
    //       instead the origin is at the upper left in vulkan, so we reverse the Y axis
    let proj = cgmath::perspective(cgmath::Rad(std::f32::consts::FRAC_PI_2), { let d = images[0].dimensions(); d[0] as f32 / d[1] as f32 }, 0.01, 100.0);
    let view = cgmath::Matrix4::look_at(cgmath::Point3::new(0.3, 0.3, 1.0), cgmath::Point3::new(0.0, 0.0, 0.0), cgmath::Vector3::new(0.0, -1.0, 0.0));
    let scale = cgmath::Matrix4::from_scale(0.01);

    let uniform_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::<vs::ty::Data>
                               ::from_data(&device, &vulkano::buffer::BufferUsage::all(), Some(queue.family()), 
                                vs::ty::Data {
                                    world : <cgmath::Matrix4<f32> as cgmath::SquareMatrix>::identity().into(),
                                    view : (view * scale).into(),
                                    proj : proj.into(),
                                })
                               .expect("failed to create buffer");

    let vs = vs::Shader::load(&device).expect("failed to create shader module");
    let fs = fs::Shader::load(&device).expect("failed to create shader module");

    mod renderpass {
        single_pass_renderpass!{
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: ::vulkano::format::Format,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: ::vulkano::format::D16Unorm,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        }
    }

    let renderpass = renderpass::CustomRenderPass::new(&device, &renderpass::Formats {
        color: (images[0].format(), 1),
        depth: (vulkano::format::D16Unorm, 1)
    }).unwrap();

    let pipeline = vulkano::pipeline::GraphicsPipeline::new(&device, vulkano::pipeline::GraphicsPipelineParams {
        vertex_input: vulkano::pipeline::vertex::TwoBuffersDefinition::new(),
        vertex_shader: vs.main_entry_point(),
        input_assembly: vulkano::pipeline::input_assembly::InputAssembly::triangle_list(),
        tessellation: None,
        geometry_shader: None,
        viewport: vulkano::pipeline::viewport::ViewportsState::Fixed {
            data: vec![(
                vulkano::pipeline::viewport::Viewport {
                    origin: [0.0, 0.0],
                    depth_range: 0.0 .. 1.0,
                    dimensions: [images[0].dimensions()[0] as f32, images[0].dimensions()[1] as f32],
                },
                vulkano::pipeline::viewport::Scissor::irrelevant()
            )],
        },
        raster: Default::default(),
        multisample: vulkano::pipeline::multisample::Multisample::disabled(),
        fragment_shader: fs.main_entry_point(),
        depth_stencil: vulkano::pipeline::depth_stencil::DepthStencil::simple_depth_test(),
        blend: vulkano::pipeline::blend::Blend::pass_through(),
        render_pass: vulkano::framebuffer::Subpass::from(renderpass.clone(), 0).unwrap(),
    }).unwrap();

    let set = Arc::new(simple_descriptor_set!(pipeline.clone(), 0, {
        uniforms: uniform_buffer.clone()
    }));

    let framebuffers = images.iter().map(|image| {
        let attachments = renderpass::AList {
            color: image.clone(),
            depth: depth_buffer.clone(),
        };

        vulkano::framebuffer::StdFramebuffer::new(renderpass.clone(), [image.dimensions()[0], image.dimensions()[1], 1], attachments).unwrap()
    }).collect::<Vec<_>>();


    let command_buffers = framebuffers.iter().map(|framebuffer| {
        Arc::new(vulkano::command_buffer::empty()
            .begin_render_pass(framebuffer.clone(), false, renderpass::ClearValues {
                 color: [0.0, 0.0, 1.0, 1.0],
                 depth: 1.0,
             })
            .draw_indexed(pipeline.clone(), vulkano::command_buffer::DynamicState::none(),
                          (vertex_buffer.clone(), normals_buffer.clone()), index_buffer.clone(), set.clone(), ())
            .end_render_pass().unwrap()
            .build_primary(&device, queue.family()).unwrap())
    }).collect::<Vec<_>>();

    let mut submissions: Vec<vulkano::command_buffer::Submission> = Vec::new();


    loop {
        submissions.retain(|s| s.destroying_would_block());

        {
            // aquiring write lock for the uniform buffer
            let mut buffer_content = uniform_buffer.write(Duration::new(1, 0)).unwrap(); 

            let rotation = cgmath::Matrix3::from_angle_y(cgmath::Rad(time::precise_time_ns() as f32 * 0.000000001));

            // since write lock implementd Deref and DerefMut traits, 
            // we can update content directly 
            buffer_content.world = cgmath::Matrix4::from(rotation).into();
        }

        let image_num = swapchain.acquire_next_image(Duration::from_millis(1)).unwrap();
        submissions.push(command_buffers[image_num].clone().submit(&queue).unwrap());
        swapchain.present(&queue, image_num).unwrap();

        for ev in window.window().poll_events() {
            match ev {
                winit::Event::Closed => return,
                _ => ()
            }
        }
    }
}
