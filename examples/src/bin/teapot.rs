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
#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;
use vulkano::command_buffer::CommandBufferBuilder;
use vulkano::sync::GpuFuture;
use vulkano::image::ImageView;
use vulkano::instance::debug::{self, DebugCallback};

use std::sync::Arc;

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    let extensions = vulkano::instance::InstanceExtensions {
        ext_debug_report: true,
        ..vulkano_win::required_extensions()
    };
    let instance = vulkano::instance::Instance::new(None, &extensions, None).expect("failed to create instance");

    let debug_callback = DebugCallback::new(&instance, debug::MessageTypes {
        error: true,
        warning: true,
        performance_warning: true,
        information: false,
        debug: false,
    }, move |msg| {
        let level = if msg.ty.error {
            "ERROR"
        } else if msg.ty.warning || msg.ty.performance_warning {
            "WARNING"
        } else {
            unreachable!();
        };
        println!("{}: {}: {}", level, msg.layer_prefix, msg.description);
    }).expect("failed to create debug callback");

    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();

    let queue = physical.queue_families().find(|&q| q.supports_graphics() &&
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
        let caps = window.surface().capabilities(physical).expect("failed to get surface capabilities");

        let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
        let present = caps.present_modes.iter().next().unwrap();
        let usage = caps.supported_usage_flags;
        let format = caps.supported_formats[0].0;

        vulkano::swapchain::Swapchain::new(device.clone(), window.surface().clone(), caps.min_image_count, format, dimensions, 1,
                                           usage, &queue, vulkano::swapchain::SurfaceTransform::Identity,
                                           vulkano::swapchain::CompositeAlpha::Opaque,
                                           present, true, None).expect("failed to create swapchain")
    };


    let depth_buffer = vulkano::image::attachment::AttachmentImage::transient(device.clone(), images[0].dimensions(), vulkano::format::D16Unorm).unwrap();

    let vertex_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(), Some(queue.family()), examples::VERTICES.iter().cloned())
                                .expect("failed to create buffer");

    let normals_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(), Some(queue.family()), examples::NORMALS.iter().cloned())
                                .expect("failed to create buffer");

    let index_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(), Some(queue.family()), examples::INDICES.iter().cloned())
                                .expect("failed to create buffer");

    // note: this teapot was meant for OpenGL where the origin is at the lower left
    //       instead the origin is at the upper left in vulkan, so we reverse the Y axis
    let proj = cgmath::perspective(cgmath::Rad(std::f32::consts::FRAC_PI_2), { let d = images[0].dimensions(); d[0] as f32 / d[1] as f32 }, 0.01, 100.0);
    let view = cgmath::Matrix4::look_at(cgmath::Point3::new(0.3, 0.3, 1.0), cgmath::Point3::new(0.0, 0.0, 0.0), cgmath::Vector3::new(0.0, -1.0, 0.0));
    let scale = cgmath::Matrix4::from_scale(0.01);

    let uniform_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::<vs::ty::Data>
                               ::from_data(device.clone(), vulkano::buffer::BufferUsage::all(), Some(queue.family()), 
                                vs::ty::Data {
                                    world : <cgmath::Matrix4<f32> as cgmath::SquareMatrix>::identity().into(),
                                    view : (view * scale).into(),
                                    proj : proj.into(),
                                })
                               .expect("failed to create buffer");

    let vs = vs::Shader::load(&device).expect("failed to create shader module");
    let fs = fs::Shader::load(&device).expect("failed to create shader module");

    let renderpass = Arc::new(
        single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: vulkano::format::Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        ).unwrap()
    );

    let pipeline = Arc::new(vulkano::pipeline::GraphicsPipeline::new(device.clone(), vulkano::pipeline::GraphicsPipelineParams {
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
    }).unwrap());

    let set = Arc::new(simple_descriptor_set!(pipeline.clone(), 0, {
        uniforms: uniform_buffer.clone()
    }));

    let framebuffers = images.iter().map(|image| {
        Arc::new(vulkano::framebuffer::Framebuffer::start(renderpass.clone())
            .add(image.clone()).unwrap()
            .add(depth_buffer.clone()).unwrap()
            .build().unwrap())
    }).collect::<Vec<_>>();


    let mut previous_frame = Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>;

    loop {
        previous_frame.cleanup_finished();

        {
            // aquiring write lock for the uniform buffer
            let mut buffer_content = uniform_buffer.write().unwrap(); 

            let rotation = cgmath::Matrix3::from_angle_y(cgmath::Rad(time::precise_time_ns() as f32 * 0.000000001));

            // since write lock implementd Deref and DerefMut traits, 
            // we can update content directly 
            buffer_content.world = cgmath::Matrix4::from(rotation).into();
        }

        let (image_num, acquire_future) = vulkano::swapchain::acquire_next_image(swapchain.clone(), std::time::Duration::new(1, 0)).unwrap();

        let command_buffer = vulkano::command_buffer::AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
            .begin_render_pass(
                framebuffers[image_num].clone(), false,
                vec![
                    [0.0, 0.0, 1.0, 1.0].into(),
                    1f32.into()
                ]).unwrap()
            .draw_indexed(
                pipeline.clone(), vulkano::command_buffer::DynamicState::none(),
                (vertex_buffer.clone(), normals_buffer.clone()), 
                index_buffer.clone(), set.clone(), ()).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();
        
        let future = previous_frame.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush().unwrap();
        previous_frame = Box::new(future) as Box<_>;

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => done = true,
                _ => ()
            }
        });
        if done { return; }
    }
}

mod vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    v_normal = transpose(inverse(mat3(worldview))) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
}
"]
    struct Dummy;
}

mod fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 0) out vec4 f_color;

const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

void main() {
    float brightness = dot(normalize(v_normal), normalize(LIGHT));
    vec3 dark_color = vec3(0.6, 0.0, 0.0);
    vec3 regular_color = vec3(1.0, 0.0, 0.0);

    f_color = vec4(mix(dark_color, regular_color, brightness), 1.0);
}
"]
    struct Dummy;
}
