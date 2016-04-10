// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

extern crate cgmath;
extern crate winit;

#[cfg(windows)]
use winit::os::windows::WindowExt;

#[macro_use]
extern crate vulkano;

#[path = "support/teapot.rs"]
mod teapot;

use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

mod vs { include!{concat!(env!("OUT_DIR"), "/shaders/examples/teapot_vs.glsl")} }
mod fs { include!{concat!(env!("OUT_DIR"), "/shaders/examples/teapot_fs.glsl")} }

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    // TODO: for the moment the AMD driver crashes if you don't pass an ApplicationInfo, but in theory it's optional
    let app = vulkano::instance::ApplicationInfo { application_name: "test", application_version: 1, engine_name: "test", engine_version: 1 };
    let extensions = vulkano::instance::InstanceExtensions {
        khr_surface: true,
        khr_win32_surface: true,
        .. vulkano::instance::InstanceExtensions::none()
    };
    let instance = vulkano::instance::Instance::new(Some(&app), &extensions, None).expect("failed to create instance");

    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let window = winit::WindowBuilder::new().build().unwrap();
    let surface = unsafe { vulkano::swapchain::Surface::from_hwnd(&instance, ptr::null() as *const () /* FIXME */, window.get_hwnd()).unwrap() };

    let queue = physical.queue_families().find(|q| q.supports_graphics() &&
                                                   surface.is_supported(q).unwrap_or(false))
                                                .expect("couldn't find a graphical queue family");

    let device_ext = vulkano::device::DeviceExtensions {
        khr_swapchain: true,
        .. vulkano::device::DeviceExtensions::none()
    };

    let (device, queues) = vulkano::device::Device::new(&physical, physical.supported_features(),
                                                        &device_ext, None, [(queue, 0.5)].iter().cloned())
                                                                .expect("failed to create device");
    let queue = queues.into_iter().next().unwrap();

    let (swapchain, images) = {
        let caps = surface.get_capabilities(&physical).expect("failed to get surface capabilities");

        let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
        let present = caps.present_modes[0];
        let usage = caps.supported_usage_flags;
        let format = caps.supported_formats[0].0;

        vulkano::swapchain::Swapchain::new(&device, &surface, 3, format, dimensions, 1,
                                           &usage, &queue, vulkano::swapchain::SurfaceTransform::Identity,
                                           vulkano::swapchain::CompositeAlpha::Opaque,
                                           present, true).expect("failed to create swapchain")
    };


    let cb_pool = vulkano::command_buffer::CommandBufferPool::new(&device, &queue.family())
                                                  .expect("failed to create command buffer pool");


    let depth_buffer = vulkano::image::attachment::AttachmentImage::transient(&device, images[0].dimensions(), vulkano::format::D16Unorm).unwrap();

    let vertex_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                               ::array(&device, teapot::VERTICES.len(),
                                       &vulkano::buffer::Usage::all(), Some(queue.family()))
                                       .expect("failed to create buffer");

    {
        let mut mapping = vertex_buffer.write(Duration::new(0, 0)).unwrap();
        for (o, i) in mapping.iter_mut().zip(teapot::VERTICES.iter()) {
            *o = *i;
        }
    }

    let normals_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                                ::array(&device, teapot::NORMALS.len(),
                                        &vulkano::buffer::Usage::all(), Some(queue.family()))
                                        .expect("failed to create buffer");

    {
        let mut mapping = normals_buffer.write(Duration::new(0, 0)).unwrap();
        for (o, i) in mapping.iter_mut().zip(teapot::NORMALS.iter()) {
            *o = *i;
        }
    }

    let index_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer
                              ::array(&device, teapot::INDICES.len(),
                                      &vulkano::buffer::Usage::all(), Some(queue.family()))
                                      .expect("failed to create buffer");

    {
        let mut mapping = index_buffer.write(Duration::new(0, 0)).unwrap();
        for (o, i) in mapping.iter_mut().zip(teapot::INDICES.iter()) {
            *o = *i;
        }
    }

    // note: this teapot was meant for OpenGL where the origin is at the lower left
    //       instead the origin is at the upper left in vulkan, so we reverse the Y axis
    let proj = cgmath::perspective(cgmath::rad(3.141592 / 2.0), { let d = images[0].dimensions(); d[0] as f32 / d[1] as f32 }, 0.01, 100.0);
    let view = cgmath::Matrix4::look_at(cgmath::Point3::new(0.3, 0.3, 1.0), cgmath::Point3::new(0.0, 0.0, 0.0), cgmath::Vector3::new(0.0, -1.0, 0.0));
    let scale = cgmath::Matrix4::from_scale(0.01);

    let uniform_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::<vs::ty::Data>
                               ::new(&device, &vulkano::buffer::Usage::all(), Some(queue.family()))
                               .expect("failed to create buffer");
    {
        let mut mapping = uniform_buffer.write(Duration::new(0, 0)).unwrap();
        mapping.worldview = (view * scale).into();
        mapping.proj = proj.into();
    }

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

    let descriptor_pool = vulkano::descriptor::descriptor_set::DescriptorPool::new(&device).unwrap();

    mod pipeline_layout {
        pipeline_layout!{
            set0: {
                uniforms: UniformBuffer<::vs::ty::Data>
            }
        }
    }

    let pipeline_layout = pipeline_layout::CustomPipeline::new(&device).unwrap();
    let set = pipeline_layout::set0::Set::new(&descriptor_pool, &pipeline_layout, &pipeline_layout::set0::Descriptors {
        uniforms: &uniform_buffer
    }).unwrap();

    let pipeline = vulkano::pipeline::GraphicsPipeline::new(&device, vulkano::pipeline::GraphicsPipelineParams {
        vertex_input: vulkano::pipeline::vertex::TwoBuffersDefinition::new(),
        vertex_shader: vs.main_entry_point(),
        input_assembly: vulkano::pipeline::input_assembly::InputAssembly::triangle_list(),
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
        layout: &pipeline_layout,
        render_pass: vulkano::framebuffer::Subpass::from(&renderpass, 0).unwrap(),
    }).unwrap();

    let framebuffers = images.iter().map(|image| {
        let attachments = renderpass::AList {
            color: &image,
            depth: &depth_buffer,
        };

        vulkano::framebuffer::Framebuffer::new(&renderpass, (image.dimensions()[0], image.dimensions()[1], 1), attachments).unwrap()
    }).collect::<Vec<_>>();


    let command_buffers = framebuffers.iter().map(|framebuffer| {
        vulkano::command_buffer::PrimaryCommandBufferBuilder::new(&cb_pool).unwrap()
            .draw_inline(&renderpass, &framebuffer, renderpass::ClearValues {
                 color: [0.0, 0.0, 1.0, 1.0].into(),
                 depth: 1.0,
             })
            .draw_indexed(&pipeline, (&vertex_buffer, &normals_buffer), &index_buffer,
                          &vulkano::command_buffer::DynamicState::none(), &set, &())
            .draw_end()
            .build().unwrap()
    }).collect::<Vec<_>>();

    let mut submissions: Vec<Arc<vulkano::command_buffer::Submission>> = Vec::new();

    loop {
        submissions.retain(|s| s.destroying_would_block());

        let image_num = swapchain.acquire_next_image(Duration::from_millis(1)).unwrap();
        submissions.push(vulkano::command_buffer::submit(&command_buffers[image_num], &queue).unwrap());
        swapchain.present(&queue, image_num).unwrap();

        for ev in window.poll_events() {
            match ev {
                winit::Event::Closed => return,
                _ => ()
            }
        }
    }
}
