// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

extern crate cgmath;
extern crate image;
extern crate winit;

#[cfg(windows)]
use winit::os::windows::WindowExt;

#[macro_use]
extern crate vulkano;

use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::mem;
use std::ptr;

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
    let instance = vulkano::instance::Instance::new(Some(&app), &extensions, &[]).expect("failed to create instance");

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
                                                        &device_ext, &[], [(queue, 0.5)].iter().cloned())
                                                                .expect("failed to create device");
    let queue = queues.into_iter().next().unwrap();

    let (swapchain, images) = {
        let caps = surface.get_capabilities(&physical).expect("failed to get surface capabilities");

        let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
        let present = caps.present_modes[0];
        let usage = caps.supported_usage_flags;

        vulkano::swapchain::Swapchain::new(&device, &surface, 3,
                                           vulkano::format::B8G8R8A8Srgb, dimensions, 1,
                                           &usage, &queue, vulkano::swapchain::SurfaceTransform::Identity,
                                           vulkano::swapchain::CompositeAlpha::Opaque,
                                           present, true).expect("failed to create swapchain")
    };


    let cb_pool = vulkano::command_buffer::CommandBufferPool::new(&device, &queue.family())
                                                  .expect("failed to create command buffer pool");





    let vertex_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::<[Vertex]>
                               ::array(&device, 4, &vulkano::buffer::Usage::all(),
                                       Some(queue.family())).expect("failed to create buffer");

    struct Vertex { position: [f32; 2] }
    impl_vertex!(Vertex, position);

    // The buffer that we created contains uninitialized data.
    // In order to fill it with data, we have to *map* it.
    {
        // The `write` function would return `Err` if the buffer was in use by the GPU. This
        // obviously can't happen here, since we haven't ask the GPU to do anything yet.
        let mut mapping = vertex_buffer.write(0).unwrap();
        mapping[0].position = [-0.5, -0.5];
        mapping[1].position = [-0.5,  0.5];
        mapping[2].position = [ 0.5, -0.5];
        mapping[3].position = [ 0.5,  0.5];
    }


    mod vs { include!{concat!(env!("OUT_DIR"), "/shaders/examples/image_vs.glsl")} }
    let vs = vs::Shader::load(&device).expect("failed to create shader module");
    mod fs { include!{concat!(env!("OUT_DIR"), "/shaders/examples/image_fs.glsl")} }
    let fs = fs::Shader::load(&device).expect("failed to create shader module");

    mod renderpass {
        single_pass_renderpass!{
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: B8G8R8A8Srgb,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        }
    }

    let renderpass = renderpass::CustomRenderPass::new(&device).unwrap();

    let texture = vulkano::image::immutable::ImmutableImage::new(&device, vulkano::image::sys::Dimensions::Dim2d { width: 93, height: 93 },
                                                                 vulkano::format::R8G8B8A8Unorm, Some(queue.family())).unwrap();


    let pixel_buffer = {
        let image = image::load_from_memory_with_format(include_bytes!("image_img.png"),
                                                        image::ImageFormat::PNG).unwrap().to_rgba();
        let image_data = image.into_raw().clone();

        // TODO: staging buffer instead
        let pixel_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::<[[u8; 4]]>
                                   ::array(&device, image_data.len(), &vulkano::buffer::Usage::all(),
                                           Some(queue.family())).expect("failed to create buffer");

        {
            let mut mapping = pixel_buffer.write(0).unwrap();
            for (o, i) in mapping.iter_mut().zip(image_data.chunks(4)) {
                o[0] = i[0];
                o[1] = i[1];
                o[2] = i[2];
                o[3] = i[3];
            }
        }

        pixel_buffer
    };


    let sampler = vulkano::sampler::Sampler::new(&device, vulkano::sampler::Filter::Linear,
                                                 vulkano::sampler::Filter::Linear, vulkano::sampler::MipmapMode::Nearest,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 0.0, 1.0, 0.0, 0.0).unwrap();

    let descriptor_pool = vulkano::descriptor_set::DescriptorPool::new(&device).unwrap();
    let descriptor_set_layout = {
        let desc = vulkano::descriptor_set::RuntimeDescriptorSetDesc {
            descriptors: vec![
                vulkano::descriptor_set::DescriptorDesc {
                    binding: 0,
                    ty: vulkano::descriptor_set::DescriptorType::CombinedImageSampler,
                    array_count: 1,
                    stages: vulkano::descriptor_set::ShaderStages::all_graphics(),
                }
            ]
        };

        vulkano::descriptor_set::DescriptorSetLayout::new(&device, desc).unwrap()
    };

    let pipeline_layout = vulkano::descriptor_set::PipelineLayout::new(&device, vulkano::descriptor_set::RuntimeDesc, vec![descriptor_set_layout.clone()]).unwrap();
    let set = vulkano::descriptor_set::DescriptorSet::new(&descriptor_pool, &descriptor_set_layout,
                                                          vec![(0, vulkano::descriptor_set::DescriptorBind::combined_image_sampler(&sampler, &texture))]).unwrap();


    let pipeline = {
        let ia = vulkano::pipeline::input_assembly::InputAssembly {
            topology: vulkano::pipeline::input_assembly::PrimitiveTopology::TriangleStrip,
            primitive_restart_enable: false,
        };

        let raster = Default::default();
        let ms = vulkano::pipeline::multisample::Multisample::disabled();
        let blend = vulkano::pipeline::blend::Blend {
            logic_op: None,
            blend_constants: Some([0.0; 4]),
        };

        let viewports = vulkano::pipeline::viewport::ViewportsState::Fixed {
            data: vec![(
                vulkano::pipeline::viewport::Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [1244.0, 699.0],
                    depth_range: 0.0 .. 1.0
                },
                vulkano::pipeline::viewport::Scissor {
                    origin: [0, 0],
                    dimensions: [1244, 699],
                }
            )],
        };

        vulkano::pipeline::GraphicsPipeline::new(&device, vulkano::pipeline::vertex::SingleBufferDefinition::new(),
                                                 &vs.main_entry_point(), &ia, &viewports,
                                                 &raster, &ms, &blend, &fs.main_entry_point(),
                                                 &pipeline_layout, vulkano::framebuffer::Subpass::from(&renderpass, 0).unwrap())
                                                 .unwrap()
    };

    let framebuffers = images.iter().map(|image| {
        let attachments = renderpass::AList {
            color: &image,
        };

        vulkano::framebuffer::Framebuffer::new(&renderpass, (1244, 699, 1), attachments).unwrap()
    }).collect::<Vec<_>>();


    let command_buffers = framebuffers.iter().map(|framebuffer| {
        vulkano::command_buffer::PrimaryCommandBufferBuilder::new(&cb_pool).unwrap()
            .copy_buffer_to_color_image(&pixel_buffer, &texture)
            //.clear_color_image(&texture, [0.0, 1.0, 0.0, 1.0])
            .draw_inline(&renderpass, &framebuffer, renderpass::ClearValues {
                color: [0.0, 0.0, 1.0, 1.0]
            })
            .draw(&pipeline, &vertex_buffer, &vulkano::command_buffer::DynamicState::none(), set.clone())
            .draw_end()
            .build().unwrap()
    }).collect::<Vec<_>>();

    loop {
        let image_num = swapchain.acquire_next_image(1000000).unwrap();
        vulkano::command_buffer::submit(&command_buffers[image_num], &queue).unwrap();
        swapchain.present(&queue, image_num).unwrap();

        for ev in window.poll_events() {
            match ev {
                winit::Event::Closed => return,
                _ => ()
            }
        }
    }
}
