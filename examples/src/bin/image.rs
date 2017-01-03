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

#[macro_use]
extern crate vulkano;
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;

use std::time::Duration;

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    let extensions = vulkano_win::required_extensions();
    let instance = vulkano::instance::Instance::new(None, &extensions, &[]).expect("failed to create instance");

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

        vulkano::swapchain::Swapchain::new(&device, &window.surface(), caps.min_image_count,
                                           vulkano::format::B8G8R8A8Srgb, dimensions, 1,
                                           &usage, &queue, vulkano::swapchain::SurfaceTransform::Identity,
                                           vulkano::swapchain::CompositeAlpha::Opaque,
                                           present, true, None).expect("failed to create swapchain")
    };


    #[derive(Debug, Clone)]
    struct Vertex { position: [f32; 2] }
    impl_vertex!(Vertex, position);

    let vertex_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::<[Vertex]>
                               ::from_iter(&device, &vulkano::buffer::BufferUsage::all(),
                                       Some(queue.family()), [
                                           Vertex { position: [-0.5, -0.5 ] },
                                           Vertex { position: [-0.5,  0.5 ] },
                                           Vertex { position: [ 0.5, -0.5 ] },
                                           Vertex { position: [ 0.5,  0.5 ] },
                                       ].iter().cloned()).expect("failed to create buffer");

    mod vs { include!{concat!(env!("OUT_DIR"), "/shaders/src/bin/image_vs.glsl")} }
    let vs = vs::Shader::load(&device).expect("failed to create shader module");
    mod fs { include!{concat!(env!("OUT_DIR"), "/shaders/src/bin/image_fs.glsl")} }
    let fs = fs::Shader::load(&device).expect("failed to create shader module");

    mod renderpass {
        single_pass_renderpass!{
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: ::vulkano::format::B8G8R8A8Srgb,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        }
    }

    let renderpass = renderpass::CustomRenderPass::new(&device, &renderpass::Formats {
        color: (vulkano::format::B8G8R8A8Srgb, 1)
    }).unwrap();

    let texture = vulkano::image::immutable::ImmutableImage::new(&device, vulkano::image::Dimensions::Dim2d { width: 93, height: 93 },
                                                                 vulkano::format::R8G8B8A8Unorm, Some(queue.family())).unwrap();


    let pixel_buffer = {
        let image = image::load_from_memory_with_format(include_bytes!("image_img.png"),
                                                        image::ImageFormat::PNG).unwrap().to_rgba();
        let image_data = image.into_raw().clone();

        let image_data_chunks = image_data.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]);

        // TODO: staging buffer instead
        vulkano::buffer::cpu_access::CpuAccessibleBuffer::<[[u8; 4]]>
            ::from_iter(&device, &vulkano::buffer::BufferUsage::all(),
                        Some(queue.family()), image_data_chunks)
                        .expect("failed to create buffer")
    };


    let sampler = vulkano::sampler::Sampler::new(&device, vulkano::sampler::Filter::Linear,
                                                 vulkano::sampler::Filter::Linear, vulkano::sampler::MipmapMode::Nearest,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 0.0, 1.0, 0.0, 0.0).unwrap();

    let descriptor_pool = vulkano::descriptor::descriptor_set::DescriptorPool::new(&device);
    mod pipeline_layout {
        pipeline_layout!{
            set0: {
                tex: CombinedImageSampler
            }
        }
    }

    let pipeline_layout = pipeline_layout::CustomPipeline::new(&device).unwrap();
    let set = pipeline_layout::set0::Set::new(&descriptor_pool, &pipeline_layout, &pipeline_layout::set0::Descriptors {
        tex: (&sampler, &texture)
    });


    let pipeline = vulkano::pipeline::GraphicsPipeline::new(&device, vulkano::pipeline::GraphicsPipelineParams {
        vertex_input: vulkano::pipeline::vertex::SingleBufferDefinition::new(),
        vertex_shader: vs.main_entry_point(),
        input_assembly: vulkano::pipeline::input_assembly::InputAssembly {
            topology: vulkano::pipeline::input_assembly::PrimitiveTopology::TriangleStrip,
            primitive_restart_enable: false,
        },
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
        depth_stencil: vulkano::pipeline::depth_stencil::DepthStencil::disabled(),
        blend: vulkano::pipeline::blend::Blend::pass_through(),
        layout: &pipeline_layout,
        render_pass: vulkano::framebuffer::Subpass::from(&renderpass, 0).unwrap(),
    }).unwrap();

    let framebuffers = images.iter().map(|image| {
        let attachments = renderpass::AList {
            color: &image,
        };

        vulkano::framebuffer::Framebuffer::new(&renderpass, [images[0].dimensions()[0], images[0].dimensions()[1], 1], attachments).unwrap()
    }).collect::<Vec<_>>();


    let command_buffers = framebuffers.iter().map(|framebuffer| {
        vulkano::command_buffer::PrimaryCommandBufferBuilder::new(&device, queue.family())
            .copy_buffer_to_color_image(&pixel_buffer, &texture, 0, 0 .. 1, [0, 0, 0],
                                        [texture.dimensions().width(), texture.dimensions().height(), 1])
            //.clear_color_image(&texture, [0.0, 1.0, 0.0, 1.0])
            .draw_inline(&renderpass, &framebuffer, renderpass::ClearValues {
                color: [0.0, 0.0, 1.0, 1.0]
            })
            .draw(&pipeline, &vertex_buffer, &vulkano::command_buffer::DynamicState::none(),
                  &set, &())
            .draw_end()
            .build()
    }).collect::<Vec<_>>();

    loop {
        let image_num = swapchain.acquire_next_image(Duration::new(10, 0)).unwrap();
        vulkano::command_buffer::submit(&command_buffers[image_num], &queue).unwrap();
        swapchain.present(&queue, image_num).unwrap();

        for ev in window.window().poll_events() {
            match ev {
                winit::Event::Closed => return,
                _ => ()
            }
        }
    }
}
