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
    let instance = vulkano::instance::Instance::new(Some(&app), &[]).expect("failed to create instance");

    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let window = winit::WindowBuilder::new().build().unwrap();
    let surface = unsafe { vulkano::swapchain::Surface::from_hwnd(&instance, ptr::null() as *const () /* FIXME */, window.get_hwnd()).unwrap() };

    let queue = physical.queue_families().find(|q| q.supports_graphics() &&
                                                   surface.is_supported(q).unwrap_or(false))
                                                .expect("couldn't find a graphical queue family");

    let (device, queues) = vulkano::device::Device::new(&physical, physical.supported_features(),
                                                        [(queue, 0.5)].iter().cloned(), &[])
                                                                .expect("failed to create device");
    let queue = queues.into_iter().next().unwrap();

    let (swapchain, images) = {
        let caps = surface.get_capabilities(&physical).expect("failed to get surface capabilities");

        let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
        let present = caps.present_modes[0];
        let usage = caps.supported_usage_flags;

        vulkano::swapchain::Swapchain::new(&device, &surface, 3,
                                           vulkano::formats::B8G8R8A8Srgb, dimensions, 1,
                                           &usage, &queue, vulkano::swapchain::SurfaceTransform::Identity,
                                           vulkano::swapchain::CompositeAlpha::Opaque,
                                           present, true).expect("failed to create swapchain")
    };


    let cb_pool = vulkano::command_buffer::CommandBufferPool::new(&device, &queue.lock().unwrap().family())
                                                  .expect("failed to create command buffer pool");





    let vertex_buffer = vulkano::buffer::Buffer::<[Vertex], _>
                               ::array(&device, 4, &vulkano::buffer::Usage::all(),
                                       vulkano::memory::HostVisible, &queue)
                                       .expect("failed to create buffer");

    struct Vertex { position: [f32; 2] }
    impl_vertex!(Vertex, position);

    // The buffer that we created contains uninitialized data.
    // In order to fill it with data, we have to *map* it.
    {
        // The `try_write` function would return `None` if the buffer was in use by the GPU. This
        // obviously can't happen here, since we haven't ask the GPU to do anything yet.
        let mut mapping = vertex_buffer.try_write().unwrap();
        mapping[0].position = [-0.5, -0.5];
        mapping[1].position = [-0.5,  0.5];
        mapping[2].position = [ 0.5, -0.5];
        mapping[3].position = [ 0.5,  0.5];
    }


    mod vs { include!{concat!(env!("OUT_DIR"), "/examples-image_vs.rs")} }
    let vs = vs::ImageShader::load(&device).expect("failed to create shader module");
    mod fs { include!{concat!(env!("OUT_DIR"), "/examples-image_fs.rs")} }
    let fs = fs::ImageShader::load(&device).expect("failed to create shader module");

    let renderpass = single_pass_renderpass!{
        device: &device,
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
    }.unwrap();

    let texture = vulkano::image::Image::<vulkano::image::Type2d, _, _>::new(&device, &vulkano::image::Usage::all(),
                                                  vulkano::memory::DeviceLocal, &queue,
                                                  vulkano::formats::R8G8B8A8Unorm, [93, 93], (), 1).unwrap();
    let texture = texture.transition(vulkano::image::Layout::ShaderReadOnlyOptimal, &cb_pool, &mut queue.lock().unwrap()).unwrap();
    let texture_view = vulkano::image::ImageView::new(&texture).expect("failed to create image view");


    let pixel_buffer = {
        let image = image::load_from_memory_with_format(include_bytes!("image_img.png"),
                                                        image::ImageFormat::PNG).unwrap().to_rgba();
        let image_data = image.into_raw().clone();

        let pixel_buffer = vulkano::buffer::Buffer::<[[u8; 4]], _>
                               ::array(&device, image_data.len(), &vulkano::buffer::Usage::all(),
                                       vulkano::memory::HostVisible, &queue)
                                       .expect("failed to create buffer");

        {
            let mut mapping = pixel_buffer.try_write().unwrap();
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
                                                          vec![(0, vulkano::descriptor_set::DescriptorBind::CombinedImageSampler(sampler.clone(), texture_view.clone(), vulkano::image::Layout::ShaderReadOnlyOptimal))]).unwrap();


    let images = images.into_iter().map(|image| {
        let image = image.transition(vulkano::image::Layout::PresentSrc, &cb_pool,
                                     &mut queue.lock().unwrap()).unwrap();
        vulkano::image::ImageView::new(&image).expect("failed to create image view")
    }).collect::<Vec<_>>();

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

        vulkano::pipeline::GraphicsPipeline::new(&device, &vs.main_entry_point(), &ia, &viewports,
                                                 &raster, &ms, &blend, &fs.main_entry_point(),
                                                 &pipeline_layout, &renderpass.subpass(0).unwrap())
                                                 .unwrap()
    };

    let framebuffers = images.iter().map(|image| {
        vulkano::framebuffer::Framebuffer::new(&renderpass, (1244, 699, 1), (image.clone() as std::sync::Arc<_>,)).unwrap()
    }).collect::<Vec<_>>();


    let command_buffers = framebuffers.iter().map(|framebuffer| {
        vulkano::command_buffer::PrimaryCommandBufferBuilder::new(&cb_pool).unwrap()
            .copy_buffer_to_color_image(&pixel_buffer, &texture)
            //.clear_color_image(&texture, [0.0, 1.0, 0.0, 1.0])
            .draw_inline(&renderpass, &framebuffer, ([0.0, 0.0, 1.0, 1.0],))
            .draw(&pipeline, vertex_buffer.clone(), &vulkano::command_buffer::DynamicState::none(), set.clone())
            .draw_end()
            .build().unwrap()
    }).collect::<Vec<_>>();

    loop {
        let image_num = swapchain.acquire_next_image(1000000).unwrap();
        let mut queue = queue.lock().unwrap();
        command_buffers[image_num].submit(&mut queue).unwrap();
        swapchain.present(&mut queue, image_num).unwrap();
        drop(queue);

        for ev in window.poll_events() {
            match ev {
                winit::Event::Closed => break,
                _ => ()
            }
        }
    }
}
