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

mod vs { include!{concat!(env!("OUT_DIR"), "/shaders/examples/teapot_vs.glsl")} }
mod fs { include!{concat!(env!("OUT_DIR"), "/shaders/examples/teapot_fs.glsl")} }

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    // TODO: for the moment the AMD driver crashes if you don't pass an ApplicationInfo, but in theory it's optional
    let app = vulkano::instance::ApplicationInfo { application_name: "test", application_version: 1, engine_name: "test", engine_version: 1 };
    let extensions = vulkano::instance::InstanceExtensions {
        khr_surface: true,
        khr_swapchain: true,
        khr_win32_surface: true,
        .. vulkano::instance::InstanceExtensions::none()
    };
    let instance = vulkano::instance::Instance::new(Some(&app), &extensions, &["VK_LAYER_LUNARG_standard_validation"]).expect("failed to create instance");

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
    };

    let (device, queues) = vulkano::device::Device::new(&physical, physical.supported_features(),
                                                        &device_ext, &["VK_LAYER_LUNARG_standard_validation"], [(queue, 0.5)].iter().cloned())
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


    let depth_buffer = vulkano::image::Image::<vulkano::image::Type2d, _, _>::new(&device, &vulkano::image::Usage::all(),
                                                  vulkano::memory::DeviceLocal, &queue,
                                                  vulkano::format::D16Unorm, images[0].dimensions(), (), 1).unwrap();
    let depth_buffer = depth_buffer.transition(vulkano::image::Layout::DepthStencilAttachmentOptimal, &cb_pool, &queue).unwrap();
    let depth_buffer = vulkano::image::ImageView::new(&depth_buffer).expect("failed to create image view");

    let vertex_buffer = vulkano::buffer::Buffer
                               ::array(&device, teapot::VERTICES.len(),
                                       &vulkano::buffer::Usage::all(),
                                       vulkano::memory::HostVisible, &queue)
                                       .expect("failed to create buffer");

    {
        let mut mapping = vertex_buffer.try_write().unwrap();
        for (o, i) in mapping.iter_mut().zip(teapot::VERTICES.iter()) {
            *o = *i;
        }
    }

    let normals_buffer = vulkano::buffer::Buffer
                                ::array(&device, teapot::NORMALS.len(),
                                        &vulkano::buffer::Usage::all(),
                                        vulkano::memory::HostVisible, &queue)
                                        .expect("failed to create buffer");

    {
        let mut mapping = normals_buffer.try_write().unwrap();
        for (o, i) in mapping.iter_mut().zip(teapot::NORMALS.iter()) {
            *o = *i;
        }
    }

    let index_buffer = vulkano::buffer::Buffer
                              ::array(&device, teapot::INDICES.len(),
                                      &vulkano::buffer::Usage::all(),
                                      vulkano::memory::HostVisible, &queue)
                                      .expect("failed to create buffer");

    {
        let mut mapping = index_buffer.try_write().unwrap();
        for (o, i) in mapping.iter_mut().zip(teapot::INDICES.iter()) {
            *o = *i;
        }
    }

    // note: this teapot was meant for OpenGL where the origin is at the lower left
    //       instead the origin is at the upper left in vulkan, so we reverse the Y axis
    let proj = cgmath::perspective(cgmath::rad(3.141592 / 2.0), { let d = images[0].dimensions(); d[0] as f32 / d[1] as f32 }, 0.01, 100.0);
    let view = cgmath::Matrix4::look_at(cgmath::Point3::new(0.3, 0.3, 1.0), cgmath::Point3::new(0.0, 0.0, 0.0), cgmath::Vector3::new(0.0, -1.0, 0.0));
    let scale = cgmath::Matrix4::from_scale(0.01);

    let uniform_buffer = vulkano::buffer::Buffer::<vs::ty::Data, _>
                               ::new(&device, &vulkano::buffer::Usage::all(),
                                     vulkano::memory::HostVisible, &queue)
                               .expect("failed to create buffer");
    {
        let mut mapping = uniform_buffer.try_write().unwrap();
        mapping.worldview = (view * scale).into();
        mapping.proj = proj.into();
    }

    let vs = vs::Shader::load(&device).expect("failed to create shader module");
    let fs = fs::Shader::load(&device).expect("failed to create shader module");

    let images = images.into_iter().map(|image| {
        let image = image.transition(vulkano::image::Layout::PresentSrc, &cb_pool,
                                     &queue).unwrap();
        vulkano::image::ImageView::new(&image).expect("failed to create image view")
    }).collect::<Vec<_>>();

    mod renderpass {
        single_pass_renderpass!{
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: B8G8R8A8Srgb,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: D16Unorm,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        }
    }

    let renderpass = vulkano::framebuffer::RenderPass::new(&device, renderpass::Layout).unwrap();

    let descriptor_pool = vulkano::descriptor_set::DescriptorPool::new(&device).unwrap();
    let descriptor_set_layout = vulkano::descriptor_set::DescriptorSetLayout::new(&device, vs::Set0).unwrap();

    mod pipeline_layout { pipeline_from_sets!(::vs::Set0); }
    let pipeline_layout = vulkano::descriptor_set::PipelineLayout::new(&device, pipeline_layout::Layout, (descriptor_set_layout.clone(),)).unwrap();
    let set = vulkano::descriptor_set::DescriptorSet::new(&descriptor_pool, &descriptor_set_layout,
                                                          uniform_buffer.clone() as std::sync::Arc<_>).unwrap();


    let pipeline = {
        let ia = vulkano::pipeline::input_assembly::InputAssembly::triangle_list();
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
                                                 &pipeline_layout, vulkano::framebuffer::Subpass::from(&renderpass, 0).unwrap())
                                                 .unwrap()
    };

    let framebuffers = images.iter().map(|image| {
        vulkano::framebuffer::Framebuffer::new(&renderpass, (1244, 699, 1), (image.clone() as std::sync::Arc<_>, depth_buffer.clone() as std::sync::Arc<_>)).unwrap()
    }).collect::<Vec<_>>();


    let command_buffers = framebuffers.iter().map(|framebuffer| {
        vulkano::command_buffer::PrimaryCommandBufferBuilder::new(&cb_pool).unwrap()
            .draw_inline(&renderpass, &framebuffer, ([0.0, 0.0, 1.0, 1.0], 1.0))
            .draw_indexed(&pipeline, (vertex_buffer.clone(), normals_buffer.clone()), &index_buffer, &vulkano::command_buffer::DynamicState::none(), set.clone())
            .draw_end()
            .build().unwrap()
    }).collect::<Vec<_>>();

    let mut submissions: Vec<vulkano::command_buffer::Submission> = Vec::new();

    loop {
        submissions.retain(|s| !s.destroying_would_block());

        let image_num = swapchain.acquire_next_image(1000000).unwrap();
        submissions.push(vulkano::command_buffer::submit(&command_buffers[image_num], &queue).unwrap());
        swapchain.present(&queue, image_num).unwrap();

        for ev in window.poll_events() {
            match ev {
                winit::Event::Closed => break,
                _ => ()
            }
        }
    }
}
