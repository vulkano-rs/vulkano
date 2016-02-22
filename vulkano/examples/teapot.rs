extern crate kernel32;
extern crate gdi32;
extern crate user32;
extern crate winapi;
extern crate cgmath;

#[macro_use]
extern crate vulkano;

#[path = "support/teapot.rs"]
mod teapot;

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

    let window = unsafe { create_window() };
    let surface = unsafe { vulkano::swapchain::Surface::from_hwnd(&instance, kernel32::GetModuleHandleW(ptr::null()), window).unwrap() };

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


    let depth_buffer = vulkano::image::Image::<vulkano::image::Type2d, vulkano::formats::D16Unorm, _>::new(&device, &vulkano::image::Usage::all(),
                                                  vulkano::memory::DeviceLocal, &queue,
                                                  images[0].dimensions(), (), 1).unwrap();
    let depth_buffer = depth_buffer.transition(vulkano::image::Layout::DepthStencilAttachmentOptimal, &cb_pool, &mut queue.lock().unwrap()).unwrap();
    let depth_buffer = vulkano::image::ImageView::new(&depth_buffer).expect("failed to create image view");

    let vertex_buffer = vulkano::buffer::Buffer::<[teapot::Vertex], _>
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

    let normals_buffer = vulkano::buffer::Buffer::<[teapot::Normal], _>
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

    let index_buffer = vulkano::buffer::Buffer::<[u16], _>
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

    let uniform_buffer = vulkano::buffer::Buffer::<([[f32; 4]; 4], [[f32; 4]; 4]), _>
                               ::new(&device, &vulkano::buffer::Usage::all(),
                                     vulkano::memory::HostVisible, &queue)
                               .expect("failed to create buffer");
    {
        let mut mapping = uniform_buffer.try_write().unwrap();
        mapping.0 = (view * scale).into();
        mapping.1 = proj.into();
    }

    mod vs { include!{concat!(env!("OUT_DIR"), "/examples-teapot_vs.rs")} }
    let vs = vs::TeapotShader::load(&device).expect("failed to create shader module");
    mod fs { include!{concat!(env!("OUT_DIR"), "/examples-teapot_fs.rs")} }
    let fs = fs::TeapotShader::load(&device).expect("failed to create shader module");

    let images = images.into_iter().map(|image| {
        let image = image.transition(vulkano::image::Layout::PresentSrc, &cb_pool,
                                     &mut queue.lock().unwrap()).unwrap();
        vulkano::image::ImageView::new(&image).expect("failed to create image view")
    }).collect::<Vec<_>>();

    let renderpass = single_pass_renderpass!{
        device: &device,
        attachments: {
            color [Clear]
        }
    }.unwrap();



    let descriptor_pool = vulkano::descriptor_set::DescriptorPool::new(&device).unwrap();

    let (pipeline_layout, set) = {
        let layout1 = vulkano::descriptor_set::DescriptorSetLayout::new(&device, Default::default()).unwrap();
        let pipeline_layout = vulkano::descriptor_set::PipelineLayout::new(&device, Default::default(), (layout1.clone(), ())).unwrap();
        let set1 = vulkano::descriptor_set::DescriptorSet::new(&descriptor_pool, &layout1, uniform_buffer.clone() as std::sync::Arc<_>).unwrap();
        (pipeline_layout, set1)
    };

    let pipeline = {
        let ia = vulkano::pipeline::input_assembly::InputAssembly {
            topology: vulkano::pipeline::input_assembly::PrimitiveTopology::TriangleList,
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
        vulkano::framebuffer::Framebuffer::new(&renderpass, (1244, 699, 1), (image.clone() as std::sync::Arc<_>, depth_buffer.clone() as std::sync::Arc<_>)).unwrap()
    }).collect::<Vec<_>>();


    let command_buffers = framebuffers.iter().map(|framebuffer| {
        vulkano::command_buffer::PrimaryCommandBufferBuilder::new(&cb_pool).unwrap()
            .draw_inline(&renderpass, &framebuffer, ([0.0, 0.0, 1.0, 1.0], 1.0))
            .draw_indexed(&pipeline, (vertex_buffer.clone(), normals_buffer.clone()), &index_buffer, &vulkano::command_buffer::DynamicState::none(), (set.clone(), ()))
            .draw_end()
            .build().unwrap()
    }).collect::<Vec<_>>();

    loop {
        let image_num = swapchain.acquire_next_image(1000000).unwrap();
        let mut queue = queue.lock().unwrap();
        command_buffers[image_num].submit(&mut queue).unwrap();
        swapchain.present(&mut queue, image_num).unwrap();
        drop(queue);

        unsafe {
            let mut msg = mem::uninitialized();
            if user32::GetMessageW(&mut msg, ptr::null_mut(), 0, 0) == 0 {
                break;
            }

            user32::TranslateMessage(&msg);
            user32::DispatchMessageW(&msg);
        }
    }
}





unsafe fn create_window() -> winapi::HWND {
    let class_name = register_window_class();

    let title: Vec<u16> = vec![b'V' as u16, b'u' as u16, b'l' as u16, b'k' as u16,
                               b'a' as u16, b'n' as u16, 0];

    user32::CreateWindowExW(winapi::WS_EX_APPWINDOW | winapi::WS_EX_WINDOWEDGE, class_name.as_ptr(),
                            title.as_ptr() as winapi::LPCWSTR,
                            winapi::WS_OVERLAPPEDWINDOW | winapi::WS_CLIPSIBLINGS |
                            winapi::WS_VISIBLE,
                            winapi::CW_USEDEFAULT, winapi::CW_USEDEFAULT,
                            winapi::CW_USEDEFAULT, winapi::CW_USEDEFAULT,
                            ptr::null_mut(), ptr::null_mut(),
                            kernel32::GetModuleHandleW(ptr::null()),
                            ptr::null_mut())
}

unsafe fn register_window_class() -> Vec<u16> {
    let class_name: Vec<u16> = OsStr::new("Window Class").encode_wide().chain(Some(0).into_iter())
                                                         .collect::<Vec<u16>>();

    let class = winapi::WNDCLASSEXW {
        cbSize: mem::size_of::<winapi::WNDCLASSEXW>() as winapi::UINT,
        style: winapi::CS_HREDRAW | winapi::CS_VREDRAW | winapi::CS_OWNDC,
        lpfnWndProc: Some(callback),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: kernel32::GetModuleHandleW(ptr::null()),
        hIcon: ptr::null_mut(),
        hCursor: ptr::null_mut(),
        hbrBackground: ptr::null_mut(),
        lpszMenuName: ptr::null(),
        lpszClassName: class_name.as_ptr(),
        hIconSm: ptr::null_mut(),
    };

    user32::RegisterClassExW(&class);
    class_name
}

unsafe extern "system" fn callback(window: winapi::HWND, msg: winapi::UINT,
                                   wparam: winapi::WPARAM, lparam: winapi::LPARAM)
                                   -> winapi::LRESULT
{
    user32::DefWindowProcW(window, msg, wparam, lparam)
}