extern crate kernel32;
extern crate gdi32;
extern crate user32;
extern crate winapi;

#[macro_use]
extern crate vulkano;

use std::sync::Arc;
use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::mem;
use std::ptr;

fn main() {
    // The first step of any vulkan program is to create an instance.
    // TODO: for the moment the AMD driver crashes if you don't pass an ApplicationInfo, but in theory it's optional
    let app = vulkano::instance::ApplicationInfo { application_name: "test", application_version: 1, engine_name: "test", engine_version: 1 };
    let instance = vulkano::instance::Instance::new(Some(&app), &["VK_LAYER_LUNARG_draw_state"]).expect("failed to create instance");

    // We then choose which physical device to use.
    //
    // In a real application, there are three things to take into consideration:
    //
    // - Some devices support some optional features that may be required by your application.
    //   You should filter out the devices that don't support your app.
    //
    // - Not all devices can draw to a certain surface. Once you create your window, you have to
    //   choose a device that is capable of drawing to it.
    //
    // - You probably want to leave the choice between the remaining devices to the user.
    //
    // Here we are just going to use the first device.
    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    // The objective of this example is to draw a triangle on a window. To do so, we first have to
    // create the window in a platform-specific way. Then we create a `Surface` object from it.
    //
    // Surface objects are cross-platform. Once you have a `Surface` everything is the same again.
    let window = unsafe { create_window() };
    let surface = unsafe { vulkano::swapchain::Surface::from_hwnd(&instance, kernel32::GetModuleHandleW(ptr::null()), window).unwrap() };

    // The next step is to choose which queue will execute our draw commands.
    //
    // Devices can provide multiple queues to run things in parallel (for example a draw queue and
    // a compute queue). This is something you have to have to manage manually in Vulkan.
    //
    // We have to specify which queues you are going to use when we create the device, therefore
    // we need to choose that now.
    let queue = physical.queue_families().find(|q| q.supports_graphics() &&
                                                   surface.is_supported(q).unwrap_or(false))
                                                .expect("couldn't find a graphical queue family");

    // Now initializing the device.
    //
    // We have to pass a list of optional Vulkan features that must be enabled. TODO: explain this
    //
    // We also have to pass a list of queues to create and their priorities relative to each other.
    // Since we create one queue, we don't really care about the priority and just pass `0.5`.
    // The list of created queues is returned by the function alongside with the device.
    let (device, queues) = vulkano::device::Device::new(&physical, physical.supported_features(),
                                                        [(queue, 0.5)].iter().cloned(),
                                                        &["VK_LAYER_LUNARG_draw_state"])
                                                                .expect("failed to create device");

    // Since we can request multiple queues, the `queues` variable is a `Vec`. Our actual queue
    // is the first element.
    let queue = queues.into_iter().next().unwrap();

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating
    // a swapchain allocates the color buffers that will contain the image that will ultimately
    // be visible on the screen. These images are returned alongside with the swapchain.
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

    // We now create a buffer that will store the shape of our triangle.
    //
    // The first parameter is the device to use, and the second parameter is how the buffer will
    // be used on the GPU side. This is a vertex buffer, and we need to tell this to the
    // implementation.
    //
    // The third parameter is where to get the memory where the buffer will be stored. It is very
    // important as it also determines the way you are going to access and modify your buffer.
    // Here we just ask for a basic host visible memory.
    //
    // Note that to store immutable data, the best way is to create two buffers. One buffer on
    // the CPU and one buffer on the GPU. We then write our data to the buffer on the CPU and
    // ask the GPU to copy it to the real buffer. This way the data is located on the most
    // efficient memory possible.
    let vertex_buffer: Arc<vulkano::buffer::Buffer<[Vertex; 3], _>> =
                                vulkano::buffer::Buffer::new(&device, &vulkano::buffer::Usage::all(),
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
        mapping[0].position = [-0.5, -0.25];
        mapping[1].position = [0.0, 0.5];
        mapping[2].position = [0.25, -0.1];
    }

    // The next step is to create the shaders.
    //
    // The shader creation API provided by the vulkano library is unsafe, for various reasons.
    //
    // Instead, in our build script we used the `vulkano-shaders` crate to parse our shaders at
    // compile time and provide a safe wrapper over vulkano's API. You can find the shaders'
    // source code in the `triangle_fs.glsl` and `triangle_vs.glsl` files.
    //
    // The code generated by the build script created a struct named `TriangleShader`, which we
    // can now use to load the shader.
    //
    // Because of some restrictions with the `include!` macro, we need to use a module.
    mod vs { include!{concat!(env!("OUT_DIR"), "/examples-triangle_vs.rs")} }
    let vs = vs::TriangleShader::load(&device).expect("failed to create shader module");
    mod fs { include!{concat!(env!("OUT_DIR"), "/examples-triangle_fs.rs")} }
    let fs = fs::TriangleShader::load(&device).expect("failed to create shader module");

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitely does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // We are going to create a command buffer below. Command buffers need to be allocated
    // from a *command buffer pool*, so we create the pool.
    let cb_pool = vulkano::command_buffer::CommandBufferPool::new(&device, &queue.lock().unwrap().family())
                                                  .expect("failed to create command buffer pool");

    // We are going to draw on the images returned when creating the swapchain. To do so, we must
    // convert them into *image views*. TODO: explain more
    let images = images.into_iter().map(|image| {
        let image = image.transition(vulkano::image::Layout::PresentSrc, &cb_pool,
                                     &mut queue.lock().unwrap()).unwrap();
        vulkano::image::ImageView::new(&image).expect("failed to create image view")
    }).collect::<Vec<_>>();

    // The next step is to create a *renderpass*, which is an object that describes where the
    // output of the graphics pipeline will go. It describes the layout of the images
    // where the colors, depth and/or stencil information will be written.
    let renderpass = renderpass!{
        device: &device,
        attachments: {
            color [Clear]
        }
    }.unwrap();

    let pipeline: Arc<vulkano::pipeline::GraphicsPipeline<Arc<vulkano::buffer::Buffer<[Vertex; 3], _>>>> = {
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
                                                 &renderpass.subpass(0).unwrap()).unwrap()
    };

    // The renderpass we created above only describes the layout of our framebuffers. We also need
    // to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.
    let framebuffers = images.iter().map(|image| {
        vulkano::framebuffer::Framebuffer::new(&renderpass, (1244, 699, 1), image).unwrap()
    }).collect::<Vec<_>>();

    // The final initialization step is to create a command buffer.
    //
    // A command buffer contains a list of commands that the GPU must execute. This can include
    // transfers between buffers, clearing images or attachments, etc. and draw commands. Here we
    // create a command buffer with a single command: drawing the triangle. The color attachment
    // is also cleared at the start.
    //
    // Since we have several images to draw on, we are also going to create one command buffer for
    // each image.
    let command_buffers = framebuffers.iter().map(|framebuffer| {
        vulkano::command_buffer::PrimaryCommandBufferBuilder::new(&cb_pool).unwrap()
            .draw_inline(&renderpass, &framebuffer, [0.0, 0.0, 1.0, 1.0])
            .draw(&pipeline, vertex_buffer.clone(), &vulkano::command_buffer::DynamicState::none())
            .draw_end()
            .build().unwrap()
    }).collect::<Vec<_>>();

    // Initialization is finally finished!

    // Note that the only thing we need now are the `command_buffers` and `swapchain` variables.
    // Everything else is kept alive internally with `Arc`s (even the vertex buffer for example),
    // so the only variable that we need is this one.

    loop {
        // Before we can draw on the output, we have to *acquire* an image from the swapchain.
        // This operation returns the index of the image that we are allowed to draw upon..
        let image_num = swapchain.acquire_next_image().unwrap();

        // Our queue is wrapped around a `Mutex`, so we have to lock it.
        let mut queue = queue.lock().unwrap();

        // In order to draw, all we need to do is submit the command buffer to the queue.
        command_buffers[image_num].submit(&mut queue).unwrap();

        // The color output should now contain our triangle. But in order to show it on the
        // screen, we have to *present* the image. Depending on the presentation mode, this may
        // be shown immediatly or on the next redraw.
        swapchain.present(&mut queue, image_num).unwrap();

        // In a real application we want to submit things to the same queue in parallel, so we
        // shouldn't keep it locked too long.
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