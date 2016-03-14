#[macro_use]
extern crate vulkano;
extern crate winit;

#[cfg(windows)]
use winit::os::windows::WindowExt;

use std::sync::Arc;
use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::mem;
use std::ptr;

fn main() {
    // The first step of any vulkan program is to create an instance.
    // TODO: for the moment the AMD driver crashes if you don't pass an ApplicationInfo, but in theory it's optional
    let app = vulkano::instance::ApplicationInfo { application_name: "test", application_version: 1, engine_name: "test", engine_version: 1 };
    let extensions = vulkano::instance::InstanceExtensions {
        khr_surface: true,
        khr_swapchain: true,
        khr_win32_surface: true,
        .. vulkano::instance::InstanceExtensions::none()
    };
    let instance = vulkano::instance::Instance::new(Some(&app), &["VK_LAYER_LUNARG_draw_state"], &extensions).expect("failed to create instance");

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
    let window = winit::WindowBuilder::new().build().unwrap();
    let surface = unsafe { vulkano::swapchain::Surface::from_hwnd(&instance, ptr::null() as *const () /* FIXME */, window.get_hwnd()).unwrap() };

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
                                           vulkano::format::B8G8R8A8Srgb, dimensions, 1,
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
    let vertex_buffer = vulkano::buffer::Buffer::<[Vertex; 3], _>
                               ::new(&device, &vulkano::buffer::Usage::all(),
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
    mod vs { include!{concat!(env!("OUT_DIR"), "/shaders/examples/triangle_vs.glsl")} }
    let vs = vs::Shader::load(&device).expect("failed to create shader module");
    mod fs { include!{concat!(env!("OUT_DIR"), "/shaders/examples/triangle_fs.glsl")} }
    let fs = fs::Shader::load(&device).expect("failed to create shader module");

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitely does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // We are going to create a command buffer below. Command buffers need to be allocated
    // from a *command buffer pool*, so we create the pool.
    let cb_pool = vulkano::command_buffer::CommandBufferPool::new(&device, &queue.family())
                                                  .expect("failed to create command buffer pool");

    // We are going to draw on the images returned when creating the swapchain. To do so, we must
    // convert them into *image views*. TODO: explain more
    let images = images.into_iter().map(|image| {
        let image = image.transition(vulkano::image::Layout::PresentSrc, &cb_pool,
                                     &queue).unwrap();
        vulkano::image::ImageView::new(&image).expect("failed to create image view")
    }).collect::<Vec<_>>();

    // The next step is to create a *renderpass*, which is an object that describes where the
    // output of the graphics pipeline will go. It describes the layout of the images
    // where the colors, depth and/or stencil information will be written.
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

    let renderpass = vulkano::framebuffer::UnsafeRenderPass::new(&device, renderpass::Layout).unwrap();

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
                                                 &vulkano::descriptor_set::PipelineLayout::new(&device, vulkano::descriptor_set::EmptyPipelineDesc, ()).unwrap(),
                                                 renderpass.subpass(0).unwrap()).unwrap()
    };

    // The renderpass we created above only describes the layout of our framebuffers. We also need
    // to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.
    let framebuffers = images.iter().map(|image| {
        vulkano::framebuffer::Framebuffer::new(&renderpass, (1244, 699, 1), (image.clone() as Arc<_>,)).unwrap()
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
            .draw_inline(&renderpass, &framebuffer, ([0.0, 0.0, 1.0, 1.0],))
            .draw(&pipeline, vertex_buffer.clone(), &vulkano::command_buffer::DynamicState::none(), ())
            .draw_end()
            .build().unwrap()
    }).collect::<Vec<_>>();

    // Initialization is finally finished!

    // Note that the only thing we need now are the `command_buffers` and `swapchain` variables.
    // Everything else is kept alive internally with `Arc`s (even the vertex buffer for example),
    // so the only variable that we need is this one.

    let mut submissions: Vec<vulkano::command_buffer::Submission> = Vec::new();

    loop {
        submissions.retain(|s| !s.destroying_would_block());

        // Before we can draw on the output, we have to *acquire* an image from the swapchain.
        // This operation returns the index of the image that we are allowed to draw upon..
        let image_num = swapchain.acquire_next_image(1000000).unwrap();

        // In order to draw, all we need to do is submit the command buffer to the queue.
        submissions.push(vulkano::command_buffer::submit(&command_buffers[image_num], &queue).unwrap());

        // The color output should now contain our triangle. But in order to show it on the
        // screen, we have to *present* the image. Depending on the presentation mode, this may
        // be shown immediatly or on the next redraw.
        swapchain.present(&queue, image_num).unwrap();

        for ev in window.poll_events() {
            match ev {
                winit::Event::Closed => break,
                _ => ()
            }
        }
    }
}
