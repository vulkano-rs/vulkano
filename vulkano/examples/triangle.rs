extern crate vulkano;

use std::thread;

fn main() {
    // The first step of any vulkan program is to create an instance.
    let instance = vulkano::instance::Instance::new(None, None).expect("failed to create instance");

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

    // Vulkan provides a cross-platform API to draw on the whole monitor. In order to avoid the
    // boiling plate of creating a window, we are going to use it.
    let surface = {
        let display = vulkano::swapchain::Display::enumerate(&physical).unwrap().next().unwrap();
        let display_mode = display.display_modes().unwrap().next().unwrap();;
        let plane = vulkano::swapchain::DisplayPlane::enumerate(&physical).unwrap().next().unwrap();
        vulkano::swapchain::Surface::from_display_mode(&display_mode, &plane).unwrap()
    };

    // The next step is to choose which queue will execute our draw commands.
    //
    // Devices can provide multiple queues to run things in parallel (for example a draw queue and
    // a compute queue). This is something you have to have to manage manually in Vulkan.
    //
    // We have to specify which queues you are going to use when you create the device, therefore
    // we need to choose that now.
    let queue = physical.queue_families().find(|q| q.supports_graphics() &&
                                                   surface.is_supported(q).unwrap_or(false))
                                                .expect("couldn't find a graphical queue family");

    // Now initializing the device.
    //
    // We have to pass a list of optional Vulkan features that must be enabled. Here we don't need
    // any of them.
    //
    // We also have to pass a list of queues to create and their priorities relative to each other.
    // Since we create one queue, we don't really care about the priority and just pass `0.5`.
    // The list of created queues is returned by the function alongside with the device.
    let (device, queues) = vulkano::device::Device::new(&physical,
                                                        &vulkano::instance::Features::none(),
                                                        [(queue, 0.5)].iter().cloned())
                                                                .expect("failed to create device");

    // Since we can request multiple queues, the `queues` variable is a `Vec`. Our actual queue
    // is the first element.
    let queue = queues.into_iter().next().unwrap();

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating
    // a swapchain allocates the color buffers that will contain the image that will be visible
    // on the screen.
    let (swapchain, images) = {
        let caps = surface.get_capabilities(&physical).expect("failed to get surface capabilities");
        println!("{:?}", caps);

        let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
        let present = caps.present_modes[0];
        let usage = caps.supported_usage_flags;

        vulkano::swapchain::Swapchain::new(&device, &surface, 3,
                                           vulkano::formats::B8G8R8A8Srgb, dimensions, 1,
                                           &usage, vulkano::swapchain::SurfaceTransform::Identity,
                                           vulkano::swapchain::CompositeAlpha::Opaque,
                                           present, true).expect("failed to create swapchain")
    };

    // We create a buffer that will store the shape of our triangle.
    //
    // The first parameter is the device to use, and the second parameter is where to get the
    // memory where the buffer will be stored. The latter is very important as it determines the
    // way you are going to access and modify your buffer. Here we just ask for a basic host
    // visible memory.
    //
    // Note that to store immutable data, the best way is to create two buffers. One buffer on
    // the CPU and one buffer on the GPU. We then write our data to the buffer on the CPU and
    // ask the GPU to copy it to the real buffer. This way the data is located on the most
    // efficient memory possible.
    let vertex_buffer: Arc<vulkano::buffer::Buffer<[Vertex; 3], _>> =
                                vulkano::buffer::Buffer::new(&device, vulkano::memory::HostVisible)
                                                                .expect("failed to create buffer");
    struct Vertex { position: [f32; 2] }

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

    // The next step is to create the shader.
    //
    // The shader creation API provided by the vulkano library is unsafe, for various reasons.
    //
    // Instead, in our build script we used the `vulkano-shaders` crate to parse our shader at
    // compile time and provide a safe wrapper over vulkano's API. You can find the shader's
    // source code in the `triangle.glsl` file.
    //
    // The code generated by the build script created a struct named `TriangleShader`, which we
    // can now use to load the shader.
    //
    // Because of some restrictions with the `include!` macro, we need to use a module.
    mod shader { include!{concat!(env!("OUT_DIR"), "/examples-triangle.rs")} }
    let shader = shader::TriangleShader::load(&device);

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitely does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // The next step is to create a *renderpass*, which is an object that describes where the
    // output of the graphics pipeline will go. It describes the layout of the images
    // where the colors, depth and/or stencil information will be written.
    let renderpass = renderpass!{
        device: &device,
        attachments: {
            color [Clear]
        }
    }.unwrap();

    // However the renderpass doesn't contain the *actual* attachments. It only describes the
    // layout of the output. In order to describe the actual attachments, we have to create a
    // *framebuffer*.
    //
    // A framebuffer is built upon a renderpass, but you can use a framebuffer with any other
    // renderpass as long as the layout is the same.
    //
    // In our situation we want to draw on the swapchain we created above. To do so, we extract
    // images from it.
    let framebuffers = images.iter().map(|image| {
        vulkano::framebuffer::Framebuffer::new(&renderpass, (1244, 699, 1), image).unwrap()
    }).collect::<Vec<_>>();

    // Don't worry, it's almost over!
    //
    // The next step is to create a *graphics pipeline*. This describes the state in which the GPU
    // must be in order to draw our triangle. It contains various information like the list of
    // shaders, depth function, primitive types, etc.
    let graphics_pipeline = ;

    // We are going to create a command buffer right below. Command buffers need to be allocated
    // from a *command buffer pool*, so we create the pool.
    let cb_pool = vulkano::command_buffer::CommandBufferPool::new(&device, &queue.lock().unwrap().family())
                                                  .expect("failed to create command buffer pool");

    // The final initialization step is to create a command buffer.
    //
    // A command buffer contains a list of commands that the GPU must execute. This can include
    // transfers between buffers, clearing images or attachments, etc. and draw commands. Here we
    // create a command buffer with two commands: clearing the attachment and drawing the triangle.
    let command_buffers = framebuffers.iter().map(|framebuffer| {
        vulkano::command_buffer::PrimaryCommandBufferBuilder::new(&cb_pool).unwrap()
            .draw_inline(&renderpass, &framebuffer, [0.0, 0.0, 1.0, 1.0])
            .draw_end()
            .build().unwrap()
    }).collect::<Vec<_>>();

    // Initialization is finally finished!

    // Note that the only thing we need now is the `command_buffer` variable. Everything else is
    // kept alive internally with `Arc`s (even the vertex buffer), so the only variable that we
    // need is this one.

    loop {
        // Before we can draw on the output, we have to *acquire* an image from the swapchain.
        // This operation returns the index of the image that we are allowed to draw upon.
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

        // Sleeping a bit in order not take up too much CPU.
        thread::sleep_ms(16);
    }
}
