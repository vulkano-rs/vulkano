// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.


// Welcome to the triangle example!
//
// This is the only example that is entirely detailed. All the other examples avoid code
// duplication by using helper functions.
//
// This example assumes that you are already more or less familiar with graphics programming
// and that you want to learn Vulkan. This means that for example it won't go into details about
// what a vertex or a shader is.


// The `vulkano` crate is the main crate that you must use to use Vulkan.
#[macro_use]
extern crate vulkano;
// However the Vulkan library doesn't provide any functionality to create and handle windows, as
// this would be out of scope. In order to open a window, we are going to use the `winit` crate.
extern crate winit;
// The `vulkano_win` crate is the link between `vulkano` and `winit`. Vulkano doesn't know about
// winit, and winit doesn't know about vulkano, so import a crate that will provide a link between
// the two.
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer;
use vulkano::command_buffer::DynamicState;
use vulkano::command_buffer::PrimaryCommandBufferBuilder;
use vulkano::command_buffer::Submission;
use vulkano::descriptor::pipeline_layout::EmptyPipeline;
use vulkano::device::Device;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::Subpass;
use vulkano::instance::Instance;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineParams;
use vulkano::pipeline::blend::Blend;
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::pipeline::input_assembly::InputAssembly;
use vulkano::pipeline::multisample::Multisample;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::ViewportsState;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::viewport::Scissor;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::Swapchain;

use std::sync::Arc;
use std::time::Duration;

fn main() {
    // The first step of any vulkan program is to create an instance.
    let instance = {
        // When we create an instance, we have to pass a list of extensions that we want to enable.
        //
        // All the window-drawing functionalities are part of non-core extensions that we need
        // to enable manually. To do so, we ask the `vulkano_win` crate for the list of extensions
        // required to draw to a window.
        let extensions = vulkano_win::required_extensions();

        // Now creating the instance.
        Instance::new(None, &extensions, None).expect("failed to create Vulkan instance")
    };

    // We then choose which physical device to use.
    //
    // In a real application, there are three things to take into consideration:
    //
    // - Some devices may not support some of the optional features that may be required by your
    //   application. You should filter out the devices that don't support your app.
    //
    // - Not all devices can draw to a certain surface. Once you create your window, you have to
    //   choose a device that is capable of drawing to it.
    //
    // - You probably want to leave the choice between the remaining devices to the user.
    //
    // For the sake of the example we are just going to use the first device, which should work
    // most of the time.
    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    // Some little debug infos.
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    // The objective of this example is to draw a triangle on a window. To do so, we first need to
    // create the window.
    //
    // This is done by creating a `WindowBuilder` from the `winit` crate, then calling the
    // `build_vk_surface` method provided by the `VkSurfaceBuild` trait from `vulkano_win`. If you
    // ever get an error about `build_vk_surface` being undefined in one of your projects, this
    // probably means that you forgot to import this trait.
    //
    // This returns a `vulkano_win::Window` object that contains both a cross-platform winit
    // window and a cross-platform Vulkan surface that represents the surface of the window.
    let window = winit::WindowBuilder::new().build_vk_surface(&instance).unwrap();

    // The next step is to choose which GPU queue will execute our draw commands.
    //
    // Devices can provide multiple queues to run commands in parallel (for example a draw queue
    // and a compute queue), similar to CPU threads. This is something you have to have to manage
    // manually in Vulkan.
    //
    // In a real-life application, we would probably use at least a graphics queue and a transfers
    // queue to handle data transfers in parallel. In this example we only use one queue.
    //
    // We have to choose which queues to use early on, because we will need this info very soon.
    let queue = physical.queue_families().find(|q| {
        // We take the first queue that supports drawing to our window.
        q.supports_graphics() && window.surface().is_supported(q).unwrap_or(false)
    }).expect("couldn't find a graphical queue family");

    // Now initializing the device. This is probably the most important object of Vulkan.
    //
    // We have to pass five parameters when creating a device:
    //
    // - Which physical device to connect to.
    //
    // - A list of optional features and extensions that our program needs to work correctly.
    //   Some parts of the Vulkan specs are optional and must be enabled manually at device
    //   creation. In this example the only thing we are going to need is the `khr_swapchain`
    //   extension that allows us to draw to a window.
    //
    // - A list of layers to enable. This is very niche, and you will usually pass `None`.
    //
    // - The list of queues that we are going to use. The exact parameter is an iterator whose
    //   items are `(Queue, f32)` where the floating-point represents the priority of the queue
    //   between 0.0 and 1.0. The priority of the queue is a hint to the implementation about how
    //   much it should prioritize queues between one another.
    //
    // The list of created queues is returned by the function alongside with the device.
    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            .. vulkano::device::DeviceExtensions::none()
        };

        Device::new(&physical, physical.supported_features(), &device_ext,
                    [(queue, 0.5)].iter().cloned()).expect("failed to create device")
    };

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retreive the first and only element of the
    // iterator and throw it away.
    let queue = queues.next().unwrap();

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating
    // a swapchain allocates the color buffers that will contain the image that will ultimately
    // be visible on the screen. These images are returned alongside with the swapchain.
    let (swapchain, images) = {
        // Querying the capabilities of the surface. When we create the swapchain we can only
        // pass values that are allowed by the capabilities.
        let caps = window.surface().get_capabilities(&physical)
                         .expect("failed to get surface capabilities");

        // We choose the dimensions of the swapchain to match the current dimensions of the window.
        // If `caps.current_extent` is `None`, this means that the window size will be determined
        // by the dimensions of the swapchain, in which case we just use a default value.
        let dimensions = caps.current_extent.unwrap_or([1280, 1024]);

        // The present mode determines the way the images will be presented on the screen. This
        // includes things such as vsync and will affect the framerate of your application. We just
        // use the first supported value, but you probably want to leave that choice to the user.
        let present = caps.present_modes.iter().next().unwrap();

        // The alpha mode indicates how the alpha value of the final image will behave. For example
        // you can choose whether the window will be opaque or transparent.
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Choosing the internal format that the images will have.
        let format = caps.supported_formats[0].0;

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(&device, &window.surface(), 2, format, dimensions, 1,
                       &caps.supported_usage_flags, &queue, SurfaceTransform::Identity, alpha,
                       present, true, None).expect("failed to create swapchain")
    };

    // We now create a buffer that will store the shape of our triangle.
    let vertex_buffer = {
        #[derive(Debug, Clone)]
        struct Vertex { position: [f32; 2] }
        impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(&device, &BufferUsage::all(), Some(queue.family()), [
            Vertex { position: [-0.5, -0.25] },
            Vertex { position: [0.0, 0.5] },
            Vertex { position: [0.25, -0.1] }
        ].iter().cloned()).expect("failed to create buffer")
    };

    // The next step is to create the shaders.
    //
    // The shader creation API provided by the vulkano library is unsafe, for various reasons.
    //
    // Instead, in our build script we used the `vulkano-shaders` crate to parse our shaders at
    // compile time and provide a safe wrapper over vulkano's API. See the `build.rs` file at the
    // root of the crate. You can find the shaders' source code in the `triangle_fs.glsl` and
    // `triangle_vs.glsl` files.
    //
    // The author knows that this system is crappy and that it would be far better to use a plugin.
    // Unfortunately plugins aren't available in stable Rust yet.
    //
    // The code generated by the build script created a struct named `TriangleShader`, which we
    // can now use to load the shader.
    //
    // Because of some restrictions with the `include!` macro, we need to use a module.
    mod vs { include!{concat!(env!("OUT_DIR"), "/shaders/src/bin/triangle_vs.glsl")} }
    let vs = vs::Shader::load(&device).expect("failed to create shader module");
    mod fs { include!{concat!(env!("OUT_DIR"), "/shaders/src/bin/triangle_fs.glsl")} }
    let fs = fs::Shader::load(&device).expect("failed to create shader module");

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitely does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // The next step is to create a *render pass*, which is an object that describes where the
    // output of the graphics pipeline will go. It describes the layout of the images
    // where the colors, depth and/or stencil information will be written.
    mod render_pass {
        use vulkano::format::Format;

        // Calling this macro creates multiple structs based on the macro's parameters:
        //
        // - `CustomRenderPass` is the main struct that represents the render pass.
        // - `Formats` can be used to indicate the list of the formats of the attachments.
        // - `AList` can be used to indicate the actual list of images that are attached.
        //
        // Render passes can also have multiple subpasses, the only restriction being that all
        // the passes will use the same framebuffer dimensions. Here we only have one pass, so
        // we use the appropriate macro.
        single_pass_renderpass!{
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `load: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load: Clear,
                    // `store: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store: Store,
                    // `format: <ty>` indicates the type of the format of the image. This has to
                    // be one of the types of the `vulkano::format` module (or alternatively one
                    // of your structs that implements the `FormatDesc` trait). Here we use the
                    // generic `vulkano::format::Format` enum because we don't know the format in
                    // advance.
                    format: Format,
                }
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {}
            }
        }
    }

    // The macro above only created the custom struct that represents our render pass. We also have
    // to actually instanciate that struct.
    //
    // To do so, we have to pass the actual values of the formats of the attachments.
    let render_pass = render_pass::CustomRenderPass::new(&device, &render_pass::Formats {
        // Use the format of the images and one sample.
        color: (images[0].format(), 1)
    }).unwrap();

    // Before we draw we have to create what is called a pipeline. This is similar to an OpenGL
    // program, but much more specific.
    let pipeline = GraphicsPipeline::new(&device, GraphicsPipelineParams {
        // We need to indicate the layout of the vertices.
        // The type `SingleBufferDefinition` actually contains a template parameter corresponding
        // to the type of each vertex. But in this code it is automatically inferred.
        vertex_input: SingleBufferDefinition::new(),

        // A Vulkan shader can in theory contain multiple entry points, so we have to specify
        // which one. The `main` word of `main_entry_point` actually corresponds to the name of
        // the entry point.
        vertex_shader: vs.main_entry_point(),

        // `InputAssembly::triangle_list()` is a shortcut to build a `InputAssembly` struct that
        // describes a list of triangles.
        input_assembly: InputAssembly::triangle_list(),

        // No tessellation shader.
        tessellation: None,

        // No geometry shader.
        geometry_shader: None,

        // TODO: switch to dynamic viewports and explain how it works
        viewport: ViewportsState::Fixed {
            data: vec![(
                Viewport {
                    origin: [0.0, 0.0],
                    depth_range: 0.0 .. 1.0,
                    dimensions: [images[0].dimensions()[0] as f32,
                                 images[0].dimensions()[1] as f32],
                },
                Scissor::irrelevant()
            )],
        },

        // The `Raster` struct can be used to customize parameters such as polygon mode or backface
        // culling.
        raster: Default::default(),

        // If we use multisampling, we can pass additional configuration.
        multisample: Multisample::disabled(),

        // See `vertex_shader`.
        fragment_shader: fs.main_entry_point(),

        // `DepthStencil::disabled()` is a shortcut to build a `DepthStencil` struct that describes
        // the fact that depth and stencil testing are disabled.
        depth_stencil: DepthStencil::disabled(),

        // `Blend::pass_through()` is a shortcut to build a `Blend` struct that describes the fact
        // that colors must be directly transferred from the fragment shader output to the
        // attachments without any change.
        blend: Blend::pass_through(),

        // Shaders can usually access resources such as images or buffers. This parameters is here
        // to indicate the layout of the accessed resources, which is also called the *pipeline
        // layout*. Here we don't access anything, so we just create an `EmptyPipeline` object.
        layout: &EmptyPipeline::new(&device).unwrap(),

        // We have to indicate which subpass of which render pass this pipeline is going to be used
        // in. The pipeline will only be usable from this particular subpass.
        render_pass: Subpass::from(&render_pass, 0).unwrap(),
    }).unwrap();

    // The render pass we created above only describes the layout of our framebuffers. Before we
    // can draw we also need to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.
    let framebuffers = images.iter().map(|image| {
        let dimensions = [image.dimensions()[0], image.dimensions()[1], 1];
        Framebuffer::new(&render_pass, dimensions, render_pass::AList {
            // The `AList` struct was generated by the render pass macro above, and contains one
            // member for each attachment.
            color: image
        }).unwrap()
    }).collect::<Vec<_>>();

    // Initialization is finally finished!

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // a `Submission` object which holds the resources for as long as they are in use by the GPU.
    //
    // Destroying a `Submission` blocks until the GPU is finished executing it. In order to avoid
    // that, we store them in a `Vec` and clean them from time to time.
    let mut submissions: Vec<Arc<Submission>> = Vec::new();

    loop {
        // Clearing the old submissions by keeping alive only the ones whose destructor would block.
        submissions.retain(|s| s.destroying_would_block());

        // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
        // no image is available (which happens if you submit draw commands too quickly), then the
        // function will block.
        // This operation returns the index of the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is a timeout after
        // which the function call will return an error.
        let image_num = swapchain.acquire_next_image(Duration::new(1, 0)).unwrap();

        // In order to draw, we have to build a *command buffer*. The command buffer object holds
        // the list of commands that are going to be executed.
        //
        // Building a command buffer is an expensive operation (usually a few hundred
        // microseconds), but it is known to be a hot path in the driver and is expected to be
        // optimized.
        //
        // Note that we have to pass a queue family when we create the command buffer. The command
        // buffer will only be executable on that given queue family.
        let command_buffer = PrimaryCommandBufferBuilder::new(&device, queue.family())
            // Before we can draw, we have to *enter a render pass*. There are two methods to do
            // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
            // not covered here.
            //
            // The third parameter contains the list of values to clear the attachments with. Only
            // the attachments that use `load: Clear` appear in this struct.
            .draw_inline(&render_pass, &framebuffers[image_num], render_pass::ClearValues {
                color: [0.0, 0.0, 1.0, 1.0]
            })

            // We are now inside the first subpass of the render pass. We add a draw command.
            //
            // The last two parameters contain the list of resources to pass to the shaders.
            // Since we used an `EmptyPipeline` object, the objects have to be `()`.
            .draw(&pipeline, &vertex_buffer, &DynamicState::none(), (), &())

            // We leave the render pass by calling `draw_end`. Note that if we had multiple
            // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
            // next subpass.
            .draw_end()

            // Finish building the command buffer by calling `build`.
            .build();

        // Now all we need to do is submit the command buffer to the queue.
        submissions.push(command_buffer::submit(&command_buffer, &queue).unwrap());

        // The color output is now expected to contain our triangle. But in order to show it on
        // the screen, we have to *present* the image by calling `present`.
        //
        // This function does not actually present the image immediately. Instead it submits a
        // present command at the end of the queue. This means that it will only be presented once
        // the GPU has finished executing the command buffer that draws the triangle.
        swapchain.present(&queue, image_num).unwrap();

        // Note that in more complex programs it is likely that one of `acquire_next_image`,
        // `command_buffer::submit`, or `present` will block for some time. This happens when the
        // GPU's queue is full and the driver has to wait until the GPU finished some work.
        //
        // Unfortunately the Vulkan API doesn't provide any way to not wait or to detect when a
        // wait would happen. Blocking may be the desired behavior, but if you don't want to
        // block you should spawn a separate thread dedicated to submissions.

        // Handling the window events in order to close the program when the user wants to close
        // it.
        for ev in window.window().poll_events() {
            match ev {
                winit::Event::Closed => return,
                _ => ()
            }
        }
    }
}
