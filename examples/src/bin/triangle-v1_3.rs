// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
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
//
// This version of the triangle example is written for Vulkan 1.3 and higher, using dynamic
// rendering instead of render pass and framebuffer objects. If your device does not support
// Vulkan 1.3, or if you want to see how to support older versions, see the original triangle
// example.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            render_pass::PipelineRenderingCreateInfo,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{LoadOp, StoreOp},
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() {
    // The first step of any Vulkan program is to create an instance.
    //
    // When we create an instance, we have to pass a list of extensions that we want to enable.
    //
    // All the window-drawing functionalities are part of non-core extensions that we need
    // to enable manually. To do so, we ask the `vulkano_win` crate for the list of extensions
    // required to draw to a window.
    let required_extensions = vulkano_win::required_extensions();

    // Now creating the instance.
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    // The objective of this example is to draw a triangle on a window. To do so, we first need to
    // create the window.
    //
    // This is done by creating a `WindowBuilder` from the `winit` crate, then calling the
    // `build_vk_surface` method provided by the `VkSurfaceBuild` trait from `vulkano_win`. If you
    // ever get an error about `build_vk_surface` being undefined in one of your projects, this
    // probably means that you forgot to import this trait.
    //
    // This returns a `vulkano::swapchain::Surface` object that contains both a cross-platform winit
    // window and a cross-platform Vulkan surface that represents the surface of the window.
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    // Choose device extensions that we're going to use.
    // In order to present images to a surface, we need a `Swapchain`, which is provided by the
    // `khr_swapchain` extension.
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    // We then choose which physical device to use. First, we enumerate all the available physical
    // devices, then apply filters to narrow them down to those that can support our needs.
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| {
            // For this example, we require at least Vulkan 1.3.
            p.api_version() >= Version::V1_3
        })
        .filter(|&p| {
            // Some devices may not support the extensions or features that your application, or
            // report properties and limits that are not sufficient for your application. These
            // should be filtered out here.
            p.supported_extensions().is_superset_of(&device_extensions)
        })
        .filter_map(|p| {
            // For each physical device, we try to find a suitable queue family that will execute
            // our draw commands.
            //
            // Devices can provide multiple queues to run commands in parallel (for example a draw
            // queue and a compute queue), similar to CPU threads. This is something you have to
            // have to manage manually in Vulkan. Queues of the same type belong to the same
            // queue family.
            //
            // Here, we look for a single queue family that is suitable for our purposes. In a
            // real-life application, you may want to use a separate dedicated transfer queue to
            // handle data transfers in parallel with graphics operations. You may also need a
            // separate queue for compute operations, if your application uses those.
            p.queue_families()
                .find(|&q| {
                    // We select a queue family that supports graphics operations. When drawing to
                    // a window surface, as we do in this example, we also need to check that queues
                    // in this queue family are capable of presenting images to the surface.
                    q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false)
                })
                // The code here searches for the first queue family that is suitable. If none is
                // found, `None` is returned to `filter_map`, which disqualifies this physical
                // device.
                .map(|q| (p, q))
        })
        // All the physical devices that pass the filters above are suitable for the application.
        // However, not every device is equal, some are preferred over others. Now, we assign
        // each physical device a score, and pick the device with the
        // lowest ("best") score.
        //
        // In this example, we simply select the best-scoring device to use in the application.
        // In a real-life setting, you may want to use the best-scoring device only as a
        // "default" or "recommended" device, and let the user choose the device themselves.
        .min_by_key(|(p, _)| {
            // We assign a lower score to device types that are likely to be faster/better.
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            }
        })
        .expect("No suitable physical device found");

    // Some little debug infos.
    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    // Now initializing the device. This is probably the most important object of Vulkan.
    //
    // The iterator of created queues is returned by the function alongside the device.
    let (device, mut queues) = Device::new(
        // Which physical device to connect to.
        physical_device,
        DeviceCreateInfo {
            // A list of optional features and extensions that our program needs to work correctly.
            // Some parts of the Vulkan specs are optional and must be enabled manually at device
            // creation. In this example the only thing we are going to need is the `khr_swapchain`
            // extension that allows us to draw to a window.
            enabled_extensions: physical_device
                // Some devices require certain extensions to be enabled if they are present
                // (e.g. `khr_portability_subset`). We add them to the device extensions that we're
                // going to enable.
                .required_extensions()
                .union(&device_extensions),

            // In order to render with Vulkan 1.3's dynamic rendering, we need to enable it here.
            // Otherwise, we are only allowed to render with a render pass object, as in the
            // standard triangle example. The feature is required to be supported on Vulkan 1.3 and
            // higher, so we don't need to check for support.
            enabled_features: Features {
                dynamic_rendering: true,
                ..Features::none()
            },

            // The list of queues that we are going to use. Here we only use one queue, from the
            // previously chosen queue family.
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],

            ..Default::default()
        },
    )
    .unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. We
    // only use one queue in this example, so we just retrieve the first and only element of the
    // iterator.
    let queue = queues.next().unwrap();

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating
    // a swapchain allocates the color buffers that will contain the image that will ultimately
    // be visible on the screen. These images are returned alongside the swapchain.
    let (mut swapchain, images) = {
        // Querying the capabilities of the surface. When we create the swapchain we can only
        // pass values that are allowed by the capabilities.
        let surface_capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        // Choosing the internal format that the images will have.
        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,

                image_format,
                // The dimensions of the window, only used to initially setup the swapchain.
                // NOTE:
                // On some drivers the swapchain dimensions are specified by
                // `surface_capabilities.current_extent` and the swapchain size must use these
                // dimensions.
                // These dimensions are always the same as the window dimensions.
                //
                // However, other drivers don't specify a value, i.e.
                // `surface_capabilities.current_extent` is `None`. These drivers will allow
                // anything, but the only sensible value is the window
                // dimensions.
                //
                // Both of these cases need the swapchain to use the window dimensions, so we just
                // use that.
                image_extent: surface.window().inner_size().into(),

                image_usage: ImageUsage::color_attachment(),

                // The alpha mode indicates how the alpha value of the final image will behave. For
                // example, you can choose whether the window will be opaque or transparent.
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .unwrap(),

                ..Default::default()
            },
        )
        .unwrap()
    };

    // We now create a buffer that will store the shape of our triangle.
    // We use #[repr(C)] here to force rustc to not do anything funky with our data, although for this
    // particular example, it doesn't actually change the in-memory representation.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
    struct Vertex {
        position: [f32; 2],
    }
    impl_vertex!(Vertex, position);

    let vertices = [
        Vertex {
            position: [-0.5, -0.25],
        },
        Vertex {
            position: [0.0, 0.5],
        },
        Vertex {
            position: [0.25, -0.1],
        },
    ];
    let vertex_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertices)
            .unwrap();

    // The next step is to create the shaders.
    //
    // The raw shader creation API provided by the vulkano library is unsafe, for various reasons.
    //
    // An overview of what the `shader!` macro generates can be found in the
    // `vulkano-shaders` crate docs. You can view them at https://docs.rs/vulkano-shaders/
    //
    // TODO: explain this in details
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
				#version 450

				layout(location = 0) in vec2 position;

				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
				}
			"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
				#version 450

				layout(location = 0) out vec4 f_color;

				void main() {
					f_color = vec4(1.0, 0.0, 0.0, 1.0);
				}
			"
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // Before we draw we have to create what is called a pipeline. This is similar to an OpenGL
    // program, but much more specific.
    let pipeline = GraphicsPipeline::start()
        // We describe the formats of attachment images where the colors, depth and/or stencil
        // information will be written. The pipeline will only be usable with this particular
        // configuration of the attachment images.
        .render_pass(PipelineRenderingCreateInfo {
            // We specify a single color attachment that will be rendered to. When we begin
            // rendering, we will specify a swapchain image to be used as this attachment, so here
            // we set its format to be the same format as the swapchain.
            color_attachment_formats: vec![Some(swapchain.image_format())],
            ..Default::default()
        })
        // We need to indicate the layout of the vertices.
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        // The content of the vertex buffer describes a list of triangles.
        .input_assembly_state(InputAssemblyState::new())
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify
        // which one.
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        // Use a resizable viewport set to draw over the entire window
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        // See `vertex_shader`.
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
        .build(device.clone())
        .unwrap();

    // Dynamic viewports allow us to recreate just the viewport when the window is resized
    // Otherwise we would have to recreate the whole pipeline.
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    // When creating the swapchain, we only created plain images. To use them as an attachment for
    // rendering, we must wrap then in an image view.
    //
    // Since we need to draw to multiple images, we are going to create a different image view for
    // each image.
    let mut attachment_image_views = window_size_dependent_setup(&images, &mut viewport);

    // Initialization is finally finished!

    // In some situations, the swapchain will become invalid by itself. This includes for example
    // when the window is resized (as the images of the swapchain will no longer match the
    // window's) or, on Android, when the application went to the background and goes back to the
    // foreground.
    //
    // In this situation, acquiring a swapchain image or presenting it will return an error.
    // Rendering to an image of that swapchain will not produce any error, but may or may not work.
    // To continue rendering, we need to recreate the swapchain by creating a new swapchain.
    // Here, we remember that we need to do this for the next loop iteration.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                // It is important to call this function from time to time, otherwise resources will keep
                // accumulating and you will eventually reach an out of memory error.
                // Calling this function polls various fences in order to determine what the GPU has
                // already processed, and frees the resources that are no longer needed.
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever the window resizes we need to recreate everything dependent on the window size.
                // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
                if recreate_swapchain {
                    // Get the new dimensions of the window.

                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: surface.window().inner_size().into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            // This error tends to happen when the user is manually resizing the window.
                            // Simply restarting the loop is the easiest way to fix this issue.
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    // Now that we have new swapchain images, we must create new image views from
                    // them as well.
                    attachment_image_views =
                        window_size_dependent_setup(&new_images, &mut viewport);
                    recreate_swapchain = false;
                }

                // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
                // no image is available (which happens if you submit draw commands too quickly), then the
                // function will block.
                // This operation returns the index of the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional timeout
                // after which the function call will return an error.
                let (image_num, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
                // will still work, but it may not display correctly. With some drivers this can be when
                // the window resizes, but it may not cause the swapchain to become out of date.
                if suboptimal {
                    recreate_swapchain = true;
                }

                // In order to draw, we have to build a *command buffer*. The command buffer object holds
                // the list of commands that are going to be executed.
                //
                // Building a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to be
                // optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The command
                // buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    // Before we can draw, we have to *enter a render pass*. We specify which
                    // attachments we are going to use for rendering here, which needs to match
                    // what was previously specified when creating the pipeline.
                    .begin_rendering(RenderingInfo {
                        // As before, we specify one color attachment, but now we specify
                        // the image view to use as well as how it should be used.
                        color_attachments: vec![Some(RenderingAttachmentInfo {
                            // `Clear` means that we ask the GPU to clear the content of this
                            // attachment at the start of rendering.
                            load_op: LoadOp::Clear,
                            // `Store` means that we ask the GPU to store the rendered output
                            // in the attachment image. We could also ask it to discard the result.
                            store_op: StoreOp::Store,
                            // The value to clear the attachment with. Here we clear it with a
                            // blue color.
                            //
                            // Only attachments that have `LoadOp::Clear` are provided with
                            // clear values, any others should use `None` as the clear value.
                            clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
                            ..RenderingAttachmentInfo::image_view(
                                // We specify image view corresponding to the currently acquired
                                // swapchain image, to use for this attachment.
                                attachment_image_views[image_num].clone(),
                            )
                        })],
                        ..Default::default()
                    })
                    .unwrap()
                    // We are now inside the first subpass of the render pass. We add a draw command.
                    //
                    // The last two parameters contain the list of resources to pass to the shaders.
                    // Since we used an `EmptyPipeline` object, the objects have to be `()`.
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    // We leave the render pass.
                    .end_rendering()
                    .unwrap();

                // Finish building the command buffer by calling `build`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to show it on
                    // the screen, we have to *present* the image by calling `present`.
                    //
                    // This function does not actually present the image immediately. Instead it submits a
                    // present command at the end of the queue. This means that it will only be presented once
                    // the GPU has finished executing the command buffer that draws the triangle.
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    viewport: &mut Viewport,
) -> Vec<Arc<ImageView<SwapchainImage<Window>>>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>()
}
