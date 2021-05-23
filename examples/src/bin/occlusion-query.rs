// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This is a modification of the triangle example, that demonstrates the basics of occlusion queries.
// Occlusion queries allow you to query whether, and sometimes how many, pixels pass the depth test
// in a range of draw calls.

use std::sync::Arc;
use vulkano::buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SubpassContents,
};
use vulkano::device::{Device, DeviceExtensions, DeviceOwned};
use vulkano::format::Format;
use vulkano::image::{view::ImageView, AttachmentImage, ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::query::{QueryControlFlags, QueryPool, QueryResultFlags, QueryType};
use vulkano::render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass};
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance =
        Instance::new(None, Version::major_minor(1, 1), &required_extensions, None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(composite_alpha)
            .build()
            .unwrap()
    };

    let vertex_buffer = {
        #[derive(Default, Debug, Clone)]
        struct Vertex {
            position: [f32; 3],
            color: [f32; 3],
        }
        vulkano::impl_vertex!(Vertex, position, color);

        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            [
                // The first triangle (red) is the same one as in the triangle example.
                Vertex {
                    position: [-0.5, -0.25, 0.5],
                    color: [1.0, 0.0, 0.0],
                },
                Vertex {
                    position: [0.0, 0.5, 0.5],
                    color: [1.0, 0.0, 0.0],
                },
                Vertex {
                    position: [0.25, -0.1, 0.5],
                    color: [1.0, 0.0, 0.0],
                },
                // The second triangle (cyan) is the same shape and position as the first,
                // but smaller, and moved behind a bit.
                // It should be completely occluded by the first triangle.
                // (You can lower its z value to put it in front)
                Vertex {
                    position: [-0.25, -0.125, 0.6],
                    color: [0.0, 1.0, 1.0],
                },
                Vertex {
                    position: [0.0, 0.25, 0.6],
                    color: [0.0, 1.0, 1.0],
                },
                Vertex {
                    position: [0.125, -0.05, 0.6],
                    color: [0.0, 1.0, 1.0],
                },
                // The third triangle (green) is the same shape and size as the first,
                // but moved to the left and behind the second.
                // It is partially occluded by the first two.
                Vertex {
                    position: [-0.25, -0.25, 0.7],
                    color: [0.0, 1.0, 0.0],
                },
                Vertex {
                    position: [0.25, 0.5, 0.7],
                    color: [0.0, 1.0, 0.0],
                },
                Vertex {
                    position: [0.5, -0.1, 0.7],
                    color: [0.0, 1.0, 0.0],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap()
    };

    // Create three buffer slices, one for each triangle.
    let buffer_slice = vertex_buffer.into_buffer_slice();
    let triangle1 = buffer_slice.clone().slice(0..3).unwrap();
    let triangle2 = buffer_slice.clone().slice(3..6).unwrap();
    let triangle3 = buffer_slice.clone().slice(6..9).unwrap();

    // Create a query pool for occlusion queries, with 3 slots.
    let query_pool = Arc::new(QueryPool::new(device.clone(), QueryType::Occlusion, 3).unwrap());

    // Create a buffer on the CPU to hold the results of the three queries.
    // Query results are always represented as either `u32` or `u64`.
    // For occlusion queries, you always need one element per query. You can ask for the number of
    // elements needed at runtime by calling `QueryType::result_size`.
    // If you retrieve query results with `with_availability` enabled, then this array needs to
    // be 6 elements long instead of 3.
    let mut query_results = [0u32; 3];

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
				#version 450

				layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 color;

                layout(location = 0) out vec3 v_color;

				void main() {
                    v_color = color;
					gl_Position = vec4(position, 1.0);
				}
			"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
				#version 450

                layout(location = 0) in vec3 v_color;
				layout(location = 0) out vec4 f_color;

				void main() {
					f_color = vec4(v_color, 1.0);
				}
			"
        }
    }

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        )
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            // Enable depth testing, which is needed for occlusion queries to make sense at all.
            // If you disable depth testing, every pixel is considered to pass the depth test, so
            // every query will return a nonzero result.
            .depth_stencil_simple_depth()
            .build(device.clone())
            .unwrap(),
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None,
    };

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| match event {
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
            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if recreate_swapchain {
                let dimensions: [u32; 2] = surface.window().inner_size().into();
                let (new_swapchain, new_images) =
                    match swapchain.recreate().dimensions(dimensions).build() {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };

                swapchain = new_swapchain;
                framebuffers = window_size_dependent_setup(
                    &new_images,
                    render_pass.clone(),
                    &mut dynamic_state,
                );
                recreate_swapchain = false;
            }

            let (image_num, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into(), 1.0.into()];

            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            // Beginning or resetting a query is unsafe for now.
            unsafe {
                builder
                    // A query must be reset before each use, including the first use.
                    // This must be done outside a render pass.
                    .reset_query_pool(query_pool.clone(), 0..3)
                    .unwrap()
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values,
                    )
                    .unwrap()
                    // Begin query 0, then draw the red triangle.
                    // Enabling the `precise` bit would give exact numeric results. This needs
                    // the `occlusion_query_precise` feature to be enabled on the device.
                    .begin_query(query_pool.clone(), 0, QueryControlFlags { precise: false })
                    .unwrap()
                    .draw(
                        pipeline.clone(),
                        &dynamic_state,
                        triangle1.clone(),
                        (),
                        (),
                        vec![],
                    )
                    .unwrap()
                    // End query 0.
                    .end_query(query_pool.clone(), 0)
                    .unwrap()
                    // Begin query 1 for the cyan triangle.
                    .begin_query(query_pool.clone(), 1, QueryControlFlags { precise: false })
                    .unwrap()
                    .draw(
                        pipeline.clone(),
                        &dynamic_state,
                        triangle2.clone(),
                        (),
                        (),
                        vec![],
                    )
                    .unwrap()
                    .end_query(query_pool.clone(), 1)
                    .unwrap()
                    // Finally, query 2 for the green triangle.
                    .begin_query(query_pool.clone(), 2, QueryControlFlags { precise: false })
                    .unwrap()
                    .draw(
                        pipeline.clone(),
                        &dynamic_state,
                        triangle3.clone(),
                        (),
                        (),
                        vec![],
                    )
                    .unwrap()
                    .end_query(query_pool.clone(), 2)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();
            }

            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
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

            // Retrieve the query results.
            // This copies the results to a variable on the CPU. You can also use the
            // `copy_query_pool_results` function on a command buffer to write results to a
            // Vulkano buffer. This could then be used to influence draw operations further down
            // the line, either in the same frame or a future frame.
            query_pool
                .queries_range(0..3)
                .unwrap()
                .get_results(
                    &mut query_results,
                    QueryResultFlags {
                        // Block the function call until the results are available.
                        // Note: if not all the queries have actually been executed, then this
                        // will wait forever for something that never happens!
                        wait: true,
                        // Blocking and waiting will never give partial results.
                        partial: false,
                        // Blocking and waiting will ensure the results are always available after
                        // the function returns.
                        //
                        // If you disable waiting, then this can be used to include the
                        // availability of each query's results. You need one extra element per
                        // query in your `query_results` buffer for this. This element will
                        // be filled with a zero/nonzero value indicating availability.
                        with_availability: false,
                    },
                )
                .unwrap();

            // If the `precise` bit was not enabled, then you're only guaranteed to get a boolean
            // result here: zero if all pixels were occluded, nonzero if only some were occluded.
            // Enabling `precise` will give the exact number of pixels.

            // Query 0 (red triangle) will always succeed, because the depth buffer starts empty
            // and will never occlude anything.
            assert_ne!(query_results[0], 0);

            // Query 1 (cyan triangle) will fail, because it's drawn completely behind the first.
            assert_eq!(query_results[1], 0);

            // Query 2 (green triangle) will succeed, because it's only partially occluded.
            assert_ne!(query_results[2], 0);
        }
        _ => (),
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    let depth_attachment = ImageView::new(
        AttachmentImage::with_usage(
            render_pass.device().clone(),
            dimensions,
            Format::D16Unorm,
            ImageUsage {
                depth_stencil_attachment: true,
                transient_attachment: true,
                ..ImageUsage::none()
            },
        )
        .unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(view)
                    .unwrap()
                    .add(depth_attachment.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}
