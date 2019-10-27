// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::{SwapchainImage, ImmutableImage, Dimensions};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::sampler::{Sampler, SamplerAddressMode, Filter, MipmapMode};
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;

use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{Window, WindowBuilder};
use winit::event::{Event, WindowEvent};

use png;
use std::io::Cursor;

use std::sync::Arc;

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    let extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &extensions, None).unwrap();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let events_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window();

    let queue_family = physical.queue_families().find(|&q|
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    ).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };
    let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
        [(queue_family, 0.5)].iter().cloned()).unwrap();
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();

        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        let initial_dimensions = {
            // convert to physical pixels
            let dimensions: (u32, u32) = window.inner_size().to_physical(window.hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        };

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
            initial_dimensions, 1, usage, &queue, SurfaceTransform::Identity, alpha,
            PresentMode::Fifo, true, None).unwrap()
    };


    #[derive(Default, Debug, Clone)]
    struct Vertex { position: [f32; 2] }
    vulkano::impl_vertex!(Vertex, position);

    let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
        device.clone(),
        BufferUsage::all(),
        [
            Vertex { position: [-0.5, -0.5 ] },
            Vertex { position: [-0.5,  0.5 ] },
            Vertex { position: [ 0.5, -0.5 ] },
            Vertex { position: [ 0.5,  0.5 ] },
        ].iter().cloned()
    ).unwrap();

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap()
    );
    
    let (texture, tex_future) = {
        let png_bytes = include_bytes!("image_img.png").to_vec();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let (info, mut reader) = decoder.read_info().unwrap();
        let dimensions = Dimensions::Dim2d { width: info.width, height: info.height };
        let mut image_data = Vec::new();
        image_data.resize((info.width * info.height * 4) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        ImmutableImage::from_iter(
            image_data.iter().cloned(),
            dimensions,
            Format::R8G8B8A8Srgb,
            queue.clone()
        ).unwrap()
    };

    let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
        MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_strip()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .blend_alpha_blending()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
        .add_sampled_image(texture.clone(), sampler.clone()).unwrap()
        .build().unwrap()
    );

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(tex_future) as Box<dyn GpuFuture>);

    events_loop.run(move |ev, _, cf| {
        *cf = ControlFlow::Poll;
        let window = surface.window();

        previous_frame_end.as_mut().unwrap().cleanup_finished();
        if recreate_swapchain {
            let dimensions = {
                let dimensions: (u32, u32) = window.inner_size().to_physical(window.hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                Err(err) => panic!("{:?}", err)
            };

            swapchain = new_swapchain;
            framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state);

            recreate_swapchain = false;
        }

        let (image_num, future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                return;
            }
            Err(err) => panic!("{:?}", err)
        };

        let clear_values = vec!([0.0, 0.0, 1.0, 1.0].into());
        let cb = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
            .unwrap()
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values).unwrap()
            .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), set.clone(), ()).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        let prev = previous_frame_end.take();
        let future = prev.unwrap().join(future)
            .then_execute(queue.clone(), cb).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                // This wait is required when using NVIDIA or running on macOS. See https://github.com/vulkano-rs/vulkano/issues/1247
                future.wait(None).unwrap();
                previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
            }
        }

        match ev {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => *cf = ControlFlow::Exit,
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
            _ => ()
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}

mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = position + vec2(0.5);
}"
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    f_color = texture(tex, tex_coords);
}"
    }
}
