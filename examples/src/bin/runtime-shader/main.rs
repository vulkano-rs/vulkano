// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
//
// This example demonstrates one way of preparing data structures and loading
// SPIRV shaders from external source (file system).
//
// Note that you will need to do all correctness checking by yourself.
//
// vert.glsl and frag.glsl must be built by yourself.
// One way of building them is to build Khronos' glslang and use
// glslangValidator tool:
// $ glslangValidator vert.glsl -V -S vert -o vert.spv
// $ glslangValidator frag.glsl -V -S frag -o frag.spv
// Vulkano uses glslangValidator to build your shaders internally.

use std::borrow::Cow;
use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use vulkano as vk;
use vulkano::buffer::cpu_access::CpuAccessibleBuffer;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SubpassContents,
};
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::pipeline::layout::PipelineLayoutDesc;
use vulkano::pipeline::shader::{
    GraphicsShaderType, ShaderInterface, ShaderInterfaceEntry, ShaderModule,
    SpecializationConstants,
};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
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

#[derive(Default, Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position, color);

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();
    let physical = vk::instance::PhysicalDevice::enumerate(&instance)
        .next()
        .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .unwrap();
    let (device, mut queues) = {
        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .unwrap()
    };
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

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
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
        )
        .unwrap(),
    );

    let vs = {
        let mut f = File::open("src/bin/runtime-shader/vert.spv")
            .expect("Can't find file src/bin/runtime-shader/vert.spv This example needs to be run from the root of the example crate.");
        let mut v = vec![];
        f.read_to_end(&mut v).unwrap();
        // Create a ShaderModule on a device the same Shader::load does it.
        // NOTE: You will have to verify correctness of the data by yourself!
        unsafe { ShaderModule::new(device.clone(), &v) }.unwrap()
    };

    let fs = {
        let mut f = File::open("src/bin/runtime-shader/frag.spv")
            .expect("Can't find file src/bin/runtime-shader/frag.spv");
        let mut v = vec![];
        f.read_to_end(&mut v).unwrap();
        unsafe { ShaderModule::new(device.clone(), &v) }.unwrap()
    };

    // This definition will tell Vulkan how input entries of our vertex shader look like
    // There are things to consider when giving out entries:
    // * There must be only one entry per one location, you can't have
    //   `color' and `position' entries both at 0..1 locations.  They also
    //   should not overlap.
    // * Format of each element must be no larger than 128 bits.
    let vertex_input = unsafe {
        ShaderInterface::new_unchecked(vec![
            ShaderInterfaceEntry {
                location: 1..2,
                format: Format::R32G32B32Sfloat,
                name: Some(Cow::Borrowed("color")),
            },
            ShaderInterfaceEntry {
                location: 0..1,
                format: Format::R32G32Sfloat,
                name: Some(Cow::Borrowed("position")),
            },
        ])
    };

    // This definition will tell Vulkan how output entries (those passed to next
    // stage) of our vertex shader look like.
    let vertex_output = unsafe {
        ShaderInterface::new_unchecked(vec![ShaderInterfaceEntry {
            location: 0..1,
            format: Format::R32G32B32Sfloat,
            name: Some(Cow::Borrowed("v_color")),
        }])
    };

    // This definition describes the layout of this stage.
    let vertex_layout = PipelineLayoutDesc::new(
        // No descriptor sets.
        vec![],
        // No push constants.
        vec![],
    )
    .unwrap();

    // Same as with our vertex shader, but for fragment one instead.
    let fragment_input = unsafe {
        ShaderInterface::new_unchecked(vec![ShaderInterfaceEntry {
            location: 0..1,
            format: Format::R32G32B32Sfloat,
            name: Some(Cow::Borrowed("v_color")),
        }])
    };

    // Note that color fragment color entry will be determined
    // automatically by Vulkano.
    let fragment_output = unsafe {
        ShaderInterface::new_unchecked(vec![ShaderInterfaceEntry {
            location: 0..1,
            format: Format::R32G32B32A32Sfloat,
            name: Some(Cow::Borrowed("f_color")),
        }])
    };

    // Layout same as with vertex shader.
    let fragment_layout = PipelineLayoutDesc::new(
        // No descriptor sets.
        vec![],
        // No push constants.
        vec![],
    )
    .unwrap();

    // NOTE: ShaderModule::*_shader_entry_point calls do not do any error
    // checking and you have to verify correctness of what you are doing by
    // yourself.
    //
    // You must be extra careful to specify correct entry point, or program will
    // crash at runtime outside of rust and you will get NO meaningful error
    // information!
    let vert_main = unsafe {
        vs.graphics_entry_point(
            CStr::from_bytes_with_nul_unchecked(b"main\0"),
            vertex_layout,
            <()>::descriptors(),
            vertex_input,
            vertex_output,
            GraphicsShaderType::Vertex,
        )
    };

    let frag_main = unsafe {
        fs.graphics_entry_point(
            CStr::from_bytes_with_nul_unchecked(b"main\0"),
            fragment_layout,
            <()>::descriptors(),
            fragment_input,
            fragment_output,
            GraphicsShaderType::Fragment,
        )
    };

    let graphics_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vert_main, ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(frag_main, ())
            .cull_mode_front()
            .front_face_counter_clockwise()
            .depth_stencil_disabled()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let mut recreate_swapchain = false;

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
            Vertex {
                position: [-1.0, 1.0],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.0, -1.0],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                position: [1.0, 1.0],
                color: [0.0, 0.0, 1.0],
            },
        ]
        .iter()
        .cloned(),
    )
    .unwrap();

    // NOTE: We don't create any descriptor sets in this example, but you should
    // note that passing wrong types, providing sets at wrong indexes will cause
    // descriptor set builder to return Err!

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

            let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();
            builder
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    SubpassContents::Inline,
                    clear_values,
                )
                .unwrap()
                .draw(
                    graphics_pipeline.clone(),
                    &dynamic_state,
                    vertex_buffer.clone(),
                    (),
                    (),
                    vec![],
                )
                .unwrap()
                .end_render_pass()
                .unwrap();
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
        }
        _ => (),
    });
}

/// This method is called once during initialization, then again whenever the window is resized
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

    images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(view)
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}
