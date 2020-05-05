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
use vulkano::descriptor::descriptor::{DescriptorDesc, DescriptorDescTy, DescriptorImageDesc, DescriptorImageDescArray, DescriptorImageDescDimensions, ShaderStages};
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, UnsafeDescriptorSetLayout};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::{SwapchainImage, ImmutableImage, Dimensions};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::sampler::{Sampler, SamplerAddressMode, Filter, MipmapMode};
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;
use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent};

use png;
use std::io::Cursor;

use std::sync::Arc;


fn main() {
    // This example is based off of the `image` example and demonstrates the usage of an image sampler array.
    // This allows you to provide an array of images to a shader, the length of which can be determined at runtime.
    // Click to adjust the push constant index into the array.

    // In order to use a variable length array within the shader requires  the GLSL extension `GL_EXT_nonuniform_qualifier`.
    // This is not currently supported by vulkano-shaders and requires runtime-loading of SPIRV. For the purposes of this example
    // the shader has the number of textures set statically.

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &required_extensions, None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();

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
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
            dimensions, 1, usage, &queue, SurfaceTransform::Identity, alpha,
            PresentMode::Fifo, FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()
    };


    #[derive(Default, Debug, Clone)]
    struct Vertex { position: [f32; 2] }
    vulkano::impl_vertex!(Vertex, position);

    let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
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

    let (texture1, tex1_future) = {
        let png_bytes = include_bytes!("image_img_1.png").to_vec();
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

    let (texture2, tex2_future) = {
        let png_bytes = include_bytes!("image_img_2.png").to_vec();
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

    let image_array = vec![texture1, texture2];

    let layout_descriptors = vec![
        Some(DescriptorDesc {
            ty: DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc {
                sampled: true,
                dimensions: DescriptorImageDescDimensions::TwoDimensional,
                format: None,
                multisampled: false,
                array_layers: DescriptorImageDescArray::NonArrayed,
            }),
            array_count: image_array.len() as u32,
            stages: ShaderStages {
                vertex: false,
                tessellation_control: false,
                tessellation_evaluation: false,
                geometry: false,
                fragment: true,
                compute: false,
            },
            readonly: true,
        })
    ];
    let layout = Arc::new(UnsafeDescriptorSetLayout::new(device.clone(), layout_descriptors.into_iter()).unwrap());

    let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
        .add_sampled_image_array(image_array.clone(), sampler.clone()).unwrap()
        .build().unwrap()
    );

    let mut pc = fs::ty::PushConstants { n_textures: image_array.len() as u32, index: 0 };

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let tex_future = tex1_future.join(tex2_future);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(tex_future) as Box<dyn GpuFuture>);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreate_swapchain = true;
            },
            Event::WindowEvent { event: WindowEvent::MouseInput { state, .. } , .. } => {
                if state == winit::event::ElementState::Pressed {
                    if pc.index == image_array.len() as u32 - 1 {
                        pc.index = 0;
                    } else {
                        pc.index += 1;
                    }
                }
            },
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e)
                    };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state);
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    },
                    Err(e) => panic!("Failed to acquire next image: {:?}", e)
                };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec!([0.0, 0.0, 1.0, 1.0].into());
                let cb = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                    .unwrap()
                    .begin_render_pass(framebuffers[image_num].clone(), false, clear_values).unwrap()
                    .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), set.clone(), pc).unwrap()
                    .end_render_pass().unwrap()
                    .build().unwrap();

                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), cb).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(Box::new(future) as Box<_>);
                    },
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                }
            },
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

layout(push_constant) uniform PushConstants {
    uint n_textures;
    uint index;
};

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D tex[2];

void main() {

    f_color = texture(tex[index % n_textures], tex_coords);
}"
    }
}
