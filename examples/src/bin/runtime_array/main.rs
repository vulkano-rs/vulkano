// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use png;
use std::io::Cursor;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::descriptor_set::layout::{
    DescriptorSetDesc, DescriptorSetLayout, DescriptorSetLayoutError,
};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::Format;
use vulkano::image::ImageAccess;
use vulkano::image::{
    view::ImageView, ImageDimensions, ImageUsage, ImmutableImage, MipmapsCount, SwapchainImage,
};
use vulkano::instance::Instance;
use vulkano::pipeline::color_blend::ColorBlendState;
use vulkano::pipeline::layout::PipelineLayout;
use vulkano::pipeline::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain::{self, AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, Version::V1_2, &required_extensions, None).unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        &Features {
            descriptor_indexing: true,
            shader_uniform_buffer_array_non_uniform_indexing: true,
            runtime_descriptor_array: true,
            descriptor_binding_variable_descriptor_count: true,
            descriptor_binding_partially_bound: true,
            ..Features::none()
        },
        &physical_device
            .required_extensions()
            .union(&device_extensions),
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical_device).unwrap();
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

    #[derive(Default, Debug, Clone)]
    struct Vertex {
        position: [f32; 2],
        tex_i: u32,
        coords: [f32; 2],
    }
    vulkano::impl_vertex!(Vertex, position, tex_i, coords);

    let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
            Vertex {
                position: [-0.1, -0.9],
                tex_i: 0,
                coords: [1.0, 0.0],
            },
            Vertex {
                position: [-0.9, -0.9],
                tex_i: 0,
                coords: [0.0, 0.0],
            },
            Vertex {
                position: [-0.9, -0.1],
                tex_i: 0,
                coords: [0.0, 1.0],
            },
            Vertex {
                position: [-0.1, -0.9],
                tex_i: 0,
                coords: [1.0, 0.0],
            },
            Vertex {
                position: [-0.9, -0.1],
                tex_i: 0,
                coords: [0.0, 1.0],
            },
            Vertex {
                position: [-0.1, -0.1],
                tex_i: 0,
                coords: [1.0, 1.0],
            },
            Vertex {
                position: [0.9, -0.9],
                tex_i: 1,
                coords: [1.0, 0.0],
            },
            Vertex {
                position: [0.1, -0.9],
                tex_i: 1,
                coords: [0.0, 0.0],
            },
            Vertex {
                position: [0.1, -0.1],
                tex_i: 1,
                coords: [0.0, 1.0],
            },
            Vertex {
                position: [0.9, -0.9],
                tex_i: 1,
                coords: [1.0, 0.0],
            },
            Vertex {
                position: [0.1, -0.1],
                tex_i: 1,
                coords: [0.0, 1.0],
            },
            Vertex {
                position: [0.9, -0.1],
                tex_i: 1,
                coords: [1.0, 1.0],
            },
        ]
        .iter()
        .cloned(),
    )
    .unwrap();

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
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
    .unwrap();

    let mascot_texture = {
        let png_bytes = include_bytes!("rust_mascot.png").to_vec();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();
        let dimensions = ImageDimensions::Dim2d {
            width: info.width,
            height: info.height,
            array_layers: 1,
        };
        let mut image_data = Vec::new();
        image_data.resize((info.width * info.height * 4) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        let image = ImmutableImage::from_iter(
            image_data.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            queue.clone(),
        )
        .unwrap()
        .0;

        ImageView::new(image).unwrap()
    };

    let vulkano_texture = {
        let png_bytes = include_bytes!("vulkano_logo.png").to_vec();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();
        let dimensions = ImageDimensions::Dim2d {
            width: info.width,
            height: info.height,
            array_layers: 1,
        };
        let mut image_data = Vec::new();
        image_data.resize((info.width * info.height * 4) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        let image = ImmutableImage::from_iter(
            image_data.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            queue.clone(),
        )
        .unwrap()
        .0;

        ImageView::new(image).unwrap()
    };

    let sampler = Sampler::new(
        device.clone(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    .unwrap();

    let pipeline_layout = {
        let mut descriptor_set_descs: Vec<_> =
            DescriptorSetDesc::from_requirements(fs.main_entry_point().descriptor_requirements());

        // Set 0, Binding 0
        descriptor_set_descs[0].set_variable_descriptor_count(0, 2);

        let descriptor_set_layouts = descriptor_set_descs
            .into_iter()
            .map(|desc| Ok(DescriptorSetLayout::new(device.clone(), desc.clone())?))
            .collect::<Result<Vec<_>, DescriptorSetLayoutError>>()
            .unwrap();

        PipelineLayout::new(
            device.clone(),
            descriptor_set_layouts,
            fs.main_entry_point().push_constant_range().iter().cloned(),
        )
        .unwrap()
    };

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    let pipeline = GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.main_entry_point(), ())
        .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()).blend_alpha())
        .render_pass(subpass)
        .with_pipeline_layout(device.clone(), pipeline_layout)
        .unwrap();

    let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
    let mut set_builder = PersistentDescriptorSet::start(layout.clone());

    set_builder
        .enter_array()
        .unwrap()
        .add_sampled_image(mascot_texture.clone(), sampler.clone())
        .unwrap()
        .add_sampled_image(vulkano_texture.clone(), sampler.clone())
        .unwrap()
        .leave_array()
        .unwrap();

    let set = set_builder.build().unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    let mut recreate_swapchain = false;
    let mut previous_frame_end: Option<Box<dyn GpuFuture>> =
        Some(Box::new(vulkano::sync::now(device.clone())));

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
                framebuffers =
                    window_size_dependent_setup(&new_images, render_pass.clone(), &mut viewport);
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

            let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    SubpassContents::Inline,
                    clear_values,
                )
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    set.clone(),
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
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
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Framebuffer::start(render_pass.clone())
                .add(view)
                .unwrap()
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>()
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        vulkan_version: "1.2",
        spirv_version: "1.5",
        src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in uint tex_i;
layout(location = 2) in vec2 coords;

layout(location = 0) out flat uint out_tex_i;
layout(location = 1) out vec2 out_coords;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    out_tex_i = tex_i;
    out_coords = coords;
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        vulkan_version: "1.2",
        spirv_version: "1.5",
        src: "
#version 450

#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in flat uint tex_i;
layout(location = 1) in vec2 coords;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D tex[];

void main() {
    f_color = texture(tex[tex_i], coords);
}"
    }
}
