// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! TODO multiview explanation

use std::fs::File;
use std::io::BufWriter;
use std::iter;
use std::path::Path;
use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SubpassContents,
};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{
    ImageAccess, ImageCreateFlags, ImageDimensions, ImageLayout, ImageUsage, SampleCount,
    StorageImage,
};
use vulkano::instance::PhysicalDevice;
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{
    AttachmentDesc, Framebuffer, LoadOp, MultiviewDesc, RenderPass, RenderPassDesc, StoreOp,
    Subpass, SubpassDesc,
};
use vulkano::sync::GpuFuture;
use vulkano::{sync, Version};

fn main() {
    let instance = Instance::new(
        None,
        Version::V1_1,
        &InstanceExtensions {
            khr_get_physical_device_properties2: true, // required to get multiview limits

            ..InstanceExtensions::none()
        },
        None,
    )
    .unwrap();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    // This example renders to two layers of the framebuffer using the multiview extension so we
    // check that at least two views are supported by the device.
    // Not checking this on a device that doesn't support two views
    // will lead to a runtime error when creating the `RenderPass`.
    // The `max_multiview_view_count` function will return `None`
    // when the `VK_KHR_get_physical_device_properties2` instance extension has not been enabled.
    if physical
        .extended_properties()
        .max_multiview_view_count()
        .unwrap_or(0)
        < 2
    {
        println!("The device doesn't support two multiview views or the VK_KHR_get_physical_device_properties2 instance extension has not been loaded");

        // A real application should probably fall back to rendering the framebuffer layers
        // in multiple passes when multiview isn't supported.
        return;
    }

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_multiview: true,

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

    let image = StorageImage::with_usage(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 512,
            height: 512,
            array_layers: 2,
        },
        Format::B8G8R8A8Srgb,
        ImageUsage {
            transfer_source: true,
            color_attachment: true,
            ..ImageUsage::none()
        },
        ImageCreateFlags::none(),
        Some(queue_family),
    )
    .unwrap();

    let image_view = ImageView::new(image.clone()).unwrap();

    let vertex_buffer = {
        #[derive(Default, Debug, Clone)]
        struct Vertex {
            position: [f32; 2],
        }
        vulkano::impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            [
                Vertex {
                    position: [-0.5, -0.25],
                },
                Vertex {
                    position: [0.0, 0.5],
                },
                Vertex {
                    position: [0.25, -0.1],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap()
    };

    // Note the `#extension GL_EXT_multiview : enable` that enables the multiview extension
    // for the shader and the use of `gl_ViewIndex` which contains a value based on which
    // view the shader is being invoked for.
    // In this example `gl_ViewIndex` is used toggle a hardcoded offset for vertex positions
    // but in a VR application you could easily use it as an index to a uniform array
    // that contains the transformation matrices for the left and right eye.
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
				#version 450
                #extension GL_EXT_multiview : enable

				layout(location = 0) in vec2 position;

				void main() {
                    gl_Position = vec4(position, 0.0, 1.0) + gl_ViewIndex * vec4(0.25, 0.25, 0.0, 0.0);
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

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let render_pass_description = RenderPassDesc::with_multiview(
        vec![AttachmentDesc {
            format: image.format(),
            samples: SampleCount::Sample1,
            load: LoadOp::Clear,
            store: StoreOp::Store,
            stencil_load: LoadOp::Clear,
            stencil_store: StoreOp::Store,
            initial_layout: ImageLayout::ColorAttachmentOptimal,
            final_layout: ImageLayout::ColorAttachmentOptimal,
        }],
        vec![SubpassDesc {
            color_attachments: vec![(0, ImageLayout::ColorAttachmentOptimal)],
            depth_stencil: None,
            input_attachments: vec![],
            resolve_attachments: vec![],
            preserve_attachments: vec![],
        }],
        vec![],
        MultiviewDesc {
            view_masks: vec![0b11],
            correlation_masks: vec![0b11],
            view_offsets: vec![],
        },
    );

    let render_pass = Arc::new(RenderPass::new(device.clone(), render_pass_description).unwrap());

    let framebuffer = Arc::new(
        Framebuffer::start(render_pass.clone())
            .add(image_view)
            .unwrap()
            .build()
            .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports(iter::once(Viewport {
                origin: [0.0, 0.0],
                dimensions: [
                    image.dimensions().width() as f32,
                    image.dimensions().height() as f32,
                ],
                depth_range: 0.0..1.0,
            }))
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let dynamic_state = DynamicState::none();

    let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

    let buffer1 = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0..image.dimensions().width() * image.dimensions().height() * 4).map(|_| 0u8),
    )
    .unwrap();
    let buffer2 = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0..image.dimensions().width() * image.dimensions().height() * 4).map(|_| 0u8),
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue_family,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // drawing commands are broadcast to each view in the view mask of the active renderpass
    // which means only a single draw call is needed to draw to multiple layers of the framebuffer
    builder
        .begin_render_pass(framebuffer.clone(), SubpassContents::Inline, clear_values)
        .unwrap()
        .draw(
            pipeline.clone(),
            &dynamic_state,
            vertex_buffer.clone(),
            (),
            (),
            vec![],
        )
        .unwrap()
        .end_render_pass()
        .unwrap();

    // copy the image layers to different buffers to save them as individual images to disk
    builder
        .copy_image_to_buffer_dimensions(
            image.clone(),
            buffer1.clone(),
            [0, 0, 0],
            image.dimensions().width_height_depth(),
            0,
            1,
            0,
        )
        .unwrap()
        .copy_image_to_buffer_dimensions(
            image.clone(),
            buffer2.clone(),
            [0, 0, 0],
            image.dimensions().width_height_depth(),
            1,
            1,
            0,
        )
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    // write each layer to its own file
    write_image_buffer_to_file(
        buffer1,
        "multiview1.png",
        image.dimensions().width(),
        image.dimensions().height(),
    );
    write_image_buffer_to_file(
        buffer2,
        "multiview2.png",
        image.dimensions().width(),
        image.dimensions().height(),
    );
}

fn write_image_buffer_to_file(
    buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    path: &str,
    width: u32,
    height: u32,
) {
    let buffer_content = buffer.read().unwrap();
    let path = Path::new(path);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::RGBA);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&buffer_content).unwrap();
}
