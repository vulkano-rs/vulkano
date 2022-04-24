// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! This example demonstrates using the `VK_KHR_multiview` extension to render to multiple
//! layers of the framebuffer in one render pass. This can significantly improve performance
//! in cases where multiple perspectives or cameras are very similar like in virtual reality
//! or other types of stereoscopic rendering where the left and right eye only differ
//! in a small position offset.

use bytemuck::{Pod, Zeroable};
use std::{fs::File, io::BufWriter, path::Path, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, BufferImageCopy, CommandBufferUsage, CopyImageToBufferInfo,
        RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo,
    },
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageDimensions, ImageLayout,
        ImageSubresourceLayers, ImageUsage, SampleCount, StorageImage,
    },
    impl_vertex,
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{
        AttachmentDescription, AttachmentReference, Framebuffer, FramebufferCreateInfo, LoadOp,
        RenderPass, RenderPassCreateInfo, StoreOp, Subpass, SubpassDescription,
    },
    sync::{self, GpuFuture},
};

fn main() {
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: InstanceExtensions {
            khr_get_physical_device_properties2: true, // required to get multiview limits
            ..InstanceExtensions::none()
        },
        ..Default::default()
    })
    .unwrap();

    let device_extensions = DeviceExtensions {
        ..DeviceExtensions::none()
    };
    let features = Features {
        // enabling the `multiview` feature will use the `VK_KHR_multiview` extension on
        // Vulkan 1.0 and the device feature on Vulkan 1.1+
        multiview: true,
        ..Features::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| {
            p.supported_extensions().is_superset_of(&device_extensions)
        })
        .filter(|&p| {
            p.supported_features().is_superset_of(&features)
        })
        .filter(|&p| {
            // This example renders to two layers of the framebuffer using the multiview
            // extension so we check that at least two views are supported by the device.
            // Not checking this on a device that doesn't support two views
            // will lead to a runtime error when creating the `RenderPass`.
            // The `max_multiview_view_count` function will return `None` when the
            // `VK_KHR_get_physical_device_properties2` instance extension has not been enabled.
            p.properties().max_multiview_view_count.unwrap_or(0) >= 2
        })
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics())
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        // A real application should probably fall back to rendering the framebuffer layers
        // in multiple passes when multiview isn't supported.
        .expect("No device supports two multiview views or the VK_KHR_get_physical_device_properties2 instance extension has not been loaded");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            enabled_features: features,
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
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
        Format::B8G8R8A8_SRGB,
        ImageUsage {
            transfer_src: true,
            color_attachment: true,
            ..ImageUsage::none()
        },
        ImageCreateFlags::none(),
        Some(queue_family),
    )
    .unwrap();

    let image_view = ImageView::new_default(image.clone()).unwrap();

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

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let render_pass_description = RenderPassCreateInfo {
        attachments: vec![AttachmentDescription {
            format: Some(image.format()),
            samples: SampleCount::Sample1,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            stencil_load_op: LoadOp::Clear,
            stencil_store_op: StoreOp::Store,
            initial_layout: ImageLayout::ColorAttachmentOptimal,
            final_layout: ImageLayout::ColorAttachmentOptimal,
            ..Default::default()
        }],
        subpasses: vec![SubpassDescription {
            // the view mask indicates which layers of the framebuffer should be rendered for each
            // subpass
            view_mask: 0b11,
            color_attachments: vec![Some(AttachmentReference {
                attachment: 0,
                layout: ImageLayout::ColorAttachmentOptimal,
                ..Default::default()
            })],
            ..Default::default()
        }],
        // the correlated view masks indicate sets of views that may be more efficient to render
        // concurrently
        correlated_view_masks: vec![0b11],
        ..Default::default()
    };

    let render_pass = RenderPass::new(device.clone(), render_pass_description).unwrap();

    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![image_view],
            ..Default::default()
        },
    )
    .unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
            Viewport {
                origin: [0.0, 0.0],
                dimensions: [
                    image.dimensions().width() as f32,
                    image.dimensions().height() as f32,
                ],
                depth_range: 0.0..1.0,
            },
        ]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let create_buffer = || {
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            (0..image.dimensions().width() * image.dimensions().height() * 4).map(|_| 0u8),
        )
        .unwrap()
    };

    let buffer1 = create_buffer();
    let buffer2 = create_buffer();

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue_family,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // drawing commands are broadcast to each view in the view mask of the active renderpass
    // which means only a single draw call is needed to draw to multiple layers of the framebuffer
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .bind_pipeline_graphics(pipeline.clone())
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .draw(vertex_buffer.len() as u32, 1, 0, 0)
        .unwrap()
        .end_render_pass()
        .unwrap();

    // copy the image layers to different buffers to save them as individual images to disk
    builder
        .copy_image_to_buffer(CopyImageToBufferInfo {
            regions: [BufferImageCopy {
                image_subresource: ImageSubresourceLayers {
                    array_layers: 0..1,
                    ..image.subresource_layers()
                },
                image_extent: image.dimensions().width_height_depth(),
                ..Default::default()
            }]
            .into(),
            ..CopyImageToBufferInfo::image_buffer(image.clone(), buffer1.clone())
        })
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo {
            regions: [BufferImageCopy {
                image_subresource: ImageSubresourceLayers {
                    array_layers: 1..2,
                    ..image.subresource_layers()
                },
                image_extent: image.dimensions().width_height_depth(),
                ..Default::default()
            }]
            .into(),
            ..CopyImageToBufferInfo::image_buffer(image.clone(), buffer2.clone())
        })
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
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&buffer_content).unwrap();
}
