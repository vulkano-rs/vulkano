// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates using the `VK_KHR_multiview` extension to render to multiple layers of
// the framebuffer in one render pass. This can significantly improve performance in cases where
// multiple perspectives or cameras are very similar like in virtual reality or other types of
// stereoscopic rendering where the left and right eye only differ in a small position offset.

use std::{fs::File, io::BufWriter, path::Path};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BufferImageCopy,
        CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageDimensions, ImageLayout,
        ImageSubresourceLayers, ImageUsage, SampleCount, StorageImage,
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, PipelineLayout,
    },
    render_pass::{
        AttachmentDescription, AttachmentReference, Framebuffer, FramebufferCreateInfo, LoadOp,
        RenderPass, RenderPassCreateInfo, StoreOp, Subpass, SubpassDescription,
    },
    shader::PipelineShaderStageCreateInfo,
    sync::{self, GpuFuture},
    VulkanLibrary,
};

fn main() {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: InstanceExtensions {
                // Required to get multiview limits.
                khr_get_physical_device_properties2: true,
                ..InstanceExtensions::empty()
            },
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        ..DeviceExtensions::empty()
    };
    let features = Features {
        // enabling the `multiview` feature will use the `VK_KHR_multiview` extension on Vulkan 1.0
        // and the device feature on Vulkan 1.1+.
        multiview: true,
        ..Features::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter(|p| p.supported_features().contains(&features))
        .filter(|p| {
            // This example renders to two layers of the framebuffer using the multiview extension
            // so we check that at least two views are supported by the device. Not checking this
            // on a device that doesn't support two views will lead to a runtime error when
            // creating the `RenderPass`. The `max_multiview_view_count` function will return
            // `None` when the `VK_KHR_get_physical_device_properties2` instance extension has not
            // been enabled.
            p.properties().max_multiview_view_count.unwrap_or(0) >= 2
        })
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::GRAPHICS))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        // A real application should probably fall back to rendering the framebuffer layers in
        // multiple passes when multiview isn't supported.
        .expect(
            "no device supports two multiview views or the \
            `VK_KHR_get_physical_device_properties2` instance extension has not been loaded",
        );

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: features,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let image = StorageImage::with_usage(
        &memory_allocator,
        ImageDimensions::Dim2d {
            width: 512,
            height: 512,
            array_layers: 2,
        },
        Format::B8G8R8A8_SRGB,
        ImageUsage::TRANSFER_SRC | ImageUsage::COLOR_ATTACHMENT,
        ImageCreateFlags::empty(),
        Some(queue.queue_family_index()),
    )
    .unwrap();

    let image_view = ImageView::new_default(image.clone()).unwrap();

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

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
    let vertex_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        vertices,
    )
    .unwrap();

    // Note the `#extension GL_EXT_multiview : enable` that enables the multiview extension for the
    // shader and the use of `gl_ViewIndex` which contains a value based on which view the shader
    // is being invoked for. In this example `gl_ViewIndex` is used to toggle a hardcoded offset
    // for vertex positions but in a VR application you could easily use it as an index to a
    // uniform array that contains the transformation matrices for the left and right eye.
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450
                #extension GL_EXT_multiview : enable

                layout(location = 0) in vec2 position;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0) + gl_ViewIndex * vec4(0.25, 0.25, 0.0, 0.0);
                }
            ",
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            ",
        }
    }

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
            // The view mask indicates which layers of the framebuffer should be rendered for each
            // subpass.
            view_mask: 0b11,
            color_attachments: vec![Some(AttachmentReference {
                attachment: 0,
                layout: ImageLayout::ColorAttachmentOptimal,
                ..Default::default()
            })],
            ..Default::default()
        }],
        // The correlated view masks indicate sets of views that may be more efficient to render
        // concurrently.
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

    let pipeline = {
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let vertex_input_state = Vertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::entry_point(vs),
            PipelineShaderStageCreateInfo::entry_point(fs),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(render_pass, 0).unwrap();
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::viewport_fixed_scissor_irrelevant([
                    Viewport {
                        origin: [0.0, 0.0],
                        dimensions: [
                            image.dimensions().width() as f32,
                            image.dimensions().height() as f32,
                        ],
                        depth_range: 0.0..1.0,
                    },
                ])),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::new(subpass.num_color_attachments())),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let create_buffer = || {
        Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            (0..image.dimensions().width() * image.dimensions().height() * 4).map(|_| 0u8),
        )
        .unwrap()
    };

    let buffer1 = create_buffer();
    let buffer2 = create_buffer();

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // Drawing commands are broadcast to each view in the view mask of the active renderpass which
    // means only a single draw call is needed to draw to multiple layers of the framebuffer.
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .bind_pipeline_graphics(pipeline)
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .draw(vertex_buffer.len() as u32, 1, 0, 0)
        .unwrap()
        .end_render_pass()
        .unwrap();

    // Copy the image layers to different buffers to save them as individual images to disk.
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

    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    // Write each layer to its own file.
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

fn write_image_buffer_to_file(buffer: Subbuffer<[u8]>, path: &str, width: u32, height: u32) {
    let buffer_content = buffer.read().unwrap();
    let path = Path::new(path);
    let file = File::create(path).unwrap();
    let w = &mut BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&buffer_content).unwrap();

    if let Ok(path) = path.canonicalize() {
        println!("Saved to {}", path.display());
    }
}
