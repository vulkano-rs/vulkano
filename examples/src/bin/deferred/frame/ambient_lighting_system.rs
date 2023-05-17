// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferUsage, SecondaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    image::ImageViewAbstract,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryUsage},
    pipeline::{
        graphics::{
            color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    },
    render_pass::Subpass,
    shader::PipelineShaderStageCreateInfo,
};

use super::LightingVertex;

/// Allows applying an ambient lighting to a scene.
pub struct AmbientLightingSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Subbuffer<[LightingVertex]>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl AmbientLightingSystem {
    /// Initializes the ambient lighting system.
    pub fn new(
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        memory_allocator: &impl MemoryAllocator,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> AmbientLightingSystem {
        // TODO: vulkano doesn't allow us to draw without a vertex buffer, otherwise we could
        //       hard-code these values in the shader
        let vertices = [
            LightingVertex {
                position: [-1.0, -1.0],
            },
            LightingVertex {
                position: [-1.0, 3.0],
            },
            LightingVertex {
                position: [3.0, -1.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator,
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
        .expect("failed to create buffer");

        let pipeline = {
            let device = gfx_queue.device();
            let vs = vs::load(device.clone())
                .expect("failed to create shader module")
                .entry_point("main")
                .expect("shader entry point not found");
            let fs = fs::load(device.clone())
                .expect("failed to create shader module")
                .entry_point("main")
                .expect("shader entry point not found");
            let vertex_input_state = LightingVertex::per_vertex()
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
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::viewport_dynamic_scissor_irrelevant()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(
                        ColorBlendState::new(subpass.num_color_attachments()).blend(
                            AttachmentBlend {
                                color_op: BlendOp::Add,
                                color_source: BlendFactor::One,
                                color_destination: BlendFactor::One,
                                alpha_op: BlendOp::Max,
                                alpha_source: BlendFactor::One,
                                alpha_destination: BlendFactor::One,
                            },
                        ),
                    ),
                    subpass: Some(subpass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        AmbientLightingSystem {
            gfx_queue,
            vertex_buffer,
            subpass,
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    /// Builds a secondary command buffer that applies ambient lighting.
    ///
    /// This secondary command buffer will read `color_input`, multiply it with `ambient_color`
    /// and write the output to the current framebuffer with additive blending (in other words
    /// the value will be added to the existing value in the framebuffer, and not replace the
    /// existing value).
    ///
    /// - `viewport_dimensions` contains the dimensions of the current framebuffer.
    /// - `color_input` is an image containing the albedo of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `ambient_color` is the color to apply.
    pub fn draw(
        &self,
        viewport_dimensions: [u32; 2],
        color_input: Arc<dyn ImageViewAbstract + 'static>,
        ambient_color: [f32; 3],
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let push_constants = fs::PushConstants {
            color: [ambient_color[0], ambient_color[1], ambient_color[2], 1.0],
        };

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::image_view(0, color_input)],
        )
        .unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        let mut builder = AutoCommandBufferBuilder::secondary(
            self.command_buffer_allocator.as_ref(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .set_viewport(0, [viewport].into_iter().collect())
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            // The `color_input` parameter of the `draw` method.
            layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_diffuse;

            layout(push_constant) uniform PushConstants {
                // The `ambient_color` parameter of the `draw` method.
                vec4 color;
            } push_constants;

            layout(location = 0) out vec4 f_color;

            void main() {
                // Load the value at the current pixel.
                vec3 in_diffuse = subpassLoad(u_diffuse).rgb;
                f_color.rgb = push_constants.color.rgb * in_diffuse;
                f_color.a = 1.0;
            }
        ",
    }
}
