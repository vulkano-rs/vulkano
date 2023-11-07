// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use cgmath::Vector3;
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
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
};

use super::LightingVertex;

/// Allows applying a directional light source to a scene.
pub struct DirectionalLightingSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Subbuffer<[LightingVertex]>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl DirectionalLightingSystem {
    /// Initializes the directional lighting system.
    pub fn new(
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> DirectionalLightingSystem {
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
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
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
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                color_blend_op: BlendOp::Add,
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                            }),
                            ..Default::default()
                        },
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        DirectionalLightingSystem {
            gfx_queue,
            vertex_buffer,
            subpass,
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    /// Builds a secondary command buffer that applies directional lighting.
    ///
    /// This secondary command buffer will read `color_input` and `normals_input`, and multiply the
    /// color with `color` and the dot product of the `direction` with the normal.
    /// It then writes the output to the current framebuffer with additive blending (in other words
    /// the value will be added to the existing value in the framebuffer, and not replace the
    /// existing value).
    ///
    /// Since `normals_input` contains normals in world coordinates, `direction` should also be in
    /// world coordinates.
    ///
    /// - `viewport_dimensions` contains the dimensions of the current framebuffer.
    /// - `color_input` is an image containing the albedo of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `normals_input` is an image containing the normals of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `direction` is the direction of the light in world coordinates.
    /// - `color` is the color to apply.
    pub fn draw(
        &self,
        viewport_dimensions: [u32; 2],
        color_input: Arc<ImageView>,
        normals_input: Arc<ImageView>,
        direction: Vector3<f32>,
        color: [f32; 3],
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let push_constants = fs::PushConstants {
            color: [color[0], color[1], color[2], 1.0],
            direction: direction.extend(0.0).into(),
        };

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, color_input),
                WriteDescriptorSet::image_view(1, normals_input),
            ],
            [],
        )
        .unwrap();

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
            depth_range: 0.0..=1.0,
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
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap()
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
            // The `normals_input` parameter of the `draw` method.
            layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;

            layout(push_constant) uniform PushConstants {
                // The `color` parameter of the `draw` method.
                vec4 color;
                // The `direction` parameter of the `draw` method.
                vec4 direction;
            } push_constants;

            layout(location = 0) out vec4 f_color;

            void main() {
                vec3 in_normal = normalize(subpassLoad(u_normals).rgb);

                // If the normal is perpendicular to the direction of the lighting, then 
                // `light_percent` will be 0. If the normal is parallel to the direction of the 
                // lightin, then `light_percent` will be 1. Any other angle will yield an 
                // intermediate value.
                float light_percent = -dot(push_constants.direction.xyz, in_normal);
                // `light_percent` must not go below 0.0. There's no such thing as negative lighting.
                light_percent = max(light_percent, 0.0);

                vec3 in_diffuse = subpassLoad(u_diffuse).rgb;
                f_color.rgb = light_percent * push_constants.color.rgb * in_diffuse;
                f_color.a = 1.0;
            }
        ",
    }
}
