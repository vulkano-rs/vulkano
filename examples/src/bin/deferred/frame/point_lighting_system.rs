// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use cgmath::{Matrix4, Vector3};
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
        PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
};

use super::LightingVertex;

pub struct PointLightingSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Subbuffer<[LightingVertex]>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl PointLightingSystem {
    /// Initializes the point lighting system.
    pub fn new(
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        memory_allocator: &impl MemoryAllocator,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> PointLightingSystem {
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
                    viewport_state: Some(ViewportState::viewport_dynamic_scissor_irrelevant()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(
                        ColorBlendState::new(subpass.num_color_attachments()).blend(
                            AttachmentBlend {
                                color_blend_op: BlendOp::Add,
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                            },
                        ),
                    ),
                    subpass: Some(subpass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        PointLightingSystem {
            gfx_queue,
            vertex_buffer,
            subpass,
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    /// Builds a secondary command buffer that applies a point lighting.
    ///
    /// This secondary command buffer will read `depth_input` and rebuild the world position of the
    /// pixel currently being processed (modulo rounding errors). It will then compare this
    /// position with `position`, and process the lighting based on the distance and orientation
    /// (similar to the directional lighting system).
    ///
    /// It then writes the output to the current framebuffer with additive blending (in other words
    /// the value will be added to the existing value in the framebuffer, and not replace the
    /// existing value).
    ///
    /// Note that in a real-world application, you probably want to pass additional parameters
    /// such as some way to indicate the distance at which the lighting decrease. In this example
    /// this value is hardcoded in the shader.
    ///
    /// - `viewport_dimensions` contains the dimensions of the current framebuffer.
    /// - `color_input` is an image containing the albedo of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `normals_input` is an image containing the normals of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `depth_input` is an image containing the depth value of each pixel of the scene. It is
    ///   the result of the deferred pass.
    /// - `screen_to_world` is a matrix that turns coordinates from framebuffer space into world
    ///   space. This matrix is used alongside with `depth_input` to determine the world
    ///   coordinates of each pixel being processed.
    /// - `position` is the position of the spot light in world coordinates.
    /// - `color` is the color of the light.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        viewport_dimensions: [u32; 2],
        color_input: Arc<ImageView>,
        normals_input: Arc<ImageView>,
        depth_input: Arc<ImageView>,
        screen_to_world: Matrix4<f32>,
        position: Vector3<f32>,
        color: [f32; 3],
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let push_constants = fs::PushConstants {
            screen_to_world: screen_to_world.into(),
            color: [color[0], color[1], color[2], 1.0],
            position: position.extend(0.0).into(),
        };

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, color_input),
                WriteDescriptorSet::image_view(1, normals_input),
                WriteDescriptorSet::image_view(2, depth_input),
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
            layout(location = 0) out vec2 v_screen_coords;

            void main() {
                v_screen_coords = position;
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
            // The `depth_input` parameter of the `draw` method.
            layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_depth;

            layout(push_constant) uniform PushConstants {
                // The `screen_to_world` parameter of the `draw` method.
                mat4 screen_to_world;
                // The `color` parameter of the `draw` method.
                vec4 color;
                // The `position` parameter of the `draw` method.
                vec4 position;
            } push_constants;

            layout(location = 0) in vec2 v_screen_coords;
            layout(location = 0) out vec4 f_color;

            void main() {
                float in_depth = subpassLoad(u_depth).x;

                // Any depth superior or equal to 1.0 means that the pixel has been untouched by 
                // the deferred pass. We don't want to deal with them.
                if (in_depth >= 1.0) {
                    discard;
                }

                // Find the world coordinates of the current pixel.
                vec4 world = push_constants.screen_to_world * vec4(v_screen_coords, in_depth, 1.0);
                world /= world.w;

                vec3 in_normal = normalize(subpassLoad(u_normals).rgb);
                vec3 light_direction = normalize(push_constants.position.xyz - world.xyz);

                // Calculate the percent of lighting that is received based on the orientation of 
                // the normal and the direction of the light.
                float light_percent = max(-dot(light_direction, in_normal), 0.0);

                float light_distance = length(push_constants.position.xyz - world.xyz);
                // Further decrease light_percent based on the distance with the light position.
                light_percent *= 1.0 / exp(light_distance);

                vec3 in_diffuse = subpassLoad(u_diffuse).rgb;
                f_color.rgb = push_constants.color.rgb * light_percent * in_diffuse;
                f_color.a = 1.0;
            }
        ",
    }
}
