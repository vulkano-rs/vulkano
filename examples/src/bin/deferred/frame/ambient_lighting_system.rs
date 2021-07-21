// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, SecondaryAutoCommandBuffer,
};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Queue;
use vulkano::image::ImageViewAbstract;
use vulkano::pipeline::blend::AttachmentBlend;
use vulkano::pipeline::blend::BlendFactor;
use vulkano::pipeline::blend::BlendOp;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::Subpass;

/// Allows applying an ambient lighting to a scene.
pub struct AmbientLightingSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pipeline: Arc<GraphicsPipeline>,
}

impl AmbientLightingSystem {
    /// Initializes the ambient lighting system.
    pub fn new(gfx_queue: Arc<Queue>, subpass: Subpass) -> AmbientLightingSystem {
        // TODO: vulkano doesn't allow us to draw without a vertex buffer, otherwise we could
        //       hard-code these values in the shader
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage::all(),
                false,
                [
                    Vertex {
                        position: [-1.0, -1.0],
                    },
                    Vertex {
                        position: [-1.0, 3.0],
                    },
                    Vertex {
                        position: [3.0, -1.0],
                    },
                ]
                .iter()
                .cloned(),
            )
            .expect("failed to create buffer")
        };

        let pipeline = {
            let vs = vs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(
                GraphicsPipeline::start()
                    .vertex_input_single_buffer::<Vertex>()
                    .vertex_shader(vs.main_entry_point(), ())
                    .triangle_list()
                    .viewports_dynamic_scissors_irrelevant(1)
                    .fragment_shader(fs.main_entry_point(), ())
                    .blend_collective(AttachmentBlend {
                        enabled: true,
                        color_op: BlendOp::Add,
                        color_source: BlendFactor::One,
                        color_destination: BlendFactor::One,
                        alpha_op: BlendOp::Max,
                        alpha_source: BlendFactor::One,
                        alpha_destination: BlendFactor::One,
                        mask_red: true,
                        mask_green: true,
                        mask_blue: true,
                        mask_alpha: true,
                    })
                    .render_pass(subpass)
                    .build(gfx_queue.device().clone())
                    .unwrap(),
            ) as Arc<_>
        };

        AmbientLightingSystem {
            gfx_queue: gfx_queue,
            vertex_buffer: vertex_buffer,
            pipeline: pipeline,
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
    ///
    pub fn draw<C>(
        &self,
        viewport_dimensions: [u32; 2],
        color_input: C,
        ambient_color: [f32; 3],
    ) -> SecondaryAutoCommandBuffer
    where
        C: ImageViewAbstract + Send + Sync + 'static,
    {
        let push_constants = fs::ty::PushConstants {
            color: [ambient_color[0], ambient_color[1], ambient_color[2], 1.0],
        };

        let layout = self
            .pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();
        let descriptor_set = PersistentDescriptorSet::start(layout.clone())
            .add_image(color_input)
            .unwrap()
            .build()
            .unwrap();

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                depth_range: 0.0..1.0,
            }]),
            ..DynamicState::none()
        };

        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::MultipleSubmit,
            self.pipeline.subpass().clone(),
        )
        .unwrap();
        builder
            .draw(
                self.pipeline.clone(),
                &dynamic_state,
                self.vertex_buffer.clone(),
                descriptor_set,
                push_constants,
            )
            .unwrap();
        builder.build().unwrap()
    }
}

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
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
}"
    }
}
