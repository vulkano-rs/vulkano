// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer,
};
use vulkano::device::Queue;
use vulkano::pipeline::depth_stencil::DepthStencilState;
use vulkano::pipeline::input_assembly::InputAssemblyState;
use vulkano::pipeline::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::Subpass;

pub struct TriangleDrawSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pipeline: Arc<GraphicsPipeline>,
}

impl TriangleDrawSystem {
    /// Initializes a triangle drawing system.
    pub fn new(gfx_queue: Arc<Queue>, subpass: Subpass) -> TriangleDrawSystem {
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
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
            .expect("failed to create buffer")
        };

        let pipeline = {
            let vs = vs::load(gfx_queue.device().clone()).expect("failed to create shader module");
            let fs = fs::load(gfx_queue.device().clone()).expect("failed to create shader module");

            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.entry_point("main").unwrap(), ())
                .input_assembly_state(InputAssemblyState::new())
                .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                .fragment_shader(fs.entry_point("main").unwrap(), ())
                .depth_stencil_state(DepthStencilState::simple_depth_test())
                .render_pass(subpass)
                .build(gfx_queue.device().clone())
                .unwrap()
        };

        TriangleDrawSystem {
            gfx_queue: gfx_queue,
            vertex_buffer: vertex_buffer,
            pipeline: pipeline,
        }
    }

    /// Builds a secondary command buffer that draws the triangle on the current subpass.
    pub fn draw(&self, viewport_dimensions: [u32; 2]) -> SecondaryAutoCommandBuffer {
        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::MultipleSubmit,
            self.pipeline.subpass().clone(),
        )
        .unwrap();
        builder
            .set_viewport(
                0,
                [Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                }],
            )
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }
}

#[repr(C)]
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

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;

void main() {
    f_color = vec4(1.0, 1.0, 1.0, 1.0);
    f_normal = vec3(0.0, 0.0, 1.0);
}"
    }
}
