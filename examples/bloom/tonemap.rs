use crate::{App, RenderContext};
use std::{slice, sync::Arc};
use vulkano::{
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexInputState,
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        DynamicState, GraphicsPipeline, Pipeline, PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
};
use vulkano_taskgraph::{command_buffer::RecordingCommandBuffer, Task, TaskContext, TaskResult};

const EXPOSURE: f32 = 1.0;

pub struct TonemapTask {
    pipeline: Option<Arc<GraphicsPipeline>>,
}

impl TonemapTask {
    pub fn new(_app: &App) -> Self {
        TonemapTask { pipeline: None }
    }

    pub fn create_pipeline(&mut self, app: &App, subpass: &Subpass) {
        let bcx = app.resources.bindless_context().unwrap();

        let pipeline = {
            let vs = vs::load(&app.device).unwrap().entry_point("main").unwrap();
            let fs = fs::load(&app.device).unwrap().entry_point("main").unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];
            let layout = bcx.pipeline_layout_from_stages(&stages).unwrap();

            GraphicsPipeline::new(
                &app.device,
                None,
                &GraphicsPipelineCreateInfo {
                    stages: &stages,
                    vertex_input_state: Some(&VertexInputState::default()),
                    input_assembly_state: Some(&InputAssemblyState::default()),
                    viewport_state: Some(&ViewportState::default()),
                    rasterization_state: Some(&RasterizationState::default()),
                    multisample_state: Some(&MultisampleState::default()),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState::default()],
                        ..Default::default()
                    }),
                    dynamic_state: &[DynamicState::Viewport],
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        self.pipeline = Some(pipeline);
    }
}

impl Task for TonemapTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        cbf.set_viewport(0, slice::from_ref(&rcx.viewport))?;
        cbf.bind_pipeline_graphics(self.pipeline.as_ref().unwrap())?;
        cbf.push_constants(
            self.pipeline.as_ref().unwrap().layout(),
            0,
            &fs::PushConstants {
                sampler_id: rcx.bloom_sampler_id,
                texture_id: rcx.bloom_sampled_image_id,
                exposure: EXPOSURE,
            },
        )?;

        unsafe { cbf.draw(3, 1, 0, 0) }?;

        Ok(())
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            const vec2[3] POSITIONS = {
                vec2(-1.0, -1.0),
                vec2(-1.0,  3.0),
                vec2( 3.0, -1.0),
            };

            const vec2[3] TEX_COORDS = {
                vec2(0.0, 0.0),
                vec2(0.0, 2.0),
                vec2(2.0, 0.0),
            };

            layout(location = 0) out vec2 v_tex_coords;

            void main() {
                gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
                v_tex_coords = TEX_COORDS[gl_VertexIndex];
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "tonemap.glsl",
        include: ["."],
    }
}
