use std::sync::Arc;

use anyhow::anyhow;

use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Subpass;
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode};

use crate::renderer::gbuffers::GBuffers;
use crate::VulkanContext;

/// This subpass draws a laughably bad black outline around objects,
/// and maps the HDR colors back to SDR.
pub struct OutlineAndToneMappingSubpass {
    pipeline: Arc<GraphicsPipeline>,
    /// All the G-buffers are attached as sampled images within the descriptor set.
    descriptor_set: Arc<PersistentDescriptorSet>,
}
impl OutlineAndToneMappingSubpass {
    pub fn new<TWindow: Send + Sync + 'static>(
        vk: VulkanContext<TWindow>,
        subpass: Subpass,
        gbuffers: Arc<GBuffers>,
    ) -> anyhow::Result<Self> {
        let pipeline = {
            let vs = vert::load(vk.device())?;
            let vertex_shader = vs
                .entry_point("main")
                .ok_or_else(|| anyhow!("Outline subpass vertex shader has no entry point"))?;
            let fs = frag::load(vk.device())?;
            let fragment_shader = fs
                .entry_point("main")
                .ok_or_else(|| anyhow!("Outline subpass fragment shader has no entry point"))?;

            GraphicsPipeline::start()
                // Basic bindings.
                .render_pass(subpass.clone())
                .vertex_shader(vertex_shader, ())
                .fragment_shader(fragment_shader, ())
                // Input definition.
                .vertex_input_state(BuffersDefinition::new())
                .input_assembly_state(
                    InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
                )
                // Settings.
                .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()))
                .depth_stencil_state(DepthStencilState::disabled())
                .build(vk.device())?
        };

        // Build a nearest-neighbor clamping sampler for reading input.
        let input_sampler = Sampler::start(vk.device())
            .filter(Filter::Nearest)
            .address_mode(SamplerAddressMode::ClampToEdge)
            .build()?;

        let descriptor_set = PersistentDescriptorSet::new(
            pipeline
                .layout()
                .descriptor_set_layouts()
                .get(0)
                .ok_or_else(|| anyhow!("Outline pipeline has no available descriptor set layouts"))?
                .clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    gbuffers.position_buffer.clone(),
                    input_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    gbuffers.normal_buffer.clone(),
                    input_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    gbuffers.base_color_buffer.clone(),
                    input_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    3,
                    gbuffers.id_buffer.clone(),
                    input_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    4,
                    gbuffers.composite_buffer.clone(),
                    input_sampler.clone(),
                ),
            ],
        )?;

        Ok(Self {
            pipeline,
            descriptor_set,
        })
    }

    pub fn build_command_buffer(
        &self,
        command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> anyhow::Result<()> {
        command_buffer_builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.descriptor_set.clone(),
            )
            .draw(3, 1, 0, 0)?;
        Ok(())
    }
}

mod vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/bin/object_oriented_deferred_cube/renderer/passes/post_pass/outline_and_tonemapping_subpass.vert"
    }
}
mod frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/object_oriented_deferred_cube/renderer/passes/post_pass/outline_and_tonemapping_subpass.frag"
    }
}
