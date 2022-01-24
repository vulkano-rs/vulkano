use std::sync::Arc;

use anyhow::anyhow;
use nalgebra::Vector4;

use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};

use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Subpass;

use crate::renderer::gbuffers::GBuffers;
use crate::scene::scene::Scene;
use crate::VulkanContext;

/// The struct containing resources specific to the lighting subpass.
/// The lighting subpass's purpose is to calculate each light's contribution
/// to a visible surface point and add them together, since light is additive.
/// It's done in the screen space, so we're just drawing a big triangle covering the whole screen
/// and use the fragment shader as some sort of "for loop" across the screen pixels.
///
/// Q: Why use a triangle? Isn't the screen rectangular?
/// A: https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
/// Q: Why not compute shader?
/// A: The reason "subpass" existed in the first place is to remove the wait between passes.
///    Subpass is basically a compute shader designed for screen space processing.
///    Also, according to unreliable internet sources,
///    compute shaders require manual tweaking the workload assignments for different GPU models
///    to be actually faster than fragment shaders.
///
/// For other infos... you have read geometry_subpass.rs first, right?
pub struct LightingSubpass {
    pipeline: Arc<GraphicsPipeline>,
    /// We're using an Vec<> here because each subpassInput seems to require their own set.
    descriptor_sets: Vec<Arc<PersistentDescriptorSet>>,
}

#[repr(C)]
struct PushConstants {
    /// Use vec4's to prevent memory issues.
    /// Although you can do it like this:
    /// ```rust
    /// position: Vector3<f32>,
    /// _pad: f32
    /// ```
    /// ```glsl
    /// vec4 position;
    /// ```
    /// but the constructor will then look stupid since Rust doesn't have inline default values,
    /// and you'll have to invent `_pad2`, `_pad3`, `_padinf` for every 3-component vector used.
    pub position: Vector4<f32>,
    pub luminance: Vector4<f32>,
}

impl LightingSubpass {
    pub fn new<TWindow: Send + Sync + 'static>(
        vk: VulkanContext<TWindow>,
        subpass: Subpass,
        gbuffers: Arc<GBuffers>,
    ) -> anyhow::Result<Self> {
        let pipeline = {
            // Load the shaders.
            let vs = vert::load(vk.device())?;
            let vertex_shader = vs
                .entry_point("main")
                .ok_or_else(|| anyhow!("Lighting subpass vertex shader has no entry point"))?;
            let fs = frag::load(vk.device())?;
            let fragment_shader = fs
                .entry_point("main")
                .ok_or_else(|| anyhow!("Lighting subpass fragment shader has no entry point"))?;

            GraphicsPipeline::start()
                // Basic bindings.
                .render_pass(subpass.clone())
                .vertex_shader(vertex_shader, ())
                .fragment_shader(fragment_shader, ())
                // Input definition.
                .vertex_input_state(
                    // We're not passing ANY vertex data at all here.
                    // More explanation later.
                    BuffersDefinition::new(),
                )
                .input_assembly_state(
                    InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
                )
                // Settings.
                .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                .color_blend_state(
                    // We are using additive blend this time.
                    ColorBlendState::new(subpass.num_color_attachments())
                        .blend(AttachmentBlend::additive()),
                )
                .depth_stencil_state(
                    // We won't want depth test for a single triangle...
                    DepthStencilState::disabled(),
                )
                .build(vk.device())?
        };

        // Create a separate descriptor set for each buffer.
        let descriptor_sets = vec![
            gbuffers.position_buffer.clone(),
            gbuffers.normal_buffer.clone(),
            gbuffers.base_color_buffer.clone(),
            gbuffers.id_buffer.clone(),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, buffer)| -> anyhow::Result<_> {
            Ok(PersistentDescriptorSet::new(
                pipeline
                    .layout()
                    .descriptor_set_layouts()
                    .get(i)
                    .ok_or_else(|| {
                        anyhow!("Lighting pipeline has no available descriptor set layouts")
                    })?
                    .clone(),
                [WriteDescriptorSet::image_view(0, buffer)],
            )?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            pipeline,
            descriptor_sets,
        })
    }

    pub fn build_command_buffer(
        &self,
        command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        scene: &Scene,
    ) -> anyhow::Result<()> {
        for (point_light, position) in &scene.point_lights {
            command_buffer_builder
                .bind_pipeline_graphics(self.pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    self.descriptor_sets.clone(),
                )
                .push_constants(
                    self.pipeline.layout().clone(),
                    0,
                    PushConstants {
                        position: position.to_homogeneous(),
                        luminance: point_light.luminance.push(0.),
                    },
                )
                // No need to bind index or vertex buffer here.
                // The vertex shader will just be invoked 3 times without any `in`s.
                // It'll generate the needed info based on the invocation index.
                .draw(3, 1, 0, 0)?;
        }

        Ok(())
    }
}

mod vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/bin/object_oriented_deferred_cube/renderer/passes/main_pass/lighting_subpass.vert"
    }
}
mod frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/object_oriented_deferred_cube/renderer/passes/main_pass/lighting_subpass.frag"
    }
}
