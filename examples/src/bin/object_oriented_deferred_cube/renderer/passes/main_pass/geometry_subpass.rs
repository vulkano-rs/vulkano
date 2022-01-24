use std::sync::Arc;

use anyhow::anyhow;
use nalgebra::Matrix4;

use vulkano::buffer::TypedBufferAccess;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, StateMode};
use vulkano::render_pass::Subpass;

use crate::asset::model::Vertex;
use crate::renderer::asset_library::AssetLibrary;
use crate::scene::scene::Scene;
use crate::VulkanContext;

/// The struct containing resources specific to the geometry subpass.
/// The geometry subpass's responsibility is to "flatten" the mesh onto the G-buffers (which is fast),
/// so the GPU only have to do slow lighting computations on what is visible.
pub struct GeometrySubpass {
    /// A pipeline describes what happens in a subpass.
    pipeline: Arc<GraphicsPipeline>,
    /// A descriptor set describes a handle to a particular set of shared resources.
    /// They are divided into sets because rebinding is expensive -
    /// so you can rebind only what you need.
    descriptor_set: Arc<PersistentDescriptorSet>,
}

/// The push constants we're passing to the pipeline.
/// Push constants are a way of passing small amount of dynamic data to shaders,
/// which is faster than updating a UBO or SSBO.
#[repr(C)]
// ^-- So Rust won't mess with our struct's runtime memory layout.
// Learn more about repr(C) and the memory layout of push constants and UBOs expected by Vulkan:
// https://doc.rust-lang.org/nomicon/other-reprs.html
// https://www.oreilly.com/library/view/opengl-programming-guide/9780132748445/app09lev1sec2.html
// (Note SSBOs can and should use another thing called std430...)
struct PushConstants {
    // Transform matrices.
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
    // You generally want to place sparse scalars at the end of struct
    // (and use 4-component vectors and matrices instead of 3)
    // for push constants, UBOs and SSBOs, to avoid memory alignment issues.
    // It's a shame that Rust doesn't support explicit struct layout like C#...
    id: u32,         // ID.
    base_color: u32, // Index of the base color texture in the texture array.
}
// Yay, no special black magic macro needed for this struct!

impl GeometrySubpass {
    pub fn new<TWindow: Send + Sync + 'static>(
        vk: VulkanContext<TWindow>,
        subpass: Subpass,
        library: Arc<AssetLibrary>,
    ) -> anyhow::Result<Self> {
        let pipeline = {
            // Load the shaders.
            let vs = vert::load(vk.device())?; // needs to be on a separate line, or it won't live long enough
            let vertex_shader = vs
                .entry_point("main")
                .ok_or_else(|| anyhow!("Geometry subpass vertex shader has no entry point"))?;
            let fs = frag::load(vk.device())?;
            let fragment_shader = fs
                .entry_point("main")
                .ok_or_else(|| anyhow!("Geometry subpass fragment shader has no entry point"))?;

            GraphicsPipeline::start()
                // Basic bindings.
                .render_pass(subpass.clone())
                .vertex_shader(vertex_shader, ())
                .fragment_shader(fragment_shader, ())
                // Input definition.
                .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
                .input_assembly_state(
                    InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
                )
                // The various settings: this part is where you should pay attention to.
                .rasterization_state(RasterizationState {
                    // Directional culling. We enable backface culling for the geometry subpass.
                    // You don't want culling if you are using fullscreen triangle for post processing,
                    // and will want frontface culling for shadow maps.
                    cull_mode: StateMode::Fixed(CullMode::Back),
                    front_face: StateMode::Fixed(FrontFace::CounterClockwise),
                    ..Default::default()
                })
                .viewport_state(
                    // Scissor testing. Most of the time you won't need that.
                    // https://gamedev.stackexchange.com/questions/40704/what-is-the-purpose-of-glscissor
                    ViewportState::viewport_dynamic_scissor_irrelevant(),
                )
                .color_blend_state(
                    // Sets how the output color are blended onto the "canvas".
                    // Think Photoshop layer blending mode + alpha, but not as powerful.
                    // (although there is an extension for more modes,
                    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_blend_operation_advanced.html
                    // https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_EXT_blend_operation_advanced
                    // it's not supported by Vulkano yet. And probably not useful for PBR 3D.)
                    ColorBlendState::new(subpass.num_color_attachments()), // We just new()'ed an DISABLED (direct replace) blend state for ALL the outputs.
                                                                           // To specify separated or other blend states, remember to modify this.
                )
                .depth_stencil_state(
                    // Depth test. We almost always want to enable it for everything 3D.
                    // TODO We're enabling a forward (less = nearer) depth test here,
                    // remember to change this if you flipped the Z-buffer direction.
                    DepthStencilState::simple_depth_test(),
                )
                .build(vk.device())?
        };

        // We'll need the textures managed by the AssetLibrary in geometry subpass.
        let descriptor_set = PersistentDescriptorSet::new(
            pipeline
                .layout()
                .descriptor_set_layouts()
                .get(0)
                .ok_or_else(|| {
                    anyhow!("Geometry pipeline has no available descriptor set layouts")
                })?
                .clone(),
            library.descriptor_writes(),
        )?;

        Ok(Self {
            pipeline,
            descriptor_set,
        })
    }

    pub fn build_command_buffer(
        &self,
        command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        viewport: Viewport,
        library: Arc<AssetLibrary>,
        scene: &Scene,
    ) -> anyhow::Result<()> {
        command_buffer_builder
            // For each subpass, we need to first bind the pipeline.
            .bind_pipeline_graphics(self.pipeline.clone())
            // And the descriptor sets if any.
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.descriptor_set.clone(),
            );

        let view_transform = scene.camera.view_transform();
        let projection_transform = scene
            .camera
            .projection_transform(viewport.dimensions[0] / viewport.dimensions[1]);

        let mut id = 1; // The object ID counter. Reserving 0 for background.
        for (object, transform) in scene.objects.iter() {
            let object = library
                .models
                .get(object)
                .ok_or_else(|| anyhow!("Model {} doesn't exist", object))?;
            command_buffer_builder
                .push_constants(
                    self.pipeline.layout().clone(),
                    0,
                    PushConstants {
                        model: *transform.matrix(),
                        view: view_transform,
                        projection: projection_transform,
                        id,
                        base_color: object.base_color,
                    },
                )
                .bind_vertex_buffers(0, object.vertex_buffer.clone())
                .bind_index_buffer(object.index_buffer.clone())
                .draw_indexed(
                    object.index_buffer.len() as u32,
                    1, // Note how "instance count" is 1 even if we aren't instancing at all
                    0,
                    0,
                    0,
                )?;
            id += 1;
        }

        Ok(()) // We don't need to return anything,
               // all the command buffer builder methods are mutating the original builder.
    }
}

// These are black magic.
// They also automatically generate the various pipeline descriptions so
// a) you don't have to write the boilerplate yourself
// b) the pipeline is guaranteed to be compatible with the shaders,
// but then unfortunately you can't implement shader hot swapping.
mod vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/bin/object_oriented_deferred_cube/renderer/passes/main_pass/geometry_subpass.vert"
    }
}
mod frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/object_oriented_deferred_cube/renderer/passes/main_pass/geometry_subpass.frag"
    }
}
