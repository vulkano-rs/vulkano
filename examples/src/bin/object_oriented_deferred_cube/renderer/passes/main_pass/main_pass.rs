use std::sync::Arc;

use anyhow::anyhow;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
};

use vulkano::device::Queue;
use vulkano::image::ImageViewAbstract;
use vulkano::ordered_passes_renderpass;
use vulkano::pipeline::graphics::viewport::Viewport;

use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};

use crate::renderer::asset_library::AssetLibrary;
use crate::renderer::gbuffers::{GBuffers, CLEAR_VALUES};
use crate::renderer::passes::main_pass::geometry_subpass::GeometrySubpass;
use crate::renderer::passes::main_pass::lighting_subpass::LightingSubpass;
use crate::scene::scene::Scene;
use crate::VulkanContext;

/// This is the main pass.
/// This struct is bound to a specific `Device` and `AssetLibrary` reference,
/// but for simplicity we'll just assume these are immutable.
/// It draws what the user sees from their perspective, before the post processing.
/// It contains two subpasses: the geometry subpass and the lighting subpass.
pub struct MainPass {
    /// This object itself defines *how* data is passed around during a render pass.
    render_pass: Arc<RenderPass>,
    library: Arc<AssetLibrary>,
    /// "Framebuffer" tells the GPU *which* data to use.
    /// It's neither the handle to the actual data (although containing them),
    /// nor by any means related to the screen (that's the swapchain you are looking for).
    framebuffer: Arc<Framebuffer>,
    /// For more information on the subpasses, read their respective files.
    geometry_subpass: GeometrySubpass,
    lighting_subpass: LightingSubpass,
}

impl MainPass {
    pub fn new<TWindow: Send + Sync + 'static>(
        vk: VulkanContext<TWindow>,
        gbuffers: Arc<GBuffers>,
        library: Arc<AssetLibrary>,
    ) -> anyhow::Result<Self> {
        // Black magic macro warning! ...at least this one is just a macro_rules! macro.
        let render_pass = ordered_passes_renderpass!(vk.device(),
            attachments: { // "Attachments" enumerates all the attachment image used.
                position: {
                    load: Clear, // We'll need to clean up what's left by the last frame here.
                    store: Store, // We need to store all the G-buffers so they can be used by the next pass...
                    format: gbuffers.position_buffer.format(),
                    samples: 1,
                },
                normal: {
                    load: Clear,
                    store: Store,
                    format: gbuffers.normal_buffer.format(),
                    samples: 1,
                },
                base_color: {
                    load: Clear,
                    store: Store,
                    format: gbuffers.base_color_buffer.format(),
                    samples: 1,
                },
                id: {
                    load: Clear,
                    store: Store,
                    format: gbuffers.id_buffer.format(),
                    samples: 1,
                },
                composite: {
                    load: Clear,
                    store: Store,
                    format: gbuffers.composite_buffer.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare, // ...except this one. It's pure trash after depth test.
                    format: gbuffers.depth_buffer.format(),
                    samples: 1,
                }
            },
            passes: [
                { // The geometry pass.
                    color: [position, normal, base_color, id], // "Color" means the output, not necessarily color data.
                    depth_stencil: {depth},
                    input: []
                },
                { // The lighting pass.
                    color: [composite],
                    depth_stencil: {},
                    input: [position, normal, base_color, id]
                }
            ]
        )?;

        let framebuffer = Framebuffer::start(render_pass.clone())
            .add(gbuffers.position_buffer.clone())?
            .add(gbuffers.normal_buffer.clone())?
            .add(gbuffers.base_color_buffer.clone())?
            .add(gbuffers.id_buffer.clone())?
            .add(gbuffers.composite_buffer.clone())?
            .add(gbuffers.depth_buffer.clone())?
            .build()?;

        let geometry_subpass = GeometrySubpass::new(
            vk.clone(),
            render_pass.clone().first_subpass(),
            library.clone(),
        )?;

        let lighting_subpass = LightingSubpass::new(
            vk.clone(),
            Subpass::from(render_pass.clone(), 1)
                .ok_or_else(|| anyhow!("Main pass doesn't have enough subpasses"))?,
            gbuffers.clone(),
        )?;

        Ok(Self {
            render_pass,
            library,
            framebuffer,
            geometry_subpass,
            lighting_subpass,
        })
    }

    /// Build a command buffer for rendering the current frame.
    pub fn build_command_buffer(
        &self,
        queue: Arc<Queue>,
        viewport: Viewport,
        scene: &Scene,
    ) -> anyhow::Result<PrimaryAutoCommandBuffer> {
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            queue.device().clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer_builder
            // A render pass needs an explicit beginning...
            .begin_render_pass(
                self.framebuffer.clone(),
                SubpassContents::Inline,
                CLEAR_VALUES,
            )?
            .set_viewport(0, [viewport.clone()]);
        self.geometry_subpass.build_command_buffer(
            &mut command_buffer_builder,
            viewport.clone(),
            self.library.clone(),
            &scene,
        )?;
        // ...an explicit transitioning of subpasses...
        command_buffer_builder.next_subpass(SubpassContents::Inline)?;
        self.lighting_subpass
            .build_command_buffer(&mut command_buffer_builder, &scene)?;
        // ...and an explicit end.
        command_buffer_builder.end_render_pass()?;

        Ok(command_buffer_builder.build()?)
    }
}
