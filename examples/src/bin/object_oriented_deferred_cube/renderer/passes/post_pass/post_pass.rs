use std::sync::Arc;

use anyhow::anyhow;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
};

use vulkano::device::Queue;
use vulkano::format::{ClearValue, Format};
use vulkano::image::view::ImageView;
use vulkano::image::SwapchainImage;
use vulkano::ordered_passes_renderpass;
use vulkano::pipeline::graphics::viewport::Viewport;

use vulkano::render_pass::{Framebuffer, RenderPass};

use crate::renderer::gbuffers::GBuffers;
use crate::renderer::passes::post_pass::outline_and_tonemapping_subpass::OutlineAndToneMappingSubpass;

use crate::VulkanContext;

/// This is the post-processing pass.
/// If your post processing is pixel-local (e.g. just turning the screen sepia),
/// you could also do your post processing in the main pass as a subpass,
/// but that's usually not the case (since almost every effect requires adjacent pixels).
/// It contains only one subpass, but you can extend it with more.
pub struct PostPass {
    render_pass: Arc<RenderPass>,
    /// We need a separate `Framebuffer` for each swapchain image.
    framebuffers: Vec<Arc<Framebuffer>>,
    outline_subpass: OutlineAndToneMappingSubpass,
}

impl PostPass {
    pub fn new<TWindow: Send + Sync + 'static>(
        vk: VulkanContext<TWindow>,
        gbuffers: Arc<GBuffers>,
        swapchain_images: Vec<Arc<SwapchainImage<TWindow>>>,
        swapchain_format: Format,
    ) -> anyhow::Result<Self> {
        let render_pass = ordered_passes_renderpass!(vk.device(),
            attachments: {
                swapchain: {
                    // Since this is the first and final subpass, and we're overwriting every pixel,
                    // we'll just use DontCare.
                    load: DontCare, // TODO Don't forget to replace with Load for subsequent subpasses!
                    store: Store, // Swapchain images need to be stored even in the final subpass, or you are presenting nothing to screen
                    format: swapchain_format,
                    samples: 1,
                }
                // We don't need all the other G-buffers here since we're passing them in
                // as storage images, so their definitions will reside in the descriptor set.
            },
            passes: [
                {
                    color: [swapchain],
                    depth_stencil: {},
                    input: []
                }
            ]
        )?;

        // Create a framebuffer for each swapchain image.
        let framebuffers = swapchain_images
            .into_iter()
            .map(|image| -> anyhow::Result<_> {
                // An ImageView is needed for swapchain images (or any image)
                // to be used in a Framebuffer.
                let view = ImageView::new(image.clone())?;
                Ok(Framebuffer::start(render_pass.clone()).add(view)?.build()?)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let outline_subpass = OutlineAndToneMappingSubpass::new(
            vk.clone(),
            render_pass.clone().first_subpass(),
            gbuffers,
        )?;

        Ok(Self {
            render_pass,
            framebuffers,
            outline_subpass,
        })
    }

    /// Build a command buffer for rendering the current frame.
    pub fn build_command_buffer(
        &self,
        queue: Arc<Queue>,
        viewport: Viewport,
        swapchain_index: usize,
    ) -> anyhow::Result<PrimaryAutoCommandBuffer> {
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            queue.device().clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer_builder
            .begin_render_pass(
                // Gets the corresponding framebuffer to the currently acquired swapchain image.
                self.framebuffers
                    .get(swapchain_index)
                    .ok_or_else(|| {
                        anyhow!(
                            "Swapchain image index out of bounds (index: {}, count: {})",
                            swapchain_index,
                            self.framebuffers.len()
                        )
                    })?
                    .clone(),
                SubpassContents::Inline,
                [ClearValue::None],
            )?
            .set_viewport(0, [viewport.clone()]);
        self.outline_subpass
            .build_command_buffer(&mut command_buffer_builder)?;
        command_buffer_builder.end_render_pass()?;

        Ok(command_buffer_builder.build()?)
    }
}
