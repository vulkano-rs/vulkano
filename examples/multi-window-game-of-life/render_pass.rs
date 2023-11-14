use crate::{app::App, pixels_draw::PixelsDrawPipeline};
use std::sync::Arc;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    device::Queue,
    format::Format,
    image::view::ImageView,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};

/// A render pass which places an incoming image over the frame, filling it.
pub struct RenderPassPlaceOverFrame {
    gfx_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pixels_draw_pipeline: PixelsDrawPipeline,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl RenderPassPlaceOverFrame {
    pub fn new(
        app: &App,
        gfx_queue: Arc<Queue>,
        output_format: Format,
    ) -> RenderPassPlaceOverFrame {
        let render_pass = vulkano::single_pass_renderpass!(
            gfx_queue.device().clone(),
            attachments: {
                color: {
                    format: output_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let pixels_draw_pipeline = PixelsDrawPipeline::new(app, gfx_queue.clone(), subpass);

        RenderPassPlaceOverFrame {
            gfx_queue,
            render_pass,
            pixels_draw_pipeline,
            command_buffer_allocator: app.command_buffer_allocator.clone(),
        }
    }

    /// Places the view exactly over the target swapchain image. The texture draw pipeline uses a
    /// quad onto which it places the view.
    pub fn render<F>(
        &self,
        before_future: F,
        image_view: Arc<ImageView>,
        target: Arc<ImageView>,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        // Get the dimensions.
        let img_dims: [u32; 2] = target.image().extent()[0..2].try_into().unwrap();

        // Create the framebuffer.
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![target],
                ..Default::default()
            },
        )
        .unwrap();

        // Create a primary command buffer builder.
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Begin the render pass.
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0; 4].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                },
            )
            .unwrap();

        // Create a secondary command buffer from the texture pipeline & send draw commands.
        let cb = self.pixels_draw_pipeline.draw(img_dims, image_view);

        // Execute above commands (subpass).
        command_buffer_builder.execute_commands(cb).unwrap();

        // End the render pass.
        command_buffer_builder
            .end_render_pass(Default::default())
            .unwrap();

        // Build the command buffer.
        let command_buffer = command_buffer_builder.build().unwrap();

        // Execute primary command buffer.
        let after_future = before_future
            .then_execute(self.gfx_queue.clone(), command_buffer)
            .unwrap();

        after_future.boxed()
    }
}
