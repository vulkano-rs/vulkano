// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    pixels_draw_pipeline::PixelsDrawPipeline,
    renderer::{FinalImageView, InterimImageView},
};
use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    device::Queue,
    format::Format,
    image::ImageAccess,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};

/// A render pass which places an incoming image over frame filling it
pub struct RenderPassPlaceOverFrame {
    gfx_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pixels_draw_pipeline: PixelsDrawPipeline,
}

impl RenderPassPlaceOverFrame {
    pub fn new(gfx_queue: Arc<Queue>, output_format: Format) -> RenderPassPlaceOverFrame {
        let render_pass = vulkano::single_pass_renderpass!(gfx_queue.device().clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: output_format,
                    samples: 1,
                }
            },
            pass: {
                    color: [color],
                    depth_stencil: {}
            }
        )
        .unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let pixels_draw_pipeline = PixelsDrawPipeline::new(gfx_queue.clone(), subpass);
        RenderPassPlaceOverFrame {
            gfx_queue,
            render_pass,
            pixels_draw_pipeline,
        }
    }

    /// Place view exactly over swapchain image target.
    /// Texture draw pipeline uses a quad onto which it places the view.
    pub fn render<F>(
        &mut self,
        before_future: F,
        view: InterimImageView,
        target: FinalImageView,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        // Get dimensions
        let img_dims = target.image().dimensions();
        // Create framebuffer (must be in same order as render pass description in `new`
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![target],
                ..Default::default()
            },
        )
        .unwrap();
        // Create primary command buffer builder
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        // Begin render pass
        command_buffer_builder
            .begin_render_pass(
                framebuffer,
                SubpassContents::SecondaryCommandBuffers,
                vec![[0.0; 4].into()],
            )
            .unwrap();
        // Create secondary command buffer from texture pipeline & send draw commands
        let cb = self
            .pixels_draw_pipeline
            .draw(img_dims.width_height(), view);
        // Execute above commands (subpass)
        command_buffer_builder.execute_commands(cb).unwrap();
        // End render pass
        command_buffer_builder.end_render_pass().unwrap();
        // Build command buffer
        let command_buffer = command_buffer_builder.build().unwrap();
        // Execute primary command buffer
        let after_future = before_future
            .then_execute(self.gfx_queue.clone(), command_buffer)
            .unwrap();

        after_future.boxed()
    }
}
