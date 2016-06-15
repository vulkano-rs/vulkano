// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::ptr;
use std::sync::Arc;

use smallvec::SmallVec;

use command_buffer::pool::CommandPool;
use command_buffer::sys::KeepAlive;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use format::ClearValue;
use framebuffer::RenderPass;
use framebuffer::RenderPassCompatible;
use framebuffer::Framebuffer;

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a command that fills a buffer with data.
pub struct BeginRenderPassCommand {
    keep_alive1: Arc<KeepAlive + 'static>,
    keep_alive2: Arc<KeepAlive + 'static>,

    device: vk::Device,

    secondary: bool,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    clear_values: SmallVec<[vk::ClearValue; 4]>,
}

impl BeginRenderPassCommand {
    /// Creates a command that starts a render pass.
    ///
    /// # Panic
    ///
    /// - Panicks if the framebuffer is not compatible with the render pass.
    ///
    pub fn new<R, F, I>(render_pass: &Arc<R>, framebuffer: &Arc<Framebuffer<F>>, clear_values: I,
                        secondary: bool) -> Result<BeginRenderPassCommand, BeginRenderPassError>
        where R: RenderPass + 'static,
              F: RenderPass + RenderPassCompatible<R> + 'static,
              I: Iterator<Item = ClearValue>,
    {
        let device = framebuffer.device().internal_object();
        assert_eq!(device, render_pass.render_pass().device().internal_object());

        if !framebuffer.is_compatible_with(render_pass) {
            return Err(BeginRenderPassError::IncompatibleRenderPass);
        }

        let clear_values = clear_values.map(|value| {
            match value {
                ClearValue::None => vk::ClearValue::color({
                    vk::ClearColorValue::float32([0.0, 0.0, 0.0, 0.0])
                }),
                ClearValue::Float(data) => vk::ClearValue::color(vk::ClearColorValue::float32(data)),
                ClearValue::Int(data) => vk::ClearValue::color(vk::ClearColorValue::int32(data)),
                ClearValue::Uint(data) => vk::ClearValue::color(vk::ClearColorValue::uint32(data)),
                ClearValue::Depth(d) => vk::ClearValue::depth_stencil({
                    vk::ClearDepthStencilValue { depth: d, stencil: 0 }
                }),
                ClearValue::Stencil(s) => vk::ClearValue::depth_stencil({
                    vk::ClearDepthStencilValue { depth: 0.0, stencil: s }
                }),
                ClearValue::DepthStencil((d, s)) => vk::ClearValue::depth_stencil({
                    vk::ClearDepthStencilValue { depth: d, stencil: s }
                }),
            }
        }).collect();

        Ok(BeginRenderPassCommand {
            keep_alive1: render_pass.clone() as Arc<_>,
            keep_alive2: framebuffer.clone() as Arc<_>,
            device: device,
            secondary: secondary,
            render_pass: render_pass.render_pass().internal_object(),
            framebuffer: framebuffer.internal_object(),
            x: 0,           // TODO: leave the choice to the user
            y: 0,           // TODO: leave the choice to the user
            width: framebuffer.dimensions()[0],         // TODO: leave the choice to the user
            height: framebuffer.dimensions()[1],        // TODO: leave the choice to the user
            clear_values: clear_values,
        })
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the command buffer is within a render pass.
    /// - Panicks if the render pass or framebuffer was not allocated with the same device as the
    ///   command buffer.
    /// - Panicks if the queue family does not support graphics operations.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            let _pool_lock = cb.pool().lock();

            // Various checks.
            assert!(!cb.within_render_pass);
            assert_eq!(self.device, cb.device().internal_object());
            assert!(cb.pool().queue_family().supports_graphics());

            let flags = if self.secondary { vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS }
                        else { vk::SUBPASS_CONTENTS_INLINE };

            let infos = vk::RenderPassBeginInfo {
                sType: vk::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                pNext: ptr::null(),
                renderPass: self.render_pass,
                framebuffer: self.framebuffer,
                renderArea: vk::Rect2D {
                    offset: vk::Offset2D { x: self.x as i32, y: self.y as i32 },
                    extent: vk::Extent2D { width: self.width, height: self.height },
                },
                clearValueCount: self.clear_values.len() as u32,
                pClearValues: self.clear_values.as_ptr(),
            };

            cb.keep_alive.push(self.keep_alive1.clone());
            cb.keep_alive.push(self.keep_alive2.clone());

            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();
                vk.CmdBeginRenderPass(cmd, &infos, flags);
            }

            cb
        }
    }
}

error_ty!{BeginRenderPassError => "Error that can happen when beginning a render pass.",
    IncompatibleRenderPass => "the framebuffer is not compatible with the render pass",
}

/// Prototype for a command that switches to the next subpass.
pub struct NextSubpassCommand {
    secondary: bool,
}

impl NextSubpassCommand {
    /// Builds a `NextSubpassCommand`.
    #[inline]
    pub fn new(secondary: bool) -> NextSubpassCommand {
        NextSubpassCommand {
            secondary: secondary,
        }
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the command buffer is not within a render pass.
    /// - Panicks if the queue family does not support graphics operations.
    ///
    pub fn submit<P>(&mut self, cb: UnsafeCommandBufferBuilder<P>) -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            let _pool_lock = cb.pool().lock();

            // Various checks.
            assert!(cb.within_render_pass);
            assert!(cb.pool().queue_family().supports_graphics());

            let flags = if self.secondary { vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS }
                        else { vk::SUBPASS_CONTENTS_INLINE };

            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();
                vk.CmdNextSubpass(cmd, flags);
            }

            cb
        }
    }
}

/// Prototype for a command that ends the render pass.
pub struct EndRenderPassCommand;

impl EndRenderPassCommand {
    /// Builds a `EndRenderPassCommand`.
    #[inline]
    pub fn new() -> EndRenderPassCommand {
        EndRenderPassCommand
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the command buffer is not within a render pass.
    /// - Panicks if the queue family does not support graphics operations.
    ///
    pub fn submit<P>(&mut self, cb: UnsafeCommandBufferBuilder<P>) -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            let _pool_lock = cb.pool().lock();

            // Various checks.
            assert!(cb.within_render_pass);
            assert!(cb.pool().queue_family().supports_graphics());

            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();
                vk.CmdEndRenderPass(cmd);
            }

            cb
        }
    }
}
