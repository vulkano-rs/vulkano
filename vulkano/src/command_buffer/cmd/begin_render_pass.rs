// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use std::ops::Range;
use std::ptr;
use smallvec::SmallVec;

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use format::ClearValue;
use framebuffer::FramebufferAbstract;
use framebuffer::RenderPass;
use framebuffer::RenderPassDescClearValues;
use framebuffer::RenderPassAbstract;
use VulkanObject;
use VulkanPointers;
use vk;

/// Command that makes the command buffer enter a render pass.
pub struct CmdBeginRenderPass<Rp, F> {
    // Inline or secondary.
    contents: vk::SubpassContents,
    // The draw area.
    rect: [Range<u32>; 2],
    // The clear values for the clear attachments.
    clear_values: SmallVec<[vk::ClearValue; 6]>,
    // The raw render pass handle to bind.
    raw_render_pass: vk::RenderPass,
    // The raw framebuffer handle to bind.
    raw_framebuffer: vk::Framebuffer,
    // The device.
    device: Arc<Device>,
    // The render pass. Can be `None` if same as framebuffer.
    render_pass: Option<Rp>,
    // The framebuffer.
    framebuffer: F,
}

impl<F> CmdBeginRenderPass<Arc<RenderPass>, F>
    where F: FramebufferAbstract
{
    /// See the documentation of the `begin_render_pass` method.
    // TODO: allow setting more parameters
    pub fn new<C>(framebuffer: F, secondary: bool, clear_values: C)
                  -> CmdBeginRenderPass<Arc<RenderPassAbstract>, F>
        where F: RenderPassDescClearValues<C>
    {
        let raw_render_pass = RenderPassAbstract::inner(&framebuffer).internal_object();
        let device = framebuffer.device().clone();
        let raw_framebuffer = FramebufferAbstract::inner(&framebuffer).internal_object();

        let clear_values = {
            framebuffer.convert_clear_values(clear_values).map(|clear_value| {
                match clear_value {
                    ClearValue::None => {
                        vk::ClearValue::color(vk::ClearColorValue::float32([0.0; 4]))
                    },
                    ClearValue::Float(val) => {
                        vk::ClearValue::color(vk::ClearColorValue::float32(val))
                    },
                    ClearValue::Int(val) => {
                        vk::ClearValue::color(vk::ClearColorValue::int32(val))
                    },
                    ClearValue::Uint(val) => {
                        vk::ClearValue::color(vk::ClearColorValue::uint32(val))
                    },
                    ClearValue::Depth(val) => {
                        vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                            depth: val, stencil: 0
                        })
                    },
                    ClearValue::Stencil(val) => {
                        vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                            depth: 0.0, stencil: val
                        })
                    },
                    ClearValue::DepthStencil((depth, stencil)) => {
                        vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                            depth: depth, stencil: stencil,
                        })
                    },
                }
            }).collect()
        };

        let rect = [0 .. framebuffer.dimensions()[0],
                    0 .. framebuffer.dimensions()[1]];

        CmdBeginRenderPass {
            contents: if secondary { vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS }
                      else { vk::SUBPASS_CONTENTS_INLINE },
            rect: rect,
            clear_values: clear_values,
            raw_render_pass: raw_render_pass,
            raw_framebuffer: raw_framebuffer,
            device: device,
            render_pass: None,
            framebuffer: framebuffer,
        }
    }
}

unsafe impl<'a, P, Rp, F> AddCommand<&'a CmdBeginRenderPass<Rp, F>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdBeginRenderPass<Rp, F>) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();

            let begin = vk::RenderPassBeginInfo {
                sType: vk::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                pNext: ptr::null(),
                renderPass: command.raw_render_pass,
                framebuffer: command.raw_framebuffer,
                renderArea: vk::Rect2D {
                    offset: vk::Offset2D {
                        x: command.rect[0].start as i32,
                        y: command.rect[1].start as i32,
                    },
                    extent: vk::Extent2D {
                        width: command.rect[0].end - command.rect[0].start,
                        height: command.rect[1].end - command.rect[1].start,
                    },
                },
                clearValueCount: command.clear_values.len() as u32,
                pClearValues: command.clear_values.as_ptr(),
            };

            vk.CmdBeginRenderPass(cmd, &begin, command.contents);
        }

        self
    }
}
