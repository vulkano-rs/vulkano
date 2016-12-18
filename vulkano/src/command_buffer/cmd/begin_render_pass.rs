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

use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use device::Device;
use format::ClearValue;
use framebuffer::AttachmentsList;
use framebuffer::FramebufferRef;
use framebuffer::RenderPass;
use framebuffer::RenderPassClearValues;
use framebuffer::RenderPassRef;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds to the end of it a command that enters a render pass.
pub struct CmdBeginRenderPass<L, Rp, F> where L: CommandsList {
    // Parent commands list.
    previous: L,
    // True if only secondary command buffers can be added.
    secondary: bool,
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

impl<L, F> CmdBeginRenderPass<L, Arc<RenderPass>, F>
    where L: CommandsList, F: FramebufferRef
{
    /// See the documentation of the `begin_render_pass` method.
    // TODO: allow setting more parameters
    pub fn new<C>(previous: L, framebuffer: F, secondary: bool, clear_values: C)
                  -> CmdBeginRenderPass<L, Arc<RenderPass>, F>
        where <<F as FramebufferRef>::RenderPass as RenderPassRef>::Desc: RenderPassClearValues<C>
    {
        let raw_render_pass = framebuffer.inner().render_pass().inner().internal_object();
        let device = framebuffer.inner().render_pass().inner().device().clone();
        let raw_framebuffer = framebuffer.inner().internal_object();

        let clear_values = {
            let desc = framebuffer.inner().render_pass().inner().desc();
            desc.convert_clear_values(clear_values).map(|clear_value| {
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

        let rect = [0 .. framebuffer.inner().dimensions()[0],
                    0 .. framebuffer.inner().dimensions()[1]];

        CmdBeginRenderPass {
            previous: previous,
            secondary: secondary,
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

unsafe impl<L, Rp, F> CommandsList for CmdBeginRenderPass<L, Rp, F>
    where L: CommandsList, F: FramebufferRef, F::Attachments: AttachmentsList
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        assert_eq!(self.device.internal_object(), builder.device().internal_object());

        debug_assert!(self.rect[0].start <= self.rect[0].end);
        debug_assert!(self.rect[1].start <= self.rect[1].end);

        self.framebuffer.inner().add_transition(builder);

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();

                let begin = vk::RenderPassBeginInfo {
                    sType: vk::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    pNext: ptr::null(),
                    renderPass: self.raw_render_pass,
                    framebuffer: self.raw_framebuffer,
                    renderArea: vk::Rect2D {
                        offset: vk::Offset2D {
                            x: self.rect[0].start as i32,
                            y: self.rect[1].start as i32,
                        },
                        extent: vk::Extent2D {
                            width: self.rect[0].end - self.rect[0].start,
                            height: self.rect[1].end - self.rect[1].start,
                        },
                    },
                    clearValueCount: self.clear_values.len() as u32,
                    pClearValues: self.clear_values.as_ptr(),
                };

                let contents = if self.secondary { vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS }
                               else { vk::SUBPASS_CONTENTS_INLINE };
                
                vk.CmdBeginRenderPass(cmd, &begin, contents);
            }
        }));
    }
}
