// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;
use std::sync::Arc;
use std::ops::Range;
use std::ptr;
use smallvec::SmallVec;

use command_buffer::StatesManager;
use command_buffer::SubmitInfo;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use device::Device;
use device::Queue;
use format::ClearValue;
use framebuffer::traits::TrackedFramebuffer;
use framebuffer::RenderPass;
use framebuffer::RenderPassClearValues;
use instance::QueueFamily;
use sync::Fence;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds to the end of it a command that goes to the next subpass
/// of the current render pass.
#[derive(Debug, Copy, Clone)]
pub struct CmdNextSubpass<L> where L: CommandsList {
    // Parent commands list.
    previous: L,
    // The parameter for vkCmdNextSubpass.
    contents: vk::SubpassContents,
}

impl<L> CmdNextSubpass<L> where L: CommandsList {
    /// See the documentation of the `next_subpass` method.
    #[inline]
    pub fn new(previous: L, secondary: bool) -> CmdNextSubpass<L> {
        // TODO: check that we're in a render pass and that the next subpass is correct

        CmdNextSubpass {
            previous: previous,
            contents: if secondary { vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS }
                      else { vk::SUBPASS_CONTENTS_INLINE },
        }
    }
}

unsafe impl<L> CommandsList for CmdNextSubpass<L> where L: CommandsList {
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();
                
                vk.CmdNextSubpass(cmd, self.contents);
            }
        }));
    }
}
