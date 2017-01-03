// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds to the end of it a command that goes to the next subpass
/// of the current render pass.
#[derive(Debug, Copy, Clone)]
pub struct CmdNextSubpass{
    // The parameter for vkCmdNextSubpass.
    contents: vk::SubpassContents,
}

impl CmdNextSubpass {
    /// See the documentation of the `next_subpass` method.
    #[inline]
    pub fn new(secondary: bool) -> CmdNextSubpass {
        CmdNextSubpass {
            contents: if secondary { vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS }
                      else { vk::SUBPASS_CONTENTS_INLINE }
        }
    }
}

unsafe impl<'a, P> AddCommand<&'a CmdNextSubpass> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdNextSubpass) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdNextSubpass(cmd, command.contents);
        }

        self
    }
}
