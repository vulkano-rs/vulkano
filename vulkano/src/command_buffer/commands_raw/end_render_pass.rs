// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use command_buffer::CommandAddError;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;

/// Command that exits the current render pass.
#[derive(Debug, Copy, Clone)]
pub struct CmdEndRenderPass;

impl CmdEndRenderPass {
    /// See the documentation of the `end_render_pass` method.
    #[inline]
    pub fn new() -> CmdEndRenderPass {
        CmdEndRenderPass
    }
}

unsafe impl<'a, P> AddCommand<&'a CmdEndRenderPass> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdEndRenderPass) -> Result<Self::Out, CommandAddError> {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdEndRenderPass(cmd);
        }

        Ok(self)
    }
}
