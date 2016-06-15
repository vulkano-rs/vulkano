// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use smallvec::SmallVec;

use command_buffer::CommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::sys::KeepAlive;
use command_buffer::sys::UnsafeCommandBufferBuilder;

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a command that switches to the next subpass.
pub struct ExecuteCommand {
    keep_alive: SmallVec<[Arc<KeepAlive>; 8]>,
    command_buffers: SmallVec<[vk::CommandBuffer; 8]>,
}

impl ExecuteCommand {
    /// Builds a `ExecuteCommand`.
    // FIXME: most checks are missing
    #[inline]
    pub fn new<I, C>(command_buffers: I) -> ExecuteCommand
        where I: Iterator<Item = Arc<C>>,
              C: CommandBuffer + 'static + Send + Sync,
    {
        let mut keep_alive = SmallVec::new();
        let mut raw_handles = SmallVec::new();
        for command_buffer in command_buffers {
            keep_alive.push(command_buffer.clone() as Arc<_>);
            raw_handles.push(command_buffer.inner_cb().internal_object());
        }

        ExecuteCommand {
            keep_alive: keep_alive,
            command_buffers: raw_handles,
        }
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the command buffer is not within a render pass.
    /// - Panicks if the queue family does not support graphics operations.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            let _pool_lock = cb.pool().lock();

            for ka in self.keep_alive.into_iter() {
                cb.keep_alive.push(ka);
            }

            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();
                vk.CmdExecuteCommands(cmd, self.command_buffers.len() as u32,
                                      self.command_buffers.as_ptr());
            }

            cb
        }
    }
}
