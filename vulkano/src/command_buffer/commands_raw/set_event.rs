// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use command_buffer::CommandAddError;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use sync::Event;
use VulkanObject;
use VulkanPointers;
use vk;

/// Command that sets or resets an event.
#[derive(Debug, Clone)]
pub struct CmdSetEvent {
    // The event to set or reset.
    event: Arc<Event>,
    // The pipeline stages after which the event should be set or reset.
    stages: vk::PipelineStageFlags,
    // If true calls `vkCmdSetEvent`, otherwise `vkCmdSetEvent`.
    set: bool,
}

// TODO: add constructor

unsafe impl DeviceOwned for CmdSetEvent {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.event.device()
    }
}

unsafe impl<'a, P> AddCommand<&'a CmdSetEvent> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdSetEvent) -> Result<Self::Out, CommandAddError> {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            if command.set {
                vk.CmdSetEvent(cmd, command.event.internal_object(), command.stages);
            } else {
                vk.CmdResetEvent(cmd, command.event.internal_object(), command.stages);
            }
        }

        Ok(self)
    }
}
