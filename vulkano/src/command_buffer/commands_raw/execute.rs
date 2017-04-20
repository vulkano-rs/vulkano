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

use command_buffer::CommandAddError;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;
use vk;

/// Command that executes a secondary command buffer.
pub struct CmdExecuteCommands<Cb> {
    // Raw list of command buffers to execute.
    raw_list: SmallVec<[vk::CommandBuffer; 4]>,
    // Command buffer to execute.
    command_buffer: Cb,
}

impl<Cb> CmdExecuteCommands<Cb> {
    /// See the documentation of the `execute_commands` method.
    #[inline]
    pub fn new(command_buffer: Cb) -> CmdExecuteCommands<Cb> {
        unimplemented!()        // TODO:
        /*let raw_list = {
            let mut l = SmallVec::new();
            l.push(command_buffer.inner());
            l
        };

        CmdExecuteCommands {
            raw_list: raw_list,
            command_buffer: command_buffer,
        }*/
    }
}

unsafe impl<Cb> DeviceOwned for CmdExecuteCommands<Cb>
    where Cb: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.command_buffer.device()
    }
}

unsafe impl<'a, P, Cb> AddCommand<&'a CmdExecuteCommands<Cb>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdExecuteCommands<Cb>) -> Result<Self::Out, CommandAddError> {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdExecuteCommands(cmd, command.raw_list.len() as u32, command.raw_list.as_ptr());
        }

        Ok(self)
    }
}
