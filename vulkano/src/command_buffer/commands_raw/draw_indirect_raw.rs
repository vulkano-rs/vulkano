// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use buffer::BufferAccess;
use command_buffer::CommandAddError;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use VulkanObject;
use vk;

pub struct CmdDrawIndirectRaw<B> {
    buffer: B,
    draw_count: u32,
    stride: u32,
}

impl<B> CmdDrawIndirectRaw<B> where B: BufferAccess {
    #[inline]
    pub unsafe fn new(buffer: B, draw_count: u32) -> CmdDrawIndirectRaw<B> {
        assert_eq!(buffer.inner().offset % 4, 0);

        // FIXME: all checks are missing here

        CmdDrawIndirectRaw {
            buffer: buffer,
            draw_count: draw_count,
            stride: 16,         // TODO:
        }
    }
}

impl<B> CmdDrawIndirectRaw<B> {
    /// Returns the buffer that contains the indirect command.
    #[inline]
    pub fn buffer(&self) -> &B {
        &self.buffer
    }
}

unsafe impl<B> DeviceOwned for CmdDrawIndirectRaw<B>
    where B: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

unsafe impl<'a, B, P> AddCommand<&'a CmdDrawIndirectRaw<B>> for UnsafeCommandBufferBuilder<P>
    where B: BufferAccess,
          P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdDrawIndirectRaw<B>) -> Result<Self::Out, CommandAddError> {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdDrawIndirect(cmd, command.buffer.inner().buffer.internal_object(),
                               command.buffer.inner().offset as vk::DeviceSize,
                               command.draw_count, command.stride);
        }

        Ok(self)
    }
}
