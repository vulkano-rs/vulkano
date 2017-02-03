// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use buffer::Buffer;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;
use vk;

pub struct CmdDrawIndirectRaw<B> {
    buffer: B,
    offset: vk::DeviceSize,
    draw_count: u32,
    stride: u32,
}

impl<B> CmdDrawIndirectRaw<B> where B: Buffer {
    #[inline]
    pub unsafe fn new(buffer: B, offset: usize, draw_count: u32) -> CmdDrawIndirectRaw<B> {
        let real_offset = offset + buffer.inner().offset;
        assert_eq!(real_offset % 4, 0);

        // FIXME: all checks are missing here

        CmdDrawIndirectRaw {
            buffer: buffer,
            offset: real_offset as vk::DeviceSize,
            draw_count: draw_count,
            stride: 16,         // TODO:
        }
    }
}

unsafe impl<'a, B, P> AddCommand<&'a CmdDrawIndirectRaw<B>> for UnsafeCommandBufferBuilder<P>
    where B: Buffer,
          P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdDrawIndirectRaw<B>) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdDrawIndirect(cmd, command.buffer.inner().buffer.internal_object(),
                               command.offset, command.draw_count, command.stride);
        }

        self
    }
}
