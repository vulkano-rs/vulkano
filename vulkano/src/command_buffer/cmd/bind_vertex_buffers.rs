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

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use pipeline::vertex::Source;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds a command that binds an index buffer at the end of it.
pub struct CmdBindVertexBuffers<B> {
    // Raw handles of the buffers to bind.
    raw_buffers: SmallVec<[vk::Buffer; 4]>,
    // Raw offsets of the buffers to bind.
    offsets: SmallVec<[vk::DeviceSize; 4]>,
    // The device of the buffer, so that we can compare it with the command buffer's device.
    device: Arc<Device>,
    // The buffers to bind. Unused, but we need to keep it alive.
    buffers: B,
}

impl<B> CmdBindVertexBuffers<B> {
    /// Builds the command.
    #[inline]
    pub fn new<S>(source_def: &S, buffers: B) -> CmdBindVertexBuffers<B>
        where S: Source<B>
    {
        let (device, raw_buffers, offsets) = {
            let (buffers, _, _) = source_def.decode(&buffers);

            let device = buffers.first().unwrap().buffer.device().clone();
            let raw_buffers = buffers.iter().map(|b| b.buffer.internal_object()).collect();
            let offsets = buffers.iter().map(|b| b.offset as vk::DeviceSize).collect();

            (device, raw_buffers, offsets)
        };

        // Note that we don't check for collisions between vertex buffers, because there shouldn't
        // be any.

        CmdBindVertexBuffers {
            raw_buffers: raw_buffers,
            offsets: offsets,
            device: device,
            buffers: buffers,
        }
    }
}

unsafe impl<'a, P, B> AddCommand<&'a CmdBindVertexBuffers<B>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdBindVertexBuffers<B>) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdBindVertexBuffers(cmd, 0, command.raw_buffers.len() as u32,
                                    command.raw_buffers.as_ptr(), command.offsets.as_ptr());
        }

        self
    }
}
