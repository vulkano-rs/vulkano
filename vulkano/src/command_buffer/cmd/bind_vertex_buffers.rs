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

use buffer::TrackedBuffer;
use buffer::TypedBuffer;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use device::Device;
use pipeline::vertex::Source;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds a command that binds an index buffer at the end of it.
pub struct CmdBindVertexBuffers<L, B> where L: CommandsList {
    // Parent commands list.
    previous: L,
    // Raw handles of the buffers to bind.
    raw_buffers: SmallVec<[vk::Buffer; 4]>,
    // Raw offsets of the buffers to bind.
    offsets: SmallVec<[vk::DeviceSize; 4]>,
    // The device of the buffer, so that we can compare it with the command buffer's device.
    device: Arc<Device>,
    // The buffers to bind. Unused, but we need to keep it alive.
    buffers: B,
}

impl<L, B> CmdBindVertexBuffers<L, B> where L: CommandsList {
    /// Builds the command.
    #[inline]
    pub fn new<S>(previous: L, source_def: &S, buffers: B) -> CmdBindVertexBuffers<L, B>
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
            previous: previous,
            raw_buffers: raw_buffers,
            offsets: offsets,
            device: device,
            buffers: buffers,
        }
    }
}

unsafe impl<L, B> CommandsList for CmdBindVertexBuffers<L, B> where L: CommandsList {
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        assert_eq!(self.device.internal_object(), builder.device().internal_object());
        debug_assert_eq!(self.raw_buffers.len(), self.offsets.len());

        // FIXME: perform buffer transitions

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();
                // TODO: don't bind if not necessary
                vk.CmdBindVertexBuffers(cmd, 0, self.raw_buffers.len() as u32,
                                        self.raw_buffers.as_ptr(), self.offsets.as_ptr());
            }
        }));
    }
}
