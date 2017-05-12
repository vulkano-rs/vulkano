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
use pipeline::vertex::VertexSource;
use VulkanObject;
use VulkanPointers;
use vk;

/// Command that binds vertex buffers to a command buffer.
pub struct CmdBindVertexBuffers<B> {
    // Actual raw state of the command.
    state: CmdBindVertexBuffersHash,
    // Offset within `state` to start binding.
    first_binding: u32,
    // Number of bindings to pass to the command.
    num_bindings: u32,
    // The device of the buffer, so that we can compare it with the command buffer's device.
    device: Arc<Device>,
    // The buffers to bind. Unused, but we need to keep it alive.
    buffers: B,
}

/// A "hash" of the bind vertex buffers command. Can be compared with a previous hash to determine
/// if two commands are identical.
///
/// > **Note**: This is not *actually* a hash, because there's no collision. If two objects are
/// > equal, then the commands are always identical.
#[derive(Clone, PartialEq, Eq)]
pub struct CmdBindVertexBuffersHash {
    // Raw handles of the buffers to bind.
    raw_buffers: SmallVec<[vk::Buffer; 4]>,
    // Raw offsets of the buffers to bind.
    offsets: SmallVec<[vk::DeviceSize; 4]>,
}

impl<B> CmdBindVertexBuffers<B> {
    /// Builds the command.
    #[inline]
    pub fn new<S>(source_def: &S, buffers: B) -> CmdBindVertexBuffers<B>
        where S: VertexSource<B>
    {
        let (device, raw_buffers, offsets) = {
            let (buffers, _, _) = source_def.decode(&buffers);

            let device = buffers.first().unwrap().buffer.device().clone();
            let raw_buffers: SmallVec<_> = buffers.iter().map(|b| b.buffer.internal_object()).collect();
            let offsets = buffers.iter().map(|b| b.offset as vk::DeviceSize).collect();

            (device, raw_buffers, offsets)
        };

        let num_bindings = raw_buffers.len() as u32;

        CmdBindVertexBuffers {
            state: CmdBindVertexBuffersHash {
                raw_buffers: raw_buffers,
                offsets: offsets,
            },
            first_binding: 0,
            num_bindings: num_bindings,
            device: device,
            buffers: buffers,
        }
    }

    /// Returns a hash that represents the command.
    #[inline]
    pub fn hash(&self) -> &CmdBindVertexBuffersHash {
        &self.state
    }

    /// Modifies the command so that it doesn't bind vertex buffers that were already bound by a
    /// previous command with the given hash.
    ///
    /// Note that this doesn't modify the hash of the command.
    pub fn diff(&mut self, previous_hash: &CmdBindVertexBuffersHash) {
        // We don't want to split the command into multiple ones, so we just trim the list of
        // vertex buffers at the start and at the end.
        let left_trim = self.state.raw_buffers
            .iter()
            .zip(self.state.offsets.iter())
            .zip(previous_hash.raw_buffers.iter())
            .zip(previous_hash.offsets.iter())
            .position(|(((&cur_buf, &cur_off), &prev_buf), &prev_off)| {
                cur_buf != prev_buf || cur_off != prev_off
            })
            .map(|p| p as u32)
            .unwrap_or(self.num_bindings);

        let right_trim = self.state.raw_buffers
            .iter()
            .zip(self.state.offsets.iter().rev())
            .zip(previous_hash.raw_buffers.iter().rev())
            .zip(previous_hash.offsets.iter().rev())
            .position(|(((&cur_buf, &cur_off), &prev_buf), &prev_off)| {
                cur_buf != prev_buf || cur_off != prev_off
            })
            .map(|p| p as u32)
            .unwrap_or(self.num_bindings);

        self.first_binding = left_trim;
        debug_assert!(left_trim <= self.state.raw_buffers.len() as u32);
        self.num_bindings = (self.state.raw_buffers.len() as u32 - left_trim).saturating_sub(right_trim);
    }
}

unsafe impl<B> DeviceOwned for CmdBindVertexBuffers<B> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<'a, P, B> AddCommand<&'a CmdBindVertexBuffers<B>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdBindVertexBuffers<B>) -> Result<Self::Out, CommandAddError> {
        unsafe {
            debug_assert_eq!(command.state.offsets.len(), command.state.raw_buffers.len());
            debug_assert!(command.num_bindings <= command.state.raw_buffers.len() as u32);

            if command.num_bindings == 0 {
                return Ok(self);
            }

            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdBindVertexBuffers(cmd, command.first_binding,
                                    command.num_bindings,
                                    command.state.raw_buffers[command.first_binding as usize..].as_ptr(),
                                    command.state.offsets[command.first_binding as usize..].as_ptr());
        }

        Ok(self)
    }
}
