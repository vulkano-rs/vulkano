// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::sync::Arc;
use smallvec::SmallVec;

use buffer::Buffer;
use command_buffer::pool::CommandPool;
use command_buffer::sys::KeepAlive;
use command_buffer::sys::UnsafeCommandBufferBuilder;

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a command that binds vertex buffers.
pub struct VertexSourceBindCommand {
    // List of buffers to keep alive.
    keep_alive: SmallVec<[Arc<KeepAlive + 'static>; 8]>,

    // The device of the buffers, or 0 if the list of buffers is empty.
    device: vk::Device,

    // List of raw buffers to pass to `vkCmdBindVertexBuffers`.
    raw_buffers: SmallVec<[vk::Buffer; 8]>,
    // List of buffer offsets to pass to `vkCmdBindVertexBuffers`.
    offsets: SmallVec<[vk::DeviceSize; 8]>,
}

impl VertexSourceBindCommand {
    /// Builds a command that binds vertex buffers to a command buffer.
    ///
    /// Note that binding points N and greater, where N is the length of `buffers`, will be in an
    /// undeterminate state. In other words, the bindings outside of range of `buffers` are not
    /// unbinded.
    ///
    /// # Panic
    ///
    /// - Panicks if the buffers were not allocated with the same device.
    ///
    pub fn new<I>(buffers: I) -> Result<VertexSourceBindCommand, VertexBufferBindError>
        where I: Iterator<Item = (Arc<Buffer>, usize)>
    {
        let mut keep_alive = SmallVec::new();
        let mut raw_buffers = SmallVec::new();
        let mut offsets = SmallVec::new();
        let mut device = None;
        let mut max_vb_input_limit = 8;

        for (buffer, offset) in buffers {
            raw_buffers.push(buffer.inner_buffer().internal_object());
            offsets.push(offset as vk::DeviceSize);

            device = match device {
                Some(d) => {
                    assert_eq!(d, buffer.inner_buffer().device().internal_object());
                    Some(d)
                },
                None => {
                    let device = buffer.inner_buffer().device();
                    max_vb_input_limit = device.physical_device().limits()
                                               .max_vertex_input_bindings();
                    Some(device.internal_object())
                },
            };

            if !buffer.inner_buffer().usage_vertex_buffer() {
                return Err(VertexBufferBindError::WrongUsageFlag);
            }

            keep_alive.push(unsafe { mem::transmute(buffer) });      // FIXME: meh
        }

        if raw_buffers.len() > max_vb_input_limit as usize {
            return Err(VertexBufferBindError::MaxVertexInputBindingsExceeded);
        }

        Ok(VertexSourceBindCommand {
            keep_alive: keep_alive,
            device: device.unwrap_or(0),
            raw_buffers: raw_buffers,
            offsets: offsets,
        })
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the buffers were not allocated with the same device as the command buffer.
    /// - Panicks if the queue doesn't not support graphics operations.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            let _pool_lock = cb.pool().lock();

            assert!(cb.pool().queue_family().supports_graphics());

            // Nothing to bind.
            if self.raw_buffers.is_empty() {
                return cb;
            }

            // Various checks.
            // Note that surprisingly the specs allow this function to be called from outside a
            // render pass.
            assert_eq!(self.device, cb.device().internal_object());
            debug_assert_eq!(self.offsets.len(), self.raw_buffers.len());

            for ka in self.keep_alive.into_iter() {
                cb.keep_alive.push(ka);
            }

            // Determine which bindings have to be refreshed.
            let mut first_binding = 0;
            for ((&curr_vb, &raw), &off) in cb.current_vertex_buffers.iter()
                                                                     .zip(self.raw_buffers.iter())
                                                                     .zip(self.offsets.iter())
            {
                if curr_vb.0 == raw && curr_vb.1 == off {
                    first_binding += 1;
                } else {
                    break;
                }
            }

            // Ignore binding if not necessary.
            if first_binding >= self.raw_buffers.len() {
                return cb;
            }

            for ((curr_vb, &raw), &off) in cb.current_vertex_buffers.iter_mut()
                                                                    .zip(self.raw_buffers.iter())
                                                                    .zip(self.offsets.iter())
                                                                    .skip(first_binding)
            {
                *curr_vb = (raw, off);
            }

            // Now binding.
            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();

                let binding_count = (self.raw_buffers.len() - first_binding) as u32;
                debug_assert!(binding_count >= 1);

                vk.CmdBindVertexBuffers(cmd, first_binding as u32, binding_count,
                                        self.raw_buffers.as_ptr().offset(first_binding as isize),
                                        self.offsets.as_ptr().offset(first_binding as isize));
            }

            cb
        }
    }
}

error_ty!{VertexBufferBindError => "Error that can happen when binding vertex buffers.",
    MaxVertexInputBindingsExceeded => "too many vertex buffers",
    WrongUsageFlag => "one of the buffers was not created with the correct usage flags",
}


// TODO: add test for binding 0 buffer
