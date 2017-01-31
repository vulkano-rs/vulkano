// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use VulkanObject;
use VulkanPointers;

/// Command that draws indexed vertices.
pub struct CmdDrawIndexedRaw {
    index_count: u32,
    instance_count: u32,
    first_vertex: u32,
    vertex_offset: i32,
    first_instance: u32,
}

impl CmdDrawIndexedRaw {
    #[inline]
    pub unsafe fn new(index_count: u32, instance_count: u32, first_vertex: u32,
                      vertex_offset: i32, first_instance: u32) -> CmdDrawIndexedRaw
    {
        CmdDrawIndexedRaw {
            index_count: index_count,
            instance_count: instance_count,
            first_vertex: first_vertex,
            vertex_offset: vertex_offset,
            first_instance: first_instance,
        }
    }
}

unsafe impl<'a, P> AddCommand<&'a CmdDrawIndexedRaw> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdDrawIndexedRaw) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdDrawIndexed(cmd, command.index_count, command.instance_count,
                              command.first_vertex, command.vertex_offset, command.first_instance);
        }

        self
    }
}
