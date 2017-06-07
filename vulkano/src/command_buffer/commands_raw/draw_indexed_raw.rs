// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use command_buffer::CommandAddError;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;

/// Command that draws indexed vertices.
///
/// > **Note**: Unless you are writing a custom implementation of a command buffer, you are
/// > encouraged to ignore this struct and use a `CmdDrawIndexed` instead.
pub struct CmdDrawIndexedRaw {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
}

impl CmdDrawIndexedRaw {
    /// Builds a new command that executes an indexed draw command.
    ///
    /// The command will use the vertex buffers, index buffer, dynamic states, descriptor sets,
    /// push constants, and graphics pipeline currently bound.
    ///
    /// This command corresponds to the `vkCmdDrawIndexed` function in Vulkan. It takes the first
    /// `index_count` indices in the index buffer starting at `first_index`, and adds the value of
    /// `vertex_offset` to each index. `instance_count` and `first_instance` are related to
    /// instancing and serve the same purpose as in other drawing commands.
    ///
    /// # Safety
    ///
    /// While building the command is always safe, care must be taken when it is added to a command
    /// buffer. A correct combination of graphics pipeline, descriptor set, push constants, vertex
    /// buffers, index buffer, and dynamic state must have been bound beforehand.
    ///
    /// There is no limit to the values of the parameters, but they must be in range of the index
    /// buffer and vertex buffer.
    ///
    #[inline]
    pub unsafe fn new(index_count: u32, instance_count: u32, first_index: u32,
                      vertex_offset: i32, first_instance: u32) -> CmdDrawIndexedRaw
    {
        CmdDrawIndexedRaw {
            index_count: index_count,
            instance_count: instance_count,
            first_index: first_index,
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
    fn add(self, command: &'a CmdDrawIndexedRaw) -> Result<Self::Out, CommandAddError> {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdDrawIndexed(cmd, command.index_count, command.instance_count,
                              command.first_index, command.vertex_offset, command.first_instance);
        }

        Ok(self)
    }
}
