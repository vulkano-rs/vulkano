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

/// Command that draws non-indexed vertices.
///
/// > **Note**: Unless you are writing a custom implementation of a command buffer, you are
/// > encouraged to ignore this struct and use a `CmdDraw` instead.
pub struct CmdDrawRaw {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

impl CmdDrawRaw {
    /// Builds a new command that executes a non-indexed draw command.
    ///
    /// The command will use the vertex buffers, dynamic states, descriptor sets, push constants,
    /// and graphics pipeline currently bound.
    ///
    /// This command corresponds to the `vkCmdDraw` function in Vulkan.
    ///
    /// # Safety
    ///
    /// While building the command is always safe, care must be taken when it is added to a command
    /// buffer. A correct combination of graphics pipeline, descriptor set, push constants, vertex
    /// buffers, and dynamic state must have been bound beforehand.
    ///
    /// There is no limit to the values of the parameters, but they must be in range of the vertex
    /// buffer.
    ///
    #[inline]
    pub unsafe fn new(vertex_count: u32, instance_count: u32, first_vertex: u32,
                      first_instance: u32) -> CmdDrawRaw
    {
        CmdDrawRaw {
            vertex_count: vertex_count,
            instance_count: instance_count,
            first_vertex: first_vertex,
            first_instance: first_instance,
        }
    }
}

unsafe impl<'a, P> AddCommand<&'a CmdDrawRaw> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdDrawRaw) -> Result<Self::Out, CommandAddError> {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdDraw(cmd, command.vertex_count, command.instance_count, command.first_vertex,
                       command.first_instance);
        }

        Ok(self)
    }
}
