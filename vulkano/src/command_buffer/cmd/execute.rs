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

use buffer::Buffer;
use command_buffer::DynamicState;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::SecondaryCommandBuffer;
use command_buffer::cmd::CommandsList;
use command_buffer::cmd::CommandsListSink;
use command_buffer::cmd::CommandsListSinkCaller;
use device::Device;
use image::Layout;
use image::Image;
use sync::AccessFlagBits;
use sync::PipelineStages;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds a command at the end of it that executes a secondary
/// command buffer.
pub struct CmdExecuteCommands<Cb, L>
    where Cb: SecondaryCommandBuffer,
          L: CommandsList
{
    // Parent commands list.
    previous: L,
    // Raw list of command buffers to execute.
    raw_list: SmallVec<[vk::CommandBuffer; 4]>,
    // Command buffer to execute.
    command_buffer: Cb,
}

impl<Cb, L> CmdExecuteCommands<Cb, L>
    where Cb: SecondaryCommandBuffer,
          L: CommandsList
{
    /// See the documentation of the `execute_commands` method.
    #[inline]
    pub fn new(previous: L, command_buffer: Cb) -> CmdExecuteCommands<Cb, L> {
        // FIXME: most checks are missing

        let raw_list = {
            let mut l = SmallVec::new();
            l.push(command_buffer.inner());
            l
        };

        CmdExecuteCommands {
            previous: previous,
            raw_list: raw_list,
            command_buffer: command_buffer,
        }
    }
}

// TODO: specialize the trait so that multiple calls to `execute` are grouped together?
unsafe impl<Cb, L> CommandsList for CmdExecuteCommands<Cb, L>
    where Cb: SecondaryCommandBuffer,
          L: CommandsList
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        assert_eq!(self.command_buffer.device().internal_object(),
                   builder.device().internal_object());

        self.command_buffer.append(&mut FilterOutCommands(builder, self.command_buffer.device()));

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();

                vk.CmdExecuteCommands(cmd, self.raw_list.len() as u32, self.raw_list.as_ptr());

                // vkCmdExecuteCommands resets the state of the command buffer.
                raw.current_state = DynamicState::none();
                raw.bound_graphics_pipeline = 0;
                raw.bound_compute_pipeline = 0;
                raw.bound_index_buffer = (0, 0, 0);
            }
        }));
    }
}

struct FilterOutCommands<'a, 'c: 'a>(&'a mut CommandsListSink<'c>, &'a Arc<Device>);

impl<'a, 'c: 'a> CommandsListSink<'c> for FilterOutCommands<'a, 'c> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.1
    }

    #[inline]
    fn add_command(&mut self, _: Box<CommandsListSinkCaller<'c> + 'c>) {}

    // FIXME: this is wrong since the underlying impl will try to perform transitions that are
    //        performed by the secondary command buffer
    #[inline]
    fn add_buffer_transition(&mut self, buffer: &'c Buffer, offset: usize, size: usize, write: bool, stages: PipelineStages, access: AccessFlagBits) {
        self.0.add_buffer_transition(buffer, offset, size, write, stages, access)
    }

    // FIXME: this is wrong since the underlying impl will try to perform transitions that are
    //        performed by the secondary command buffer
    #[inline]
    fn add_image_transition(&mut self, image: &'c Image, first_layer: u32, num_layers: u32, first_mipmap: u32, num_mipmaps: u32, write: bool, layout: Layout, stages: PipelineStages, access: AccessFlagBits) {
        self.0.add_image_transition(image,
                                    first_layer,
                                    num_layers,
                                    first_mipmap,
                                    num_mipmaps,
                                    write,
                                    layout,
                                    stages,
                                    access)
    }

    #[inline]
    fn add_image_transition_notification(&mut self, image: &'c Image, first_layer: u32, num_layers: u32, first_mipmap: u32, num_mipmaps: u32, layout: Layout, stages: PipelineStages, access: AccessFlagBits) {
        self.0.add_image_transition_notification(image,
                                                 first_layer,
                                                 num_layers,
                                                 first_mipmap,
                                                 num_mipmaps,
                                                 layout,
                                                 stages,
                                                 access)
    }
}
