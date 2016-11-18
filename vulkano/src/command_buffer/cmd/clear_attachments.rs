// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;

use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds at the end of it a command that clears framebuffer
/// attachments.
pub struct CmdClearAttachments<L> {
    // Parent commands list.
    previous: L,
    // The attachments to clear.
    attachments: SmallVec<[vk::ClearAttachment; 8]>,
    // The rectangles to clear.
    rects: SmallVec<[vk::ClearRect; 4]>,
}

// TODO: add constructor

unsafe impl<L> CommandsList for CmdClearAttachments<L>
    where L: CommandsList
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        // According to the Vulkan specifications, the `vkCmdClearAttachments` command doesn't
        // need any pipeline barrier.
        // Since the thing that is cleared is an attachment of the framebuffer, there's no need to
        // provide any additional form of synchronization.

        if self.attachments.is_empty() || self.rects.is_empty() {
            return;
        }

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();

                vk.CmdClearAttachments(cmd, self.attachments.len() as u32,
                                       self.attachments.as_ptr(), self.rects.len() as u32,
                                       self.rects.as_ptr());
            }
        }));
    }
}
