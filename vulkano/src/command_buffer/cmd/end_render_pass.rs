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

use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use VulkanPointers;

/// Wraps around a commands list and adds to the end of it a command that ends the current render
/// pass.
#[derive(Debug, Copy, Clone)]
pub struct CmdEndRenderPass<L> where L: CommandsList {
    // Parent commands list.
    previous: L,
}

impl<L> CmdEndRenderPass<L> where L: CommandsList {
    /// See the documentation of the `end_render_pass` method.
    #[inline]
    pub fn new(previous: L) -> Result<CmdEndRenderPass<L>, CmdEndRenderPassError> {
        // TODO: check that we're in a render pass and that the next subpass is correct

        Ok(CmdEndRenderPass {
            previous: previous,
        })
    }
}

unsafe impl<L> CommandsList for CmdEndRenderPass<L> where L: CommandsList {
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();
                vk.CmdEndRenderPass(cmd);
            }
        }));
    }
}

/// Error that can happen when creating a `CmdEndRenderPass`.
#[derive(Debug, Copy, Clone)]
pub enum CmdEndRenderPassError {
    /// It's not possible to end the render pass before you went over all the subpasses.
    SubpassesRemaining,
}

impl error::Error for CmdEndRenderPassError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdEndRenderPassError::SubpassesRemaining => {
                "it's not possible to end the render pass before you went over all the subpasses"
            },
        }
    }
}

impl fmt::Display for CmdEndRenderPassError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
