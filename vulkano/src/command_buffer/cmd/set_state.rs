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

use command_buffer::RawCommandBufferPrototype;
use command_buffer::DynamicState;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use device::Device;
use VulkanObject;
use VulkanPointers;

/// Wraps around a commands list and adds to the end of it a command that sets the state of the
/// pipeline to the given one.
///
/// Only the values that are `Some` are touched. Parameters that are `None` are left untouched.
/// A state is not modified if the same state is already current.
pub struct CmdSetState<L>
    where L: CommandsList
{
    // Parent commands list.
    previous: L,
    // The device.
    device: Arc<Device>,
    // The state to set.
    dynamic_state: DynamicState,
}

impl<L> CmdSetState<L>
    where L: CommandsList
{
    /// Builds a command.
    ///
    /// Since this command checks whether the dynamic state is supported by the device, you have
    /// to pass the device as well when building the command.
    // TODO: should check the limits and features of the device
    pub fn new(previous: L, device: Arc<Device>, state: DynamicState) -> CmdSetState<L> {
        CmdSetState {
            previous: previous,
            device: device,
            dynamic_state: DynamicState {
                // This constructor is explicitely layed out so that we don't forget to
                // modify this code if we add a new member to `DynamicState`.
                line_width: state.line_width,
                viewports: state.viewports,
                scissors: state.scissors,
            },
        }
    }
}

unsafe impl<L> CommandsList for CmdSetState<L>
    where L: CommandsList
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        assert_eq!(self.device.internal_object(),
                   builder.device().internal_object());

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();

                if let Some(line_width) = self.dynamic_state.line_width {
                    if raw.current_state.line_width != Some(line_width) {
                        vk.CmdSetLineWidth(cmd, line_width);
                        raw.current_state.line_width = Some(line_width);
                    }
                }

                if let Some(ref viewports) = self.dynamic_state.viewports {
                    // TODO: cache state
                    let viewports = viewports.iter().map(|v| v.clone().into()).collect::<SmallVec<[_; 16]>>();
                    vk.CmdSetViewport(cmd, 0, viewports.len() as u32, viewports.as_ptr());
                }

                if let Some(ref scissors) = self.dynamic_state.scissors {
                    // TODO: cache state
                    let scissors = scissors.iter().map(|v| v.clone().into()).collect::<SmallVec<[_; 16]>>();
                    vk.CmdSetScissor(cmd, 0, scissors.len() as u32, scissors.as_ptr());
                }
            }
        }));
    }
}
