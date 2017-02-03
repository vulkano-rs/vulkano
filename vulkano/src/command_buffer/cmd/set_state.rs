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

use command_buffer::DynamicState;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;

/// Command that sets the state of the pipeline to the given one.
///
/// Only the values that are `Some` are modified. Parameters that are `None` are left untouched.
pub struct CmdSetState {
    // The device.
    device: Arc<Device>,
    // The state to set.
    dynamic_state: DynamicState,
}

impl CmdSetState {
    /// Builds a command.
    ///
    /// Since this command checks whether the dynamic state is supported by the device, you have
    /// to pass the device as well when building the command.
    // TODO: should check the limits and features of the device
    pub fn new(device: Arc<Device>, state: DynamicState) -> CmdSetState {
        CmdSetState {
            device: device,
            dynamic_state: DynamicState {
                // This constructor is explicitely layed out so that we don't forget to
                // modify the code of this module if we add a new member to `DynamicState`.
                line_width: state.line_width,
                viewports: state.viewports,
                scissors: state.scissors,
            },
        }
    }

    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the state that is going to be set.
    #[inline]
    pub fn state(&self) -> &DynamicState {
        &self.dynamic_state
    }
}

unsafe impl<'a, P> AddCommand<&'a CmdSetState> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdSetState) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();

            if let Some(line_width) = command.dynamic_state.line_width {
                vk.CmdSetLineWidth(cmd, line_width);
            }

            if let Some(ref viewports) = command.dynamic_state.viewports {
                let viewports = viewports.iter().map(|v| v.clone().into()).collect::<SmallVec<[_; 16]>>();
                vk.CmdSetViewport(cmd, 0, viewports.len() as u32, viewports.as_ptr());
            }

            if let Some(ref scissors) = command.dynamic_state.scissors {
                let scissors = scissors.iter().map(|v| v.clone().into()).collect::<SmallVec<[_; 16]>>();
                vk.CmdSetScissor(cmd, 0, scissors.len() as u32, scissors.as_ptr());
            }
        }

        self
    }
}
