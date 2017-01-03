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
use std::sync::Arc;

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use descriptor::pipeline_layout::PipelineLayoutRef;
use descriptor::pipeline_layout::PipelineLayoutPushConstantsCompatible;
use device::Device;
use VulkanObject;
use VulkanPointers;

/// Wraps around a commands list and adds at the end of it a command that updates push constants.
pub struct CmdPushConstants<Pc, Pl> {
    // The device of the pipeline object, so that we can compare it with the command buffer's
    // device.
    device: Arc<Device>,
    // The push constant data.
    push_constants: Pc,
    // The pipeline layout.
    pipeline_layout: Pl,
}

impl<Pc, Pl> CmdPushConstants<Pc, Pl>
    where Pl: PipelineLayoutRef
{
    /// Builds the command.
    ///
    /// Returns an error if the push constants are not compatible with the pipeline layout.
    #[inline]
    pub fn new(pipeline_layout: Pl, push_constants: Pc)
               -> Result<CmdPushConstants<Pc, Pl>, CmdPushConstantsError> 
    {
        if !PipelineLayoutPushConstantsCompatible::is_compatible(pipeline_layout.desc(), &push_constants) {
            return Err(CmdPushConstantsError::IncompatibleData);
        }

        let device = pipeline_layout.device().clone();

        Ok(CmdPushConstants {
            device: device,
            push_constants: push_constants,
            pipeline_layout: pipeline_layout,
        })
    }
}

unsafe impl<'a, P, Pc, Pl> AddCommand<&'a CmdPushConstants<Pc, Pl>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool,
          Pl: PipelineLayoutRef
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdPushConstants<Pc, Pl>) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();

            let data_raw = &command.push_constants as *const Pc as *const u8;
            
            for num_range in 0 .. command.pipeline_layout.desc().num_push_constants_ranges() {
                let range = match command.pipeline_layout.desc().push_constants_range(num_range) {
                    Some(r) => r,
                    None => continue
                };

                debug_assert_eq!(range.offset % 4, 0);
                debug_assert_eq!(range.size % 4, 0);

                vk.CmdPushConstants(cmd, command.pipeline_layout.sys().internal_object(),
                                    range.stages.into(), range.offset as u32, range.size as u32,
                                    data_raw.offset(range.offset as isize) as *const _);
            }
        }

        self
    }
}

/// Error that can happen when creating a `CmdPushConstants`.
#[derive(Debug, Copy, Clone)]
pub enum CmdPushConstantsError {
    /// The push constants are not compatible with the pipeline layout.
    // TODO: inner error
    IncompatibleData,
}

impl error::Error for CmdPushConstantsError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdPushConstantsError::IncompatibleData => {
                "the push constants are not compatible with the pipeline layout"
            },
        }
    }
}

impl fmt::Display for CmdPushConstantsError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
