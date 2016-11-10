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
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::pipeline_layout::PipelineLayoutRef;
use descriptor::pipeline_layout::PipelineLayoutPushConstantsCompatible;
use device::Device;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds at the end of it a command that updates push constants.
pub struct CmdPushConstants<L, Pc, Pl> where L: CommandsList {
    // Parent commands list.
    previous: L,
    // The device of the pipeline object, so that we can compare it with the command buffer's
    // device.
    device: Arc<Device>,
    // The push constant data.
    push_constants: Pc,
    // The pipeline layout.
    pipeline_layout: Pl,
}

impl<L, Pc, Pl> CmdPushConstants<L, Pc, Pl>
    where L: CommandsList, Pl: PipelineLayoutRef
{
    /// Builds the command.
    ///
    /// Returns an error if the push constants are not compatible with the pipeline layout.
    #[inline]
    pub fn new(previous: L, pipeline_layout: Pl, push_constants: Pc)
               -> Result<CmdPushConstants<L, Pc, Pl>, CmdPushConstantsError> 
    {
        if !PipelineLayoutPushConstantsCompatible::is_compatible(pipeline_layout.desc(), &push_constants) {
            return Err(CmdPushConstantsError::IncompatibleData);
        }

        let device = pipeline_layout.device().clone();

        Ok(CmdPushConstants {
            previous: previous,
            device: device,
            push_constants: push_constants,
            pipeline_layout: pipeline_layout,
        })
    }
}

unsafe impl<L, Pc, Pl> CommandsList for CmdPushConstants<L, Pc, Pl>
    where L: CommandsList, Pl: PipelineLayoutRef
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        assert_eq!(self.device.internal_object(), builder.device().internal_object());

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();

                let data_raw = &self.push_constants as *const Pc as *const u8;
                
                for num_range in 0 .. self.pipeline_layout.desc().num_push_constants_ranges() {
                    let range = match self.pipeline_layout.desc().push_constants_range(num_range) {
                        Some(r) => r,
                        None => continue
                    };

                    debug_assert_eq!(range.offset % 4, 0);
                    debug_assert_eq!(range.size % 4, 0);

                    vk.CmdPushConstants(cmd, self.pipeline_layout.sys().internal_object(),
                                        range.stages.into(), range.offset as u32, range.size as u32,
                                        data_raw.offset(range.offset as isize) as *const _);
                }
            }
        }));
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
