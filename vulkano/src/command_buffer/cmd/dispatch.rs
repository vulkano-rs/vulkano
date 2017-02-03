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

use command_buffer::cmd::CmdBindDescriptorSets;
use command_buffer::cmd::CmdBindDescriptorSetsError;
use command_buffer::cmd::CmdBindPipeline;
use command_buffer::cmd::CmdPushConstants;
use command_buffer::cmd::CmdPushConstantsError;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use descriptor::PipelineLayoutAbstract;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use device::DeviceOwned;
use pipeline::ComputePipeline;
use VulkanPointers;

/// Command that executes a compute shader.
pub struct CmdDispatch<L, Pl, S, Pc>
    where L: CommandsList, Pl: PipelineLayoutAbstract, S: TrackedDescriptorSetsCollection
{
    // Parent commands list.
    previous: CmdPushConstants<
                CmdBindDescriptorSets<
                    CmdBindPipeline<L, Arc<ComputePipeline<Pl>>>,
                    S, Arc<ComputePipeline<Pl>>
                >,
                Pc, Arc<ComputePipeline<Pl>>
              >,

    // Dispatch dimensions.
    dimensions: [u32; 3],
}

impl<L, Pl, S, Pc> CmdDispatch<L, Pl, S, Pc>
    where L: CommandsList, Pl: PipelineLayoutAbstract, S: TrackedDescriptorSetsCollection
{
    /// See the documentation of the `dispatch` method.
    pub fn new(previous: L, pipeline: Arc<ComputePipeline<Pl>>, sets: S, dimensions: [u32; 3],
               push_constants: Pc) -> Result<CmdDispatch<L, Pl, S, Pc>, CmdDispatchError>
    {
        let previous = CmdBindPipeline::bind_compute_pipeline(previous, pipeline.clone());
        let device = previous.device().clone();
        let previous = CmdBindDescriptorSets::new(previous, false, pipeline.clone(), sets)?;
        let previous = CmdPushConstants::new(previous, pipeline.clone(), push_constants)?;

        // FIXME: check dimensions limits

        Ok(CmdDispatch {
            previous: previous,
            dimensions: dimensions,
        })
    }
}

unsafe impl<L, Pl, S, Pc> CommandsList for CmdDispatch<L, Pl, S, Pc>
    where L: CommandsList, Pl: PipelineLayoutAbstract, S: TrackedDescriptorSetsCollection
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();
                vk.CmdDispatch(cmd, self.dimensions[0], self.dimensions[1], self.dimensions[2]);
            }
        }));
    }
}

/// Error that can happen when creating a `CmdDispatch`.
#[derive(Debug, Copy, Clone)]
pub enum CmdDispatchError {
    /// The dispatch dimensions are larger than the hardware limits.
    DimensionsTooLarge,
    /// Error while binding descriptor sets.
    BindDescriptorSetsError(CmdBindDescriptorSetsError),
    /// Error while setting push constants.
    PushConstantsError(CmdPushConstantsError),
}

impl From<CmdBindDescriptorSetsError> for CmdDispatchError {
    #[inline]
    fn from(err: CmdBindDescriptorSetsError) -> CmdDispatchError {
        CmdDispatchError::BindDescriptorSetsError(err)
    }
}

impl From<CmdPushConstantsError> for CmdDispatchError {
    #[inline]
    fn from(err: CmdPushConstantsError) -> CmdDispatchError {
        CmdDispatchError::PushConstantsError(err)
    }
}

impl error::Error for CmdDispatchError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdDispatchError::DimensionsTooLarge => {
                "the dispatch dimensions are larger than the hardware limits"
            },
            CmdDispatchError::BindDescriptorSetsError(_) => {
                "error while binding descriptor sets"
            },
            CmdDispatchError::PushConstantsError(_) => {
                "error while setting push constants"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            CmdDispatchError::DimensionsTooLarge => None,
            CmdDispatchError::BindDescriptorSetsError(ref err) => Some(err),
            CmdDispatchError::PushConstantsError(ref err) => Some(err),
        }
    }
}

impl fmt::Display for CmdDispatchError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
