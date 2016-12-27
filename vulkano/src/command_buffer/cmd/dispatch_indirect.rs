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
use std::mem;
use std::sync::Arc;

use buffer::Buffer;
use buffer::TypedBuffer;
use command_buffer::cmd::CmdBindDescriptorSets;
use command_buffer::cmd::CmdBindDescriptorSetsError;
use command_buffer::cmd::CmdBindPipeline;
use command_buffer::cmd::CmdPushConstants;
use command_buffer::cmd::CmdPushConstantsError;
use command_buffer::DispatchIndirectCommand;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use descriptor::PipelineLayoutRef;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use pipeline::ComputePipeline;
use sync::AccessFlagBits;
use sync::PipelineStages;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds an indirect dispatch command at the end of it.
pub struct CmdDispatchIndirect<L, B, Pl, S, Pc>
    where L: CommandsList,
          Pl: PipelineLayoutRef,
          S: TrackedDescriptorSetsCollection
{
    // Parent commands list.
    previous: CmdPushConstants<CmdBindDescriptorSets<CmdBindPipeline<L, Arc<ComputePipeline<Pl>>>, S, Arc<ComputePipeline<Pl>>>, Pc, Arc<ComputePipeline<Pl>>>,

    raw_buffer: vk::Buffer,
    raw_offset: vk::DeviceSize,

    // The buffer.
    buffer: B,
}

impl<L, B, Pl, S, Pc> CmdDispatchIndirect<L, B, Pl, S, Pc>
    where L: CommandsList,
          Pl: PipelineLayoutRef,
          S: TrackedDescriptorSetsCollection
{
    /// This function is unsafe because the values in the buffer must be less or equal than
    /// `VkPhysicalDeviceLimits::maxComputeWorkGroupCount`.
    pub unsafe fn new(previous: L, pipeline: Arc<ComputePipeline<Pl>>, sets: S, push_constants: Pc, buffer: B) -> Result<CmdDispatchIndirect<L, B, Pl, S, Pc>, CmdDispatchIndirectError>
        where B: TypedBuffer<Content = DispatchIndirectCommand>
    {
        let previous = CmdBindPipeline::bind_compute_pipeline(previous, pipeline.clone());
        let device = previous.device().clone();
        let previous = CmdBindDescriptorSets::new(previous, false, pipeline.clone(), sets)?;
        let previous = CmdPushConstants::new(previous, pipeline.clone(), push_constants)?;

        let (raw_buffer, raw_offset) = {
            let inner = buffer.inner();

            if !inner.buffer.usage_indirect_buffer() {
                return Err(CmdDispatchIndirectError::MissingBufferUsage);
            }

            if inner.offset % 4 != 0 {
                return Err(CmdDispatchIndirectError::WrongAlignment);
            }

            (inner.buffer.internal_object(), inner.offset as vk::DeviceSize)
        };

        Ok(CmdDispatchIndirect {
            previous: previous,
            raw_buffer: raw_buffer,
            raw_offset: raw_offset,
            buffer: buffer,
        })
    }
}

unsafe impl<L, B, Pl, S, Pc> CommandsList for CmdDispatchIndirect<L, B, Pl, S, Pc>
    where L: CommandsList,
          B: Buffer,
          Pl: PipelineLayoutRef,
          S: TrackedDescriptorSetsCollection
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        {
            let stages = PipelineStages { compute_shader: true, ..PipelineStages::none() };
            let access = AccessFlagBits { indirect_command_read: true, ..AccessFlagBits::none() };
            builder.add_buffer_transition(&self.buffer,
                                          0,
                                          mem::size_of::<DispatchIndirectCommand>(),
                                          false,
                                          stages,
                                          access);
        }

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();
                vk.CmdDispatchIndirect(cmd, self.raw_buffer, self.raw_offset);
            }
        }));
    }
}

/// Error that can happen when creating a `CmdDispatch`.
#[derive(Debug, Copy, Clone)]
pub enum CmdDispatchIndirectError {
    /// The buffer must have the "indirect" usage.
    MissingBufferUsage,
    /// The buffer must be 4-bytes-aligned.
    WrongAlignment,
    /// Error while binding descriptor sets.
    BindDescriptorSetsError(CmdBindDescriptorSetsError),
    /// Error while setting push constants.
    PushConstantsError(CmdPushConstantsError),
}

impl From<CmdBindDescriptorSetsError> for CmdDispatchIndirectError {
    #[inline]
    fn from(err: CmdBindDescriptorSetsError) -> CmdDispatchIndirectError {
        CmdDispatchIndirectError::BindDescriptorSetsError(err)
    }
}

impl From<CmdPushConstantsError> for CmdDispatchIndirectError {
    #[inline]
    fn from(err: CmdPushConstantsError) -> CmdDispatchIndirectError {
        CmdDispatchIndirectError::PushConstantsError(err)
    }
}

impl error::Error for CmdDispatchIndirectError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdDispatchIndirectError::MissingBufferUsage => "the buffer must have the indirect usage.",
            CmdDispatchIndirectError::WrongAlignment => "the buffer must be 4-bytes-aligned",
            CmdDispatchIndirectError::BindDescriptorSetsError(_) => "error while binding descriptor sets",
            CmdDispatchIndirectError::PushConstantsError(_) => "error while setting push constants",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            CmdDispatchIndirectError::MissingBufferUsage => None,
            CmdDispatchIndirectError::WrongAlignment => None,
            CmdDispatchIndirectError::BindDescriptorSetsError(ref err) => Some(err),
            CmdDispatchIndirectError::PushConstantsError(ref err) => Some(err),
        }
    }
}

impl fmt::Display for CmdDispatchIndirectError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
