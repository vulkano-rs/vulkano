// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::sync::Arc;

use buffer::TrackedBuffer;
use buffer::TypedBuffer;
use command_buffer::cmd::CmdBindDescriptorSets;
use command_buffer::cmd::CmdBindPipeline;
use command_buffer::cmd::CmdPushConstants;
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
    where L: CommandsList, Pl: PipelineLayoutRef, S: TrackedDescriptorSetsCollection
{
    // Parent commands list.
    previous: CmdPushConstants<
                CmdBindDescriptorSets<
                    CmdBindPipeline<L, Arc<ComputePipeline<Pl>>>,
                    S, Arc<ComputePipeline<Pl>>
                >,
                Pc, Arc<ComputePipeline<Pl>>
              >,

    raw_buffer: vk::Buffer,
    raw_offset: vk::DeviceSize,

    // The buffer.
    buffer: B,
}

impl<L, B, Pl, S, Pc> CmdDispatchIndirect<L, B, Pl, S, Pc>
    where L: CommandsList, Pl: PipelineLayoutRef, S: TrackedDescriptorSetsCollection
{
    /// This function is unsafe because the values in the buffer must be less or equal than
    /// `VkPhysicalDeviceLimits::maxComputeWorkGroupCount`.
    pub unsafe fn new(previous: L, pipeline: Arc<ComputePipeline<Pl>>, sets: S, push_constants: Pc,
                      buffer: B) -> CmdDispatchIndirect<L, B, Pl, S, Pc>
        where B: TypedBuffer<Content = DispatchIndirectCommand>
    {
        let previous = CmdBindPipeline::bind_compute_pipeline(previous, pipeline.clone());
        let device = previous.device().clone();
        let previous = CmdBindDescriptorSets::new(previous, false, pipeline.clone(), sets).unwrap() /* TODO: error */;
        let previous = CmdPushConstants::new(previous, pipeline.clone(), push_constants).unwrap() /* TODO: error */;

        let (raw_buffer, raw_offset) = {
            let inner = buffer.inner();
            if !inner.buffer.usage_indirect_buffer() {
                panic!()        // TODO: error
            }
            assert_eq!(inner.offset % 4, 0);
            (inner.buffer.internal_object(), inner.offset as vk::DeviceSize)
        };

        CmdDispatchIndirect {
            previous: previous,
            raw_buffer: raw_buffer,
            raw_offset: raw_offset,
            buffer: buffer,
        }
    }
}

unsafe impl<L, B, Pl, S, Pc> CommandsList for CmdDispatchIndirect<L, B, Pl, S, Pc>
    where L: CommandsList, B: TrackedBuffer,
          Pl: PipelineLayoutRef, S: TrackedDescriptorSetsCollection
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        {
            let stages = PipelineStages { compute_shader: true, .. PipelineStages::none() };
            let access = AccessFlagBits { indirect_command_read: true, .. AccessFlagBits::none() };
            builder.add_buffer_transition(&self.buffer, 0,
                                          mem::size_of::<DispatchIndirectCommand>(), false,
                                          stages, access);
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
