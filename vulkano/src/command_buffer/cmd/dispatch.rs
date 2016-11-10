// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;
use std::sync::Arc;
use smallvec::SmallVec;

use command_buffer::DynamicState;
use command_buffer::StatesManager;
use command_buffer::SubmitInfo;
use command_buffer::cmd::CmdBindDescriptorSets;
use command_buffer::cmd::CmdBindPipeline;
use command_buffer::cmd::CmdBindVertexBuffers;
use command_buffer::cmd::CmdPushConstants;
use command_buffer::cmd::CmdSetState;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use descriptor::PipelineLayoutRef;
use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use device::Device;
use device::Queue;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::vertex::Source;
use sync::Fence;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds a dispatch command at the end of it.
pub struct CmdDispatch<L, Pl, S, Pc>
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

    // Dispatch dimensions.
    dimensions: [u32; 3],
}

impl<L, Pl, S, Pc> CmdDispatch<L, Pl, S, Pc>
    where L: CommandsList, Pl: PipelineLayoutRef, S: TrackedDescriptorSetsCollection
{
    /// See the documentation of the `dispatch` method.
    pub fn new(previous: L, pipeline: Arc<ComputePipeline<Pl>>, sets: S, dimensions: [u32; 3],
               push_constants: Pc) -> CmdDispatch<L, Pl, S, Pc>
    {
        let previous = CmdBindPipeline::bind_compute_pipeline(previous, pipeline.clone());
        let device = previous.device().clone();
        let previous = CmdBindDescriptorSets::new(previous, false, pipeline.clone(), sets).unwrap() /* TODO: error */;
        let previous = CmdPushConstants::new(previous, pipeline.clone(), push_constants).unwrap() /* TODO: error */;

        // TODO: check dimensions limits

        CmdDispatch {
            previous: previous,
            dimensions: dimensions,
        }
    }
}

unsafe impl<L, Pl, S, Pc> CommandsList for CmdDispatch<L, Pl, S, Pc>
    where L: CommandsList, Pl: PipelineLayoutRef, S: TrackedDescriptorSetsCollection
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
