// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::TrackedBuffer;
use buffer::TypedBuffer;
use command_buffer::DynamicState;
use command_buffer::cmd::CmdBindDescriptorSets;
use command_buffer::cmd::CmdBindIndexBuffer;
use command_buffer::cmd::CmdBindPipeline;
use command_buffer::cmd::CmdBindVertexBuffers;
use command_buffer::cmd::CmdPushConstants;
use command_buffer::cmd::CmdSetState;
use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use descriptor::PipelineLayoutRef;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::Source;
use VulkanPointers;

/// Wraps around a commands list and adds a draw command at the end of it.
pub struct CmdDrawIndexed<L, V, Ib, Pv, Pl, Prp, S, Pc>
    where L: CommandsList, Pl: PipelineLayoutRef, S: TrackedDescriptorSetsCollection
{
    // Parent commands list.
    previous: CmdBindIndexBuffer<
                CmdBindVertexBuffers<
                    CmdPushConstants<
                        CmdBindDescriptorSets<
                            CmdSetState<
                                CmdBindPipeline<L, Arc<GraphicsPipeline<Pv, Pl, Prp>>>
                            >,
                            S, Arc<GraphicsPipeline<Pv, Pl, Prp>>
                        >,
                        Pc, Arc<GraphicsPipeline<Pv, Pl, Prp>>
                    >,
                    V
                >,
                Ib
              >,

    // Parameters for vkCmdDrawIndexedIndexed.
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
}

impl<L, V, I, Ib, Pv, Pl, Prp, S, Pc> CmdDrawIndexed<L, V, Ib, Pv, Pl, Prp, S, Pc>
    where L: CommandsList,
          Pl: PipelineLayoutRef,
          S: TrackedDescriptorSetsCollection,
          Ib: TrackedBuffer + TypedBuffer<Content = [I]>,
          I: Index + 'static
{
    /// See the documentation of the `draw` method.
    pub fn new(previous: L, pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
               dynamic: DynamicState, vertices: V, index_buffer: Ib, sets: S, push_constants: Pc)
               -> CmdDrawIndexed<L, V, Ib, Pv, Pl, Prp, S, Pc>
        where Pv: Source<V>
    {
        let index_count = index_buffer.len();
        let (_, _, instance_count) = pipeline.vertex_definition().decode(&vertices);

        let previous = CmdBindPipeline::bind_graphics_pipeline(previous, pipeline.clone());
        let device = previous.device().clone();
        let previous = CmdSetState::new(previous, device, dynamic);
        let previous = CmdBindDescriptorSets::new(previous, true, pipeline.clone(), sets).unwrap() /* TODO: error */;
        let previous = CmdPushConstants::new(previous, pipeline.clone(), push_constants).unwrap() /* TODO: error */;
        let previous = CmdBindVertexBuffers::new(previous, pipeline.vertex_definition(), vertices);
        let previous = CmdBindIndexBuffer::new(previous, index_buffer);

        // TODO: check that dynamic state is not missing some elements required by the pipeline

        CmdDrawIndexed {
            previous: previous,
            index_count: index_count as u32,
            instance_count: instance_count as u32,
            first_index: 0,
            vertex_offset: 0,
            first_instance: 0,
        }
    }
}

unsafe impl<L, V, Ib, Pv, Pl, Prp, S, Pc> CommandsList for CmdDrawIndexed<L, V, Ib, Pv, Pl, Prp, S, Pc>
    where L: CommandsList, Pl: PipelineLayoutRef, S: TrackedDescriptorSetsCollection,
          Ib: TrackedBuffer
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();
                vk.CmdDrawIndexed(cmd, self.index_count, self.instance_count, self.first_index,
                                  self.vertex_offset, self.first_instance);
            }
        }));
    }
}
