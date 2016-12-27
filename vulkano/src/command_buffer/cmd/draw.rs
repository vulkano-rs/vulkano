// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use command_buffer::DynamicState;
use command_buffer::cmd::CmdBindDescriptorSets;
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
use pipeline::vertex::Source;
use VulkanPointers;

/// Wraps around a commands list and adds a draw command at the end of it.
pub struct CmdDraw<L, V, Pv, Pl, Prp, S, Pc>
    where L: CommandsList,
          Pl: PipelineLayoutRef,
          S: TrackedDescriptorSetsCollection
{
    // Parent commands list.
    previous: CmdBindVertexBuffers<CmdPushConstants<CmdBindDescriptorSets<CmdSetState<CmdBindPipeline<L, Arc<GraphicsPipeline<Pv, Pl, Prp>>>>, S, Arc<GraphicsPipeline<Pv, Pl, Prp>>>, Pc, Arc<GraphicsPipeline<Pv, Pl, Prp>>>, V>,

    // Parameters for vkCmdDraw.
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

impl<L, V, Pv, Pl, Prp, S, Pc> CmdDraw<L, V, Pv, Pl, Prp, S, Pc>
    where L: CommandsList,
          Pl: PipelineLayoutRef,
          S: TrackedDescriptorSetsCollection
{
    /// See the documentation of the `draw` method.
    pub fn new(previous: L, pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>, dynamic: DynamicState, vertices: V, sets: S, push_constants: Pc) -> CmdDraw<L, V, Pv, Pl, Prp, S, Pc>
        where Pv: Source<V>
    {
        let (_, vertex_count, instance_count) = pipeline.vertex_definition().decode(&vertices);

        let previous = CmdBindPipeline::bind_graphics_pipeline(previous, pipeline.clone());
        let device = previous.device().clone();
        let previous = CmdSetState::new(previous, device, dynamic);
        let previous = CmdBindDescriptorSets::new(previous, true, pipeline.clone(), sets).unwrap() /* TODO: error */;
        let previous = CmdPushConstants::new(previous, pipeline.clone(), push_constants).unwrap() /* TODO: error */;
        let previous = CmdBindVertexBuffers::new(previous, pipeline.vertex_definition(), vertices);

        // TODO: check that dynamic state is not missing some elements required by the pipeline

        CmdDraw {
            previous: previous,
            vertex_count: vertex_count as u32,
            instance_count: instance_count as u32,
            first_vertex: 0,
            first_instance: 0,
        }
    }
}

unsafe impl<L, V, Pv, Pl, Prp, S, Pc> CommandsList for CmdDraw<L, V, Pv, Pl, Prp, S, Pc>
    where L: CommandsList,
          Pl: PipelineLayoutRef,
          S: TrackedDescriptorSetsCollection
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();
                vk.CmdDraw(cmd,
                           self.vertex_count,
                           self.instance_count,
                           self.first_vertex,
                           self.first_instance);
            }
        }));
    }
}
