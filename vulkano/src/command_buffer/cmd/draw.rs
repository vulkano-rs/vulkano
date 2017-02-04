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
use command_buffer::cb::AddCommand;
use command_buffer::cmd::CmdBindDescriptorSets;
use command_buffer::cmd::CmdBindPipeline;
use command_buffer::cmd::CmdBindVertexBuffers;
use command_buffer::cmd::CmdDrawRaw;
use command_buffer::cmd::CmdPushConstants;
use command_buffer::cmd::CmdSetState;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::PipelineLayoutAbstract;
use pipeline::GraphicsPipeline;
use pipeline::vertex::VertexSource;

/// Command that draws non-indexed vertices.
pub struct CmdDraw<V, Pv, Pl, Prp, S, Pc> {
    vertex_buffers: CmdBindVertexBuffers<V>,
    push_constants: CmdPushConstants<Pc, Arc<GraphicsPipeline<Pv, Pl, Prp>>>,
    descriptor_sets: CmdBindDescriptorSets<S, Arc<GraphicsPipeline<Pv, Pl, Prp>>>,
    set_state: CmdSetState,
    bind_pipeline: CmdBindPipeline<Arc<GraphicsPipeline<Pv, Pl, Prp>>>,
    draw_raw: CmdDrawRaw,
}

impl<V, Pv, Pl, Prp, S, Pc> CmdDraw<V, Pv, Pl, Prp, S, Pc>
    where Pl: PipelineLayoutAbstract, S: DescriptorSetsCollection
{
    /// See the documentation of the `draw` method.
    pub fn new(pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
               dynamic: DynamicState, vertices: V, sets: S, push_constants: Pc)
               -> CmdDraw<V, Pv, Pl, Prp, S, Pc>
        where Pv: VertexSource<V>
    {
        let (_, vertex_count, instance_count) = pipeline.vertex_definition().decode(&vertices);

        let bind_pipeline = CmdBindPipeline::bind_graphics_pipeline(pipeline.clone());
        let device = bind_pipeline.device().clone();
        let set_state = CmdSetState::new(device, dynamic);
        let descriptor_sets = CmdBindDescriptorSets::new(true, pipeline.clone(), sets).unwrap() /* TODO: error */;
        let push_constants = CmdPushConstants::new(pipeline.clone(), push_constants).unwrap() /* TODO: error */;
        let vertex_buffers = CmdBindVertexBuffers::new(pipeline.vertex_definition(), vertices);
        let draw_raw = unsafe { CmdDrawRaw::new(vertex_count as u32, instance_count as u32, 0, 0) };

        CmdDraw {
            vertex_buffers: vertex_buffers,
            push_constants: push_constants,
            descriptor_sets: descriptor_sets,
            set_state: set_state,
            bind_pipeline: bind_pipeline,
            draw_raw: draw_raw,
        }
    }
}

unsafe impl<Cb, V, Pv, Pl, Prp, S, Pc, O, O1, O2, O3, O4, O5> AddCommand<CmdDraw<V, Pv, Pl, Prp, S, Pc>> for Cb
    where Cb: AddCommand<CmdBindVertexBuffers<V>, Out = O1>,
          O1: AddCommand<CmdPushConstants<Pc, Arc<GraphicsPipeline<Pv, Pl, Prp>>>, Out = O2>,
          O2: AddCommand<CmdBindDescriptorSets<S, Arc<GraphicsPipeline<Pv, Pl, Prp>>>, Out = O3>,
          O3: AddCommand<CmdSetState, Out = O4>,
          O4: AddCommand<CmdBindPipeline<Arc<GraphicsPipeline<Pv, Pl, Prp>>>, Out = O5>,
          O5: AddCommand<CmdDrawRaw, Out = O>
{
    type Out = O;

    #[inline]
    fn add(self, command: CmdDraw<V, Pv, Pl, Prp, S, Pc>) -> O {
        self.add(command.vertex_buffers)
            .add(command.push_constants)
            .add(command.descriptor_sets)
            .add(command.set_state)
            .add(command.bind_pipeline)
            .add(command.draw_raw)
    }
}
