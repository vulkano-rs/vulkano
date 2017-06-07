// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use command_buffer::CommandAddError;
use command_buffer::DynamicState;
use command_buffer::cb::AddCommand;
use command_buffer::commands_raw::CmdBindDescriptorSets;
use command_buffer::commands_raw::CmdBindPipeline;
use command_buffer::commands_raw::CmdBindVertexBuffers;
use command_buffer::commands_raw::CmdDrawRaw;
use command_buffer::commands_raw::CmdPushConstants;
use command_buffer::commands_raw::CmdSetState;
use descriptor::descriptor_set::DescriptorSetsCollection;
use pipeline::GraphicsPipelineAbstract;
use pipeline::vertex::VertexSource;

/// Command that draws non-indexed vertices.
pub struct CmdDraw<V, P, S, Pc> {
    vertex_buffers: CmdBindVertexBuffers<V>,
    push_constants: CmdPushConstants<Pc, P>,
    descriptor_sets: CmdBindDescriptorSets<S, P>,
    set_state: CmdSetState,
    bind_pipeline: CmdBindPipeline<P>,
    draw_raw: CmdDrawRaw,
}

impl<V, P, S, Pc> CmdDraw<V, P, S, Pc>
    where P: GraphicsPipelineAbstract, S: DescriptorSetsCollection
{
    /// See the documentation of the `draw` method.
    pub fn new(pipeline: P, dynamic: DynamicState, vertices: V, sets: S, push_constants: Pc)
               -> CmdDraw<V, P, S, Pc>
        where P: VertexSource<V> + Clone
    {
        let (_, vertex_count, instance_count) = pipeline.decode(&vertices);

        let bind_pipeline = CmdBindPipeline::bind_graphics_pipeline(pipeline.clone());
        let device = bind_pipeline.device().clone();
        let set_state = CmdSetState::new(device, dynamic);
        let descriptor_sets = CmdBindDescriptorSets::new(true, pipeline.clone(), sets).unwrap() /* TODO: error */;
        let push_constants = CmdPushConstants::new(pipeline.clone(), push_constants).unwrap() /* TODO: error */;
        let vertex_buffers = CmdBindVertexBuffers::new(&pipeline, vertices);
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

unsafe impl<Cb, V, P, S, Pc, O, O1, O2, O3, O4, O5> AddCommand<CmdDraw<V, P, S, Pc>> for Cb
    where Cb: AddCommand<CmdBindVertexBuffers<V>, Out = O1>,
          O1: AddCommand<CmdPushConstants<Pc, P>, Out = O2>,
          O2: AddCommand<CmdBindDescriptorSets<S, P>, Out = O3>,
          O3: AddCommand<CmdSetState, Out = O4>,
          O4: AddCommand<CmdBindPipeline<P>, Out = O5>,
          O5: AddCommand<CmdDrawRaw, Out = O>
{
    type Out = O;

    #[inline]
    fn add(self, command: CmdDraw<V, P, S, Pc>) -> Result<Self::Out, CommandAddError> {
        Ok(self.add(command.vertex_buffers)?
               .add(command.push_constants)?
               .add(command.descriptor_sets)?
               .add(command.set_state)?
               .add(command.bind_pipeline)?
               .add(command.draw_raw)?)
    }
}
