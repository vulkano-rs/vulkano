// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use command_buffer::DynamicState;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use VulkanObject;
use vk;

/// Keep track of the state of a command buffer builder, so that you don't need to bind objects
/// that were already bound.
///
/// > **Important**: Executing a secondary command buffer invalidates the state of a command buffer
/// > builder. When you do so, you need to call `invalidate()`.
pub struct StateCacher {
    // The dynamic state to synchronize with `CmdSetState`.
    dynamic_state: DynamicState,
    // The compute pipeline currently bound. 0 if nothing bound.
    compute_pipeline: vk::Pipeline,
    // The graphics pipeline currently bound. 0 if nothing bound.
    graphics_pipeline: vk::Pipeline,
}

/// Outcome of an operation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StateCacherOutcome {
    /// The caller needs to perform the state change in the actual command buffer builder.
    NeedChange,
    /// The state change is not necessary.
    AlreadyOk,
}

impl StateCacher {
    /// Builds a new `StateCacher`.
    #[inline]
    pub fn new() -> StateCacher {
        StateCacher {
            dynamic_state: DynamicState::none(),
            compute_pipeline: 0,
            graphics_pipeline: 0,
        }
    }

    /// Resets the cache to its default state. You **must** call this after executing a secondary
    /// command buffer.
    #[inline]
    pub fn invalidate(&mut self) {
        self.dynamic_state = DynamicState::none();
        self.compute_pipeline = 0;
        self.graphics_pipeline = 0;
    }

    /// Checks whether we need to bind a graphics pipeline. Returns `StateCacherOutcome::AlreadyOk`
    /// if the pipeline was already bound earlier, and `StateCacherOutcome::NeedChange` if you need
    /// to actually bind the pipeline.
    pub fn bind_graphics_pipeline<P>(&mut self, pipeline: &P) -> StateCacherOutcome
        where P: GraphicsPipelineAbstract
    {
        let inner = GraphicsPipelineAbstract::inner(pipeline).internal_object();
        if inner == self.graphics_pipeline {
            StateCacherOutcome::AlreadyOk
        } else {
            self.graphics_pipeline = inner;
            StateCacherOutcome::NeedChange
        }
    }

    /// Checks whether we need to bind a compute pipeline. Returns `StateCacherOutcome::AlreadyOk`
    /// if the pipeline was already bound earlier, and `StateCacherOutcome::NeedChange` if you need
    /// to actually bind the pipeline.
    pub fn bind_compute_pipeline<P>(&mut self, pipeline: &P) -> StateCacherOutcome
        where P: ComputePipelineAbstract
    {
        let inner = pipeline.inner().internal_object();
        if inner == self.compute_pipeline {
            StateCacherOutcome::AlreadyOk
        } else {
            self.compute_pipeline = inner;
            StateCacherOutcome::NeedChange
        }
    }
}

/*
unsafe impl<I, O> AddCommand<commands_raw::CmdSetState> for StateCacher<I>
    where I: AddCommand<commands_raw::CmdSetState, Out = O>
{
    type Out = StateCacher<O>;

    #[inline]
    fn add(mut self, command: commands_raw::CmdSetState) -> Result<Self::Out, CommandAddError> {
        // We need to synchronize `self.dynamic_state` with the state in `command`.
        // While doing so, we tweak `command` to erase the states that are the same as what's
        // already in `self.dynamic_state`.

        let mut command_state = command.state().clone();

        // Handle line width.
        if let Some(new_val) = command_state.line_width {
            if self.dynamic_state.line_width == Some(new_val) {
                command_state.line_width = None;
            } else {
                self.dynamic_state.line_width = Some(new_val);
            }
        }

        // TODO: missing implementations

        Ok(StateCacher {
            inner: self.inner.add(commands_raw::CmdSetState::new(command.device().clone(), command_state))?,
            dynamic_state: self.dynamic_state,
            graphics_pipeline: self.graphics_pipeline,
            compute_pipeline: self.compute_pipeline,
            vertex_buffers: self.vertex_buffers,
        })
    }
}

unsafe impl<I, O, B> AddCommand<commands_raw::CmdBindVertexBuffers<B>> for StateCacher<I>
    where I: AddCommand<commands_raw::CmdBindVertexBuffers<B>, Out = O>
{
    type Out = StateCacher<O>;

    #[inline]
    fn add(mut self, mut command: commands_raw::CmdBindVertexBuffers<B>)
           -> Result<Self::Out, CommandAddError>
    {
        match &mut self.vertex_buffers {
            &mut Some(ref mut curr) => {
                if *curr != *command.hash() {
                    let new_hash = command.hash().clone();
                    command.diff(curr);
                    *curr = new_hash;
                }
            },
            curr @ &mut None => {
                *curr = Some(command.hash().clone());
            }
        };

        Ok(StateCacher {
            inner: self.inner.add(command)?,
            dynamic_state: self.dynamic_state,
            graphics_pipeline: self.graphics_pipeline,
            compute_pipeline: self.compute_pipeline,
            vertex_buffers: self.vertex_buffers,
        })
    }
}*/
