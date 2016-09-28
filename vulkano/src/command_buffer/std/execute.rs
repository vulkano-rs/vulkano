// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use smallvec::SmallVec;

use command_buffer::states_manager::StatesManager;
use command_buffer::std::CommandsListPossibleInsideRenderPass;
use command_buffer::std::CommandsListPossibleOutsideRenderPass;
use command_buffer::std::CommandsListBase;
use command_buffer::std::CommandsList;
use command_buffer::std::CommandsListOutput;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use device::Queue;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use sync::Fence;

/// Wraps around a commands list and adds a command at the end of it that executes a secondary
/// command buffer.
pub struct ExecuteCommand<Cb, L> where Cb: CommandsListOutput, L: CommandsListBase {
    // Parent commands list.
    previous: L,
    // Command buffer to execute.
    command_buffer: Cb,
}

impl<Cb, L> ExecuteCommand<Cb, L>
    where Cb: CommandsListOutput, L: CommandsListBase
{
    /// See the documentation of the `execute` method.
    #[inline]
    pub fn new(previous: L, command_buffer: Cb) -> ExecuteCommand<Cb, L> {
        // FIXME: check that the number of subpasses is correct

        ExecuteCommand {
            previous: previous,
            command_buffer: command_buffer,
        }
    }
}

// TODO: specialize `execute()` so that multiple calls to `execute` are grouped together 
unsafe impl<Cb, L> CommandsListBase for ExecuteCommand<Cb, L>
    where Cb: CommandsListOutput, L: CommandsListBase
{
    #[inline]
    fn num_commands(&self) -> usize {
        self.previous.num_commands() + 1
    }

    #[inline]
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()> {
        // FIXME: check the secondary cb's queue validity
        self.previous.check_queue_validity(queue)
    }

    #[inline]
    fn buildable_state(&self) -> bool {
        self.previous.buildable_state()
    }

    #[inline]
    fn extract_states(&mut self) -> StatesManager {
        self.previous.extract_states()
    }

    #[inline]
    fn is_compute_pipeline_bound<Pl>(&self, pipeline: &Arc<ComputePipeline<Pl>>) -> bool {
        // Bindings are always invalidated after a execute command ends.
        false
    }

    #[inline]
    fn is_graphics_pipeline_bound<Pv, Pl, Prp>(&self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Prp>>)
                                                -> bool
    {
        // Bindings are always invalidated after a execute command ends.
        false
    }
}

// TODO: specialize `execute()` so that multiple calls to `execute` are grouped together 
unsafe impl<Cb, L> CommandsList for ExecuteCommand<Cb, L>
    where Cb: CommandsListOutput, L: CommandsList
{
    type Pool = L::Pool;
    type Output = ExecuteCommandCb<Cb, L::Output>;

    unsafe fn raw_build<I, F>(self, in_s: &mut StatesManager, out: &mut StatesManager,
                              additional_elements: F, barriers: I,
                              final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        // We split the barriers in two: those to apply after our command, and those to
        // transfer to the parent so that they are applied before our command.

        let my_command_num = self.num_commands();

        // The transitions to apply immediately after our command.
        let mut transitions_to_apply = PipelineBarrierBuilder::new();

        // The barriers to transfer to the parent.
        let barriers = barriers.filter_map(|(after_command_num, barrier)| {
            if after_command_num >= my_command_num || !transitions_to_apply.is_empty() {
                transitions_to_apply.merge(barrier);
                None
            } else {
                Some((after_command_num, barrier))
            }
        }).collect::<SmallVec<[_; 8]>>();

        // Passing to the parent.
        let parent = {
            let local_cb_to_exec = self.command_buffer.inner();
            self.previous.raw_build(in_s, out, |cb| {
                cb.execute_commands(Some(local_cb_to_exec));
                cb.pipeline_barrier(transitions_to_apply);
                additional_elements(cb);
            }, barriers.into_iter(), final_barrier)
        };

        ExecuteCommandCb {
            previous: parent,
            command_buffer: self.command_buffer,
        }
    }
}

unsafe impl<Cb, L> CommandsListPossibleInsideRenderPass for ExecuteCommand<Cb, L>
    where Cb: CommandsListOutput, L: CommandsListPossibleInsideRenderPass + CommandsListBase
{
    type RenderPass = L::RenderPass;

    #[inline]
    fn current_subpass_num(&self) -> u32 {
        self.previous.current_subpass_num()
    }

    #[inline]
    fn secondary_subpass(&self) -> bool {
        debug_assert!(self.previous.secondary_subpass());
        true
    }

    #[inline]
    fn render_pass(&self) -> &Self::RenderPass {
        self.previous.render_pass()
    }
}

unsafe impl<Cb, L> CommandsListPossibleOutsideRenderPass for ExecuteCommand<Cb, L>
    where Cb: CommandsListOutput, L: CommandsListPossibleOutsideRenderPass + CommandsListBase
{
}

/// Wraps around a command buffer and adds an execute command at the end of it.
pub struct ExecuteCommandCb<Cb, L> where Cb: CommandsListOutput, L: CommandsListOutput {
    // The previous commands.
    previous: L,
    // The secondary command buffer to execute.
    command_buffer: Cb,
}

unsafe impl<Cb, L> CommandsListOutput for ExecuteCommandCb<Cb, L>
    where Cb: CommandsListOutput, L: CommandsListOutput
{
    type Pool = L::Pool;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        self.previous.inner()
    }

    #[inline]
    unsafe fn on_submit<F>(&self, states: &StatesManager, queue: &Arc<Queue>, mut fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        self.previous.on_submit(states, queue, &mut fence)
    }
}
