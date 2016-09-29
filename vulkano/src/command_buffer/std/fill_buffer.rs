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

use buffer::TrackedBuffer;
use buffer::TrackedBufferPipelineBarrierRequest;
use command_buffer::states_manager::StatesManager;
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
use sync::AccessFlagBits;
use sync::Fence;
use sync::PipelineStages;

/// Wraps around a commands list and adds a fill buffer command at the end of it.
pub struct FillCommand<L, B>
    where B: TrackedBuffer, L: CommandsListBase
{
    // Parent commands list.
    previous: L,
    // The buffer to fill.
    buffer: B,
    // The data to fill the buffer with.
    data: u32,
    // Pipeline barrier to perform before this command.
    barrier: Option<TrackedBufferPipelineBarrierRequest>,
    // States of the resources, or `None` if it has been extracted.
    resources_states: Option<StatesManager>,
}

impl<L, B> FillCommand<L, B>
    where B: TrackedBuffer,
          L: CommandsListBase + CommandsListPossibleOutsideRenderPass,
{
    /// See the documentation of the `fill_buffer` method.
    pub fn new(mut previous: L, buffer: B, data: u32) -> FillCommand<L, B> {
        let mut states = previous.extract_states();

        // Determining the new state of the buffer, and the optional pipeline barrier to add
        // before our command in the final output.
        let barrier = {
            let stage = PipelineStages { transfer: true, .. PipelineStages::none() };
            let access = AccessFlagBits { transfer_write: true, .. AccessFlagBits::none() };

            buffer.transition(&mut states, previous.num_commands() + 1, 0, buffer.size(), true,
                              stage, access)
        };

        // Minor safety check.
        if let Some(ref barrier) = barrier {
            assert!(barrier.after_command_num <= previous.num_commands());
        }

        FillCommand {
            previous: previous,
            buffer: buffer,
            data: data,
            barrier: barrier,
            resources_states: Some(states),
        }
    }
}

unsafe impl<L, B> CommandsListBase for FillCommand<L, B>
    where B: TrackedBuffer,
          L: CommandsListBase,
{
    #[inline]
    fn num_commands(&self) -> usize {
        self.previous.num_commands() + 1
    }

    #[inline]
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()> {
        // No restriction
        self.previous.check_queue_validity(queue)
    }

    #[inline]
    fn buildable_state(&self) -> bool {
        true
    }

    #[inline]
    fn extract_states(&mut self) -> StatesManager {
        self.resources_states.take().unwrap()
    }

    #[inline]
    fn is_compute_pipeline_bound<Pl>(&self, pipeline: &Arc<ComputePipeline<Pl>>) -> bool {

        self.previous.is_compute_pipeline_bound(pipeline)
    }

    #[inline]
    fn is_graphics_pipeline_bound<Pv, Pl, Prp>(&self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Prp>>)
                                                -> bool
    {
        self.previous.is_graphics_pipeline_bound(pipeline)
    }
}

unsafe impl<L, B> CommandsList for FillCommand<L, B>
    where B: TrackedBuffer,
          L: CommandsList,
{
    type Pool = L::Pool;
    type Output = FillCommandCb<L::Output, B>;

    unsafe fn raw_build<I, F>(mut self, in_s: &mut StatesManager, out: &mut StatesManager,
                              additional_elements: F, barriers: I,
                              mut final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        if let Some(t) = self.buffer.finish(in_s, out) {
            final_barrier.add_buffer_barrier_request(&self.buffer, t);
        }

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

        // The local barrier requested by this command, or `None` if no barrier requested.
        let my_barrier = if let Some(my_barrier) = self.barrier.take() {
            let mut t = PipelineBarrierBuilder::new();
            let c_num = my_barrier.after_command_num;
            t.add_buffer_barrier_request(&self.buffer, my_barrier);
            Some((c_num, t))
        } else {
            None
        };

        // Passing to the parent.
        let my_buffer = self.buffer;
        let my_data = self.data;
        let parent = self.previous.raw_build(in_s, out, |cb| {
            cb.fill_buffer(&my_buffer, my_data);
            cb.pipeline_barrier(transitions_to_apply);
            additional_elements(cb);
        }, my_barrier.into_iter().chain(barriers.into_iter()), final_barrier);

        FillCommandCb {
            previous: parent,
            buffer: my_buffer,
        }
    }
}

unsafe impl<L, B> CommandsListPossibleOutsideRenderPass for FillCommand<L, B>
    where B: TrackedBuffer,
          L: CommandsListBase,
{
}

/// Wraps around a command buffer and adds an update buffer command at the end of it.
pub struct FillCommandCb<L, B> where B: TrackedBuffer, L: CommandsListOutput {
    // The previous commands.
    previous: L,
    // The buffer to update.
    buffer: B,
}

unsafe impl<L, B> CommandsListOutput for FillCommandCb<L, B>
    where B: TrackedBuffer, L: CommandsListOutput
{
    type Pool = L::Pool;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        self.previous.inner()
    }

    unsafe fn on_submit<F>(&self, states: &StatesManager, queue: &Arc<Queue>, mut fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        // We query the parent.
        let mut parent = self.previous.on_submit(states, queue, &mut fence);

        // Then build our own output that modifies the parent's.
        let submit_infos = self.buffer.on_submit(states, queue, fence);

        parent.semaphores_wait.extend(submit_infos.pre_semaphore.into_iter());
        parent.semaphores_signal.extend(submit_infos.post_semaphore.into_iter());

        if let Some(pre) = submit_infos.pre_barrier {
            parent.pre_pipeline_barrier.add_buffer_barrier_request(&self.buffer, pre);
        }

        if let Some(post) = submit_infos.post_barrier {
            parent.post_pipeline_barrier.add_buffer_barrier_request(&self.buffer, post);
        }

        parent
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use buffer::BufferUsage;
    use buffer::CpuAccessibleBuffer;
    use command_buffer::std::PrimaryCbBuilder;
    use command_buffer::std::CommandsListBase;
    use command_buffer::submit::CommandBuffer;

    #[test]
    fn basic() {
        let (device, queue) = gfx_dev_and_queue!();

        let buffer = CpuAccessibleBuffer::from_data(&device, &BufferUsage::transfer_dest(),
                                                    Some(queue.family()), 0u32).unwrap();

        let _ = PrimaryCbBuilder::new(&device, queue.family())
                    .fill_buffer(buffer.clone(), 128u32)
                    .build()
                    .submit(&queue);

        let content = buffer.read(Duration::from_secs(0)).unwrap();
        assert_eq!(*content, 128);
    }
}
