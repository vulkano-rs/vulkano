// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::any::Any;
use std::iter::Chain;
use std::option::IntoIter as OptionIntoIter;
use std::sync::Arc;
use smallvec::SmallVec;

use buffer::traits::CommandBufferState;
use buffer::traits::CommandListState;
use buffer::traits::PipelineBarrierRequest;
use buffer::traits::TrackedBuffer;
use command_buffer::std::OutsideRenderPass;
use command_buffer::std::ResourcesStates;
use command_buffer::std::StdCommandsList;
use command_buffer::submit::CommandBuffer;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use device::Queue;
use image::traits::TrackedImage;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use sync::AccessFlagBits;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

/// Wraps around a commands list and adds a fill buffer command at the end of it.
pub struct FillCommand<L, B>
    where B: TrackedBuffer, L: StdCommandsList
{
    // Parent commands list.
    previous: L,
    // The buffer to fill.
    buffer: B,
    // Current state of the buffer to fill, or `None` if it has been extracted.
    buffer_state: Option<B::CommandListState>,
    // The data to fill the buffer with.
    data: u32,
    // Pipeline barrier to perform before this command.
    barrier: Option<PipelineBarrierRequest>,
}

impl<L, B> FillCommand<L, B>
    where B: TrackedBuffer,
          L: StdCommandsList + OutsideRenderPass,
{
    /// See the documentation of the `fill_buffer` method.
    pub fn new(mut previous: L, buffer: B, data: u32) -> FillCommand<L, B> {
        // Determining the new state of the buffer, and the optional pipeline barrier to add
        // before our command in the final output.
        let (state, barrier) = unsafe {
            let stage = PipelineStages { transfer: true, .. PipelineStages::none() };
            let access = AccessFlagBits { transfer_write: true, .. AccessFlagBits::none() };

            previous.extract_buffer_state(&buffer)
                    .unwrap_or(buffer.initial_state())
                    .transition(previous.num_commands() + 1, buffer.inner(),
                                0, buffer.size(), true, stage, access)
        };

        // Minor safety check.
        if let Some(ref barrier) = barrier {
            assert!(barrier.after_command_num <= previous.num_commands());
        }

        FillCommand {
            previous: previous,
            buffer: buffer,
            buffer_state: Some(state),
            data: data,
            barrier: barrier,
        }
    }
}

unsafe impl<L, B> StdCommandsList for FillCommand<L, B>
    where B: TrackedBuffer,
          L: StdCommandsList,
{
    type Pool = L::Pool;
    type Output = FillCommandCb<L::Output, B>;

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
    fn is_compute_pipeline_bound<Pl>(&self, pipeline: &Arc<ComputePipeline<Pl>>) -> bool {

        self.previous.is_compute_pipeline_bound(pipeline)
    }

    #[inline]
    fn is_graphics_pipeline_bound<Pv, Pl, Prp>(&self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Prp>>)
                                                -> bool
    {
        self.previous.is_graphics_pipeline_bound(pipeline)
    }

    unsafe fn raw_build<I, F>(mut self, additional_elements: F, barriers: I,
                              mut final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        // Computing the finished state, or `None` if we don't have to manage it.
        let finished_state = match self.buffer_state.take().map(|s| s.finish()) {
            Some((s, t)) => {
                if let Some(t) = t {
                    final_barrier.add_buffer_barrier_request(self.buffer.inner(), t);
                }
                Some(s)
            },
            None => None,
        };

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
            t.add_buffer_barrier_request(self.buffer.inner(), my_barrier);
            Some((c_num, t))
        } else {
            None
        };

        // Passing to the parent.
        let my_buffer = self.buffer;
        let my_data = self.data;
        let parent = self.previous.raw_build(|cb| {
            cb.fill_buffer(my_buffer.inner(), 0, my_buffer.size(), my_data);
            cb.pipeline_barrier(transitions_to_apply);
            additional_elements(cb);
        }, my_barrier.into_iter().chain(barriers.into_iter()), final_barrier);

        FillCommandCb {
            previous: parent,
            buffer: my_buffer,
            buffer_state: finished_state,
        }
    }
}

unsafe impl<L, B> ResourcesStates for FillCommand<L, B>
    where B: TrackedBuffer,
          L: StdCommandsList,
{
    unsafe fn extract_buffer_state<Ob>(&mut self, buffer: &Ob)
                                               -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        if self.buffer.is_same_buffer(buffer) {
            let s: &mut Option<Ob::CommandListState> = (&mut self.buffer_state as &mut Any)
                                                                        .downcast_mut().unwrap();
            Some(s.take().unwrap())

        } else {
            self.previous.extract_buffer_state(buffer)
        }
    }

    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        if self.buffer.is_same_image(image) {
            let s: &mut Option<I::CommandListState> = (&mut self.buffer_state as &mut Any)
                                                                        .downcast_mut().unwrap();
            Some(s.take().unwrap())

        } else {
            self.previous.extract_image_state(image)
        }
    }
}

unsafe impl<L, B> OutsideRenderPass for FillCommand<L, B>
    where B: TrackedBuffer,
          L: StdCommandsList,
{
}

/// Wraps around a command buffer and adds an update buffer command at the end of it.
pub struct FillCommandCb<L, B> where B: TrackedBuffer, L: CommandBuffer {
    // The previous commands.
    previous: L,
    // The buffer to update.
    buffer: B,
    // The state of the buffer to update, or `None` if we don't manage it. Will be used to
    // determine which semaphores or barriers to add when submitting.
    buffer_state: Option<B::FinishedState>,
}

unsafe impl<L, B> CommandBuffer for FillCommandCb<L, B>
    where B: TrackedBuffer, L: CommandBuffer
{
    type Pool = L::Pool;
    type SemaphoresWaitIterator = Chain<L::SemaphoresWaitIterator,
                                        OptionIntoIter<(Arc<Semaphore>, PipelineStages)>>;
    type SemaphoresSignalIterator = Chain<L::SemaphoresSignalIterator,
                                          OptionIntoIter<Arc<Semaphore>>>;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        self.previous.inner()
    }

    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, mut fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        // We query the parent.
        let parent = self.previous.on_submit(queue, &mut fence);

        // Then build our own output that modifies the parent's.

        if let Some(ref buffer_state) = self.buffer_state {
            let submit_infos = buffer_state.on_submit(&self.buffer, queue, fence);

            let mut out = SubmitInfo {
                semaphores_wait: parent.semaphores_wait.chain(submit_infos.pre_semaphore.into_iter()),
                semaphores_signal: parent.semaphores_signal.chain(submit_infos.post_semaphore.into_iter()),
                pre_pipeline_barrier: parent.pre_pipeline_barrier,
                post_pipeline_barrier: parent.post_pipeline_barrier,
            };

            if let Some(pre) = submit_infos.pre_barrier {
                out.pre_pipeline_barrier.add_buffer_barrier_request(self.buffer.inner(), pre);
            }

            if let Some(post) = submit_infos.post_barrier {
                out.post_pipeline_barrier.add_buffer_barrier_request(self.buffer.inner(), post);
            }

            out

        } else {
            SubmitInfo {
                semaphores_wait: parent.semaphores_wait.chain(None.into_iter()),
                semaphores_signal: parent.semaphores_signal.chain(None.into_iter()),
                pre_pipeline_barrier: parent.pre_pipeline_barrier,
                post_pipeline_barrier: parent.post_pipeline_barrier,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use buffer::BufferUsage;
    use buffer::CpuAccessibleBuffer;
    use command_buffer::std::PrimaryCbBuilder;
    use command_buffer::std::StdCommandsList;
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
