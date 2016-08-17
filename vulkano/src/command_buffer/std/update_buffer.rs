// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::any::Any;
use std::iter;
use std::iter::Empty;
use std::sync::Arc;
use smallvec::SmallVec;

use buffer::traits::CommandBufferState;
use buffer::traits::CommandListState;
use buffer::traits::PipelineBarrierRequest;
use buffer::traits::TrackedBuffer;
use command_buffer::std::OutsideRenderPass;
use command_buffer::std::StdCommandsList;
use command_buffer::submit::CommandBuffer;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use device::Queue;
use instance::QueueFamily;
use sync::AccessFlagBits;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

pub struct UpdateCommand<'a, L, B, D: ?Sized> where B: TrackedBuffer, L: StdCommandsList, D: 'static {
    previous: L,
    buffer: B,
    buffer_state: Option<B::CommandListState>,
    data: &'a D,
    transition: Option<PipelineBarrierRequest>,
}

impl<'a, L, B, D: ?Sized> UpdateCommand<'a, L, B, D>
    where B: TrackedBuffer,
          L: StdCommandsList + OutsideRenderPass,
          D: Copy + 'static,
{
    pub fn new(mut previous: L, buffer: B, data: &'a D) -> UpdateCommand<'a, L, B, D> {
        let stage = PipelineStages {
            transfer: true,
            .. PipelineStages::none()
        };

        let access = AccessFlagBits {
            transfer_write: true,
            .. AccessFlagBits::none()
        };

        let (state, transition) = unsafe {
            previous.extract_current_buffer_state(&buffer)
                    .unwrap_or(buffer.initial_state())
                    .transition(previous.num_commands() + 1, buffer.inner(),
                                0, buffer.size(), true, stage, access)
        };

        if let Some(ref transition) = transition {
            assert!(transition.after_command_num <= previous.num_commands());
        }

        UpdateCommand {
            previous: previous,
            buffer: buffer,
            buffer_state: Some(state),
            data: data,
            transition: transition,
        }
    }
}

unsafe impl<'a, L, B, D: ?Sized> StdCommandsList for UpdateCommand<'a, L, B, D>
    where B: TrackedBuffer,
          L: StdCommandsList,
          D: Copy + 'static,
{
    type Pool = L::Pool;
    type Output = UpdateCommandCb<L, B>;

    #[inline]
    fn num_commands(&self) -> usize {
        self.previous.num_commands() + 1
    }

    #[inline]
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()> {
        // No restriction
        self.previous.check_queue_validity(queue)
    }

    unsafe fn extract_current_buffer_state<Ob>(&mut self, buffer: &Ob)
                                               -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        if self.buffer.is_same(buffer) {
            let s: &mut Option<Ob::CommandListState> = (&mut self.buffer_state as &mut Any)
                                                                        .downcast_mut().unwrap();
            Some(s.take().unwrap())

        } else {
            self.previous.extract_current_buffer_state(buffer)
        }
    }

    unsafe fn raw_build<I, F>(mut self, additional_elements: F, transitions: I,
                              mut final_transitions: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        let finished_state = match self.buffer_state.take().map(|s| s.finish()) {
            Some((s, t)) => {
                if let Some(t) = t {
                    final_transitions.add_buffer_barrier_request(self.buffer.inner(), t);
                }
                Some(s)
            },
            None => None,
        };

        // We split the transitions in two: those to apply after the actual command, and those to
        // transfer to the parent so that they are applied before the actual command.

        let my_command_num = self.num_commands();

        let mut transitions_to_apply = PipelineBarrierBuilder::new();
        let mut transitions = transitions.filter_map(|(after_command_num, transition)| {
            if after_command_num >= my_command_num || !transitions_to_apply.is_empty() {
                transitions_to_apply.merge(transition);
                None
            } else {
                Some((after_command_num, transition))
            }
        }).collect::<SmallVec<[_; 8]>>();

        let my_transition = if let Some(my_transition) = self.transition.take() {
            let mut t = PipelineBarrierBuilder::new();
            let c_num = my_transition.after_command_num;
            t.add_buffer_barrier_request(self.buffer.inner(), my_transition);
            Some((c_num, t))
        } else {
            None
        };

        let transitions = my_transition.into_iter().chain(transitions.into_iter());

        let my_buffer = self.buffer;
        let my_data = self.data;
        let parent = self.previous.raw_build(|cb| {
            cb.update_buffer(my_buffer.inner(), 0, my_buffer.size(), my_data);
            cb.pipeline_barrier(transitions_to_apply);
            additional_elements(cb);
        }, transitions, final_transitions);

        UpdateCommandCb {
            previous: parent,
            buffer: my_buffer,
            buffer_state: finished_state,
        }
    }
}

pub struct UpdateCommandCb<L, B> where B: TrackedBuffer, L: StdCommandsList {
    previous: L::Output,
    buffer: B,
    buffer_state: Option<B::FinishedState>,
}

unsafe impl<L, B> CommandBuffer for UpdateCommandCb<L, B>
    where B: TrackedBuffer, L: StdCommandsList
{
    type Pool = L::Pool;
    type SemaphoresWaitIterator = Empty<(Arc<Semaphore>, PipelineStages)>;
    type SemaphoresSignalIterator = Empty<Arc<Semaphore>>;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        self.previous.inner()
    }

    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, mut fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        let parent = self.previous.on_submit(queue, &mut fence);

        let mut my_output = SubmitInfo {
            semaphores_wait: iter::empty(),     // FIXME:
            semaphores_signal: iter::empty(),     // FIXME:
            pre_pipeline_barrier: parent.pre_pipeline_barrier,
            post_pipeline_barrier: parent.post_pipeline_barrier,
        };

        if let Some(ref buffer_state) = self.buffer_state {
            let submit_infos = buffer_state.on_submit(&self.buffer, queue, fence);

            if let Some(pre) = submit_infos.pre_barrier {
                my_output.pre_pipeline_barrier.add_buffer_barrier_request(self.buffer.inner(), pre);
            }

            if let Some(post) = submit_infos.post_barrier {
                my_output.post_pipeline_barrier.add_buffer_barrier_request(self.buffer.inner(), post);
            }
        }

        my_output
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
    fn basic_submit() {
        let (device, queue) = gfx_dev_and_queue!();

        let buffer = CpuAccessibleBuffer::from_data(&device, &BufferUsage::transfer_dest(),
                                                    Some(queue.family()), 0u32).unwrap();

        let _ = PrimaryCbBuilder::new(&device, queue.family())
                    .update_buffer(buffer.clone(), &128u32)
                    .build()
                    .submit(&queue);

        let content = buffer.read(Duration::from_secs(0)).unwrap();
        assert_eq!(*content, 128);
    }
}
