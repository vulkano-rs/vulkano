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

use buffer::traits::TrackedBuffer;
use command_buffer::states_manager::StatesManager;
use command_buffer::std::OutsideRenderPass;
use command_buffer::std::StdCommandsList;
use command_buffer::submit::CommandBuffer;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use descriptor::PipelineLayout;
use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use device::Queue;
use image::traits::TrackedImage;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use sync::Fence;
use VulkanObject;

/// Wraps around a commands list and adds a dispatch command at the end of it.
pub struct DispatchCommand<'a, L, Pl, S, Pc>
    where L: StdCommandsList, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection, Pc: 'a
{
    // Parent commands list.
    previous: L,
    // The compute pipeline.
    pipeline: Arc<ComputePipeline<Pl>>,
    // The descriptor sets to bind.
    sets: S,
    // Pipeline barrier to inject in the final command buffer.
    pipeline_barrier: (usize, PipelineBarrierBuilder),
    // The push constants.   TODO: use Cow
    push_constants: &'a Pc,
    // Dispatch dimensions.
    dimensions: [u32; 3],
    // States of the resources, or `None` if it has been extracted.
    resources_states: Option<StatesManager>,
}

impl<'a, L, Pl, S, Pc> DispatchCommand<'a, L, Pl, S, Pc>
    where L: StdCommandsList + OutsideRenderPass, Pl: PipelineLayout,
          S: TrackedDescriptorSetsCollection, Pc: 'a
{
    /// See the documentation of the `dispatch` method.
    pub fn new(mut previous: L, pipeline: Arc<ComputePipeline<Pl>>, sets: S, dimensions: [u32; 3],
               push_constants: &'a Pc) -> DispatchCommand<'a, L, Pl, S, Pc>
    {
        let mut states = previous.extract_states();

        let (barrier_loc, barrier) = unsafe {
            sets.transition(&mut states)
        };

        DispatchCommand {
            previous: previous,
            pipeline: pipeline,
            sets: sets,
            pipeline_barrier: (barrier_loc, barrier),
            push_constants: push_constants,
            dimensions: dimensions,
            resources_states: Some(states),
        }
    }
}

unsafe impl<'a, L, Pl, S, Pc> StdCommandsList for DispatchCommand<'a, L, Pl, S, Pc>
    where L: StdCommandsList, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection, Pc: 'a
{
    type Pool = L::Pool;
    type Output = DispatchCommandCb<L::Output, Pl, S>;

    #[inline]
    fn num_commands(&self) -> usize {
        self.previous.num_commands() + 1
    }

    #[inline]
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()> {
        if !queue.supports_compute() {
            return Err(());
        }

        self.previous.check_queue_validity(queue)
    }

    #[inline]
    fn is_compute_pipeline_bound<OPl>(&self, pipeline: &Arc<ComputePipeline<OPl>>) -> bool {
        pipeline.internal_object() == self.pipeline.internal_object()
    }

    #[inline]
    fn is_graphics_pipeline_bound<Pv, OPl, Prp>(&self, pipeline: &Arc<GraphicsPipeline<Pv, OPl, Prp>>)
                                                 -> bool
    {
        self.previous.is_graphics_pipeline_bound(pipeline)
    }

    #[inline]
    fn extract_states(&mut self) -> StatesManager {
        self.resources_states.take().unwrap()
    }

    #[inline]
    fn buildable_state(&self) -> bool {
        true
    }

    unsafe fn raw_build<I, F>(self, additional_elements: F, barriers: I,
                              mut final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        let my_command_num = self.num_commands();

        // Computing the finished state of the sets.
        let (finished_state, fb) = self.sets.finish(self.sets_state);
        final_barrier.merge(fb);

        // We split the barriers in two: those to apply after our command, and those to
        // transfer to the parent so that they are applied before our command.

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

        // Moving out some values, otherwise Rust complains that the closure below uses `self`
        // while it's partially moved out.
        let my_barrier = self.pipeline_barrier;
        let my_pipeline = self.pipeline;
        let my_sets = self.sets;
        let my_push_constants = self.push_constants;
        let my_dimensions = self.dimensions;
        let bind_pipeline = !self.previous.is_compute_pipeline_bound(&my_pipeline);

        // Passing to the parent.
        let parent = self.previous.raw_build(|cb| {
            // TODO: is the pipeline layout always the same as in the compute pipeline? 
            if bind_pipeline {
                cb.bind_pipeline_compute(&my_pipeline);
            }

            let sets: SmallVec<[_; 8]> = my_sets.list().collect();      // TODO: ideally shouldn't collect, but there are lifetime problems
            cb.bind_descriptor_sets(false, &**my_pipeline.layout(), 0,
                                    sets.iter().map(|s| s.inner()), iter::empty());         // TODO: dynamic ranges, and don't bind if not necessary
            cb.push_constants(&**my_pipeline.layout(), ShaderStages::all(), 0,        // TODO: stages
                              &my_push_constants);
            cb.dispatch(my_dimensions[0], my_dimensions[1], my_dimensions[2]);
            cb.pipeline_barrier(transitions_to_apply);
            additional_elements(cb);
        }, Some(my_barrier).into_iter().chain(barriers.into_iter()), final_barrier);

        DispatchCommandCb {
            previous: parent,
            pipeline: my_pipeline,
            sets: my_sets,
            sets_state: finished_state,
        }
    }
}

unsafe impl<'a, L, Pl, S, Pc> OutsideRenderPass for DispatchCommand<'a, L, Pl, S, Pc>
    where L: StdCommandsList, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection, Pc: 'a
{
}

/// Wraps around a command buffer and adds an update buffer command at the end of it.
pub struct DispatchCommandCb<L, Pl, S>
    where L: CommandBuffer, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection
{
    // The previous commands.
    previous: L,
    // The barrier. We store it here to keep it alive.
    pipeline: Arc<ComputePipeline<Pl>>,
    // The descriptor sets. Stored here to keep them alive.
    sets: S,
}

unsafe impl<L, Pl, S> CommandBuffer for DispatchCommandCb<L, Pl, S>
    where L: CommandBuffer, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection
{
    type Pool = L::Pool;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        self.previous.inner()
    }

    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, mut fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        // We query the parent.
        let mut parent = self.previous.on_submit(queue, &mut fence);

        // We query our sets.
        let my_infos = self.sets.on_submit(&self.sets_state, queue, fence);

        // We merge the two.
        SubmitInfo {
            semaphores_wait: {
                parent.semaphores_wait.extend(my_infos.semaphores_wait.into_iter());
                parent.semaphores_wait
            },
            semaphores_signal: {
                parent.semaphores_signal.extend(my_infos.semaphores_signal.into_iter());
                parent.semaphores_signal
            },
            pre_pipeline_barrier: {
                let mut b = parent.pre_pipeline_barrier;
                b.merge(my_infos.pre_pipeline_barrier);
                b
            },
            post_pipeline_barrier: {
                let mut b = parent.post_pipeline_barrier;
                b.merge(my_infos.post_pipeline_barrier);
                b
            },
        }
    }
}
