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

use buffer::Buffer;
use command_buffer::DynamicState;
use command_buffer::states_manager::StatesManager;
use command_buffer::std::CommandsListPossibleInsideRenderPass;
use command_buffer::std::CommandsListBase;
use command_buffer::std::CommandsList;
use command_buffer::std::CommandsListOutput;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use descriptor::PipelineLayout;
use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use device::Device;
use device::Queue;
use instance::QueueFamily;
use pipeline::GraphicsPipeline;
use pipeline::vertex::Source;
use sync::Fence;
use VulkanObject;
use vk;

/// Wraps around a commands list and adds a draw command at the end of it.
pub struct DrawCommand<'a, L, Pv, Pl, Prp, S, Pc>
    where L: CommandsListBase, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection, Pc: 'a
{
    // Parent commands list.
    previous: L,
    // The graphics pipeline.
    pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
    // The descriptor sets to bind.
    sets: S,
    // Pipeline barrier to inject in the final command buffer.
    pipeline_barrier: (usize, PipelineBarrierBuilder),
    // The push constants.   TODO: use Cow
    push_constants: &'a Pc,
    // FIXME: strong typing and state transitions
    vertex_buffers: SmallVec<[Arc<Buffer>; 4]>,
    // States of the resources, or `None` if it has been extracted.
    resources_states: Option<StatesManager>,
    // Actual type of draw.
    inner: DrawInner,
}

enum DrawInner {
    Regular {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },

    Indexed {
        vertex_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    },

    // TODO: indirect rendering
}

impl<'a, L, Pv, Pl, Prp, S, Pc> DrawCommand<'a, L, Pv, Pl, Prp, S, Pc>
    where L: CommandsListBase + CommandsListPossibleInsideRenderPass, Pl: PipelineLayout,
          S: TrackedDescriptorSetsCollection, Pc: 'a
{
    /// See the documentation of the `draw` method.
    pub fn regular<V>(mut previous: L, pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
                      dynamic: &DynamicState, vertices: V, sets: S, push_constants: &'a Pc)
                      -> DrawCommand<'a, L, Pv, Pl, Prp, S, Pc>
        where Pv: Source<V>
    {
        let mut states = previous.extract_states();

        let (barrier_loc, barrier) = unsafe {
            sets.transition(&mut states)
        };

        // FIXME: lot of stuff missing here

        let (buffers, num_vertices, num_instances) = pipeline.vertex_definition().decode(vertices);
        let buffers = buffers.collect();

        DrawCommand {
            previous: previous,
            pipeline: pipeline,
            sets: sets,
            pipeline_barrier: (barrier_loc, barrier),
            push_constants: push_constants,
            vertex_buffers: buffers,
            resources_states: Some(states),
            inner: DrawInner::Regular {
                vertex_count: num_vertices as u32,
                instance_count: num_instances as u32,
                first_vertex: 0,
                first_instance: 0,
            },
        }
    }
}

unsafe impl<'a, L, Pv, Pl, Prp, S, Pc> CommandsListBase for DrawCommand<'a, L, Pv, Pl, Prp, S, Pc>
    where L: CommandsListBase, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection, Pc: 'a
{
    #[inline]
    fn num_commands(&self) -> usize {
        self.previous.num_commands() + 1
    }

    #[inline]
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()> {
        if !queue.supports_graphics() {
            return Err(());
        }

        self.previous.check_queue_validity(queue)
    }

    #[inline]
    fn is_compute_pipeline_bound(&self, pipeline: vk::Pipeline) -> bool {
        self.previous.is_compute_pipeline_bound(pipeline)
    }

    #[inline]
    fn is_graphics_pipeline_bound(&self, pipeline: vk::Pipeline) -> bool {
        self.pipeline.internal_object() == pipeline
    }

    #[inline]
    fn extract_states(&mut self) -> StatesManager {
        self.resources_states.take().unwrap()
    }

    #[inline]
    fn buildable_state(&self) -> bool {
        self.previous.buildable_state()
    }
}

unsafe impl<'a, L, Pv, Pl, Prp, S, Pc> CommandsList for DrawCommand<'a, L, Pv, Pl, Prp, S, Pc>
    where L: CommandsList, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection, Pc: 'a
{
    type Pool = L::Pool;
    type Output = DrawCommandCb<L::Output, Pv, Pl, Prp, S>;


    unsafe fn raw_build<I, F>(self, in_s: &mut StatesManager, out: &mut StatesManager,
                              additional_elements: F, barriers: I,
                              mut final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        let my_command_num = self.num_commands();

        // Computing the finished state of the sets.
        final_barrier.merge(self.sets.finish(in_s, out));

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
        let bind_pipeline = !self.previous.is_graphics_pipeline_bound(my_pipeline.internal_object());
        let my_sets = self.sets;
        let my_push_constants = self.push_constants;
        let my_vertex_buffers = self.vertex_buffers;
        let my_inner = self.inner;

        // Passing to the parent.
        let parent = self.previous.raw_build(in_s, out, |cb| {
            // TODO: is the pipeline layout always the same as in the graphics pipeline? 
            if bind_pipeline {
                cb.bind_pipeline_graphics(&my_pipeline);
            }

            let sets: SmallVec<[_; 8]> = my_sets.list().collect();      // TODO: ideally shouldn't collect, but there are lifetime problems
            cb.bind_descriptor_sets(true, &**my_pipeline.layout(), 0,
                                    sets.iter().map(|s| s.inner()), iter::empty());         // TODO: dynamic ranges, and don't bind if not necessary
            cb.push_constants(&**my_pipeline.layout(), ShaderStages::all(), 0,        // TODO: stages
                              &my_push_constants);

            cb.bind_vertex_buffers(0, my_vertex_buffers.iter().map(|buf| (buf.inner().buffer, 0)));

            match my_inner {
                DrawInner::Regular { vertex_count, instance_count,
                                     first_vertex, first_instance } =>
                {
                    cb.draw(vertex_count, instance_count, first_vertex, first_instance);
                },
                DrawInner::Indexed { vertex_count, instance_count, first_index,
                                     vertex_offset, first_instance } =>
                {
                    cb.draw_indexed(vertex_count, instance_count, first_index,
                                    vertex_offset, first_instance);
                },
            }

            cb.pipeline_barrier(transitions_to_apply);
            additional_elements(cb);
        }, Some(my_barrier).into_iter().chain(barriers.into_iter()), final_barrier);

        DrawCommandCb {
            previous: parent,
            pipeline: my_pipeline,
            sets: my_sets,
            vertex_buffers: my_vertex_buffers,
        }
    }
}

unsafe impl<'a, L, Pv, Pl, Prp, S, Pc> CommandsListPossibleInsideRenderPass for DrawCommand<'a, L, Pv, Pl, Prp, S, Pc>
    where L: CommandsListBase + CommandsListPossibleInsideRenderPass, Pl: PipelineLayout,
          S: TrackedDescriptorSetsCollection, Pc: 'a
{
    type RenderPass = L::RenderPass;

    #[inline]
    fn current_subpass_num(&self) -> u32 {
        self.previous.current_subpass_num()
    }

    #[inline]
    fn secondary_subpass(&self) -> bool {
        self.previous.secondary_subpass()
    }

    #[inline]
    fn render_pass(&self) -> &Self::RenderPass {
        self.previous.render_pass()
    }
}

/// Wraps around a command buffer and adds an update buffer command at the end of it.
pub struct DrawCommandCb<L, Pv, Pl, Prp, S>
    where L: CommandsListOutput, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection
{
    // The previous commands.
    previous: L,
    // The barrier. We store it here to keep it alive.
    pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
    // The descriptor sets. Stored here to keep them alive.
    sets: S,
    // FIXME: strong typing and state transitions
    vertex_buffers: SmallVec<[Arc<Buffer>; 4]>,
}

unsafe impl<L, Pv, Pl, Prp, S> CommandsListOutput for DrawCommandCb<L, Pv, Pl, Prp, S>
    where L: CommandsListOutput, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection
{
    #[inline]
    fn inner(&self) -> vk::CommandBuffer {
        self.previous.inner()
    }

    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.previous.device()
    }

    unsafe fn on_submit<F>(&self, states: &StatesManager, queue: &Arc<Queue>, mut fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        // We query the parent.
        let mut parent = self.previous.on_submit(states, queue, &mut fence);

        // We query our sets.
        let my_infos = self.sets.on_submit(states, queue, fence);

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
