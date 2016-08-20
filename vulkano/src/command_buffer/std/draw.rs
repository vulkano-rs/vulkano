// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;
use std::iter::Chain;
use std::sync::Arc;
use smallvec::SmallVec;

use buffer::traits::Buffer;
use buffer::traits::TrackedBuffer;
use command_buffer::DynamicState;
use command_buffer::std::InsideRenderPass;
use command_buffer::std::StdCommandsList;
use command_buffer::submit::CommandBuffer;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use descriptor::PipelineLayout;
use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollection;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollectionState;
use descriptor::descriptor_set::collection::TrackedDescriptorSetsCollectionFinished;
use device::Queue;
use image::traits::TrackedImage;
use instance::QueueFamily;
use pipeline::GraphicsPipeline;
use pipeline::vertex::Source;
use sync::Fence;

/// Wraps around a commands list and adds a draw command at the end of it.
pub struct DrawCommand<'a, L, Pv, Pl, Prp, S, Pc>
    where L: StdCommandsList, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection, Pc: 'a
{
    // Parent commands list.
    previous: L,
    // The graphics pipeline.
    pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
    // The descriptor sets to bind.
    sets: S,
    // The state of the descriptor sets.
    sets_state: S::State,
    // Pipeline barrier to inject in the final command buffer.
    pipeline_barrier: (usize, PipelineBarrierBuilder),
    // The push constants.   TODO: use Cow
    push_constants: &'a Pc,
    // FIXME: strong typing and state transitions
    vertex_buffers: SmallVec<[Arc<Buffer>; 4]>,
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
    where L: StdCommandsList + InsideRenderPass, Pl: PipelineLayout,
          S: TrackedDescriptorSetsCollection, Pc: 'a
{
    /// See the documentation of the `draw` method.
    pub fn regular<V>(mut previous: L, pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
                      dynamic: &DynamicState, vertices: V, sets: S, push_constants: &'a Pc)
                      -> DrawCommand<'a, L, Pv, Pl, Prp, S, Pc>
        where Pv: Source<V>
    {
        let (sets_state, barrier_loc, barrier) = unsafe {
            sets.extract_from_commands_list_and_transition(&mut previous)
        };

        // FIXME: lot of stuff missing here

        let (buffers, num_vertices, num_instances) = pipeline.vertex_definition().decode(vertices);
        let buffers = buffers.collect();

        DrawCommand {
            previous: previous,
            pipeline: pipeline,
            sets: sets,
            sets_state: sets_state,
            pipeline_barrier: (barrier_loc, barrier),
            push_constants: push_constants,
            vertex_buffers: buffers,
            inner: DrawInner::Regular {
                vertex_count: num_vertices as u32,
                instance_count: num_instances as u32,
                first_vertex: 0,
                first_instance: 0,
            },
        }
    }
}

unsafe impl<'a, L, Pv, Pl, Prp, S, Pc> StdCommandsList for DrawCommand<'a, L, Pv, Pl, Prp, S, Pc>
    where L: StdCommandsList, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection, Pc: 'a
{
    type Pool = L::Pool;
    type Output = DrawCommandCb<L::Output, Pv, Pl, Prp, S>;

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

    unsafe fn extract_current_buffer_state<Ob>(&mut self, buffer: &Ob)
                                               -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        if let Some(s) = self.sets_state.extract_buffer_state(buffer) {
            return Some(s);
        }

        self.previous.extract_current_buffer_state(buffer)
    }

    unsafe fn extract_current_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        if let Some(s) = self.sets_state.extract_image_state(image) {
            return Some(s);
        }

        self.previous.extract_current_image_state(image)
    }

    #[inline]
    fn buildable_state(&self) -> bool {
        self.previous.buildable_state()
    }

    unsafe fn raw_build<I, F>(self, additional_elements: F, barriers: I,
                              mut final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        let my_command_num = self.num_commands();

        // Computing the finished state of the sets.
        let (finished_state, fb) = self.sets_state.finish();
        final_barrier.merge(fb);

        // We split the barriers in two: those to apply after our command, and those to
        // transfer to the parent so that they are applied before our command.

        // The transitions to apply immediately after our command.
        let mut transitions_to_apply = PipelineBarrierBuilder::new();

        // The barriers to transfer to the parent.
        let mut barriers = barriers.filter_map(|(after_command_num, barrier)| {
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
        let my_vertex_buffers = self.vertex_buffers;
        let my_inner = self.inner;

        // Passing to the parent.
        let parent = self.previous.raw_build(|cb| {
            // TODO: is the pipeline layout always the same as in the graphics pipeline? 
            cb.bind_pipeline_graphics(&my_pipeline);       // TODO: don't bind if not necessary
            let sets: SmallVec<[_; 8]> = my_sets.list().collect();      // TODO: ideally shouldn't collect, but there are lifetime problems
            cb.bind_descriptor_sets(true, &**my_pipeline.layout(), 0,
                                    sets.iter().map(|s| s.inner()), iter::empty());         // TODO: dynamic ranges, and don't bind if not necessary
            cb.push_constants(&**my_pipeline.layout(), ShaderStages::all(), 0,        // TODO: stages
                              &my_push_constants);

            cb.bind_vertex_buffers(0, my_vertex_buffers.iter().map(|buf| (buf.inner(), 0)));

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
            sets_state: finished_state,
            vertex_buffers: my_vertex_buffers,
        }
    }
}

unsafe impl<'a, L, Pv, Pl, Prp, S, Pc> InsideRenderPass for DrawCommand<'a, L, Pv, Pl, Prp, S, Pc>
    where L: StdCommandsList + InsideRenderPass, Pl: PipelineLayout,
          S: TrackedDescriptorSetsCollection, Pc: 'a
{
    type RenderPass = L::RenderPass;
    type Framebuffer = L::Framebuffer;

    #[inline]
    fn current_subpass(&self) -> u32 {
        self.previous.current_subpass()
    }

    #[inline]
    fn secondary_subpass(&self) -> bool {
        self.previous.secondary_subpass()
    }

    #[inline]
    fn render_pass(&self) -> &Arc<Self::RenderPass> {
        self.previous.render_pass()
    }

    #[inline]
    fn framebuffer(&self) -> &Self::Framebuffer {
        self.previous.framebuffer()
    }
}

/// Wraps around a command buffer and adds an update buffer command at the end of it.
pub struct DrawCommandCb<L, Pv, Pl, Prp, S>
    where L: CommandBuffer, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection
{
    // The previous commands.
    previous: L,
    // The barrier. We store it here to keep it alive.
    pipeline: Arc<GraphicsPipeline<Pv, Pl, Prp>>,
    // The descriptor sets. Stored here to keep them alive.
    sets: S,
    // State of the descriptor sets.
    sets_state: S::Finished,
    // FIXME: strong typing and state transitions
    vertex_buffers: SmallVec<[Arc<Buffer>; 4]>,
}

unsafe impl<L, Pv, Pl, Prp, S> CommandBuffer for DrawCommandCb<L, Pv, Pl, Prp, S>
    where L: CommandBuffer, Pl: PipelineLayout, S: TrackedDescriptorSetsCollection
{
    type Pool = L::Pool;
    type SemaphoresWaitIterator = Chain<L::SemaphoresWaitIterator,
                                        <S::Finished as TrackedDescriptorSetsCollectionFinished>::
                                            SemaphoresWaitIterator>;
    type SemaphoresSignalIterator = Chain<L::SemaphoresSignalIterator,
                                          <S::Finished as TrackedDescriptorSetsCollectionFinished>::
                                            SemaphoresSignalIterator>;

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

        // We query our sets.
        let my_infos = self.sets_state.on_submit(queue, fence);

        // We merge the two.
        SubmitInfo {
            semaphores_wait: parent.semaphores_wait.chain(my_infos.semaphores_wait),
            semaphores_signal: parent.semaphores_signal.chain(my_infos.semaphores_signal),
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
