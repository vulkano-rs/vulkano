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
use std::ops::Range;
use smallvec::SmallVec;

use buffer::traits::TrackedBuffer;
use command_buffer::std::InsideRenderPass;
use command_buffer::std::OutsideRenderPass;
use command_buffer::std::ResourcesStates;
use command_buffer::std::StdCommandsList;
use command_buffer::submit::CommandBuffer;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use device::Queue;
use format::ClearValue;
use framebuffer::traits::TrackedFramebuffer;
use framebuffer::RenderPass;
use framebuffer::RenderPassClearValues;
use image::traits::TrackedImage;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use sync::Fence;

/// Wraps around a commands list and adds an update buffer command at the end of it.
pub struct BeginRenderPassCommand<L, Rp, F>
    where L: StdCommandsList, Rp: RenderPass, F: TrackedFramebuffer
{
    // Parent commands list.
    previous: L,
    // True if only secondary command buffers can be added.
    secondary: bool,
    rect: [Range<u32>; 2],
    clear_values: SmallVec<[ClearValue; 6]>,
    // If `None`, then the render pass used in the command is the framebuffer's. If `Some`, then it
    // is a different (but compatible) render pass.
    render_pass: Option<Rp>,
    framebuffer: F,
    framebuffer_state: F::State,
    barrier_position: usize,
    barrier: PipelineBarrierBuilder,
}

impl<L, F> BeginRenderPassCommand<L, F::RenderPass, F>
    where L: StdCommandsList + OutsideRenderPass, F: TrackedFramebuffer
{
    /// See the documentation of the `begin_render_pass` method.
    // TODO: allow setting more parameters
    pub fn new<C>(mut previous: L, framebuffer: F, secondary: bool, clear_values: C)
                  -> BeginRenderPassCommand<L, F::RenderPass, F>
        where F::RenderPass: RenderPassClearValues<C>
    {
        let (state, barrier_pos, barrier) = unsafe {
            framebuffer.extract_and_transition(previous.num_commands() + 1, &mut previous)
        };

        let clear_values = framebuffer.render_pass().convert_clear_values(clear_values)
                                      .collect();

        let rect = [0 .. framebuffer.dimensions()[0], 0 .. framebuffer.dimensions()[1]];

        BeginRenderPassCommand {
            previous: previous,
            secondary: secondary,
            rect: rect,
            clear_values: clear_values,
            render_pass: None,
            framebuffer: framebuffer,
            framebuffer_state: state,
            barrier_position: barrier_pos,
            barrier: barrier,
        }
    }
}

unsafe impl<L, Rp, Fb> StdCommandsList for BeginRenderPassCommand<L, Rp, Fb>
    where L: StdCommandsList, Rp: RenderPass, Fb: TrackedFramebuffer
{
    type Pool = L::Pool;
    type Output = BeginRenderPassCommandCb<L::Output, Rp, Fb>;

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
    fn buildable_state(&self) -> bool {
        // We are no longer in a buildable state after entering a render pass.
        false
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

    unsafe fn raw_build<I, F>(self, additional_elements: F, barriers: I,
                              mut final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        let my_command_num = self.num_commands();
        let barriers = barriers.map(move |(n, b)| { assert!(n < my_command_num); (n, b) });

        let (finished_fb_state, local_final_barrier) = {
            self.framebuffer.finish(self.framebuffer_state)
        };
        final_barrier.merge(local_final_barrier);

        let my_render_pass = self.render_pass;
        let my_framebuffer = self.framebuffer;
        let my_clear_values = self.clear_values;
        let my_rect = self.rect;
        let my_secondary = self.secondary;
        
        let barriers_for_parent = Some((self.barrier_position, self.barrier)).into_iter()
                                                                             .chain(barriers);

        let parent = self.previous.raw_build(|cb| {
            cb.begin_render_pass(my_render_pass.as_ref().map(|rp| rp.inner())
                                               .unwrap_or(my_framebuffer.render_pass().inner()),
                                 &my_framebuffer, my_clear_values.into_iter(),
                                 my_rect, my_secondary);
            additional_elements(cb);
        }, barriers_for_parent, final_barrier);

        BeginRenderPassCommandCb {
            previous: parent,
            render_pass: my_render_pass,
            framebuffer: my_framebuffer,
            state: finished_fb_state,
        }
    }
}

unsafe impl<L, Rp, F> ResourcesStates for BeginRenderPassCommand<L, Rp, F>
    where L: StdCommandsList, Rp: RenderPass, F: TrackedFramebuffer
{
    #[inline]
    unsafe fn extract_buffer_state<Ob>(&mut self, buffer: &Ob)
                                               -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        if let Some(buf) = self.framebuffer.extract_buffer_state(&mut self.framebuffer_state, buffer) {
            return Some(buf);
        }

        self.previous.extract_buffer_state(buffer)
    }

    #[inline]
    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        if let Some(img) = self.framebuffer.extract_image_state(&mut self.framebuffer_state, image) {
            return Some(img);
        }

        self.previous.extract_image_state(image)
    }
}

unsafe impl<L, Rp, F> InsideRenderPass for BeginRenderPassCommand<L, Rp, F>
    where L: StdCommandsList, Rp: RenderPass, F: TrackedFramebuffer
{
    type RenderPass = Rp;
    type Framebuffer = F;

    #[inline]
    fn current_subpass(&self) -> u32 {
        0
    }

    #[inline]
    fn secondary_subpass(&self) -> bool {
        self.secondary
    }

    #[inline]
    fn render_pass(&self) -> &Self::RenderPass {
        if let Some(ref rp) = self.render_pass {
            rp
        } else {
            panic!()        // TODO:
            //self.framebuffer.render_pass()
        }
    }

    #[inline]
    fn framebuffer(&self) -> &Self::Framebuffer {
        &self.framebuffer
    }
}

/// Wraps around a command buffer and adds an update buffer command at the end of it.
pub struct BeginRenderPassCommandCb<L, Rp, F>
    where L: CommandBuffer, Rp: RenderPass, F: TrackedFramebuffer
{
    // The previous commands.
    previous: L,
    render_pass: Option<Rp>,
    framebuffer: F,
    state: F::Finished,
}

unsafe impl<L, Rp, Fb> CommandBuffer for BeginRenderPassCommandCb<L, Rp, Fb>
    where L: CommandBuffer, Rp: RenderPass, Fb: TrackedFramebuffer
{
    type Pool = L::Pool;
    type SemaphoresWaitIterator = L::SemaphoresWaitIterator;
    type SemaphoresSignalIterator = L::SemaphoresSignalIterator;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        self.previous.inner()
    }

    #[inline]
    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, mut fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        // FIXME: merge semaphore iterators

        let framebuffer_submit_reqs = self.framebuffer.on_submit(&self.state, queue, &mut fence);
        let parent_reqs = self.previous.on_submit(queue, &mut fence);

        assert!(framebuffer_submit_reqs.semaphores_wait.count() == 0);        // not implemented
        assert!(framebuffer_submit_reqs.semaphores_signal.count() == 0);      // not implemented

        SubmitInfo {
            semaphores_wait: parent_reqs.semaphores_wait,
            semaphores_signal: parent_reqs.semaphores_signal,
            pre_pipeline_barrier: {
                let mut b = parent_reqs.pre_pipeline_barrier;
                b.merge(framebuffer_submit_reqs.pre_pipeline_barrier);
                b
            },
            post_pipeline_barrier: {
                let mut b = parent_reqs.post_pipeline_barrier;
                b.merge(framebuffer_submit_reqs.post_pipeline_barrier);
                b
            },
        }
    }
}

/// Wraps around a commands list and adds a command at the end of it that jumps to the next subpass.
pub struct NextSubpassCommand<L> where L: StdCommandsList {
    // Parent commands list.
    previous: L,
    // True if only secondary command buffers can be added.
    secondary: bool,
}

impl<L> NextSubpassCommand<L> where L: StdCommandsList + InsideRenderPass {
    /// See the documentation of the `next_subpass` method.
    #[inline]
    pub fn new(previous: L, secondary: bool) -> NextSubpassCommand<L> {
        // FIXME: put this check
        //assert!(previous.current_subpass() + 1 < previous.render_pass().num_subpasses());      // TODO: error instead

        NextSubpassCommand {
            previous: previous,
            secondary: secondary,
        }
    }
}

unsafe impl<L> StdCommandsList for NextSubpassCommand<L>
    where L: StdCommandsList + InsideRenderPass
{
    type Pool = L::Pool;
    type Output = NextSubpassCommandCb<L::Output>;

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
    fn buildable_state(&self) -> bool {
        false
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

    unsafe fn raw_build<I, F>(self, additional_elements: F, barriers: I,
                              final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        let secondary = self.secondary;

        let parent = self.previous.raw_build(|cb| {
            cb.next_subpass(secondary);
            additional_elements(cb);
        }, barriers, final_barrier);

        NextSubpassCommandCb {
            previous: parent,
        }
    }
}

unsafe impl<L> ResourcesStates for NextSubpassCommand<L>
    where L: StdCommandsList + InsideRenderPass
{
    #[inline]
    unsafe fn extract_buffer_state<Ob>(&mut self, buffer: &Ob)
                                               -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        self.previous.extract_buffer_state(buffer)
    }

    #[inline]
    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        self.previous.extract_image_state(image)
    }
}

unsafe impl<L> InsideRenderPass for NextSubpassCommand<L>
    where L: StdCommandsList + InsideRenderPass
{
    type RenderPass = L::RenderPass;
    type Framebuffer = L::Framebuffer;

    #[inline]
    fn current_subpass(&self) -> u32 {
        self.previous.current_subpass() + 1
    }

    #[inline]
    fn secondary_subpass(&self) -> bool {
        self.secondary
    }

    #[inline]
    fn render_pass(&self) -> &Self::RenderPass {
        self.previous.render_pass()
    }

    #[inline]
    fn framebuffer(&self) -> &Self::Framebuffer {
        self.previous.framebuffer()
    }
}

/// Wraps around a command buffer and adds an end render pass command at the end of it.
pub struct NextSubpassCommandCb<L> where L: CommandBuffer {
    // The previous commands.
    previous: L,
}

unsafe impl<L> CommandBuffer for NextSubpassCommandCb<L> where L: CommandBuffer {
    type Pool = L::Pool;
    type SemaphoresWaitIterator = L::SemaphoresWaitIterator;
    type SemaphoresSignalIterator = L::SemaphoresSignalIterator;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        self.previous.inner()
    }

    #[inline]
    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, mut fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        self.previous.on_submit(queue, &mut fence)
    }
}

/// Wraps around a commands list and adds an end render pass command at the end of it.
pub struct EndRenderPassCommand<L> where L: StdCommandsList {
    // Parent commands list.
    previous: L,
}

impl<L> EndRenderPassCommand<L> where L: StdCommandsList + InsideRenderPass {
    /// See the documentation of the `end_render_pass` method.
    #[inline]
    pub fn new(previous: L) -> EndRenderPassCommand<L> {
        // FIXME: check that the number of subpasses is correct

        EndRenderPassCommand {
            previous: previous,
        }
    }
}

unsafe impl<L> StdCommandsList for EndRenderPassCommand<L> where L: StdCommandsList {
    type Pool = L::Pool;
    type Output = EndRenderPassCommandCb<L::Output>;

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

    unsafe fn raw_build<I, F>(self, additional_elements: F, barriers: I,
                              final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        // We need to flush all the barriers because regular (ie. non-self-referencing) barriers
        // aren't allowed inside render passes.

        let mut pipeline_barrier = PipelineBarrierBuilder::new();
        for (num, barrier) in barriers {
            debug_assert!(num <= self.num_commands());
            pipeline_barrier.merge(barrier);
        }

        let parent = self.previous.raw_build(|cb| {
            cb.end_render_pass();
            cb.pipeline_barrier(pipeline_barrier);
            additional_elements(cb);
        }, iter::empty(), final_barrier);

        EndRenderPassCommandCb {
            previous: parent,
        }
    }
}

unsafe impl<L> ResourcesStates for EndRenderPassCommand<L> where L: StdCommandsList {
    #[inline]
    unsafe fn extract_buffer_state<Ob>(&mut self, buffer: &Ob)
                                               -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        self.previous.extract_buffer_state(buffer)
    }

    #[inline]
    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        self.previous.extract_image_state(image)
    }
}

unsafe impl<L> OutsideRenderPass for EndRenderPassCommand<L> where L: StdCommandsList {
}

/// Wraps around a command buffer and adds an end render pass command at the end of it.
pub struct EndRenderPassCommandCb<L> where L: CommandBuffer {
    // The previous commands.
    previous: L,
}

unsafe impl<L> CommandBuffer for EndRenderPassCommandCb<L> where L: CommandBuffer {
    type Pool = L::Pool;
    type SemaphoresWaitIterator = L::SemaphoresWaitIterator;
    type SemaphoresSignalIterator = L::SemaphoresSignalIterator;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        self.previous.inner()
    }

    #[inline]
    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, mut fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        self.previous.on_submit(queue, &mut fence)
    }
}
