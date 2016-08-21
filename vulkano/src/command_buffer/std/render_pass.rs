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
use framebuffer::Framebuffer;
use framebuffer::RenderPass;
use framebuffer::RenderPassClearValues;
use image::traits::TrackedImage;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use sync::Fence;

/// Wraps around a commands list and adds an update buffer command at the end of it.
pub struct BeginRenderPassCommand<L, Rp, Rpf>
    where L: StdCommandsList, Rp: RenderPass, Rpf: RenderPass
{
    // Parent commands list.
    previous: L,
    // True if only secondary command buffers can be added.
    secondary: bool,
    rect: [Range<u32>; 2],
    clear_values: SmallVec<[ClearValue; 6]>,
    render_pass: Arc<Rp>,
    framebuffer: Arc<Framebuffer<Rpf>>,
}

impl<L, Rp> BeginRenderPassCommand<L, Rp, Rp>
    where L: StdCommandsList + OutsideRenderPass, Rp: RenderPass
{
    /// See the documentation of the `begin_render_pass` method.
    // TODO: allow setting more parameters
    pub fn new<C>(previous: L, framebuffer: Arc<Framebuffer<Rp>>, secondary: bool, clear_values: C)
                  -> BeginRenderPassCommand<L, Rp, Rp>
        where Rp: RenderPassClearValues<C>
    {
        // FIXME: transition states of the images in the framebuffer

        let clear_values = framebuffer.render_pass().convert_clear_values(clear_values)
                                      .collect();

        BeginRenderPassCommand {
            previous: previous,
            secondary: secondary,
            rect: [0 .. framebuffer.width(), 0 .. framebuffer.height()],
            clear_values: clear_values,
            render_pass: framebuffer.render_pass().clone(),
            framebuffer: framebuffer.clone(),
        }
    }
}

unsafe impl<L, Rp, Rpf> StdCommandsList for BeginRenderPassCommand<L, Rp, Rpf>
    where L: StdCommandsList, Rp: RenderPass, Rpf: RenderPass
{
    type Pool = L::Pool;
    type Output = BeginRenderPassCommandCb<L::Output, Rp, Rpf>;

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
                              final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<L::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        let my_command_num = self.num_commands();
        let barriers = barriers.map(move |(n, b)| { assert!(n < my_command_num); (n, b) });

        let my_render_pass = self.render_pass;
        let my_framebuffer = self.framebuffer;
        let mut my_clear_values = self.clear_values;
        let my_rect = self.rect;
        let my_secondary = self.secondary;

        let parent = self.previous.raw_build(|cb| {
            cb.begin_render_pass(my_render_pass.inner(), &my_framebuffer,
                                 my_clear_values.into_iter(), my_rect, my_secondary);
            additional_elements(cb);
        }, barriers, final_barrier);

        BeginRenderPassCommandCb {
            previous: parent,
            render_pass: my_render_pass,
            framebuffer: my_framebuffer,
        }
    }
}

unsafe impl<L, Rp, Rpf> ResourcesStates for BeginRenderPassCommand<L, Rp, Rpf>
    where L: StdCommandsList, Rp: RenderPass, Rpf: RenderPass
{
    unsafe fn extract_buffer_state<Ob>(&mut self, buffer: &Ob)
                                               -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        // FIXME: state of images in the framebuffer
        self.previous.extract_buffer_state(buffer)
    }

    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        // FIXME: state of images in the framebuffer
        self.previous.extract_image_state(image)
    }
}

unsafe impl<L, Rp, Rpf> InsideRenderPass for BeginRenderPassCommand<L, Rp, Rpf>
    where L: StdCommandsList, Rp: RenderPass, Rpf: RenderPass
{
    type RenderPass = Rp;
    type Framebuffer = Arc<Framebuffer<Rpf>>;

    #[inline]
    fn current_subpass(&self) -> u32 {
        0
    }

    #[inline]
    fn secondary_subpass(&self) -> bool {
        self.secondary
    }

    #[inline]
    fn render_pass(&self) -> &Arc<Self::RenderPass> {
        &self.render_pass
    }

    #[inline]
    fn framebuffer(&self) -> &Self::Framebuffer {
        &self.framebuffer
    }
}

/// Wraps around a command buffer and adds an update buffer command at the end of it.
pub struct BeginRenderPassCommandCb<L, Rp, Rpf>
    where L: CommandBuffer, Rp: RenderPass, Rpf: RenderPass
{
    // The previous commands.
    previous: L,
    render_pass: Arc<Rp>,
    framebuffer: Arc<Framebuffer<Rpf>>,
}

unsafe impl<L, Rp, Rpf> CommandBuffer for BeginRenderPassCommandCb<L, Rp, Rpf>
    where L: CommandBuffer, Rp: RenderPass, Rpf: RenderPass
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
        self.previous.on_submit(queue, &mut fence)
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
    fn render_pass(&self) -> &Arc<Self::RenderPass> {
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
