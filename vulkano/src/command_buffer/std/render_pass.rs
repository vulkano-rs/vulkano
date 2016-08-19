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

use buffer::traits::TrackedBuffer;
use command_buffer::std::InsideRenderPass;
use command_buffer::std::OutsideRenderPass;
use command_buffer::std::StdCommandsList;
use command_buffer::submit::CommandBuffer;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use device::Queue;
use framebuffer::Framebuffer;
use framebuffer::RenderPass;
use image::traits::TrackedImage;
use instance::QueueFamily;
use sync::Fence;

/// Wraps around a commands list and adds an update buffer command at the end of it.
pub struct BeginRenderPassCommand<L, Rp, Rpf>
    where L: StdCommandsList, Rp: RenderPass, Rpf: RenderPass
{
    // Parent commands list.
    previous: L,
    render_pass: Arc<Rp>,
    framebuffer: Arc<Framebuffer<Rpf>>,
}

impl<L, Rp> BeginRenderPassCommand<L, Rp, Rp>
    where L: StdCommandsList + OutsideRenderPass, Rp: RenderPass
{
    /// See the documentation of the `begin_render_pass` method.
    pub fn new(previous: L, framebuffer: Arc<Framebuffer<Rp>>)
        -> BeginRenderPassCommand<L, Rp, Rp>
    {
        // FIXME: transition states of the images in the framebuffer

        BeginRenderPassCommand {
            previous: previous,
            render_pass: framebuffer.render_pass().clone(),
            framebuffer: framebuffer.clone(),
        }
    }
}

unsafe impl<L, Rp, Rpf> StdCommandsList for BeginRenderPassCommand<L, Rp, Rpf>
    where L: StdCommandsList, Rp: RenderPass, Rpf: RenderPass
{
    type Pool = L::Pool;
    type Output = BeginRenderPassCommandCb<L, Rp, Rpf>;

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

    unsafe fn extract_current_buffer_state<Ob>(&mut self, buffer: &Ob)
                                               -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        // FIXME: state of images in the framebuffer
        self.previous.extract_current_buffer_state(buffer)
    }

    unsafe fn extract_current_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        // FIXME: state of images in the framebuffer
        self.previous.extract_current_image_state(image)
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

        BeginRenderPassCommandCb {
            previous: parent,
            render_pass: self.render_pass,
            framebuffer: self.framebuffer,
        }
    }
}

unsafe impl<L, Rp, Rpf> InsideRenderPass for BeginRenderPassCommand<L, Rp, Rpf>
    where L: StdCommandsList, Rp: RenderPass, Rpf: RenderPass
{
    type RenderPass = Rp;
    type Framebuffer = Arc<Framebuffer<Rpf>>;

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
    where L: StdCommandsList, Rp: RenderPass, Rpf: RenderPass
{
    // The previous commands.
    previous: L::Output,
    render_pass: Arc<Rp>,
    framebuffer: Arc<Framebuffer<Rpf>>,
}

unsafe impl<L, Rp, Rpf> CommandBuffer for BeginRenderPassCommandCb<L, Rp, Rpf>
    where L: StdCommandsList, Rp: RenderPass, Rpf: RenderPass
{
    type Pool = L::Pool;
    type SemaphoresWaitIterator = <L::Output as CommandBuffer>::SemaphoresWaitIterator;
    type SemaphoresSignalIterator = <L::Output as CommandBuffer>::SemaphoresSignalIterator;

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
        EndRenderPassCommand {
            previous: previous,
        }
    }
}

unsafe impl<L> StdCommandsList for EndRenderPassCommand<L> where L: StdCommandsList {
    type Pool = L::Pool;
    type Output = EndRenderPassCommandCb<L>;

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
    unsafe fn extract_current_buffer_state<Ob>(&mut self, buffer: &Ob)
                                               -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        self.previous.extract_current_buffer_state(buffer)
    }

    #[inline]
    unsafe fn extract_current_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        self.previous.extract_current_image_state(image)
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

unsafe impl<L> OutsideRenderPass for EndRenderPassCommand<L> where L: StdCommandsList {
}

/// Wraps around a command buffer and adds an end render pass command at the end of it.
pub struct EndRenderPassCommandCb<L> where L: StdCommandsList {
    // The previous commands.
    previous: L::Output,
}

unsafe impl<L> CommandBuffer for EndRenderPassCommandCb<L> where L: StdCommandsList {
    type Pool = L::Pool;
    type SemaphoresWaitIterator = <L::Output as CommandBuffer>::SemaphoresWaitIterator;
    type SemaphoresSignalIterator = <L::Output as CommandBuffer>::SemaphoresSignalIterator;

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
