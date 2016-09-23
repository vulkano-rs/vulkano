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
use image::traits::TrackedImage;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use sync::Fence;

/// Wraps around a commands list and adds a command at the end of it that executes a secondary
/// command buffer.
pub struct ExecuteCommand<Cb, L> where Cb: CommandBuffer, L: StdCommandsList {
    // Parent commands list.
    previous: L,
    // Command buffer to execute.
    command_buffer: Cb,
}

impl<Cb, L> ExecuteCommand<Cb, L>
    where Cb: CommandBuffer, L: StdCommandsList
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
unsafe impl<Cb, L> StdCommandsList for ExecuteCommand<Cb, L>
    where Cb: CommandBuffer, L: StdCommandsList
{
    type Pool = L::Pool;
    type Output = ExecuteCommandCb<Cb, L::Output>;

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

    unsafe fn raw_build<I, F>(self, additional_elements: F, barriers: I,
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
            self.previous.raw_build(|cb| {
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

unsafe impl<Cb, L> ResourcesStates for ExecuteCommand<Cb, L>
    where Cb: CommandBuffer, L: StdCommandsList
{
    #[inline]
    unsafe fn extract_buffer_state<Ob>(&mut self, buffer: &Ob)
                                               -> Option<Ob::CommandListState>
        where Ob: TrackedBuffer
    {
        // FIXME:
        self.previous.extract_buffer_state(buffer)
    }

    #[inline]
    unsafe fn extract_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        // FIXME:
        self.previous.extract_image_state(image)
    }
}

unsafe impl<Cb, L> InsideRenderPass for ExecuteCommand<Cb, L>
    where Cb: CommandBuffer, L: InsideRenderPass
{
    type RenderPass = L::RenderPass;
    type Framebuffer = L::Framebuffer;

    #[inline]
    fn current_subpass(&self) -> u32 {
        self.previous.current_subpass()
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

    #[inline]
    fn framebuffer(&self) -> &Self::Framebuffer {
        self.previous.framebuffer()
    }
}

unsafe impl<Cb, L> OutsideRenderPass for ExecuteCommand<Cb, L>
    where Cb: CommandBuffer, L: OutsideRenderPass
{
}

/// Wraps around a command buffer and adds an execute command at the end of it.
pub struct ExecuteCommandCb<Cb, L> where Cb: CommandBuffer, L: CommandBuffer {
    // The previous commands.
    previous: L,
    // The secondary command buffer to execute.
    command_buffer: Cb,
}

unsafe impl<Cb, L> CommandBuffer for ExecuteCommandCb<Cb, L>
    where Cb: CommandBuffer, L: CommandBuffer
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
