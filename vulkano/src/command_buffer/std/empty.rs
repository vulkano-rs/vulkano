// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::traits::TrackedBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::pool::StandardCommandPool;
use command_buffer::states_manager::StatesManager;
use command_buffer::std::CommandsListPossibleOutsideRenderPass;
use command_buffer::std::CommandsListBase;
use command_buffer::std::CommandsList;
use command_buffer::std::CommandsListOutput;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use command_buffer::sys::Flags;
use command_buffer::sys::Kind;
use device::Device;
use device::Queue;
use framebuffer::EmptySinglePassRenderPass;
use framebuffer::StdFramebuffer;
use framebuffer::framebuffer::EmptyAttachmentsList;
use image::traits::TrackedImage;
use instance::QueueFamily;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use sync::Fence;

pub struct PrimaryCbBuilder<P = Arc<StandardCommandPool>> where P: CommandPool {
    pool: P,
    flags: Flags,
}

impl PrimaryCbBuilder<Arc<StandardCommandPool>> {
    /// Builds a new primary command buffer builder.
    #[inline]
    pub fn new(device: &Arc<Device>, family: QueueFamily)
               -> PrimaryCbBuilder<Arc<StandardCommandPool>>
    {
        PrimaryCbBuilder::with_pool(Device::standard_command_pool(device, family))
    }
}

impl<P> PrimaryCbBuilder<P> where P: CommandPool {
    /// Builds a new primary command buffer builder that uses a specific pool.
    pub fn with_pool(pool: P) -> PrimaryCbBuilder<P> {
        PrimaryCbBuilder {
            pool: pool,
            flags: Flags::SimultaneousUse,      // TODO: allow customization
        }
    }
}

unsafe impl<P> CommandsListBase for PrimaryCbBuilder<P> where P: CommandPool {
    #[inline]
    fn num_commands(&self) -> usize {
        0
    }

    #[inline]
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()> {
        Ok(())
    }

    #[inline]
    fn is_compute_pipeline_bound<Pl>(&self, pipeline: &Arc<ComputePipeline<Pl>>) -> bool {

        false
    }

    #[inline]
    fn is_graphics_pipeline_bound<Pv, Pl, Prp>(&self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Prp>>)
                                                -> bool
    {
        false
    }

    #[inline]
    fn extract_states(&mut self) -> StatesManager {
        StatesManager::new()
    }

    #[inline]
    fn buildable_state(&self) -> bool {
        true
    }
}

unsafe impl<P> CommandsList for PrimaryCbBuilder<P> where P: CommandPool {
    type Pool = P;
    type Output = PrimaryCb<P>;

    unsafe fn raw_build<I, F>(self, _: &mut StatesManager, _: &mut StatesManager,
                              additional_elements: F, barriers: I,
                              final_barrier: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<Self::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        let kind = Kind::Primary::<EmptySinglePassRenderPass,
                                   StdFramebuffer<EmptySinglePassRenderPass, EmptyAttachmentsList>>;
        let mut cb = UnsafeCommandBufferBuilder::new(self.pool, kind,
                                                     self.flags).unwrap();  // TODO: handle error

        // Since we're at the start of the command buffer, there's no need wonder when to add the
        // barriers. We have no choice but to add them immediately.
        let mut pipeline_barrier = PipelineBarrierBuilder::new();
        for (_, barrier) in barriers {
            pipeline_barrier.merge(barrier);
        }
        cb.pipeline_barrier(pipeline_barrier);

        // Then add the rest.
        additional_elements(&mut cb);
        cb.pipeline_barrier(final_barrier);
        
        PrimaryCb {
            cb: cb.build().unwrap(),        // TODO: handle error
        }
    }
}

unsafe impl<P> CommandsListPossibleOutsideRenderPass for PrimaryCbBuilder<P> where P: CommandPool {}

pub struct PrimaryCb<P = Arc<StandardCommandPool>> where P: CommandPool {
    cb: UnsafeCommandBuffer<P>,
}

unsafe impl<P> CommandsListOutput for PrimaryCb<P> where P: CommandPool {
    type Pool = P;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        &self.cb
    }

    unsafe fn on_submit<F>(&self, states: &StatesManager, queue: &Arc<Queue>, fence: F) -> SubmitInfo
        where F: FnMut() -> Arc<Fence>
    {
        // TODO: Must handle non-SimultaneousUse and Once flags ; for now the `SimultaneousUse`
        //       flag is mandatory, so there's no safety issue. However it will need to be handled
        //       before allowing other flags to be used.

        SubmitInfo {
            semaphores_wait: vec![],
            semaphores_signal: vec![],
            pre_pipeline_barrier: PipelineBarrierBuilder::new(),
            post_pipeline_barrier: PipelineBarrierBuilder::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use command_buffer::std::PrimaryCbBuilder;
    use command_buffer::std::CommandsListBase;
    use command_buffer::submit::CommandBuffer;

    #[test]
    fn basic_submit() {
        let (device, queue) = gfx_dev_and_queue!();
        let _ = PrimaryCbBuilder::new(&device, queue.family()).build().submit(&queue);
    }
}
