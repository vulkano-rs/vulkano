// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;
use std::iter::Empty;
use std::sync::Arc;

use buffer::traits::TrackedBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::pool::StandardCommandPool;
use command_buffer::std::OutsideRenderPass;
use command_buffer::std::StdCommandsList;
use command_buffer::submit::CommandBuffer;
use command_buffer::submit::SubmitInfo;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use command_buffer::sys::Flags;
use command_buffer::sys::Kind;
use device::Device;
use device::Queue;
use framebuffer::EmptySinglePassRenderPass;
use image::traits::TrackedImage;
use instance::QueueFamily;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

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

unsafe impl<P> StdCommandsList for PrimaryCbBuilder<P> where P: CommandPool {
    type Pool = P;
    type Output = PrimaryCb<P>;

    #[inline]
    fn num_commands(&self) -> usize {
        0
    }

    #[inline]
    fn check_queue_validity(&self, queue: QueueFamily) -> Result<(), ()> {
        Ok(())
    }

    #[inline]
    unsafe fn extract_current_buffer_state<B>(&mut self, buffer: &B) -> Option<B::CommandListState>
        where B: TrackedBuffer
    {
        None
    }

    #[inline]
    unsafe fn extract_current_image_state<I>(&mut self, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        None
    }

    unsafe fn raw_build<I, F>(self, additional_elements: F, transitions: I,
                              final_transitions: PipelineBarrierBuilder) -> Self::Output
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<Self::Pool>),
              I: Iterator<Item = (usize, PipelineBarrierBuilder)>
    {
        let mut pipeline_barrier = PipelineBarrierBuilder::new();
        for (_, transition) in transitions {
            pipeline_barrier.merge(transition);
        }

        let kind = Kind::Primary::<EmptySinglePassRenderPass, EmptySinglePassRenderPass>;
        let mut cb = UnsafeCommandBufferBuilder::new(self.pool, kind,
                                                     self.flags).unwrap();  // TODO: handle
        cb.pipeline_barrier(pipeline_barrier);
        additional_elements(&mut cb);
        cb.pipeline_barrier(final_transitions);
        
        PrimaryCb {
            cb: cb.build().unwrap(),        // TODO: handle error
        }
    }
}

unsafe impl<P> OutsideRenderPass for PrimaryCbBuilder<P> where P: CommandPool {}

pub struct PrimaryCb<P = Arc<StandardCommandPool>> where P: CommandPool {
    cb: UnsafeCommandBuffer<P>,
}

unsafe impl<P> CommandBuffer for PrimaryCb<P> where P: CommandPool {
    type Pool = P;
    type SemaphoresWaitIterator = Empty<(Arc<Semaphore>, PipelineStages)>;
    type SemaphoresSignalIterator = Empty<Arc<Semaphore>>;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        &self.cb
    }

    unsafe fn on_submit<F>(&self, queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        // TODO: must handle SimultaneousUse and Once flags

        SubmitInfo {
            semaphores_wait: iter::empty(),
            semaphores_signal: iter::empty(),
            pre_pipeline_barrier: PipelineBarrierBuilder::new(),
            post_pipeline_barrier: PipelineBarrierBuilder::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use command_buffer::std::PrimaryCbBuilder;
    use command_buffer::std::StdCommandsList;
    use command_buffer::submit::CommandBuffer;

    #[test]
    fn basic_submit() {
        let (device, queue) = gfx_dev_and_queue!();
        let _ = PrimaryCbBuilder::new(&device, queue.family()).build().submit(&queue);
    }
}
