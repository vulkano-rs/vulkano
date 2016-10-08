// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::fmt;
use std::ptr;
use std::sync::Arc;
use std::time::Duration;
use smallvec::SmallVec;

use command_buffer::sys::PipelineBarrierBuilder;
use device::Device;
use device::Queue;
use sync::Fence;
use sync::FenceWaitError;
use sync::PipelineStages;
use sync::Semaphore;

use check_errors;
use vk;
use VulkanObject;
use VulkanPointers;
use SynchronizedVulkanObject;

/// Trait for objects that can be submitted to the GPU.
pub unsafe trait Submit {
    /// Submits the command buffer.
    ///
    /// Note that since submitting has a fixed overhead, you should try, if possible, to submit
    /// multiple command buffers at once instead.
    ///
    /// This is a simple shortcut for creating a `Submit` object.
    #[inline]
    fn submit(self, queue: &Arc<Queue>) -> Submission<Self> where Self: Sized {
        submit(self, queue)
    }

    /// Returns the inner object.
    // TODO: crappy API
    fn inner(&self) -> vk::CommandBuffer;

    /// Returns the device this object belongs to.
    fn device(&self) -> &Arc<Device>;

    /// Called slightly before the object is submitted. The function must return a `SubmitBuilder`
    /// containing the list of things to submit.
    ///
    /// # Safety for the caller
    ///
    /// This function must only be called if there's actually a submission with the returned
    /// parameters that follows.
    ///
    /// This function is supposed to be called only by vulkano's internals. It is recommended
    /// that you never call it.
    ///
    /// # Safety for the implementation
    ///
    /// To write.
    unsafe fn append_submission(&self, base: SubmitBuilder, queue: &Arc<Queue>) -> SubmitBuilder;
}

/// Information about how the submitting function should synchronize the submission.
// TODO: move or remove
pub struct SubmitInfo {
    /// List of semaphores to wait upon before the command buffer starts execution.
    pub semaphores_wait: Vec<(Arc<Semaphore>, PipelineStages)>,
    /// List of semaphores to signal after the command buffer has finished.
    pub semaphores_signal: Vec<Arc<Semaphore>>,
    /// Pipeline barrier to execute on the queue and immediately before the command buffer.
    /// Ignored if empty.
    pub pre_pipeline_barrier: PipelineBarrierBuilder,
    /// Pipeline barrier to execute on the queue and immediately after the command buffer.
    /// Ignored if empty.
    pub post_pipeline_barrier: PipelineBarrierBuilder,
}

impl SubmitInfo {
    #[inline]
    pub fn empty() -> SubmitInfo {
        SubmitInfo {
            semaphores_wait: Vec::new(),
            semaphores_signal: Vec::new(),
            pre_pipeline_barrier: PipelineBarrierBuilder::new(),
            post_pipeline_barrier: PipelineBarrierBuilder::new(),
        }
    }
}

/// Allows building a submission.
// TODO: can be optimized by storing all the semaphores in a single vec and all command buffers
// in a single vec
pub struct SubmitBuilder {
    semaphores_storage: SmallVec<[vk::Semaphore; 16]>,
    dest_stages_storage: SmallVec<[vk::PipelineStageFlags; 8]>,
    command_buffers_storage: SmallVec<[vk::CommandBuffer; 4]>,
    submits: SmallVec<[SubmitBuilderSubmit; 2]>,
    keep_alive_semaphores: SmallVec<[Arc<Semaphore>; 8]>,
}

#[derive(Default)]
struct SubmitBuilderSubmit {
    batches: SmallVec<[vk::SubmitInfo; 4]>,
    fence: Option<Arc<Fence>>,
}

impl SubmitBuilder {
    /// Builds a new empty `SubmitBuilder`.
    #[inline]
    pub fn new() -> SubmitBuilder {
        SubmitBuilder {
            semaphores_storage: SmallVec::new(),
            dest_stages_storage: SmallVec::new(),
            command_buffers_storage: SmallVec::new(),
            submits: SmallVec::new(),
            keep_alive_semaphores: SmallVec::new(),
        }
    }

    /// Adds a fence to signal.
    ///
    /// > **Note**: Due to the way the Vulkan API is designed, you are strongly encouraged to use
    /// > only one fence and signal at the very end of the submission.
    #[inline]
    pub fn add_fence(mut self, fence: Arc<Fence>) -> SubmitBuilder {
        if self.submits.last().map(|b| b.fence.is_some()).unwrap_or(true) {
            self.submits.push(Default::default());
        }

        {
            let mut last = self.submits.last_mut().unwrap();
            debug_assert!(last.fence.is_none());
            last.fence = Some(fence);
        }

        self
    }

    /// Adds a semaphore to wait upon.
    #[inline]
    pub fn add_wait_semaphore(mut self, semaphore: Arc<Semaphore>, stages: PipelineStages)
                              -> SubmitBuilder
    {
        if self.submits.last().map(|b| b.fence.is_some()).unwrap_or(true) {
            self.submits.push(Default::default());
        }

        {
            let mut submit = self.submits.last_mut().unwrap();
            if submit.batches.last().map(|b| b.signalSemaphoreCount != 0 ||
                                             b.commandBufferCount != 0)
                                    .unwrap_or(true)
            {
                submit.batches.push(SubmitBuilder::empty_vk_submit_info());
            }

            submit.batches.last_mut().unwrap().waitSemaphoreCount += 1;
            self.dest_stages_storage.push(stages.into());
            self.semaphores_storage.push(semaphore.internal_object());
            self.keep_alive_semaphores.push(semaphore);
        }

        self
    }

    // TODO: API shouldn't expose vk ; instead take a `&'a UnsafeCommandBuffer` where `'a` is a
    //       lifetime on `Submit::append_submission`.
    #[inline]
    pub fn add_command_buffer(mut self, command_buffer: vk::CommandBuffer) -> SubmitBuilder {
        if self.submits.last().map(|b| b.fence.is_some()).unwrap_or(true) {
            self.submits.push(Default::default());
        }

        {
            let mut submit = self.submits.last_mut().unwrap();
            if submit.batches.last().map(|b| b.signalSemaphoreCount != 0).unwrap_or(true) {
                submit.batches.push(SubmitBuilder::empty_vk_submit_info());
            }

            self.command_buffers_storage.push(command_buffer);
            submit.batches.last_mut().unwrap().commandBufferCount += 1;
        }

        self
    }

    /// Adds a semaphore to signal after all the previous elements have completed.
    #[inline]
    pub fn add_signal_semaphore(mut self, semaphore: Arc<Semaphore>) -> SubmitBuilder {
        if self.submits.last().map(|b| b.fence.is_some()).unwrap_or(true) {
            self.submits.push(Default::default());
        }

        {
            let mut submit = self.submits.last_mut().unwrap();
            if submit.batches.is_empty() {
                submit.batches.push(SubmitBuilder::empty_vk_submit_info());
            }

            submit.batches.last_mut().unwrap().signalSemaphoreCount += 1;
            self.semaphores_storage.push(semaphore.internal_object());
            self.keep_alive_semaphores.push(semaphore);
        }

        self
    }

    #[inline]
    fn empty_vk_submit_info() -> vk::SubmitInfo {
        vk::SubmitInfo {
            sType: vk::STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: ptr::null(),
            waitSemaphoreCount: 0,
            pWaitSemaphores: ptr::null(),
            pWaitDstStageMask: ptr::null(),
            commandBufferCount: 0,
            pCommandBuffers: ptr::null(),
            signalSemaphoreCount: 0,
            pSignalSemaphores: ptr::null(),
        }
    }
}

pub fn submit<S>(submit: S, queue: &Arc<Queue>) -> Submission<S>
    where S: Submit
{
    unsafe {
        let mut builder = submit.append_submission(SubmitBuilder::new(), queue);

        let last_fence = if let Some(last) = builder.submits.last_mut() {
            if last.fence.is_none() {
                last.fence = Some(Fence::new(submit.device().clone()));
            }

            last.fence.as_ref().unwrap().clone()
            
        } else {
            Fence::new(submit.device().clone())     // TODO: meh
        };

        {
            let vk = queue.device().pointers();
            let queue = queue.internal_object_guard();

            let mut next_semaphore = 0;
            let mut next_wait_stage = 0;
            let mut next_command_buffer = 0;

            for submit in builder.submits.iter_mut() {
                for batch in submit.batches.iter_mut() {
                    batch.pWaitSemaphores = builder.semaphores_storage.as_ptr().offset(next_semaphore);
                    batch.pWaitDstStageMask = builder.dest_stages_storage.as_ptr().offset(next_wait_stage);
                    next_semaphore += batch.waitSemaphoreCount as isize;
                    next_wait_stage += batch.waitSemaphoreCount as isize;
                    batch.pCommandBuffers = builder.command_buffers_storage.as_ptr().offset(next_command_buffer);
                    next_command_buffer += batch.commandBufferCount as isize;
                    batch.pSignalSemaphores = builder.semaphores_storage.as_ptr().offset(next_semaphore);
                    next_semaphore += batch.signalSemaphoreCount as isize;
                }

                let fence = submit.fence.as_ref().map(|f| f.internal_object()).unwrap_or(0);
                check_errors(vk.QueueSubmit(*queue, submit.batches.len() as u32,
                                            submit.batches.as_ptr(), fence)).unwrap();        // TODO: handle errors (trickier than it looks)

            }

            debug_assert_eq!(next_semaphore as usize, builder.semaphores_storage.len());
            debug_assert_eq!(next_wait_stage as usize, builder.dest_stages_storage.len());
            debug_assert_eq!(next_command_buffer as usize, builder.command_buffers_storage.len());
        }

        Submission {
            queue: queue.clone(),
            fence: last_fence,
            keep_alive_semaphores: builder.keep_alive_semaphores,
            submit: submit,
        }
    }
}

/// Returned when you submit one or multiple command buffers.
///
/// This object holds the resources that are used by the GPU and that must be kept alive for at
/// least as long as the GPU is executing the submission. Therefore destroying a `Submission`
/// object will block until the GPU is finished executing.
///
/// Whenever you submit a command buffer, you are encouraged to store the returned `Submission`
/// in a long-living container such as a `Vec`. From time to time, you can clean the obsolete
/// objects by checking whether `destroying_would_block()` returns false. For example, if you use
/// a `Vec` you can do `vec.retain(|s| s.destroying_would_block())`.
// TODO: docs
// # Leak safety
//
// The `Submission` object can hold borrows of command buffers. In order for it to be safe to leak
// a `Submission`, the borrowed object themselves must be protected by a fence.
#[must_use]
pub struct Submission<S = Box<Submit>> {
    fence: Arc<Fence>,      // TODO: make optional
    queue: Arc<Queue>,
    keep_alive_semaphores: SmallVec<[Arc<Semaphore>; 8]>,
    submit: S,
}

impl<S> fmt::Debug for Submission<S> {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        // TODO: better impl?
        write!(fmt, "<Vulkan submission>")
    }
}

impl<S> Submission<S> {
    /// Returns `true` if destroying this `Submission` object would block the CPU for some time.
    #[inline]
    pub fn destroying_would_block(&self) -> bool {
        !self.finished()
    }

    /// Returns `true` if the GPU has finished executing this submission.
    #[inline]
    pub fn finished(&self) -> bool {
        self.fence.ready().unwrap_or(false)     // TODO: what to do in case of error?   
    }

    /// Waits until the submission has finished.
    #[inline]
    pub fn wait(&self, timeout: Duration) -> Result<(), FenceWaitError> {
        self.fence.wait(timeout)
    }

    /// Returns the queue the submission was submitted to.
    #[inline]
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

impl<S> Drop for Submission<S> {
    fn drop(&mut self) {
        self.fence.wait(Duration::from_secs(10)).unwrap();      // TODO: handle some errors
    }
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::iter::Empty;
    use std::sync::Arc;

    use command_buffer::pool::StandardCommandPool;
    use command_buffer::submit::CommandBuffer;
    use command_buffer::submit::SubmitInfo;
    use command_buffer::sys::Kind;
    use command_buffer::sys::Flags;
    use command_buffer::sys::PipelineBarrierBuilder;
    use command_buffer::sys::UnsafeCommandBuffer;
    use command_buffer::sys::UnsafeCommandBufferBuilder;
    use device::Device;
    use device::Queue;
    use framebuffer::framebuffer::EmptyAttachmentsList;
    use framebuffer::EmptySinglePassRenderPass;
    use framebuffer::StdFramebuffer;
    use sync::Fence;
    use sync::PipelineStages;
    use sync::Semaphore;

    #[test]
    fn basic_submit() {
        struct Basic { inner: UnsafeCommandBuffer<Arc<StandardCommandPool>> }
        unsafe impl CommandBuffer for Basic {
            type Pool = Arc<StandardCommandPool>;

            fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> { &self.inner }

            unsafe fn on_submit<F>(&self, _: &Arc<Queue>, fence: F)
                                   -> SubmitInfo<Self::SemaphoresWaitIterator,
                                                 Self::SemaphoresSignalIterator>
                where F: FnOnce() -> Arc<Fence>
            {
                SubmitInfo::empty()
            }
        }

        let (device, queue) = gfx_dev_and_queue!();

        let pool = Device::standard_command_pool(&device, queue.family());
        let kind = Kind::Primary::<EmptySinglePassRenderPass, StdFramebuffer<EmptySinglePassRenderPass, EmptyAttachmentsList>>;

        let cb = UnsafeCommandBufferBuilder::new(pool, kind, Flags::OneTimeSubmit).unwrap();
        let cb = Basic { inner: cb.build().unwrap() };

        let _s = cb.submit(&queue);
    }
}
