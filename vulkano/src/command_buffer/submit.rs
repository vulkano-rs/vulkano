// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ptr;
use std::sync::Arc;
use std::time::Duration;
use smallvec::SmallVec;

use command_buffer::pool::CommandPool;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use device::Queue;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

use check_errors;
use vk;
use VulkanObject;
use VulkanPointers;
use SynchronizedVulkanObject;

/// Trait for objects that represent commands ready to be executed by the GPU.
pub unsafe trait CommandBuffer {
    /// Submits the command buffer.
    ///
    /// Note that since submitting has a fixed overhead, you should try, if possible, to submit
    /// multiple command buffers at once instead.
    ///
    /// This is a simple shortcut for creating a `Submit` object.
    #[inline]
    fn submit(self, queue: &Arc<Queue>) -> Submission where Self: Sized {
        Submit::new().add(self).submit(queue)
    }

    /// Type of the pool that was used to allocate the command buffer.
    type Pool: CommandPool;

    /// Iterator that returns the list of semaphores to wait upon before the command buffer is
    /// submitted.
    type SemaphoresWaitIterator: Iterator<Item = (Arc<Semaphore>, PipelineStages)>;

    /// Iterator that returns the list of semaphores to signal after the command buffer has
    /// finished execution.
    type SemaphoresSignalIterator: Iterator<Item = Arc<Semaphore>>;

    /// Returns the inner object.
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool>;

    /// Called slightly before the command buffer is submitted. Signals the command buffers that it
    /// is going to be submitted on the given queue. The function must return the list of
    /// semaphores to wait upon and transitions to perform.
    ///
    /// The `fence` parameter is a closure that can be used to pull a fence if required. If a fence
    /// is pulled, it is guaranteed that it will be signaled after the command buffer ends.
    ///
    /// # Safety for the caller
    ///
    /// This function must only be called if there's actually a submission that follows. If a
    /// fence is pulled, then it must eventually be signaled. All the semaphores that are waited
    /// upon must become unsignaled, and all the semaphores that are supposed to be signaled must
    /// become signaled.
    ///
    /// This function is supposed to be called only by vulkano's internals. It is recommended
    /// that you never call it.
    ///
    /// # Safety for the implementation
    ///
    /// The implementation must ensure that the command buffer doesn't get destroyed before the
    /// fence is signaled, or before a fence of a later submission to the same queue is signaled.
    ///
    unsafe fn on_submit<F>(&self, queue_family: u32, queue_within_family: u32, fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnOnce() -> Arc<Fence>;
}

/// Information about how the submitting function should synchronize the submission.
pub struct SubmitInfo<Swi, Ssi> {
    /// List of semaphores to wait upon before the command buffer starts execution.
    pub semaphores_wait: Swi,
    /// List of semaphores to signal after the command buffer has finished.
    pub semaphores_signal: Ssi,
    /// Pipeline barrier to execute on the queue and immediately before the command buffer.
    /// Ignored if empty.
    pub pre_pipeline_barrier: PipelineBarrierBuilder,
    /// Pipeline barrier to execute on the queue and immediately after the command buffer.
    /// Ignored if empty.
    pub post_pipeline_barrier: PipelineBarrierBuilder,
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
pub struct Submission {
    fence: Arc<Fence>,      // TODO: make optional
    keep_alive: SmallVec<[Arc<KeepAlive>; 4]>,
}

impl Submission {
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
}

impl Drop for Submission {
    fn drop(&mut self) {
        self.fence.wait(Duration::from_secs(10)).unwrap();      // TODO: handle some errors
    }
}

trait KeepAlive {}
impl<T> KeepAlive for T {}

#[derive(Debug, Copy, Clone)]
pub struct Submit<L> {
    list: L,
}

impl Submit<()> {
    /// Builds an empty submission list.
    #[inline]
    pub fn new() -> Submit<()> {
        Submit { list: () }
    }
}

impl<L> Submit<L> where L: SubmitList {
    /// Adds a command buffer to submit to the list.
    ///
    /// In the Vulkan API, a submission is divided into batches that each contain one or more
    /// command buffers. Vulkano will automatically determine which command buffers can be grouped
    /// into the same batch.
    #[inline]
    pub fn add<C>(self, command_buffer: C) -> Submit<(C, L)> where C: CommandBuffer {
        Submit { list: (command_buffer, self.list) }
    }

    /// Submits the list of command buffers.
    pub fn submit(self, queue: &Arc<Queue>) -> Submission {
        let SubmitListOpaque { fence, wait_semaphores, wait_stages, command_buffers,
                               signal_semaphores, mut submits, keep_alive }
                             = self.list.infos(queue.family().id(), queue.id_within_family());

        // TODO: for now we always create a Fence in order to put it in the submission
        let fence = fence.unwrap_or_else(|| Fence::new(queue.device().clone()));

        // Filling the pointers inside `submits`.
        unsafe {
            debug_assert_eq!(wait_semaphores.len(), wait_stages.len());

            let mut next_wait = 0;
            let mut next_cb = 0;
            let mut next_signal = 0;

            for submit in submits.iter_mut() {
                debug_assert!(submit.waitSemaphoreCount as usize + next_wait as usize <=
                              wait_semaphores.len());
                debug_assert!(submit.commandBufferCount as usize + next_cb as usize <=
                              command_buffers.len());
                debug_assert!(submit.signalSemaphoreCount as usize + next_signal as usize <=
                              signal_semaphores.len());

                submit.pWaitSemaphores = wait_semaphores.as_ptr().offset(next_wait);
                submit.pWaitDstStageMask = wait_stages.as_ptr().offset(next_wait);
                submit.pCommandBuffers = command_buffers.as_ptr().offset(next_cb);
                submit.pSignalSemaphores = signal_semaphores.as_ptr().offset(next_signal);

                next_wait += submit.waitSemaphoreCount as isize;
                next_cb += submit.commandBufferCount as isize;
                next_signal += submit.signalSemaphoreCount as isize;
            }

            debug_assert_eq!(next_wait as usize, wait_semaphores.len());
            debug_assert_eq!(next_wait as usize, wait_stages.len());
            debug_assert_eq!(next_cb as usize, command_buffers.len());
            debug_assert_eq!(next_signal as usize, signal_semaphores.len());
        }

        unsafe {
            let vk = queue.device().pointers();
            let queue = queue.internal_object_guard();
            //let fence = fence.as_ref().map(|f| f.internal_object()).unwrap_or(0);
            let fence = fence.internal_object();
            check_errors(vk.QueueSubmit(*queue, submits.len() as u32, submits.as_ptr(),
                                        fence)).unwrap();        // TODO: handle errors (trickier than it looks)
        }

        Submission {
            keep_alive: keep_alive,
            fence: fence,
        }
    }
}

/* TODO: All that stuff below is undocumented */

pub struct SubmitListOpaque {
    fence: Option<Arc<Fence>>,
    wait_semaphores: SmallVec<[vk::Semaphore; 16]>,
    wait_stages: SmallVec<[vk::PipelineStageFlags; 16]>,
    command_buffers: SmallVec<[vk::CommandBuffer; 16]>,
    signal_semaphores: SmallVec<[vk::Semaphore; 16]>,
    submits: SmallVec<[vk::SubmitInfo; 8]>,
    keep_alive: SmallVec<[Arc<KeepAlive>; 4]>,
}

pub unsafe trait SubmitList {
    fn infos(self, queue_family: u32, queue_within_family: u32) -> SubmitListOpaque;
}

unsafe impl SubmitList for () {
    fn infos(self, _: u32, _: u32) -> SubmitListOpaque {
        SubmitListOpaque {
            fence: None,
            wait_semaphores: SmallVec::new(),
            wait_stages: SmallVec::new(),
            command_buffers: SmallVec::new(),
            signal_semaphores: SmallVec::new(),
            submits: SmallVec::new(),
            keep_alive: SmallVec::new(),
        }
    }
}

unsafe impl<C, R> SubmitList for (C, R) where C: CommandBuffer, R: SubmitList {
    fn infos(self, queue_family: u32, queue_within_family: u32) -> SubmitListOpaque {
        // TODO: attempt to group multiple submits into one when possible

        let (current, rest) = self;

        let mut infos = rest.infos(queue_family, queue_within_family);
        let device = current.inner().device().clone();
        let current_infos = unsafe { current.on_submit(queue_family, queue_within_family, || {
            if let Some(fence) = infos.fence.as_ref() {
                return fence.clone();
            }

            let new_fence = Fence::new(device);
            infos.fence = Some(new_fence.clone());
            new_fence
        })};

        // TODO: not implemented ; where to store the created command buffers?
        assert!(current_infos.pre_pipeline_barrier.is_empty());
        assert!(current_infos.post_pipeline_barrier.is_empty());

        let mut new_submit = vk::SubmitInfo {
            sType: vk::STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: ptr::null(),
            waitSemaphoreCount: 0,
            pWaitSemaphores: ptr::null(),
            pWaitDstStageMask: ptr::null(),
            commandBufferCount: 1,
            pCommandBuffers: ptr::null(),
            signalSemaphoreCount: 0,
            pSignalSemaphores: ptr::null(),
        };

        for (semaphore, stage) in current_infos.semaphores_wait {
            infos.wait_semaphores.push(semaphore.internal_object());
            infos.wait_stages.push(stage.into());
            infos.keep_alive.push(semaphore);
            new_submit.waitSemaphoreCount += 1;
        }

        for semaphore in current_infos.semaphores_signal {
            infos.signal_semaphores.push(semaphore.internal_object());
            infos.keep_alive.push(semaphore);
            new_submit.signalSemaphoreCount += 1;
        }

        infos.command_buffers.push(current.inner().internal_object());

        infos.submits.push(new_submit);

        infos
    }
}
