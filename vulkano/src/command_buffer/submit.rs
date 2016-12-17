// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;
use std::time::Duration;
use smallvec::SmallVec;

use command_buffer::cb::UnsyncedCommandBuffer;
use command_buffer::pool::CommandPool;
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
// TODO: is Box<Error> appropriate? maybe something else?
pub unsafe trait Submit {
    /// Submits the object to the queue.
    ///
    /// Since submitting has a fixed overhead, you should try, if possible, to submit multiple
    /// command buffers at once instead. To do so, you can use the `chain` method.
    ///
    /// `s.submit(queue)` is a shortcut for `s.submit_precise(queue).boxed()`.
    // TODO: add example
    #[inline]
    fn submit(self, queue: &Arc<Queue>) -> Result<Submission, Box<Error>>
        where Self: Sized + 'static
    {
        Ok(try!(self.submit_precise(queue)).boxed())
    }

    /// Submits the object to the queue.
    ///
    /// Since submitting has a fixed overhead, you should try, if possible, to submit multiple
    /// command buffers at once instead. To do so, you can use the `chain` method.
    ///
    /// Contrary to `submit`, this method preserves strong typing in the submission. This means
    /// that it has a lower overhead but it is less convenient to store in a container.
    // TODO: add example
    #[inline]
    fn submit_precise(self, queue: &Arc<Queue>) -> Result<Submission<Self>, Box<Error>>
        where Self: Sized + 'static
    {
        submit(self, queue)
    }

    /// Consumes this object and another one to return a `SubmitChain` object that will submit both
    /// at once.
    ///
    /// `self` will be executed first, and then `other` afterwards.
    ///
    /// # Panic
    ///
    /// - Panics if the two objects don't belong to the same `Device`.
    ///
    // TODO: add test for panic
    // TODO: add example
    #[inline]
    fn chain<S>(self, other: S) -> SubmitChain<Self, S> where Self: Sized, S: Submit {
        assert_eq!(self.device().internal_object(),
                   other.device().internal_object());
        SubmitChain { first: self, second: other }
    }

    /// Returns the device this object belongs to.
    fn device(&self) -> &Arc<Device>;

    /// Called slightly before the object is submitted. The function must modify an existing
    /// `SubmitBuilder` object to append the list of things to submit to it.
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
    /// TODO: To write.
    unsafe fn append_submission<'a>(&'a self, base: SubmitBuilder<'a>, queue: &Arc<Queue>)
                                    -> Result<SubmitBuilder<'a>, Box<Error>>;
}

unsafe impl<S: ?Sized> Submit for Box<S> where S: Submit {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        (**self).device()
    }

    #[inline]
    unsafe fn append_submission<'a>(&'a self, base: SubmitBuilder<'a>, queue: &Arc<Queue>)
                                    -> Result<SubmitBuilder<'a>, Box<Error>>
    {
        (**self).append_submission(base, queue)
    }
}

unsafe impl<S: ?Sized> Submit for Arc<S> where S: Submit {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        (**self).device()
    }

    #[inline]
    unsafe fn append_submission<'a>(&'a self, base: SubmitBuilder<'a>, queue: &Arc<Queue>)
                                    -> Result<SubmitBuilder<'a>, Box<Error>>
    {
        (**self).append_submission(base, queue)
    }
}

/// Allows building a submission.
///
/// This object contains a list of operations that the GPU should perform in order. You can add new
/// operations to the list with the various `add_*` methods.
///
/// > **Note**: Command buffers submitted one after another are not executed in order. Instead they
/// > are only guarateed to *start* in the order they were added. The object that implements
/// > `Submit` should be aware of that fact and add appropriate pipeline barriers to the command
/// > buffers.
///
/// # Safety
///
/// While it is safe to build a `SubmitBuilder` from scratch, the only way to actually submit
/// something is through the `Submit` trait which is unsafe to implement.
// TODO: can be optimized by storing all the semaphores in a single vec and all command buffers
// in a single vec
// TODO: add sparse bindings and swapchain presents
pub struct SubmitBuilder<'a> {
    semaphores_storage: SmallVec<[vk::Semaphore; 16]>,
    dest_stages_storage: SmallVec<[vk::PipelineStageFlags; 8]>,
    command_buffers_storage: SmallVec<[vk::CommandBuffer; 4]>,
    submits: SmallVec<[SubmitBuilderSubmit; 2]>,
    keep_alive_semaphores: SmallVec<[Arc<Semaphore>; 8]>,
    keep_alive_fences: SmallVec<[Arc<Fence>; 2]>,
    marker: PhantomData<&'a ()>,
}

#[derive(Default)]
struct SubmitBuilderSubmit {
    batches: SmallVec<[vk::SubmitInfo; 4]>,
    fence: Option<Arc<Fence>>,
}

impl<'a> SubmitBuilder<'a> {
    /// Builds a new empty `SubmitBuilder`.
    #[inline]
    pub fn new() -> SubmitBuilder<'a> {
        SubmitBuilder {
            semaphores_storage: SmallVec::new(),
            dest_stages_storage: SmallVec::new(),
            command_buffers_storage: SmallVec::new(),
            submits: SmallVec::new(),
            keep_alive_semaphores: SmallVec::new(),
            keep_alive_fences: SmallVec::new(),
            marker: PhantomData,
        }
    }

    /// Adds an operation that signals a fence.
    ///
    /// > **Note**: Due to the way the Vulkan API is designed, you are strongly encouraged to use
    /// > only one fence and signal at the very end of the submission.
    ///
    /// The fence is signalled after all previous operations of the `SubmitBuilder` are finished.
    #[inline]
    pub fn add_fence_signal(mut self, fence: Arc<Fence>) -> SubmitBuilder<'a> {
        if self.submits.last().map(|b| b.fence.is_some()).unwrap_or(true) {
            self.submits.push(Default::default());
        }

        {
            let mut last = self.submits.last_mut().unwrap();
            debug_assert!(last.fence.is_none());
            self.keep_alive_fences.push(fence.clone());
            last.fence = Some(fence);
        }

        self
    }

    /// Adds an operation that waits on a semaphore.
    ///
    /// Only the given `stages` of the command buffers added afterwards will wait upon
    /// the semaphore. Other stages not included in `stages` can execute before waiting.
    ///
    /// The semaphore must be signalled by a previous submission.
    #[inline]
    pub fn add_wait_semaphore(mut self, semaphore: Arc<Semaphore>, stages: PipelineStages)
                              -> SubmitBuilder<'a>
    {
        // TODO: check device of the semaphore with a debug_assert
        // TODO: if stages contains tessellation or geometry stages, make sure the corresponding
        // feature is active with a debug_assert
        debug_assert!({ let f: vk::PipelineStageFlagBits = stages.into(); f != 0 });

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

    /// Adds an operation that executes a command buffer.
    ///
    /// > **Note**: Command buffers submitted one after another are not executed in order. Instead
    /// > they are only guarateed to *start* in the order they were added. The object that
    /// > implements `Submit` should be aware of that fact and add appropriate pipeline barriers
    /// > to the command buffers.
    ///
    /// Thanks to the lifetime requirement, the command buffer must outlive the `Submit` object
    /// that builds this `SubmitBuilder`. Consequently keeping the `Submit` object alive is enough
    /// to guarantee that the command buffer is kept alive as well.
    #[inline]
    pub fn add_command_buffer<L, P>(self, command_buffer: &'a UnsyncedCommandBuffer<L, P>)
                                    -> SubmitBuilder<'a>
        where P: CommandPool
    {
        self.add_command_buffer_raw(command_buffer.internal_object())
    }

    // TODO: remove in favor of `add_command_buffer`?
    #[inline]
    pub fn add_command_buffer_raw(mut self, command_buffer: vk::CommandBuffer)
                                  -> SubmitBuilder<'a>
    {
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

    /// Adds an operation that signals a semaphore.
    ///
    /// The semaphore is signalled after all previous operations of the `SubmitBuilder` are
    /// finished.
    ///
    /// The semaphore must be unsignalled.
    #[inline]
    pub fn add_signal_semaphore(mut self, semaphore: Arc<Semaphore>) -> SubmitBuilder<'a> {
        // TODO: check device of the semaphore with a debug_assert

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

// Implementation for `Submit::submit`.
fn submit<S>(submit: S, queue: &Arc<Queue>) -> Result<Submission<S>, Box<Error>>
    where S: Submit + 'static
{
    let last_fence;
    let keep_alive_semaphores;
    let keep_alive_fences;

    unsafe {
        let mut builder = try!(submit.append_submission(SubmitBuilder::new(), queue));
        keep_alive_semaphores = builder.keep_alive_semaphores;
        keep_alive_fences = builder.keep_alive_fences;

        last_fence = if let Some(last) = builder.submits.last_mut() {
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
                    batch.pWaitSemaphores = builder.semaphores_storage
                                                   .as_ptr().offset(next_semaphore);
                    batch.pWaitDstStageMask = builder.dest_stages_storage
                                                     .as_ptr().offset(next_wait_stage);
                    next_semaphore += batch.waitSemaphoreCount as isize;
                    next_wait_stage += batch.waitSemaphoreCount as isize;
                    batch.pCommandBuffers = builder.command_buffers_storage
                                                   .as_ptr().offset(next_command_buffer);
                    next_command_buffer += batch.commandBufferCount as isize;
                    batch.pSignalSemaphores = builder.semaphores_storage
                                                     .as_ptr().offset(next_semaphore);
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
    }

    Ok(Submission {
        queue: queue.clone(),
        fence: FenceWithWaiting(last_fence),
        keep_alive_semaphores: keep_alive_semaphores,
        keep_alive_fences: keep_alive_fences,
        submit: submit,
    })
}

/// Chain of two `Submit` objects. See `Submit::chain`.
pub struct SubmitChain<A, B> {
    first: A,
    second: B,
}

unsafe impl<A, B> Submit for SubmitChain<A, B> where A: Submit, B: Submit {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        debug_assert_eq!(self.first.device().internal_object(),
                         self.second.device().internal_object());
        
        self.first.device()
    }

    #[inline]
    unsafe fn append_submission<'a>(&'a self, base: SubmitBuilder<'a>, queue: &Arc<Queue>)
                                    -> Result<SubmitBuilder<'a>, Box<Error>>
    {
        let builder = try!(self.first.append_submission(base, queue));
        self.second.append_submission(builder, queue)
    }
}

/// Returned when you submit something to a queue.
///
/// This object holds the resources that are used by the GPU and that must be kept alive for at
/// least as long as the GPU is executing the submission. Therefore destroying a `Submission`
/// object will block until the GPU is finished executing.
///
/// Whenever you submit a command buffer, you are encouraged to store the returned `Submission`
/// in a long-living container such as a `Vec`. From time to time, you can clean the obsolete
/// objects by checking whether `destroying_would_block()` returns false. For example, if you use
/// a `Vec` you can do `vec.retain(|s| s.destroying_would_block())`.
///
/// # Leak safety
///
/// One of the roles of the `Submission` object is to keep alive the objects used by the GPU during
/// the submission. However if the user calls `std::mem::forget` on the `Submission`, all borrows
/// are immediately free'd. This is known as *the leakpocalypse*.
///
/// In order to avoid this problem, only `'static` objects can be put in a `Submission`.
// TODO: ^ decide whether we allow to add an unsafe non-static constructor
#[must_use]
pub struct Submission<S: 'static = Box<Submit>> {
    fence: FenceWithWaiting,      // TODO: make optional
    queue: Arc<Queue>,
    keep_alive_semaphores: SmallVec<[Arc<Semaphore>; 8]>,
    keep_alive_fences: SmallVec<[Arc<Fence>; 2]>,
    submit: S,
}

struct FenceWithWaiting(Arc<Fence>);
impl Drop for FenceWithWaiting {
    fn drop(&mut self) {
        self.0.wait(Duration::from_secs(10)).unwrap();      // TODO: handle some errors
    }
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
        self.fence.0.ready().unwrap_or(false)     // TODO: what to do in case of error?   
    }

    /// Waits until the submission has finished.
    #[inline]
    pub fn wait(&self, timeout: Duration) -> Result<(), FenceWaitError> {
        self.fence.0.wait(timeout)
    }

    /// Returns the queue the submission was submitted to.
    #[inline]
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

impl<S> Submission<S> where S: Submit + 'static {
    /// Turns this submission into a boxed submission.
    pub fn boxed(self) -> Submission {
        Submission {
            fence: self.fence,
            queue: self.queue,
            keep_alive_semaphores: self.keep_alive_semaphores,
            keep_alive_fences: self.keep_alive_fences,
            submit: Box::new(self.submit) as Box<_>,
        }
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
