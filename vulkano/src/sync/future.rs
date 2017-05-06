// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error::Error;
use std::mem;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use buffer::BufferAccess;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferExecFuture;
use command_buffer::submit::SubmitAnyBuilder;
use command_buffer::submit::SubmitCommandBufferBuilder;
use command_buffer::submit::SubmitSemaphoresWaitBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use swapchain::Swapchain;
use swapchain::PresentFuture;
use sync::AccessFlagBits;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

use VulkanObject;

/// Represents an event that will happen on the GPU in the future.
// TODO: consider switching all methods to take `&mut self` for optimization purposes
pub unsafe trait GpuFuture: DeviceOwned {
    /// If possible, checks whether the submission has finished. If so, gives up ownership of the
    /// resources used by these submissions.
    ///
    /// It is highly recommended to call `cleanup_finished` from time to time. Doing so will
    /// prevent memory usage from increasing over time, and will also destroy the locks on 
    /// resources used by the GPU.
    fn cleanup_finished(&mut self);

    /// Builds a submission that, if submitted, makes sure that the event represented by this
    /// `GpuFuture` will happen, and possibly contains extra elements (eg. a semaphore wait or an
    /// event wait) that makes the dependency with subsequent operations work.
    ///
    /// It is the responsibility of the caller to ensure that the submission is going to be
    /// submitted only once. However keep in mind that this function can perfectly be called
    /// multiple times (as long as the returned object is only submitted once).
    ///
    /// It is however the responsibility of the implementation to not return the same submission
    /// from multiple different future objects. For example if you implement `GpuFuture` on
    /// `Arc<Foo>` then `build_submission()` must always return `SubmitAnyBuilder::Empty`,
    /// otherwise it would be possible for the user to clone the `Arc` and make the same
    /// submission be submitted multiple times.
    ///
    /// Once the caller has submitted the submission and has determined that the GPU has finished
    /// executing it, it should call `signal_finished`. Failure to do so will incur a large runtime
    /// overhead, as the future will have to block to make sure that it is finished.
    // TODO: better error type
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Box<Error>>;

    /// Flushes the future and submits to the GPU the actions that will permit this future to
    /// occur.
    ///
    /// The implementation must remember that it was flushed. If the function is called multiple
    /// times, only the first time must result in a flush.
    // TODO: better error type
    fn flush(&self) -> Result<(), Box<Error>>;

    /// Sets the future to its "complete" state, meaning that it can safely be destroyed.
    ///
    /// This must only be done if you called `build_submission()`, submitted the returned
    /// submission, and determined that it was finished.
    unsafe fn signal_finished(&self);

    /// Returns the queue that triggers the event. Returns `None` if unknown or irrelevant.
    ///
    /// If this function returns `None` and `queue_change_allowed` returns `false`, then a panic
    /// is likely to occur if you use this future. This is only a problem if you implement
    /// the `GpuFuture` trait yourself for a type outside of vulkano.
    fn queue(&self) -> Option<&Arc<Queue>>;

    /// Returns `true` if elements submitted after this future can be submitted to a different
    /// queue than the other returned by `queue()`.
    fn queue_change_allowed(&self) -> bool;

    /// Checks whether submitting something after this future grants access (exclusive or shared,
    /// depending on the parameter) to the given buffer on the given queue.
    ///
    /// If the access is granted, returns the pipeline stage and access flags of the latest usage
    /// of this resource, or `None` if irrelevant.
    ///
    /// > **Note**: Returning `Ok` means "access granted", while returning `Err` means
    /// > "don't know". Therefore returning `Err` is never unsafe.
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>;

    /// Checks whether submitting something after this future grants access (exclusive or shared,
    /// depending on the parameter) to the given image on the given queue.
    ///
    /// If the access is granted, returns the pipeline stage and access flags of the latest usage
    /// of this resource, or `None` if irrelevant.
    ///
    /// > **Note**: Returning `Ok` means "access granted", while returning `Err` means
    /// > "don't know". Therefore returning `Err` is never unsafe.
    ///
    /// > **Note**: Keep in mind that changing the layout of an image also requires exclusive
    /// > access.
    fn check_image_access(&self, image: &ImageAccess, exclusive: bool, queue: &Queue)
                         -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>;

    /// Joins this future with another one, representing the moment when both events have happened.
    // TODO: handle errors
    fn join<F>(self, other: F) -> JoinFuture<Self, F>
        where Self: Sized, F: GpuFuture
    {
        assert_eq!(self.device().internal_object(), other.device().internal_object());

        if !self.queue_change_allowed() && !other.queue_change_allowed() {
            assert!(self.queue().unwrap().is_same(other.queue().unwrap()));
        }

        JoinFuture {
            first: self,
            second: other,
        }
    }

    /// Executes a command buffer after this future.
    ///
    /// > **Note**: This is just a shortcut function. The actual implementation is in the
    /// > `CommandBuffer` trait.
    #[inline]
    fn then_execute<Cb>(self, queue: Arc<Queue>, command_buffer: Cb)
                        -> CommandBufferExecFuture<Self, Cb>
        where Self: Sized, Cb: CommandBuffer + 'static
    {
        command_buffer.execute_after(self, queue)
    }

    /// Executes a command buffer after this future, on the same queue as the future.
    ///
    /// > **Note**: This is just a shortcut function. The actual implementation is in the
    /// > `CommandBuffer` trait.
    #[inline]
    fn then_execute_same_queue<Cb>(self, command_buffer: Cb) -> CommandBufferExecFuture<Self, Cb>
        where Self: Sized, Cb: CommandBuffer + 'static
    {
        let queue = self.queue().unwrap().clone();
        command_buffer.execute_after(self, queue)
    }

    /// Signals a semaphore after this future. Returns another future that represents the signal.
    ///
    /// Call this function when you want to execute some operations on a queue and want to see the
    /// result on another queue.
    #[inline]
    fn then_signal_semaphore(self) -> SemaphoreSignalFuture<Self> where Self: Sized {
        let device = self.device().clone();

        assert!(self.queue().is_some());        // TODO: document

        SemaphoreSignalFuture {
            previous: self,
            semaphore: Semaphore::new(device).unwrap(),
            wait_submitted: Mutex::new(false),
            finished: AtomicBool::new(false),
        }
    }

    /// Signals a semaphore after this future and flushes it. Returns another future that
    /// represents the moment when the semaphore is signalled.
    ///
    /// This is a just a shortcut for `then_signal_semaphore()` followed with `flush()`.
    ///
    /// When you want to execute some operations A on a queue and some operations B on another
    /// queue that need to see the results of A, it can be a good idea to submit A as soon as
    /// possible while you're preparing B.
    ///
    /// If you ran A and B on the same queue, you would have to decide between submitting A then
    /// B, or A and B simultaneously. Both approaches have their trade-offs. But if A and B are
    /// on two different queues, then you would need two submits anyway and it is always
    /// advantageous to submit A as soon as possible.
    #[inline]
    fn then_signal_semaphore_and_flush(self) -> Result<SemaphoreSignalFuture<Self>, Box<Error>>
        where Self: Sized
    {
        let f = self.then_signal_semaphore();
        f.flush()?;
        Ok(f)
    }

    /// Signals a fence after this future. Returns another future that represents the signal.
    ///
    /// > **Note**: More often than not you want to immediately flush the future after calling this
    /// > function. If so, consider using `then_signal_fence_and_flush`.
    #[inline]
    fn then_signal_fence(self) -> FenceSignalFuture<Self> where Self: Sized {
        let device = self.device().clone();

        assert!(self.queue().is_some());        // TODO: document

        let fence = Fence::new(device.clone()).unwrap();
        FenceSignalFuture {
            device: device,
            state: Mutex::new(FenceSignalFutureState::Pending(self, fence)),
        }
    }

    /// Signals a fence after this future. Returns another future that represents the signal.
    ///
    /// This is a just a shortcut for `then_signal_fence()` followed with `flush()`.
    #[inline]
    fn then_signal_fence_and_flush(self) -> Result<FenceSignalFuture<Self>, Box<Error>>
        where Self: Sized
    {
        let f = self.then_signal_fence();
        f.flush()?;
        Ok(f)
    }

    /// Presents a swapchain image after this future.
    ///
    /// You should only ever do this indirectly after a `SwapchainAcquireFuture` of the same image,
    /// otherwise an error will occur when flushing.
    ///
    /// > **Note**: This is just a shortcut for the `Swapchain::present()` function.
    #[inline]
    fn then_swapchain_present(self, queue: Arc<Queue>, swapchain: Arc<Swapchain>,
                              image_index: usize) -> PresentFuture<Self>
        where Self: Sized
    {
        Swapchain::present(swapchain, self, queue, image_index)
    }
}

unsafe impl<F: ?Sized> GpuFuture for Box<F> where F: GpuFuture {
    #[inline]
    fn cleanup_finished(&mut self) {
        (**self).cleanup_finished()
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Box<Error>> {
        (**self).build_submission()
    }

    #[inline]
    fn flush(&self) -> Result<(), Box<Error>> {
        (**self).flush()
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        (**self).signal_finished()
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        (**self).queue_change_allowed()
    }

    #[inline]
    fn queue(&self) -> Option<&Arc<Queue>> {
        (**self).queue()
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        (**self).check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        (**self).check_image_access(image, exclusive, queue)
    }
}

/// A dummy future that represents "now".
#[must_use]
pub struct DummyFuture {
    device: Arc<Device>,
}

impl DummyFuture {
    /// Builds a new dummy future.
    #[inline]
    pub fn new(device: Arc<Device>) -> DummyFuture {
        DummyFuture {
            device: device,
        }
    }
}

unsafe impl GpuFuture for DummyFuture {
    #[inline]
    fn cleanup_finished(&mut self) {
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Box<Error>> {
        Ok(SubmitAnyBuilder::Empty)
    }

    #[inline]
    fn flush(&self) -> Result<(), Box<Error>> {
        Ok(())
    }

    #[inline]
    unsafe fn signal_finished(&self) {
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        true
    }

    #[inline]
    fn queue(&self) -> Option<&Arc<Queue>> {
        None
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        Err(())
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        Err(())
    }
}

unsafe impl DeviceOwned for DummyFuture {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// Represents a semaphore being signaled after a previous event.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct SemaphoreSignalFuture<F> where F: GpuFuture {
    previous: F,
    semaphore: Semaphore,
    // True if the signaling command has already been submitted.
    // If flush is called multiple times, we want to block so that only one flushing is executed.
    // Therefore we use a `Mutex<bool>` and not an `AtomicBool`.
    wait_submitted: Mutex<bool>,
    finished: AtomicBool,
}

unsafe impl<F> GpuFuture for SemaphoreSignalFuture<F> where F: GpuFuture {
    #[inline]
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Box<Error>> {
        // Flushing the signaling part, since it must always be submitted before the waiting part.
        try!(self.flush());

        let mut sem = SubmitSemaphoresWaitBuilder::new();
        sem.add_wait_semaphore(&self.semaphore);
        Ok(SubmitAnyBuilder::SemaphoresWait(sem))
    }

    fn flush(&self) -> Result<(), Box<Error>> {
        unsafe {
            let mut wait_submitted = self.wait_submitted.lock().unwrap();

            if *wait_submitted {
                return Ok(());
            }

            let queue = self.previous.queue().unwrap().clone();

            match try!(self.previous.build_submission()) {
                SubmitAnyBuilder::Empty => {
                    let mut builder = SubmitCommandBufferBuilder::new();
                    builder.add_signal_semaphore(&self.semaphore);
                    try!(builder.submit(&queue));
                },
                SubmitAnyBuilder::SemaphoresWait(sem) => {
                    let mut builder: SubmitCommandBufferBuilder = sem.into();
                    builder.add_signal_semaphore(&self.semaphore);
                    try!(builder.submit(&queue));
                },
                SubmitAnyBuilder::CommandBuffer(mut builder) => {
                    debug_assert_eq!(builder.num_signal_semaphores(), 0);
                    builder.add_signal_semaphore(&self.semaphore);
                    try!(builder.submit(&queue));
                },
                SubmitAnyBuilder::QueuePresent(present) => {
                    try!(present.submit(&queue));
                    let mut builder = SubmitCommandBufferBuilder::new();
                    builder.add_signal_semaphore(&self.semaphore);
                    try!(builder.submit(&queue));       // FIXME: problematic because if we return an error and flush() is called again, then we'll submit the present twice
                },
            };

            // Only write `true` here in order to try again next time if an error occurs.
            *wait_submitted = true;
            Ok(())
        }
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        debug_assert!(*self.wait_submitted.lock().unwrap());
        self.finished.store(true, Ordering::SeqCst);
        self.previous.signal_finished();
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        true
    }

    #[inline]
    fn queue(&self) -> Option<&Arc<Queue>> {
        self.previous.queue()
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        self.previous.check_buffer_access(buffer, exclusive, queue).map(|_| None)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        self.previous.check_image_access(image, exclusive, queue).map(|_| None)
    }
}

unsafe impl<F> DeviceOwned for SemaphoreSignalFuture<F> where F: GpuFuture {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.semaphore.device()
    }
}

impl<F> Drop for SemaphoreSignalFuture<F> where F: GpuFuture {
    fn drop(&mut self) {
        unsafe {
            if !*self.finished.get_mut() {
                // TODO: handle errors?
                self.flush().unwrap();
                // Block until the queue finished.
                self.queue().unwrap().wait().unwrap();
                self.previous.signal_finished();
            }
        }
    }
}

/// Represents a fence being signaled after a previous event.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct FenceSignalFuture<F> where F: GpuFuture {
    // Current state. See the docs of `FenceSignalFutureState`.
    state: Mutex<FenceSignalFutureState<F>>,
    // The device of the future.
    device: Arc<Device>,
}

// This future can be in three different states: pending (ie. newly-created), submitted (ie. the
// command that submits the fence has been submitted), or cleaned (ie. the previous future has
// been dropped).
enum FenceSignalFutureState<F> {
    // Newly-created. Not submitted yet.
    Pending(F, Fence),

    // Partially submitted to the queue. Only happens in situations where submitting requires two
    // steps, and when the first step succeeded while the second step failed.
    //
    // Note that if there's ever a submit operation that needs three steps we will need to rework
    // this code, as it was designed for two-step operations only.
    PartiallyFlushed(F, Fence),

    // Submitted to the queue.
    Flushed(F, Fence),

    // The submission is finished. The previous future and the fence have been cleaned.
    Cleaned,

    // A function panicked while the state was being modified. Should never happen.
    Poisonned,
}

impl<F> FenceSignalFuture<F> where F: GpuFuture {
    // Implementation of `cleanup_finished`, but takes a `&self` instead of a `&mut self`.
    // This is an external function so that we can also call it from an `Arc<FenceSignalFuture>`.
    #[inline]
    fn cleanup_finished_impl(&self) {
        let mut state = self.state.lock().unwrap();

        match *state {
            FenceSignalFutureState::Flushed(_, ref fence) => {
                match fence.wait(Duration::from_secs(0)) {
                    Ok(()) => (),
                    Err(_) => return,
                }
            },
            _ => return,
        };

        // This code can only be reached if we're already flushed and waiting on the fence
        // succeeded.
        *state = FenceSignalFutureState::Cleaned;
    }

    // Implementation of `flush`. You must lock the state and pass the mutex guard here.
    fn flush_impl(&self, state: &mut MutexGuard<FenceSignalFutureState<F>>)
                  -> Result<(), Box<Error>>
    {
        unsafe {
            // In this function we temporarily replace the current state with `Poisonned` at the
            // beginning, and we take care to always put back a value into `state` before
            // returning (even in case of error).
            let old_state = mem::replace(&mut **state, FenceSignalFutureState::Poisonned);

            let (previous, fence, partially_flushed) = match old_state {
                FenceSignalFutureState::Pending(prev, fence) => {
                    (prev, fence, false)
                },
                FenceSignalFutureState::PartiallyFlushed(prev, fence) => {
                    (prev, fence, true)
                },
                other => {
                    // We were already flushed in the past, or we're already poisonned. Don't do
                    // anything.
                    **state = other;
                    return Ok(());
                },
            };

            // TODO: meh for unwrap
            let queue = previous.queue().unwrap().clone();

            // There are three possible outcomes for the flush operation: success, partial success
            // in which case `result` will contain `Err(OutcomeErr::Partial)`, or total failure
            // in which case `result` will contain `Err(OutcomeErr::Full)`.
            enum OutcomeErr<E> { Partial(E), Full(E) }
            let result = match try!(previous.build_submission()) {
                SubmitAnyBuilder::Empty => {
                    debug_assert!(!partially_flushed);
                    let mut b = SubmitCommandBufferBuilder::new();
                    b.set_fence_signal(&fence);
                    b.submit(&queue).map_err(|err| OutcomeErr::Full(err.into()))
                },
                SubmitAnyBuilder::SemaphoresWait(sem) => {
                    debug_assert!(!partially_flushed);
                    let b: SubmitCommandBufferBuilder = sem.into();
                    debug_assert!(!b.has_fence());
                    b.submit(&queue).map_err(|err| OutcomeErr::Full(err.into()))
                },
                SubmitAnyBuilder::CommandBuffer(mut cb_builder) => {
                    debug_assert!(!partially_flushed);
                    // The assert below could technically be a debug assertion as it is part of the
                    // safety contract of the trait. However it is easy to get this wrong if you
                    // write a custom implementation, and if so the consequences would be
                    // disastrous and hard to debug. Therefore we prefer to just use a regular
                    // assertion.
                    assert!(!cb_builder.has_fence());
                    cb_builder.set_fence_signal(&fence);
                    cb_builder.submit(&queue).map_err(|err| OutcomeErr::Full(err.into()))
                },
                SubmitAnyBuilder::QueuePresent(present) => {
                    let intermediary_result = if partially_flushed {
                        Ok(())
                    } else {
                        present.submit(&queue)
                    };
                    match intermediary_result {
                        Ok(()) => {
                            let mut b = SubmitCommandBufferBuilder::new();
                            b.set_fence_signal(&fence);
                            b.submit(&queue).map_err(|err| OutcomeErr::Partial(err.into()))
                        },
                        Err(err) => {
                            Err(OutcomeErr::Full(err.into()))
                        }
                    }
                },
            };

            // Restore the state before returning.
            match result {
                Ok(()) => {
                    **state = FenceSignalFutureState::Flushed(previous, fence);
                    Ok(())
                },
                Err(OutcomeErr::Partial(err)) => {
                    **state = FenceSignalFutureState::PartiallyFlushed(previous, fence);
                    Err(err)
                },
                Err(OutcomeErr::Full(err)) => {
                    **state = FenceSignalFutureState::Pending(previous, fence);
                    Err(err)
                },
            }
        }
    }
}

impl<F> FenceSignalFutureState<F> {
    #[inline]
    fn get_prev(&self) -> Option<&F> {
        match *self {
            FenceSignalFutureState::Pending(ref prev, _) => Some(prev),
            FenceSignalFutureState::PartiallyFlushed(ref prev, _) => Some(prev),
            FenceSignalFutureState::Flushed(ref prev, _) => Some(prev),
            FenceSignalFutureState::Cleaned => None,
            FenceSignalFutureState::Poisonned => None,
        }
    }
}

unsafe impl<F> GpuFuture for FenceSignalFuture<F> where F: GpuFuture {
    #[inline]
    fn cleanup_finished(&mut self) {
        self.cleanup_finished_impl()
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Box<Error>> {
        let mut state = self.state.lock().unwrap();
        try!(self.flush_impl(&mut state));

        match *state {
            FenceSignalFutureState::Flushed(_, ref fence) => {
                try!(fence.wait(Duration::from_secs(600)));     // TODO: arbitrary timeout?
            },
            FenceSignalFutureState::Cleaned | FenceSignalFutureState::Poisonned => (),
            FenceSignalFutureState::Pending(_, _)  => unreachable!(),
            FenceSignalFutureState::PartiallyFlushed(_, _) => unreachable!(),
        }

        Ok(SubmitAnyBuilder::Empty)
    }

    #[inline]
    fn flush(&self) -> Result<(), Box<Error>> {
        let mut state = self.state.lock().unwrap();
        self.flush_impl(&mut state)
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        let state = self.state.lock().unwrap();
        match *state {
            FenceSignalFutureState::Flushed(ref prev, _) => {
                prev.signal_finished();
            },
            FenceSignalFutureState::Cleaned | FenceSignalFutureState::Poisonned => (),
            _ => unreachable!(),
        }
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        true
    }

    #[inline]
    fn queue(&self) -> Option<&Arc<Queue>> {
        // FIXME: reimplement correctly ; either find a solution or change the API to take &mut self
        None
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()> {
        let state = self.state.lock().unwrap();
        if let Some(previous) = state.get_prev() {
            previous.check_buffer_access(buffer, exclusive, queue)
        } else {
            Err(())
        }
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, ()> {
        let state = self.state.lock().unwrap();
        if let Some(previous) = state.get_prev() {
            previous.check_image_access(image, exclusive, queue)
        } else {
            Err(())
        }
    }
}

unsafe impl<F> DeviceOwned for FenceSignalFuture<F> where F: GpuFuture {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl<F> Drop for FenceSignalFuture<F> where F: GpuFuture {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();

        // We ignore any possible error while submitting for now. Problems are handled below.
        let _ = self.flush_impl(&mut state);

        match mem::replace(&mut *state, FenceSignalFutureState::Cleaned) {
            FenceSignalFutureState::Flushed(previous, fence) => {
                // This is a normal situation. Submitting worked.
                // TODO: arbitrary timeout?
                // TODO: handle errors?
                fence.wait(Duration::from_secs(600)).unwrap();
                unsafe { previous.signal_finished(); }
            },
            FenceSignalFutureState::Cleaned => {
                // Also a normal situation. The user called `cleanup_finished()` before dropping.
            },
            FenceSignalFutureState::Poisonned => {
                // The previous future was already dropped and blocked the current queue.
            },
            FenceSignalFutureState::Pending(_, _) |
            FenceSignalFutureState::PartiallyFlushed(_, _) => {
                // Flushing produced an error. There's nothing more we can do except drop the 
                // previous future and let it block the current queue.
            },
        }
    }
}

unsafe impl<F> GpuFuture for Arc<FenceSignalFuture<F>> where F: GpuFuture {
    #[inline]
    fn cleanup_finished(&mut self) {
        self.cleanup_finished_impl()
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Box<Error>> {
        // Note that this is sound because we always return `SubmitAnyBuilder::Empty`. See the
        // documentation of `build_submission`.
        (**self).build_submission()
    }

    #[inline]
    fn flush(&self) -> Result<(), Box<Error>> {
        (**self).flush()
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        (**self).signal_finished()
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        (**self).queue_change_allowed()
    }

    #[inline]
    fn queue(&self) -> Option<&Arc<Queue>> {
        (**self).queue()
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        (**self).check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        (**self).check_image_access(image, exclusive, queue)
    }
}

/// Two futures joined into one.
#[must_use]
pub struct JoinFuture<A, B> {
    first: A,
    second: B,
}

unsafe impl<A, B> DeviceOwned for JoinFuture<A, B> where A: DeviceOwned, B: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        let device = self.first.device();
        debug_assert_eq!(self.second.device().internal_object(), device.internal_object());
        device
    }
}

unsafe impl<A, B> GpuFuture for JoinFuture<A, B> where A: GpuFuture, B: GpuFuture {
    #[inline]
    fn cleanup_finished(&mut self) {
        self.first.cleanup_finished();
        self.second.cleanup_finished();
    }

    #[inline]
    fn flush(&self) -> Result<(), Box<Error>> {
        // Since each future remembers whether it has been flushed, there's no safety issue here
        // if we call this function multiple times.
        try!(self.first.flush());
        try!(self.second.flush());
        Ok(())
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Box<Error>> {
        let first = try!(self.first.build_submission());
        let second = try!(self.second.build_submission());

        Ok(match (first, second) {
            (SubmitAnyBuilder::Empty, b) => b,
            (a, SubmitAnyBuilder::Empty) => a,
            (SubmitAnyBuilder::SemaphoresWait(mut a), SubmitAnyBuilder::SemaphoresWait(b)) => {
                a.merge(b);
                SubmitAnyBuilder::SemaphoresWait(a)
            },
            (SubmitAnyBuilder::SemaphoresWait(a), SubmitAnyBuilder::CommandBuffer(b)) => {
                try!(b.submit(&self.second.queue().clone().unwrap()));
                SubmitAnyBuilder::SemaphoresWait(a)
            },
            (SubmitAnyBuilder::CommandBuffer(a), SubmitAnyBuilder::SemaphoresWait(b)) => {
                try!(a.submit(&self.first.queue().clone().unwrap()));
                SubmitAnyBuilder::SemaphoresWait(b)
            },
            (SubmitAnyBuilder::SemaphoresWait(a), SubmitAnyBuilder::QueuePresent(b)) => {
                try!(b.submit(&self.second.queue().clone().unwrap()));
                SubmitAnyBuilder::SemaphoresWait(a)
            },
            (SubmitAnyBuilder::QueuePresent(a), SubmitAnyBuilder::SemaphoresWait(b)) => {
                try!(a.submit(&self.first.queue().clone().unwrap()));
                SubmitAnyBuilder::SemaphoresWait(b)
            },
            (SubmitAnyBuilder::CommandBuffer(a), SubmitAnyBuilder::CommandBuffer(b)) => {
                // TODO: we may want to add debug asserts here
                let new = a.merge(b);
                SubmitAnyBuilder::CommandBuffer(new)
            },
            (SubmitAnyBuilder::QueuePresent(a), SubmitAnyBuilder::QueuePresent(b)) => {
                try!(a.submit(&self.first.queue().clone().unwrap()));
                try!(b.submit(&self.second.queue().clone().unwrap()));
                SubmitAnyBuilder::Empty
            },
            (SubmitAnyBuilder::CommandBuffer(a), SubmitAnyBuilder::QueuePresent(b)) => {
                unimplemented!()
            },
            (SubmitAnyBuilder::QueuePresent(a), SubmitAnyBuilder::CommandBuffer(b)) => {
                unimplemented!()
            },
        })
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        self.first.signal_finished();
        self.second.signal_finished();
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        self.first.queue_change_allowed() && self.second.queue_change_allowed()
    }

    #[inline]
    fn queue(&self) -> Option<&Arc<Queue>> {
        match (self.first.queue(), self.second.queue()) {
            (Some(q1), Some(q2)) => if q1.is_same(&q2) {
                Some(q1)
            } else if self.first.queue_change_allowed() {
                Some(q2)
            } else if self.second.queue_change_allowed() {
                Some(q1)
            } else {
                None
            },
            (Some(q), None) => Some(q),
            (None, Some(q)) => Some(q),
            (None, None) => None,
        }
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        let first = self.first.check_buffer_access(buffer, exclusive, queue);
        let second = self.second.check_buffer_access(buffer, exclusive, queue);
        debug_assert!(!exclusive || !(first.is_ok() && second.is_ok()), "Two futures gave \
                                                                         exclusive access to the \
                                                                         same resource");
        match (first, second) {
            (Ok(v), Err(_)) | (Err(_), Ok(v)) => Ok(v),
            (Err(()), Err(())) => Err(()),
            (Ok(None), Ok(None)) => Ok(None),
            (Ok(Some(a)), Ok(None)) | (Ok(None), Ok(Some(a))) => Ok(Some(a)),
            (Ok(Some((a1, a2))), Ok(Some((b1, b2)))) => {
                Ok(Some((a1 | b1, a2 | b2)))
            },
        }
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        let first = self.first.check_image_access(image, exclusive, queue);
        let second = self.second.check_image_access(image, exclusive, queue);
        debug_assert!(!exclusive || !(first.is_ok() && second.is_ok()), "Two futures gave \
                                                                         exclusive access to the \
                                                                         same resource");
        match (first, second) {
            (Ok(v), Err(_)) | (Err(_), Ok(v)) => Ok(v),
            (Err(()), Err(())) => Err(()),
            (Ok(None), Ok(None)) => Ok(None),
            (Ok(Some(a)), Ok(None)) | (Ok(None), Ok(Some(a))) => Ok(Some(a)),
            (Ok(Some((a1, a2))), Ok(Some((b1, b2)))) => {
                Ok(Some((a1 | b1, a2 | b2)))
            },
        }
    }
}
