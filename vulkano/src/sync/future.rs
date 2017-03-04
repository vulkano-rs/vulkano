// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error::Error;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use buffer::Buffer;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferExecFuture;
use command_buffer::submit::SubmitAnyBuilder;
use command_buffer::submit::SubmitCommandBufferBuilder;
use command_buffer::submit::SubmitSemaphoresWaitBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::Image;
use swapchain::Swapchain;
use swapchain::PresentFuture;
use sync::Fence;
use sync::FenceWaitError;
use sync::Semaphore;

use SafeDeref;
use VulkanObject;

/// Represents an event that will happen on the GPU in the future.
// TODO: unsound if put inside an Arc
pub unsafe trait GpuFuture: DeviceOwned {
    /// Returns `true` if the event happened on the GPU.
    ///
    /// If this returns `false`, then the destuctor of this future will block until it is the case.
    ///
    /// If you didn't call `flush()` yet, then this function will return `false`.
    // TODO: what if user submits a cb without fence, calls flush, and then calls is_finished()
    // expecting it to return true eventually?
    fn is_finished(&self) -> bool;

    /// Builds a submission that, if submitted, makes sure that the event represented by this
    /// `GpuFuture` will happen, and possibly contains extra elements (eg. a semaphore wait or an
    /// event wait) that makes the dependency with subsequent operations work.
    ///
    /// It is the responsibility of the caller to ensure that the submission is going to be
    /// submitted only once. However keep in mind that this function can perfectly be called
    /// multiple times (as long as the returned object is only submitted once).
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
    /// > **Note**: Returning `true` means "access granted", while returning `false` means
    /// > "don't know". Therefore returning `false` is never unsafe.
    fn check_buffer_access(&self, buffer: &Buffer, exclusive: bool, queue: &Queue) -> bool;

    /// Checks whether submitting something after this future grants access (exclusive or shared,
    /// depending on the parameter) to the given image on the given queue.
    ///
    /// > **Note**: Returning `true` means "access granted", while returning `false` means
    /// > "don't know". Therefore returning `false` is never unsafe.
    ///
    /// > **Note**: Keep in mind that changing the layout of an image also requires exclusive
    /// > access.
    fn check_image_access(&self, image: &Image, exclusive: bool, queue: &Queue) -> bool;

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

    /// Signals a fence after this future. Returns another future that represents the signal.
    #[inline]
    fn then_signal_fence(self) -> FenceSignalFuture<Self> where Self: Sized {
        let device = self.device().clone();

        assert!(self.queue().is_some());        // TODO: document

        FenceSignalFuture {
            previous: Some(self),
            fence: Fence::new(device).unwrap(),
            flushed: Mutex::new(false),
        }
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

unsafe impl<T> GpuFuture for T where T: SafeDeref, T::Target: GpuFuture {
    #[inline]
    fn is_finished(&self) -> bool {
        (**self).is_finished()
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
    fn check_buffer_access(&self, buffer: &Buffer, exclusive: bool, queue: &Queue) -> bool {
        (**self).check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(&self, image: &Image, exclusive: bool, queue: &Queue) -> bool {
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
    fn is_finished(&self) -> bool {
        true
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
    fn check_buffer_access(&self, buffer: &Buffer, exclusive: bool, queue: &Queue) -> bool {
        false
    }

    #[inline]
    fn check_image_access(&self, image: &Image, exclusive: bool, queue: &Queue) -> bool {
        false
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
    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
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
    fn check_buffer_access(&self, buffer: &Buffer, exclusive: bool, queue: &Queue) -> bool {
        self.previous.check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(&self, image: &Image, exclusive: bool, queue: &Queue) -> bool {
        self.previous.check_image_access(image, exclusive, queue)
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
    previous: Option<F>,
    fence: Fence,
    // True if the signaling command has already been submitted.
    // If flush is called multiple times, we want to block so that only one flushing is executed.
    // Therefore we use a `Mutex<bool>` and not an `AtomicBool`.
    flushed: Mutex<bool>,
}

impl<F> FenceSignalFuture<F> where F: GpuFuture {
    /// Waits until the fence is signaled, or at least until the number of nanoseconds of the
    /// timeout has elapsed.
    pub fn wait(&self, timeout: Duration) -> Result<(), FenceWaitError> {
        // FIXME: flush?
        self.fence.wait(timeout)
    }
}

unsafe impl<F> GpuFuture for FenceSignalFuture<F> where F: GpuFuture {
    #[inline]
    fn is_finished(&self) -> bool {
        if !*self.flushed.lock().unwrap() {
            return false;
        }

        self.fence.wait(Duration::from_secs(0)).is_ok()
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Box<Error>> {
        try!(self.flush());
        self.fence.wait(Duration::from_secs(600)).unwrap();     // TODO: handle errors
        Ok(SubmitAnyBuilder::Empty)
    }

    #[inline]
    fn flush(&self) -> Result<(), Box<Error>> {
        unsafe {
            let mut flushed = self.flushed.lock().unwrap();

            if *flushed {
                return Ok(());
            }
            
            debug_assert!(self.previous.is_some());
            let queue = self.previous.as_ref().unwrap().queue().unwrap().clone();

            match try!(self.previous.as_ref().unwrap().build_submission()) {
                SubmitAnyBuilder::Empty => {
                    let mut b = SubmitCommandBufferBuilder::new();
                    b.set_fence_signal(&self.fence);
                    try!(b.submit(&queue));
                },
                SubmitAnyBuilder::SemaphoresWait(sem) => {
                    let b: SubmitCommandBufferBuilder = sem.into();
                    debug_assert!(!b.has_fence());
                    try!(b.submit(&queue));
                },
                SubmitAnyBuilder::CommandBuffer(mut cb_builder) => {
                    debug_assert!(!cb_builder.has_fence());
                    cb_builder.set_fence_signal(&self.fence);
                    try!(cb_builder.submit(&queue));
                },
                SubmitAnyBuilder::QueuePresent(present) => {
                    try!(present.submit(&queue));
                    let mut b = SubmitCommandBufferBuilder::new();
                    b.set_fence_signal(&self.fence);
                    try!(b.submit(&queue));       // FIXME: problematic because if we return an error and flush() is called again, then we'll submit the present twice
                },
            };

            // Only write `true` here in order to try again next time if an error occurs.
            *flushed = true;
            Ok(())
        }
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        debug_assert!(*self.flushed.lock().unwrap());
        if let Some(ref previous) = self.previous {
            previous.signal_finished();
        }
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        true
    }

    #[inline]
    fn queue(&self) -> Option<&Arc<Queue>> {
        self.previous.as_ref().and_then(|p| p.queue())
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &Buffer, exclusive: bool, queue: &Queue) -> bool {
        if let Some(ref previous) = self.previous {
            previous.check_buffer_access(buffer, exclusive, queue)
        } else {
            false
        }
    }

    #[inline]
    fn check_image_access(&self, image: &Image, exclusive: bool, queue: &Queue) -> bool {
        if let Some(ref previous) = self.previous {
            previous.check_image_access(image, exclusive, queue)
        } else {
            false
        }
    }
}

unsafe impl<F> DeviceOwned for FenceSignalFuture<F> where F: GpuFuture {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.fence.device()
    }
}

impl<F> Drop for FenceSignalFuture<F> where F: GpuFuture {
    fn drop(&mut self) {
        self.flush().unwrap();          // TODO: handle error?
        self.fence.wait(Duration::from_secs(600)).unwrap();     // TODO: handle some errors

        unsafe {
            if let Some(ref previous) = self.previous {
                previous.signal_finished();
            }
        }
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
    fn is_finished(&self) -> bool {
        self.first.is_finished() && self.second.is_finished()
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
    fn check_buffer_access(&self, buffer: &Buffer, exclusive: bool, queue: &Queue) -> bool {
        let first = self.first.check_buffer_access(buffer, exclusive, queue);
        let second = self.second.check_buffer_access(buffer, exclusive, queue);
        debug_assert!(!exclusive || !(first && second), "Two futures gave exclusive access to the \
                                                         same resource");
        first || second
    }

    #[inline]
    fn check_image_access(&self, image: &Image, exclusive: bool, queue: &Queue) -> bool {
        let first = self.first.check_image_access(image, exclusive, queue);
        let second = self.second.check_image_access(image, exclusive, queue);
        debug_assert!(!exclusive || !(first && second), "Two futures gave exclusive access to the \
                                                         same resource");
        first || second
    }
}
