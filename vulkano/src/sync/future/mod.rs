// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Represents an event that will happen on the GPU in the future.
//!
//! Whenever you ask the GPU to start an operation by using a function of the vulkano library (for
//! example executing a command buffer), this function will return a *future*. A future is an
//! object that implements [the `GpuFuture` trait](crate::sync::GpuFuture) and that represents the
//! point in time when this operation is over.
//!
//! No function in vulkano immediately sends an operation to the GPU (with the exception of some
//! unsafe low-level functions). Instead they return a future that is in the pending state. Before
//! the GPU actually starts doing anything, you have to *flush* the future by calling the `flush()`
//! method or one of its derivatives.
//!
//! Futures serve several roles:
//!
//! - Futures can be used to build dependencies between operations and makes it possible to ask
//!   that an operation starts only after a previous operation is finished.
//! - Submitting an operation to the GPU is a costly operation. By chaining multiple operations
//!   with futures you will submit them all at once instead of one by one, thereby reducing this
//!   cost.
//! - Futures keep alive the resources and objects used by the GPU so that they don't get destroyed
//!   while they are still in use.
//!
//! The last point means that you should keep futures alive in your program for as long as their
//! corresponding operation is potentially still being executed by the GPU. Dropping a future
//! earlier will block the current thread (after flushing, if necessary) until the GPU has finished
//! the operation, which is usually not what you want.
//!
//! If you write a function that submits an operation to the GPU in your program, you are
//! encouraged to let this function return the corresponding future and let the caller handle it.
//! This way the caller will be able to chain multiple futures together and decide when it wants to
//! keep the future alive or drop it.
//!
//! # Executing an operation after a future
//!
//! Respecting the order of operations on the GPU is important, as it is what *proves* vulkano that
//! what you are doing is indeed safe. For example if you submit two operations that modify the
//! same buffer, then you need to execute one after the other instead of submitting them
//! independently. Failing to do so would mean that these two operations could potentially execute
//! simultaneously on the GPU, which would be unsafe.
//!
//! This is done by calling one of the methods of the `GpuFuture` trait. For example calling
//! `prev_future.then_execute(command_buffer)` takes ownership of `prev_future` and will make sure
//! to only start executing `command_buffer` after the moment corresponding to `prev_future`
//! happens. The object returned by the `then_execute` function is itself a future that corresponds
//! to the moment when the execution of `command_buffer` ends.
//!
//! ## Between two different GPU queues
//!
//! When you want to perform an operation after another operation on two different queues, you
//! **must** put a *semaphore* between them. Failure to do so would result in a runtime error.
//! Adding a semaphore is a simple as replacing `prev_future.then_execute(...)` with
//! `prev_future.then_signal_semaphore().then_execute(...)`.
//!
//! > **Note**: A common use-case is using a transfer queue (ie. a queue that is only capable of
//! > performing transfer operations) to write data to a buffer, then read that data from the
//! > rendering queue.
//!
//! What happens when you do so is that the first queue will execute the first set of operations
//! (represented by `prev_future` in the example), then put a semaphore in the signalled state.
//! Meanwhile the second queue blocks (if necessary) until that same semaphore gets signalled, and
//! then only will execute the second set of operations.
//!
//! Since you want to avoid blocking the second queue as much as possible, you probably want to
//! flush the operation to the first queue as soon as possible. This can easily be done by calling
//! `then_signal_semaphore_and_flush()` instead of `then_signal_semaphore()`.
//!
//! ## Between several different GPU queues
//!
//! The `then_signal_semaphore()` method is appropriate when you perform an operation in one queue,
//! and want to see the result in another queue. However in some situations you want to start
//! multiple operations on several different queues.
//!
//! TODO: this is not yet implemented
//!
//! # Fences
//!
//! A `Fence` is an object that is used to signal the CPU when an operation on the GPU is finished.
//!
//! Signalling a fence is done by calling `then_signal_fence()` on a future. Just like semaphores,
//! you are encouraged to use `then_signal_fence_and_flush()` instead.
//!
//! Signalling a fence is kind of a "terminator" to a chain of futures

pub use self::{
    fence_signal::{FenceSignalFuture, FenceSignalFutureBehavior},
    join::JoinFuture,
    now::{now, NowFuture},
    semaphore_signal::SemaphoreSignalFuture,
};
use super::{
    fence::{Fence, FenceError},
    semaphore::Semaphore,
};
use crate::{
    buffer::Buffer,
    command_buffer::{
        CommandBufferExecError, CommandBufferExecFuture, PrimaryCommandBufferAbstract,
        ResourceUseRef, SubmitInfo,
    },
    device::{DeviceOwned, Queue},
    image::{sys::Image, ImageLayout},
    memory::BindSparseInfo,
    swapchain::{self, PresentFuture, PresentInfo, Swapchain, SwapchainPresentInfo},
    DeviceSize, OomError, RuntimeError,
};
use smallvec::SmallVec;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    ops::Range,
    sync::Arc,
};

mod fence_signal;
mod join;
mod now;
mod semaphore_signal;

/// Represents an event that will happen on the GPU in the future.
///
/// See the documentation of the `sync` module for explanations about futures.
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
    /// Also note that calling `flush()` on the future  may change the value returned by
    /// `build_submission()`.
    ///
    /// It is however the responsibility of the implementation to not return the same submission
    /// from multiple different future objects. For example if you implement `GpuFuture` on
    /// `Arc<Foo>` then `build_submission()` must always return `SubmitAnyBuilder::Empty`,
    /// otherwise it would be possible for the user to clone the `Arc` and make the same
    /// submission be submitted multiple times.
    ///
    /// It is also the responsibility of the implementation to ensure that it works if you call
    /// `build_submission()` and submits the returned value without calling `flush()` first. In
    /// other words, `build_submission()` should perform an implicit flush if necessary.
    ///
    /// Once the caller has submitted the submission and has determined that the GPU has finished
    /// executing it, it should call `signal_finished`. Failure to do so will incur a large runtime
    /// overhead, as the future will have to block to make sure that it is finished.
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError>;

    /// Flushes the future and submits to the GPU the actions that will permit this future to
    /// occur.
    ///
    /// The implementation must remember that it was flushed. If the function is called multiple
    /// times, only the first time must result in a flush.
    fn flush(&self) -> Result<(), FlushError>;

    /// Sets the future to its "complete" state, meaning that it can safely be destroyed.
    ///
    /// This must only be done if you called `build_submission()`, submitted the returned
    /// submission, and determined that it was finished.
    ///
    /// The implementation must be aware that this function can be called multiple times on the
    /// same future.
    unsafe fn signal_finished(&self);

    /// Returns the queue that triggers the event. Returns `None` if unknown or irrelevant.
    ///
    /// If this function returns `None` and `queue_change_allowed` returns `false`, then a panic
    /// is likely to occur if you use this future. This is only a problem if you implement
    /// the `GpuFuture` trait yourself for a type outside of vulkano.
    fn queue(&self) -> Option<Arc<Queue>>;

    /// Returns `true` if elements submitted after this future can be submitted to a different
    /// queue than the other returned by `queue()`.
    fn queue_change_allowed(&self) -> bool;

    /// Checks whether submitting something after this future grants access (exclusive or shared,
    /// depending on the parameter) to the given buffer on the given queue.
    ///
    /// > **Note**: Returning `Ok` means "access granted", while returning `Err` means
    /// > "don't know". Therefore returning `Err` is never unsafe.
    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<(), AccessCheckError>;

    /// Checks whether submitting something after this future grants access (exclusive or shared,
    /// depending on the parameter) to the given image on the given queue.
    ///
    /// Implementations must ensure that the image is in the given layout. However if the `layout`
    /// is `Undefined` then the implementation should accept any actual layout.
    ///
    /// > **Note**: Returning `Ok` means "access granted", while returning `Err` means
    /// > "don't know". Therefore returning `Err` is never unsafe.
    ///
    /// > **Note**: Keep in mind that changing the layout of an image also requires exclusive
    /// > access.
    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<(), AccessCheckError>;

    /// Checks whether accessing a swapchain image is permitted.
    ///
    /// > **Note**: Setting `before` to `true` should skip checking the current future and always
    /// > forward the call to the future before.
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), AccessCheckError>;

    /// Joins this future with another one, representing the moment when both events have happened.
    // TODO: handle errors
    fn join<F>(self, other: F) -> JoinFuture<Self, F>
    where
        Self: Sized,
        F: GpuFuture,
    {
        join::join(self, other)
    }

    /// Executes a command buffer after this future.
    ///
    /// > **Note**: This is just a shortcut function. The actual implementation is in the
    /// > `CommandBuffer` trait.
    fn then_execute<Cb>(
        self,
        queue: Arc<Queue>,
        command_buffer: Cb,
    ) -> Result<CommandBufferExecFuture<Self>, CommandBufferExecError>
    where
        Self: Sized,
        Cb: PrimaryCommandBufferAbstract + 'static,
    {
        command_buffer.execute_after(self, queue)
    }

    /// Executes a command buffer after this future, on the same queue as the future.
    ///
    /// > **Note**: This is just a shortcut function. The actual implementation is in the
    /// > `CommandBuffer` trait.
    fn then_execute_same_queue<Cb>(
        self,
        command_buffer: Cb,
    ) -> Result<CommandBufferExecFuture<Self>, CommandBufferExecError>
    where
        Self: Sized,
        Cb: PrimaryCommandBufferAbstract + 'static,
    {
        let queue = self.queue().unwrap();
        command_buffer.execute_after(self, queue)
    }

    /// Signals a semaphore after this future. Returns another future that represents the signal.
    ///
    /// Call this function when you want to execute some operations on a queue and want to see the
    /// result on another queue.
    #[inline]
    fn then_signal_semaphore(self) -> SemaphoreSignalFuture<Self>
    where
        Self: Sized,
    {
        semaphore_signal::then_signal_semaphore(self)
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
    fn then_signal_semaphore_and_flush(self) -> Result<SemaphoreSignalFuture<Self>, FlushError>
    where
        Self: Sized,
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
    fn then_signal_fence(self) -> FenceSignalFuture<Self>
    where
        Self: Sized,
    {
        fence_signal::then_signal_fence(self, FenceSignalFutureBehavior::Continue)
    }

    /// Signals a fence after this future. Returns another future that represents the signal.
    ///
    /// This is a just a shortcut for `then_signal_fence()` followed with `flush()`.
    #[inline]
    fn then_signal_fence_and_flush(self) -> Result<FenceSignalFuture<Self>, FlushError>
    where
        Self: Sized,
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
    fn then_swapchain_present(
        self,
        queue: Arc<Queue>,
        swapchain_info: SwapchainPresentInfo,
    ) -> PresentFuture<Self>
    where
        Self: Sized,
    {
        swapchain::present(self, queue, swapchain_info)
    }

    /// Turn the current future into a `Box<dyn GpuFuture>`.
    ///
    /// This is a helper function that calls `Box::new(yourFuture) as Box<dyn GpuFuture>`.
    #[inline]
    fn boxed(self) -> Box<dyn GpuFuture>
    where
        Self: Sized + 'static,
    {
        Box::new(self) as _
    }

    /// Turn the current future into a `Box<dyn GpuFuture + Send>`.
    ///
    /// This is a helper function that calls `Box::new(yourFuture) as Box<dyn GpuFuture + Send>`.
    #[inline]
    fn boxed_send(self) -> Box<dyn GpuFuture + Send>
    where
        Self: Sized + Send + 'static,
    {
        Box::new(self) as _
    }

    /// Turn the current future into a `Box<dyn GpuFuture + Sync>`.
    ///
    /// This is a helper function that calls `Box::new(yourFuture) as Box<dyn GpuFuture + Sync>`.
    #[inline]
    fn boxed_sync(self) -> Box<dyn GpuFuture + Sync>
    where
        Self: Sized + Sync + 'static,
    {
        Box::new(self) as _
    }

    /// Turn the current future into a `Box<dyn GpuFuture + Send + Sync>`.
    ///
    /// This is a helper function that calls `Box::new(yourFuture) as Box<dyn GpuFuture + Send +
    /// Sync>`.
    #[inline]
    fn boxed_send_sync(self) -> Box<dyn GpuFuture + Send + Sync>
    where
        Self: Sized + Send + Sync + 'static,
    {
        Box::new(self) as _
    }
}

unsafe impl<F: ?Sized> GpuFuture for Box<F>
where
    F: GpuFuture,
{
    fn cleanup_finished(&mut self) {
        (**self).cleanup_finished()
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        (**self).build_submission()
    }

    fn flush(&self) -> Result<(), FlushError> {
        (**self).flush()
    }

    unsafe fn signal_finished(&self) {
        (**self).signal_finished()
    }

    fn queue_change_allowed(&self) -> bool {
        (**self).queue_change_allowed()
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        (**self).queue()
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        (**self).check_buffer_access(buffer, range, exclusive, queue)
    }

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        (**self).check_image_access(image, range, exclusive, expected_layout, queue)
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), AccessCheckError> {
        (**self).check_swapchain_image_acquired(swapchain, image_index, before)
    }
}

/// Contains all the possible submission builders.
#[derive(Debug)]
pub enum SubmitAnyBuilder {
    Empty,
    SemaphoresWait(SmallVec<[Arc<Semaphore>; 8]>),
    CommandBuffer(SubmitInfo, Option<Arc<Fence>>),
    QueuePresent(PresentInfo),
    BindSparse(SmallVec<[BindSparseInfo; 1]>, Option<Arc<Fence>>),
}

impl SubmitAnyBuilder {
    /// Returns true if equal to `SubmitAnyBuilder::Empty`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, SubmitAnyBuilder::Empty)
    }
}

/// Access to a resource was denied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessError {
    /// Exclusive access is denied.
    ExclusiveDenied,

    /// The resource is already in use, and there is no tracking of concurrent usages.
    AlreadyInUse,

    UnexpectedImageLayout {
        allowed: ImageLayout,
        requested: ImageLayout,
    },

    /// Trying to use an image without transitioning it from the "undefined" or "preinitialized"
    /// layouts first.
    ImageNotInitialized {
        /// The layout that was requested for the image.
        requested: ImageLayout,
    },

    /// Trying to use a buffer that still contains garbage data.
    BufferNotInitialized,

    /// Trying to use a swapchain image without depending on a corresponding acquire image future.
    SwapchainImageNotAcquired,
}

impl Error for AccessError {}

impl Display for AccessError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
                AccessError::ExclusiveDenied => "only shared access is allowed for this resource",
                AccessError::AlreadyInUse => {
                    "the resource is already in use, and there is no tracking of concurrent usages"
                }
                AccessError::UnexpectedImageLayout { .. } => {
                    unimplemented!() // TODO: find a description
                }
                AccessError::ImageNotInitialized { .. } => {
                    "trying to use an image without transitioning it from the undefined or \
                    preinitialized layouts first"
                }
                AccessError::BufferNotInitialized => {
                    "trying to use a buffer that still contains garbage data"
                }
                AccessError::SwapchainImageNotAcquired => {
                    "trying to use a swapchain image without depending on a corresponding acquire \
                    image future"
                }
            }
        )
    }
}

/// Error that can happen when checking whether we have access to a resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessCheckError {
    /// Access to the resource has been denied.
    Denied(AccessError),
    /// The resource is unknown, therefore we cannot possibly answer whether we have access or not.
    Unknown,
}

impl Error for AccessCheckError {}

impl Display for AccessCheckError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
                AccessCheckError::Denied(_) => "access to the resource has been denied",
                AccessCheckError::Unknown => "the resource is unknown",
            }
        )
    }
}

impl From<AccessError> for AccessCheckError {
    fn from(err: AccessError) -> AccessCheckError {
        AccessCheckError::Denied(err)
    }
}

/// Error that can happen when creating a graphics pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlushError {
    /// Access to a resource has been denied.
    AccessError(AccessError),

    /// Not enough memory.
    OomError(OomError),

    /// The connection to the device has been lost.
    DeviceLost,

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,

    /// The surface has changed in a way that makes the swapchain unusable. You must query the
    /// surface's new properties and recreate a new swapchain if you want to continue drawing.
    OutOfDate,

    /// The swapchain has lost or doesn't have full screen exclusivity possibly for
    /// implementation-specific reasons outside of the application’s control.
    FullScreenExclusiveModeLost,

    /// The flush operation needed to block, but the timeout has elapsed.
    Timeout,

    /// A non-zero present_id must be greater than any non-zero present_id passed previously
    /// for the same swapchain.
    PresentIdLessThanOrEqual,

    /// Access to a resource has been denied.
    ResourceAccessError {
        error: AccessError,
        use_ref: Option<ResourceUseRef>,
    },

    /// The command buffer or one of the secondary command buffers it executes was created with the
    /// "one time submit" flag, but has already been submitted it the past.
    OneTimeSubmitAlreadySubmitted,

    /// The command buffer or one of the secondary command buffers it executes is already in use by
    /// the GPU and was not created with the "concurrent" flag.
    ExclusiveAlreadyInUse,
}

impl Error for FlushError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            FlushError::AccessError(err) => Some(err),
            FlushError::OomError(err) => Some(err),
            FlushError::ResourceAccessError { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl Display for FlushError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
                FlushError::AccessError(_) => "access to a resource has been denied",
                FlushError::OomError(_) => "not enough memory",
                FlushError::DeviceLost => "the connection to the device has been lost",
                FlushError::SurfaceLost => "the surface of this swapchain is no longer valid",
                FlushError::OutOfDate => "the swapchain needs to be recreated",
                FlushError::FullScreenExclusiveModeLost => {
                    "the swapchain no longer has full screen exclusivity"
                }
                FlushError::Timeout => {
                    "the flush operation needed to block, but the timeout has elapsed"
                }
                FlushError::PresentIdLessThanOrEqual => {
                    "present id is less than or equal to previous"
                }
                FlushError::ResourceAccessError { .. } => "access to a resource has been denied",
                FlushError::OneTimeSubmitAlreadySubmitted => {
                    "the command buffer or one of the secondary command buffers it executes was \
                    created with the \"one time submit\" flag, but has already been submitted in \
                    the past"
                }
                FlushError::ExclusiveAlreadyInUse => {
                    "the command buffer or one of the secondary command buffers it executes is \
                    already in use was not created with the \"concurrent\" flag"
                }
            }
        )
    }
}

impl From<AccessError> for FlushError {
    fn from(err: AccessError) -> FlushError {
        FlushError::AccessError(err)
    }
}

impl From<RuntimeError> for FlushError {
    fn from(err: RuntimeError) -> Self {
        match err {
            RuntimeError::OutOfHostMemory | RuntimeError::OutOfDeviceMemory => {
                Self::OomError(err.into())
            }
            RuntimeError::DeviceLost => Self::DeviceLost,
            RuntimeError::SurfaceLost => Self::SurfaceLost,
            RuntimeError::OutOfDate => Self::OutOfDate,
            RuntimeError::FullScreenExclusiveModeLost => Self::FullScreenExclusiveModeLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<FenceError> for FlushError {
    fn from(err: FenceError) -> FlushError {
        match err {
            FenceError::OomError(err) => FlushError::OomError(err),
            FenceError::Timeout => FlushError::Timeout,
            FenceError::DeviceLost => FlushError::DeviceLost,
            _ => unreachable!(),
        }
    }
}
