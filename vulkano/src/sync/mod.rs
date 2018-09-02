// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Synchronization on the GPU.
//!
//! Just like for CPU code, you have to ensure that buffers and images are not accessed mutably by
//! multiple GPU queues simultaneously and that they are not accessed mutably by the CPU and by the
//! GPU simultaneously.
//!
//! This safety is enforced at runtime by vulkano but it is not magic and you will require some
//! knowledge if you want to avoid errors.
//!
//! # Futures
//!
//! Whenever you ask the GPU to start an operation by using a function of the vulkano library (for
//! example executing a command buffer), this function will return a *future*. A future is an
//! object that implements [the `GpuFuture` trait](trait.GpuFuture.html) and that represents the
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
//! Signalling a fence is kind of a "terminator" to a chain of futures.
//!
//! TODO: lots of problems with how to use fences
//! TODO: talk about fence + semaphore simultaneously
//! TODO: talk about using fences to clean up

use device::Queue;
use std::sync::Arc;

pub use self::event::Event;
pub use self::fence::Fence;
pub use self::fence::FenceWaitError;
pub use self::future::AccessCheckError;
pub use self::future::AccessError;
pub use self::future::FenceSignalFuture;
pub use self::future::FlushError;
pub use self::future::GpuFuture;
pub use self::future::JoinFuture;
pub use self::future::NowFuture;
pub use self::future::SemaphoreSignalFuture;
pub use self::future::now;
pub use self::pipeline::AccessFlagBits;
pub use self::pipeline::PipelineStages;
pub use self::semaphore::Semaphore;

mod event;
mod fence;
mod future;
mod pipeline;
mod semaphore;

/// Declares in which queue(s) a resource can be used.
///
/// When you create a buffer or an image, you have to tell the Vulkan library in which queue
/// families it will be used. The vulkano library requires you to tell in which queue family
/// the resource will be used, even for exclusive mode.
#[derive(Debug, Clone, PartialEq, Eq)]
// TODO: remove
pub enum SharingMode {
    /// The resource is used is only one queue family.
    Exclusive(u32),
    /// The resource is used in multiple queue families. Can be slower than `Exclusive`.
    Concurrent(Vec<u32>), // TODO: Vec is too expensive here
}

impl<'a> From<&'a Arc<Queue>> for SharingMode {
    #[inline]
    fn from(queue: &'a Arc<Queue>) -> SharingMode {
        SharingMode::Exclusive(queue.family().id())
    }
}

impl<'a> From<&'a [&'a Arc<Queue>]> for SharingMode {
    #[inline]
    fn from(queues: &'a [&'a Arc<Queue>]) -> SharingMode {
        SharingMode::Concurrent(queues.iter().map(|queue| queue.family().id()).collect())
    }
}

/// Declares in which queue(s) a resource can be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sharing<I>
    where I: Iterator<Item = u32>
{
    /// The resource is used is only one queue family.
    Exclusive,
    /// The resource is used in multiple queue families. Can be slower than `Exclusive`.
    Concurrent(I),
}
