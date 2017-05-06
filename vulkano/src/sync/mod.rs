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
//! Whenever you want ask the GPU to start an operation (for example executing a command buffer),
//! you need to call a function from vulkano that returns a *future*. A future is an object that
//! implements [the `GpuFuture` trait](trait.GpuFuture.html) and that represents the point in time
//! when the operation is over.
//!
//! Futures serve several roles:
//!
//! - Futures can be used to build dependencies between submissions so that you can ask that an
//!   operation starts only after a previous operation is finished.
//! - Submitting an operation to the GPU is a costly operation. When chaining multiple operations
//!   with futures you will submit them all at once instead of one by one, thereby reducing this
//!   cost.
//! - Futures keep alive the resources and objects used by the GPU so that they don't get destroyed
//!   while they are still in use.
//!
//! The last point means that you should keep futures alive in your program for as long as their
//! corresponding operation is potentially still being executed by the GPU. Dropping a future
//! earlier will block the current thread until the GPU has finished the operation, which is not
//! usually not what you want.
//!
//! In other words if you write a function in your program that submits an operation to the GPU,
//! you should always make this function return the corresponding future and let the caller handle
//! it.
//!
//! # Dependencies between futures
//!
//! Building dependencies between futures is important, as it is what *proves* vulkano that
//! some an operation is indeed safe. For example if you submit two operations that modify the same
//! buffer, then you need to make sure that one of them gets executed after the other. Failing to
//! add a dependency would mean that these two operations could potentially execute simultaneously
//! on the GPU, which would be unsafe.
//!
//! Adding a dependency is done by calling one of the methods of the `GpuFuture` trait. For example
//! calling `prev_future.then_execute(command_buffer)` takes ownership of `prev_future` and returns
//! a new future in which `command_buffer` starts executing after the moment corresponding to
//! `prev_future` happens. The new future corresponds to the moment when the execution of
//! `command_buffer` ends.
//!
//! ## Between different GPU queues
//!
//! When you want to perform an operation after another operation on two different queues, you
//! **must** put a *semaphore* between them. Failure to do so would result in a runtime error.
//!
//! Adding a semaphore is a simple as replacing `prev_future.then_execute(...)` with
//! `prev_future.then_signal_semaphore().then_execute(...)`.
//!
//! In practice you usually want to use `then_signal_semaphore_and_flush()` instead of
//! `then_signal_semaphore()`, as the execution will start sooner.
//!
//! TODO: using semaphores to dispatch to multiple queues
//!
//! # Fences
//!
//! A `Fence` is an object that is used to signal the CPU when an operation on the GPU is finished.
//!
//! If you want to perform an operation on the CPU after an operation on the GPU is finished (eg.
//! if you want to read on the CPU data written by the GPU), then you need to ask the GPU to signal
//! a fence after the operation and wait for that fence to be *signalled* on the CPU.
//!
//! TODO: talk about fence + semaphore simultaneously
//! TODO: talk about using fences to clean up

use std::sync::Arc;
use device::Queue;

pub use self::event::Event;
pub use self::fence::Fence;
pub use self::fence::FenceWaitError;
pub use self::future::DummyFuture;
pub use self::future::GpuFuture;
pub use self::future::SemaphoreSignalFuture;
pub use self::future::FenceSignalFuture;
pub use self::future::JoinFuture;
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
/// families it will be used. The vulkano library requires you to tell in which queue famiily
/// the resource will be used, even for exclusive mode.
#[derive(Debug, Clone, PartialEq, Eq)]
// TODO: remove
pub enum SharingMode {
    /// The resource is used is only one queue family.
    Exclusive(u32),
    /// The resource is used in multiple queue families. Can be slower than `Exclusive`.
    Concurrent(Vec<u32>),       // TODO: Vec is too expensive here
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
        SharingMode::Concurrent(queues.iter().map(|queue| {
            queue.family().id()
        }).collect())
    }
}

/// Declares in which queue(s) a resource can be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sharing<I> where I: Iterator<Item = u32> {
    /// The resource is used is only one queue family.
    Exclusive,
    /// The resource is used in multiple queue families. Can be slower than `Exclusive`.
    Concurrent(I),
}
