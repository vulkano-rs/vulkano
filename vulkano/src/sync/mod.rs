// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Synchronization primitives for Vulkan objects.
//! 
//! In Vulkan, you have to manually ensure two things:
//! 
//! - That a buffer or an image are not read and written simultaneously (similarly to the CPU).
//! - That writes to a buffer or an image are propagated to other queues by inserting memory
//!   barriers.
//!
//! But don't worry ; this is automatically enforced by this library (as long as you don't use
//! any unsafe function). See the `memory` module for more info.
//!

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
