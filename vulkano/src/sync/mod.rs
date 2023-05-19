// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
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

#[allow(unused)]
pub(crate) use self::pipeline::{PipelineStageAccess, PipelineStageAccessSet};
pub use self::{
    future::{now, FlushError, GpuFuture},
    pipeline::{
        AccessFlags, BufferMemoryBarrier, DependencyFlags, DependencyInfo, ImageMemoryBarrier,
        MemoryBarrier, PipelineMemoryAccess, PipelineStage, PipelineStages,
        QueueFamilyOwnershipTransfer,
    },
};
use crate::device::Queue;
use std::sync::Arc;

pub mod event;
pub mod fence;
pub mod future;
mod pipeline;
pub mod semaphore;

/// Declares in which queue(s) a resource can be used.
///
/// When you create a buffer or an image, you have to tell the Vulkan library in which queue
/// families it will be used. The vulkano library requires you to tell in which queue family
/// the resource will be used, even for exclusive mode.
#[derive(Debug, Clone, PartialEq, Eq)]
// TODO: remove
pub enum SharingMode {
    /// The resource is used is only one queue family.
    Exclusive,
    /// The resource is used in multiple queue families. Can be slower than `Exclusive`.
    Concurrent(Vec<u32>), // TODO: Vec is too expensive here
}

impl<'a> From<&'a Arc<Queue>> for SharingMode {
    #[inline]
    fn from(_queue: &'a Arc<Queue>) -> SharingMode {
        SharingMode::Exclusive
    }
}

impl<'a> From<&'a [&'a Arc<Queue>]> for SharingMode {
    #[inline]
    fn from(queues: &'a [&'a Arc<Queue>]) -> SharingMode {
        SharingMode::Concurrent(
            queues
                .iter()
                .map(|queue| queue.queue_family_index())
                .collect(),
        )
    }
}

/// Declares in which queue(s) a resource can be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sharing<I>
where
    I: IntoIterator<Item = u32>,
{
    /// The resource is used is only one queue family.
    Exclusive,
    /// The resource is used in multiple queue families. Can be slower than `Exclusive`.
    Concurrent(I),
}

/// How the memory of a resource is currently being accessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CurrentAccess {
    /// The resource is currently being accessed exclusively by the CPU.
    CpuExclusive,

    /// The resource is currently being accessed exclusively by the GPU.
    /// The GPU can have multiple exclusive accesses, if they are separated by synchronization.
    ///
    /// `gpu_writes` must not be 0. If it's decremented to 0, switch to `Shared`.
    GpuExclusive { gpu_reads: usize, gpu_writes: usize },

    /// The resource is not currently being accessed, or is being accessed for reading only.
    Shared { cpu_reads: usize, gpu_reads: usize },
}
