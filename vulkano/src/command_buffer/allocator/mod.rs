// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! In the Vulkan API, command buffers must be allocated from *command pools*.
//!
//! A command pool holds and manages the memory of one or more command buffers. If you destroy a
//! command pool, all of its command buffers are automatically destroyed.
//!
//! In vulkano, creating a command buffer requires passing an implementation of the
//! [`CommandBufferAllocator`] trait, which you can implement yourself or use the vulkano-provided
//! [`StandardCommandBufferAllocator`].

pub use self::standard::StandardCommandBufferAllocator;
use super::{pool::CommandPoolAlloc, CommandBufferLevel};
use crate::{device::DeviceOwned, OomError};

pub mod standard;

/// Types that manage the memory of command buffers.
///
/// # Safety
///
/// A Vulkan command pool must be externally synchronized as if it owned the command buffers that
/// were allocated from it. This includes allocating from the pool, freeing from the pool, resetting
/// the pool or individual command buffers, and most importantly recording commands to command
/// buffers. The implementation of `CommandBufferAllocator` is expected to manage this.
///
/// The destructors of the [`CommandBufferBuilderAlloc`] and the [`CommandBufferAlloc`] are expected
/// to free the command buffer, reset the command buffer, or add it to a pool so that it gets
/// reused. If the implementation frees or resets the command buffer, it must not forget that this
/// operation must be externally synchronized.
pub unsafe trait CommandBufferAllocator: DeviceOwned {
    /// See [`allocate`](Self::allocate).
    type Iter: Iterator<Item = Self::Builder>;

    /// Represents a command buffer that has been allocated and that is currently being built.
    type Builder: CommandBufferBuilderAlloc<Alloc = Self::Alloc>;

    /// Represents a command buffer that has been allocated and that is pending execution or is
    /// being executed.
    type Alloc: CommandBufferAlloc;

    /// Allocates command buffers.
    ///
    /// Returns an iterator that contains the requested amount of allocated command buffers.
    fn allocate(
        &self,
        level: CommandBufferLevel,
        command_buffer_count: u32,
    ) -> Result<Self::Iter, OomError>;

    /// Returns the index of the queue family that this pool targets.
    fn queue_family_index(&self) -> u32;
}

/// A command buffer allocated from a pool and that can be recorded.
///
/// # Safety
///
/// See [`CommandBufferAllocator`] for information about safety.
pub unsafe trait CommandBufferBuilderAlloc: DeviceOwned {
    /// Return type of `into_alloc`.
    type Alloc: CommandBufferAlloc;

    /// Returns the internal object that contains the command buffer.
    fn inner(&self) -> &CommandPoolAlloc;

    /// Turns this builder into a command buffer that is pending execution.
    fn into_alloc(self) -> Self::Alloc;

    /// Returns the index of the queue family that the pool targets.
    fn queue_family_index(&self) -> u32;
}

/// A command buffer allocated from a pool that has finished being recorded.
///
/// # Safety
///
/// See [`CommandBufferAllocator`] for information about safety.
pub unsafe trait CommandBufferAlloc: DeviceOwned + Send + Sync + 'static {
    /// Returns the internal object that contains the command buffer.
    fn inner(&self) -> &CommandPoolAlloc;

    /// Returns the index of the queue family that the pool targets.
    fn queue_family_index(&self) -> u32;
}
