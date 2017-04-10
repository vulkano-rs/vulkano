// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! In the Vulkan API, command buffers must be allocated from *command pools*.
//! 
//! A command pool holds and manages the memory of one or more command buffers. If you destroy a
//! command pool, all of its command buffers are automatically destroyed.
//! 
//! In vulkano, creating a command buffer requires passing an implementation of the `CommandPool`
//! trait. By default vulkano will use the `StandardCommandPool` struct, but you can implement
//! this trait yourself by wrapping around the `UnsafeCommandPool` type.

use instance::QueueFamily;

use device::DeviceOwned;
use OomError;
use VulkanObject;
use vk;

pub use self::standard::StandardCommandPool;
pub use self::standard::StandardCommandPoolFinished;
pub use self::sys::UnsafeCommandPool;
pub use self::sys::UnsafeCommandPoolAllocIter;
pub use self::sys::CommandPoolTrimError;

mod standard;
mod sys;

/// Types that manage the memory of command buffers.
pub unsafe trait CommandPool: DeviceOwned {
    /// See `alloc()`.
    type Iter: Iterator<Item = AllocatedCommandBuffer>;
    /// See `lock()`.
    type Lock;
    /// See `finish()`.
    type Finished: CommandPoolFinished;

    /// Allocates command buffers from this pool.
    fn alloc(&self, secondary: bool, count: u32) -> Result<Self::Iter, OomError>;

    /// Frees command buffers from this pool.
    ///
    /// # Safety
    ///
    /// - The command buffers must have been allocated from this pool.
    /// - `secondary` must have the same value as what was passed to `alloc`.
    ///
    unsafe fn free<I>(&self, secondary: bool, command_buffers: I)
        where I: Iterator<Item = AllocatedCommandBuffer>;

    /// Once a command buffer has finished being built, it should call this method in order to
    /// produce a `Finished` object.
    ///
    /// The `Finished` object must hold the pool alive.
    ///
    /// The point of this object is to change the Send/Sync strategy after a command buffer has
    /// finished being built compared to before.
    fn finish(self) -> Self::Finished;

    /// Before any command buffer allocated from this pool can be modified, the pool itself must
    /// be locked by calling this method.
    ///
    /// All the operations are atomic at the thread level, so the point of this lock is to
    /// prevent the pool from being accessed from multiple threads in parallel.
    fn lock(&self) -> Self::Lock;

    /// Returns true if command buffers can be reset individually. In other words, if the pool
    /// was created with `reset_cb` set to true.
    fn can_reset_invidual_command_buffers(&self) -> bool;

    /// Returns the queue family that this pool targets.
    fn queue_family(&self) -> QueueFamily;
}

/// See `CommandPool::finish()`.
pub unsafe trait CommandPoolFinished: DeviceOwned {
    /// Frees command buffers.
    ///
    /// # Safety
    ///
    /// - The command buffers must have been allocated from this pool.
    /// - `secondary` must have the same value as what was passed to `alloc`.
    ///
    unsafe fn free<I>(&self, secondary: bool, command_buffers: I)
        where I: Iterator<Item = AllocatedCommandBuffer>;

    /// Returns the queue family that this pool targets.
    fn queue_family(&self) -> QueueFamily;
}

/// Opaque type that represents a command buffer allocated from a pool.
pub struct AllocatedCommandBuffer(vk::CommandBuffer);

impl From<vk::CommandBuffer> for AllocatedCommandBuffer {
    #[inline]
    fn from(cmd: vk::CommandBuffer) -> AllocatedCommandBuffer {
        AllocatedCommandBuffer(cmd)
    }
}

unsafe impl VulkanObject for AllocatedCommandBuffer {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.0
    }
}
