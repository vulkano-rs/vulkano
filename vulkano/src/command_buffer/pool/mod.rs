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

use OomError;
use device::DeviceOwned;

pub use self::standard::StandardCommandPool;
pub use self::sys::CommandPoolTrimError;
pub use self::sys::UnsafeCommandPool;
pub use self::sys::UnsafeCommandPoolAlloc;
pub use self::sys::UnsafeCommandPoolAllocIter;

pub mod standard;
mod sys;

/// Types that manage the memory of command buffers.
///
/// # Safety
///
/// A Vulkan command pool must be externally synchronized as if it owned the command buffers that
/// were allocated from it. This includes allocating from the pool, freeing from the pool,
/// resetting the pool or individual command buffers, and most importantly recording commands to
/// command buffers.
///
/// The implementation of `CommandPool` is expected to manage this. For as long as a `Builder`
/// is alive, the trait implementation is expected to lock the pool that allocated the `Builder`
/// for the current thread.
///
/// > **Note**: This may be modified in the future to allow different implementation strategies.
///
/// The destructors of the `CommandPoolBuilderAlloc` and the `CommandPoolAlloc` are expected to
/// free the command buffer, reset the command buffer, or add it to a pool so that it gets reused.
/// If the implementation frees or resets the command buffer, it must not forget that this
/// operation must lock the pool.
///
pub unsafe trait CommandPool: DeviceOwned {
    /// See `alloc()`.
    type Iter: Iterator<Item = Self::Builder>;
    /// Represents a command buffer that has been allocated and that is currently being built.
    type Builder: CommandPoolBuilderAlloc<Alloc = Self::Alloc>;
    /// Represents a command buffer that has been allocated and that is pending execution or is
    /// being executed.
    type Alloc: CommandPoolAlloc;

    /// Allocates command buffers from this pool.
    ///
    /// Returns an iterator that contains an bunch of allocated command buffers.
    fn alloc(&self, secondary: bool, count: u32) -> Result<Self::Iter, OomError>;

    /// Returns the queue family that this pool targets.
    fn queue_family(&self) -> QueueFamily;
}

/// A command buffer allocated from a pool and that can be recorded.
///
/// # Safety
///
/// See `CommandPool` for information about safety.
///
pub unsafe trait CommandPoolBuilderAlloc: DeviceOwned {
    /// Return type of `into_alloc`.
    type Alloc: CommandPoolAlloc;

    /// Returns the internal object that contains the command buffer.
    fn inner(&self) -> &UnsafeCommandPoolAlloc;

    /// Turns this builder into a command buffer that is pending execution.
    fn into_alloc(self) -> Self::Alloc;

    /// Returns the queue family that the pool targets.
    fn queue_family(&self) -> QueueFamily;
}

/// A command buffer allocated from a pool that has finished being recorded.
///
/// # Safety
///
/// See `CommandPool` for information about safety.
///
pub unsafe trait CommandPoolAlloc: DeviceOwned {
    /// Returns the internal object that contains the command buffer.
    fn inner(&self) -> &UnsafeCommandPoolAlloc;

    /// Returns the queue family that the pool targets.
    fn queue_family(&self) -> QueueFamily;
}
