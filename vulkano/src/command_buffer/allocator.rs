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

use super::{
    pool::{
        CommandBufferAllocateInfo, CommandPool, CommandPoolAlloc, CommandPoolCreateInfo,
        CommandPoolCreationError,
    },
    CommandBufferLevel,
};
use crate::{
    device::{Device, DeviceOwned},
    OomError,
};
use crossbeam_queue::SegQueue;
use smallvec::SmallVec;
use std::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop, sync::Arc, vec::IntoIter};

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
        queue_family_index: u32,
        level: CommandBufferLevel,
        command_buffer_count: u32,
    ) -> Result<Self::Iter, OomError>;
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

/// Standard implementation of a command buffer allocator.
///
/// A thread can have as many `StandardCommandBufferAllocator`s as needed, but they can't be shared
/// between threads. This is done so that there are no locks involved when creating command
/// buffers. You are encouraged to create one allocator per frame in flight per thread.
///
/// Command buffers can't be moved between threads during the building process, but finished command
/// buffers can. When a command buffer is dropped, it is returned back to the pool for reuse.
#[derive(Debug)]
pub struct StandardCommandBufferAllocator {
    device: Arc<Device>,
    /// Each queue family index points directly to its pool.
    pools: SmallVec<[UnsafeCell<Option<Arc<Pool>>>; 8]>,
}

impl StandardCommandBufferAllocator {
    /// Creates a new `StandardCommandBufferAllocator`.
    #[inline]
    pub fn new(device: Arc<Device>) -> Self {
        let pools = device
            .physical_device()
            .queue_family_properties()
            .iter()
            .map(|_| UnsafeCell::new(None))
            .collect();

        StandardCommandBufferAllocator { device, pools }
    }
}

unsafe impl CommandBufferAllocator for StandardCommandBufferAllocator {
    type Iter = IntoIter<StandardCommandBufferBuilderAlloc>;

    type Builder = StandardCommandBufferBuilderAlloc;

    type Alloc = StandardCommandBufferAlloc;

    /// # Panics
    ///
    /// - Panics if the queue family index is not active on the device.
    #[inline]
    fn allocate(
        &self,
        queue_family_index: u32,
        level: CommandBufferLevel,
        command_buffer_count: u32,
    ) -> Result<Self::Iter, OomError> {
        // VUID-vkCreateCommandPool-queueFamilyIndex-01937
        assert!(self
            .device
            .active_queue_family_indices()
            .contains(&queue_family_index));

        let pool = unsafe { &mut *self.pools[queue_family_index as usize].get() };
        if pool.is_none() {
            *pool = Some(Pool::new(self.device.clone(), queue_family_index)?);
        }

        pool.as_ref().unwrap().allocate(level, command_buffer_count)
    }
}

unsafe impl DeviceOwned for StandardCommandBufferAllocator {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

#[derive(Debug)]
struct Pool {
    // The Vulkan pool specific to a device's queue family.
    inner: CommandPool,
    // List of existing primary command buffers that are available for reuse.
    primary_pool: SegQueue<CommandPoolAlloc>,
    // List of existing secondary command buffers that are available for reuse.
    secondary_pool: SegQueue<CommandPoolAlloc>,
}

impl Pool {
    fn new(device: Arc<Device>, queue_family_index: u32) -> Result<Arc<Self>, OomError> {
        CommandPool::new(
            device,
            CommandPoolCreateInfo {
                queue_family_index,
                reset_command_buffer: true,
                ..Default::default()
            },
        )
        .map(|inner| {
            Arc::new(Pool {
                inner,
                primary_pool: Default::default(),
                secondary_pool: Default::default(),
            })
        })
        .map_err(|err| match err {
            CommandPoolCreationError::OomError(err) => err,
            // We check that the provided queue family index is active on the device, so it can't
            // be out of range.
            CommandPoolCreationError::QueueFamilyIndexOutOfRange { .. } => unreachable!(),
        })
    }

    fn allocate(
        self: &Arc<Self>,
        level: CommandBufferLevel,
        mut command_buffer_count: u32,
    ) -> Result<IntoIter<StandardCommandBufferBuilderAlloc>, OomError> {
        // The final output.
        let mut output = Vec::with_capacity(command_buffer_count as usize);

        // First, pick from already-existing command buffers.
        {
            let existing = match level {
                CommandBufferLevel::Primary => &self.primary_pool,
                CommandBufferLevel::Secondary => &self.secondary_pool,
            };

            for _ in 0..command_buffer_count as usize {
                if let Some(cmd) = existing.pop() {
                    output.push(StandardCommandBufferBuilderAlloc {
                        inner: StandardCommandBufferAlloc {
                            cmd: ManuallyDrop::new(cmd),
                            pool: self.clone(),
                        },
                        dummy_avoid_send_sync: PhantomData,
                    });
                } else {
                    break;
                }
            }
        }

        // Then allocate the rest.
        if output.len() < command_buffer_count as usize {
            command_buffer_count -= output.len() as u32;

            for cmd in self
                .inner
                .allocate_command_buffers(CommandBufferAllocateInfo {
                    level,
                    command_buffer_count,
                    ..Default::default()
                })?
            {
                output.push(StandardCommandBufferBuilderAlloc {
                    inner: StandardCommandBufferAlloc {
                        cmd: ManuallyDrop::new(cmd),
                        pool: self.clone(),
                    },
                    dummy_avoid_send_sync: PhantomData,
                });
            }
        }

        // Final output.
        Ok(output.into_iter())
    }
}

/// Command buffer allocated from a [`StandardCommandBufferAllocator`] that is currently being
/// built.
pub struct StandardCommandBufferBuilderAlloc {
    // The only difference between a `StandardCommandBufferBuilder` and a
    // `StandardCommandBufferAlloc` is that the former must not implement `Send` and `Sync`.
    // Therefore we just share the structs.
    inner: StandardCommandBufferAlloc,
    // Unimplemented `Send` and `Sync` from the builder.
    dummy_avoid_send_sync: PhantomData<*const u8>,
}

unsafe impl CommandBufferBuilderAlloc for StandardCommandBufferBuilderAlloc {
    type Alloc = StandardCommandBufferAlloc;

    #[inline]
    fn inner(&self) -> &CommandPoolAlloc {
        self.inner.inner()
    }

    #[inline]
    fn into_alloc(self) -> Self::Alloc {
        self.inner
    }

    #[inline]
    fn queue_family_index(&self) -> u32 {
        self.inner.queue_family_index()
    }
}

unsafe impl DeviceOwned for StandardCommandBufferBuilderAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

/// Command buffer allocated from a [`StandardCommandBufferAllocator`].
pub struct StandardCommandBufferAlloc {
    // The actual command buffer. Extracted in the `Drop` implementation.
    cmd: ManuallyDrop<CommandPoolAlloc>,
    // We hold a reference to the command pool for our destructor.
    pool: Arc<Pool>,
}

unsafe impl Send for StandardCommandBufferAlloc {}
unsafe impl Sync for StandardCommandBufferAlloc {}

unsafe impl CommandBufferAlloc for StandardCommandBufferAlloc {
    #[inline]
    fn inner(&self) -> &CommandPoolAlloc {
        &self.cmd
    }

    #[inline]
    fn queue_family_index(&self) -> u32 {
        self.pool.inner.queue_family_index()
    }
}

unsafe impl DeviceOwned for StandardCommandBufferAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.pool.inner.device()
    }
}

impl Drop for StandardCommandBufferAlloc {
    #[inline]
    fn drop(&mut self) {
        let cmd = unsafe { ManuallyDrop::take(&mut self.cmd) };

        match cmd.level() {
            CommandBufferLevel::Primary => self.pool.primary_pool.push(cmd),
            CommandBufferLevel::Secondary => self.pool.secondary_pool.push(cmd),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CommandBufferAllocator, CommandBufferBuilderAlloc, StandardCommandBufferAllocator,
    };
    use crate::{command_buffer::CommandBufferLevel, VulkanObject};

    #[test]
    fn reuse_command_buffers() {
        let (device, queue) = gfx_dev_and_queue!();

        let allocator = StandardCommandBufferAllocator::new(device);

        let cb = allocator
            .allocate(queue.queue_family_index(), CommandBufferLevel::Primary, 1)
            .unwrap()
            .next()
            .unwrap();
        let raw = cb.inner().internal_object();
        drop(cb);

        let cb2 = allocator
            .allocate(queue.queue_family_index(), CommandBufferLevel::Primary, 1)
            .unwrap()
            .next()
            .unwrap();
        assert_eq!(raw, cb2.inner().internal_object());
    }
}
