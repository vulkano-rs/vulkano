// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Traits and types for managing the allocation of command buffers and command pools.
//!
//! In Vulkano, creating a command buffer requires passing an implementation of the
//! [`CommandBufferAllocator`] trait. You can implement this trait yourself, or use the
//! Vulkano-provided [`StandardCommandBufferAllocator`].

use super::{
    pool::{
        CommandBufferAllocateInfo, CommandPool, CommandPoolAlloc, CommandPoolCreateInfo,
        CommandPoolResetFlags,
    },
    CommandBufferLevel,
};
use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    Validated, VulkanError,
};
use crossbeam_queue::ArrayQueue;
use smallvec::{IntoIter, SmallVec};
use std::{
    cell::{Cell, UnsafeCell},
    error::Error,
    fmt::Display,
    marker::PhantomData,
    mem::ManuallyDrop,
    sync::Arc,
    thread,
};
use thread_local::ThreadLocal;

const MAX_POOLS: usize = 32;

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
    ) -> Result<Self::Iter, VulkanError>;
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
/// The intended way to use this allocator is to have one that is used globally for the duration of
/// the program, in order to avoid creating and destroying [`CommandPool`]s, as that is expensive.
/// Alternatively, you can have one locally on a thread for the duration of the thread.
///
/// Internally, this allocator keeps one or more `CommandPool`s per queue family index per thread,
/// using Thread-Local Storage. When a thread first allocates, an entry is reserved for the thread
/// and queue family combination. After a thread exits and the allocator wasn't dropped yet, its
/// entries are freed, but the pools it used are not dropped. The next time a new thread allocates
/// for the first time, the entries are reused along with the pools. If all threads drop their
/// reference to the allocator, all entries along with the allocator are dropped, even if the
/// threads didn't exit yet, which is why you should keep the allocator alive for as long as you
/// need to allocate so that the pools can keep being reused.
///
/// This allocator only needs to lock when a thread first allocates or when a thread that
/// previously allocated exits. In all other cases, allocation is lock-free.
///
/// Command buffers can't be moved between threads during the building process, but finished command
/// buffers can. When a command buffer is dropped, it is returned back to the pool for reuse.
#[derive(Debug)]
pub struct StandardCommandBufferAllocator {
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    // Each queue family index points directly to its entry.
    pools: ThreadLocal<SmallVec<[UnsafeCell<Option<Entry>>; 8]>>,
    create_info: StandardCommandBufferAllocatorCreateInfo,
}

impl StandardCommandBufferAllocator {
    /// Creates a new `StandardCommandBufferAllocator`.
    #[inline]
    pub fn new(device: Arc<Device>, create_info: StandardCommandBufferAllocatorCreateInfo) -> Self {
        StandardCommandBufferAllocator {
            device: InstanceOwnedDebugWrapper(device),
            pools: ThreadLocal::new(),
            create_info,
        }
    }

    /// Tries to reset the [`CommandPool`] that's currently in use for the given queue family index
    /// on the current thread.
    ///
    /// If successful, the memory of the pool can be reused again along with all command buffers
    /// allocated from it. This is only possible if all command buffers allocated from the pool
    /// have been dropped.
    ///
    /// This has no effect if the entry wasn't initialized yet or if the entry was [cleared].
    ///
    /// # Panics
    ///
    /// - Panics if `queue_family_index` is not less than the number of queue families.
    ///
    /// [cleared]: Self::clear
    #[inline]
    pub fn try_reset_pool(
        &self,
        queue_family_index: u32,
        flags: CommandPoolResetFlags,
    ) -> Result<(), Validated<ResetCommandPoolError>> {
        if let Some(entry) = unsafe { &mut *self.entry(queue_family_index) }.as_mut() {
            entry.try_reset_pool(flags)
        } else {
            Ok(())
        }
    }

    /// Clears the entry for the given queue family index and the current thread. This does not
    /// mean that the pools are dropped immediately. A pool is kept alive for as long as command
    /// buffers allocated from it exist.
    ///
    /// This has no effect if the entry was not initialized yet.
    ///
    /// # Panics
    ///
    /// - Panics if `queue_family_index` is not less than the number of queue families.
    #[inline]
    pub fn clear(&self, queue_family_index: u32) {
        unsafe { *self.entry(queue_family_index) = None };
    }

    fn entry(&self, queue_family_index: u32) -> *mut Option<Entry> {
        let pools = self.pools.get_or(|| {
            self.device
                .physical_device()
                .queue_family_properties()
                .iter()
                .map(|_| UnsafeCell::new(None))
                .collect()
        });

        pools[queue_family_index as usize].get()
    }
}

unsafe impl CommandBufferAllocator for StandardCommandBufferAllocator {
    type Iter = IntoIter<[StandardCommandBufferBuilderAlloc; 1]>;

    type Builder = StandardCommandBufferBuilderAlloc;

    type Alloc = StandardCommandBufferAlloc;

    /// Allocates command buffers.
    ///
    /// Returns an iterator that contains the requested amount of allocated command buffers.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family index is not active on the device.
    /// - Panics if `command_buffer_count` exceeds the count configured for the pool corresponding
    ///   to `level`.
    #[inline]
    fn allocate(
        &self,
        queue_family_index: u32,
        level: CommandBufferLevel,
        command_buffer_count: u32,
    ) -> Result<Self::Iter, VulkanError> {
        // VUID-vkCreateCommandPool-queueFamilyIndex-01937
        assert!(self
            .device
            .active_queue_family_indices()
            .contains(&queue_family_index));

        let entry = unsafe { &mut *self.entry(queue_family_index) };
        if entry.is_none() {
            let reserve = Arc::new(ArrayQueue::new(MAX_POOLS));
            *entry = Some(Entry {
                pool: Pool::new(
                    self.device.clone(),
                    queue_family_index,
                    reserve.clone(),
                    &self.create_info,
                )?,
                reserve,
            });
        }
        let entry = entry.as_mut().unwrap();

        // First try to allocate from existing command buffers.
        if let Some(allocs) = entry.pool.allocate(level, command_buffer_count) {
            return Ok(allocs);
        }

        // Else try to reset the pool.
        if entry
            .try_reset_pool(CommandPoolResetFlags::empty())
            .is_err()
        {
            // If that fails too try to grab a pool from the reserve.
            entry.pool = if let Some(inner) = entry.reserve.pop() {
                Arc::new(Pool {
                    inner: ManuallyDrop::new(inner),
                    reserve: entry.reserve.clone(),
                })
            } else {
                // Else we are unfortunately forced to create a new pool.
                Pool::new(
                    self.device.clone(),
                    queue_family_index,
                    entry.reserve.clone(),
                    &self.create_info,
                )?
            };
        }

        Ok(entry.pool.allocate(level, command_buffer_count).unwrap())
    }
}

unsafe impl CommandBufferAllocator for Arc<StandardCommandBufferAllocator> {
    type Iter = IntoIter<[StandardCommandBufferBuilderAlloc; 1]>;

    type Builder = StandardCommandBufferBuilderAlloc;

    type Alloc = StandardCommandBufferAlloc;

    #[inline]
    fn allocate(
        &self,
        queue_family_index: u32,
        level: CommandBufferLevel,
        command_buffer_count: u32,
    ) -> Result<Self::Iter, VulkanError> {
        (**self).allocate(queue_family_index, level, command_buffer_count)
    }
}

unsafe impl DeviceOwned for StandardCommandBufferAllocator {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

#[derive(Debug)]
struct Entry {
    // Contains the actual Vulkan command pool that is currently in use.
    pool: Arc<Pool>,
    // When a `Pool` is dropped, it returns itself here for reuse.
    reserve: Arc<ArrayQueue<PoolInner>>,
}

// This is needed because of the blanket impl of `Send` on `Arc<T>`, which requires that `T` is
// `Send + Sync`. `Pool` is `Send + !Sync` because `CommandPool` is `!Sync`. That's fine however
// because we never access the Vulkan command pool concurrently. Same goes for the `Cell`s.
unsafe impl Send for Entry {}

impl Entry {
    fn try_reset_pool(
        &mut self,
        flags: CommandPoolResetFlags,
    ) -> Result<(), Validated<ResetCommandPoolError>> {
        if let Some(pool) = Arc::get_mut(&mut self.pool) {
            unsafe {
                pool.inner.inner.reset(flags).map_err(|err| match err {
                    Validated::Error(err) => {
                        Validated::Error(ResetCommandPoolError::VulkanError(err))
                    }
                    Validated::ValidationError(err) => err.into(),
                })?
            };

            *pool.inner.primary_allocations.get_mut() = 0;
            *pool.inner.secondary_allocations.get_mut() = 0;

            Ok(())
        } else {
            Err(ResetCommandPoolError::InUse.into())
        }
    }
}

#[derive(Debug)]
struct Pool {
    inner: ManuallyDrop<PoolInner>,
    // Where we return the `PoolInner` in our `Drop` impl.
    reserve: Arc<ArrayQueue<PoolInner>>,
}

#[derive(Debug)]
struct PoolInner {
    // The Vulkan pool specific to a device's queue family.
    inner: CommandPool,
    // List of existing primary command buffers that are available for reuse.
    primary_pool: Option<ArrayQueue<CommandPoolAlloc>>,
    // List of existing secondary command buffers that are available for reuse.
    secondary_pool: Option<ArrayQueue<CommandPoolAlloc>>,
    // How many command buffers have been allocated from `self.primary_pool`.
    primary_allocations: Cell<usize>,
    // How many command buffers have been allocated from `self.secondary_pool`.
    secondary_allocations: Cell<usize>,
}

impl Pool {
    fn new(
        device: Arc<Device>,
        queue_family_index: u32,
        reserve: Arc<ArrayQueue<PoolInner>>,
        create_info: &StandardCommandBufferAllocatorCreateInfo,
    ) -> Result<Arc<Self>, VulkanError> {
        let inner = CommandPool::new(
            device,
            CommandPoolCreateInfo {
                queue_family_index,
                ..Default::default()
            },
        )
        .map_err(Validated::unwrap)?;

        let primary_pool = if create_info.primary_buffer_count > 0 {
            let pool = ArrayQueue::new(create_info.primary_buffer_count);

            for alloc in inner.allocate_command_buffers(CommandBufferAllocateInfo {
                level: CommandBufferLevel::Primary,
                command_buffer_count: create_info.primary_buffer_count as u32,
                ..Default::default()
            })? {
                let _ = pool.push(alloc);
            }

            Some(pool)
        } else {
            None
        };

        let secondary_pool = if create_info.secondary_buffer_count > 0 {
            let pool = ArrayQueue::new(create_info.secondary_buffer_count);

            for alloc in inner.allocate_command_buffers(CommandBufferAllocateInfo {
                level: CommandBufferLevel::Secondary,
                command_buffer_count: create_info.secondary_buffer_count as u32,
                ..Default::default()
            })? {
                let _ = pool.push(alloc);
            }

            Some(pool)
        } else {
            None
        };

        Ok(Arc::new(Pool {
            inner: ManuallyDrop::new(PoolInner {
                inner,
                primary_pool,
                secondary_pool,
                primary_allocations: Cell::new(0),
                secondary_allocations: Cell::new(0),
            }),
            reserve,
        }))
    }

    fn allocate(
        self: &Arc<Self>,
        level: CommandBufferLevel,
        command_buffer_count: u32,
    ) -> Option<IntoIter<[StandardCommandBufferBuilderAlloc; 1]>> {
        let command_buffer_count = command_buffer_count as usize;

        match level {
            CommandBufferLevel::Primary => {
                if let Some(pool) = &self.inner.primary_pool {
                    let count = self.inner.primary_allocations.get();
                    if count + command_buffer_count <= pool.capacity() {
                        let mut output = SmallVec::<[_; 1]>::with_capacity(command_buffer_count);
                        for _ in 0..command_buffer_count {
                            output.push(StandardCommandBufferBuilderAlloc {
                                inner: StandardCommandBufferAlloc {
                                    inner: ManuallyDrop::new(pool.pop().unwrap()),
                                    pool: self.clone(),
                                },
                                _marker: PhantomData,
                            });
                        }

                        self.inner
                            .primary_allocations
                            .set(count + command_buffer_count);

                        Some(output.into_iter())
                    } else if command_buffer_count > pool.capacity() {
                        panic!(
                            "command buffer count ({}) exceeds the capacity of the primary command \
                            buffer pool ({})",
                            command_buffer_count, pool.capacity(),
                        );
                    } else {
                        None
                    }
                } else {
                    panic!(
                        "attempted to allocate a primary command buffer when the primary command \
                        buffer pool was configured to be empty",
                    );
                }
            }
            CommandBufferLevel::Secondary => {
                if let Some(pool) = &self.inner.secondary_pool {
                    let count = self.inner.secondary_allocations.get();
                    if count + command_buffer_count <= pool.capacity() {
                        let mut output = SmallVec::<[_; 1]>::with_capacity(command_buffer_count);
                        for _ in 0..command_buffer_count {
                            output.push(StandardCommandBufferBuilderAlloc {
                                inner: StandardCommandBufferAlloc {
                                    inner: ManuallyDrop::new(pool.pop().unwrap()),
                                    pool: self.clone(),
                                },
                                _marker: PhantomData,
                            });
                        }

                        self.inner
                            .secondary_allocations
                            .set(count + command_buffer_count);

                        Some(output.into_iter())
                    } else if command_buffer_count > pool.capacity() {
                        panic!(
                            "command buffer count ({}) exceeds the capacity of the secondary \
                            command buffer pool ({})",
                            command_buffer_count,
                            pool.capacity(),
                        );
                    } else {
                        None
                    }
                } else {
                    panic!(
                        "attempted to allocate a secondary command buffer when the secondary \
                        command buffer pool was configured to be empty",
                    );
                }
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };

        if thread::panicking() {
            return;
        }

        unsafe { inner.inner.reset(CommandPoolResetFlags::empty()) }.unwrap();
        inner.primary_allocations.set(0);
        inner.secondary_allocations.set(0);

        // If there is not enough space in the reserve, we destroy the pool. The only way this can
        // happen is if something is resource hogging, forcing new pools to be created such that
        // the number exceeds `MAX_POOLS`, and then drops them all at once.
        let _ = self.reserve.push(inner);
    }
}

/// Parameters to create a new [`StandardCommandBufferAllocator`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandardCommandBufferAllocatorCreateInfo {
    /// How many primary command buffers should be allocated per pool.
    ///
    /// Each time a thread allocates using some queue family index, and either no pools were
    /// initialized yet or all pools are full, a new pool is created for that thread and queue
    /// family combination. This option tells the allocator how many primary command buffers should
    /// be allocated for that pool. It always allocates exactly this many command buffers at once
    /// for the pool, as that is more performant than allocating them one-by-one. What this means
    /// is that you should make sure that this is not too large, so that you don't end up wasting
    /// too much memory. You also don't want this to be too low, because that on the other hand
    /// would mean that the pool would have to be reset more often, or that more pools would need
    /// to be created, depending on the lifetime of the command buffers.
    ///
    /// The default value is `256`.
    pub primary_buffer_count: usize,

    /// Same as `primary_buffer_count` except for secondary command buffers.
    ///
    /// The default value is `256`.
    pub secondary_buffer_count: usize,

    pub _ne: crate::NonExhaustive,
}

impl Default for StandardCommandBufferAllocatorCreateInfo {
    #[inline]
    fn default() -> Self {
        StandardCommandBufferAllocatorCreateInfo {
            primary_buffer_count: 256,
            secondary_buffer_count: 256,
            _ne: crate::NonExhaustive(()),
        }
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
    _marker: PhantomData<*const ()>,
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
    inner: ManuallyDrop<CommandPoolAlloc>,
    // We hold a reference to the pool for our destructor.
    pool: Arc<Pool>,
}

// It's fine to share `Pool` between threads because we never access the Vulkan command pool
// concurrently. Same goes for the `Cell`s.
unsafe impl Send for StandardCommandBufferAlloc {}
unsafe impl Sync for StandardCommandBufferAlloc {}

unsafe impl CommandBufferAlloc for StandardCommandBufferAlloc {
    #[inline]
    fn inner(&self) -> &CommandPoolAlloc {
        &self.inner
    }

    #[inline]
    fn queue_family_index(&self) -> u32 {
        self.pool.inner.inner.queue_family_index()
    }
}

unsafe impl DeviceOwned for StandardCommandBufferAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.pool.inner.inner.device()
    }
}

impl Drop for StandardCommandBufferAlloc {
    #[inline]
    fn drop(&mut self) {
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };
        let pool = match inner.level() {
            CommandBufferLevel::Primary => &self.pool.inner.primary_pool,
            CommandBufferLevel::Secondary => &self.pool.inner.secondary_pool,
        };
        // This can't panic, because if an allocation from a particular kind of pool was made, then
        // the pool must exist.
        let _ = pool.as_ref().unwrap().push(inner);
    }
}

/// Error that can be returned when resetting a [`CommandPool`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResetCommandPoolError {
    /// A runtime error occurred.
    VulkanError(VulkanError),

    /// The `CommandPool` is still in use.
    InUse,
}

impl Error for ResetCommandPoolError {}

impl Display for ResetCommandPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VulkanError(_) => write!(f, "a runtime error occurred"),
            Self::InUse => write!(f, "the command pool is still in use"),
        }
    }
}

impl From<VulkanError> for ResetCommandPoolError {
    fn from(err: VulkanError) -> Self {
        Self::VulkanError(err)
    }
}

impl From<ResetCommandPoolError> for Validated<ResetCommandPoolError> {
    fn from(err: ResetCommandPoolError) -> Self {
        Self::Error(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VulkanObject;
    use std::thread;

    #[test]
    fn threads_use_different_pools() {
        let (device, queue) = gfx_dev_and_queue!();

        let allocator = StandardCommandBufferAllocator::new(device, Default::default());

        let pool1 = allocator
            .allocate(queue.queue_family_index(), CommandBufferLevel::Primary, 1)
            .unwrap()
            .next()
            .unwrap()
            .into_alloc()
            .pool
            .inner
            .inner
            .handle();

        thread::spawn(move || {
            let pool2 = allocator
                .allocate(queue.queue_family_index(), CommandBufferLevel::Primary, 1)
                .unwrap()
                .next()
                .unwrap()
                .into_alloc()
                .pool
                .inner
                .inner
                .handle();
            assert_ne!(pool1, pool2);
        })
        .join()
        .unwrap();
    }
}
