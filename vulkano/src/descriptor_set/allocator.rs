// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! In the Vulkan API, descriptor sets must be allocated from *descriptor pools*.
//!
//! A descriptor pool holds and manages the memory of one or more descriptor sets. If you destroy a
//! descriptor pool, all of its descriptor sets are automatically destroyed.
//!
//! In vulkano, creating a descriptor set requires passing an implementation of the
//! [`DescriptorSetAllocator`] trait, which you can implement yourself or use the vulkano-provided
//! [`StandardDescriptorSetAllocator`].

use self::sorted_map::SortedMap;
use super::{
    layout::DescriptorSetLayout,
    pool::{DescriptorPool, DescriptorPoolCreateInfo, DescriptorSetAllocateInfo},
    sys::UnsafeDescriptorSet,
};
use crate::{
    descriptor_set::layout::DescriptorType,
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    Validated, VulkanError,
};
use crossbeam_queue::ArrayQueue;
use std::{cell::UnsafeCell, mem::ManuallyDrop, num::NonZeroU64, sync::Arc, thread};
use thread_local::ThreadLocal;

const MAX_POOLS: usize = 32;

const MAX_SETS: usize = 256;

/// Types that manage the memory of descriptor sets.
///
/// # Safety
///
/// A Vulkan descriptor pool must be externally synchronized as if it owned the descriptor sets that
/// were allocated from it. This includes allocating from the pool, freeing from the pool and
/// resetting the pool or individual descriptor sets. The implementation of `DescriptorSetAllocator`
/// is expected to manage this.
///
/// The destructor of the [`DescriptorSetAlloc`] is expected to free the descriptor set, reset the
/// descriptor set, or add it to a pool so that it gets reused. If the implementation frees or
/// resets the descriptor set, it must not forget that this operation must be externally
/// synchronized.
pub unsafe trait DescriptorSetAllocator: DeviceOwned {
    /// Object that represented an allocated descriptor set.
    ///
    /// The destructor of this object should free the descriptor set.
    type Alloc: DescriptorSetAlloc;

    /// Allocates a descriptor set.
    fn allocate(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<Self::Alloc, Validated<VulkanError>>;
}

/// An allocated descriptor set.
pub trait DescriptorSetAlloc: Send + Sync {
    /// Returns the inner unsafe descriptor set object.
    fn inner(&self) -> &UnsafeDescriptorSet;

    /// Returns the inner unsafe descriptor set object.
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet;

    /// Returns the descriptor pool that the descriptor set was allocated from.
    fn pool(&self) -> &DescriptorPool;
}

/// Standard implementation of a descriptor set allocator.
///
/// The intended way to use this allocator is to have one that is used globally for the duration of
/// the program, in order to avoid creating and destroying [`DescriptorPool`]s, as that is
/// expensive. Alternatively, you can have one locally on a thread for the duration of the thread.
///
/// Internally, this allocator uses one or more `DescriptorPool`s per descriptor set layout per
/// thread, using Thread-Local Storage. When a thread first allocates, an entry is reserved for the
/// thread and descriptor set layout combination. After a thread exits and the allocator wasn't
/// dropped yet, its entries are freed, but the pools it used are not dropped. The next time a new
/// thread allocates for the first time, the entries are reused along with the pools. If all
/// threads drop their reference to the allocator, all entries along with the allocator are
/// dropped, even if the threads didn't exit yet, which is why you should keep the allocator alive
/// for as long as you need to allocate so that the pools can keep being reused.
///
/// This allocator only needs to lock when a thread first allocates or when a thread that
/// previously allocated exits. In all other cases, allocation is lock-free.
///
/// [`DescriptorPool`]: crate::descriptor_set::pool::DescriptorPool
#[derive(Debug)]
pub struct StandardDescriptorSetAllocator {
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    pools: ThreadLocal<UnsafeCell<SortedMap<NonZeroU64, Entry>>>,
}

#[derive(Debug)]
enum Entry {
    Fixed(FixedEntry),
    Variable(VariableEntry),
}

// This is needed because of the blanket impl of `Send` on `Arc<T>`, which requires that `T` is
// `Send + Sync`. `FixedPool` and `VariablePool` are `Send + !Sync` because `DescriptorPool` is
// `!Sync`. That's fine however because we never access the `DescriptorPool` concurrently.
unsafe impl Send for Entry {}

impl StandardDescriptorSetAllocator {
    /// Creates a new `StandardDescriptorSetAllocator`.
    #[inline]
    pub fn new(device: Arc<Device>) -> StandardDescriptorSetAllocator {
        StandardDescriptorSetAllocator {
            device: InstanceOwnedDebugWrapper(device),
            pools: ThreadLocal::new(),
        }
    }

    /// Clears the entry for the given descriptor set layout and the current thread. This does not
    /// mean that the pools are dropped immediately. A pool is kept alive for as long as descriptor
    /// sets allocated from it exist.
    ///
    /// This has no effect if the entry was not initialized yet.
    #[inline]
    pub fn clear(&self, layout: &Arc<DescriptorSetLayout>) {
        unsafe { &mut *self.pools.get_or(Default::default).get() }.remove(layout.id())
    }

    /// Clears all entries for the current thread. This does not mean that the pools are dropped
    /// immediately. A pool is kept alive for as long as descriptor sets allocated from it exist.
    ///
    /// This has no effect if no entries were initialized yet.
    #[inline]
    pub fn clear_all(&self) {
        unsafe { *self.pools.get_or(Default::default).get() = SortedMap::default() };
    }
}

unsafe impl DescriptorSetAllocator for StandardDescriptorSetAllocator {
    type Alloc = StandardDescriptorSetAlloc;

    /// Allocates a descriptor set.
    #[inline]
    fn allocate(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<StandardDescriptorSetAlloc, Validated<VulkanError>> {
        let max_count = layout.variable_descriptor_count();
        let pools = self.pools.get_or(Default::default);

        let entry = unsafe { &mut *pools.get() }.get_or_try_insert(layout.id(), || {
            if max_count == 0 {
                FixedEntry::new(layout.clone()).map(Entry::Fixed)
            } else {
                VariableEntry::new(layout.clone()).map(Entry::Variable)
            }
        })?;

        match entry {
            Entry::Fixed(entry) => entry.allocate(),
            Entry::Variable(entry) => entry.allocate(variable_descriptor_count),
        }
    }
}

unsafe impl<T: DescriptorSetAllocator> DescriptorSetAllocator for Arc<T> {
    type Alloc = T::Alloc;

    #[inline]
    fn allocate(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<Self::Alloc, Validated<VulkanError>> {
        (**self).allocate(layout, variable_descriptor_count)
    }
}

unsafe impl DeviceOwned for StandardDescriptorSetAllocator {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

#[derive(Debug)]
struct FixedEntry {
    // The `FixedPool` struct contains an actual Vulkan pool. Every time it is full we create
    // a new pool and replace the current one with the new one.
    pool: Arc<FixedPool>,
    // The amount of sets available to use when we create a new Vulkan pool.
    set_count: usize,
    // The descriptor set layout that this pool is for.
    layout: Arc<DescriptorSetLayout>,
}

impl FixedEntry {
    fn new(layout: Arc<DescriptorSetLayout>) -> Result<Self, Validated<VulkanError>> {
        Ok(FixedEntry {
            pool: FixedPool::new(&layout, MAX_SETS)?,
            set_count: MAX_SETS,
            layout,
        })
    }

    fn allocate(&mut self) -> Result<StandardDescriptorSetAlloc, Validated<VulkanError>> {
        let inner = if let Some(inner) = self.pool.reserve.pop() {
            inner
        } else {
            self.set_count *= 2;
            self.pool = FixedPool::new(&self.layout, self.set_count)?;

            self.pool.reserve.pop().unwrap()
        };

        Ok(StandardDescriptorSetAlloc {
            inner: ManuallyDrop::new(inner),
            parent: AllocParent::Fixed(self.pool.clone()),
        })
    }
}

#[derive(Debug)]
struct FixedPool {
    // The actual Vulkan descriptor pool. This field isn't actually used anywhere, but we need to
    // keep the pool alive in order to keep the descriptor sets valid.
    inner: DescriptorPool,
    // List of descriptor sets. When `alloc` is called, a descriptor will be extracted from this
    // list. When a `SingleLayoutPoolAlloc` is dropped, its descriptor set is put back in this list.
    reserve: ArrayQueue<UnsafeDescriptorSet>,
}

impl FixedPool {
    fn new(
        layout: &Arc<DescriptorSetLayout>,
        set_count: usize,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        let inner = DescriptorPool::new(
            layout.device().clone(),
            DescriptorPoolCreateInfo {
                max_sets: set_count as u32,
                pool_sizes: layout
                    .descriptor_counts()
                    .iter()
                    .map(|(&ty, &count)| {
                        assert!(ty != DescriptorType::InlineUniformBlock);
                        (ty, count * set_count as u32)
                    })
                    .collect(),
                ..Default::default()
            },
        )
        .map_err(Validated::unwrap)?;

        let allocate_infos = (0..set_count).map(|_| DescriptorSetAllocateInfo::new(layout));

        let allocs = unsafe {
            inner
                .allocate_descriptor_sets(allocate_infos)
                .map_err(|err| match err {
                    Validated::ValidationError(_) => err,
                    Validated::Error(vk_err) => match vk_err {
                        VulkanError::OutOfHostMemory | VulkanError::OutOfDeviceMemory => err,
                        VulkanError::FragmentedPool => {
                            // This can't happen as we don't free individual sets.
                            unreachable!();
                        }
                        VulkanError::OutOfPoolMemory => {
                            // We created the pool with an exact size.
                            unreachable!();
                        }
                        _ => {
                            // Shouldn't ever be returned.
                            unreachable!();
                        }
                    },
                })?
        };

        let reserve = ArrayQueue::new(set_count);
        for alloc in allocs {
            let _ = reserve.push(alloc);
        }

        Ok(Arc::new(FixedPool { inner, reserve }))
    }
}

#[derive(Debug)]
struct VariableEntry {
    // The `VariablePool` struct contains an actual Vulkan pool. Every time it is full
    // we grab one from the reserve, or create a new pool if there are none.
    pool: Arc<VariablePool>,
    // When a `VariablePool` is dropped, it returns its Vulkan pool here for reuse.
    reserve: Arc<ArrayQueue<DescriptorPool>>,
    // The descriptor set layout that this pool is for.
    layout: Arc<DescriptorSetLayout>,
    // The number of sets currently allocated from the Vulkan pool.
    allocations: usize,
}

impl VariableEntry {
    fn new(layout: Arc<DescriptorSetLayout>) -> Result<Self, Validated<VulkanError>> {
        let reserve = Arc::new(ArrayQueue::new(MAX_POOLS));

        Ok(VariableEntry {
            pool: VariablePool::new(&layout, reserve.clone())?,
            reserve,
            layout,
            allocations: 0,
        })
    }

    fn allocate(
        &mut self,
        variable_descriptor_count: u32,
    ) -> Result<StandardDescriptorSetAlloc, Validated<VulkanError>> {
        if self.allocations >= MAX_SETS {
            self.pool = if let Some(inner) = self.reserve.pop() {
                Arc::new(VariablePool {
                    inner: ManuallyDrop::new(inner),
                    reserve: self.reserve.clone(),
                })
            } else {
                VariablePool::new(&self.layout, self.reserve.clone())?
            };
            self.allocations = 0;
        }

        let allocate_info = DescriptorSetAllocateInfo {
            variable_descriptor_count,
            ..DescriptorSetAllocateInfo::new(&self.layout)
        };

        let mut sets = unsafe {
            self.pool
                .inner
                .allocate_descriptor_sets([allocate_info])
                .map_err(|err| match err {
                    Validated::ValidationError(_) => err,
                    Validated::Error(vk_err) => match vk_err {
                        VulkanError::OutOfHostMemory | VulkanError::OutOfDeviceMemory => err,
                        VulkanError::FragmentedPool => {
                            // This can't happen as we don't free individual sets.
                            unreachable!();
                        }
                        VulkanError::OutOfPoolMemory => {
                            // We created the pool to fit the maximum variable descriptor count.
                            unreachable!();
                        }
                        _ => {
                            // Shouldn't ever be returned.
                            unreachable!();
                        }
                    },
                })?
        };
        self.allocations += 1;

        Ok(StandardDescriptorSetAlloc {
            inner: ManuallyDrop::new(sets.next().unwrap()),
            parent: AllocParent::Variable(self.pool.clone()),
        })
    }
}

#[derive(Debug)]
struct VariablePool {
    // The actual Vulkan descriptor pool.
    inner: ManuallyDrop<DescriptorPool>,
    // Where we return the Vulkan descriptor pool in our `Drop` impl.
    reserve: Arc<ArrayQueue<DescriptorPool>>,
}

impl VariablePool {
    fn new(
        layout: &Arc<DescriptorSetLayout>,
        reserve: Arc<ArrayQueue<DescriptorPool>>,
    ) -> Result<Arc<Self>, VulkanError> {
        DescriptorPool::new(
            layout.device().clone(),
            DescriptorPoolCreateInfo {
                max_sets: MAX_SETS as u32,
                pool_sizes: layout
                    .descriptor_counts()
                    .iter()
                    .map(|(&ty, &count)| {
                        assert!(ty != DescriptorType::InlineUniformBlock);
                        (ty, count * MAX_SETS as u32)
                    })
                    .collect(),
                ..Default::default()
            },
        )
        .map(|inner| {
            Arc::new(Self {
                inner: ManuallyDrop::new(inner),
                reserve,
            })
        })
        .map_err(Validated::unwrap)
    }
}

impl Drop for VariablePool {
    fn drop(&mut self) {
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };

        if thread::panicking() {
            return;
        }

        unsafe { inner.reset() }.unwrap();

        // If there is not enough space in the reserve, we destroy the pool. The only way this can
        // happen is if something is resource hogging, forcing new pools to be created such that
        // the number exceeds `MAX_POOLS`, and then drops them all at once.
        let _ = self.reserve.push(inner);
    }
}

/// A descriptor set allocated from a [`StandardDescriptorSetAllocator`].
#[derive(Debug)]
pub struct StandardDescriptorSetAlloc {
    // The actual descriptor set.
    inner: ManuallyDrop<UnsafeDescriptorSet>,
    // The pool where we allocated from. Needed for our `Drop` impl.
    parent: AllocParent,
}

#[derive(Debug)]
enum AllocParent {
    Fixed(Arc<FixedPool>),
    Variable(Arc<VariablePool>),
}

impl AllocParent {
    #[inline]
    fn pool(&self) -> &DescriptorPool {
        match self {
            Self::Fixed(pool) => &pool.inner,
            Self::Variable(pool) => &pool.inner,
        }
    }
}

// This is needed because of the blanket impl of `Send` on `Arc<T>`, which requires that `T` is
// `Send + Sync`. `FixedPool` and `VariablePool` are `Send + !Sync` because `DescriptorPool` is
// `!Sync`. That's fine however because we never access the `DescriptorPool` concurrently.
unsafe impl Send for StandardDescriptorSetAlloc {}
unsafe impl Sync for StandardDescriptorSetAlloc {}

impl DescriptorSetAlloc for StandardDescriptorSetAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        &self.inner
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet {
        &mut self.inner
    }

    #[inline]
    fn pool(&self) -> &DescriptorPool {
        self.parent.pool()
    }
}

impl Drop for StandardDescriptorSetAlloc {
    #[inline]
    fn drop(&mut self) {
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };

        match &self.parent {
            AllocParent::Fixed(pool) => {
                let _ = pool.reserve.push(inner);
            }
            AllocParent::Variable(_) => {}
        }
    }
}

mod sorted_map {
    use smallvec::SmallVec;

    /// Minimal implementation of a `SortedMap`. This outperforms both a [`BTreeMap`] and
    /// [`HashMap`] for small numbers of elements. In Vulkan, having too many descriptor set
    /// layouts is highly discouraged, which is why this optimization makes sense.
    #[derive(Debug)]
    pub(super) struct SortedMap<K, V> {
        inner: SmallVec<[(K, V); 8]>,
    }

    impl<K, V> Default for SortedMap<K, V> {
        fn default() -> Self {
            Self {
                inner: SmallVec::default(),
            }
        }
    }

    impl<K: Ord + Copy, V> SortedMap<K, V> {
        pub fn get_or_try_insert<E>(
            &mut self,
            key: K,
            f: impl FnOnce() -> Result<V, E>,
        ) -> Result<&mut V, E> {
            match self.inner.binary_search_by_key(&key, |&(k, _)| k) {
                Ok(index) => Ok(&mut self.inner[index].1),
                Err(index) => {
                    self.inner.insert(index, (key, f()?));
                    Ok(&mut self.inner[index].1)
                }
            }
        }

        pub fn remove(&mut self, key: K) {
            if let Ok(index) = self.inner.binary_search_by_key(&key, |&(k, _)| k) {
                self.inner.remove(index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        descriptor_set::layout::{
            DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
        },
        shader::ShaderStages,
        VulkanObject,
    };
    use std::thread;

    #[test]
    fn threads_use_different_pools() {
        let (device, _) = gfx_dev_and_queue!();

        let layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: [(
                    0,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::all_graphics(),
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                    },
                )]
                .into(),
                ..Default::default()
            },
        )
        .unwrap();

        let allocator = StandardDescriptorSetAllocator::new(device);

        let pool1 =
            if let AllocParent::Fixed(pool) = &allocator.allocate(&layout, 0).unwrap().parent {
                pool.inner.handle()
            } else {
                unreachable!()
            };

        thread::spawn(move || {
            let pool2 =
                if let AllocParent::Fixed(pool) = &allocator.allocate(&layout, 0).unwrap().parent {
                    pool.inner.handle()
                } else {
                    unreachable!()
                };
            assert_ne!(pool1, pool2);
        })
        .join()
        .unwrap();
    }
}
