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
    pool::{
        DescriptorPool, DescriptorPoolAlloc, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo,
        DescriptorSetAllocateInfo,
    },
};
use crate::{
    descriptor_set::layout::DescriptorType,
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    Validated, VulkanError,
};
use crossbeam_queue::ArrayQueue;
use std::{
    cell::UnsafeCell,
    fmt::{Debug, Error as FmtError, Formatter},
    mem,
    num::NonZeroU64,
    ptr,
    sync::Arc,
    thread,
};
use thread_local::ThreadLocal;

const MAX_POOLS: usize = 32;

/// Types that manage the memory of descriptor sets.
///
/// # Safety
///
/// A Vulkan descriptor pool must be externally synchronized as if it owned the descriptor sets
/// that were allocated from it. This includes allocating from the pool, freeing from the pool and
/// resetting the pool or individual descriptor sets. The implementation of
/// `DescriptorSetAllocator` is expected to manage this.
///
/// The destructor of the [`DescriptorSetAlloc`] is expected to free the descriptor set, reset the
/// descriptor set, or add it to a pool so that it gets reused. If the implementation frees or
/// resets the descriptor set, it must not forget that this operation must be externally
/// synchronized.
pub unsafe trait DescriptorSetAllocator: DeviceOwned + Send + Sync + 'static {
    /// Allocates a descriptor set.
    fn allocate(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<DescriptorSetAlloc, Validated<VulkanError>>;

    /// Deallocates the given `allocation`.
    ///
    /// # Safety
    ///
    /// - `allocation` must refer to a **currently allocated** allocation of `self`.
    unsafe fn deallocate(&self, allocation: DescriptorSetAlloc);
}

impl Debug for dyn DescriptorSetAllocator {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("DescriptorSetAllocator")
            .finish_non_exhaustive()
    }
}

/// An allocation made using a [descriptor set allocator].
///
/// [descriptor set allocator]: DescriptorSetAllocator
#[derive(Debug)]
pub struct DescriptorSetAlloc {
    /// The internal object that contains the descriptor set.
    pub inner: DescriptorPoolAlloc,

    /// The descriptor pool that the descriptor set was allocated from.
    pub pool: Arc<DescriptorPool>,

    /// An opaque handle identifying the allocation inside the allocator.
    pub handle: AllocationHandle,
}

unsafe impl Send for DescriptorSetAlloc {}
unsafe impl Sync for DescriptorSetAlloc {}

/// An opaque handle identifying an allocation inside an allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(not(doc), repr(transparent))]
pub struct AllocationHandle(*mut ());

unsafe impl Send for AllocationHandle {}
unsafe impl Sync for AllocationHandle {}

impl AllocationHandle {
    /// Creates a null `AllocationHandle`.
    ///
    /// Use this if you don't have anything that you need to associate with the allocation.
    #[inline]
    pub const fn null() -> Self {
        AllocationHandle(ptr::null_mut())
    }

    /// Stores a pointer in an `AllocationHandle`.
    ///
    /// Use this if you want to associate an allocation with some (host) heap allocation.
    #[inline]
    pub const fn from_ptr(ptr: *mut ()) -> Self {
        AllocationHandle(ptr)
    }

    /// Stores an index inside an `AllocationHandle`.
    ///
    /// Use this if you want to associate an allocation with some index.
    #[allow(clippy::useless_transmute)]
    #[inline]
    pub const fn from_index(index: usize) -> Self {
        // SAFETY: `usize` and `*mut ()` have the same layout.
        AllocationHandle(unsafe { mem::transmute::<usize, *mut ()>(index) })
    }

    /// Retrieves a previously-stored pointer from the `AllocationHandle`.
    ///
    /// If this handle hasn't been created using [`from_ptr`] then this will return an invalid
    /// pointer, dereferencing which is undefined behavior.
    ///
    /// [`from_ptr`]: Self::from_ptr
    #[inline]
    pub const fn as_ptr(self) -> *mut () {
        self.0
    }

    /// Retrieves a previously-stored index from the `AllocationHandle`.
    ///
    /// If this handle hasn't been created using [`from_index`] then this will return a bogus
    /// result.
    ///
    /// [`from_index`]: Self::from_index
    #[allow(clippy::transmutes_expressible_as_ptr_casts)]
    #[inline]
    pub const fn as_index(self) -> usize {
        // SAFETY: `usize` and `*mut ()` have the same layout.
        unsafe { mem::transmute::<*mut (), usize>(self.0) }
    }
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
    create_info: StandardDescriptorSetAllocatorCreateInfo,
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
    pub fn new(
        device: Arc<Device>,
        create_info: StandardDescriptorSetAllocatorCreateInfo,
    ) -> StandardDescriptorSetAllocator {
        StandardDescriptorSetAllocator {
            device: InstanceOwnedDebugWrapper(device),
            pools: ThreadLocal::new(),
            create_info,
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
    #[inline]
    fn allocate(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<DescriptorSetAlloc, Validated<VulkanError>> {
        let is_fixed = layout.variable_descriptor_count() == 0;
        let pools = self.pools.get_or_default();

        let entry = unsafe { &mut *pools.get() }.get_or_try_insert(layout.id(), || {
            if is_fixed {
                FixedEntry::new(layout, &self.create_info).map(Entry::Fixed)
            } else {
                VariableEntry::new(layout, &self.create_info).map(Entry::Variable)
            }
        })?;

        match entry {
            Entry::Fixed(entry) => entry.allocate(layout, &self.create_info),
            Entry::Variable(entry) => {
                entry.allocate(layout, variable_descriptor_count, &self.create_info)
            }
        }
    }

    #[inline]
    unsafe fn deallocate(&self, allocation: DescriptorSetAlloc) {
        let is_fixed = allocation.inner.variable_descriptor_count() == 0;
        let ptr = allocation.handle.as_ptr();

        if is_fixed {
            // SAFETY: The caller must guarantee that `allocation` refers to one allocated by
            // `self`, therefore `ptr` must be the same one we gave out on allocation. We also know
            // that the pointer must be valid, because the caller must guarantee that the same
            // allocation isn't deallocated more than once. That means that since we cloned the
            // `Arc` on allocation, at least that strong reference must still keep it alive, and we
            // can safely drop this clone at the end of the scope here.
            let reserve = unsafe { Arc::from_raw(ptr.cast::<ArrayQueue<DescriptorPoolAlloc>>()) };

            let _ = reserve.push(allocation.inner);
        } else {
            // SAFETY: Same as the `Arc::from_raw` above.
            let reserve = unsafe { Arc::from_raw(ptr.cast::<ArrayQueue<Arc<DescriptorPool>>>()) };

            let pool = allocation.pool;

            // We have to make sure that we don't reset the pool under these conditions, because
            // 1. it could cause a panic while panicking
            // 2. the pool could still be in use by `VariableEntry`, in which case the count would
            //    be at last 2 (one for our reference and one in the `VariableEntry`), however
            //    there could also be other references in other allocations, or the user could have
            //    created a reference themself (which will most certainly cause a leak)
            // respectively.
            if thread::panicking() || Arc::strong_count(&pool) != 1 {
                return;
            }

            // SAFETY: We checked that the pool has a single strong reference above, and we own
            // this last reference, therefore it's impossible that a new reference would be created
            // outside of this scope.
            unsafe { pool.reset() }.unwrap();

            // If there is not enough space in the reserve, we destroy the pool. The only way this
            // can happen is if something is resource hogging, forcing new pools to be created such
            // that the number exceeds `MAX_POOLS`, and then drops them all at once.
            let _ = reserve.push(pool);
        }
    }
}

unsafe impl<T: DescriptorSetAllocator> DescriptorSetAllocator for Arc<T> {
    #[inline]
    fn allocate(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<DescriptorSetAlloc, Validated<VulkanError>> {
        (**self).allocate(layout, variable_descriptor_count)
    }

    #[inline]
    unsafe fn deallocate(&self, allocation: DescriptorSetAlloc) {
        (**self).deallocate(allocation)
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
    pool: Arc<DescriptorPool>,
    reserve: Arc<ArrayQueue<DescriptorPoolAlloc>>,
}

impl FixedEntry {
    fn new(
        layout: &Arc<DescriptorSetLayout>,
        create_info: &StandardDescriptorSetAllocatorCreateInfo,
    ) -> Result<Self, Validated<VulkanError>> {
        let pool = DescriptorPool::new(
            layout.device().clone(),
            DescriptorPoolCreateInfo {
                flags: create_info
                    .update_after_bind
                    .then_some(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .unwrap_or_default(),
                max_sets: create_info.set_count as u32,
                pool_sizes: layout
                    .descriptor_counts()
                    .iter()
                    .map(|(&ty, &count)| {
                        assert!(ty != DescriptorType::InlineUniformBlock);
                        (ty, count * create_info.set_count as u32)
                    })
                    .collect(),
                ..Default::default()
            },
        )
        .map_err(Validated::unwrap)?;

        let allocate_infos =
            (0..create_info.set_count).map(|_| DescriptorSetAllocateInfo::new(layout.clone()));

        let allocs =
            unsafe { pool.allocate_descriptor_sets(allocate_infos) }.map_err(|err| match err {
                Validated::ValidationError(_) => err,
                Validated::Error(vk_err) => match vk_err {
                    VulkanError::OutOfHostMemory | VulkanError::OutOfDeviceMemory => err,
                    // This can't happen as we don't free individual sets.
                    VulkanError::FragmentedPool => unreachable!(),
                    // We created the pool with an exact size.
                    VulkanError::OutOfPoolMemory => unreachable!(),
                    // Shouldn't ever be returned.
                    _ => unreachable!(),
                },
            })?;

        let reserve = ArrayQueue::new(create_info.set_count);

        for alloc in allocs {
            let _ = reserve.push(alloc);
        }

        Ok(FixedEntry {
            pool: Arc::new(pool),
            reserve: Arc::new(reserve),
        })
    }

    fn allocate(
        &mut self,
        layout: &Arc<DescriptorSetLayout>,
        create_info: &StandardDescriptorSetAllocatorCreateInfo,
    ) -> Result<DescriptorSetAlloc, Validated<VulkanError>> {
        let inner = if let Some(inner) = self.reserve.pop() {
            inner
        } else {
            *self = FixedEntry::new(layout, create_info)?;

            self.reserve.pop().unwrap()
        };

        Ok(DescriptorSetAlloc {
            inner,
            pool: self.pool.clone(),
            handle: AllocationHandle::from_ptr(Arc::into_raw(self.reserve.clone()) as _),
        })
    }
}

#[derive(Debug)]
struct VariableEntry {
    pool: Arc<DescriptorPool>,
    reserve: Arc<ArrayQueue<Arc<DescriptorPool>>>,
    // The number of sets currently allocated from the Vulkan pool.
    allocations: usize,
}

impl VariableEntry {
    fn new(
        layout: &DescriptorSetLayout,
        create_info: &StandardDescriptorSetAllocatorCreateInfo,
    ) -> Result<Self, Validated<VulkanError>> {
        Self::with_reserve(layout, create_info, Arc::new(ArrayQueue::new(MAX_POOLS)))
    }

    fn with_reserve(
        layout: &DescriptorSetLayout,
        create_info: &StandardDescriptorSetAllocatorCreateInfo,
        reserve: Arc<ArrayQueue<Arc<DescriptorPool>>>,
    ) -> Result<Self, Validated<VulkanError>> {
        let pool = DescriptorPool::new(
            layout.device().clone(),
            DescriptorPoolCreateInfo {
                flags: create_info
                    .update_after_bind
                    .then_some(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .unwrap_or_default(),
                max_sets: create_info.set_count as u32,
                pool_sizes: layout
                    .descriptor_counts()
                    .iter()
                    .map(|(&ty, &count)| {
                        assert!(ty != DescriptorType::InlineUniformBlock);
                        (ty, count * create_info.set_count as u32)
                    })
                    .collect(),
                ..Default::default()
            },
        )
        .map_err(Validated::unwrap)?;

        Ok(VariableEntry {
            pool: Arc::new(pool),
            reserve,
            allocations: 0,
        })
    }

    fn allocate(
        &mut self,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
        create_info: &StandardDescriptorSetAllocatorCreateInfo,
    ) -> Result<DescriptorSetAlloc, Validated<VulkanError>> {
        if self.allocations >= create_info.set_count {
            // This can happen if there's only ever one allocation alive at any point in time. In
            // that case, when deallocating the last set before reaching `set_count`, there will be
            // 2 references to the pool (one here and one in the allocation) and so the pool won't
            // be returned to the reserve when deallocating. However, since there are no other
            // allocations alive, there would be no other allocations that could return it to the
            // reserve. To avoid dropping the pool unneccessarily, we simply continue using it. In
            // the case where there are other references, we drop ours, at which point an
            // allocation still holding a reference will be able to put the pool into the reserve
            // when deallocated. If the user created a reference themself that will most certainly
            // lead to a memory leak.
            if Arc::strong_count(&self.pool) == 1 {
                // SAFETY: We checked that the pool has a single strong reference above, and we own
                // this last reference, therefore it's impossible that a new reference would be
                // created outside of this scope.
                unsafe { self.pool.reset() }.unwrap();

                self.allocations = 0;
            } else {
                *self = if let Some(pool) = self.reserve.pop() {
                    VariableEntry {
                        pool,
                        reserve: self.reserve.clone(),
                        allocations: 0,
                    }
                } else {
                    VariableEntry::with_reserve(layout, create_info, self.reserve.clone())?
                };
            }
        }

        let allocate_info = DescriptorSetAllocateInfo {
            variable_descriptor_count,
            ..DescriptorSetAllocateInfo::new(layout.clone())
        };

        let mut sets = unsafe { self.pool.allocate_descriptor_sets([allocate_info]) }.map_err(
            |err| match err {
                Validated::ValidationError(_) => err,
                Validated::Error(vk_err) => match vk_err {
                    VulkanError::OutOfHostMemory | VulkanError::OutOfDeviceMemory => err,
                    // This can't happen as we don't free individual sets.
                    VulkanError::FragmentedPool => unreachable!(),
                    // We created the pool to fit the maximum variable descriptor count.
                    VulkanError::OutOfPoolMemory => unreachable!(),
                    // Shouldn't ever be returned.
                    _ => unreachable!(),
                },
            },
        )?;

        self.allocations += 1;

        Ok(DescriptorSetAlloc {
            inner: sets.next().unwrap(),
            pool: self.pool.clone(),
            handle: AllocationHandle::from_ptr(Arc::into_raw(self.reserve.clone()) as _),
        })
    }
}

/// Parameters to create a new `StandardDescriptorSetAllocator`.
#[derive(Clone, Debug)]
pub struct StandardDescriptorSetAllocatorCreateInfo {
    /// How many descriptor sets should be allocated per pool.
    ///
    /// Each time a thread allocates using some descriptor set layout, and either no pools were
    /// initialized yet or all pools are full, a new pool is allocated for that thread and
    /// descriptor set layout combination. This option tells the allocator how many descriptor sets
    /// should be allocated for that pool. For fixed-size descriptor set layouts, it always
    /// allocates exactly this many descriptor sets at once for the pool, as that is more
    /// performant than allocating them one-by-one. For descriptor set layouts with a variable
    /// descriptor count, it allocates a pool capable of holding exactly this many descriptor sets,
    /// but doesn't allocate any descriptor sets since the variable count isn't known. What this
    /// means is that you should make sure that this isn't too large, so that you don't end up
    /// wasting too much memory. You also don't want this to be too low, because that on the other
    /// hand would mean that the pool would have to be reset more often, or that more pools would
    /// need to be created, depending on the lifetime of the descriptor sets.
    ///
    /// The default value is `32`.
    pub set_count: usize,

    /// Whether to allocate descriptor pools with the
    /// [`DescriptorPoolCreateFlags::UPDATE_AFTER_BIND`] flag set.
    ///
    /// The default value is `false`.
    pub update_after_bind: bool,

    pub _ne: crate::NonExhaustive,
}

impl Default for StandardDescriptorSetAllocatorCreateInfo {
    #[inline]
    fn default() -> Self {
        StandardDescriptorSetAllocatorCreateInfo {
            set_count: 32,
            update_after_bind: false,
            _ne: crate::NonExhaustive(()),
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
