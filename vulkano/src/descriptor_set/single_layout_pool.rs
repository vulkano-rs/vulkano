// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    layout::DescriptorSetLayout,
    pool::{
        DescriptorPoolAlloc, DescriptorPoolAllocError, DescriptorSetAllocateInfo,
        UnsafeDescriptorPool, UnsafeDescriptorPoolCreateInfo,
    },
    sys::UnsafeDescriptorSet,
    DescriptorSet, DescriptorSetCreationError, DescriptorSetInner, DescriptorSetResources,
    WriteDescriptorSet,
};
use crate::{
    device::{Device, DeviceOwned},
    OomError, VulkanObject,
};
use crossbeam_queue::ArrayQueue;
use std::{
    cell::UnsafeCell,
    hash::{Hash, Hasher},
    mem::ManuallyDrop,
    sync::Arc,
};

const MAX_SETS: usize = 32;

const MAX_POOLS: usize = 32;

/// `SingleLayoutDescSetPool` is a convenience wrapper provided by Vulkano not to be confused with
/// `VkDescriptorPool`. Its function is to provide access to pool(s) to allocate descriptor sets
/// from and optimizes for a specific layout which must not have a variable descriptor count. If
/// you need a variable descriptor count see [`SingleLayoutVariableDescSetPool`]. For a more general
/// purpose pool see [`StandardDescriptorPool`].
///
/// [`StandardDescriptorPool`]: super::pool::standard::StandardDescriptorPool
#[derive(Debug)]
pub struct SingleLayoutDescSetPool {
    // The `SingleLayoutPool` struct contains an actual Vulkan pool. Every time it is full we create
    // a new pool and replace the current one with the new one.
    inner: Arc<SingleLayoutPool>,
    // The amount of sets available to use when we create a new Vulkan pool.
    set_count: usize,
    // The descriptor set layout that this pool is for.
    layout: Arc<DescriptorSetLayout>,
}

impl SingleLayoutDescSetPool {
    /// Initializes a new pool. The pool is configured to allocate sets that corresponds to the
    /// parameters passed to this function.
    ///
    /// # Panics
    ///
    /// - Panics if the provided `layout` is for push descriptors rather than regular descriptor
    ///   sets.
    /// - Panics if the provided `layout` has a binding with a variable descriptor count.
    pub fn new(layout: Arc<DescriptorSetLayout>) -> Result<Self, OomError> {
        assert!(
            !layout.push_descriptor(),
            "the provided descriptor set layout is for push descriptors, and cannot be used to \
            build a descriptor set object",
        );
        assert!(
            layout.variable_descriptor_count() == 0,
            "the provided descriptor set layout has a binding with a variable descriptor count, \
            which cannot be used with SingleLayoutDescSetPool",
        );

        Ok(Self {
            inner: SingleLayoutPool::new(&layout, MAX_SETS)?,
            set_count: MAX_SETS,
            layout,
        })
    }

    /// Returns a new descriptor set, either by creating a new one or returning an existing one
    /// from the internal reserve.
    #[inline]
    pub fn next(
        &mut self,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Result<Arc<SingleLayoutDescSet>, DescriptorSetCreationError> {
        let alloc = self.next_alloc()?;
        let inner = DescriptorSetInner::new(
            alloc.inner().internal_object(),
            self.layout.clone(),
            0,
            descriptor_writes,
        )?;

        Ok(Arc::new(SingleLayoutDescSet { alloc, inner }))
    }

    pub(crate) fn next_alloc(&mut self) -> Result<SingleLayoutPoolAlloc, OomError> {
        loop {
            if let Some(existing) = self.inner.reserve.pop() {
                return Ok(SingleLayoutPoolAlloc {
                    pool: self.inner.clone(),
                    inner: ManuallyDrop::new(existing),
                });
            }

            self.set_count *= 2;

            self.inner = SingleLayoutPool::new(&self.layout, self.set_count)?;
        }
    }
}

#[derive(Debug)]
struct SingleLayoutPool {
    // The actual Vulkan descriptor pool. This field isn't actually used anywhere, but we need to
    // keep the pool alive in order to keep the descriptor sets valid.
    _inner: UnsafeDescriptorPool,
    // List of descriptor sets. When `alloc` is called, a descriptor will be extracted from this
    // list. When a `SingleLayoutPoolAlloc` is dropped, its descriptor set is put back in this list.
    reserve: ArrayQueue<UnsafeDescriptorSet>,
}

impl SingleLayoutPool {
    fn new(layout: &Arc<DescriptorSetLayout>, set_count: usize) -> Result<Arc<Self>, OomError> {
        let mut inner = UnsafeDescriptorPool::new(
            layout.device().clone(),
            UnsafeDescriptorPoolCreateInfo {
                max_sets: set_count as u32,
                pool_sizes: layout
                    .descriptor_counts()
                    .iter()
                    .map(|(&ty, &count)| (ty, count * set_count as u32))
                    .collect(),
                ..Default::default()
            },
        )?;

        let allocate_infos = (0..set_count).map(|_| DescriptorSetAllocateInfo {
            layout,
            variable_descriptor_count: 0,
        });

        let reserve = match unsafe { inner.allocate_descriptor_sets(allocate_infos) } {
            Ok(alloc_iter) => {
                let reserve = ArrayQueue::new(set_count);

                for alloc in alloc_iter {
                    reserve.push(alloc).unwrap();
                }

                reserve
            }
            Err(DescriptorPoolAllocError::OutOfHostMemory) => {
                return Err(OomError::OutOfHostMemory);
            }
            Err(DescriptorPoolAllocError::OutOfDeviceMemory) => {
                return Err(OomError::OutOfDeviceMemory);
            }
            Err(DescriptorPoolAllocError::FragmentedPool) => {
                // This can't happen as we don't free individual sets.
                unreachable!();
            }
            Err(DescriptorPoolAllocError::OutOfPoolMemory) => {
                // We created the pool with an exact size.
                unreachable!();
            }
        };

        Ok(Arc::new(Self {
            _inner: inner,
            reserve,
        }))
    }
}

#[derive(Debug)]
pub(crate) struct SingleLayoutPoolAlloc {
    // The actual descriptor set.
    inner: ManuallyDrop<UnsafeDescriptorSet>,
    // The `SingleLayoutPool` where we allocated from. We need to keep a copy of it in each
    // allocation so that we can put back the allocation in the list in our `Drop` impl.
    pool: Arc<SingleLayoutPool>,
}

impl DescriptorPoolAlloc for SingleLayoutPoolAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        &self.inner
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet {
        &mut self.inner
    }
}

impl Drop for SingleLayoutPoolAlloc {
    fn drop(&mut self) {
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };
        self.pool.reserve.push(inner).unwrap();
    }
}

/// A descriptor set created from a [`SingleLayoutDescSetPool`].
pub struct SingleLayoutDescSet {
    alloc: SingleLayoutPoolAlloc,
    inner: DescriptorSetInner,
}

unsafe impl DescriptorSet for SingleLayoutDescSet {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.alloc.inner()
    }

    #[inline]
    fn layout(&self) -> &Arc<DescriptorSetLayout> {
        self.inner.layout()
    }

    #[inline]
    fn resources(&self) -> &DescriptorSetResources {
        self.inner.resources()
    }
}

unsafe impl DeviceOwned for SingleLayoutDescSet {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.layout().device()
    }
}

impl PartialEq for SingleLayoutDescSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner().internal_object() == other.inner().internal_object()
            && self.device() == other.device()
    }
}

impl Eq for SingleLayoutDescSet {}

impl Hash for SingleLayoutDescSet {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().internal_object().hash(state);
        self.device().hash(state);
    }
}

/// Much like [`SingleLayoutDescSetPool`], except that it allows you to allocate descriptor sets
/// with a variable descriptor count. As this has more overhead, you should only use this pool if
/// you need the functionality and prefer [`SingleLayoutDescSetPool`] otherwise. For a more general
/// purpose pool see [`StandardDescriptorPool`].
///
/// [`StandardDescriptorPool`]: super::pool::standard::StandardDescriptorPool
#[derive(Debug)]
pub struct SingleLayoutVariableDescSetPool {
    // The `SingleLayoutVariablePool` struct contains an actual Vulkan pool. Every time it is full
    // we grab one from the reserve, or create a new pool if there are none.
    inner: Arc<SingleLayoutVariablePool>,
    // When a `SingleLayoutVariablePool` is dropped, it returns its Vulkan pool here for reuse.
    reserve: Arc<ArrayQueue<UnsafeDescriptorPool>>,
    // The descriptor set layout that this pool is for.
    layout: Arc<DescriptorSetLayout>,
    // The number of sets currently allocated from the Vulkan pool.
    allocated_sets: usize,
}

impl SingleLayoutVariableDescSetPool {
    /// Initializes a new pool. The pool is configured to allocate sets that corresponds to the
    /// parameters passed to this function.
    ///
    /// # Panics
    ///
    /// - Panics if the provided `layout` is for push descriptors rather than regular descriptor
    ///   sets.
    pub fn new(layout: Arc<DescriptorSetLayout>) -> Result<Self, OomError> {
        assert!(
            !layout.push_descriptor(),
            "the provided descriptor set layout is for push descriptors, and cannot be used to \
            build a descriptor set object",
        );

        let reserve = Arc::new(ArrayQueue::new(MAX_POOLS));

        Ok(Self {
            inner: SingleLayoutVariablePool::new(&layout, reserve.clone())?,
            reserve,
            layout,
            allocated_sets: 0,
        })
    }

    /// Allocates a new descriptor set.
    ///
    /// # Panics
    ///
    /// - Panics if the provided `variable_descriptor_count` exceeds the maximum for the layout.
    #[inline]
    pub fn next(
        &mut self,
        variable_descriptor_count: u32,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Result<SingleLayoutVariableDescSet, DescriptorSetCreationError> {
        let max_count = self.layout.variable_descriptor_count();

        assert!(
            variable_descriptor_count <= max_count,
            "the provided variable_descriptor_count ({}) is greater than the maximum number of \
            variable count descriptors in the set ({})",
            variable_descriptor_count,
            max_count,
        );

        let alloc = self.next_alloc(variable_descriptor_count)?;
        let inner = DescriptorSetInner::new(
            alloc.inner().internal_object(),
            self.layout.clone(),
            0,
            descriptor_writes,
        )?;

        Ok(SingleLayoutVariableDescSet { inner, alloc })
    }

    pub(crate) fn next_alloc(
        &mut self,
        variable_descriptor_count: u32,
    ) -> Result<SingleLayoutVariablePoolAlloc, OomError> {
        if self.allocated_sets >= MAX_SETS {
            self.inner = if let Some(unsafe_pool) = self.reserve.pop() {
                Arc::new(SingleLayoutVariablePool {
                    inner: UnsafeCell::new(ManuallyDrop::new(unsafe_pool)),
                    reserve: self.reserve.clone(),
                })
            } else {
                SingleLayoutVariablePool::new(&self.layout, self.reserve.clone())?
            };
            self.allocated_sets = 0;
        }

        let inner = {
            let unsafe_pool = unsafe { &mut *self.inner.inner.get() };

            let allocate_info = DescriptorSetAllocateInfo {
                layout: &self.layout,
                variable_descriptor_count,
            };

            match unsafe { unsafe_pool.allocate_descriptor_sets([allocate_info]) } {
                Ok(mut sets) => sets.next().unwrap(),
                Err(DescriptorPoolAllocError::OutOfHostMemory) => {
                    return Err(OomError::OutOfHostMemory);
                }
                Err(DescriptorPoolAllocError::OutOfDeviceMemory) => {
                    return Err(OomError::OutOfDeviceMemory);
                }
                Err(DescriptorPoolAllocError::FragmentedPool) => {
                    // This can't happen as we don't free individual sets.
                    unreachable!();
                }
                Err(DescriptorPoolAllocError::OutOfPoolMemory) => {
                    // We created the pool to fit the maximum variable descriptor count.
                    unreachable!();
                }
            }
        };

        self.allocated_sets += 1;

        Ok(SingleLayoutVariablePoolAlloc {
            inner,
            _pool: self.inner.clone(),
        })
    }
}

#[derive(Debug)]
struct SingleLayoutVariablePool {
    // The actual Vulkan descriptor pool.
    inner: UnsafeCell<ManuallyDrop<UnsafeDescriptorPool>>,
    // Where we return the Vulkan descriptor pool in our `Drop` impl.
    reserve: Arc<ArrayQueue<UnsafeDescriptorPool>>,
}

impl SingleLayoutVariablePool {
    fn new(
        layout: &Arc<DescriptorSetLayout>,
        reserve: Arc<ArrayQueue<UnsafeDescriptorPool>>,
    ) -> Result<Arc<Self>, OomError> {
        let unsafe_pool = UnsafeDescriptorPool::new(
            layout.device().clone(),
            UnsafeDescriptorPoolCreateInfo {
                max_sets: MAX_SETS as u32,
                pool_sizes: layout
                    .descriptor_counts()
                    .iter()
                    .map(|(&ty, &count)| (ty, count * MAX_SETS as u32))
                    .collect(),
                ..Default::default()
            },
        )?;

        Ok(Arc::new(Self {
            inner: UnsafeCell::new(ManuallyDrop::new(unsafe_pool)),
            reserve,
        }))
    }
}

impl Drop for SingleLayoutVariablePool {
    fn drop(&mut self) {
        let mut inner = unsafe { ManuallyDrop::take(&mut *self.inner.get()) };
        // TODO: This should not return `Result`, resetting a pool can't fail.
        unsafe { inner.reset() }.unwrap();

        // If there is not enough space in the reserve, we destroy the pool. The only way this can
        // happen is if something is resource hogging, forcing new pools to be created such that
        // the number exceeds `MAX_POOLS`, and then drops them all at once.
        let _ = self.reserve.push(inner);
    }
}

#[derive(Debug)]
pub(crate) struct SingleLayoutVariablePoolAlloc {
    // The actual descriptor set.
    inner: UnsafeDescriptorSet,
    // The `SingleLayoutVariablePool` where we allocated from. We need to keep a copy of it in each
    // allocation so that we can put back the pool in the reserve once all allocations have been
    // dropped.
    _pool: Arc<SingleLayoutVariablePool>,
}

unsafe impl Send for SingleLayoutVariablePoolAlloc {}
unsafe impl Sync for SingleLayoutVariablePoolAlloc {}

impl DescriptorPoolAlloc for SingleLayoutVariablePoolAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        &self.inner
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet {
        &mut self.inner
    }
}

/// A descriptor set created from a [`SingleLayoutVariableDescSetPool`].
pub struct SingleLayoutVariableDescSet {
    alloc: SingleLayoutVariablePoolAlloc,
    inner: DescriptorSetInner,
}

unsafe impl DescriptorSet for SingleLayoutVariableDescSet {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.alloc.inner()
    }

    #[inline]
    fn layout(&self) -> &Arc<DescriptorSetLayout> {
        self.inner.layout()
    }

    #[inline]
    fn resources(&self) -> &DescriptorSetResources {
        self.inner.resources()
    }
}

unsafe impl DeviceOwned for SingleLayoutVariableDescSet {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.layout().device()
    }
}

impl PartialEq for SingleLayoutVariableDescSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner().internal_object() == other.inner().internal_object()
            && self.device() == other.device()
    }
}

impl Eq for SingleLayoutVariableDescSet {}

impl Hash for SingleLayoutVariableDescSet {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().internal_object().hash(state);
        self.device().hash(state);
    }
}
