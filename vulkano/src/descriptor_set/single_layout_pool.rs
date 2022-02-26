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
use crossbeam_queue::SegQueue;
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

/// `SingleLayoutDescSetPool` is a convenience wrapper provided by Vulkano not to be confused with
/// `VkDescriptorPool`. Its function is to provide access to pool(s) to allocate `DescriptorSet`'s
/// from and optimizes for a specific layout. For a more general purpose pool see `descriptor_set::pool::StdDescriptorPool`.
pub struct SingleLayoutDescSetPool {
    // The `SingleLayoutPool` struct contains an actual Vulkan pool. Every time it is full we create
    // a new pool and replace the current one with the new one.
    inner: Option<Arc<SingleLayoutPool>>,
    // The Vulkan device.
    device: Arc<Device>,
    // The amount of sets available to use when we create a new Vulkan pool.
    set_count: usize,
    // The descriptor layout that this pool is for.
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
    pub fn new(layout: Arc<DescriptorSetLayout>) -> Self {
        assert!(
            !layout.push_descriptor(),
            "the provided descriptor set layout is for push descriptors, and cannot be used to build a descriptor set object"
        );
        assert!(
            layout.variable_descriptor_count() == 0,
            "the provided descriptor set layout has a binding with a variable descriptor count, which cannot be used with SingleLayoutDescSetPool"
        );

        Self {
            inner: None,
            device: layout.device().clone(),
            set_count: 4,
            layout,
        }
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

    fn next_alloc(&mut self) -> Result<SingleLayoutPoolAlloc, OomError> {
        loop {
            let mut not_enough_sets = false;

            if let Some(ref mut p_inner) = self.inner {
                if let Some(existing) = p_inner.reserve.pop() {
                    return Ok(SingleLayoutPoolAlloc {
                        pool: p_inner.clone(),
                        inner: Some(existing),
                    });
                } else {
                    not_enough_sets = true;
                }
            }

            if not_enough_sets {
                self.set_count *= 2;
            }

            let count = *self.layout.descriptors_count() * self.set_count as u32;
            let mut unsafe_pool = UnsafeDescriptorPool::new(
                self.device.clone(),
                UnsafeDescriptorPoolCreateInfo {
                    max_sets: self.set_count as u32,
                    pool_sizes: count.into(),
                    ..Default::default()
                },
            )?;

            let reserve = unsafe {
                match unsafe_pool.allocate_descriptor_sets((0..self.set_count).map(|_| {
                    DescriptorSetAllocateInfo {
                        layout: self.layout.as_ref(),
                        variable_descriptor_count: 0,
                    }
                })) {
                    Ok(alloc_iter) => {
                        let reserve = SegQueue::new();

                        for alloc in alloc_iter {
                            reserve.push(alloc);
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
                        unreachable!()
                    }
                    Err(DescriptorPoolAllocError::OutOfPoolMemory) => unreachable!(),
                }
            };

            self.inner = Some(Arc::new(SingleLayoutPool {
                inner: unsafe_pool,
                reserve,
            }));
        }
    }
}

struct SingleLayoutPool {
    // The actual Vulkan descriptor pool. This field isn't actually used anywhere, but we need to
    // keep the pool alive in order to keep the descriptor sets valid.
    inner: UnsafeDescriptorPool,

    // List of descriptor sets. When `alloc` is called, a descriptor will be extracted from this
    // list. When a `SingleLayoutPoolAlloc` is dropped, its descriptor set is put back in this list.
    reserve: SegQueue<UnsafeDescriptorSet>,
}

struct SingleLayoutPoolAlloc {
    // The `SingleLayoutPool` were we allocated from. We need to keep a copy of it in each allocation
    // so that we can put back the allocation in the list in our `Drop` impl.
    pool: Arc<SingleLayoutPool>,

    // The actual descriptor set, wrapped inside an `Option` so that we can extract it in our
    // `Drop` impl.
    inner: Option<UnsafeDescriptorSet>,
}

impl DescriptorPoolAlloc for SingleLayoutPoolAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.inner.as_ref().unwrap()
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet {
        self.inner.as_mut().unwrap()
    }
}

impl Drop for SingleLayoutPoolAlloc {
    fn drop(&mut self) {
        let inner = self.inner.take().unwrap();
        self.pool.reserve.push(inner);
    }
}

/// A descriptor set created from a `SingleLayoutDescSetPool`.
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
