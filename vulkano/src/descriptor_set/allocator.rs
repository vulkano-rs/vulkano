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

use super::{
    layout::DescriptorSetLayout,
    single_layout_pool::{
        SingleLayoutDescriptorSetPool, SingleLayoutPoolAlloc,
        SingleLayoutVariableDescriptorSetPool, SingleLayoutVariablePoolAlloc,
    },
    sys::UnsafeDescriptorSet,
};
use crate::{
    device::{Device, DeviceOwned},
    OomError,
};
use ahash::HashMap;
use std::{cell::UnsafeCell, sync::Arc};
use thread_local::ThreadLocal;

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
    ) -> Result<Self::Alloc, OomError>;
}

/// An allocated descriptor set.
pub trait DescriptorSetAlloc: Send + Sync {
    /// Returns the inner unsafe descriptor set object.
    fn inner(&self) -> &UnsafeDescriptorSet;

    /// Returns the inner unsafe descriptor set object.
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet;
}

/// Standard implementation of a descriptor set allocator.
///
/// The intended way to use this allocator is to have one that is used globally for the duration of
/// the program, in order to avoid creating and destroying [`DescriptorPool`]s, as that is
/// expensive. Alternatively, you can have one locally on a thread for the duration of the thread.
///
/// Internally, this allocator uses one [`SingleLayoutDescriptorSetPool`] /
/// [`SingleLayoutVariableDescriptorSetPool`] per descriptor set layout per thread, using
/// Thread-Local Storage. When a thread first allocates, an entry is reserved for it in the TLS.
/// After a thread exits and the allocator wasn't dropped yet, its entry is freed, but the pools
/// it used are not dropped. The next time a new thread allocates for the first time, the entry is
/// reused along with the pools. If all threads drop their reference to the allocator, all entries
/// along with the allocator are dropped, even if the threads didn't exit yet, which is why you
/// should keep the allocator alive for as long as you need to allocate so that the pools can keep
/// being reused.
///
/// This allocator only needs to lock when a thread first allocates or when a thread that
/// previously allocated exits. In all other cases, allocation is lock-free.
///
/// [`DescriptorPool`]: crate::descriptor_set::pool::DescriptorPool
#[derive(Debug)]
pub struct StandardDescriptorSetAllocator {
    device: Arc<Device>,
    pools: ThreadLocal<UnsafeCell<HashMap<Arc<DescriptorSetLayout>, Pool>>>,
}

#[derive(Debug)]
enum Pool {
    Fixed(SingleLayoutDescriptorSetPool),
    Variable(SingleLayoutVariableDescriptorSetPool),
}

impl StandardDescriptorSetAllocator {
    /// Creates a new `StandardDescriptorSetAllocator`.
    #[inline]
    pub fn new(device: Arc<Device>) -> StandardDescriptorSetAllocator {
        StandardDescriptorSetAllocator {
            device,
            pools: ThreadLocal::new(),
        }
    }
}

unsafe impl DescriptorSetAllocator for StandardDescriptorSetAllocator {
    type Alloc = StandardDescriptorSetAlloc;

    /// Allocates a descriptor set.
    ///
    /// # Panics
    ///
    /// - Panics if the provided `layout` is for push descriptors rather than regular descriptor
    ///   sets.
    /// - Panics if the provided `variable_descriptor_count` is greater than the maximum number of
    ///   variable count descriptors in the set.
    #[inline]
    fn allocate(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<StandardDescriptorSetAlloc, OomError> {
        assert!(
            !layout.push_descriptor(),
            "the provided descriptor set layout is for push descriptors, and cannot be used to \
            build a descriptor set object",
        );

        let max_count = layout.variable_descriptor_count();

        assert!(
            variable_descriptor_count <= max_count,
            "the provided variable_descriptor_count ({}) is greater than the maximum number of \
            variable count descriptors in the set ({})",
            variable_descriptor_count,
            max_count,
        );

        let pools = self.pools.get_or(|| UnsafeCell::new(HashMap::default()));
        let pools = unsafe { &mut *pools.get() };

        // We do this instead of using `HashMap::entry` directly because that would involve cloning
        // an `Arc` every time. `hash_raw_entry` is still not stabilized >:(
        let pool = if let Some(pool) = pools.get_mut(layout) {
            pool
        } else {
            pools.entry(layout.clone()).or_insert(if max_count == 0 {
                Pool::Fixed(SingleLayoutDescriptorSetPool::new(layout.clone())?)
            } else {
                Pool::Variable(SingleLayoutVariableDescriptorSetPool::new(layout.clone())?)
            })
        };

        let inner = match pool {
            Pool::Fixed(pool) => PoolAlloc::Fixed(pool.next_alloc()?),
            Pool::Variable(pool) => {
                PoolAlloc::Variable(pool.next_alloc(variable_descriptor_count)?)
            }
        };

        Ok(StandardDescriptorSetAlloc { inner })
    }
}

unsafe impl DeviceOwned for StandardDescriptorSetAllocator {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// A descriptor set allocated from a [`StandardDescriptorSetAllocator`].
#[derive(Debug)]
pub struct StandardDescriptorSetAlloc {
    // The actual descriptor alloc.
    inner: PoolAlloc,
}

#[derive(Debug)]
enum PoolAlloc {
    Fixed(SingleLayoutPoolAlloc),
    Variable(SingleLayoutVariablePoolAlloc),
}

impl DescriptorSetAlloc for StandardDescriptorSetAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        match &self.inner {
            PoolAlloc::Fixed(alloc) => alloc.inner(),
            PoolAlloc::Variable(alloc) => alloc.inner(),
        }
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet {
        match &mut self.inner {
            PoolAlloc::Fixed(alloc) => alloc.inner_mut(),
            PoolAlloc::Variable(alloc) => alloc.inner_mut(),
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

        let pool1 = if let PoolAlloc::Fixed(alloc) = allocator.allocate(&layout, 0).unwrap().inner {
            alloc.pool().handle()
        } else {
            unreachable!()
        };

        thread::spawn(move || {
            let pool2 =
                if let PoolAlloc::Fixed(alloc) = allocator.allocate(&layout, 0).unwrap().inner {
                    alloc.pool().handle()
                } else {
                    unreachable!()
                };
            assert_ne!(pool1, pool2);
        })
        .join()
        .unwrap();
    }
}
