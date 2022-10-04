// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{DescriptorPool, DescriptorPoolAlloc};
use crate::{
    descriptor_set::{
        layout::DescriptorSetLayout,
        single_layout_pool::{
            SingleLayoutPoolAlloc, SingleLayoutVariableDescSetPool, SingleLayoutVariablePoolAlloc,
        },
        sys::UnsafeDescriptorSet,
        SingleLayoutDescSetPool,
    },
    device::{Device, DeviceOwned},
    OomError,
};
use ahash::HashMap;
use std::sync::Arc;

/// Standard implementation of a descriptor pool.
///
/// Interally, this implementation uses one [`SingleLayoutDescSetPool`] /
/// [`SingleLayoutVariableDescSetPool`] per descriptor set layout.
#[derive(Debug)]
pub struct StandardDescriptorPool {
    device: Arc<Device>,
    pools: HashMap<Arc<DescriptorSetLayout>, Pool>,
}

#[derive(Debug)]
enum Pool {
    Fixed(SingleLayoutDescSetPool),
    Variable(SingleLayoutVariableDescSetPool),
}

impl StandardDescriptorPool {
    /// Builds a new `StandardDescriptorPool`.
    pub fn new(device: Arc<Device>) -> StandardDescriptorPool {
        StandardDescriptorPool {
            device,
            pools: HashMap::default(),
        }
    }
}

unsafe impl DescriptorPool for StandardDescriptorPool {
    type Alloc = StandardDescriptorPoolAlloc;

    fn allocate(
        &mut self,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<StandardDescriptorPoolAlloc, OomError> {
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

        // We do this instead of using `HashMap::entry` directly because that would involve cloning
        // an `Arc` every time. `hash_raw_entry` is still not stabilized >:(
        let pool = if let Some(pool) = self.pools.get_mut(layout) {
            pool
        } else {
            self.pools
                .entry(layout.clone())
                .or_insert(if max_count == 0 {
                    Pool::Fixed(SingleLayoutDescSetPool::new(layout.clone())?)
                } else {
                    Pool::Variable(SingleLayoutVariableDescSetPool::new(layout.clone())?)
                })
        };

        let inner = match pool {
            Pool::Fixed(pool) => PoolAlloc::Fixed(pool.next_alloc()?),
            Pool::Variable(pool) => {
                PoolAlloc::Variable(pool.next_alloc(variable_descriptor_count)?)
            }
        };

        Ok(StandardDescriptorPoolAlloc { inner })
    }
}

unsafe impl DeviceOwned for StandardDescriptorPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// A descriptor set allocated from a `StandardDescriptorPool`.
#[derive(Debug)]
pub struct StandardDescriptorPoolAlloc {
    // The actual descriptor alloc.
    inner: PoolAlloc,
}

#[derive(Debug)]
enum PoolAlloc {
    Fixed(SingleLayoutPoolAlloc),
    Variable(SingleLayoutVariablePoolAlloc),
}

impl DescriptorPoolAlloc for StandardDescriptorPoolAlloc {
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
