// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    device::{Device, DeviceOwned},
    memory::{
        pool::{
            AllocLayout, MappingRequirement, MemoryPool, MemoryPoolAlloc,
            StandardHostVisibleMemoryTypePool, StandardHostVisibleMemoryTypePoolAlloc,
            StandardNonHostVisibleMemoryTypePool, StandardNonHostVisibleMemoryTypePoolAlloc,
        },
        DeviceMemory, DeviceMemoryError, MappedDeviceMemory,
    },
    DeviceSize,
};
use parking_lot::Mutex;
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
};

#[derive(Debug)]
pub struct StandardMemoryPool {
    device: Arc<Device>,

    // For each memory type index, stores the associated pool.
    pools: Mutex<HashMap<(u32, AllocLayout, MappingRequirement), Pool>>,
}

impl StandardMemoryPool {
    /// Creates a new pool.
    #[inline]
    pub fn new(device: Arc<Device>) -> Arc<StandardMemoryPool> {
        let cap = device
            .physical_device()
            .memory_properties()
            .memory_types
            .len();

        Arc::new(StandardMemoryPool {
            device,
            pools: Mutex::new(HashMap::with_capacity(cap)),
        })
    }
}

fn generic_allocation(
    mem_pool: Arc<StandardMemoryPool>,
    memory_type_index: u32,
    size: DeviceSize,
    alignment: DeviceSize,
    layout: AllocLayout,
    map: MappingRequirement,
) -> Result<StandardMemoryPoolAlloc, DeviceMemoryError> {
    let mut pools = mem_pool.pools.lock();

    let memory_properties = mem_pool.device().physical_device().memory_properties();
    let memory_type = memory_properties
        .memory_types
        .get(memory_type_index as usize)
        .ok_or(DeviceMemoryError::MemoryTypeIndexOutOfRange {
            memory_type_index,
            memory_type_count: memory_properties.memory_types.len() as u32,
        })?;

    let memory_type_host_visible = memory_type.property_flags.host_visible;
    assert!(memory_type_host_visible || map == MappingRequirement::DoNotMap);

    match pools.entry((memory_type_index, layout, map)) {
        Entry::Occupied(entry) => match *entry.get() {
            Pool::HostVisible(ref pool) => {
                let alloc = pool.alloc(size, alignment)?;
                let inner = StandardMemoryPoolAllocInner::HostVisible(alloc);
                Ok(StandardMemoryPoolAlloc {
                    inner,
                    _pool: mem_pool.clone(),
                })
            }
            Pool::NonHostVisible(ref pool) => {
                let alloc = pool.alloc(size, alignment)?;
                let inner = StandardMemoryPoolAllocInner::NonHostVisible(alloc);
                Ok(StandardMemoryPoolAlloc {
                    inner,
                    _pool: mem_pool.clone(),
                })
            }
        },

        Entry::Vacant(entry) => {
            if memory_type_host_visible {
                let pool = StandardHostVisibleMemoryTypePool::new(
                    mem_pool.device.clone(),
                    memory_type_index,
                );
                entry.insert(Pool::HostVisible(pool.clone()));
                let alloc = pool.alloc(size, alignment)?;
                let inner = StandardMemoryPoolAllocInner::HostVisible(alloc);
                Ok(StandardMemoryPoolAlloc {
                    inner,
                    _pool: mem_pool.clone(),
                })
            } else {
                let pool = StandardNonHostVisibleMemoryTypePool::new(
                    mem_pool.device.clone(),
                    memory_type_index,
                );
                entry.insert(Pool::NonHostVisible(pool.clone()));
                let alloc = pool.alloc(size, alignment)?;
                let inner = StandardMemoryPoolAllocInner::NonHostVisible(alloc);
                Ok(StandardMemoryPoolAlloc {
                    inner,
                    _pool: mem_pool.clone(),
                })
            }
        }
    }
}

unsafe impl MemoryPool for Arc<StandardMemoryPool> {
    type Alloc = StandardMemoryPoolAlloc;

    fn alloc_generic(
        &self,
        memory_type_index: u32,
        size: DeviceSize,
        alignment: DeviceSize,
        layout: AllocLayout,
        map: MappingRequirement,
    ) -> Result<StandardMemoryPoolAlloc, DeviceMemoryError> {
        generic_allocation(
            self.clone(),
            memory_type_index,
            size,
            alignment,
            layout,
            map,
        )
    }
}

unsafe impl DeviceOwned for StandardMemoryPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

#[derive(Debug)]
enum Pool {
    HostVisible(Arc<StandardHostVisibleMemoryTypePool>),
    NonHostVisible(Arc<StandardNonHostVisibleMemoryTypePool>),
}

#[derive(Debug)]
pub struct StandardMemoryPoolAlloc {
    inner: StandardMemoryPoolAllocInner,
    _pool: Arc<StandardMemoryPool>,
}

impl StandardMemoryPoolAlloc {
    #[inline]
    pub fn size(&self) -> DeviceSize {
        match self.inner {
            StandardMemoryPoolAllocInner::NonHostVisible(ref mem) => mem.size(),
            StandardMemoryPoolAllocInner::HostVisible(ref mem) => mem.size(),
        }
    }
}

unsafe impl MemoryPoolAlloc for StandardMemoryPoolAlloc {
    #[inline]
    fn memory(&self) -> &DeviceMemory {
        match self.inner {
            StandardMemoryPoolAllocInner::NonHostVisible(ref mem) => mem.memory(),
            StandardMemoryPoolAllocInner::HostVisible(ref mem) => mem.memory().as_ref(),
        }
    }

    #[inline]
    fn mapped_memory(&self) -> Option<&MappedDeviceMemory> {
        match self.inner {
            StandardMemoryPoolAllocInner::NonHostVisible(_) => None,
            StandardMemoryPoolAllocInner::HostVisible(ref mem) => Some(mem.memory()),
        }
    }

    #[inline]
    fn offset(&self) -> DeviceSize {
        match self.inner {
            StandardMemoryPoolAllocInner::NonHostVisible(ref mem) => mem.offset(),
            StandardMemoryPoolAllocInner::HostVisible(ref mem) => mem.offset(),
        }
    }
}

#[derive(Debug)]
enum StandardMemoryPoolAllocInner {
    NonHostVisible(StandardNonHostVisibleMemoryTypePoolAlloc),
    HostVisible(StandardHostVisibleMemoryTypePoolAlloc),
}
