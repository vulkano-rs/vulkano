// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::device::physical::MemoryType;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::memory::pool::AllocLayout;
use crate::memory::pool::MappingRequirement;
use crate::memory::pool::MemoryPool;
use crate::memory::pool::MemoryPoolAlloc;
use crate::memory::pool::StdHostVisibleMemoryTypePool;
use crate::memory::pool::StdHostVisibleMemoryTypePoolAlloc;
use crate::memory::pool::StdNonHostVisibleMemoryTypePool;
use crate::memory::pool::StdNonHostVisibleMemoryTypePoolAlloc;
use crate::memory::DeviceMemory;
use crate::memory::DeviceMemoryAllocationError;
use crate::memory::MappedDeviceMemory;
use crate::DeviceSize;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug)]
pub struct StdMemoryPool {
    device: Arc<Device>,

    // For each memory type index, stores the associated pool.
    pools: Mutex<HashMap<(u32, AllocLayout, MappingRequirement), Pool>>,
}

impl StdMemoryPool {
    /// Creates a new pool.
    #[inline]
    pub fn new(device: Arc<Device>) -> Arc<StdMemoryPool> {
        let cap = device.physical_device().memory_types().len();

        Arc::new(StdMemoryPool {
            device: device.clone(),
            pools: Mutex::new(HashMap::with_capacity(cap)),
        })
    }
}

fn generic_allocation(
    mem_pool: Arc<StdMemoryPool>,
    memory_type: MemoryType,
    size: DeviceSize,
    alignment: DeviceSize,
    layout: AllocLayout,
    map: MappingRequirement,
) -> Result<StdMemoryPoolAlloc, DeviceMemoryAllocationError> {
    let mut pools = mem_pool.pools.lock().unwrap();

    let memory_type_host_visible = memory_type.is_host_visible();
    assert!(memory_type_host_visible || map == MappingRequirement::DoNotMap);

    match pools.entry((memory_type.id(), layout, map)) {
        Entry::Occupied(entry) => match entry.get() {
            &Pool::HostVisible(ref pool) => {
                let alloc = StdHostVisibleMemoryTypePool::alloc(&pool, size, alignment)?;
                let inner = StdMemoryPoolAllocInner::HostVisible(alloc);
                Ok(StdMemoryPoolAlloc {
                    inner,
                    pool: mem_pool.clone(),
                })
            }
            &Pool::NonHostVisible(ref pool) => {
                let alloc = StdNonHostVisibleMemoryTypePool::alloc(&pool, size, alignment)?;
                let inner = StdMemoryPoolAllocInner::NonHostVisible(alloc);
                Ok(StdMemoryPoolAlloc {
                    inner,
                    pool: mem_pool.clone(),
                })
            }
        },

        Entry::Vacant(entry) => {
            if memory_type_host_visible {
                let pool = StdHostVisibleMemoryTypePool::new(mem_pool.device.clone(), memory_type);
                entry.insert(Pool::HostVisible(pool.clone()));
                let alloc = StdHostVisibleMemoryTypePool::alloc(&pool, size, alignment)?;
                let inner = StdMemoryPoolAllocInner::HostVisible(alloc);
                Ok(StdMemoryPoolAlloc {
                    inner,
                    pool: mem_pool.clone(),
                })
            } else {
                let pool =
                    StdNonHostVisibleMemoryTypePool::new(mem_pool.device.clone(), memory_type);
                entry.insert(Pool::NonHostVisible(pool.clone()));
                let alloc = StdNonHostVisibleMemoryTypePool::alloc(&pool, size, alignment)?;
                let inner = StdMemoryPoolAllocInner::NonHostVisible(alloc);
                Ok(StdMemoryPoolAlloc {
                    inner,
                    pool: mem_pool.clone(),
                })
            }
        }
    }
}

unsafe impl MemoryPool for Arc<StdMemoryPool> {
    type Alloc = StdMemoryPoolAlloc;

    fn alloc_generic(
        &self,
        memory_type: MemoryType,
        size: DeviceSize,
        alignment: DeviceSize,
        layout: AllocLayout,
        map: MappingRequirement,
    ) -> Result<StdMemoryPoolAlloc, DeviceMemoryAllocationError> {
        generic_allocation(self.clone(), memory_type, size, alignment, layout, map)
    }
}

unsafe impl DeviceOwned for StdMemoryPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

#[derive(Debug)]
enum Pool {
    HostVisible(Arc<StdHostVisibleMemoryTypePool>),
    NonHostVisible(Arc<StdNonHostVisibleMemoryTypePool>),
}

#[derive(Debug)]
pub struct StdMemoryPoolAlloc {
    inner: StdMemoryPoolAllocInner,
    pool: Arc<StdMemoryPool>,
}

impl StdMemoryPoolAlloc {
    #[inline]
    pub fn size(&self) -> DeviceSize {
        match self.inner {
            StdMemoryPoolAllocInner::NonHostVisible(ref mem) => mem.size(),
            StdMemoryPoolAllocInner::HostVisible(ref mem) => mem.size(),
        }
    }
}

unsafe impl MemoryPoolAlloc for StdMemoryPoolAlloc {
    #[inline]
    fn memory(&self) -> &DeviceMemory {
        match self.inner {
            StdMemoryPoolAllocInner::NonHostVisible(ref mem) => mem.memory(),
            StdMemoryPoolAllocInner::HostVisible(ref mem) => mem.memory().as_ref(),
        }
    }

    #[inline]
    fn mapped_memory(&self) -> Option<&MappedDeviceMemory> {
        match self.inner {
            StdMemoryPoolAllocInner::NonHostVisible(_) => None,
            StdMemoryPoolAllocInner::HostVisible(ref mem) => Some(mem.memory()),
        }
    }

    #[inline]
    fn offset(&self) -> DeviceSize {
        match self.inner {
            StdMemoryPoolAllocInner::NonHostVisible(ref mem) => mem.offset(),
            StdMemoryPoolAllocInner::HostVisible(ref mem) => mem.offset(),
        }
    }
}

#[derive(Debug)]
enum StdMemoryPoolAllocInner {
    NonHostVisible(StdNonHostVisibleMemoryTypePoolAlloc),
    HostVisible(StdHostVisibleMemoryTypePoolAlloc),
}
