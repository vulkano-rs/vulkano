// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use fnv::FnvHasher;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::Arc;
use std::sync::Mutex;

use device::Device;
use device::DeviceOwned;
use instance::MemoryType;
use memory::pool::AllocLayout;
use memory::pool::MappingRequirement;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdHostVisibleMemoryTypePool;
use memory::pool::StdHostVisibleMemoryTypePoolAlloc;
use memory::pool::StdNonHostVisibleMemoryTypePool;
use memory::pool::StdNonHostVisibleMemoryTypePoolAlloc;
use memory::DeviceMemory;
use memory::DeviceMemoryAllocError;
use memory::MappedDeviceMemory;

#[derive(Debug)]
pub struct StdMemoryPool {
    device: Arc<Device>,

    // For each memory type index, stores the associated pool.
    pools:
        Mutex<HashMap<(u32, AllocLayout, MappingRequirement), Pool, BuildHasherDefault<FnvHasher>>>,
}

impl StdMemoryPool {
    /// Creates a new pool.
    #[inline]
    pub fn new(device: Arc<Device>) -> Arc<StdMemoryPool> {
        let cap = device.physical_device().memory_types().len();
        let hasher = BuildHasherDefault::<FnvHasher>::default();

        Arc::new(StdMemoryPool {
            device: device.clone(),
            pools: Mutex::new(HashMap::with_capacity_and_hasher(cap, hasher)),
        })
    }
}

fn generic_allocation(
    mem_pool: Arc<StdMemoryPool>,
    memory_type: MemoryType,
    size: usize,
    alignment: usize,
    layout: AllocLayout,
    map: MappingRequirement,
) -> Result<StdMemoryPoolAlloc, DeviceMemoryAllocError> {
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

/// Same as `generic_allocation` but with exportable memory option.
#[cfg(target_os = "linux")]
fn generic_exportable_allocation(
    mem_pool: Arc<StdMemoryPool>,
    memory_type: MemoryType,
    size: usize,
    alignment: usize,
    layout: AllocLayout,
    map: MappingRequirement,
) -> Result<StdMemoryPoolAlloc, DeviceMemoryAllocError> {
    let mut pools = mem_pool.pools.lock().unwrap();

    let memory_type_host_visible = memory_type.is_host_visible();
    assert!(memory_type_host_visible || map == MappingRequirement::DoNotMap);

    match pools.entry((memory_type.id(), layout, map)) {
        Entry::Occupied(entry) => match entry.get() {
            &Pool::HostVisible(ref pool) => {
                let alloc = StdHostVisibleMemoryTypePool::alloc_exportable(&pool, size, alignment)?;
                let inner = StdMemoryPoolAllocInner::HostVisible(alloc);
                Ok(StdMemoryPoolAlloc {
                    inner,
                    pool: mem_pool.clone(),
                })
            }
            &Pool::NonHostVisible(ref pool) => {
                let alloc =
                    StdNonHostVisibleMemoryTypePool::alloc_exportable(&pool, size, alignment)?;
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
                let alloc = StdHostVisibleMemoryTypePool::alloc_exportable(&pool, size, alignment)?;
                let inner = StdMemoryPoolAllocInner::HostVisible(alloc);
                Ok(StdMemoryPoolAlloc {
                    inner,
                    pool: mem_pool.clone(),
                })
            } else {
                let pool =
                    StdNonHostVisibleMemoryTypePool::new(mem_pool.device.clone(), memory_type);
                entry.insert(Pool::NonHostVisible(pool.clone()));
                let alloc =
                    StdNonHostVisibleMemoryTypePool::alloc_exportable(&pool, size, alignment)?;
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
        size: usize,
        alignment: usize,
        layout: AllocLayout,
        map: MappingRequirement,
    ) -> Result<StdMemoryPoolAlloc, DeviceMemoryAllocError> {
        generic_allocation(self.clone(), memory_type, size, alignment, layout, map)
    }

    /// Same as `alloc_generic` but with exportable memory option.
    #[cfg(target_os = "linux")]
    fn alloc_generic_exportable(
        &self,
        memory_type: MemoryType,
        size: usize,
        alignment: usize,
        layout: AllocLayout,
        map: MappingRequirement,
    ) -> Result<StdMemoryPoolAlloc, DeviceMemoryAllocError> {
        generic_exportable_allocation(self.clone(), memory_type, size, alignment, layout, map)
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
    pub fn size(&self) -> usize {
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
    fn offset(&self) -> usize {
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
