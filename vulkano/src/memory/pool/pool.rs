// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use fnv::FnvHasher;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::BuildHasherDefault;
use std::sync::Arc;
use std::sync::Mutex;

use device::Device;
use instance::MemoryType;
use memory::DeviceMemory;
use memory::MappedDeviceMemory;
use memory::DeviceMemoryAllocError;
use memory::pool::AllocLayout;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdHostVisibleMemoryTypePool;
use memory::pool::StdHostVisibleMemoryTypePoolAlloc;
use memory::pool::StdNonHostVisibleMemoryTypePool;
use memory::pool::StdNonHostVisibleMemoryTypePoolAlloc;

#[derive(Debug)]
pub struct StdMemoryPool {
    device: Arc<Device>,

    // For each memory type index, stores the associated pool.
    pools: Mutex<HashMap<(u32, AllocLayout), Pool, BuildHasherDefault<FnvHasher>>>,
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

unsafe impl MemoryPool for Arc<StdMemoryPool> {
    type Alloc = StdMemoryPoolAlloc;

    fn alloc(&self, memory_type: MemoryType, size: usize, alignment: usize, layout: AllocLayout)
             -> Result<StdMemoryPoolAlloc, DeviceMemoryAllocError> {
        let mut pools = self.pools.lock().unwrap();

        match pools.entry((memory_type.id(), layout)) {
            Entry::Occupied(entry) => {
                match entry.get() {
                    &Pool::HostVisible(ref pool) => {
                        let alloc = StdHostVisibleMemoryTypePool::alloc(&pool, size, alignment)?;
                        let inner = StdMemoryPoolAllocInner::HostVisible(alloc);
                        Ok(StdMemoryPoolAlloc {
                               inner: inner,
                               pool: self.clone(),
                           })
                    },
                    &Pool::NonHostVisible(ref pool) => {
                        let alloc = StdNonHostVisibleMemoryTypePool::alloc(&pool, size, alignment)?;
                        let inner = StdMemoryPoolAllocInner::NonHostVisible(alloc);
                        Ok(StdMemoryPoolAlloc {
                               inner: inner,
                               pool: self.clone(),
                           })
                    },
                }
            },

            Entry::Vacant(entry) => {
                match memory_type.is_host_visible() {
                    true => {
                        let pool = StdHostVisibleMemoryTypePool::new(self.device.clone(),
                                                                     memory_type);
                        entry.insert(Pool::HostVisible(pool.clone()));
                        let alloc = StdHostVisibleMemoryTypePool::alloc(&pool, size, alignment)?;
                        let inner = StdMemoryPoolAllocInner::HostVisible(alloc);
                        Ok(StdMemoryPoolAlloc {
                               inner: inner,
                               pool: self.clone(),
                           })
                    },
                    false => {
                        let pool = StdNonHostVisibleMemoryTypePool::new(self.device.clone(),
                                                                        memory_type);
                        entry.insert(Pool::NonHostVisible(pool.clone()));
                        let alloc = StdNonHostVisibleMemoryTypePool::alloc(&pool, size, alignment)?;
                        let inner = StdMemoryPoolAllocInner::NonHostVisible(alloc);
                        Ok(StdMemoryPoolAlloc {
                               inner: inner,
                               pool: self.clone(),
                           })
                    },
                }
            },
        }
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
