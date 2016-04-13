// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::BuildHasherDefault;
use std::sync::Arc;
use std::sync::Mutex;
use fnv::FnvHasher;

use device::Device;
use instance::MemoryType;
use memory::pool::HostVisibleMemoryTypePool;
use memory::pool::HostVisibleMemoryTypePoolAlloc;
use memory::pool::NonHostVisibleMemoryTypePool;
use memory::pool::NonHostVisibleMemoryTypePoolAlloc;
use memory::DeviceMemory;
use memory::MappedDeviceMemory;
use OomError;

pub struct MemoryPool {
    device: Arc<Device>,

    // For each memory type index, stores the associated pool.
    pools: Mutex<HashMap<u32, Pool, BuildHasherDefault<FnvHasher>>>,
}

impl MemoryPool {
    /// Creates a new pool.
    #[inline]
    pub fn new(device: &Arc<Device>) -> Arc<MemoryPool> {
        let cap = device.physical_device().memory_types().len();
        let hasher = BuildHasherDefault::<FnvHasher>::default();

        Arc::new(MemoryPool {
            device: device.clone(),
            pools: Mutex::new(HashMap::with_capacity_and_hasher(cap, hasher)),
        })
    }

    /// Allocates memory from the pool.
    ///
    /// # Panic
    ///
    /// - Panicks if `device` and `memory_type` don't belong to the same physical device.
    /// - Panicks if `size` is 0.
    /// - Panicks if `alignment` is 0.
    ///
    pub fn alloc(&self, memory_type: MemoryType, size: usize, alignment: usize)
                 -> Result<MemoryPoolAlloc, OomError>
    {
        let mut pools = self.pools.lock().unwrap();

        match pools.entry(memory_type.id()) {
            Entry::Occupied(entry) => {
                match entry.get() {
                    &Pool::HostVisible(ref pool) => {
                        let alloc = try!(HostVisibleMemoryTypePool::alloc(&pool, size, alignment));
                        let inner = MemoryPoolAllocInner::HostVisible(alloc);
                        Ok(MemoryPoolAlloc { inner: inner })
                    },
                    &Pool::NonHostVisible(ref pool) => {
                        let alloc = try!(NonHostVisibleMemoryTypePool::alloc(&pool, size, alignment));
                        let inner = MemoryPoolAllocInner::NonHostVisible(alloc);
                        Ok(MemoryPoolAlloc { inner: inner })
                    },
                }
            },

            Entry::Vacant(entry) => {
                match memory_type.is_host_visible() {
                    true => {
                        let pool = HostVisibleMemoryTypePool::new(&self.device, memory_type);
                        entry.insert(Pool::HostVisible(pool.clone()));
                        let alloc = try!(HostVisibleMemoryTypePool::alloc(&pool, size, alignment));
                        let inner = MemoryPoolAllocInner::HostVisible(alloc);
                        Ok(MemoryPoolAlloc { inner: inner })
                    },
                    false => {
                        let pool = NonHostVisibleMemoryTypePool::new(&self.device, memory_type);
                        entry.insert(Pool::NonHostVisible(pool.clone()));
                        let alloc = try!(NonHostVisibleMemoryTypePool::alloc(&pool, size, alignment));
                        let inner = MemoryPoolAllocInner::NonHostVisible(alloc);
                        Ok(MemoryPoolAlloc { inner: inner })
                    },
                }
            },
        }
    }
}

enum Pool {
    HostVisible(Arc<HostVisibleMemoryTypePool>),
    NonHostVisible(Arc<NonHostVisibleMemoryTypePool>),
}

pub struct MemoryPoolAlloc {
    inner: MemoryPoolAllocInner
}

impl MemoryPoolAlloc {
    #[inline]
    pub fn memory(&self) -> Memory {
        match self.inner {
            MemoryPoolAllocInner::NonHostVisible(ref mem) => Memory::Unmapped(mem.memory()),
            MemoryPoolAllocInner::HostVisible(ref mem) => Memory::Mapped(mem.memory()),
        }
    }

    #[inline]
    pub fn offset(&self) -> usize {
        match self.inner {
            MemoryPoolAllocInner::NonHostVisible(ref mem) => mem.offset(),
            MemoryPoolAllocInner::HostVisible(ref mem) => mem.offset(),
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        match self.inner {
            MemoryPoolAllocInner::NonHostVisible(ref mem) => mem.size(),
            MemoryPoolAllocInner::HostVisible(ref mem) => mem.size(),
        }
    }
}

enum MemoryPoolAllocInner {
    NonHostVisible(NonHostVisibleMemoryTypePoolAlloc),
    HostVisible(HostVisibleMemoryTypePoolAlloc),
}

pub enum Memory<'a> {
    Unmapped(&'a DeviceMemory),
    Mapped(&'a MappedDeviceMemory),
}
