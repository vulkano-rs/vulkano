// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;

use device::Device;
use instance::Instance;
use instance::MemoryType;
use memory::DeviceMemory;
use memory::DeviceMemoryAllocError;

/// Memory pool that operates on a given memory type.
#[derive(Debug)]
pub struct StdNonHostVisibleMemoryTypePool {
    device: Arc<Device>,
    memory_type: u32,
    // TODO: obviously very inefficient
    occupied: Mutex<Vec<(Arc<DeviceMemory>, Vec<Range<usize>>)>>,
}

impl StdNonHostVisibleMemoryTypePool {
    /// Creates a new pool that will operate on the given memory type.
    ///
    /// # Panic
    ///
    /// - Panics if the `device` and `memory_type` don't belong to the same physical device.
    ///
    #[inline]
    pub fn new(
        device: Arc<Device>,
        memory_type: MemoryType,
    ) -> Arc<StdNonHostVisibleMemoryTypePool> {
        assert_eq!(
            &**device.physical_device().instance() as *const Instance,
            &**memory_type.physical_device().instance() as *const Instance
        );
        assert_eq!(
            device.physical_device().index(),
            memory_type.physical_device().index()
        );

        Arc::new(StdNonHostVisibleMemoryTypePool {
            device: device.clone(),
            memory_type: memory_type.id(),
            occupied: Mutex::new(Vec::new()),
        })
    }

    /// Allocates memory from the pool.
    ///
    /// # Panic
    ///
    /// - Panics if `size` is 0.
    /// - Panics if `alignment` is 0.
    ///
    pub fn alloc(
        me: &Arc<Self>,
        size: usize,
        alignment: usize,
    ) -> Result<StdNonHostVisibleMemoryTypePoolAlloc, DeviceMemoryAllocError> {
        assert!(size != 0);
        assert!(alignment != 0);

        #[inline]
        fn align(val: usize, al: usize) -> usize {
            al * (1 + (val - 1) / al)
        }

        // Find a location.
        let mut occupied = me.occupied.lock().unwrap();

        // Try finding an entry in already-allocated chunks.
        for &mut (ref dev_mem, ref mut entries) in occupied.iter_mut() {
            // Try find some free space in-between two entries.
            for i in 0..entries.len().saturating_sub(1) {
                let entry1 = entries[i].clone();
                let entry1_end = align(entry1.end, alignment);
                let entry2 = entries[i + 1].clone();
                if entry1_end + size <= entry2.start {
                    entries.insert(i + 1, entry1_end..entry1_end + size);
                    return Ok(StdNonHostVisibleMemoryTypePoolAlloc {
                        pool: me.clone(),
                        memory: dev_mem.clone(),
                        offset: entry1_end,
                        size: size,
                    });
                }
            }

            // Try append at the end.
            let last_end = entries.last().map(|e| align(e.end, alignment)).unwrap_or(0);
            if last_end + size <= dev_mem.size() {
                entries.push(last_end..last_end + size);
                return Ok(StdNonHostVisibleMemoryTypePoolAlloc {
                    pool: me.clone(),
                    memory: dev_mem.clone(),
                    offset: last_end,
                    size: size,
                });
            }
        }

        // We need to allocate a new block.
        let new_block = {
            const MIN_BLOCK_SIZE: usize = 8 * 1024 * 1024; // 8 MB
            let to_alloc = cmp::max(MIN_BLOCK_SIZE, size.next_power_of_two());
            let new_block = DeviceMemory::alloc(me.device.clone(), me.memory_type(), to_alloc)?;
            Arc::new(new_block)
        };

        occupied.push((new_block.clone(), vec![0..size]));
        Ok(StdNonHostVisibleMemoryTypePoolAlloc {
            pool: me.clone(),
            memory: new_block,
            offset: 0,
            size: size,
        })
    }

    /// Returns the device this pool operates on.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the memory type this pool operates on.
    #[inline]
    pub fn memory_type(&self) -> MemoryType {
        self.device
            .physical_device()
            .memory_type_by_id(self.memory_type)
            .unwrap()
    }
}

#[derive(Debug)]
pub struct StdNonHostVisibleMemoryTypePoolAlloc {
    pool: Arc<StdNonHostVisibleMemoryTypePool>,
    memory: Arc<DeviceMemory>,
    offset: usize,
    size: usize,
}

impl StdNonHostVisibleMemoryTypePoolAlloc {
    #[inline]
    pub fn memory(&self) -> &DeviceMemory {
        &self.memory
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for StdNonHostVisibleMemoryTypePoolAlloc {
    fn drop(&mut self) {
        let mut occupied = self.pool.occupied.lock().unwrap();

        let entries = occupied
            .iter_mut()
            .find(|e| &*e.0 as *const DeviceMemory == &*self.memory)
            .unwrap();

        entries.1.retain(|e| e.start != self.offset);
    }
}
