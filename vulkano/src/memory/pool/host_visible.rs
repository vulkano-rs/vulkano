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
use crate::instance::Instance;
use crate::memory::DeviceMemory;
use crate::memory::DeviceMemoryAllocError;
use crate::memory::MappedDeviceMemory;
use crate::DeviceSize;
use std::cmp;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;

/// Memory pool that operates on a given memory type.
#[derive(Debug)]
pub struct StdHostVisibleMemoryTypePool {
    device: Arc<Device>,
    memory_type: u32,
    // TODO: obviously very inefficient
    occupied: Mutex<Vec<(Arc<MappedDeviceMemory>, Vec<Range<DeviceSize>>)>>,
}

impl StdHostVisibleMemoryTypePool {
    /// Creates a new pool that will operate on the given memory type.
    ///
    /// # Panic
    ///
    /// - Panics if the `device` and `memory_type` don't belong to the same physical device.
    /// - Panics if the memory type is not host-visible.
    ///
    #[inline]
    pub fn new(device: Arc<Device>, memory_type: MemoryType) -> Arc<StdHostVisibleMemoryTypePool> {
        assert_eq!(
            &**device.physical_device().instance() as *const Instance,
            &**memory_type.physical_device().instance() as *const Instance
        );
        assert_eq!(
            device.physical_device().index(),
            memory_type.physical_device().index()
        );
        assert!(memory_type.is_host_visible());

        Arc::new(StdHostVisibleMemoryTypePool {
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
        size: DeviceSize,
        alignment: DeviceSize,
    ) -> Result<StdHostVisibleMemoryTypePoolAlloc, DeviceMemoryAllocError> {
        assert!(size != 0);
        assert!(alignment != 0);

        #[inline]
        fn align(val: DeviceSize, al: DeviceSize) -> DeviceSize {
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
                    return Ok(StdHostVisibleMemoryTypePoolAlloc {
                        pool: me.clone(),
                        memory: dev_mem.clone(),
                        offset: entry1_end,
                        size: size,
                    });
                }
            }

            // Try append at the end.
            let last_end = entries.last().map(|e| align(e.end, alignment)).unwrap_or(0);
            if last_end + size <= (**dev_mem).as_ref().size() {
                entries.push(last_end..last_end + size);
                return Ok(StdHostVisibleMemoryTypePoolAlloc {
                    pool: me.clone(),
                    memory: dev_mem.clone(),
                    offset: last_end,
                    size: size,
                });
            }
        }

        // We need to allocate a new block.
        let new_block = {
            const MIN_BLOCK_SIZE: DeviceSize = 8 * 1024 * 1024; // 8 MB
            let to_alloc = cmp::max(MIN_BLOCK_SIZE, size.next_power_of_two());
            let new_block =
                DeviceMemory::alloc_and_map(me.device.clone(), me.memory_type(), to_alloc)?;
            Arc::new(new_block)
        };

        occupied.push((new_block.clone(), vec![0..size]));
        Ok(StdHostVisibleMemoryTypePoolAlloc {
            pool: me.clone(),
            memory: new_block,
            offset: 0,
            size,
        })
    }

    /// Same as `alloc` but with exportable memory fd on Linux.
    #[cfg(target_os = "linux")]
    pub fn alloc_with_exportable_fd(
        me: &Arc<Self>,
        size: DeviceSize,
        alignment: DeviceSize,
    ) -> Result<StdHostVisibleMemoryTypePoolAlloc, DeviceMemoryAllocError> {
        assert!(size != 0);
        assert!(alignment != 0);

        #[inline]
        fn align(val: DeviceSize, al: DeviceSize) -> DeviceSize {
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
                    return Ok(StdHostVisibleMemoryTypePoolAlloc {
                        pool: me.clone(),
                        memory: dev_mem.clone(),
                        offset: entry1_end,
                        size,
                    });
                }
            }

            // Try append at the end.
            let last_end = entries.last().map(|e| align(e.end, alignment)).unwrap_or(0);
            if last_end + size <= (**dev_mem).as_ref().size() {
                entries.push(last_end..last_end + size);
                return Ok(StdHostVisibleMemoryTypePoolAlloc {
                    pool: me.clone(),
                    memory: dev_mem.clone(),
                    offset: last_end,
                    size,
                });
            }
        }

        // We need to allocate a new block.
        let new_block = {
            const MIN_BLOCK_SIZE: DeviceSize = 8 * 1024 * 1024; // 8 MB
            let to_alloc = cmp::max(MIN_BLOCK_SIZE, size.next_power_of_two());
            let new_block = DeviceMemory::alloc_and_map_with_exportable_fd(
                me.device.clone(),
                me.memory_type(),
                to_alloc,
            )?;
            Arc::new(new_block)
        };

        occupied.push((new_block.clone(), vec![0..size]));
        Ok(StdHostVisibleMemoryTypePoolAlloc {
            pool: me.clone(),
            memory: new_block,
            offset: 0,
            size,
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
pub struct StdHostVisibleMemoryTypePoolAlloc {
    pool: Arc<StdHostVisibleMemoryTypePool>,
    memory: Arc<MappedDeviceMemory>,
    offset: DeviceSize,
    size: DeviceSize,
}

impl StdHostVisibleMemoryTypePoolAlloc {
    #[inline]
    pub fn memory(&self) -> &MappedDeviceMemory {
        &self.memory
    }

    #[inline]
    pub fn offset(&self) -> DeviceSize {
        self.offset
    }

    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.size
    }
}

impl Drop for StdHostVisibleMemoryTypePoolAlloc {
    fn drop(&mut self) {
        let mut occupied = self.pool.occupied.lock().unwrap();

        let entries = occupied
            .iter_mut()
            .find(|e| &*e.0 as *const MappedDeviceMemory == &*self.memory)
            .unwrap();

        entries.1.retain(|e| e.start != self.offset);
    }
}
