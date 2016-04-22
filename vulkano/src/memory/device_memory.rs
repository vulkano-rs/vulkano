// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::ptr;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Range;
use std::os::raw::c_void;
use std::sync::Arc;

use instance::MemoryType;
use device::Device;
use memory::Content;
use OomError;
use SafeDeref;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Represents memory that has been allocated.
#[derive(Debug)]
pub struct DeviceMemory<D = Arc<Device>> where D: SafeDeref<Target = Device> {
    memory: vk::DeviceMemory,
    device: D,
    size: usize,
    memory_type_index: u32,
}

impl<D> DeviceMemory<D> where D: SafeDeref<Target = Device> {
    /// Allocates a chunk of memory from the device.
    ///
    /// Some platforms may have a limit on the maximum size of a single allocation. For example,
    /// certain systems may fail to create allocations with a size greater than or equal to 4GB. 
    ///
    /// # Panic
    ///
    /// - Panicks if `size` is 0.
    /// - Panicks if `memory_type` doesn't belong to the same physical device as `device`.
    ///
    // TODO: VK_ERROR_TOO_MANY_OBJECTS error
    #[inline]
    pub fn alloc(device: &D, memory_type: &MemoryType, size: usize)
                 -> Result<DeviceMemory<D>, OomError>
        where D: Clone
    {
        assert!(size >= 1);
        assert_eq!(device.physical_device().internal_object(),
                   memory_type.physical_device().internal_object());

        if size > memory_type.heap().size() {
            return Err(OomError::OutOfDeviceMemory);
        }

        let vk = device.pointers();

        let memory = unsafe {
            let infos = vk::MemoryAllocateInfo {
                sType: vk::STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                pNext: ptr::null(),
                allocationSize: size as u64,
                memoryTypeIndex: memory_type.id(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.AllocateMemory(device.internal_object(), &infos,
                                                ptr::null(), &mut output)));
            output
        };

        Ok(DeviceMemory {
            memory: memory,
            device: device.clone(),
            size: size,
            memory_type_index: memory_type.id(),
        })
    }

    /// Allocates a chunk of memory and maps it.
    ///
    /// # Panic
    ///
    /// - Panicks if `memory_type` doesn't belong to the same physical device as `device`.
    /// - Panicks if the memory type is not host-visible.
    ///
    pub fn alloc_and_map(device: &D, memory_type: &MemoryType, size: usize)
                         -> Result<MappedDeviceMemory<D>, OomError>
        where D: Clone
    {
        let vk = device.pointers();

        assert!(memory_type.is_host_visible());
        let mem = try!(DeviceMemory::alloc(device, memory_type, size));

        let coherent = memory_type.is_host_coherent();

        let ptr = unsafe {
            let mut output = mem::uninitialized();
            try!(check_errors(vk.MapMemory(device.internal_object(), mem.memory, 0,
                                           mem.size as vk::DeviceSize, 0 /* reserved flags */,
                                           &mut output)));
            output
        };

        Ok(MappedDeviceMemory {
            memory: mem,
            pointer: ptr,
            coherent: coherent,
        })
    }

    /// Returns the memory type this chunk was allocated on.
    #[inline]
    pub fn memory_type(&self) -> MemoryType {
        self.device.physical_device().memory_type_by_id(self.memory_type_index).unwrap()
    }

    /// Returns the size in bytes of that memory chunk.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the device associated with this allocation.
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

unsafe impl<D> VulkanObject for DeviceMemory<D> where D: SafeDeref<Target = Device> {
    type Object = vk::DeviceMemory;

    #[inline]
    fn internal_object(&self) -> vk::DeviceMemory {
        self.memory
    }
}

impl<D> Drop for DeviceMemory<D> where D: SafeDeref<Target = Device> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let device = self.device();
            let vk = device.pointers();
            vk.FreeMemory(device.internal_object(), self.memory, ptr::null());
        }
    }
}

/// Represents memory that has been allocated and mapped in CPU accessible space.
#[derive(Debug)]
pub struct MappedDeviceMemory<D = Arc<Device>> where D: SafeDeref<Target = Device> {
    memory: DeviceMemory<D>,
    pointer: *mut c_void,
    coherent: bool,
}

impl<D> MappedDeviceMemory<D> where D: SafeDeref<Target = Device> {
    /// Returns the underlying `DeviceMemory`.
    // TODO: impl AsRef instead
    #[inline]
    pub fn memory(&self) -> &DeviceMemory<D> {
        &self.memory
    }

    /// Gives access to the content of the memory.
    ///
    /// # Safety
    ///
    /// - Type safety is not checked. You must ensure that `T` corresponds to the content of the
    ///   buffer.
    /// - Accesses are not synchronized. Synchronization must be handled outside of
    ///   the `MappedDeviceMemory`.
    ///
    #[inline]
    pub unsafe fn read_write<T: ?Sized>(&self, range: Range<usize>) -> CpuAccess<T, D>
        where T: Content + 'static
    {
        let vk = self.memory.device().pointers();
        let pointer = T::ref_from_ptr((self.pointer as usize + range.start) as *mut _,
                                      range.end - range.start).unwrap();       // TODO: error

        if !self.coherent {
            let range = vk::MappedMemoryRange {
                sType: vk::STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                pNext: ptr::null(),
                memory: self.memory.internal_object(),
                offset: range.start as u64,
                size: (range.end - range.start) as u64,
            };

            // TODO: check result?
            vk.InvalidateMappedMemoryRanges(self.memory.device().internal_object(), 1, &range);
        }

        CpuAccess {
            pointer: pointer,
            mem: self,
            coherent: self.coherent,
            range: range,
        }
    }
}

unsafe impl<D> Send for MappedDeviceMemory<D> where D: SafeDeref<Target = Device> {}
unsafe impl<D> Sync for MappedDeviceMemory<D> where D: SafeDeref<Target = Device> {}

impl<D> Drop for MappedDeviceMemory<D> where D: SafeDeref<Target = Device> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let device = self.memory.device();
            let vk = device.pointers();
            vk.UnmapMemory(device.internal_object(), self.memory.memory);
        }
    }
}

/// Object that can be used to read or write the content of a `MappedDeviceMemory`.
pub struct CpuAccess<'a, T: ?Sized + 'a, D = Arc<Device>> where D: SafeDeref<Target = Device> + 'a {
    pointer: *mut T,
    mem: &'a MappedDeviceMemory<D>,
    coherent: bool,
    range: Range<usize>,
}

impl<'a, T: ?Sized + 'a, D: 'a> CpuAccess<'a, T, D> where D: SafeDeref<Target = Device> {
    /// Makes a new `CpuAccess` to access a sub-part of the current `CpuAccess`.
    #[inline]
    pub fn map<U: ?Sized + 'a, F>(self, f: F) -> CpuAccess<'a, U, D>
        where F: FnOnce(*mut T) -> *mut U
    {
        CpuAccess {
            pointer: f(self.pointer),
            mem: self.mem,
            coherent: self.coherent,
            range: self.range.clone(),  // TODO: ?
        }
    }
}

unsafe impl<'a, T: ?Sized + 'a, D: 'a> Send for CpuAccess<'a, T, D> where D: SafeDeref<Target = Device> {}
unsafe impl<'a, T: ?Sized + 'a, D: 'a> Sync for CpuAccess<'a, T, D> where D: SafeDeref<Target = Device> {}

impl<'a, T: ?Sized + 'a, D: 'a> Deref for CpuAccess<'a, T, D> where D: SafeDeref<Target = Device> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

impl<'a, T: ?Sized + 'a, D: 'a> DerefMut for CpuAccess<'a, T, D> where D: SafeDeref<Target = Device> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer }
    }
}

impl<'a, T: ?Sized + 'a, D: 'a> Drop for CpuAccess<'a, T, D> where D: SafeDeref<Target = Device> {
    #[inline]
    fn drop(&mut self) {
        // If the memory doesn't have the `coherent` flag, we need to flush the data.
        if !self.coherent {
            let vk = self.mem.memory().device().pointers();

            let range = vk::MappedMemoryRange {
                sType: vk::STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                pNext: ptr::null(),
                memory: self.mem.memory().internal_object(),
                offset: self.range.start as u64,
                size: (self.range.end - self.range.start) as u64,
            };

            // TODO: check result?
            unsafe {
                vk.FlushMappedMemoryRanges(self.mem.memory().device().internal_object(),
                                           1, &range);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use OomError;
    use memory::DeviceMemory;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device.physical_device().memory_types().next().unwrap();
        let _ = DeviceMemory::alloc(&device, &mem_ty, 256).unwrap();
    }

    #[test]
    #[should_panic]
    fn zero_size() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device.physical_device().memory_types().next().unwrap();
        let _ = DeviceMemory::alloc(&device, &mem_ty, 0);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn oom_single() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device.physical_device().memory_types().filter(|m| !m.is_lazily_allocated())
                           .next().unwrap();
    
        match DeviceMemory::alloc(&device, &mem_ty, 0xffffffffffffffff) {
            Err(OomError::OutOfDeviceMemory) => (),
            _ => panic!()
        }
    }

    #[test]
    #[ignore]       // TODO: fails on AMD + Windows
    fn oom_multi() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device.physical_device().memory_types().filter(|m| !m.is_lazily_allocated())
                           .next().unwrap();
        let heap_size = mem_ty.heap().size();
    
        for _ in 0 .. 4 {
            match DeviceMemory::alloc(&device, &mem_ty, heap_size / 3) {
                Err(OomError::OutOfDeviceMemory) => return,     // test succeeded
                _ => ()
            }
        }

        panic!()
    }
}
