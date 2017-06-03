// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::fmt;
use std::mem;
use std::ptr;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Range;
use std::os::raw::c_void;
use std::sync::Arc;

use instance::MemoryType;
use device::Device;
use device::DeviceOwned;
use memory::Content;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Represents memory that has been allocated.
///
/// The destructor of `DeviceMemory` automatically frees the memory.
///
/// # Example
///
/// ```
/// use vulkano::memory::DeviceMemory;
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let mem_ty = device.physical_device().memory_types().next().unwrap();
/// 
/// // Allocates 1kB of memory.
/// let memory = DeviceMemory::alloc(device.clone(), mem_ty, 1024).unwrap();
/// ```
pub struct DeviceMemory {
    memory: vk::DeviceMemory,
    device: Arc<Device>,
    size: usize,
    memory_type_index: u32,
}

impl DeviceMemory {
    /// Allocates a chunk of memory from the device.
    ///
    /// Some platforms may have a limit on the maximum size of a single allocation. For example,
    /// certain systems may fail to create allocations with a size greater than or equal to 4GB. 
    ///
    /// # Panic
    ///
    /// - Panics if `size` is 0.
    /// - Panics if `memory_type` doesn't belong to the same physical device as `device`.
    ///
    // TODO: VK_ERROR_TOO_MANY_OBJECTS error
    #[inline]
    pub fn alloc(device: Arc<Device>, memory_type: MemoryType, size: usize)
                 -> Result<DeviceMemory, OomError>
    {
        assert!(size >= 1);
        assert_eq!(device.physical_device().internal_object(),
                   memory_type.physical_device().internal_object());

        if size > memory_type.heap().size() {
            return Err(OomError::OutOfDeviceMemory);
        }

        let memory = unsafe {
            let vk = device.pointers();

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
            device: device,
            size: size,
            memory_type_index: memory_type.id(),
        })
    }

    /// Allocates a chunk of memory and maps it.
    ///
    /// # Panic
    ///
    /// - Panics if `memory_type` doesn't belong to the same physical device as `device`.
    /// - Panics if the memory type is not host-visible.
    ///
    pub fn alloc_and_map(device: Arc<Device>, memory_type: MemoryType, size: usize)
                         -> Result<MappedDeviceMemory, OomError>
    {
        let vk = device.pointers();

        assert!(memory_type.is_host_visible());
        let mem = try!(DeviceMemory::alloc(device.clone(), memory_type, size));

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
}

unsafe impl DeviceOwned for DeviceMemory {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl fmt::Debug for DeviceMemory {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("DeviceMemory")
            .field("device", &*self.device)
            .field("memory_type", &self.memory_type())
            .field("size", &self.size)
            .finish()
    }
}

unsafe impl VulkanObject for DeviceMemory {
    type Object = vk::DeviceMemory;

    #[inline]
    fn internal_object(&self) -> vk::DeviceMemory {
        self.memory
    }
}

impl Drop for DeviceMemory {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.FreeMemory(self.device.internal_object(), self.memory, ptr::null());
        }
    }
}

/// Represents memory that has been allocated and mapped in CPU accessible space.
///
/// Can be obtained with `DeviceMemory::alloc_and_map`. The function will panic if the memory type
/// is not host-accessible.
///
/// In order to access the content of the allocated memory, you can use the `read_write` method.
/// This method returns a guard object that derefs to the content. 
///
/// # Example
///
/// ```
/// use vulkano::memory::DeviceMemory;
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// // The memory type must be mappable. 
/// let mem_ty = device.physical_device().memory_types()
///                     .filter(|t| t.is_host_visible())
///                     .next().unwrap();    // Vk specs guarantee that this can't fail
///
/// // Allocates 1kB of memory.
/// let memory = DeviceMemory::alloc_and_map(device.clone(), mem_ty, 1024).unwrap();
///
/// // Get access to the content. Note that this is very unsafe for two reasons: 1) the content is
/// // uninitialized, and 2) the access is unsynchronized.
/// unsafe {
///     let mut content = memory.read_write::<[u8]>(0 .. 1024);
///     content[12] = 54;       // `content` derefs to a `&[u8]` or a `&mut [u8]`
/// }
/// ```
pub struct MappedDeviceMemory {
    memory: DeviceMemory,
    pointer: *mut c_void,
    coherent: bool,
}

// Note that `MappedDeviceMemory` doesn't implement `Drop`, as we don't need to unmap memory before
// freeing it.
//
// Vulkan specs, documentation of `vkFreeMemory`:
// > If a memory object is mapped at the time it is freed, it is implicitly unmapped.
//

impl MappedDeviceMemory {
    /// Unmaps the memory. It will no longer be accessible from the CPU.
    pub fn unmap(self) -> DeviceMemory {
        unsafe {
            let device = self.memory.device();
            let vk = device.pointers();
            vk.UnmapMemory(device.internal_object(), self.memory.memory);
        }

        self.memory
    }

    /// Gives access to the content of the memory.
    ///
    /// This function takes care of calling `vkInvalidateMappedMemoryRanges` and
    /// `vkFlushMappedMemoryRanges` on the given range. You are therefore encouraged to use the
    /// smallest range as possible, and to not call this function multiple times in a row for
    /// several small changes.
    ///
    /// # Safety
    ///
    /// - Type safety is not checked. You must ensure that `T` corresponds to the content of the
    ///   buffer.
    /// - Accesses are not synchronized. Synchronization must be handled outside of
    ///   the `MappedDeviceMemory`.
    ///
    #[inline]
    pub unsafe fn read_write<T: ?Sized>(&self, range: Range<usize>) -> CpuAccess<T>
        where T: Content
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

impl AsRef<DeviceMemory> for MappedDeviceMemory {
    #[inline]
    fn as_ref(&self) -> &DeviceMemory {
        &self.memory
    }
}

impl AsMut<DeviceMemory> for MappedDeviceMemory {
    #[inline]
    fn as_mut(&mut self) -> &mut DeviceMemory {
        &mut self.memory
    }
}

unsafe impl DeviceOwned for MappedDeviceMemory {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.memory.device()
    }
}

unsafe impl Send for MappedDeviceMemory {}
unsafe impl Sync for MappedDeviceMemory {}

impl fmt::Debug for MappedDeviceMemory {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_tuple("MappedDeviceMemory")
            .field(&self.memory)
            .finish()
    }
}

/// Object that can be used to read or write the content of a `MappedDeviceMemory`.
///
/// This object derefs to the content, just like a `MutexGuard` for example.
pub struct CpuAccess<'a, T: ?Sized + 'a> {
    pointer: *mut T,
    mem: &'a MappedDeviceMemory,
    coherent: bool,
    range: Range<usize>,
}

impl<'a, T: ?Sized + 'a> CpuAccess<'a, T> {
    /// Builds a new `CpuAccess` to access a sub-part of the current `CpuAccess`.
    ///
    /// This function is unstable. Don't use it directly.
    // TODO: unsafe?
    // TODO: decide what to do with this
    #[doc(hidden)]
    #[inline]
    pub fn map<U: ?Sized + 'a, F>(self, f: F) -> CpuAccess<'a, U>
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

unsafe impl<'a, T: ?Sized + 'a> Send for CpuAccess<'a, T> {}
unsafe impl<'a, T: ?Sized + 'a> Sync for CpuAccess<'a, T> {}

impl<'a, T: ?Sized + 'a> Deref for CpuAccess<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for CpuAccess<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer }
    }
}

impl<'a, T: ?Sized + 'a> Drop for CpuAccess<'a, T> {
    #[inline]
    fn drop(&mut self) {
        // If the memory doesn't have the `coherent` flag, we need to flush the data.
        if !self.coherent {
            let vk = self.mem.as_ref().device().pointers();

            let range = vk::MappedMemoryRange {
                sType: vk::STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                pNext: ptr::null(),
                memory: self.mem.as_ref().internal_object(),
                offset: self.range.start as u64,
                size: (self.range.end - self.range.start) as u64,
            };

            // TODO: check result?
            unsafe {
                vk.FlushMappedMemoryRanges(self.mem.as_ref().device().internal_object(),
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
        let _ = DeviceMemory::alloc(device.clone(), mem_ty, 256).unwrap();
    }

    #[test]
    #[should_panic]
    fn zero_size() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device.physical_device().memory_types().next().unwrap();
        let _ = DeviceMemory::alloc(device.clone(), mem_ty, 0);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn oom_single() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device.physical_device().memory_types().filter(|m| !m.is_lazily_allocated())
                           .next().unwrap();
    
        match DeviceMemory::alloc(device.clone(), mem_ty, 0xffffffffffffffff) {
            Err(OomError::OutOfDeviceMemory) => (),
            _ => panic!()
        }
    }

    #[test]
    #[ignore]       // TODO: test fails for now on Mesa+Intel
    fn oom_multi() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device.physical_device().memory_types().filter(|m| !m.is_lazily_allocated())
                           .next().unwrap();
        let heap_size = mem_ty.heap().size();

        let mut allocs = Vec::new();
    
        for _ in 0 .. 4 {
            match DeviceMemory::alloc(device.clone(), mem_ty, heap_size / 3) {
                Err(OomError::OutOfDeviceMemory) => return,     // test succeeded
                Ok(a) => allocs.push(a),
                _ => ()
            }
        }

        panic!()
    }
}
