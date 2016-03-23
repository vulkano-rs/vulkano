use std::mem;
use std::ptr;
use std::ops::Deref;
use std::ops::DerefMut;
use std::os::raw::c_void;
use std::sync::Arc;

use instance::MemoryType;
use device::Device;
use memory::Content;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Represents memory that has been allocated.
pub struct DeviceMemory {
    device: Arc<Device>,
    memory: vk::DeviceMemory,
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
    /// - Panicks if `size` is 0.
    /// - Panicks if `memory_type` doesn't belong to the same physical device as `device`.
    ///
    // TODO: VK_ERROR_TOO_MANY_OBJECTS error
    #[inline]
    pub fn alloc(device: &Arc<Device>, memory_type: &MemoryType, size: usize)
                 -> Result<DeviceMemory, OomError>
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
            device: device.clone(),
            memory: memory,
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
    pub fn alloc_and_map(device: &Arc<Device>, memory_type: &MemoryType, size: usize)
                         -> Result<MappedDeviceMemory, OomError>
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
    pub fn device(&self) -> &Arc<Device> {
        &self.device
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
pub struct MappedDeviceMemory {
    memory: DeviceMemory,
    pointer: *mut c_void,
    coherent: bool,
}

impl MappedDeviceMemory {
    /// Returns the underlying `DeviceMemory`.
    // TODO: impl AsRef instead
    #[inline]
    pub fn memory(&self) -> &DeviceMemory {
        &self.memory
    }

    // TODO: remove
    #[inline]
    pub fn mapping_pointer(&self) -> *mut c_void {
        self.pointer
    }

    #[inline]
    pub unsafe fn read<T>(&self) -> CpuAccess<T> where T: Content + 'static {
        self.write()
    }

    #[inline]
    pub unsafe fn write<T>(&self) -> CpuAccess<T> where T: Content + 'static {
        let vk = self.memory.device().pointers();
        let pointer = T::ref_from_ptr(self.pointer, self.memory.size()).unwrap();       // TODO: error

        if !self.coherent {
            let range = vk::MappedMemoryRange {
                sType: vk::STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                pNext: ptr::null(),
                memory: self.memory.internal_object(),
                offset: 0,
                size: vk::WHOLE_SIZE,
            };

            // TODO: check result?
            vk.InvalidateMappedMemoryRanges(self.memory.device().internal_object(), 1, &range);
        }

        CpuAccess {
            mem: self,
            coherent: self.coherent,
            pointer: pointer,
        }
    }
}

impl Drop for MappedDeviceMemory {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.memory.device.pointers();
            vk.UnmapMemory(self.memory.device.internal_object(), self.memory.memory);
        }
    }
}

/// Object that can be used to read or write the content of a `MappedDeviceMemory`.
///
/// Note that this object holds a mutex guard on the chunk. If another thread tries to access
/// this memory's content or tries to submit a GPU command that uses this memory, it will block.
pub struct CpuAccess<'a, T: ?Sized + 'a> {
    pointer: *mut T,
    mem: &'a MappedDeviceMemory,
    coherent: bool,
}

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
        if !self.coherent {
            let vk = self.mem.memory().device().pointers();

            let range = vk::MappedMemoryRange {
                sType: vk::STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                pNext: ptr::null(),
                memory: self.mem.memory().internal_object(),
                offset: 0,
                size: vk::WHOLE_SIZE,
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
