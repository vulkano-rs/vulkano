use std::mem;
use std::ptr;
use std::os::raw::c_void;
use std::sync::Arc;

use instance::MemoryType;
use device::Device;
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
    /// # Panic
    ///
    /// - Panicks if `memory_type` doesn't belong to the same physical device as `device`.
    ///
    #[inline]
    pub fn alloc(device: &Arc<Device>, memory_type: &MemoryType, size: usize)
                 -> Result<DeviceMemory, OomError>
    {
        assert_eq!(device.physical_device().internal_object(),
                   memory_type.physical_device().internal_object());

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

impl VulkanObject for DeviceMemory {
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
}

impl MappedDeviceMemory {
    /// Returns the underlying `DeviceMemory`.
    #[inline]
    pub fn memory(&self) -> &DeviceMemory {
        &self.memory
    }

    /// Returns a pointer to the mapping.
    ///
    /// Note that access to this pointer is not safe at all.
    pub fn mapping_pointer(&self) -> *mut c_void {
        self.pointer
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
