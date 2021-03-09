// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Range;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::os::unix::io::FromRawFd;

use check_errors;
use device::Device;
use device::DeviceOwned;
use instance::MemoryType;
use memory::Content;
use memory::DedicatedAlloc;
use memory::ExternalMemoryHandleType;
use vk;
use Error;
use OomError;
use VulkanObject;

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
/// // Allocates 1KB of memory.
/// let memory = DeviceMemory::alloc(device.clone(), mem_ty, 1024).unwrap();
/// ```
pub struct DeviceMemory {
    memory: vk::DeviceMemory,
    device: Arc<Device>,
    size: usize,
    memory_type_index: u32,
    handle_types: ExternalMemoryHandleType,
}

/// Represents a builder for the device memory object.
///
/// # Example
///
/// ```
/// use vulkano::memory::DeviceMemoryBuilder;
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let mem_ty = device.physical_device().memory_types().next().unwrap();
///
/// // Allocates 1KB of memory.
/// let memory = DeviceMemoryBuilder::new(device, mem_ty, 1024).build().unwrap();
/// ```
pub struct DeviceMemoryBuilder<'a> {
    device: Arc<Device>,
    memory_type: MemoryType<'a>,
    allocate: vk::MemoryAllocateInfo,
    dedicated_info: Option<vk::MemoryDedicatedAllocateInfoKHR>,
    export_info: Option<vk::ExportMemoryAllocateInfo>,
    import_info: Option<vk::ImportMemoryFdInfoKHR>,
    handle_types: ExternalMemoryHandleType,
}

impl<'a> DeviceMemoryBuilder<'a> {
    /// Returns a new `DeviceMemoryBuilder` given the required device, memory type and size fields.
    ///
    /// # Panic
    ///
    /// - Panics if `size` is 0.
    /// - Panics if `memory_type` doesn't belong to the same physical device as `device`.
    pub fn new(device: Arc<Device>, memory_type: MemoryType, size: usize) -> DeviceMemoryBuilder {
        assert!(size > 0);
        assert_eq!(
            device.physical_device().internal_object(),
            memory_type.physical_device().internal_object()
        );

        let allocate = vk::MemoryAllocateInfo {
            sType: vk::STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: ptr::null(),
            allocationSize: size as u64,
            memoryTypeIndex: memory_type.id(),
        };

        DeviceMemoryBuilder {
            device,
            memory_type,
            allocate,
            dedicated_info: None,
            export_info: None,
            import_info: None,
            handle_types: ExternalMemoryHandleType::none(),
        }
    }

    /// Sets an optional field for dedicated allocations in the `DeviceMemoryBuilder`.  To maintain
    /// backwards compatibility, this function does nothing when dedicated allocation has not been
    /// enabled on the device.
    ///
    /// # Panic
    ///
    /// - Panics if the dedicated allocation info has already been set.
    pub fn dedicated_info(mut self, dedicated: DedicatedAlloc<'a>) -> DeviceMemoryBuilder {
        assert!(self.dedicated_info.is_none());

        if self.device.loaded_extensions().khr_dedicated_allocation {
            self.dedicated_info = match dedicated {
                DedicatedAlloc::Buffer(buffer) => Some(vk::MemoryDedicatedAllocateInfoKHR {
                    sType: vk::STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
                    pNext: ptr::null(),
                    image: 0,
                    buffer: buffer.internal_object(),
                }),
                DedicatedAlloc::Image(image) => Some(vk::MemoryDedicatedAllocateInfoKHR {
                    sType: vk::STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
                    pNext: ptr::null(),
                    image: image.internal_object(),
                    buffer: 0,
                }),
                DedicatedAlloc::None => return self,
            };

            let ptr = self
                .dedicated_info
                .as_ref()
                .map(|i| i as *const vk::MemoryDedicatedAllocateInfoKHR)
                .unwrap_or(ptr::null()) as *const _;

            if let Some(ref mut export_info) = self.export_info {
                export_info.pNext = ptr;
            } else {
                self.allocate.pNext = ptr;
            }
        }

        self
    }

    /// Sets an optional field for exportable allocations in the `DeviceMemoryBuilder`.
    ///
    /// # Panic
    ///
    /// - Panics if the export info has already been set.
    /// - Panics if the extensions associated with `handle_types` have not been loaded by the
    ///   by the device.
    pub fn export_info(
        mut self,
        handle_types: ExternalMemoryHandleType,
    ) -> DeviceMemoryBuilder<'a> {
        assert!(self.export_info.is_none());
        // TODO: check exportFromImportedHandleTypes instead.
        assert!(self.import_info.is_none());

        // Only extensions tested with Vulkano so far.
        assert!(self.device.loaded_extensions().khr_external_memory);
        assert!(self.device.loaded_extensions().khr_external_memory_fd);

        let handle_bits = handle_types.to_bits();
        if handle_bits & vk::EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT != 0 {
            assert!(self.device.loaded_extensions().ext_external_memory_dmabuf);
        }

        let unsupported = handle_bits
            & !(vk::EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                | vk::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT);
        assert!(unsupported == 0);

        let export_info = vk::ExportMemoryAllocateInfo {
            sType: vk::STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            pNext: ptr::null(),
            handleTypes: handle_bits,
        };

        self.export_info = Some(export_info);
        let ptr = self
            .export_info
            .as_ref()
            .map(|i| i as *const vk::ExportMemoryAllocateInfo)
            .unwrap_or(ptr::null()) as *const _;

        if let Some(ref mut dedicated_info) = self.dedicated_info {
            dedicated_info.pNext = ptr;
        } else {
            self.allocate.pNext = ptr;
        }

        self.handle_types = handle_types;
        self
    }

    /// Creates a `DeviceMemory` object on success, consuming the `DeviceMemoryBuilder`.  An error
    /// is returned if the requested allocation is too large or if the total number of allocations
    /// would exceed per-device limits.
    pub fn build(self) -> Result<DeviceMemory, DeviceMemoryAllocError> {
        // Note: This check is disabled because MoltenVK doesn't report correct heap sizes yet.
        // This check was re-enabled because Mesa aborts if `size` is Very Large.
        //
        // Conversions won't panic since it's based on `vkDeviceSize`, which is a u64 in the VK
        // header.  Not sure why we bother with usizes.
        let reported_heap_size = self.memory_type.heap().size() as u64;
        if reported_heap_size != 0 && self.allocate.allocationSize > reported_heap_size {
            return Err(DeviceMemoryAllocError::OomError(
                OomError::OutOfDeviceMemory,
            ));
        }

        let memory = unsafe {
            let physical_device = self.device.physical_device();
            let mut allocation_count = self
                .device
                .allocation_count()
                .lock()
                .expect("Poisoned mutex");
            if *allocation_count >= physical_device.limits().max_memory_allocation_count() {
                return Err(DeviceMemoryAllocError::TooManyObjects);
            }
            let vk = self.device.pointers();

            let mut output = MaybeUninit::uninit();
            check_errors(vk.AllocateMemory(
                self.device.internal_object(),
                &self.allocate,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            *allocation_count += 1;
            output.assume_init()
        };

        Ok(DeviceMemory {
            memory: memory,
            device: self.device,
            size: self.allocate.allocationSize as usize,
            memory_type_index: self.memory_type.id(),
            handle_types: self.handle_types,
        })
    }
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
    #[inline]
    pub fn alloc(
        device: Arc<Device>,
        memory_type: MemoryType,
        size: usize,
    ) -> Result<DeviceMemory, DeviceMemoryAllocError> {
        DeviceMemoryBuilder::new(device, memory_type, size).build()
    }

    /// Same as `alloc`, but allows specifying a resource that will be bound to the memory.
    ///
    /// If a buffer or an image is specified in `resource`, then the returned memory must not be
    /// bound to a different buffer or image.
    ///
    /// If the `VK_KHR_dedicated_allocation` extension is enabled on the device, then it will be
    /// used by this method. Otherwise the `resource` parameter will be ignored.
    #[inline]
    pub fn dedicated_alloc(
        device: Arc<Device>,
        memory_type: MemoryType,
        size: usize,
        resource: DedicatedAlloc,
    ) -> Result<DeviceMemory, DeviceMemoryAllocError> {
        DeviceMemoryBuilder::new(device, memory_type, size)
            .dedicated_info(resource)
            .build()
    }

    /// Allocates a chunk of memory and maps it.
    ///
    /// # Panic
    ///
    /// - Panics if `memory_type` doesn't belong to the same physical device as `device`.
    /// - Panics if the memory type is not host-visible.
    ///
    #[inline]
    pub fn alloc_and_map(
        device: Arc<Device>,
        memory_type: MemoryType,
        size: usize,
    ) -> Result<MappedDeviceMemory, DeviceMemoryAllocError> {
        DeviceMemory::dedicated_alloc_and_map(device, memory_type, size, DedicatedAlloc::None)
    }

    /// Equivalent of `dedicated_alloc` for `alloc_and_map`.
    pub fn dedicated_alloc_and_map(
        device: Arc<Device>,
        memory_type: MemoryType,
        size: usize,
        resource: DedicatedAlloc,
    ) -> Result<MappedDeviceMemory, DeviceMemoryAllocError> {
        let vk = device.pointers();

        assert!(memory_type.is_host_visible());
        let mem = DeviceMemory::dedicated_alloc(device.clone(), memory_type, size, resource)?;

        Self::map_allocation(device.clone(), mem)
    }

    /// Same as `alloc`, but allows exportable file descriptor on Linux.
    #[inline]
    #[cfg(target_os = "linux")]
    pub fn alloc_with_exportable_fd(
        device: Arc<Device>,
        memory_type: MemoryType,
        size: usize,
    ) -> Result<DeviceMemory, DeviceMemoryAllocError> {
        DeviceMemoryBuilder::new(device, memory_type, size)
            .export_info(ExternalMemoryHandleType {
                opaque_fd: true,
                ..ExternalMemoryHandleType::none()
            })
            .build()
    }

    /// Same as `dedicated_alloc`, but allows exportable file descriptor on Linux.
    #[inline]
    #[cfg(target_os = "linux")]
    pub fn dedicated_alloc_with_exportable_fd(
        device: Arc<Device>,
        memory_type: MemoryType,
        size: usize,
        resource: DedicatedAlloc,
    ) -> Result<DeviceMemory, DeviceMemoryAllocError> {
        DeviceMemoryBuilder::new(device, memory_type, size)
            .export_info(ExternalMemoryHandleType {
                opaque_fd: true,
                ..ExternalMemoryHandleType::none()
            })
            .dedicated_info(resource)
            .build()
    }

    /// Same as `alloc_and_map`, but allows exportable file descriptor on Linux.
    #[inline]
    #[cfg(target_os = "linux")]
    pub fn alloc_and_map_with_exportable_fd(
        device: Arc<Device>,
        memory_type: MemoryType,
        size: usize,
    ) -> Result<MappedDeviceMemory, DeviceMemoryAllocError> {
        DeviceMemory::dedicated_alloc_and_map_with_exportable_fd(
            device,
            memory_type,
            size,
            DedicatedAlloc::None,
        )
    }

    /// Same as `dedicated_alloc_and_map`, but allows exportable file descriptor on Linux.
    #[inline]
    #[cfg(target_os = "linux")]
    pub fn dedicated_alloc_and_map_with_exportable_fd(
        device: Arc<Device>,
        memory_type: MemoryType,
        size: usize,
        resource: DedicatedAlloc,
    ) -> Result<MappedDeviceMemory, DeviceMemoryAllocError> {
        let vk = device.pointers();

        assert!(memory_type.is_host_visible());
        let mem = DeviceMemory::dedicated_alloc_with_exportable_fd(
            device.clone(),
            memory_type,
            size,
            resource,
        )?;

        Self::map_allocation(device.clone(), mem)
    }

    fn map_allocation(
        device: Arc<Device>,
        mem: DeviceMemory,
    ) -> Result<MappedDeviceMemory, DeviceMemoryAllocError> {
        let vk = device.pointers();
        let coherent = mem.memory_type().is_host_coherent();
        let ptr = unsafe {
            let mut output = MaybeUninit::uninit();
            check_errors(vk.MapMemory(
                device.internal_object(),
                mem.memory,
                0,
                mem.size as vk::DeviceSize,
                0, /* reserved flags */
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(MappedDeviceMemory {
            memory: mem,
            pointer: ptr,
            coherent,
        })
    }

    /// Returns the memory type this chunk was allocated on.
    #[inline]
    pub fn memory_type(&self) -> MemoryType {
        self.device
            .physical_device()
            .memory_type_by_id(self.memory_type_index)
            .unwrap()
    }

    /// Returns the size in bytes of that memory chunk.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Exports the device memory into a Unix file descriptor.  The caller retains ownership of the
    /// file, as per the Vulkan spec.
    ///
    /// # Panic
    ///
    /// - Panics if the user requests an invalid handle type for this device memory object.
    #[inline]
    #[cfg(target_os = "linux")]
    pub fn export_fd(
        &self,
        handle_type: ExternalMemoryHandleType,
    ) -> Result<File, DeviceMemoryAllocError> {
        let vk = self.device.pointers();

        let bits = handle_type.to_bits();
        assert!(
            bits == vk::EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                || bits == vk::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
        );
        assert!(handle_type.to_bits() & self.handle_types.to_bits() != 0);

        let fd = unsafe {
            let info = vk::MemoryGetFdInfoKHR {
                sType: vk::STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                pNext: ptr::null(),
                memory: self.memory,
                handleType: handle_type.to_bits(),
            };

            let mut output = MaybeUninit::uninit();
            check_errors(vk.GetMemoryFdKHR(
                self.device.internal_object(),
                &info,
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        let file = unsafe { File::from_raw_fd(fd) };
        Ok(file)
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

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_DEVICE_MEMORY;

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
            let mut allocation_count = self
                .device
                .allocation_count()
                .lock()
                .expect("Poisoned mutex");
            *allocation_count -= 1;
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
/// // Allocates 1KB of memory.
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
    where
        T: Content,
    {
        let vk = self.memory.device().pointers();
        let pointer = T::ref_from_ptr(
            (self.pointer as usize + range.start) as *mut _,
            range.end - range.start,
        )
        .unwrap(); // TODO: error

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
    where
        F: FnOnce(*mut T) -> *mut U,
    {
        CpuAccess {
            pointer: f(self.pointer),
            mem: self.mem,
            coherent: self.coherent,
            range: self.range.clone(), // TODO: ?
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
                vk.FlushMappedMemoryRanges(self.mem.as_ref().device().internal_object(), 1, &range);
            }
        }
    }
}

/// Error type returned by functions related to `DeviceMemory`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeviceMemoryAllocError {
    /// Not enough memory available.
    OomError(OomError),
    /// The maximum number of allocations has been exceeded.
    TooManyObjects,
    /// Memory map failed.
    MemoryMapFailed,
}

impl error::Error for DeviceMemoryAllocError {
    #[inline]
    fn cause(&self) -> Option<&dyn error::Error> {
        match *self {
            DeviceMemoryAllocError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for DeviceMemoryAllocError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                DeviceMemoryAllocError::OomError(_) => "not enough memory available",
                DeviceMemoryAllocError::TooManyObjects => {
                    "the maximum number of allocations has been exceeded"
                }
                DeviceMemoryAllocError::MemoryMapFailed => "memory map failed",
            }
        )
    }
}

impl From<Error> for DeviceMemoryAllocError {
    #[inline]
    fn from(err: Error) -> DeviceMemoryAllocError {
        match err {
            e @ Error::OutOfHostMemory | e @ Error::OutOfDeviceMemory => {
                DeviceMemoryAllocError::OomError(e.into())
            }
            Error::TooManyObjects => DeviceMemoryAllocError::TooManyObjects,
            Error::MemoryMapFailed => DeviceMemoryAllocError::MemoryMapFailed,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for DeviceMemoryAllocError {
    #[inline]
    fn from(err: OomError) -> DeviceMemoryAllocError {
        DeviceMemoryAllocError::OomError(err)
    }
}

#[cfg(test)]
mod tests {
    use memory::DeviceMemory;
    use memory::DeviceMemoryAllocError;
    use OomError;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device.physical_device().memory_types().next().unwrap();
        let _ = DeviceMemory::alloc(device.clone(), mem_ty, 256).unwrap();
    }

    #[test]
    fn zero_size() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device.physical_device().memory_types().next().unwrap();
        assert_should_panic!({
            let _ = DeviceMemory::alloc(device.clone(), mem_ty, 0);
        });
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn oom_single() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device
            .physical_device()
            .memory_types()
            .filter(|m| !m.is_lazily_allocated())
            .next()
            .unwrap();

        match DeviceMemory::alloc(device.clone(), mem_ty, 0xffffffffffffffff) {
            Err(DeviceMemoryAllocError::OomError(OomError::OutOfDeviceMemory)) => (),
            _ => panic!(),
        }
    }

    #[test]
    #[ignore] // TODO: test fails for now on Mesa+Intel
    fn oom_multi() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device
            .physical_device()
            .memory_types()
            .filter(|m| !m.is_lazily_allocated())
            .next()
            .unwrap();
        let heap_size = mem_ty.heap().size();

        let mut allocs = Vec::new();

        for _ in 0..4 {
            match DeviceMemory::alloc(device.clone(), mem_ty, heap_size / 3) {
                Err(DeviceMemoryAllocError::OomError(OomError::OutOfDeviceMemory)) => return, // test succeeded
                Ok(a) => allocs.push(a),
                _ => (),
            }
        }

        panic!()
    }

    #[test]
    fn allocation_count() {
        let (device, _) = gfx_dev_and_queue!();
        let mem_ty = device.physical_device().memory_types().next().unwrap();
        assert_eq!(*device.allocation_count().lock().unwrap(), 0);
        let mem1 = DeviceMemory::alloc(device.clone(), mem_ty, 256).unwrap();
        assert_eq!(*device.allocation_count().lock().unwrap(), 1);
        {
            let mem2 = DeviceMemory::alloc(device.clone(), mem_ty, 256).unwrap();
            assert_eq!(*device.allocation_count().lock().unwrap(), 2);
        }
        assert_eq!(*device.allocation_count().lock().unwrap(), 1);
    }
}
