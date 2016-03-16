use std::mem;
use std::ptr;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::TryLockError;

use buffer::GpuAccessSynchronization as BufferGpuAccessSynchronization;
use buffer::BufferMemorySource;
use buffer::BufferMemorySourceChunk;
use buffer::GpuAccessRange;
use image::ImageMemorySource;
use image::ImageMemorySourceChunk;
use image::GpuAccessRange as ImageAccessRange;
use image::GpuAccessSynchronization as ImageGpuAccessSynchronization;
use memory::ChunkProperties;
use memory::Content;
use memory::CpuAccessible;
use memory::CpuWriteAccessible;
use memory::MemorySource;
use memory::MemorySourceChunk;
use memory::DeviceMemory;
use memory::MappedDeviceMemory;
use memory::ChunkRange;
use sync::Fence;
use sync::Semaphore;

use device::Device;
use device::Queue;

use OomError;
use VulkanObject;
use VulkanPointers;
use vk;

/// Dummy marker whose strategy is to allocate a new chunk of memory for each allocation.
///
/// The memory will not be accessible since it is not necessarily in host-visible memory.
///
/// This is good for large buffers, but inefficient is you use a lot of small buffers.
///
/// The memory is locked globally. That means that it doesn't matter whether you access the buffer
/// for reading or writing (like a `Mutex`).
#[derive(Debug, Copy, Clone)]
pub struct DeviceLocal;

unsafe impl MemorySource for DeviceLocal {
    type Chunk = DeviceLocalChunk;

    #[inline]
    fn is_sparse(&self) -> bool {
        false
    }

    #[inline]
    fn allocate(self, device: &Arc<Device>, size: usize, alignment: usize, memory_type_bits: u32)
                -> Result<DeviceLocalChunk, OomError>
    {
        // We try to find a device-local memory type, but fall back to any memory type if we don't
        // find any.

        let device_local = device.physical_device().memory_types()
                                 .filter(|t| (memory_type_bits & (1 << t.id())) != 0)
                                 .filter(|t| t.is_device_local());

        let any = device.physical_device().memory_types()
                        .filter(|t| (memory_type_bits & (1 << t.id())) != 0);

        let mem_ty = device_local.chain(any).next().expect("could not find any memory type");

        let mem = try!(DeviceMemory::alloc(device, &mem_ty, size));

        // note: alignment doesn't need to be checked because allocating memory is guaranteed to
        //       fulfill any alignment requirement

        Ok(DeviceLocalChunk {
            mem: mem,
            semaphore: Mutex::new(None),
        })
    }
}

unsafe impl BufferMemorySource for DeviceLocal {
    type Chunk = DeviceLocalChunk;

    #[inline]
    fn allocate(self, device: &Arc<Device>, size: usize, alignment: usize, memory_type_bits: u32)
                -> Result<Self::Chunk, OomError>
    {
        MemorySource::allocate(self, device, size, alignment, memory_type_bits)
    }
}

unsafe impl ImageMemorySource for DeviceLocal {
    type Chunk = DeviceLocalChunk;

    #[inline]
    fn allocate(self, device: &Arc<Device>, size: usize, alignment: usize, memory_type_bits: u32)
                -> Result<Self::Chunk, OomError>
    {
        MemorySource::allocate(self, device, size, alignment, memory_type_bits)
    }
}

/// A chunk allocated from a `DeviceLocal`.
pub struct DeviceLocalChunk {
    mem: DeviceMemory,
    semaphore: Mutex<Option<Arc<Semaphore>>>,
}

unsafe impl MemorySourceChunk for DeviceLocalChunk {
    #[inline]
    unsafe fn gpu_access(&self, _write: bool, _range: ChunkRange, _: &Arc<Queue>,
                         _: Option<Arc<Fence>>, mut semaphore: Option<Arc<Semaphore>>)
                         -> Option<Arc<Semaphore>>
    {
        assert!(semaphore.is_some());

        let mut self_semaphore = self.semaphore.lock().unwrap();
        mem::swap(&mut *self_semaphore, &mut semaphore);

        semaphore
    }

    #[inline]
    fn requires_fence(&self) -> bool {
        false
    }

    #[inline]
    fn properties(&self) -> ChunkProperties {
        ChunkProperties::Regular {
            memory: &self.mem,
            offset: 0,
            size: self.mem.size(),
        }
    }

    #[inline]
    fn may_alias(&self) -> bool {
        false
    }
}

unsafe impl BufferMemorySourceChunk for DeviceLocalChunk {
    #[inline]
    fn properties(&self) -> ChunkProperties {
        ChunkProperties::Regular {
            memory: &self.mem,
            offset: 0,
            size: self.mem.size(),
        }
    }

    unsafe fn gpu_access(&self, queue: &Arc<Queue>, submission_id: u64, _ranges: &[GpuAccessRange],
                         fence: Option<&Arc<Fence>>) -> BufferGpuAccessSynchronization
    {
        let mut semaphore = Some(Semaphore::new(queue.device()).unwrap());        // TODO: error

        let mut self_semaphore = self.semaphore.lock().unwrap();
        mem::swap(&mut *self_semaphore, &mut semaphore);

        BufferGpuAccessSynchronization {
            pre_semaphore: semaphore,
            post_semaphore: self_semaphore.clone(),
        }
    }
}

unsafe impl ImageMemorySourceChunk for DeviceLocalChunk {
    #[inline]
    fn properties(&self) -> ChunkProperties {
        ChunkProperties::Regular {
            memory: &self.mem,
            offset: 0,
            size: self.mem.size(),
        }
    }

    unsafe fn gpu_access(&self, queue: &Arc<Queue>, submission_id: u64, ranges: &[ImageAccessRange],
                         _fence: Option<&Arc<Fence>>) -> ImageGpuAccessSynchronization
    {
        let mut semaphore = Some(Semaphore::new(queue.device()).unwrap());      // TODO: error
        let mut self_semaphore = self.semaphore.lock().unwrap();
        mem::swap(&mut *self_semaphore, &mut semaphore);

        ImageGpuAccessSynchronization {
            pre_semaphore: semaphore,
            post_semaphore: None,       // FIXME:
        }
    }
}

/// Dummy marker whose strategy is to allocate a new chunk of memory for each allocation.
///
/// Guaranteed to allocate from a host-visible memory type.
///
/// This is good for large buffers, but inefficient is you use a lot of small buffers.
///
/// The memory is locked globally. That means that it doesn't matter whether you access the buffer
/// for reading or writing (like a `Mutex`).
#[derive(Debug, Copy, Clone)]
pub struct HostVisible;

unsafe impl MemorySource for HostVisible {
    type Chunk = HostVisibleChunk;

    #[inline]
    fn is_sparse(&self) -> bool {
        false
    }

    #[inline]
    fn allocate(self, device: &Arc<Device>, size: usize, alignment: usize, memory_type_bits: u32)
                -> Result<HostVisibleChunk, OomError>
    {
        let mem_ty = device.physical_device().memory_types()
                           .filter(|t| (memory_type_bits & (1 << t.id())) != 0)
                           .filter(|t| t.is_host_visible())
                           .next().unwrap();
        let mem = try!(DeviceMemory::alloc_and_map(device, &mem_ty, size));

        // note: alignment doesn't need to be checked because allocating memory is guaranteed to
        //       fulfill any alignment requirement

        Ok(HostVisibleChunk {
            mem: mem,
            coherent: mem_ty.is_host_coherent(),
            lock: Mutex::new((None, None)),
        })
    }
}

unsafe impl BufferMemorySource for HostVisible {
    type Chunk = HostVisibleChunk;

    #[inline]
    fn allocate(self, device: &Arc<Device>, size: usize, alignment: usize, memory_type_bits: u32)
                -> Result<Self::Chunk, OomError>
    {
        MemorySource::allocate(self, device, size, alignment, memory_type_bits)
    }
}

unsafe impl ImageMemorySource for HostVisible {
    type Chunk = HostVisibleChunk;

    #[inline]
    fn allocate(self, device: &Arc<Device>, size: usize, alignment: usize, memory_type_bits: u32)
                -> Result<Self::Chunk, OomError>
    {
        MemorySource::allocate(self, device, size, alignment, memory_type_bits)
    }
}

/// A chunk allocated from a `HostVisible`.
pub struct HostVisibleChunk {
    mem: MappedDeviceMemory,
    coherent: bool,
    lock: Mutex<(Option<Arc<Semaphore>>, Option<Arc<Fence>>)>,
}

unsafe impl MemorySourceChunk for HostVisibleChunk {
    #[inline]
    unsafe fn gpu_access(&self, _write: bool, _range: ChunkRange, _: &Arc<Queue>,
                         fence: Option<Arc<Fence>>, mut semaphore: Option<Arc<Semaphore>>)
                         -> Option<Arc<Semaphore>>
    {
        assert!(fence.is_some());
        assert!(semaphore.is_some());

        let mut self_lock = self.lock.lock().unwrap();
        mem::swap(&mut self_lock.0, &mut semaphore);
        self_lock.1 = fence;

        semaphore
    }

    #[inline]
    fn properties(&self) -> ChunkProperties {
        ChunkProperties::Regular {
            memory: &self.mem.memory(),
            offset: 0,
            size: self.mem.memory().size(),
        }
    }

    #[inline]
    fn may_alias(&self) -> bool {
        false
    }
}

unsafe impl BufferMemorySourceChunk for HostVisibleChunk {
    #[inline]
    fn properties(&self) -> ChunkProperties {
        MemorySourceChunk::properties(self)
    }

    unsafe fn gpu_access(&self, queue: &Arc<Queue>, submission_id: u64, _ranges: &[GpuAccessRange],
                         fence: Option<&Arc<Fence>>) -> BufferGpuAccessSynchronization
    {
        let mut semaphore = Some(Semaphore::new(queue.device()).unwrap());        // TODO: error

        let mut self_lock = self.lock.lock().unwrap();
        mem::swap(&mut self_lock.0, &mut semaphore);
        self_lock.1 = Some(fence.unwrap().clone());

        BufferGpuAccessSynchronization {
            pre_semaphore: semaphore,
            post_semaphore: self_lock.0.clone(),
        }
    }
}

unsafe impl ImageMemorySourceChunk for HostVisibleChunk {
    #[inline]
    fn properties(&self) -> ChunkProperties {
        MemorySourceChunk::properties(self)
    }

    unsafe fn gpu_access(&self, queue: &Arc<Queue>, submission_id: u64, ranges: &[ImageAccessRange],
                         fence: Option<&Arc<Fence>>) -> ImageGpuAccessSynchronization
    {
        let mut semaphore = Some(Semaphore::new(queue.device()).unwrap());      // TODO: error

        let mut self_lock = self.lock.lock().unwrap();
        mem::swap(&mut self_lock.0, &mut semaphore);
        self_lock.1 = Some(fence.unwrap().clone());

        ImageGpuAccessSynchronization {
            pre_semaphore: semaphore,
            post_semaphore: self_lock.0.clone(),
        }
    }
}

unsafe impl<'a, T: ?Sized + 'a> CpuAccessible<'a, T> for HostVisibleChunk
    where T: Content
{
    type Read = GpuAccess<'a, T>;

    #[inline]
    fn read(&'a self, timeout_ns: u64) -> GpuAccess<'a, T> {
        self.write(timeout_ns)
    }

    #[inline]
    fn try_read(&'a self) -> Option<GpuAccess<'a, T>> {
        self.try_write()
    }
}

unsafe impl<'a, T: ?Sized + 'a> CpuWriteAccessible<'a, T> for HostVisibleChunk
    where T: Content
{
    type Write = GpuAccess<'a, T>;

    #[inline]
    fn write(&'a self, timeout_ns: u64) -> GpuAccess<'a, T> {
        let vk = self.mem.memory().device().pointers();
        let pointer = T::ref_from_ptr(self.mem.mapping_pointer(), self.mem.memory().size()).unwrap();       // TODO: error

        let mut lock = self.lock.lock().unwrap();
        if let Some(ref fence) = lock.1 {
            fence.wait(timeout_ns).unwrap();        // FIXME: error
        }
        lock.1 = None;

        if !self.coherent {
            let range = vk::MappedMemoryRange {
                sType: vk::STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                pNext: ptr::null(),
                memory: self.mem.memory().internal_object(),
                offset: 0,
                size: vk::WHOLE_SIZE,
            };

            // TODO: check result?
            unsafe {
                vk.InvalidateMappedMemoryRanges(self.mem.memory().device().internal_object(),
                                                1, &range);
            }
        }

        GpuAccess {
            mem: &self.mem,
            coherent: self.coherent,
            guard: lock,
            pointer: pointer,
        }
    }

    #[inline]
    fn try_write(&'a self) -> Option<GpuAccess<'a, T>> {
        let vk = self.mem.memory().device().pointers();
        let pointer = T::ref_from_ptr(self.mem.mapping_pointer(), self.mem.memory().size()).unwrap();       // TODO: error

        let mut lock = match self.lock.try_lock() {
            Ok(l) => l,
            Err(TryLockError::Poisoned(_)) => panic!(),
            Err(TryLockError::WouldBlock) => return None,
        };

        if let Some(ref fence) = lock.1 {
            if fence.ready() != Ok(true) {      // TODO: we ignore ready()'s error here?
                return None;
            }
        }

        lock.1 = None;

        if !self.coherent {
            let range = vk::MappedMemoryRange {
                sType: vk::STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                pNext: ptr::null(),
                memory: self.mem.memory().internal_object(),
                offset: 0,
                size: vk::WHOLE_SIZE,
            };

            // TODO: check result?
            unsafe {
                vk.InvalidateMappedMemoryRanges(self.mem.memory().device().internal_object(),
                                                1, &range);
            }
        }

        Some(GpuAccess {
            mem: &self.mem,
            coherent: self.coherent,
            guard: lock,
            pointer: pointer,
        })
    }
}

/// Object that can be used to read or write the content of a `HostVisibleChunk`.
///
/// Note that this object holds a mutex guard on the chunk. If another thread tries to access
/// this memory's content or tries to submit a GPU command that uses this memory, it will block.
pub struct GpuAccess<'a, T: ?Sized + 'a> {
    mem: &'a MappedDeviceMemory,
    pointer: *mut T,
    guard: MutexGuard<'a, (Option<Arc<Semaphore>>, Option<Arc<Fence>>)>,
    coherent: bool,
}

impl<'a, T: ?Sized + 'a> Deref for GpuAccess<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for GpuAccess<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer }
    }
}

impl<'a, T: ?Sized + 'a> Drop for GpuAccess<'a, T> {
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
