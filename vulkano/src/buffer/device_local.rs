// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Buffer whose content is read-written by the GPU only.
//!
//! Each access from the CPU or from the GPU locks the whole buffer for either reading or writing.
//! You can read the buffer multiple times simultaneously from multiple queues. Trying to read and
//! write simultaneously, or write and write simultaneously will block with a semaphore.

use smallvec::SmallVec;
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;
use std::sync::Mutex;

use buffer::BufferUsage;
use buffer::sys::BufferCreationError;
use buffer::sys::SparseLevel;
use buffer::sys::UnsafeBuffer;
use buffer::traits::BufferAccess;
use buffer::traits::BufferInner;
use buffer::traits::TypedBufferAccess;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use instance::QueueFamily;
use memory::DedicatedAlloc;
use memory::DeviceMemoryAllocError;
use memory::pool::AllocFromRequirementsFilter;
use memory::pool::AllocLayout;
use memory::pool::MappingRequirement;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::PotentialDedicatedAllocation;
use memory::pool::StdMemoryPoolAlloc;
use sync::AccessError;
use sync::Sharing;

/// Buffer whose content is in device-local memory.
///
/// This buffer type is useful in order to store intermediary data. For example you execute a
/// compute shader that writes to this buffer, then read the content of the buffer in a following
/// compute or graphics pipeline.
///
/// The `DeviceLocalBuffer` will be in device-local memory, unless the device doesn't provide any
/// device-local memory.
#[derive(Debug)]
pub struct DeviceLocalBuffer<T: ?Sized, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>> {
    // Inner content.
    inner: UnsafeBuffer,

    // The memory held by the buffer.
    memory: A,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    // Number of times this buffer is locked on the GPU side.
    gpu_lock: Mutex<GpuAccess>,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

#[derive(Debug, Copy, Clone)]
enum GpuAccess {
    None,
    NonExclusive { num: u32 },
    Exclusive { num: u32 },
}

impl<T> DeviceLocalBuffer<T> {
    /// Builds a new buffer. Only allowed for sized data.
    // TODO: unsafe because uninitialized data
    #[inline]
    pub fn new<'a, I>(device: Arc<Device>, usage: BufferUsage, queue_families: I)
                      -> Result<Arc<DeviceLocalBuffer<T>>, DeviceMemoryAllocError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe { DeviceLocalBuffer::raw(device, mem::size_of::<T>(), usage, queue_families) }
    }
}

impl<T> DeviceLocalBuffer<[T]> {
    /// Builds a new buffer. Can be used for arrays.
    // TODO: unsafe because uninitialized data
    #[inline]
    pub fn array<'a, I>(device: Arc<Device>, len: usize, usage: BufferUsage, queue_families: I)
                        -> Result<Arc<DeviceLocalBuffer<[T]>>, DeviceMemoryAllocError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe { DeviceLocalBuffer::raw(device, len * mem::size_of::<T>(), usage, queue_families) }
    }
}

impl<T: ?Sized> DeviceLocalBuffer<T> {
    /// Builds a new buffer without checking the size.
    ///
    /// # Safety
    ///
    /// You must ensure that the size that you pass is correct for `T`.
    ///
    pub unsafe fn raw<'a, I>(device: Arc<Device>, size: usize, usage: BufferUsage,
                             queue_families: I)
                             -> Result<Arc<DeviceLocalBuffer<T>>, DeviceMemoryAllocError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        let queue_families = queue_families
            .into_iter()
            .map(|f| f.id())
            .collect::<SmallVec<[u32; 4]>>();

        let (buffer, mem_reqs) = {
            let sharing = if queue_families.len() >= 2 {
                Sharing::Concurrent(queue_families.iter().cloned())
            } else {
                Sharing::Exclusive
            };

            match UnsafeBuffer::new(device.clone(), size, usage, sharing, SparseLevel::none()) {
                Ok(b) => b,
                Err(BufferCreationError::AllocError(err)) => return Err(err),
                Err(_) => unreachable!(),        // We don't use sparse binding, therefore the other
                // errors can't happen
            }
        };

        let mem = MemoryPool::alloc_from_requirements(&Device::standard_pool(&device),
                                    &mem_reqs,
                                    AllocLayout::Linear,
                                    MappingRequirement::DoNotMap,
                                    DedicatedAlloc::Buffer(&buffer),
                                    |t| if t.is_device_local() {
                                        AllocFromRequirementsFilter::Preferred
                                    } else {
                                        AllocFromRequirementsFilter::Allowed
                                    })?;
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        buffer.bind_memory(mem.memory(), mem.offset())?;

        Ok(Arc::new(DeviceLocalBuffer {
                        inner: buffer,
                        memory: mem,
                        queue_families: queue_families,
                        gpu_lock: Mutex::new(GpuAccess::None),
                        marker: PhantomData,
                    }))
    }
}

impl<T: ?Sized, A> DeviceLocalBuffer<T, A> {
    /// Returns the queue families this buffer can be used on.
    // TODO: use a custom iterator
    #[inline]
    pub fn queue_families(&self) -> Vec<QueueFamily> {
        self.queue_families
            .iter()
            .map(|&num| {
                     self.device()
                         .physical_device()
                         .queue_family_by_id(num)
                         .unwrap()
                 })
            .collect()
    }
}

unsafe impl<T: ?Sized, A> DeviceOwned for DeviceLocalBuffer<T, A> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<T: ?Sized, A> BufferAccess for DeviceLocalBuffer<T, A>
    where T: 'static + Send + Sync
{
    #[inline]
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.inner,
            offset: 0,
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }

    #[inline]
    fn conflicts_buffer(&self, other: &BufferAccess) -> bool {
        self.conflict_key() == other.conflict_key() // TODO:
    }

    #[inline]
    fn conflicts_image(&self, other: &ImageAccess) -> bool {
        false
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.inner.key()
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive: bool, _: &Queue) -> Result<(), AccessError> {
        let mut lock = self.gpu_lock.lock().unwrap();
        match &mut *lock {
            a @ &mut GpuAccess::None => {
                if exclusive {
                    *a = GpuAccess::Exclusive { num: 1 };
                } else {
                    *a = GpuAccess::NonExclusive { num: 1 };
                }

                Ok(())
            },
            &mut GpuAccess::NonExclusive { ref mut num } => {
                if exclusive {
                    Err(AccessError::AlreadyInUse)
                } else {
                    *num += 1;
                    Ok(())
                }
            },
            &mut GpuAccess::Exclusive { .. } => {
                Err(AccessError::AlreadyInUse)
            },
        }
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        let mut lock = self.gpu_lock.lock().unwrap();
        match *lock {
            GpuAccess::None => panic!(),
            GpuAccess::NonExclusive { ref mut num } => {
                debug_assert!(*num >= 1);
                *num += 1;
            },
            GpuAccess::Exclusive { ref mut num } => {
                debug_assert!(*num >= 1);
                *num += 1;
            },
        }
    }

    #[inline]
    unsafe fn unlock(&self) {
        let mut lock = self.gpu_lock.lock().unwrap();

        match *lock {
            GpuAccess::None => panic!("Tried to unlock a buffer that isn't locked"),
            GpuAccess::NonExclusive { ref mut num } => {
                assert!(*num >= 1);
                *num -= 1;
                if *num >= 1 {
                    return;
                }
            },
            GpuAccess::Exclusive { ref mut num } => {
                assert!(*num >= 1);
                *num -= 1;
                if *num >= 1 {
                    return;
                }
            },
        };

        *lock = GpuAccess::None;
    }
}

unsafe impl<T: ?Sized, A> TypedBufferAccess for DeviceLocalBuffer<T, A>
    where T: 'static + Send + Sync
{
    type Content = T;
}
