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

use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use smallvec::SmallVec;

use buffer::sys::BufferCreationError;
use buffer::sys::SparseLevel;
use buffer::sys::UnsafeBuffer;
use buffer::sys::Usage;
use buffer::traits::Buffer;
use buffer::traits::BufferInner;
use buffer::traits::IntoBuffer;
use buffer::traits::TypedBuffer;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use instance::QueueFamily;
use memory::pool::AllocLayout;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdMemoryPool;
use sync::Sharing;

use OomError;
use SafeDeref;

/// Buffer whose content is accessible by the CPU.
#[derive(Debug)]
pub struct DeviceLocalBuffer<T: ?Sized, A = Arc<StdMemoryPool>> where A: MemoryPool {
    // Inner content.
    inner: UnsafeBuffer,

    // The memory held by the buffer.
    memory: A::Alloc,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    // Number of times this buffer is locked on the GPU side.
    gpu_lock: AtomicUsize,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

impl<T> DeviceLocalBuffer<T> {
    /// Builds a new buffer. Only allowed for sized data.
    #[inline]
    pub fn new<'a, I>(device: &Arc<Device>, usage: &Usage, queue_families: I)
                      -> Result<Arc<DeviceLocalBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            DeviceLocalBuffer::raw(device, mem::size_of::<T>(), usage, queue_families)
        }
    }
}

impl<T> DeviceLocalBuffer<[T]> {
    /// Builds a new buffer. Can be used for arrays.
    #[inline]
    pub fn array<'a, I>(device: &Arc<Device>, len: usize, usage: &Usage, queue_families: I)
                      -> Result<Arc<DeviceLocalBuffer<[T]>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            DeviceLocalBuffer::raw(device, len * mem::size_of::<T>(), usage, queue_families)
        }
    }
}

impl<T: ?Sized> DeviceLocalBuffer<T> {
    /// Builds a new buffer without checking the size.
    ///
    /// # Safety
    ///
    /// You must ensure that the size that you pass is correct for `T`.
    ///
    pub unsafe fn raw<'a, I>(device: &Arc<Device>, size: usize, usage: &Usage, queue_families: I)
                             -> Result<Arc<DeviceLocalBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        let queue_families = queue_families.into_iter().map(|f| f.id())
                                           .collect::<SmallVec<[u32; 4]>>();

        let (buffer, mem_reqs) = {
            let sharing = if queue_families.len() >= 2 {
                Sharing::Concurrent(queue_families.iter().cloned())
            } else {
                Sharing::Exclusive
            };

            match UnsafeBuffer::new(device, size, &usage, sharing, SparseLevel::none()) {
                Ok(b) => b,
                Err(BufferCreationError::OomError(err)) => return Err(err),
                Err(_) => unreachable!()        // We don't use sparse binding, therefore the other
                                                // errors can't happen
            }
        };

        let mem_ty = {
            let device_local = device.physical_device().memory_types()
                                     .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                                     .filter(|t| t.is_device_local());
            let any = device.physical_device().memory_types()
                            .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0);
            device_local.chain(any).next().unwrap()
        };

        let mem = try!(MemoryPool::alloc(&Device::standard_pool(device), mem_ty,
                                         mem_reqs.size, mem_reqs.alignment, AllocLayout::Linear));
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        try!(buffer.bind_memory(mem.memory(), mem.offset()));

        Ok(Arc::new(DeviceLocalBuffer {
            inner: buffer,
            memory: mem,
            queue_families: queue_families,
            gpu_lock: AtomicUsize::new(0),
            marker: PhantomData,
        }))
    }
}

impl<T: ?Sized, A> DeviceLocalBuffer<T, A> where A: MemoryPool {
    /// Returns the device used to create this buffer.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }

    /// Returns the queue families this buffer can be used on.
    // TODO: use a custom iterator
    #[inline]
    pub fn queue_families(&self) -> Vec<QueueFamily> {
        self.queue_families.iter().map(|&num| {
            self.device().physical_device().queue_family_by_id(num).unwrap()
        }).collect()
    }
}

/// Access to a device local buffer.
#[derive(Debug, Copy, Clone)]
pub struct DeviceLocalBufferAccess<P>(P);

unsafe impl<T: ?Sized, A> IntoBuffer for Arc<DeviceLocalBuffer<T, A>>
    where T: 'static + Send + Sync,
          A: MemoryPool
{
    type Target = DeviceLocalBufferAccess<Arc<DeviceLocalBuffer<T, A>>>;

    #[inline]
    fn into_buffer(self) -> Self::Target {
        DeviceLocalBufferAccess(self)
    }
}

unsafe impl<P, T: ?Sized, A> Buffer for DeviceLocalBufferAccess<P>
    where P: SafeDeref<Target = DeviceLocalBuffer<T, A>>,
          T: 'static + Send + Sync,
          A: MemoryPool
{
    #[inline]
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.0.inner,
            offset: 0,
        }
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, _: &Queue) -> bool {
        let val = self.0.gpu_lock.fetch_add(1, Ordering::SeqCst);
        if val == 1 {
            true
        } else {
            self.0.gpu_lock.fetch_sub(1, Ordering::SeqCst);
            false
        }
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        let val = self.0.gpu_lock.fetch_add(1, Ordering::SeqCst);
        debug_assert!(val >= 1);
    }
}

unsafe impl<P, T: ?Sized, A> TypedBuffer for DeviceLocalBufferAccess<P>
    where P: SafeDeref<Target = DeviceLocalBuffer<T, A>>,
          T: 'static + Send + Sync,
          A: MemoryPool
{
    type Content = T;
}

unsafe impl<P, T: ?Sized, A> DeviceOwned for DeviceLocalBufferAccess<P>
    where P: SafeDeref<Target = DeviceLocalBuffer<T, A>>,
          T: 'static + Send + Sync,
          A: MemoryPool
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.0.inner.device()
    }
}
