// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Buffer that is written once then read for as long as it is alive.
//! 
//! Use this buffer when you have data that you never modify.
//!
//! Only the first ever command buffer that uses this buffer can write to it (for example by
//! copying from another buffer). Any subsequent command buffer **must** only read from the buffer,
//! or a panic will happen.
//! 
//! The buffer will be stored in device-local memory if possible
//!

use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use smallvec::SmallVec;

use buffer::sys::BufferCreationError;
use buffer::sys::SparseLevel;
use buffer::sys::UnsafeBuffer;
use buffer::sys::Usage;
use buffer::traits::Buffer;
use buffer::traits::TypedBuffer;
use command_buffer::Submission;
use device::Device;
use instance::QueueFamily;
use memory::pool::AllocLayout;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdMemoryPool;
use sync::Sharing;

use OomError;

/// Buffer that is written once then read for as long as it is alive.
pub struct ImmutableBuffer<T: ?Sized, A = Arc<StdMemoryPool>> where A: MemoryPool {
    // Inner content.
    inner: UnsafeBuffer,

    memory: A::Alloc,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    latest_write_submission: Mutex<Option<Weak<Submission>>>,        // TODO: can use `Weak::new()` once it's stabilized

    started_reading: AtomicBool,

    marker: PhantomData<Box<T>>,
}

impl<T> ImmutableBuffer<T> {
    /// Builds a new buffer. Only allowed for sized data.
    #[inline]
    pub fn new<'a, I>(device: &Arc<Device>, usage: &Usage, queue_families: I)
                      -> Result<Arc<ImmutableBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            ImmutableBuffer::raw(device, mem::size_of::<T>(), usage, queue_families)
        }
    }
}

impl<T> ImmutableBuffer<[T]> {
    /// Builds a new buffer. Can be used for arrays.
    #[inline]
    pub fn array<'a, I>(device: &Arc<Device>, len: usize, usage: &Usage, queue_families: I)
                      -> Result<Arc<ImmutableBuffer<[T]>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            ImmutableBuffer::raw(device, len * mem::size_of::<T>(), usage, queue_families)
        }
    }
}

impl<T: ?Sized> ImmutableBuffer<T> {
    /// Builds a new buffer without checking the size.
    ///
    /// # Safety
    ///
    /// You must ensure that the size that you pass is correct for `T`.
    ///
    pub unsafe fn raw<'a, I>(device: &Arc<Device>, size: usize, usage: &Usage, queue_families: I)
                             -> Result<Arc<ImmutableBuffer<T>>, OomError>
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

        Ok(Arc::new(ImmutableBuffer {
            inner: buffer,
            memory: mem,
            queue_families: queue_families,
            latest_write_submission: Mutex::new(None),
            started_reading: AtomicBool::new(false),
            marker: PhantomData,
        }))
    }
}

impl<T: ?Sized, A> ImmutableBuffer<T, A> where A: MemoryPool {
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

unsafe impl<T: ?Sized, A> Buffer for ImmutableBuffer<T, A>
    where T: 'static + Send + Sync, A: MemoryPool
{
    #[inline]
    fn inner(&self) -> &UnsafeBuffer {
        &self.inner
    }
}

unsafe impl<T: ?Sized, A> TypedBuffer for ImmutableBuffer<T, A>
    where T: 'static + Send + Sync, A: MemoryPool
{
    type Content = T;
}
