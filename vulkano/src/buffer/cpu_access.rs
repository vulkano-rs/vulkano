// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use smallvec::SmallVec;

use buffer::sys::UnsafeBuffer;
use buffer::sys::Usage;
use buffer::traits::AccessRange;
use buffer::traits::Buffer;
use buffer::traits::TypedBuffer;
use command_buffer::Submission;
use device::Device;
use instance::QueueFamily;
use memory::Content;
use memory::CpuAccess as MemCpuAccess;
use memory::DeviceMemory;
use memory::MappedDeviceMemory;
use sync::FenceWaitError;
use sync::Sharing;

use OomError;

pub struct CpuAccessibleBuffer<T: ?Sized> {
    // Inner content.
    inner: UnsafeBuffer,

    memory: MappedDeviceMemory,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    latest_submission: Mutex<Option<Arc<Submission>>>,

    marker: PhantomData<*const T>,
}

impl<T> CpuAccessibleBuffer<T> {
    #[inline]
    pub fn new<'a, I>(device: &Arc<Device>, usage: &Usage, queue_families: I)
                      -> Result<Arc<CpuAccessibleBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            CpuAccessibleBuffer::raw(device, mem::size_of::<T>(), usage, queue_families)
        }
    }
}

impl<T> CpuAccessibleBuffer<[T]> {
    #[inline]
    pub fn array<'a, I>(device: &Arc<Device>, len: usize, usage: &Usage, queue_families: I)
                      -> Result<Arc<CpuAccessibleBuffer<[T]>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            CpuAccessibleBuffer::raw(device, len * mem::size_of::<T>(), usage, queue_families)
        }
    }
}

impl<T: ?Sized> CpuAccessibleBuffer<T> {
    pub unsafe fn raw<'a, I>(device: &Arc<Device>, size: usize, usage: &Usage, queue_families: I)
                             -> Result<Arc<CpuAccessibleBuffer<T>>, OomError>
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

            try!(UnsafeBuffer::new(device, size, &usage, sharing))
        };

        let mem_ty = device.physical_device().memory_types()
                           .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                           .filter(|t| t.is_host_visible())
                           .next().unwrap();    // Vk specs guarantee that this can't fail

        // note: alignment doesn't need to be checked because allocating memory is guaranteed to
        //       fulfill any alignment requirement

        let mem = try!(DeviceMemory::alloc_and_map(device, &mem_ty, mem_reqs.size));

        try!(buffer.bind_memory(mem.memory(), 0 .. mem_reqs.size));

        Ok(Arc::new(CpuAccessibleBuffer {
            inner: buffer,
            memory: mem,
            queue_families: queue_families,
            latest_submission: Mutex::new(None),
            marker: PhantomData,
        }))
    }
}

impl<T: ?Sized> CpuAccessibleBuffer<T> where T: Content + 'static {
    #[inline]
    pub fn read(&self, timeout_ns: u64) -> Result<CpuAccess<T>, FenceWaitError> {
        self.write(timeout_ns)
    }

    #[inline]
    pub fn write(&self, timeout_ns: u64) -> Result<CpuAccess<T>, FenceWaitError> {
        let submission = self.latest_submission.lock().unwrap();

        if let Some(submission) = submission.as_ref() {
            try!(submission.wait(timeout_ns));
        }

        Ok(CpuAccess {
            inner: unsafe { self.memory.write() },
            lock: submission,
        })
    }
}

unsafe impl<T: ?Sized> Buffer for CpuAccessibleBuffer<T> {
    #[inline]
    fn inner_buffer(&self) -> &UnsafeBuffer {
        &self.inner
    }
    
    #[inline]
    fn blocks(&self, _: Range<usize>) -> Vec<usize> {
        vec![0]
    }

    #[inline]
    fn block_memory_range(&self, _: usize) -> Range<usize> {
        0 .. self.size()
    }

    fn needs_fence(&self, _: bool, _: Range<usize>) -> Option<bool> {
        Some(true)
    }

    #[inline]
    fn host_accesses(&self, _: usize) -> bool {
        true
    }

    unsafe fn gpu_access(&self, _: &mut Iterator<Item = AccessRange>, submission: &Arc<Submission>)
                         -> Vec<Arc<Submission>>
    {
        let queue_id = submission.queue().family().id();
        if self.queue_families.iter().find(|&&id| id == queue_id).is_none() {
            panic!("Trying to submit to family {} a buffer suitable for families {:?}",
                   queue_id, self.queue_families);
        }

        let dependency = {
            let mut latest_submission = self.latest_submission.lock().unwrap();
            mem::replace(&mut *latest_submission, Some(submission.clone()))
        };

        if let Some(dependency) = dependency {
            vec![dependency]
        } else {
            vec![]
        }
    }
}

unsafe impl<T: ?Sized + 'static> TypedBuffer for CpuAccessibleBuffer<T> {
    type Content = T;
}

/// Object that can be used to read or write the content of a `CpuAccessBuffer`.
///
/// Note that this object holds a mutex guard on the chunk. If another thread tries to access
/// this buffer's content or tries to submit a GPU command that uses this buffer, it will block.
pub struct CpuAccess<'a, T: ?Sized + 'a> {
    inner: MemCpuAccess<'a, T>,
    lock: MutexGuard<'a, Option<Arc<Submission>>>,
}

impl<'a, T: ?Sized + 'a> Deref for CpuAccess<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.inner.deref()
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for CpuAccess<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.inner.deref_mut()
    }
}
