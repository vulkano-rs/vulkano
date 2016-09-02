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
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;
use smallvec::SmallVec;

use buffer::sys::BufferCreationError;
use buffer::sys::SparseLevel;
use buffer::sys::UnsafeBuffer;
use buffer::sys::Usage;
use buffer::traits::AccessRange;
use buffer::traits::Buffer;
use buffer::traits::GpuAccessResult;
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

/// Buffer whose content is accessible by the CPU.
#[derive(Debug)]
pub struct DeviceLocalBuffer<T: ?Sized, A = Arc<StdMemoryPool>> where A: MemoryPool {
    // Inner content.
    inner: UnsafeBuffer,

    // The memory held by the buffer.
    memory: A::Alloc,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    // Latest submission that uses this buffer.
    // Also used to block any attempt to submit this buffer while it is accessed by the CPU.
    latest_submission: Mutex<LatestSubmission>,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

#[derive(Debug)]
struct LatestSubmission {
    read_submissions: SmallVec<[Weak<Submission>; 4]>,
    write_submission: Option<Weak<Submission>>,         // TODO: can use `Weak::new()` once it's stabilized
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
            latest_submission: Mutex::new(LatestSubmission {
                read_submissions: SmallVec::new(),
                write_submission: None,
            }),
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

unsafe impl<T: ?Sized, A> Buffer for DeviceLocalBuffer<T, A>
    where T: 'static + Send + Sync, A: MemoryPool
{
    #[inline]
    fn inner(&self) -> &UnsafeBuffer {
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
        Some(false)
    }

    #[inline]
    fn host_accesses(&self, _: usize) -> bool {
        false
    }

    unsafe fn gpu_access(&self, ranges: &mut Iterator<Item = AccessRange>,
                         submission: &Arc<Submission>) -> GpuAccessResult
    {
        let queue_id = submission.queue().family().id();
        if self.queue_families.iter().find(|&&id| id == queue_id).is_none() {
            panic!("Trying to submit to family {} a buffer suitable for families {:?}",
                   queue_id, self.queue_families);
        }

        let is_written = {
            let mut written = false;
            while let Some(r) = ranges.next() { if r.write { written = true; break; } }
            written
        };

        let mut submissions = self.latest_submission.lock().unwrap();

        let dependencies = if is_written {
            let write_dep = mem::replace(&mut submissions.write_submission,
                                         Some(Arc::downgrade(submission)));

            let read_submissions = mem::replace(&mut submissions.read_submissions,
                                                SmallVec::new());

            // We use a temporary variable to bypass a lifetime error in rustc.
            let list = read_submissions.into_iter()
                                       .chain(write_dep.into_iter())
                                       .filter_map(|s| s.upgrade())
                                       .collect::<Vec<_>>();
            list

        } else {
            submissions.read_submissions.push(Arc::downgrade(submission));
            submissions.write_submission.clone().and_then(|s| s.upgrade()).into_iter().collect()
        };

        GpuAccessResult {
            dependencies: dependencies,
            additional_wait_semaphore: None,
            additional_signal_semaphore: None,
        }
    }
}

unsafe impl<T: ?Sized, A> TypedBuffer for DeviceLocalBuffer<T, A>
    where T: 'static + Send + Sync, A: MemoryPool
{
    type Content = T;
}
