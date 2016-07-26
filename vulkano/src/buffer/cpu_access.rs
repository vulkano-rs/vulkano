// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Buffer whose content is accessible to the CPU.
//! 
//! The `CpuAccessibleBuffer` is a basic general-purpose buffer. It can be used in any situation
//! but may not perform as well as other buffer types.
//! 
//! Each access from the CPU or from the GPU locks the whole buffer for either reading or writing.
//! You can read the buffer multiple times simultaneously. Trying to read and write simultaneously,
//! or write and write simultaneously will block.

use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;
use std::sync::Weak;
use std::time::Duration;
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
use memory::Content;
use memory::CpuAccess as MemCpuAccess;
use memory::pool::AllocLayout;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdMemoryPool;
use sync::FenceWaitError;
use sync::Sharing;

use OomError;

/// Buffer whose content is accessible by the CPU.
#[derive(Debug)]
pub struct CpuAccessibleBuffer<T: ?Sized, A = Arc<StdMemoryPool>> where A: MemoryPool {
    // Inner content.
    inner: UnsafeBuffer,

    // The memory held by the buffer.
    memory: A::Alloc,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    // Latest submission that uses this buffer.
    // Also used to block any attempt to submit this buffer while it is accessed by the CPU.
    latest_submission: RwLock<LatestSubmission>,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

#[derive(Debug)]
struct LatestSubmission {
    read_submissions: Mutex<Vec<Weak<Submission>>>,
    write_submission: Option<Weak<Submission>>,         // TODO: can use `Weak::new()` once it's stabilized
}

impl<T> CpuAccessibleBuffer<T> {
    /// Deprecated. Use `from_data` instead.
    #[deprecated]
    #[inline]
    pub fn new<'a, I>(device: &Arc<Device>, usage: &Usage, queue_families: I)
                      -> Result<Arc<CpuAccessibleBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            CpuAccessibleBuffer::raw(device, mem::size_of::<T>(), usage, queue_families)
        }
    }

    /// Builds a new buffer with some data in it. Only allowed for sized data.
    pub fn from_data<'a, I>(device: &Arc<Device>, usage: &Usage, queue_families: I, data: T)
                            -> Result<Arc<CpuAccessibleBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>,
              T: Content + 'static,
    {
        unsafe {
            let uninitialized = try!(
                CpuAccessibleBuffer::raw(device, mem::size_of::<T>(), usage, queue_families)
            );

            // Note that we are in panic-unsafety land here. However a panic should never ever
            // happen here, so in theory we are safe.
            // TODO: check whether that's true ^

            {
                let mut mapping = uninitialized.write(Duration::new(0, 0)).unwrap();
                ptr::write::<T>(&mut *mapping, data)
            }

            Ok(uninitialized)
        }
    }

    /// Builds a new uninitialized buffer. Only allowed for sized data.
    #[inline]
    pub unsafe fn uninitialized<'a, I>(device: &Arc<Device>, usage: &Usage, queue_families: I)
                                       -> Result<Arc<CpuAccessibleBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            CpuAccessibleBuffer::raw(device, mem::size_of::<T>(), usage, queue_families)
        }
    }
}

impl<T> CpuAccessibleBuffer<[T]> {
    /// Builds a new buffer that contains an array `T`. The initial data comes from an iterator
    /// that produces that list of Ts.
    pub fn from_iter<'a, I, Q>(device: &Arc<Device>, usage: &Usage, queue_families: Q, data: I)
                               -> Result<Arc<CpuAccessibleBuffer<[T]>>, OomError>
        where I: ExactSizeIterator<Item = T>,
              T: Content + 'static,
              Q: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            let uninitialized = try!(
                CpuAccessibleBuffer::uninitialized_array(device, data.len(), usage, queue_families)
            );

            // Note that we are in panic-unsafety land here. However a panic should never ever
            // happen here, so in theory we are safe.
            // TODO: check whether that's true ^

            {
                let mut mapping = uninitialized.write(Duration::new(0, 0)).unwrap();

                for (i, o) in data.zip(mapping.iter_mut()) {
                    ptr::write(o, i);
                }
            }

            Ok(uninitialized)
        }
    }

    /// Deprecated. Use `uninitialized_array` or `from_iter` instead.
    // TODO: remove
    #[inline]
    #[deprecated]
    pub fn array<'a, I>(device: &Arc<Device>, len: usize, usage: &Usage, queue_families: I)
                      -> Result<Arc<CpuAccessibleBuffer<[T]>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            CpuAccessibleBuffer::uninitialized_array(device, len, usage, queue_families)
        }
    }

    /// Builds a new buffer. Can be used for arrays.
    #[inline]
    pub unsafe fn uninitialized_array<'a, I>(device: &Arc<Device>, len: usize, usage: &Usage,
                                             queue_families: I)
                                             -> Result<Arc<CpuAccessibleBuffer<[T]>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            CpuAccessibleBuffer::raw(device, len * mem::size_of::<T>(), usage, queue_families)
        }
    }
}

impl<T: ?Sized> CpuAccessibleBuffer<T> {
    /// Builds a new buffer without checking the size.
    ///
    /// # Safety
    ///
    /// You must ensure that the size that you pass is correct for `T`.
    ///
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

            match UnsafeBuffer::new(device, size, &usage, sharing, SparseLevel::none()) {
                Ok(b) => b,
                Err(BufferCreationError::OomError(err)) => return Err(err),
                Err(_) => unreachable!()        // We don't use sparse binding, therefore the other
                                                // errors can't happen
            }
        };

        let mem_ty = device.physical_device().memory_types()
                           .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                           .filter(|t| t.is_host_visible())
                           .next().unwrap();    // Vk specs guarantee that this can't fail

        let mem = try!(MemoryPool::alloc(&Device::standard_pool(device), mem_ty,
                                         mem_reqs.size, mem_reqs.alignment, AllocLayout::Linear));
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        debug_assert!(mem.mapped_memory().is_some());
        try!(buffer.bind_memory(mem.memory(), mem.offset()));

        Ok(Arc::new(CpuAccessibleBuffer {
            inner: buffer,
            memory: mem,
            queue_families: queue_families,
            latest_submission: RwLock::new(LatestSubmission {
                read_submissions: Mutex::new(vec![]),
                write_submission: None,
            }),
            marker: PhantomData,
        }))
    }
}

impl<T: ?Sized, A> CpuAccessibleBuffer<T, A> where A: MemoryPool {
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

impl<T: ?Sized, A> CpuAccessibleBuffer<T, A> where T: Content + 'static, A: MemoryPool {
    /// Locks the buffer in order to write its content.
    ///
    /// If the buffer is currently in use by the GPU, this function will block until either the
    /// buffer is available or the timeout is reached. A value of `0` for the timeout is valid and
    /// means that the function should never block.
    ///
    /// After this function successfully locks the buffer, any attempt to submit a command buffer
    /// that uses it will block until you unlock it.
    // TODO: remove timeout parameter since CPU-side locking can't use it
    #[inline]
    pub fn read(&self, timeout: Duration) -> Result<ReadLock<T>, FenceWaitError> {
        let submission = self.latest_submission.read().unwrap();

        // TODO: should that set the write_submission to None?
        if let Some(submission) = submission.write_submission.clone().and_then(|s| s.upgrade()) {
            try!(submission.wait(timeout));
        }

        let offset = self.memory.offset();
        let range = offset .. offset + self.inner.size();

        Ok(ReadLock {
            inner: unsafe { self.memory.mapped_memory().unwrap().read_write(range) },
            lock: submission,
        })
    }

    /// Locks the buffer in order to write its content.
    ///
    /// If the buffer is currently in use by the GPU, this function will block until either the
    /// buffer is available or the timeout is reached. A value of `0` for the timeout is valid and
    /// means that the function should never block.
    ///
    /// After this function successfully locks the buffer, any attempt to submit a command buffer
    /// that uses it will block until you unlock it.
    // TODO: remove timeout parameter since CPU-side locking can't use it
    #[inline]
    pub fn write(&self, timeout: Duration) -> Result<WriteLock<T>, FenceWaitError> {
        let mut submission = self.latest_submission.write().unwrap();

        {
            let mut read_submissions = submission.read_submissions.get_mut().unwrap();
            for submission in read_submissions.drain(..) {
                if let Some(submission) = submission.upgrade() {
                    try!(submission.wait(timeout));
                }
            }
        }

        if let Some(submission) = submission.write_submission.take().and_then(|s| s.upgrade()) {
            try!(submission.wait(timeout));
        }

        let offset = self.memory.offset();
        let range = offset .. offset + self.inner.size();

        Ok(WriteLock {
            inner: unsafe { self.memory.mapped_memory().unwrap().read_write(range) },
            lock: submission,
        })
    }
}

unsafe impl<T: ?Sized, A> Buffer for CpuAccessibleBuffer<T, A>
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
        Some(true)
    }

    #[inline]
    fn host_accesses(&self, _: usize) -> bool {
        true
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

        let dependencies = if is_written {
            let mut submissions = self.latest_submission.write().unwrap();

            let write_dep = mem::replace(&mut submissions.write_submission,
                                         Some(Arc::downgrade(submission)));

            let mut read_submissions = submissions.read_submissions.get_mut().unwrap();
            let read_submissions = mem::replace(&mut *read_submissions, Vec::new());
            read_submissions.into_iter()
                            .chain(write_dep.into_iter())
                            .filter_map(|s| s.upgrade())
                            .collect::<Vec<_>>()

        } else {
            let submissions = self.latest_submission.read().unwrap();

            let mut read_submissions = submissions.read_submissions.lock().unwrap();
            read_submissions.push(Arc::downgrade(submission));

            submissions.write_submission.clone().and_then(|s| s.upgrade()).into_iter().collect()
        };

        GpuAccessResult {
            dependencies: dependencies,
            additional_wait_semaphore: None,
            additional_signal_semaphore: None,
        }
    }
}

unsafe impl<T: ?Sized, A> TypedBuffer for CpuAccessibleBuffer<T, A>
    where T: 'static + Send + Sync, A: MemoryPool
{
    type Content = T;
}

/// Object that can be used to read or write the content of a `CpuAccessBuffer`.
///
/// Note that this object holds a rwlock read guard on the chunk. If another thread tries to access
/// this buffer's content or tries to submit a GPU command that uses this buffer, it will block.
pub struct ReadLock<'a, T: ?Sized + 'a> {
    inner: MemCpuAccess<'a, T>,
    lock: RwLockReadGuard<'a, LatestSubmission>,
}

impl<'a, T: ?Sized + 'a> ReadLock<'a, T> {
    /// Makes a new `ReadLock` to access a sub-part of the current `ReadLock`.
    #[inline]
    pub fn map<U: ?Sized + 'a, F>(self, f: F) -> ReadLock<'a, U>
        where F: FnOnce(&mut T) -> &mut U
    {
        ReadLock {
            inner: self.inner.map(|ptr| unsafe { f(&mut *ptr) as *mut _ }),
            lock: self.lock,
        }
    }
}

impl<'a, T: ?Sized + 'a> Deref for ReadLock<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.inner.deref()
    }
}

/// Object that can be used to read or write the content of a `CpuAccessBuffer`.
///
/// Note that this object holds a rwlock write guard on the chunk. If another thread tries to access
/// this buffer's content or tries to submit a GPU command that uses this buffer, it will block.
pub struct WriteLock<'a, T: ?Sized + 'a> {
    inner: MemCpuAccess<'a, T>,
    lock: RwLockWriteGuard<'a, LatestSubmission>,
}

impl<'a, T: ?Sized + 'a> WriteLock<'a, T> {
    /// Makes a new `WriteLock` to access a sub-part of the current `WriteLock`.
    #[inline]
    pub fn map<U: ?Sized + 'a, F>(self, f: F) -> WriteLock<'a, U>
        where F: FnOnce(&mut T) -> &mut U
    {
        WriteLock {
            inner: self.inner.map(|ptr| unsafe { f(&mut *ptr) as *mut _ }),
            lock: self.lock,
        }
    }
}

impl<'a, T: ?Sized + 'a> Deref for WriteLock<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.inner.deref()
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for WriteLock<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.inner.deref_mut()
    }
}
