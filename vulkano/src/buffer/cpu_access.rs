// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
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

use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use buffer::sys::BufferCreationError;
use buffer::sys::SparseLevel;
use buffer::sys::UnsafeBuffer;
use buffer::traits::BufferAccess;
use buffer::traits::BufferInner;
use buffer::traits::TypedBufferAccess;
use buffer::BufferUsage;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use instance::QueueFamily;
use memory::pool::AllocFromRequirementsFilter;
use memory::pool::AllocLayout;
use memory::pool::MappingRequirement;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::PotentialDedicatedAllocation;
use memory::pool::StdMemoryPoolAlloc;
use memory::Content;
use memory::CpuAccess as MemCpuAccess;
use memory::DedicatedAlloc;
use memory::DeviceMemoryAllocError;
use parking_lot::RwLock;
use parking_lot::RwLockReadGuard;
use parking_lot::RwLockWriteGuard;
use sync::AccessError;
use sync::Sharing;

/// Buffer whose content is accessible by the CPU.
///
/// Setting the `host_cached` field on the various initializers to `true` will make it so
/// the `CpuAccessibleBuffer` prefers to allocate from host_cached memory. Host cached
/// memory caches GPU data on the CPU side. This can be more performant in cases where
/// the cpu needs to read data coming off the GPU.
#[derive(Debug)]
pub struct CpuAccessibleBuffer<T: ?Sized, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>> {
    // Inner content.
    inner: UnsafeBuffer,

    // The memory held by the buffer.
    memory: A,

    // Access pattern of the buffer.
    // Every time the user tries to read or write the buffer from the CPU, this `RwLock` is kept
    // locked and its content is checked to verify that we are allowed access. Every time the user
    // tries to submit this buffer for the GPU, this `RwLock` is briefly locked and modified.
    access: RwLock<CurrentGpuAccess>,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

#[derive(Debug)]
enum CurrentGpuAccess {
    NonExclusive {
        // Number of non-exclusive GPU accesses. Can be 0.
        num: AtomicUsize,
    },
    Exclusive {
        // Number of exclusive locks. Cannot be 0. If 0 is reached, we must jump to `NonExclusive`.
        num: usize,
    },
}

impl<T> CpuAccessibleBuffer<T> {
    /// Builds a new buffer with some data in it. Only allowed for sized data.
    pub fn from_data(
        device: Arc<Device>,
        usage: BufferUsage,
        host_cached: bool,
        data: T,
    ) -> Result<Arc<CpuAccessibleBuffer<T>>, DeviceMemoryAllocError>
    where
        T: Content + 'static,
    {
        unsafe {
            let uninitialized = CpuAccessibleBuffer::raw(
                device,
                mem::size_of::<T>(),
                usage,
                host_cached,
                iter::empty(),
            )?;

            // Note that we are in panic-unsafety land here. However a panic should never ever
            // happen here, so in theory we are safe.
            // TODO: check whether that's true ^

            {
                let mut mapping = uninitialized.write().unwrap();
                ptr::write::<T>(&mut *mapping, data)
            }

            Ok(uninitialized)
        }
    }

    /// Builds a new uninitialized buffer. Only allowed for sized data.
    #[inline]
    pub unsafe fn uninitialized(
        device: Arc<Device>,
        usage: BufferUsage,
        host_cached: bool,
    ) -> Result<Arc<CpuAccessibleBuffer<T>>, DeviceMemoryAllocError> {
        CpuAccessibleBuffer::raw(
            device,
            mem::size_of::<T>(),
            usage,
            host_cached,
            iter::empty(),
        )
    }
}

impl<T> CpuAccessibleBuffer<[T]> {
    /// Builds a new buffer that contains an array `T`. The initial data comes from an iterator
    /// that produces that list of Ts.
    pub fn from_iter<I>(
        device: Arc<Device>,
        usage: BufferUsage,
        host_cached: bool,
        data: I,
    ) -> Result<Arc<CpuAccessibleBuffer<[T]>>, DeviceMemoryAllocError>
    where
        I: ExactSizeIterator<Item = T>,
        T: Content + 'static,
    {
        unsafe {
            let uninitialized =
                CpuAccessibleBuffer::uninitialized_array(device, data.len(), usage, host_cached)?;

            // Note that we are in panic-unsafety land here. However a panic should never ever
            // happen here, so in theory we are safe.
            // TODO: check whether that's true ^

            {
                let mut mapping = uninitialized.write().unwrap();

                for (i, o) in data.zip(mapping.iter_mut()) {
                    ptr::write(o, i);
                }
            }

            Ok(uninitialized)
        }
    }

    /// Builds a new buffer. Can be used for arrays.
    #[inline]
    pub unsafe fn uninitialized_array(
        device: Arc<Device>,
        len: usize,
        usage: BufferUsage,
        host_cached: bool,
    ) -> Result<Arc<CpuAccessibleBuffer<[T]>>, DeviceMemoryAllocError> {
        CpuAccessibleBuffer::raw(
            device,
            len * mem::size_of::<T>(),
            usage,
            host_cached,
            iter::empty(),
        )
    }
}

impl<T: ?Sized> CpuAccessibleBuffer<T> {
    /// Builds a new buffer without checking the size.
    ///
    /// # Safety
    ///
    /// You must ensure that the size that you pass is correct for `T`.
    ///
    pub unsafe fn raw<'a, I>(
        device: Arc<Device>,
        size: usize,
        usage: BufferUsage,
        host_cached: bool,
        queue_families: I,
    ) -> Result<Arc<CpuAccessibleBuffer<T>>, DeviceMemoryAllocError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
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
                Err(_) => unreachable!(), // We don't use sparse binding, therefore the other
                                          // errors can't happen
            }
        };

        let mem = MemoryPool::alloc_from_requirements(
            &Device::standard_pool(&device),
            &mem_reqs,
            AllocLayout::Linear,
            MappingRequirement::Map,
            DedicatedAlloc::Buffer(&buffer),
            |m| {
                if m.is_host_cached() {
                    if host_cached {
                        AllocFromRequirementsFilter::Preferred
                    } else {
                        AllocFromRequirementsFilter::Allowed
                    }
                } else {
                    if host_cached {
                        AllocFromRequirementsFilter::Allowed
                    } else {
                        AllocFromRequirementsFilter::Preferred
                    }
                }
            },
        )?;
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        debug_assert!(mem.mapped_memory().is_some());
        buffer.bind_memory(mem.memory(), mem.offset())?;

        Ok(Arc::new(CpuAccessibleBuffer {
            inner: buffer,
            memory: mem,
            access: RwLock::new(CurrentGpuAccess::NonExclusive {
                num: AtomicUsize::new(0),
            }),
            queue_families: queue_families,
            marker: PhantomData,
        }))
    }
}

impl<T: ?Sized, A> CpuAccessibleBuffer<T, A> {
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

impl<T: ?Sized, A> CpuAccessibleBuffer<T, A>
where
    T: Content + 'static,
    A: MemoryPoolAlloc,
{
    /// Locks the buffer in order to read its content from the CPU.
    ///
    /// If the buffer is currently used in exclusive mode by the GPU, this function will return
    /// an error. Similarly if you called `write()` on the buffer and haven't dropped the lock,
    /// this function will return an error as well.
    ///
    /// After this function successfully locks the buffer, any attempt to submit a command buffer
    /// that uses it in exclusive mode will fail. You can still submit this buffer for non-exclusive
    /// accesses (ie. reads).
    #[inline]
    pub fn read(&self) -> Result<ReadLock<T>, ReadLockError> {
        let lock = match self.access.try_read() {
            Some(l) => l,
            // TODO: if a user simultaneously calls .write(), and write() is currently finding out
            //       that the buffer is in fact GPU locked, then we will return a CpuWriteLocked
            //       error instead of a GpuWriteLocked ; is this a problem? how do we fix this?
            None => return Err(ReadLockError::CpuWriteLocked),
        };

        if let CurrentGpuAccess::Exclusive { .. } = *lock {
            return Err(ReadLockError::GpuWriteLocked);
        }

        let offset = self.memory.offset();
        let range = offset..offset + self.inner.size();

        Ok(ReadLock {
            inner: unsafe { self.memory.mapped_memory().unwrap().read_write(range) },
            lock: lock,
        })
    }

    /// Locks the buffer in order to write its content from the CPU.
    ///
    /// If the buffer is currently in use by the GPU, this function will return an error. Similarly
    /// if you called `read()` on the buffer and haven't dropped the lock, this function will
    /// return an error as well.
    ///
    /// After this function successfully locks the buffer, any attempt to submit a command buffer
    /// that uses it and any attempt to call `read()` will return an error.
    #[inline]
    pub fn write(&self) -> Result<WriteLock<T>, WriteLockError> {
        let lock = match self.access.try_write() {
            Some(l) => l,
            // TODO: if a user simultaneously calls .read() or .write(), and the function is
            //       currently finding out that the buffer is in fact GPU locked, then we will
            //       return a CpuLocked error instead of a GpuLocked ; is this a problem?
            //       how do we fix this?
            None => return Err(WriteLockError::CpuLocked),
        };

        match *lock {
            CurrentGpuAccess::NonExclusive { ref num } if num.load(Ordering::SeqCst) == 0 => (),
            _ => return Err(WriteLockError::GpuLocked),
        }

        let offset = self.memory.offset();
        let range = offset..offset + self.inner.size();

        Ok(WriteLock {
            inner: unsafe { self.memory.mapped_memory().unwrap().read_write(range) },
            lock: lock,
        })
    }
}

unsafe impl<T: ?Sized, A> BufferAccess for CpuAccessibleBuffer<T, A>
where
    T: 'static + Send + Sync,
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
    fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool {
        self.conflict_key() == other.conflict_key() // TODO:
    }

    #[inline]
    fn conflicts_image(&self, other: &dyn ImageAccess) -> bool {
        false
    }

    #[inline]
    fn conflict_key(&self) -> (u64, usize) {
        (self.inner.key(), 0)
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive_access: bool, _: &Queue) -> Result<(), AccessError> {
        if exclusive_access {
            let mut lock = match self.access.try_write() {
                Some(lock) => lock,
                None => return Err(AccessError::AlreadyInUse),
            };

            match *lock {
                CurrentGpuAccess::NonExclusive { ref num } if num.load(Ordering::SeqCst) == 0 => (),
                _ => return Err(AccessError::AlreadyInUse),
            };

            *lock = CurrentGpuAccess::Exclusive { num: 1 };
            Ok(())
        } else {
            let lock = match self.access.try_read() {
                Some(lock) => lock,
                None => return Err(AccessError::AlreadyInUse),
            };

            match *lock {
                CurrentGpuAccess::Exclusive { .. } => return Err(AccessError::AlreadyInUse),
                CurrentGpuAccess::NonExclusive { ref num } => num.fetch_add(1, Ordering::SeqCst),
            };

            Ok(())
        }
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        // First, handle if we have a non-exclusive access.
        {
            // Since the buffer is in use by the GPU, it is invalid to hold a write-lock to
            // the buffer. The buffer can still be briefly in a write-locked state for the duration
            // of the check though.
            let read_lock = self.access.read();
            if let CurrentGpuAccess::NonExclusive { ref num } = *read_lock {
                let prev = num.fetch_add(1, Ordering::SeqCst);
                debug_assert!(prev >= 1);
                return;
            }
        }

        // If we reach here, this means that `access` contains `CurrentGpuAccess::Exclusive`.
        {
            // Same remark as above, but for writing.
            let mut write_lock = self.access.write();
            if let CurrentGpuAccess::Exclusive { ref mut num } = *write_lock {
                *num += 1;
            } else {
                unreachable!()
            }
        }
    }

    #[inline]
    unsafe fn unlock(&self) {
        // First, handle if we had a non-exclusive access.
        {
            // Since the buffer is in use by the GPU, it is invalid to hold a write-lock to
            // the buffer. The buffer can still be briefly in a write-locked state for the duration
            // of the check though.
            let read_lock = self.access.read();
            if let CurrentGpuAccess::NonExclusive { ref num } = *read_lock {
                let prev = num.fetch_sub(1, Ordering::SeqCst);
                debug_assert!(prev >= 1);
                return;
            }
        }

        // If we reach here, this means that `access` contains `CurrentGpuAccess::Exclusive`.
        {
            // Same remark as above, but for writing.
            let mut write_lock = self.access.write();
            if let CurrentGpuAccess::Exclusive { ref mut num } = *write_lock {
                if *num != 1 {
                    *num -= 1;
                    return;
                }
            } else {
                // Can happen if we lock in exclusive mode N times, and unlock N+1 times with the
                // last two unlocks happen simultaneously.
                panic!()
            }

            *write_lock = CurrentGpuAccess::NonExclusive {
                num: AtomicUsize::new(0),
            };
        }
    }
}

unsafe impl<T: ?Sized, A> TypedBufferAccess for CpuAccessibleBuffer<T, A>
where
    T: 'static + Send + Sync,
{
    type Content = T;
}

unsafe impl<T: ?Sized, A> DeviceOwned for CpuAccessibleBuffer<T, A> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl<T: ?Sized, A> PartialEq for CpuAccessibleBuffer<T, A>
where
    T: 'static + Send + Sync,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T: ?Sized, A> Eq for CpuAccessibleBuffer<T, A> where T: 'static + Send + Sync {}

impl<T: ?Sized, A> Hash for CpuAccessibleBuffer<T, A>
where
    T: 'static + Send + Sync,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

/// Object that can be used to read or write the content of a `CpuAccessibleBuffer`.
///
/// Note that this object holds a rwlock read guard on the chunk. If another thread tries to access
/// this buffer's content or tries to submit a GPU command that uses this buffer, it will block.
pub struct ReadLock<'a, T: ?Sized + 'a> {
    inner: MemCpuAccess<'a, T>,
    lock: RwLockReadGuard<'a, CurrentGpuAccess>,
}

impl<'a, T: ?Sized + 'a> ReadLock<'a, T> {
    /// Makes a new `ReadLock` to access a sub-part of the current `ReadLock`.
    #[inline]
    pub fn map<U: ?Sized + 'a, F>(self, f: F) -> ReadLock<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
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

/// Error when attempting to CPU-read a buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadLockError {
    /// The buffer is already locked for write mode by the CPU.
    CpuWriteLocked,
    /// The buffer is already locked for write mode by the GPU.
    GpuWriteLocked,
}

impl error::Error for ReadLockError {}

impl fmt::Display for ReadLockError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                ReadLockError::CpuWriteLocked => {
                    "the buffer is already locked for write mode by the CPU"
                }
                ReadLockError::GpuWriteLocked => {
                    "the buffer is already locked for write mode by the GPU"
                }
            }
        )
    }
}

/// Object that can be used to read or write the content of a `CpuAccessibleBuffer`.
///
/// Note that this object holds a rwlock write guard on the chunk. If another thread tries to access
/// this buffer's content or tries to submit a GPU command that uses this buffer, it will block.
pub struct WriteLock<'a, T: ?Sized + 'a> {
    inner: MemCpuAccess<'a, T>,
    lock: RwLockWriteGuard<'a, CurrentGpuAccess>,
}

impl<'a, T: ?Sized + 'a> WriteLock<'a, T> {
    /// Makes a new `WriteLock` to access a sub-part of the current `WriteLock`.
    #[inline]
    pub fn map<U: ?Sized + 'a, F>(self, f: F) -> WriteLock<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
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

/// Error when attempting to CPU-write a buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteLockError {
    /// The buffer is already locked by the CPU.
    CpuLocked,
    /// The buffer is already locked by the GPU.
    GpuLocked,
}

impl error::Error for WriteLockError {}

impl fmt::Display for WriteLockError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                WriteLockError::CpuLocked => "the buffer is already locked by the CPU",
                WriteLockError::GpuLocked => "the buffer is already locked by the GPU",
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use buffer::{BufferUsage, CpuAccessibleBuffer};

    #[test]
    fn create_empty_buffer() {
        let (device, queue) = gfx_dev_and_queue!();

        const EMPTY: [i32; 0] = [];

        let _ = CpuAccessibleBuffer::from_data(device, BufferUsage::all(), false, EMPTY.iter());
    }
}
