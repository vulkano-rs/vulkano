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

use super::{
    sys::UnsafeBuffer, BufferAccess, BufferAccessObject, BufferContents, BufferInner, BufferUsage,
};
use crate::{
    buffer::{sys::UnsafeBufferCreateInfo, BufferCreationError, TypedBufferAccess},
    device::{physical::QueueFamily, Device, DeviceOwned, Queue},
    memory::{
        pool::{
            AllocFromRequirementsFilter, AllocLayout, MappingRequirement, MemoryPoolAlloc,
            PotentialDedicatedAllocation, StdMemoryPoolAlloc,
        },
        DedicatedAllocation, DeviceMemoryAllocationError, MappedDeviceMemory, MemoryPool,
    },
    sync::{AccessError, Sharing},
    DeviceSize,
};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use smallvec::SmallVec;
use std::{
    error, fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::size_of,
    ops::{Deref, DerefMut, Range},
    ptr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

/// Buffer whose content is accessible by the CPU.
///
/// Setting the `host_cached` field on the various initializers to `true` will make it so
/// the `CpuAccessibleBuffer` prefers to allocate from host_cached memory. Host cached
/// memory caches GPU data on the CPU side. This can be more performant in cases where
/// the cpu needs to read data coming off the GPU.
#[derive(Debug)]
pub struct CpuAccessibleBuffer<T, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>>
where
    T: BufferContents + ?Sized,
{
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

impl<T> CpuAccessibleBuffer<T>
where
    T: BufferContents,
{
    /// Builds a new buffer with some data in it. Only allowed for sized data.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    pub fn from_data(
        device: Arc<Device>,
        usage: BufferUsage,
        host_cached: bool,
        data: T,
    ) -> Result<Arc<CpuAccessibleBuffer<T>>, DeviceMemoryAllocationError> {
        unsafe {
            let uninitialized = CpuAccessibleBuffer::raw(
                device,
                size_of::<T>() as DeviceSize,
                usage,
                host_cached,
                [],
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
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    #[inline]
    pub unsafe fn uninitialized(
        device: Arc<Device>,
        usage: BufferUsage,
        host_cached: bool,
    ) -> Result<Arc<CpuAccessibleBuffer<T>>, DeviceMemoryAllocationError> {
        CpuAccessibleBuffer::raw(device, size_of::<T>() as DeviceSize, usage, host_cached, [])
    }
}

impl<T> CpuAccessibleBuffer<[T]>
where
    [T]: BufferContents,
{
    /// Builds a new buffer that contains an array `T`. The initial data comes from an iterator
    /// that produces that list of Ts.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `data` is empty.
    pub fn from_iter<I>(
        device: Arc<Device>,
        usage: BufferUsage,
        host_cached: bool,
        data: I,
    ) -> Result<Arc<CpuAccessibleBuffer<[T]>>, DeviceMemoryAllocationError>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let data = data.into_iter();

        unsafe {
            let uninitialized = CpuAccessibleBuffer::uninitialized_array(
                device,
                data.len() as DeviceSize,
                usage,
                host_cached,
            )?;

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
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `len` is zero.
    #[inline]
    pub unsafe fn uninitialized_array(
        device: Arc<Device>,
        len: DeviceSize,
        usage: BufferUsage,
        host_cached: bool,
    ) -> Result<Arc<CpuAccessibleBuffer<[T]>>, DeviceMemoryAllocationError> {
        CpuAccessibleBuffer::raw(
            device,
            len * size_of::<T>() as DeviceSize,
            usage,
            host_cached,
            [],
        )
    }
}

impl<T> CpuAccessibleBuffer<T>
where
    T: BufferContents + ?Sized,
{
    /// Builds a new buffer without checking the size.
    ///
    /// # Safety
    ///
    /// - You must ensure that the size that you pass is correct for `T`.
    ///
    /// # Panics
    ///
    /// - Panics if `size` is zero.
    pub unsafe fn raw<'a, I>(
        device: Arc<Device>,
        size: DeviceSize,
        usage: BufferUsage,
        host_cached: bool,
        queue_families: I,
    ) -> Result<Arc<CpuAccessibleBuffer<T>>, DeviceMemoryAllocationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        let queue_families = queue_families
            .into_iter()
            .map(|f| f.id())
            .collect::<SmallVec<[u32; 4]>>();

        let buffer = {
            match UnsafeBuffer::new(
                device.clone(),
                UnsafeBufferCreateInfo {
                    sharing: if queue_families.len() >= 2 {
                        Sharing::Concurrent(queue_families.clone())
                    } else {
                        Sharing::Exclusive
                    },
                    size,
                    usage,
                    ..Default::default()
                },
            ) {
                Ok(b) => b,
                Err(BufferCreationError::AllocError(err)) => return Err(err),
                Err(_) => unreachable!(), // We don't use sparse binding, therefore the other
                                          // errors can't happen
            }
        };
        let mem_reqs = buffer.memory_requirements();

        let mem = MemoryPool::alloc_from_requirements(
            &Device::standard_pool(&device),
            &mem_reqs,
            AllocLayout::Linear,
            MappingRequirement::Map,
            Some(DedicatedAllocation::Buffer(&buffer)),
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

impl<T, A> CpuAccessibleBuffer<T, A>
where
    T: BufferContents + ?Sized,
{
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

impl<T, A> CpuAccessibleBuffer<T, A>
where
    T: BufferContents + ?Sized,
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

        let mapped_memory = self.memory.mapped_memory().unwrap();
        let offset = self.memory.offset();
        let range = offset..offset + self.inner.size();

        let bytes = unsafe {
            // If there are other read locks being held at this point, they also called
            // `invalidate_range` when locking. The GPU can't write data while the CPU holds a read
            // lock, so there will no new data and this call will do nothing.
            // TODO: probably still more efficient to call it only if we're the first to acquire a
            // read lock, but the number of CPU locks isn't currently tracked anywhere.
            mapped_memory.invalidate_range(range.clone()).unwrap();
            mapped_memory.read(range.clone()).unwrap()
        };

        Ok(ReadLock {
            data: T::from_bytes(bytes).unwrap(),
            range,
            lock,
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

        let mapped_memory = self.memory.mapped_memory().unwrap();
        let offset = self.memory.offset();
        let range = offset..offset + self.inner.size();

        let bytes = unsafe {
            mapped_memory.invalidate_range(range.clone()).unwrap();
            mapped_memory.write(range.clone()).unwrap()
        };

        Ok(WriteLock {
            data: T::from_bytes_mut(bytes).unwrap(),
            mapped_memory,
            range,
            lock,
        })
    }
}

unsafe impl<T, A> BufferAccess for CpuAccessibleBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    #[inline]
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.inner,
            offset: 0,
        }
    }

    #[inline]
    fn size(&self) -> DeviceSize {
        self.inner.size()
    }

    #[inline]
    fn conflict_key(&self) -> (u64, u64) {
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

impl<T, A> BufferAccessObject for Arc<CpuAccessibleBuffer<T, A>>
where
    T: BufferContents + ?Sized,
    A: Send + Sync + 'static,
{
    #[inline]
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess> {
        self.clone()
    }
}

unsafe impl<T, A> TypedBufferAccess for CpuAccessibleBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    type Content = T;
}

unsafe impl<T, A> DeviceOwned for CpuAccessibleBuffer<T, A>
where
    T: BufferContents + ?Sized,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl<T, A> PartialEq for CpuAccessibleBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T, A> Eq for CpuAccessibleBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
}

impl<T, A> Hash for CpuAccessibleBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
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
#[derive(Debug)]
pub struct ReadLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    data: &'a T,
    range: Range<DeviceSize>,
    lock: RwLockReadGuard<'a, CurrentGpuAccess>,
}

impl<'a, T> Deref for ReadLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.data
    }
}

/// Object that can be used to read or write the content of a `CpuAccessibleBuffer`.
///
/// Note that this object holds a rwlock write guard on the chunk. If another thread tries to access
/// this buffer's content or tries to submit a GPU command that uses this buffer, it will block.
#[derive(Debug)]
pub struct WriteLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    data: &'a mut T,
    mapped_memory: &'a MappedDeviceMemory,
    range: Range<DeviceSize>,
    lock: RwLockWriteGuard<'a, CurrentGpuAccess>,
}

impl<'a, T> Drop for WriteLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.mapped_memory.flush_range(self.range.clone()).unwrap();
        }
    }
}

impl<'a, T> Deref for WriteLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.data
    }
}

impl<'a, T> DerefMut for WriteLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.data
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
    use crate::buffer::{BufferUsage, CpuAccessibleBuffer};

    #[test]
    fn create_empty_buffer() {
        let (device, queue) = gfx_dev_and_queue!();

        const EMPTY: [i32; 0] = [];

        assert_should_panic!({
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, EMPTY)
                .unwrap();
            CpuAccessibleBuffer::from_iter(device, BufferUsage::all(), false, EMPTY.into_iter())
                .unwrap();
        });
    }
}
