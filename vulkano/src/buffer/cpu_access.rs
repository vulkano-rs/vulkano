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
    sys::{Buffer, BufferMemory, RawBuffer},
    BufferAccess, BufferAccessObject, BufferContents, BufferError, BufferInner, BufferUsage,
};
use crate::{
    buffer::{sys::BufferCreateInfo, TypedBufferAccess},
    device::{Device, DeviceOwned},
    memory::{
        allocator::{
            AllocationCreateInfo, AllocationCreationError, AllocationType, DeviceLayout,
            MemoryAllocatePreference, MemoryAllocator, MemoryUsage,
        },
        DedicatedAllocation,
    },
    sync::Sharing,
    DeviceSize,
};
use smallvec::SmallVec;
use std::{
    alloc::Layout,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Deref, DerefMut, Range},
    ptr,
    sync::Arc,
};

/// Buffer whose content is accessible by the CPU.
///
/// Setting the `host_cached` field on the various initializers to `true` will make it so
/// the `CpuAccessibleBuffer` prefers to allocate from host_cached memory. Host cached
/// memory caches GPU data on the CPU side. This can be more performant in cases where
/// the cpu needs to read data coming off the GPU.
#[derive(Debug)]
pub struct CpuAccessibleBuffer<T>
where
    T: BufferContents + ?Sized,
{
    inner: Arc<Buffer>,
    marker: PhantomData<Box<T>>,
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
    /// - Panics if `T` has an alignment greater than `64`.
    pub fn from_data(
        allocator: &(impl MemoryAllocator + ?Sized),
        usage: BufferUsage,
        host_cached: bool,
        data: T,
    ) -> Result<Arc<CpuAccessibleBuffer<T>>, AllocationCreationError> {
        unsafe {
            let uninitialized = CpuAccessibleBuffer::raw(
                allocator,
                DeviceLayout::from_layout(Layout::new::<T>())
                    .expect("can't allocate memory for zero-sized types"),
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
    /// - Panics if `T` has an alignment greater than `64`.
    pub unsafe fn uninitialized(
        allocator: &(impl MemoryAllocator + ?Sized),
        usage: BufferUsage,
        host_cached: bool,
    ) -> Result<Arc<CpuAccessibleBuffer<T>>, AllocationCreationError> {
        CpuAccessibleBuffer::raw(
            allocator,
            DeviceLayout::from_layout(Layout::new::<T>())
                .expect("can't allocate memory for zero-sized types"),
            usage,
            host_cached,
            [],
        )
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
    /// - Panics if `T` has an alignment greater than `64`.
    /// - Panics if `data` is empty.
    pub fn from_iter<I>(
        allocator: &(impl MemoryAllocator + ?Sized),
        usage: BufferUsage,
        host_cached: bool,
        data: I,
    ) -> Result<Arc<CpuAccessibleBuffer<[T]>>, AllocationCreationError>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let data = data.into_iter();

        unsafe {
            let uninitialized = CpuAccessibleBuffer::uninitialized_array(
                allocator,
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
    /// - Panics if `T` has an alignment greater than `64`.
    /// - Panics if `len` is zero.
    pub unsafe fn uninitialized_array(
        allocator: &(impl MemoryAllocator + ?Sized),
        len: DeviceSize,
        usage: BufferUsage,
        host_cached: bool,
    ) -> Result<Arc<CpuAccessibleBuffer<[T]>>, AllocationCreationError> {
        CpuAccessibleBuffer::raw(
            allocator,
            DeviceLayout::from_layout(Layout::array::<T>(len.try_into().unwrap()).unwrap())
                .expect("can't allocate memory for zero-sized types"),
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
    /// - Panics if `layout.alignment()` exceeds `64`.
    pub unsafe fn raw(
        allocator: &(impl MemoryAllocator + ?Sized),
        layout: DeviceLayout,
        usage: BufferUsage,
        host_cached: bool,
        queue_family_indices: impl IntoIterator<Item = u32>,
    ) -> Result<Arc<CpuAccessibleBuffer<T>>, AllocationCreationError> {
        assert!(layout.alignment().as_devicesize() <= 64);

        let queue_family_indices: SmallVec<[_; 4]> = queue_family_indices.into_iter().collect();

        let raw_buffer = RawBuffer::new(
            allocator.device().clone(),
            BufferCreateInfo {
                sharing: if queue_family_indices.len() >= 2 {
                    Sharing::Concurrent(queue_family_indices)
                } else {
                    Sharing::Exclusive
                },
                size: layout.size(),
                usage,
                ..Default::default()
            },
        )
        .map_err(|err| match err {
            BufferError::AllocError(err) => err,
            // We don't use sparse-binding, therefore the other errors can't happen.
            _ => unreachable!(),
        })?;
        let mut requirements = *raw_buffer.memory_requirements();
        requirements.layout = requirements.layout.align_to(layout.alignment()).unwrap();
        let create_info = AllocationCreateInfo {
            requirements,
            allocation_type: AllocationType::Linear,
            usage: if host_cached {
                MemoryUsage::Download
            } else {
                MemoryUsage::Upload
            },
            allocate_preference: MemoryAllocatePreference::Unknown,
            dedicated_allocation: Some(DedicatedAllocation::Buffer(&raw_buffer)),
            ..Default::default()
        };

        match allocator.allocate_unchecked(create_info) {
            Ok(mut allocation) => {
                debug_assert!(
                    allocation.offset() % requirements.layout.alignment().as_nonzero() == 0
                );
                debug_assert!(allocation.size() == requirements.layout.size());

                // The implementation might require a larger size than we wanted. With this it is
                // easier to invalidate and flush the whole buffer. It does not affect the
                // allocation in any way.
                allocation.shrink(layout.size());
                let inner = Arc::new(
                    raw_buffer
                        .bind_memory_unchecked(allocation)
                        .map_err(|(err, _, _)| err)?,
                );

                Ok(Arc::new(CpuAccessibleBuffer {
                    inner,
                    marker: PhantomData,
                }))
            }
            Err(err) => Err(err),
        }
    }
}

impl<T> CpuAccessibleBuffer<T>
where
    T: BufferContents + ?Sized,
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
    pub fn read(&self) -> Result<ReadLock<'_, T>, ReadLockError> {
        let allocation = match self.inner.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        let range = self.inner().offset..self.inner().offset + self.size();
        let mut state = self.inner.state();

        unsafe {
            state.check_cpu_read(range.clone())?;
            state.cpu_read_lock(range.clone());
        }

        let bytes = unsafe {
            // If there are other read locks being held at this point, they also called
            // `invalidate_range` when locking. The GPU can't write data while the CPU holds a read
            // lock, so there will no new data and this call will do nothing.
            // TODO: probably still more efficient to call it only if we're the first to acquire a
            // read lock, but the number of CPU locks isn't currently tracked anywhere.
            allocation.invalidate_range(0..self.size()).unwrap();
            allocation.mapped_slice().unwrap()
        };

        Ok(ReadLock {
            buffer: self,
            range,
            data: T::from_bytes(bytes).unwrap(),
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
    pub fn write(&self) -> Result<WriteLock<'_, T>, WriteLockError> {
        let allocation = match self.inner.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        let range = self.inner().offset..self.inner().offset + self.size();
        let mut state = self.inner.state();

        unsafe {
            state.check_cpu_write(range.clone())?;
            state.cpu_write_lock(range.clone());
        }

        let bytes = unsafe {
            allocation.invalidate_range(0..self.size()).unwrap();
            allocation.write(0..self.size()).unwrap()
        };

        Ok(WriteLock {
            buffer: self,
            range,
            data: T::from_bytes_mut(bytes).unwrap(),
        })
    }
}

unsafe impl<T> BufferAccess for CpuAccessibleBuffer<T>
where
    T: BufferContents + ?Sized,
{
    fn inner(&self) -> BufferInner<'_> {
        BufferInner {
            buffer: &self.inner,
            offset: 0,
        }
    }

    fn size(&self) -> DeviceSize {
        self.inner.size()
    }
}

impl<T> BufferAccessObject for Arc<CpuAccessibleBuffer<T>>
where
    T: BufferContents + ?Sized,
{
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess> {
        self.clone()
    }
}

unsafe impl<T> TypedBufferAccess for CpuAccessibleBuffer<T>
where
    T: BufferContents + ?Sized,
{
    type Content = T;
}

unsafe impl<T> DeviceOwned for CpuAccessibleBuffer<T>
where
    T: BufferContents + ?Sized,
{
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl<T> PartialEq for CpuAccessibleBuffer<T>
where
    T: BufferContents + ?Sized,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T> Eq for CpuAccessibleBuffer<T> where T: BufferContents + ?Sized {}

impl<T> Hash for CpuAccessibleBuffer<T>
where
    T: BufferContents + ?Sized,
{
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
    T: BufferContents + ?Sized,
{
    buffer: &'a CpuAccessibleBuffer<T>,
    range: Range<DeviceSize>,
    data: &'a T,
}

impl<'a, T> Drop for ReadLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    fn drop(&mut self) {
        unsafe {
            let mut state = self.buffer.inner.state();
            state.cpu_read_unlock(self.range.clone());
        }
    }
}

impl<'a, T> Deref for ReadLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    type Target = T;

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
    T: BufferContents + ?Sized,
{
    buffer: &'a CpuAccessibleBuffer<T>,
    range: Range<DeviceSize>,
    data: &'a mut T,
}

impl<'a, T> Drop for WriteLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    fn drop(&mut self) {
        let allocation = match self.buffer.inner.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        unsafe {
            allocation.flush_range(0..self.buffer.size()).unwrap();

            let mut state = self.buffer.inner.state();
            state.cpu_write_unlock(self.range.clone());
        }
    }
}

impl<'a, T> Deref for WriteLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    type Target = T;

    fn deref(&self) -> &T {
        self.data
    }
}

impl<'a, T> DerefMut for WriteLock<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
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

impl Error for ReadLockError {}

impl Display for ReadLockError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
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

impl Error for WriteLockError {}

impl Display for WriteLockError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
                WriteLockError::CpuLocked => "the buffer is already locked by the CPU",
                WriteLockError::GpuLocked => "the buffer is already locked by the GPU",
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::StandardMemoryAllocator;

    #[test]
    fn create_empty_buffer() {
        let (device, _queue) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        const EMPTY: [i32; 0] = [];

        assert_should_panic!({
            CpuAccessibleBuffer::from_data(
                &memory_allocator,
                BufferUsage::TRANSFER_DST,
                false,
                EMPTY,
            )
            .unwrap();
            CpuAccessibleBuffer::from_iter(
                &memory_allocator,
                BufferUsage::TRANSFER_DST,
                false,
                EMPTY.into_iter(),
            )
            .unwrap();
        });
    }
}
