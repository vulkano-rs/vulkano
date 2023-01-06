// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{allocator::Arena, Buffer, BufferContents, BufferError, BufferMemory};
use crate::{
    device::{Device, DeviceOwned},
    memory::{self, allocator::DeviceAlignment, is_aligned},
    DeviceSize, NonZeroDeviceSize,
};
use bytemuck::PodCastError;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::{self, align_of, size_of},
    ops::{Deref, DerefMut, Range, RangeBounds},
    sync::Arc,
};

/// A subpart of a buffer.
///
/// This type doesn't correspond to any Vulkan object, it exists for API convenience.
#[derive(Debug)]
#[repr(C)]
pub struct Subbuffer<T: ?Sized> {
    offset: DeviceSize,
    size: DeviceSize,
    parent: SubbufferParent,
    marker: PhantomData<Arc<T>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum SubbufferParent {
    Arena(Arc<Arena>),
    Buffer(Arc<Buffer>),
}

impl<T: ?Sized> Subbuffer<T> {
    pub(super) fn from_buffer(buffer: Arc<Buffer>) -> Self {
        Subbuffer {
            offset: 0,
            size: buffer.size(),
            parent: SubbufferParent::Buffer(buffer),
            marker: PhantomData,
        }
    }

    pub(super) fn from_arena(arena: Arc<Arena>, offset: DeviceSize, size: DeviceSize) -> Self {
        Subbuffer {
            offset,
            size,
            parent: SubbufferParent::Arena(arena),
            marker: PhantomData,
        }
    }

    /// Returns the offset of the subbuffer, in bytes, relative to the buffer.
    pub fn offset(&self) -> DeviceSize {
        self.offset
    }

    /// Returns the offset of the subbuffer, in bytes, relative to the [`DeviceMemory`] block.
    fn memory_offset(&self) -> DeviceSize {
        let allocation = match self.buffer().memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        allocation.offset() + self.offset
    }

    /// Returns the size of the subbuffer in bytes.
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    /// Returns the buffer that this subbuffer is a part of.
    pub fn buffer(&self) -> &Arc<Buffer> {
        match &self.parent {
            SubbufferParent::Arena(arena) => arena.buffer(),
            SubbufferParent::Buffer(buffer) => buffer,
        }
    }

    /// Returns the device address for this subbuffer.
    pub fn device_address(&self) -> Result<NonZeroDeviceSize, BufferError> {
        self.buffer().device_address().map(|ptr| {
            // SAFETY: The original address came from the Vulkan implementation, and allocation
            // sizes are guaranteed to not exceed `DeviceLayout::MAX_SIZE`, so the offset better be
            // in range.
            unsafe { NonZeroDeviceSize::new_unchecked(ptr.get() + self.offset) }
        })
    }

    /// Changes the `T` generic parameter of the subbffer to the desired type.
    ///
    /// You should **always** prefer the safe functions [`try_from_bytes`], [`into_bytes`],
    /// [`try_cast`], [`try_cast_slice`] or [`into_slice`].
    ///
    /// # Safety
    ///
    /// - Correct offset and size must be ensured before using this `Subbuffer` on the device.
    ///
    /// [`try_from_bytes`]: Self::try_from_bytes
    /// [`into_bytes`]: Self::into_bytes
    /// [`try_cast`]: Self::try_cast
    /// [`try_cast_slice`]: Self::try_cast_slice
    /// [`into_slice`]: Self::into_slice
    pub unsafe fn reinterpret<U: ?Sized>(self) -> Subbuffer<U> {
        // SAFETY: All `Subbuffer`s share the same layout.
        mem::transmute::<Subbuffer<T>, Subbuffer<U>>(self)
    }

    /// Same as [`reinterpret`], except it works with a reference to the subbuffer.
    ///
    /// [`reinterpret`]: Self::reinterpret
    pub unsafe fn reinterpret_ref<U: ?Sized>(&self) -> &Subbuffer<U> {
        assert!(size_of::<Subbuffer<T>>() == size_of::<Subbuffer<U>>());
        assert!(align_of::<Subbuffer<T>>() == align_of::<Subbuffer<U>>());

        // SAFETY: All `Subbuffer`s share the same layout.
        mem::transmute::<&Subbuffer<T>, &Subbuffer<U>>(self)
    }

    /// Casts the subbuffer to a slice of raw bytes.
    pub fn into_bytes(self) -> Subbuffer<[u8]> {
        unsafe { self.reinterpret() }
    }

    /// Same as [`into_bytes`], except it works with a reference to the subbuffer.
    ///
    /// [`into_bytes`]: Self::into_bytes
    pub fn as_bytes(&self) -> &Subbuffer<[u8]> {
        unsafe { self.reinterpret_ref() }
    }
}

impl<T> Subbuffer<T>
where
    T: BufferContents + ?Sized,
{
    /// Locks the subbuffer in order to read its content from the host.
    ///
    /// If the subbuffer is currently used in exclusive mode by the device, this function will
    /// return an error. Similarly if you called [`write`] on the buffer and haven't dropped the
    /// lock, this function will return an error as well.
    ///
    /// After this function successfully locks the subbuffer, any attempt to submit a command buffer
    /// that uses it in exclusive mode will fail. You can still submit this subbuffer for
    /// non-exclusive accesses (ie. reads).
    ///
    /// [`write`]: Self::write
    pub fn read(&self) -> Result<BufferReadGuard<'_, T>, BufferError> {
        let allocation = match self.buffer().memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => todo!("`Subbuffer::read` doesn't support sparse binding yet"),
        };

        let range = self.offset..self.offset + self.size;

        if let Some(atom_size) = allocation.atom_size() {
            if !is_aligned(range.start, atom_size) || !is_aligned(range.end, atom_size) {
                return Err(BufferError::SubbufferNotAlignedToAtomSize { range, atom_size });
            }
        }

        let mut state = self.buffer().state();
        state.check_cpu_read(range.clone())?;
        unsafe { state.cpu_read_lock(range.clone()) };

        if allocation.atom_size().is_some() {
            // If there are other read locks being held at this point, they also called
            // `invalidate_range` when locking. The GPU can't write data while the CPU holds a read
            // lock, so there will be no new data and this call will do nothing.
            // TODO: probably still more efficient to call it only if we're the first to acquire a
            // read lock, but the number of CPU locks isn't currently tracked anywhere.
            unsafe { allocation.invalidate_range(range.clone()) }?;
        }

        let bytes = unsafe { allocation.read(range) }.ok_or(BufferError::MemoryNotHostVisible)?;
        let data = T::from_bytes(bytes).unwrap();

        Ok(BufferReadGuard {
            subbuffer: self,
            data,
        })
    }

    /// Locks the subbuffer in order to write its content from the host.
    ///
    /// If the subbuffer is currently in use by the device, this function will return an error.
    /// Similarly if you called [`read`] on the subbuffer and haven't dropped the lock, this
    /// function will return an error as well.
    ///
    /// After this function successfully locks the buffer, any attempt to submit a command buffer
    /// that uses it and any attempt to call `read` will return an error.
    ///
    /// [`read`]: Self::read
    pub fn write(&self) -> Result<BufferWriteGuard<'_, T>, BufferError> {
        let allocation = match self.buffer().memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => todo!("`Subbuffer::write` doesn't support sparse binding yet"),
        };

        let range = self.offset..self.offset + self.size;

        if let Some(atom_size) = allocation.atom_size() {
            if !is_aligned(range.start, atom_size) || !is_aligned(range.end, atom_size) {
                return Err(BufferError::SubbufferNotAlignedToAtomSize { range, atom_size });
            }
        }

        let mut state = self.buffer().state();
        state.check_cpu_write(range.clone())?;
        unsafe { state.cpu_write_lock(range.clone()) };

        if allocation.atom_size().is_some() {
            unsafe { allocation.invalidate_range(range.clone()) }?;
        }

        let bytes = unsafe { allocation.write(range) }.ok_or(BufferError::MemoryNotHostVisible)?;
        let data = T::from_bytes_mut(bytes).unwrap();

        Ok(BufferWriteGuard {
            subbuffer: self,
            data,
        })
    }
}

impl<T> Subbuffer<T> {
    /// Converts the subbuffer to a slice of one element.
    pub fn into_slice(self) -> Subbuffer<[T]> {
        unsafe { self.reinterpret() }
    }
}

impl<T> Subbuffer<T>
where
    T: BufferContents,
{
    /// Tries to cast a subbuffer of raw bytes to a `Subbuffer<T>`.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `T` has an alignment greater than `64`.
    pub fn try_from_bytes(subbuffer: Subbuffer<[u8]>) -> Result<Self, PodCastError> {
        assert_valid_type_param::<T>();

        if subbuffer.size() != size_of::<T>() as DeviceSize {
            Err(PodCastError::SizeMismatch)
        } else if !is_aligned(subbuffer.memory_offset(), DeviceAlignment::of::<T>()) {
            Err(PodCastError::TargetAlignmentGreaterAndInputNotAligned)
        } else {
            Ok(unsafe { subbuffer.reinterpret() })
        }
    }

    /// Tries to cast the subbuffer to a different type.
    ///
    /// # Panics
    ///
    /// - Panics if `U` has zero size.
    /// - Panics if `U` has an alignment greater than `64`.
    pub fn try_cast<U>(self) -> Result<Subbuffer<U>, PodCastError>
    where
        U: BufferContents,
    {
        assert_valid_type_param::<U>();

        if size_of::<U>() != size_of::<T>() {
            Err(PodCastError::SizeMismatch)
        } else if align_of::<U>() > align_of::<T>()
            && !is_aligned(self.memory_offset(), DeviceAlignment::of::<U>())
        {
            Err(PodCastError::TargetAlignmentGreaterAndInputNotAligned)
        } else {
            Ok(unsafe { self.reinterpret() })
        }
    }
}

impl<T> Subbuffer<[T]> {
    /// Returns the number of elements in the slice.
    pub fn len(&self) -> DeviceSize {
        assert_valid_type_param::<T>();

        debug_assert!(self.size() % size_of::<T>() as DeviceSize == 0);

        self.size / size_of::<T>() as DeviceSize
    }

    /// Reduces the subbuffer to just one element of the slice, or returns [`None`] if the `index`
    /// is out of bounds.
    pub fn index(self, index: DeviceSize) -> Option<Subbuffer<T>> {
        (index < self.len()).then_some(Subbuffer {
            offset: self.offset + index * size_of::<T>() as DeviceSize,
            size: size_of::<T>() as DeviceSize,
            parent: self.parent,
            marker: PhantomData,
        })
    }

    /// Reduces the subbuffer to just a range of the slice, or returns [`None`] if the `range` is
    /// out of bounds.
    pub fn slice(mut self, range: impl RangeBounds<DeviceSize>) -> Option<Subbuffer<[T]>> {
        let Range { start, end } = memory::range(range, ..self.len())?;

        self.offset += start * size_of::<T>() as DeviceSize;
        self.size = (end - start) * size_of::<T>() as DeviceSize;

        Some(self)
    }
}

impl<T> Subbuffer<[T]>
where
    [T]: BufferContents,
{
    /// Tries to cast the slice to a different element type.
    ///
    /// # Panics
    ///
    /// - Panics if `U` has zero size.
    /// - Panics if `U` has an alignment greater than `64`.
    pub fn try_cast_slice<U>(self) -> Result<Subbuffer<[U]>, PodCastError>
    where
        [U]: BufferContents,
    {
        assert_valid_type_param::<U>();

        if size_of::<U>() != size_of::<T>() && self.size() % size_of::<U>() as DeviceSize != 0 {
            Err(PodCastError::OutputSliceWouldHaveSlop)
        } else if align_of::<U>() > align_of::<T>()
            && !is_aligned(self.memory_offset(), DeviceAlignment::of::<U>())
        {
            Err(PodCastError::TargetAlignmentGreaterAndInputNotAligned)
        } else {
            Ok(unsafe { self.reinterpret() })
        }
    }
}

#[inline(always)]
fn assert_valid_type_param<T>() {
    assert!(size_of::<T>() != 0);
    assert!(align_of::<T>() <= 64);
}

impl From<Arc<Buffer>> for Subbuffer<[u8]> {
    fn from(buffer: Arc<Buffer>) -> Self {
        Self::from_buffer(buffer)
    }
}

impl<T: ?Sized> Clone for Subbuffer<T> {
    fn clone(&self) -> Self {
        Subbuffer {
            parent: self.parent.clone(),
            ..*self
        }
    }
}

unsafe impl<T: ?Sized> DeviceOwned for Subbuffer<T> {
    fn device(&self) -> &Arc<Device> {
        self.buffer().device()
    }
}

impl<T: ?Sized> PartialEq for Subbuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.parent == other.parent && self.offset == other.offset && self.size == other.size
    }
}

impl<T: ?Sized> Eq for Subbuffer<T> {}

impl<T: ?Sized> Hash for Subbuffer<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.parent.hash(state);
        self.offset.hash(state);
        self.size.hash(state);
    }
}

/// RAII structure used to release the CPU access of a subbuffer when dropped.
///
/// This structure is created by the [`read`] method on [`Subbuffer`].
///
/// [`read`]: Subbuffer::read
#[derive(Debug)]
pub struct BufferReadGuard<'a, T: ?Sized> {
    subbuffer: &'a Subbuffer<T>,
    data: &'a T,
}

impl<T: ?Sized> Drop for BufferReadGuard<'_, T> {
    fn drop(&mut self) {
        let range = self.subbuffer.offset..self.subbuffer.offset + self.subbuffer.size;
        let mut state = self.subbuffer.buffer().state();
        unsafe { state.cpu_read_unlock(range) };
    }
}

impl<T: ?Sized> Deref for BufferReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

/// RAII structure used to release the CPU write access of a subbuffer when dropped.
///
/// This structure is created by the [`write`] method on [`Subbuffer`].
///
/// [`write`]: Subbuffer::write
#[derive(Debug)]
pub struct BufferWriteGuard<'a, T: ?Sized> {
    subbuffer: &'a Subbuffer<T>,
    data: &'a mut T,
}

impl<T: ?Sized> Drop for BufferWriteGuard<'_, T> {
    fn drop(&mut self) {
        let range = self.subbuffer.offset..self.subbuffer.offset + self.subbuffer.size;
        let allocation = match self.subbuffer.buffer().memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        if allocation.atom_size().is_some() {
            unsafe { allocation.flush_range(range.clone()).unwrap() };
        }

        let mut state = self.subbuffer.buffer().state();
        unsafe { state.cpu_write_unlock(range) };
    }
}

impl<T: ?Sized> Deref for BufferWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<T: ?Sized> DerefMut for BufferWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
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
