// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A subpart of a buffer.

use super::{allocator::Arena, Buffer, BufferError, BufferMemory};
use crate::{
    device::{Device, DeviceOwned},
    macros::try_opt,
    memory::{
        self,
        allocator::{align_down, align_up, DeviceLayout},
        is_aligned, DeviceAlignment,
    },
    DeviceSize, NonZeroDeviceSize,
};
use bytemuck::{AnyBitPattern, PodCastError};
use std::{
    alloc::Layout,
    cmp,
    error::Error,
    ffi::c_void,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::{self, align_of, size_of},
    ops::{Deref, DerefMut, Range, RangeBounds},
    ptr::{self, NonNull},
    sync::Arc,
};

pub use vulkano_macros::BufferContents;

/// A subpart of a buffer.
///
/// This type doesn't correspond to any Vulkan object, it exists for API convenience. Most Vulkan
/// functions that work with buffers take the buffer as argument as well as an offset and size
/// within the buffer, which we can represent with a single subbuffer instead.
///
/// `Subbuffer` also has a type parameter, which is a hint for how the data is going to be
/// interpreted by the host or device (or both). This is useful so that we can allocate
/// (sub)buffers that are correctly aligned and have the correct size for their content, and for
/// type-safety. For example, when reading/writing a subbuffer from the host, you can use
/// [`Subbuffer::read`]/[`Subbuffer::write`] without worrying about the alignment and size being
/// correct and about converting your data from/to raw bytes.
///
/// There are two ways to get a `Subbuffer`:
///
/// - By using the functions on [`Buffer`], which create a new buffer and memory allocation each
///   time, and give you a `Subbuffer` that has an entire `Buffer` dedicated to it.
/// - By using the [`SubbufferAllocator`], which creates `Subbuffer`s by suballocating existing
///   `Buffer`s such that the `Buffer`s can keep being reused.
///
/// Alternatively, you can also create a `Buffer` manually and convert it to a `Subbuffer<[u8]>`.
///
/// [`SubbufferAllocator`]: super::allocator::SubbufferAllocator
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

    /// Returns the range the subbuffer occupies, in bytes, relative to the buffer.
    pub(crate) fn range(&self) -> Range<DeviceSize> {
        self.offset..self.offset + self.size
    }

    /// Returns the buffer that this subbuffer is a part of.
    pub fn buffer(&self) -> &Arc<Buffer> {
        match &self.parent {
            SubbufferParent::Arena(arena) => arena.buffer(),
            SubbufferParent::Buffer(buffer) => buffer,
        }
    }

    /// Returns the mapped pointer to the start of the subbuffer if the memory is host-visible,
    /// otherwise returns [`None`].
    pub fn mapped_ptr(&self) -> Option<NonNull<c_void>> {
        match self.buffer().memory() {
            BufferMemory::Normal(a) => a.mapped_ptr().map(|ptr| {
                // SAFETY: The original address came from the Vulkan implementation, and allocation
                // sizes are guaranteed to not exceed `isize::MAX` when there's a mapped pointer,
                // so the offset better be in range.
                unsafe { NonNull::new_unchecked(ptr.as_ptr().add(self.offset as usize)) }
            }),
            BufferMemory::Sparse => unreachable!(),
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

    /// Casts the subbuffer to a slice of raw bytes.
    pub fn into_bytes(self) -> Subbuffer<[u8]> {
        unsafe { self.reinterpret_unchecked_inner() }
    }

    /// Same as [`into_bytes`], except it works with a reference to the subbuffer.
    ///
    /// [`into_bytes`]: Self::into_bytes
    pub fn as_bytes(&self) -> &Subbuffer<[u8]> {
        unsafe { self.reinterpret_ref_unchecked_inner() }
    }

    #[inline(always)]
    unsafe fn reinterpret_unchecked_inner<U: ?Sized>(self) -> Subbuffer<U> {
        // SAFETY: All `Subbuffer`s share the same layout.
        mem::transmute::<Subbuffer<T>, Subbuffer<U>>(self)
    }

    #[inline(always)]
    unsafe fn reinterpret_ref_unchecked_inner<U: ?Sized>(&self) -> &Subbuffer<U> {
        assert!(size_of::<Subbuffer<T>>() == size_of::<Subbuffer<U>>());
        assert!(align_of::<Subbuffer<T>>() == align_of::<Subbuffer<U>>());

        // SAFETY: All `Subbuffer`s share the same layout.
        mem::transmute::<&Subbuffer<T>, &Subbuffer<U>>(self)
    }
}

impl<T> Subbuffer<T>
where
    T: BufferContents + ?Sized,
{
    /// Changes the `T` generic parameter of the subbffer to the desired type without checking if
    /// the contents are correctly aligned and sized.
    ///
    /// **NEVER use this function** unless you absolutely have to, and even then, open an issue on
    /// GitHub instead. **An unaligned / incorrectly sized subbuffer is undefined behavior _both on
    /// the Rust and the Vulkan side!_**
    ///
    /// # Safety
    ///
    /// - `self.memory_offset()` must be properly aligned for `U`.
    /// - `self.size()` must be valid for `U`, which means:
    ///   - If `U` is sized, the size must match exactly.
    ///   - If `U` is unsized, then the subbuffer size minus the size of the head (sized part) of
    ///     the DST must be evenly divisible by the size of the element type.
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reinterpret_unchecked<U>(self) -> Subbuffer<U>
    where
        U: BufferContents + ?Sized,
    {
        let element_size = U::LAYOUT.element_size().unwrap_or(1);
        debug_assert!(is_aligned(self.memory_offset(), U::LAYOUT.alignment()));
        debug_assert!(self.size >= U::LAYOUT.head_size());
        debug_assert!((self.size - U::LAYOUT.head_size()) % element_size == 0);

        self.reinterpret_unchecked_inner()
    }

    /// Same as [`reinterpret_unchecked`], except it works with a reference to the subbuffer.
    ///
    /// # Safety
    ///
    /// Please read the safety docs on [`reinterpret_unchecked`] carefully.
    ///
    /// [`reinterpret_unchecked`]: Self::reinterpret_unchecked
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reinterpret_ref_unchecked<U>(&self) -> &Subbuffer<U>
    where
        U: BufferContents + ?Sized,
    {
        let element_size = U::LAYOUT.element_size().unwrap_or(1);
        debug_assert!(is_aligned(self.memory_offset(), U::LAYOUT.alignment()));
        debug_assert!(self.size >= U::LAYOUT.head_size());
        debug_assert!((self.size - U::LAYOUT.head_size()) % element_size == 0);

        self.reinterpret_ref_unchecked_inner()
    }

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
    /// If the memory backing the buffer is not [host-coherent], then this function will lock a
    /// range that is potentially larger than the subbuffer, because the range given to
    /// [`invalidate_range`] must be aligned to the [`non_coherent_atom_size`]. This means that for
    /// example if your Vulkan implementation reports an atom size of 64, and you tried to put 2
    /// subbuffers of size 32 in the same buffer, one at offset 0 and one at offset 32, while the
    /// buffer is backed by non-coherent memory, then invalidating one subbuffer would also
    /// invalidate the other subbuffer. This can lead to data races and is therefore not allowed.
    /// What you should do in that case is ensure that each subbuffer is aligned to the
    /// non-coherent atom size, so in this case one would be at offset 0 and the other at offset
    /// 64. [`SubbufferAllocator`] does this automatically.
    ///
    /// [host-coherent]: crate::memory::MemoryPropertyFlags::HOST_COHERENT
    /// [`invalidate_range`]: crate::memory::allocator::MemoryAlloc::invalidate_range
    /// [`non_coherent_atom_size`]: crate::device::Properties::non_coherent_atom_size
    /// [`write`]: Self::write
    /// [`SubbufferAllocator`]: super::allocator::SubbufferAllocator
    pub fn read(&self) -> Result<BufferReadGuard<'_, T>, BufferError> {
        let allocation = match self.buffer().memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => todo!("`Subbuffer::read` doesn't support sparse binding yet"),
        };

        let range = if let Some(atom_size) = allocation.atom_size() {
            // This works because the suballocators align allocations to the non-coherent atom size
            // when the memory is host-visible but not host-coherent.
            let start = align_down(self.offset, atom_size);
            let end = cmp::min(
                align_up(self.offset + self.size, atom_size),
                allocation.size(),
            );

            Range { start, end }
        } else {
            self.range()
        };

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

        let mapped_ptr = self.mapped_ptr().ok_or(BufferError::MemoryNotHostVisible)?;
        // SAFETY: `Subbuffer` guarantees that its contents are laid out correctly for `T`.
        let data = unsafe { &*T::from_ffi(mapped_ptr.as_ptr(), self.size as usize) };

        Ok(BufferReadGuard {
            subbuffer: self,
            data,
            range,
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
    /// If the memory backing the buffer is not [host-coherent], then this function will lock a
    /// range that is potentially larger than the subbuffer, because the range given to
    /// [`flush_range`] must be aligned to the [`non_coherent_atom_size`]. This means that for
    /// example if your Vulkan implementation reports an atom size of 64, and you tried to put 2
    /// subbuffers of size 32 in the same buffer, one at offset 0 and one at offset 32, while the
    /// buffer is backed by non-coherent memory, then flushing one subbuffer would also flush the
    /// other subbuffer. This can lead to data races and is therefore not allowed. What you should
    /// do in that case is ensure that each subbuffer is aligned to the non-coherent atom size, so
    /// in this case one would be at offset 0 and the other at offset 64. [`SubbufferAllocator`]
    /// does this automatically.
    ///
    /// [host-coherent]: crate::memory::MemoryPropertyFlags::HOST_COHERENT
    /// [`flush_range`]: crate::memory::allocator::MemoryAlloc::flush_range
    /// [`non_coherent_atom_size`]: crate::device::Properties::non_coherent_atom_size
    /// [`read`]: Self::read
    /// [`SubbufferAllocator`]: super::allocator::SubbufferAllocator
    pub fn write(&self) -> Result<BufferWriteGuard<'_, T>, BufferError> {
        let allocation = match self.buffer().memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => todo!("`Subbuffer::write` doesn't support sparse binding yet"),
        };

        let range = if let Some(atom_size) = allocation.atom_size() {
            // This works because the suballocators align allocations to the non-coherent atom size
            // when the memory is host-visible but not host-coherent.
            let start = align_down(self.offset, atom_size);
            let end = cmp::min(
                align_up(self.offset + self.size, atom_size),
                allocation.size(),
            );

            Range { start, end }
        } else {
            self.range()
        };

        let mut state = self.buffer().state();
        state.check_cpu_write(range.clone())?;
        unsafe { state.cpu_write_lock(range.clone()) };

        if allocation.atom_size().is_some() {
            unsafe { allocation.invalidate_range(range.clone()) }?;
        }

        let mapped_ptr = self.mapped_ptr().ok_or(BufferError::MemoryNotHostVisible)?;
        // SAFETY: `Subbuffer` guarantees that its contents are laid out correctly for `T`.
        let data = unsafe { &mut *T::from_ffi(mapped_ptr.as_ptr(), self.size as usize) };

        Ok(BufferWriteGuard {
            subbuffer: self,
            data,
            range,
        })
    }
}

impl<T> Subbuffer<T> {
    /// Converts the subbuffer to a slice of one element.
    pub fn into_slice(self) -> Subbuffer<[T]> {
        unsafe { self.reinterpret_unchecked_inner() }
    }

    /// Same as [`into_slice`], except it works with a reference to the subbuffer.
    ///
    /// [`into_slice`]: Self::into_slice
    pub fn as_slice(&self) -> &Subbuffer<[T]> {
        unsafe { self.reinterpret_ref_unchecked_inner() }
    }
}

impl<T> Subbuffer<T>
where
    T: BufferContents,
{
    /// Tries to cast a subbuffer of raw bytes to a `Subbuffer<T>`.
    pub fn try_from_bytes(subbuffer: Subbuffer<[u8]>) -> Result<Self, PodCastError> {
        if subbuffer.size() != size_of::<T>() as DeviceSize {
            Err(PodCastError::SizeMismatch)
        } else if !is_aligned(subbuffer.memory_offset(), DeviceAlignment::of::<T>()) {
            Err(PodCastError::TargetAlignmentGreaterAndInputNotAligned)
        } else {
            Ok(unsafe { subbuffer.reinterpret_unchecked() })
        }
    }

    /// Tries to cast the subbuffer to a different type.
    pub fn try_cast<U>(self) -> Result<Subbuffer<U>, PodCastError>
    where
        U: BufferContents,
    {
        if size_of::<U>() != size_of::<T>() {
            Err(PodCastError::SizeMismatch)
        } else if align_of::<U>() > align_of::<T>()
            && !is_aligned(self.memory_offset(), DeviceAlignment::of::<U>())
        {
            Err(PodCastError::TargetAlignmentGreaterAndInputNotAligned)
        } else {
            Ok(unsafe { self.reinterpret_unchecked() })
        }
    }
}

impl<T> Subbuffer<[T]> {
    /// Returns the number of elements in the slice.
    pub fn len(&self) -> DeviceSize {
        debug_assert!(self.size % size_of::<T>() as DeviceSize == 0);

        self.size / size_of::<T>() as DeviceSize
    }

    /// Reduces the subbuffer to just one element of the slice.
    ///
    /// # Panics
    ///
    /// - Panics if `index` is out of bounds.
    pub fn index(self, index: DeviceSize) -> Subbuffer<T> {
        assert!(index <= self.len());

        unsafe { self.index_unchecked(index) }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn index_unchecked(self, index: DeviceSize) -> Subbuffer<T> {
        Subbuffer {
            offset: self.offset + index * size_of::<T>() as DeviceSize,
            size: size_of::<T>() as DeviceSize,
            parent: self.parent,
            marker: PhantomData,
        }
    }

    /// Reduces the subbuffer to just a range of the slice.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is out of bounds.
    /// - Panics if `range` is empty.
    pub fn slice(mut self, range: impl RangeBounds<DeviceSize>) -> Subbuffer<[T]> {
        let Range { start, end } = memory::range(range, ..self.len()).unwrap();

        self.offset += start * size_of::<T>() as DeviceSize;
        self.size = (end - start) * size_of::<T>() as DeviceSize;
        assert!(self.size != 0);

        self
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn slice_unchecked(mut self, range: impl RangeBounds<DeviceSize>) -> Subbuffer<[T]> {
        let Range { start, end } = memory::range(range, ..self.len()).unwrap_unchecked();

        self.offset += start * size_of::<T>() as DeviceSize;
        self.size = (end - start) * size_of::<T>() as DeviceSize;
        debug_assert!(self.size != 0);

        self
    }

    /// Splits the subbuffer into two at an index.
    ///
    /// # Panics
    ///
    /// - Panics if `mid` is not greater than `0`.
    /// - Panics if `mid` is not less than `self.len()`.
    pub fn split_at(self, mid: DeviceSize) -> (Subbuffer<[T]>, Subbuffer<[T]>) {
        assert!(0 < mid && mid < self.len());

        unsafe { self.split_at_unchecked(mid) }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn split_at_unchecked(self, mid: DeviceSize) -> (Subbuffer<[T]>, Subbuffer<[T]>) {
        (
            self.clone().slice_unchecked(..mid),
            self.slice_unchecked(mid..),
        )
    }
}

impl Subbuffer<[u8]> {
    /// Creates a new `Subbuffer<[u8]>` spanning the whole buffer.
    #[inline]
    pub fn new(buffer: Arc<Buffer>) -> Self {
        Subbuffer {
            offset: 0,
            size: buffer.size(),
            parent: SubbufferParent::Buffer(buffer),
            marker: PhantomData,
        }
    }

    /// Casts the slice to a different element type while ensuring correct alignment for the type.
    ///
    /// The offset of the subbuffer is rounded up to the alignment of `T` and the size abjusted for
    /// the padding, then the size is rounded down to the nearest multiple of `T`'s size.
    ///
    /// # Panics
    ///
    /// - Panics if the aligned offset would be out of bounds.
    pub fn cast_aligned<T>(self) -> Subbuffer<[T]>
    where
        T: BufferContents,
    {
        let layout = DeviceLayout::from_layout(Layout::new::<T>()).unwrap();
        let aligned = self.align_to(layout);

        unsafe { aligned.reinterpret_unchecked() }
    }

    /// Aligns the subbuffer to the given `layout` by rounding the offset up to
    /// `layout.alignment()` and adjusting the size for the padding, and then rounding the size
    /// down to the nearest multiple of `layout.size()`.
    ///
    /// # Panics
    ///
    /// - Panics if the aligned offset would be out of bounds.
    /// - Panics if `layout.alignment()` exceeds `64`.
    #[inline]
    pub fn align_to(mut self, layout: DeviceLayout) -> Subbuffer<[u8]> {
        assert!(layout.alignment().as_devicesize() <= 64);

        let offset = self.memory_offset();
        let padding_front = align_up(offset, layout.alignment()) - offset;

        self.offset += padding_front;
        self.size = self.size.checked_sub(padding_front).unwrap();
        self.size -= self.size % layout.size();

        self
    }
}

impl<T> Subbuffer<[T]>
where
    T: BufferContents,
{
    /// Tries to cast the slice to a different element type.
    pub fn try_cast_slice<U>(self) -> Result<Subbuffer<[U]>, PodCastError>
    where
        U: BufferContents,
    {
        if size_of::<U>() != size_of::<T>() && self.size() % size_of::<U>() as DeviceSize != 0 {
            Err(PodCastError::OutputSliceWouldHaveSlop)
        } else if align_of::<U>() > align_of::<T>()
            && !is_aligned(self.memory_offset(), DeviceAlignment::of::<U>())
        {
            Err(PodCastError::TargetAlignmentGreaterAndInputNotAligned)
        } else {
            Ok(unsafe { self.reinterpret_unchecked() })
        }
    }
}

impl From<Arc<Buffer>> for Subbuffer<[u8]> {
    #[inline]
    fn from(buffer: Arc<Buffer>) -> Self {
        Self::new(buffer)
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
    range: Range<DeviceSize>,
}

impl<T: ?Sized> Drop for BufferReadGuard<'_, T> {
    fn drop(&mut self) {
        let mut state = self.subbuffer.buffer().state();
        unsafe { state.cpu_read_unlock(self.range.clone()) };
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
    range: Range<DeviceSize>,
}

impl<T: ?Sized> Drop for BufferWriteGuard<'_, T> {
    fn drop(&mut self) {
        let allocation = match self.subbuffer.buffer().memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        if allocation.atom_size().is_some() {
            unsafe { allocation.flush_range(self.range.clone()).unwrap() };
        }

        let mut state = self.subbuffer.buffer().state();
        unsafe { state.cpu_write_unlock(self.range.clone()) };
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

/// Trait for types of data that can be put in a buffer.
///
/// This trait is not intended to be implemented manually (ever) and attempting so will make you
/// one sad individual very quickly. Rather you should use [the derive macro]. Note also that there
/// are blanket implementations of this trait: you don't need to implement it if the type in
/// question already implements bytemuck's [`AnyBitPattern`]. Most if not all linear algebra crates
/// have a feature flag that you can enable for bytemuck support. The trait is also already
/// implemented for all slices where the element type implements `BufferContents`.
///
/// # Examples
///
/// Deriving the trait for sized types:
///
/// ```
/// # use vulkano::buffer::BufferContents;
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct MyData {
///     x: f32,
///     y: f32,
///     array: [i32; 12],
/// }
/// ```
///
/// Deriving the trait for unsized types works the same:
///
/// ```
/// # use vulkano::buffer::BufferContents;
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct MyData {
///     x: f32,
///     y: f32,
///     slice: [i32],
/// }
/// ```
///
/// This even works if the last field is a user-defined DST too:
///
/// ```
/// # use vulkano::buffer::BufferContents;
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct MyData {
///     x: f32,
///     y: f32,
///     other: OtherData,
/// }
///
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct OtherData {
///     slice: [i32],
/// }
/// ```
///
/// You can also use generics if you please:
///
/// ```
/// # use vulkano::buffer::BufferContents;
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct MyData<T, U> {
///     x: T,
///     y: T,
///     slice: [U],
/// }
/// ```
///
/// This even works with dependently-sized types:
///
/// ```
/// # use vulkano::buffer::BufferContents;
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct MyData<T>
/// where
///     T: ?Sized,
/// {
///     x: f32,
///     y: f32,
///     z: T,
/// }
/// ```
///
/// [the derive macro]: vulkano_macros::BufferContents
//
// If you absolutely *must* implement this trait by hand, here are the safety requirements (but
// please open an issue on GitHub instead):
//
// - The type must be a struct and all fields must implement `BufferContents`.
// - `LAYOUT` must be the correct layout for the type, which also means the type must either be
//   sized or if it's unsized then its metadata must be the same as that of a slice. Implementing
//   `BufferContents` for any other kind of DST is instantaneous horrifically undefined behavior.
// - `from_ffi` must create a pointer with the same address as the `data` parameter that is passed
//   in. The pointer is expected to be aligned properly already.
// - `from_ffi` must create a pointer that is expected to be valid for reads (and potentially
//   writes) for exactly `range` bytes. The `data` and `range` are expected to be valid for the
//   `LAYOUT`.
pub unsafe trait BufferContents: Send + Sync + 'static {
    /// The layout of the contents.
    const LAYOUT: BufferContentsLayout;

    /// Creates a pointer to `Self` from a pointer to the start of the data and a range in bytes.
    ///
    /// # Safety
    ///
    /// - If `Self` is sized, then `range` must match the size exactly.
    /// - If `Self` is unsized, then the `range` minus the size of the head (sized part) of the DST
    ///   must be evenly divisible by the size of the element type.
    #[doc(hidden)]
    unsafe fn from_ffi(data: *mut c_void, range: usize) -> *mut Self;
}

unsafe impl<T> BufferContents for T
where
    T: AnyBitPattern + Send + Sync,
{
    const LAYOUT: BufferContentsLayout =
        if let Some(layout) = BufferContentsLayout::from_sized(Layout::new::<T>()) {
            layout
        } else {
            panic!("zero-sized types are not valid buffer contents");
        };

    #[inline(always)]
    unsafe fn from_ffi(data: *mut c_void, range: usize) -> *mut Self {
        debug_assert!(range == size_of::<T>());
        debug_assert!(data as usize % align_of::<T>() == 0);

        data.cast()
    }
}

unsafe impl<T> BufferContents for [T]
where
    T: BufferContents,
{
    const LAYOUT: BufferContentsLayout = BufferContentsLayout(BufferContentsLayoutInner::Unsized {
        head_layout: None,
        element_layout: T::LAYOUT.unwrap_sized(),
    });

    #[inline(always)]
    unsafe fn from_ffi(data: *mut c_void, range: usize) -> *mut Self {
        debug_assert!(range % size_of::<T>() == 0);
        debug_assert!(data as usize % align_of::<T>() == 0);
        let len = range / size_of::<T>();

        ptr::slice_from_raw_parts_mut(data.cast(), len)
    }
}

/// Describes the layout required for a type so that it can be read from/written to a buffer. This
/// is used to allocate (sub)buffers generically.
///
/// This is similar to [`DeviceLayout`] except that this exists for the sole purpose of describing
/// the layout of buffer contents specifically. Which means for example that the sizedness of the
/// type is captured, as well as the layout of the head and tail if the layout is for unsized data,
/// in order to be able to represent everything that Vulkan can stuff in a buffer.
///
/// `BufferContentsLayout` also has an additional invariant compared to `DeviceLayout`: the
/// alignment of the data must not exceed `64`. This is because that's the guaranteed alignment
/// that all `DeviceMemory` blocks must be aligned to at minimum, and hence any greater alignment
/// can't be guaranteed. Other than that, the invariant that sizes must be non-zero applies here as
/// well, for both sized data and the element type of unsized data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferContentsLayout(BufferContentsLayoutInner);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BufferContentsLayoutInner {
    Sized(DeviceLayout),
    Unsized {
        head_layout: Option<DeviceLayout>,
        element_layout: DeviceLayout,
    },
}

impl BufferContentsLayout {
    /// Returns the size of the head (sized part). If the data has no sized part, then this will
    /// return 0.
    #[inline]
    pub const fn head_size(&self) -> DeviceSize {
        match &self.0 {
            BufferContentsLayoutInner::Sized(sized) => sized.size(),
            BufferContentsLayoutInner::Unsized {
                head_layout: None, ..
            } => 0,
            BufferContentsLayoutInner::Unsized {
                head_layout: Some(head_layout),
                ..
            } => head_layout.size(),
        }
    }

    /// Returns the size of the element type if the data is unsized, or returns [`None`].
    /// Guaranteed to be non-zero.
    #[inline]
    pub const fn element_size(&self) -> Option<DeviceSize> {
        match &self.0 {
            BufferContentsLayoutInner::Sized(_) => None,
            BufferContentsLayoutInner::Unsized { element_layout, .. } => {
                Some(element_layout.size())
            }
        }
    }

    /// Returns the alignment required for the data. Guaranteed to not exceed `64`.
    #[inline]
    pub const fn alignment(&self) -> DeviceAlignment {
        match &self.0 {
            BufferContentsLayoutInner::Sized(sized) => sized.alignment(),
            BufferContentsLayoutInner::Unsized {
                head_layout: None,
                element_layout,
            } => element_layout.alignment(),
            BufferContentsLayoutInner::Unsized {
                head_layout: Some(head_layout),
                ..
            } => head_layout.alignment(),
        }
    }

    /// Returns the [`DeviceLayout`] for the data for the given `len`, or returns [`None`] on
    /// arithmetic overflow or if the total size would exceed [`DeviceLayout::MAX_SIZE`].
    #[inline]
    pub const fn layout_for_len(&self, len: NonZeroDeviceSize) -> Option<DeviceLayout> {
        match &self.0 {
            BufferContentsLayoutInner::Sized(sized) => Some(*sized),
            BufferContentsLayoutInner::Unsized {
                head_layout,
                element_layout,
            } => {
                let (tail_layout, _) = try_opt!(element_layout.repeat(len));

                if let Some(head_layout) = head_layout {
                    let (layout, _) = try_opt!(head_layout.extend(tail_layout));

                    Some(layout.pad_to_alignment())
                } else {
                    Some(tail_layout)
                }
            }
        }
    }

    /// Creates a new `BufferContentsLayout` from a sized layout. This is inteded for use by the
    /// derive macro only.
    #[doc(hidden)]
    #[inline]
    pub const fn from_sized(sized: Layout) -> Option<Self> {
        assert!(
            sized.align() <= 64,
            "types with alignments above 64 are not valid buffer contents",
        );

        if let Ok(sized) = DeviceLayout::from_layout(sized) {
            Some(Self(BufferContentsLayoutInner::Sized(sized)))
        } else {
            None
        }
    }

    /// Creates a new `BufferContentsLayout` from a head and element layout. This is inteded for
    /// use by the derive macro only.
    #[doc(hidden)]
    #[inline]
    pub const fn from_head_element_layout(
        head_layout: Layout,
        element_layout: Layout,
    ) -> Option<Self> {
        if head_layout.align() > 64 || element_layout.align() > 64 {
            panic!("types with alignments above 64 are not valid buffer contents");
        }

        // The head of a `BufferContentsLayout` can be zero-sized.
        // TODO: Replace with `Result::ok` once its constness is stabilized.
        let head_layout = if let Ok(head_layout) = DeviceLayout::from_layout(head_layout) {
            Some(head_layout)
        } else {
            None
        };

        if let Ok(element_layout) = DeviceLayout::from_layout(element_layout) {
            Some(Self(BufferContentsLayoutInner::Unsized {
                head_layout,
                element_layout,
            }))
        } else {
            None
        }
    }

    /// Extends the given `previous` [`Layout`] by `self`. This is intended for use by the derive
    /// macro only.
    #[doc(hidden)]
    #[inline]
    pub const fn extend_from_layout(self, previous: &Layout) -> Option<Self> {
        assert!(
            previous.align() <= 64,
            "types with alignments above 64 are not valid buffer contents",
        );

        match self.0 {
            BufferContentsLayoutInner::Sized(sized) => {
                let (sized, _) = try_opt!(sized.extend_from_layout(previous));

                Some(Self(BufferContentsLayoutInner::Sized(sized)))
            }
            BufferContentsLayoutInner::Unsized {
                head_layout: None,
                element_layout,
            } => {
                // The head of a `BufferContentsLayout` can be zero-sized.
                // TODO: Replace with `Result::ok` once its constness is stabilized.
                let head_layout = if let Ok(head_layout) = DeviceLayout::from_layout(*previous) {
                    Some(head_layout)
                } else {
                    None
                };

                Some(Self(BufferContentsLayoutInner::Unsized {
                    head_layout,
                    element_layout,
                }))
            }
            BufferContentsLayoutInner::Unsized {
                head_layout: Some(head_layout),
                element_layout,
            } => {
                let (head_layout, _) = try_opt!(head_layout.extend_from_layout(previous));

                Some(Self(BufferContentsLayoutInner::Unsized {
                    head_layout: Some(head_layout),
                    element_layout,
                }))
            }
        }
    }

    /// Creates a new `BufferContentsLayout` by rounding up the size of the head to the nearest
    /// multiple of its alignment if the layout is sized, or by rounding up the size of the head to
    /// the nearest multiple of the alignment of the element type and aligning the head to the
    /// alignment of the element type if there is a sized part. Doesn't do anything if there is no
    /// sized part. Returns [`None`] if the new head size would exceed [`DeviceLayout::MAX_SIZE`].
    /// This is inteded for use by the derive macro only.
    #[doc(hidden)]
    #[inline]
    pub const fn pad_to_alignment(&self) -> Option<Self> {
        match &self.0 {
            BufferContentsLayoutInner::Sized(sized) => Some(Self(
                BufferContentsLayoutInner::Sized(sized.pad_to_alignment()),
            )),
            BufferContentsLayoutInner::Unsized {
                head_layout: None,
                element_layout,
            } => Some(Self(BufferContentsLayoutInner::Unsized {
                head_layout: None,
                element_layout: *element_layout,
            })),
            BufferContentsLayoutInner::Unsized {
                head_layout: Some(head_layout),
                element_layout,
            } => {
                // We must pad the head to the alignment of the element type, *not* the alignment
                // of the head.
                //
                // Consider a head layout of `(u8, u8, u8)` and an element layout of `u32`. If we
                // padded the head to its own alignment, like is the case for sized layouts, it
                // wouldn't change the size. Yet there is padding between the head and the first
                // element of the slice.
                //
                // The reverse is true: consider a head layout of `(u16, u8)` and an element layout
                // of `u8`. If we padded the head to its own alignment, it would be too large.
                let padded_head_size =
                    head_layout.size() + head_layout.padding_needed_for(element_layout.alignment());

                // SAFETY: `BufferContentsLayout`'s invariant guarantees that the alignment of the
                // element type doesn't exceed 64, which together with the overflow invariant of
                // `DeviceLayout` means that this can't overflow.
                let padded_head_size =
                    unsafe { NonZeroDeviceSize::new_unchecked(padded_head_size) };

                // We have to align the head to the alignment of the element type, so that the
                // struct as a whole is aligned correctly when a different struct is extended with
                // this one.
                //
                // Note that this is *not* the same as aligning the head to the alignment of the
                // element type and then padding the layout to its alignment. Consider the same
                // layout from above, with a head layout of `(u16, u8)` and an element layout of
                // `u8`. If we aligned the head to the element type and then padded it to its own
                // alignment, we would get the same wrong result as above. This instead ensures the
                // head is padded to the element and aligned to it, without the alignment of the
                // head interfering.
                let alignment =
                    DeviceAlignment::max(head_layout.alignment(), element_layout.alignment());

                if let Some(head_layout) = DeviceLayout::new(padded_head_size, alignment) {
                    Some(Self(BufferContentsLayoutInner::Unsized {
                        head_layout: Some(head_layout),
                        element_layout: *element_layout,
                    }))
                } else {
                    None
                }
            }
        }
    }

    pub(super) const fn unwrap_sized(self) -> DeviceLayout {
        match self.0 {
            BufferContentsLayoutInner::Sized(sized) => sized,
            BufferContentsLayoutInner::Unsized { .. } => {
                panic!("called `BufferContentsLayout::unwrap_sized` on an unsized layout");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        buffer::{
            sys::{BufferCreateInfo, RawBuffer},
            BufferAllocateInfo, BufferUsage,
        },
        memory::{
            allocator::{
                AllocationCreateInfo, AllocationType, DeviceLayout, MemoryAllocator, MemoryUsage,
                StandardMemoryAllocator,
            },
            MemoryRequirements,
        },
    };

    #[test]
    fn derive_buffer_contents() {
        #[derive(BufferContents)]
        #[repr(C)]
        struct Test1(u32, u64, u8);

        assert_eq!(Test1::LAYOUT.head_size() as usize, size_of::<Test1>());
        assert_eq!(Test1::LAYOUT.element_size(), None);
        assert_eq!(
            Test1::LAYOUT.alignment().as_devicesize() as usize,
            align_of::<Test1>(),
        );

        #[derive(BufferContents)]
        #[repr(C)]
        struct Composite1(Test1, [f32; 10], Test1);

        assert_eq!(
            Composite1::LAYOUT.head_size() as usize,
            size_of::<Composite1>(),
        );
        assert_eq!(Composite1::LAYOUT.element_size(), None);
        assert_eq!(
            Composite1::LAYOUT.alignment().as_devicesize() as usize,
            align_of::<Composite1>(),
        );

        #[derive(BufferContents)]
        #[repr(C)]
        struct Test2(u64, u8, [u32]);

        assert_eq!(
            Test2::LAYOUT.head_size() as usize,
            size_of::<u64>() + size_of::<u32>(),
        );
        assert_eq!(
            Test2::LAYOUT.element_size().unwrap() as usize,
            size_of::<u32>(),
        );
        assert_eq!(
            Test2::LAYOUT.alignment().as_devicesize() as usize,
            align_of::<u64>(),
        );

        #[derive(BufferContents)]
        #[repr(C)]
        struct Composite2(Test1, [f32; 10], Test2);

        assert_eq!(
            Composite2::LAYOUT.head_size() as usize,
            size_of::<Test1>() + size_of::<[f32; 10]>() + size_of::<u64>() + size_of::<u32>(),
        );
        assert_eq!(
            Composite2::LAYOUT.element_size().unwrap() as usize,
            size_of::<u32>(),
        );
        assert_eq!(
            Composite2::LAYOUT.alignment().as_devicesize() as usize,
            align_of::<u64>(),
        );
    }

    #[test]
    fn split_at() {
        let (device, _) = gfx_dev_and_queue!();
        let allocator = StandardMemoryAllocator::new_default(device);

        let buffer = Buffer::new_slice::<u32>(
            &allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            6,
        )
        .unwrap();

        {
            let (left, right) = buffer.clone().split_at(2);
            assert!(left.len() == 2);
            assert!(right.len() == 4);
        }

        {
            let (left, right) = buffer.clone().split_at(5);
            assert!(left.len() == 5);
            assert!(right.len() == 1);
        }

        {
            assert_should_panic!({ buffer.clone().split_at(0) });
        }

        {
            assert_should_panic!({ buffer.split_at(6) });
        }
    }

    #[test]
    fn cast_aligned() {
        let (device, _) = gfx_dev_and_queue!();
        let allocator = StandardMemoryAllocator::new_default(device.clone());

        let raw_buffer = RawBuffer::new(
            device,
            BufferCreateInfo {
                size: 32,
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
        )
        .unwrap();

        let requirements = MemoryRequirements {
            layout: DeviceLayout::from_size_alignment(32, 1).unwrap(),
            memory_type_bits: 1,
            prefers_dedicated_allocation: false,
            requires_dedicated_allocation: false,
        };

        // Allocate some junk in the same block as the buffer.
        let _junk = allocator
            .allocate(AllocationCreateInfo {
                requirements: MemoryRequirements {
                    layout: DeviceLayout::from_size_alignment(17, 1).unwrap(),
                    ..requirements
                },
                allocation_type: AllocationType::Linear,
                usage: MemoryUsage::GpuOnly,
                ..Default::default()
            })
            .unwrap();

        let allocation = allocator
            .allocate(AllocationCreateInfo {
                requirements,
                allocation_type: AllocationType::Linear,
                usage: MemoryUsage::GpuOnly,
                ..Default::default()
            })
            .unwrap();

        let buffer = Buffer::from_raw(raw_buffer, BufferMemory::Normal(allocation));
        let buffer = Subbuffer::from(Arc::new(buffer));

        assert!(buffer.memory_offset() >= 17);

        {
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            #[repr(C, align(16))]
            struct Test([u8; 16]);

            let aligned = buffer.clone().cast_aligned::<Test>();
            assert_eq!(aligned.memory_offset() % 16, 0);
            assert_eq!(aligned.size(), 16);
        }

        {
            let aligned = buffer.clone().cast_aligned::<[u8; 16]>();
            assert_eq!(aligned.size() % 16, 0);
        }

        {
            let layout = DeviceLayout::from_size_alignment(32, 16).unwrap();
            let aligned = buffer.clone().align_to(layout);
            assert!(is_aligned(aligned.memory_offset(), layout.alignment()));
            assert_eq!(aligned.size(), 0);
        }

        {
            let layout = DeviceLayout::from_size_alignment(1, 64).unwrap();
            assert_should_panic!({ buffer.align_to(layout) });
        }
    }
}
