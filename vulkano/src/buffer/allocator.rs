// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Efficiently suballocates buffers into smaller subbuffers.

use super::{
    sys::{Buffer, BufferCreateInfo, RawBuffer},
    BufferAccess, BufferAccessObject, BufferContents, BufferError, BufferInner, BufferUsage,
    TypedBufferAccess,
};
use crate::{
    buffer::sys::BufferMemory,
    device::{Device, DeviceOwned},
    memory::{
        allocator::{
            align_up, AllocationCreateInfo, AllocationCreationError, AllocationType,
            MemoryAllocatePreference, MemoryAllocator, MemoryUsage, StandardMemoryAllocator,
        },
        DedicatedAllocation,
    },
    DeviceSize,
};
use crossbeam_queue::ArrayQueue;
use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    mem::{align_of, size_of, ManuallyDrop},
    num::NonZeroU64,
    ptr,
    sync::Arc,
};

const MAX_ARENAS: usize = 32;

// TODO: Add `CpuSubbuffer::read` to read the content of a subbuffer.
//       But that's hard to do because we must prevent `increase_gpu_lock` from working while a
//       a buffer is locked.

/// Efficiently suballocates buffers into smaller subbuffers.
///
/// This allocator is especially suitable when you want to upload or download some data regularly
/// (for example, at each frame for a video game).
///
/// # Algorithm
///
/// The allocator keeps a pool of *arenas*. An arena is simply a buffer in which *arena allocation*
/// takes place, also known as *bump allocation* or *linear allocation*. Every time you allocate,
/// one of these arenas is suballocated. If there is no arena that is currently available, one will
/// be allocated. After all subbuffers allocated from an arena are dropped, the arena is
/// automatically returned to the arena pool. If you try to allocate a subbuffer larger than the
/// current size of an arena, the arenas are automatically resized.
///
/// No memory is allocated when the allocator is created, be it on the Vulkan or Rust side. That
/// only happens once you allocate a subbuffer.
///
/// # Usage
///
/// Ideally, one arena should be able to fit all data you need to update per frame, so that each
/// arena is submitted and freed once per frame. This way, the arena pool would also contain as
/// many arenas as there are frames in flight on the thread. Otherwise, if your arenas are not able
/// to fit everything each frame, what will likely happen is that each subbuffer will be
/// allocated from an individual arena. This can impact efficiency both in terms of memory usage
/// (because each arena has the same size, even if some of the subbuffers are way smaller) as well
/// as performance, because the data could end up more physically separated in memory, which means
/// the GPU would need to hop from place to place a lot more during a frame.
///
/// Ideally the result is something roughly like this:
///
/// ```plain
/// +---------------------------------------------------------------------------------------------+
/// |                                        Memory Block                                         |
/// |-----+------+-----------------------+---------+-----------------------+------+---------+-----|
/// |     |      |     Frame 1 Arena     |         |     Frame 2 Arena     |      |         |     |
/// | ••• | Tex. |-------+-------+-------| Attach. |-------+-------+-------| Tex. | Attach. | ••• |
/// |     |      | Vert. | Indx. | Unif. |         | Vert. | Indx. | Unif. |      |         |     |
/// +-----+------+-------+-------+-------+---------+-------+-------+-------+------+---------+-----+
/// ```
///
/// # Examples
///
/// ```
/// use vulkano::buffer::allocator::CpuBufferAllocator;
/// use vulkano::command_buffer::{
///     AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
/// };
/// use vulkano::sync::GpuFuture;
/// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// # let command_buffer_allocator: vulkano::command_buffer::allocator::StandardCommandBufferAllocator = return;
///
/// // Create the buffer allocator.
/// let buffer_allocator = CpuBufferAllocator::new(memory_allocator.clone(), Default::default());
///
/// for n in 0..25u32 {
///     // Each loop allocates a new subbuffer and stores `data` in it.
///     let data: [f32; 4] = [1.0, 0.5, n as f32 / 24.0, 0.0];
///     let subbuffer = buffer_allocator.from_data(data).unwrap();
///
///     // You can then use `subbuffer` as if it was an entirely separate buffer.
///     AutoCommandBufferBuilder::primary(
///         &command_buffer_allocator,
///         queue.queue_family_index(),
///         CommandBufferUsage::OneTimeSubmit,
///     )
///     .unwrap()
///     // For the sake of the example we just call `update_buffer` on the buffer, even though
///     // it is pointless to do that.
///     .update_buffer(&[0.2, 0.3, 0.4, 0.5], subbuffer.clone(), 0)
///     .unwrap()
///     .build().unwrap()
///     .execute(queue.clone())
///     .unwrap()
///     .then_signal_fence_and_flush()
///     .unwrap();
/// }
/// ```
#[derive(Debug)]
pub struct CpuBufferAllocator<A = Arc<StandardMemoryAllocator>> {
    state: UnsafeCell<CpuBufferAllocatorState<A>>,
}

impl<A> CpuBufferAllocator<A>
where
    A: MemoryAllocator,
{
    /// Creates a new `CpuBufferAllocator`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.memory_usage` is [`MemoryUsage::GpuOnly`].
    pub fn new(memory_allocator: A, create_info: CpuBufferAllocatorCreateInfo) -> Self {
        let CpuBufferAllocatorCreateInfo {
            arena_size,
            buffer_usage,
            memory_usage,
            _ne: _,
        } = create_info;

        assert!(memory_usage != MemoryUsage::GpuOnly);

        let properties = memory_allocator.device().physical_device().properties();
        let buffer_alignment = [
            buffer_usage
                .contains(BufferUsage::UNIFORM_BUFFER)
                .then_some(properties.min_uniform_buffer_offset_alignment),
            buffer_usage
                .contains(BufferUsage::STORAGE_BUFFER)
                .then_some(properties.min_storage_buffer_offset_alignment),
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(1);

        CpuBufferAllocator {
            state: UnsafeCell::new(CpuBufferAllocatorState {
                memory_allocator,
                buffer_usage,
                memory_usage,
                buffer_alignment,
                arena_size,
                arena: None,
                free_start: 0,
                reserve: None,
            }),
        }
    }

    /// Returns the current size of the arenas.
    pub fn arena_size(&self) -> DeviceSize {
        unsafe { &*self.state.get() }.arena_size
    }

    /// Sets the arena size to the provided `size`.
    ///
    /// The next time you allocate a subbuffer, a new arena will be allocated with the new size,
    /// and all subsequently allocated arenas will also share the new size.
    pub fn set_arena_size(&self, size: DeviceSize) {
        let state = unsafe { &mut *self.state.get() };
        state.arena_size = size;
        state.arena = None;
        state.reserve = None;
    }

    /// Ensures that the size of the current arena is at least `size`.
    ///
    /// If `size` is greater than the current arena size, then a new arena will be allocated with
    /// the new size, and all subsequently allocated arenas will also share the new size. Otherwise
    /// this has no effect.
    pub fn reserve(&self, size: DeviceSize) -> Result<(), AllocationCreationError> {
        if size > self.arena_size() {
            let state = unsafe { &mut *self.state.get() };
            state.arena_size = size;
            state.reserve = None;
            state.arena = Some(state.next_arena()?);
        }

        Ok(())
    }

    /// Allocates a subbuffer and writes `data` in it.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `T` has an alignment greater than `64`.
    pub fn from_data<T>(&self, data: T) -> Result<Arc<CpuSubbuffer<T>>, AllocationCreationError>
    where
        T: BufferContents,
    {
        assert!(size_of::<T>() > 0);
        assert!(align_of::<T>() <= 64);

        let state = unsafe { &mut *self.state.get() };

        let size = size_of::<T>() as DeviceSize;
        let offset = state.allocate(size, align_of::<T>() as DeviceSize)?;
        let arena = state.arena.as_ref().unwrap().clone();
        let allocation = match arena.inner.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        unsafe {
            let bytes = allocation.write(offset..offset + size).unwrap();
            let mapping = T::from_bytes_mut(bytes).unwrap();

            ptr::write(mapping, data);

            if let Some(atom_size) = allocation.atom_size() {
                let size = align_up(size, atom_size.get());
                let end = DeviceSize::min(offset + size, allocation.size());
                allocation.flush_range(offset..end).unwrap();
            }
        }

        Ok(Arc::new(CpuSubbuffer {
            id: CpuSubbuffer::<T>::next_id(),
            offset,
            size,
            arena,
            _marker: PhantomData,
        }))
    }

    /// Allocates a subbuffer and writes all elements of `iter` in it.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `T` has an alignment greater than `64`.
    pub fn from_iter<T, I>(
        &self,
        iter: I,
    ) -> Result<Arc<CpuSubbuffer<[T]>>, AllocationCreationError>
    where
        [T]: BufferContents,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        assert!(size_of::<T>() > 0);
        assert!(align_of::<T>() <= 64);

        let iter = iter.into_iter();
        let state = unsafe { &mut *self.state.get() };

        let size = (size_of::<T>() * iter.len()) as DeviceSize;
        let offset = state.allocate(size, align_of::<T>() as DeviceSize)?;
        let arena = state.arena.as_ref().unwrap().clone();
        let allocation = match arena.inner.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        unsafe {
            let bytes = allocation.write(offset..offset + size).unwrap();
            let mapping = <[T]>::from_bytes_mut(bytes).unwrap();

            for (o, i) in mapping.iter_mut().zip(iter) {
                ptr::write(o, i);
            }

            if let Some(atom_size) = allocation.atom_size() {
                let size = align_up(size, atom_size.get());
                let end = DeviceSize::min(offset + size, allocation.size());
                allocation.flush_range(offset..end).unwrap();
            }
        }

        Ok(Arc::new(CpuSubbuffer {
            id: CpuSubbuffer::<T>::next_id(),
            offset,
            size,
            arena,
            _marker: PhantomData,
        }))
    }
}

#[derive(Debug)]
struct CpuBufferAllocatorState<A> {
    memory_allocator: A,
    buffer_usage: BufferUsage,
    memory_usage: MemoryUsage,
    // The alignment required for the subbuffers.
    buffer_alignment: DeviceSize,
    // The current size of the arenas.
    arena_size: DeviceSize,
    // Contains the buffer that is currently being suballocated.
    arena: Option<Arc<Arena>>,
    // Offset pointing to the start of free memory within the arena.
    free_start: DeviceSize,
    // When an `Arena` is dropped, it returns itself here for reuse.
    reserve: Option<Arc<ArrayQueue<Arc<Buffer>>>>,
}

impl<A> CpuBufferAllocatorState<A>
where
    A: MemoryAllocator,
{
    fn allocate(
        &mut self,
        size: DeviceSize,
        alignment: DeviceSize,
    ) -> Result<DeviceSize, AllocationCreationError> {
        let alignment = DeviceSize::max(alignment, self.buffer_alignment);

        loop {
            if self.arena.is_none() {
                // If the requested size is larger than the arenas, we need to resize them.
                if self.arena_size < size {
                    self.arena_size = size * 2;
                    // We need to drop our reference to the old pool to make sure the arenas are
                    // dropped once no longer in use, and replace it with a new pool that will not
                    // be polluted with the outdates arenas.
                    self.reserve = None;
                }
                self.arena = Some(self.next_arena()?);
                self.free_start = 0;
            }

            let arena = self.arena.as_ref().unwrap();
            let allocation = match arena.inner.memory() {
                BufferMemory::Normal(a) => a,
                BufferMemory::Sparse => unreachable!(),
            };
            let arena_offset = allocation.offset();
            let atom_size = allocation.atom_size().map(NonZeroU64::get).unwrap_or(1);

            let alignment = DeviceSize::max(alignment, atom_size);
            let offset = align_up(arena_offset + self.free_start, alignment);

            if offset + size <= arena_offset + self.arena_size {
                let offset = offset - arena_offset;
                self.free_start = offset + size;

                return Ok(offset);
            }

            // We reached the end of the arena, grab the next one.
            self.arena = None;
        }
    }

    fn next_arena(&mut self) -> Result<Arc<Arena>, AllocationCreationError> {
        if self.reserve.is_none() {
            self.reserve = Some(Arc::new(ArrayQueue::new(MAX_ARENAS)));
        }
        let reserve = self.reserve.as_ref().unwrap();

        reserve
            .pop()
            .map(Ok)
            .unwrap_or_else(|| self.create_arena())
            .map(|inner| {
                Arc::new(Arena {
                    inner: ManuallyDrop::new(inner),
                    reserve: reserve.clone(),
                })
            })
    }

    fn create_arena(&self) -> Result<Arc<Buffer>, AllocationCreationError> {
        let raw_buffer = RawBuffer::new(
            self.memory_allocator.device().clone(),
            BufferCreateInfo {
                size: self.arena_size,
                usage: self.buffer_usage,
                ..Default::default()
            },
        )
        .map_err(|err| match err {
            BufferError::AllocError(err) => err,
            // We don't use sparse-binding, therefore the other errors can't happen.
            _ => unreachable!(),
        })?;
        let mut requirements = *raw_buffer.memory_requirements();
        requirements.alignment = DeviceSize::max(requirements.alignment, self.buffer_alignment);
        let create_info = AllocationCreateInfo {
            requirements,
            allocation_type: AllocationType::Linear,
            usage: self.memory_usage,
            allocate_preference: MemoryAllocatePreference::Unknown,
            dedicated_allocation: Some(DedicatedAllocation::Buffer(&raw_buffer)),
            ..Default::default()
        };

        match unsafe { self.memory_allocator.allocate_unchecked(create_info) } {
            Ok(mut alloc) => {
                debug_assert!(alloc.offset() % requirements.alignment == 0);
                debug_assert!(alloc.size() == requirements.size);
                alloc.shrink(self.arena_size);
                let inner = Arc::new(
                    unsafe { raw_buffer.bind_memory_unchecked(alloc) }
                        .map_err(|(err, _, _)| err)?,
                );

                Ok(inner)
            }
            Err(err) => Err(err),
        }
    }
}

#[derive(Debug)]
struct Arena {
    inner: ManuallyDrop<Arc<Buffer>>,
    // Where we return the arena in our `Drop` impl.
    reserve: Arc<ArrayQueue<Arc<Buffer>>>,
}

impl Drop for Arena {
    fn drop(&mut self) {
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };
        let _ = self.reserve.push(inner);
    }
}

/// Parameters to create a new [`CpuBufferAllocator`].
pub struct CpuBufferAllocatorCreateInfo {
    /// Initial size of an arena in bytes.
    ///
    /// Ideally this should fit all the data you need to update per frame. So for example, if you
    /// need to allocate buffers of size 1K, 2K and 5K each frame, then this should be 8K. If your
    /// data is dynamically-sized then try to make an educated guess or simply leave the default.
    ///
    /// The default value is `0`.
    pub arena_size: DeviceSize,

    /// The buffer usage that all allocated buffers should have.
    ///
    /// The default value is [`BufferUsage::TRANSFER_SRC`].
    pub buffer_usage: BufferUsage,

    /// The memory usage that all buffers should be allocated with.
    ///
    /// Must not be [`MemoryUsage::GpuOnly`].
    ///
    /// The default value is [`MemoryUsage::Upload`].
    pub memory_usage: MemoryUsage,

    pub _ne: crate::NonExhaustive,
}

impl Default for CpuBufferAllocatorCreateInfo {
    #[inline]
    fn default() -> Self {
        CpuBufferAllocatorCreateInfo {
            arena_size: 0,
            buffer_usage: BufferUsage::TRANSFER_SRC,
            memory_usage: MemoryUsage::Upload,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A subbuffer allocated using a [`CpuBufferAllocator`].
#[derive(Debug)]
pub struct CpuSubbuffer<T: ?Sized> {
    id: NonZeroU64,
    // Offset within the arena.
    offset: DeviceSize,
    // Size of the subbuffer.
    size: DeviceSize,
    // We need to keep a reference to the arena so it won't be reset.
    arena: Arc<Arena>,
    _marker: PhantomData<Box<T>>,
}

unsafe impl<T> BufferAccess for CpuSubbuffer<T>
where
    T: BufferContents + ?Sized,
{
    fn inner(&self) -> BufferInner<'_> {
        BufferInner {
            buffer: &self.arena.inner,
            offset: self.offset,
        }
    }

    fn size(&self) -> DeviceSize {
        self.size
    }
}

impl<T> BufferAccessObject for Arc<CpuSubbuffer<T>>
where
    T: BufferContents + ?Sized,
{
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess> {
        self.clone()
    }
}

unsafe impl<T> TypedBufferAccess for CpuSubbuffer<T>
where
    T: BufferContents + ?Sized,
{
    type Content = T;
}

unsafe impl<T> DeviceOwned for CpuSubbuffer<T>
where
    T: ?Sized,
{
    fn device(&self) -> &Arc<Device> {
        self.arena.inner.device()
    }
}

crate::impl_id_counter!(CpuSubbuffer<T>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        let buffer_allocator = CpuBufferAllocator::new(memory_allocator, Default::default());
        assert_eq!(buffer_allocator.arena_size(), 0);

        buffer_allocator.reserve(83).unwrap();
        assert_eq!(buffer_allocator.arena_size(), 83);
    }

    #[test]
    fn capacity_increase() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        let buffer_allocator = CpuBufferAllocator::new(memory_allocator, Default::default());
        assert_eq!(buffer_allocator.arena_size(), 0);

        buffer_allocator.from_data(12u32).unwrap();
        assert_eq!(buffer_allocator.arena_size(), 8);
    }
}
