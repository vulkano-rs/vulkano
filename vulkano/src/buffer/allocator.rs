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
    sys::BufferCreateInfo, Buffer, BufferAllocateError, BufferContents, BufferMemory, BufferUsage,
    Subbuffer,
};
use crate::{
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    memory::{
        allocator::{
            align_up, AllocationCreateInfo, DeviceLayout, HostAccessType, MemoryAllocator,
            MemoryAllocatorError, MemoryLocationPreference, StandardMemoryAllocator,
        },
        DeviceAlignment,
    },
    DeviceSize, NonZeroDeviceSize, Validated,
};
use crossbeam_queue::ArrayQueue;
use std::{
    cell::UnsafeCell,
    cmp,
    hash::{Hash, Hasher},
    mem::ManuallyDrop,
    sync::Arc,
};

const MAX_ARENAS: usize = 32;

/// Efficiently suballocates buffers into smaller subbuffers.
///
/// This allocator is especially suitable when you want to upload or download some data regularly
/// (for example, at each frame for a video game).
///
/// # Algorithm
///
/// The allocator keeps a pool of *arenas*. An arena is simply a buffer in which *arena allocation*
/// takes place, also known as *bump allocation* or *linear allocation*. Every time you allocate,
/// one of these arenas is suballocated. An arena is suballocated until it runs out of space, at
/// which point a free one is taken from the pool. If there is no arena that is currently
/// available, one will be allocated. After all subbuffers allocated from an arena are dropped, the
/// arena is automatically returned to the arena pool for reuse. If you try to allocate a subbuffer
/// larger than the current size of an arena, the arenas are automatically resized.
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
/// Download or device-only usage is much the same. Try to make the arenas fit all the data you
/// need to store at once.
///
/// # Examples
///
/// ```
/// use vulkano::buffer::allocator::SubbufferAllocator;
/// use vulkano::command_buffer::{
///     AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
/// };
/// use vulkano::sync::GpuFuture;
/// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// # let command_buffer_allocator: vulkano::command_buffer::allocator::StandardCommandBufferAllocator = return;
///
/// // Create the buffer allocator.
/// let buffer_allocator = SubbufferAllocator::new(memory_allocator.clone(), Default::default());
///
/// for n in 0..25u32 {
///     // Each loop allocates a new subbuffer and stores `data` in it.
///     let data: [f32; 4] = [1.0, 0.5, n as f32 / 24.0, 0.0];
///     let subbuffer = buffer_allocator.allocate_sized().unwrap();
///     *subbuffer.write().unwrap() = data;
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
///     .update_buffer(subbuffer.clone(), &[0.2, 0.3, 0.4, 0.5])
///     .unwrap()
///     .build().unwrap()
///     .execute(queue.clone())
///     .unwrap()
///     .then_signal_fence_and_flush()
///     .unwrap();
/// }
/// ```
#[derive(Debug)]
pub struct SubbufferAllocator<A = Arc<StandardMemoryAllocator>> {
    state: UnsafeCell<SubbufferAllocatorState<A>>,
}

impl<A> SubbufferAllocator<A>
where
    A: MemoryAllocator,
{
    /// Creates a new `SubbufferAllocator`.
    pub fn new(memory_allocator: A, create_info: SubbufferAllocatorCreateInfo) -> Self {
        let SubbufferAllocatorCreateInfo {
            arena_size,
            buffer_usage,
            location_preference,
            host_access,
            _ne: _,
        } = create_info;

        let properties = memory_allocator.device().physical_device().properties();
        let buffer_alignment = [
            buffer_usage
                .intersects(BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER)
                .then_some(properties.min_texel_buffer_offset_alignment),
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
        .unwrap_or(DeviceAlignment::MIN);

        SubbufferAllocator {
            state: UnsafeCell::new(SubbufferAllocatorState {
                memory_allocator,
                buffer_usage,
                location_preference,
                host_access,
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
    pub fn reserve(&self, size: DeviceSize) -> Result<(), MemoryAllocatorError> {
        if size > self.arena_size() {
            let state = unsafe { &mut *self.state.get() };
            state.arena_size = size;
            state.reserve = None;
            state.arena = Some(state.next_arena()?);
        }

        Ok(())
    }

    /// Allocates a subbuffer for sized data.
    pub fn allocate_sized<T>(&self) -> Result<Subbuffer<T>, MemoryAllocatorError>
    where
        T: BufferContents,
    {
        let layout = T::LAYOUT.unwrap_sized();

        unsafe { &mut *self.state.get() }
            .allocate(layout)
            .map(|subbuffer| unsafe { subbuffer.reinterpret_unchecked() })
    }

    /// Allocates a subbuffer for a slice.
    ///
    /// # Panics
    ///
    /// - Panics if `len` is zero.
    pub fn allocate_slice<T>(&self, len: DeviceSize) -> Result<Subbuffer<[T]>, MemoryAllocatorError>
    where
        T: BufferContents,
    {
        self.allocate_unsized(len)
    }

    /// Allocates a subbuffer for unsized data.
    ///
    /// # Panics
    ///
    /// - Panics if `len` is zero.
    pub fn allocate_unsized<T>(&self, len: DeviceSize) -> Result<Subbuffer<T>, MemoryAllocatorError>
    where
        T: BufferContents + ?Sized,
    {
        let len = NonZeroDeviceSize::new(len).expect("empty slices are not valid buffer contents");
        let layout = T::LAYOUT.layout_for_len(len).unwrap();

        unsafe { &mut *self.state.get() }
            .allocate(layout)
            .map(|subbuffer| unsafe { subbuffer.reinterpret_unchecked() })
    }

    /// Allocates a subbuffer with the given `layout`.
    ///
    /// # Panics
    ///
    /// - Panics if `layout.alignment()` exceeds `64`.
    pub fn allocate(&self, layout: DeviceLayout) -> Result<Subbuffer<[u8]>, MemoryAllocatorError> {
        assert!(layout.alignment().as_devicesize() <= 64);

        unsafe { &mut *self.state.get() }.allocate(layout)
    }
}

unsafe impl<A> DeviceOwned for SubbufferAllocator<A>
where
    A: MemoryAllocator,
{
    fn device(&self) -> &Arc<Device> {
        unsafe { &*self.state.get() }.memory_allocator.device()
    }
}

#[derive(Debug)]
struct SubbufferAllocatorState<A> {
    memory_allocator: A,
    buffer_usage: BufferUsage,
    location_preference: MemoryLocationPreference,
    host_access: HostAccessType,
    // The alignment required for the subbuffers.
    buffer_alignment: DeviceAlignment,
    // The current size of the arenas.
    arena_size: DeviceSize,
    // Contains the buffer that is currently being suballocated.
    arena: Option<Arc<Arena>>,
    // Offset pointing to the start of free memory within the arena.
    free_start: DeviceSize,
    // When an `Arena` is dropped, it returns itself here for reuse.
    reserve: Option<Arc<ArrayQueue<Arc<Buffer>>>>,
}

impl<A> SubbufferAllocatorState<A>
where
    A: MemoryAllocator,
{
    fn allocate(&mut self, layout: DeviceLayout) -> Result<Subbuffer<[u8]>, MemoryAllocatorError> {
        let size = layout.size();
        let alignment = cmp::max(layout.alignment(), self.buffer_alignment);

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
            let allocation = match arena.buffer.memory() {
                BufferMemory::Normal(a) => a,
                BufferMemory::Sparse => unreachable!(),
            };
            let arena_offset = allocation.offset();
            let atom_size = allocation.atom_size().unwrap_or(DeviceAlignment::MIN);

            let alignment = cmp::max(alignment, atom_size);
            let offset = align_up(arena_offset + self.free_start, alignment);

            if offset + size <= arena_offset + self.arena_size {
                let offset = offset - arena_offset;
                self.free_start = offset + size;

                return Ok(Subbuffer::from_arena(arena.clone(), offset, layout.size()));
            }

            // We reached the end of the arena, grab the next one.
            self.arena = None;
        }
    }

    fn next_arena(&mut self) -> Result<Arc<Arena>, MemoryAllocatorError> {
        if self.reserve.is_none() {
            self.reserve = Some(Arc::new(ArrayQueue::new(MAX_ARENAS)));
        }
        let reserve = self.reserve.as_ref().unwrap();

        reserve
            .pop()
            .map(Ok)
            .unwrap_or_else(|| self.create_arena())
            .map(|buffer| {
                Arc::new(Arena {
                    buffer: ManuallyDrop::new(DeviceOwnedDebugWrapper(buffer)),
                    reserve: reserve.clone(),
                })
            })
    }

    fn create_arena(&self) -> Result<Arc<Buffer>, MemoryAllocatorError> {
        Buffer::new(
            &self.memory_allocator,
            BufferCreateInfo {
                usage: self.buffer_usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                location_preference: self.location_preference,
                host_access: self.host_access,
                ..Default::default()
            },
            DeviceLayout::from_size_alignment(self.arena_size, 1).unwrap(),
        )
        .map_err(|err| match err {
            Validated::Error(BufferAllocateError::AllocateMemory(err)) => err,
            // We don't use sparse-binding, concurrent sharing or external memory, therefore the
            // other errors can't happen.
            _ => unreachable!(),
        })
    }
}

#[derive(Debug)]
pub(super) struct Arena {
    buffer: ManuallyDrop<DeviceOwnedDebugWrapper<Arc<Buffer>>>,
    // Where we return the arena in our `Drop` impl.
    reserve: Arc<ArrayQueue<Arc<Buffer>>>,
}

impl Arena {
    pub(super) fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) }.0;
        let _ = self.reserve.push(buffer);
    }
}

impl PartialEq for Arena {
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer
    }
}

impl Eq for Arena {}

impl Hash for Arena {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.buffer.hash(state);
    }
}

/// Parameters to create a new [`SubbufferAllocator`].
pub struct SubbufferAllocatorCreateInfo {
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

    /// The preferred memory location the buffers should be allocated with.
    ///
    /// The default value is [`MemoryLocationPreference::Host`].
    pub location_preference: MemoryLocationPreference,

    /// The way the buffers are going to be accessed from the host.
    ///
    /// The default value is [`HostAccessType::SequentialWrite`].
    pub host_access: HostAccessType,

    pub _ne: crate::NonExhaustive,
}

impl Default for SubbufferAllocatorCreateInfo {
    #[inline]
    fn default() -> Self {
        SubbufferAllocatorCreateInfo {
            arena_size: 0,
            buffer_usage: BufferUsage::TRANSFER_SRC,
            location_preference: MemoryLocationPreference::Host,
            host_access: HostAccessType::SequentialWrite,
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        let buffer_allocator = SubbufferAllocator::new(memory_allocator, Default::default());
        assert_eq!(buffer_allocator.arena_size(), 0);

        buffer_allocator.reserve(83).unwrap();
        assert_eq!(buffer_allocator.arena_size(), 83);
    }

    #[test]
    fn capacity_increase() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        let buffer_allocator = SubbufferAllocator::new(memory_allocator, Default::default());
        assert_eq!(buffer_allocator.arena_size(), 0);

        buffer_allocator.allocate_sized::<u32>().unwrap();
        assert_eq!(buffer_allocator.arena_size(), 8);
    }
}
