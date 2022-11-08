// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

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
            suballocator::align_up, AllocationCreateInfo, AllocationCreationError, AllocationType,
            MemoryAllocatePreference, MemoryAllocator, MemoryUsage, StandardMemoryAllocator,
        },
        DedicatedAllocation,
    },
    DeviceSize, VulkanError,
};
use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, MutexGuard,
    },
};

// TODO: Add `CpuBufferPoolSubbuffer::read` to read the content of a subbuffer.
//       But that's hard to do because we must prevent `increase_gpu_lock` from working while a
//       a buffer is locked.

/// Ring buffer from which "sub-buffers" can be individually allocated.
///
/// This buffer is especially suitable when you want to upload or download some data regularly
/// (for example, at each frame for a video game).
///
/// # Usage
///
/// A `CpuBufferPool` is similar to a ring buffer. You start by creating an empty pool, then you
/// grab elements from the pool and use them, and if the pool is full it will automatically grow
/// in size.
///
/// Contrary to a `Vec`, elements automatically free themselves when they are dropped (ie. usually
/// when you call `cleanup_finished()` on a future, or when you drop that future).
///
/// # Arc-like
///
/// The `CpuBufferPool` struct internally contains an `Arc`. You can clone the `CpuBufferPool` for
/// a cheap cost, and all the clones will share the same underlying buffer.
///
/// # Example
///
/// ```
/// use vulkano::buffer::CpuBufferPool;
/// use vulkano::command_buffer::AutoCommandBufferBuilder;
/// use vulkano::command_buffer::CommandBufferUsage;
/// use vulkano::command_buffer::PrimaryCommandBufferAbstract;
/// use vulkano::sync::GpuFuture;
/// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// # let command_buffer_allocator: vulkano::command_buffer::allocator::StandardCommandBufferAllocator = return;
///
/// // Create the ring buffer.
/// let buffer = CpuBufferPool::upload(memory_allocator);
///
/// for n in 0 .. 25u32 {
///     // Each loop grabs a new entry from that ring buffer and stores ` data` in it.
///     let data: [f32; 4] = [1.0, 0.5, n as f32 / 24.0, 0.0];
///     let sub_buffer = buffer.from_data(data).unwrap();
///
///     // You can then use `sub_buffer` as if it was an entirely separate buffer.
///     AutoCommandBufferBuilder::primary(
///         &command_buffer_allocator,
///         queue.queue_family_index(),
///         CommandBufferUsage::OneTimeSubmit,
///     )
///     .unwrap()
///     // For the sake of the example we just call `update_buffer` on the buffer, even though
///     // it is pointless to do that.
///     .update_buffer(&[0.2, 0.3, 0.4, 0.5], sub_buffer.clone(), 0)
///     .unwrap()
///     .build().unwrap()
///     .execute(queue.clone())
///     .unwrap()
///     .then_signal_fence_and_flush()
///     .unwrap();
/// }
/// ```
pub struct CpuBufferPool<T, A = StandardMemoryAllocator>
where
    [T]: BufferContents,
    A: MemoryAllocator + ?Sized,
{
    // The memory pool to use for allocations.
    allocator: Arc<A>,

    // Current buffer from which elements are grabbed.
    current_buffer: Mutex<Option<Arc<ActualBuffer>>>,

    // Buffer usage.
    buffer_usage: BufferUsage,

    memory_usage: MemoryUsage,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

// One buffer of the pool.
#[derive(Debug)]
struct ActualBuffer {
    inner: Arc<Buffer>,

    // List of the chunks that are reserved.
    chunks_in_use: Mutex<Vec<ActualBufferChunk>>,

    // The index of the chunk that should be available next for the ring buffer.
    next_index: AtomicU64,

    // Number of elements in the buffer.
    capacity: DeviceSize,
}

// Access pattern of one subbuffer.
#[derive(Debug)]
struct ActualBufferChunk {
    // First element number within the actual buffer.
    index: DeviceSize,

    // Number of occupied elements within the actual buffer.
    len: DeviceSize,

    // Number of `CpuBufferPoolSubbuffer` objects that point to this subbuffer.
    num_cpu_accesses: usize,
}

/// A subbuffer allocated from a `CpuBufferPool`.
///
/// When this object is destroyed, the subbuffer is automatically reclaimed by the pool.
pub struct CpuBufferPoolChunk<T>
where
    [T]: BufferContents,
{
    buffer: Arc<ActualBuffer>,

    // Index of the subbuffer within `buffer`. In number of elements.
    index: DeviceSize,

    // Number of bytes to add to `index * mem::size_of::<T>()` to obtain the start of the data in
    // the buffer. Necessary for alignment purposes.
    align_offset: DeviceSize,

    // Size of the subbuffer in number of elements, as requested by the user.
    // If this is 0, then no entry was added to `chunks_in_use`.
    requested_len: DeviceSize,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

/// A subbuffer allocated from a `CpuBufferPool`.
///
/// When this object is destroyed, the subbuffer is automatically reclaimed by the pool.
pub struct CpuBufferPoolSubbuffer<T>
where
    [T]: BufferContents,
{
    // This struct is just a wrapper around `CpuBufferPoolChunk`.
    chunk: CpuBufferPoolChunk<T>,
}

impl<T, A> CpuBufferPool<T, A>
where
    [T]: BufferContents,
    A: MemoryAllocator + ?Sized,
{
    /// Builds a `CpuBufferPool`.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `memory_usage` is [`MemoryUsage::GpuOnly`].
    pub fn new(
        allocator: Arc<A>,
        buffer_usage: BufferUsage,
        memory_usage: MemoryUsage,
    ) -> CpuBufferPool<T, A> {
        assert!(size_of::<T>() > 0);
        assert!(memory_usage != MemoryUsage::GpuOnly);

        CpuBufferPool {
            allocator,
            current_buffer: Mutex::new(None),
            buffer_usage,
            memory_usage,
            marker: PhantomData,
        }
    }

    /// Builds a `CpuBufferPool` meant for simple uploads.
    ///
    /// Shortcut for a pool that can only be used as transfer source and with exclusive queue
    /// family accesses.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    pub fn upload(allocator: Arc<A>) -> CpuBufferPool<T, A> {
        CpuBufferPool::new(
            allocator,
            BufferUsage {
                transfer_src: true,
                ..BufferUsage::empty()
            },
            MemoryUsage::Upload,
        )
    }

    /// Builds a `CpuBufferPool` meant for simple downloads.
    ///
    /// Shortcut for a pool that can only be used as transfer destination and with exclusive queue
    /// family accesses.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    pub fn download(allocator: Arc<A>) -> CpuBufferPool<T, A> {
        CpuBufferPool::new(
            allocator,
            BufferUsage {
                transfer_dst: true,
                ..BufferUsage::empty()
            },
            MemoryUsage::Download,
        )
    }

    /// Builds a `CpuBufferPool` meant for usage as a uniform buffer.
    ///
    /// Shortcut for a pool that can only be used as uniform buffer and with exclusive queue
    /// family accesses.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    pub fn uniform_buffer(allocator: Arc<A>) -> CpuBufferPool<T, A> {
        CpuBufferPool::new(
            allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            MemoryUsage::Upload,
        )
    }

    /// Builds a `CpuBufferPool` meant for usage as a vertex buffer.
    ///
    /// Shortcut for a pool that can only be used as vertex buffer and with exclusive queue
    /// family accesses.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    pub fn vertex_buffer(allocator: Arc<A>) -> CpuBufferPool<T, A> {
        CpuBufferPool::new(
            allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            MemoryUsage::Upload,
        )
    }

    /// Builds a `CpuBufferPool` meant for usage as a indirect buffer.
    ///
    /// Shortcut for a pool that can only be used as indirect buffer and with exclusive queue
    /// family accesses.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    pub fn indirect_buffer(allocator: Arc<A>) -> CpuBufferPool<T, A> {
        CpuBufferPool::new(
            allocator,
            BufferUsage {
                indirect_buffer: true,
                ..BufferUsage::empty()
            },
            MemoryUsage::Upload,
        )
    }
}

impl<T, A> CpuBufferPool<T, A>
where
    [T]: BufferContents,
    A: MemoryAllocator + ?Sized,
{
    /// Returns the current capacity of the pool, in number of elements.
    pub fn capacity(&self) -> DeviceSize {
        match *self.current_buffer.lock().unwrap() {
            None => 0,
            Some(ref buf) => buf.capacity,
        }
    }

    /// Makes sure that the capacity is at least `capacity`. Allocates memory if it is not the
    /// case.
    ///
    /// Since this can involve a memory allocation, an `OomError` can happen.
    pub fn reserve(&self, capacity: DeviceSize) -> Result<(), AllocationCreationError> {
        if capacity == 0 {
            return Ok(());
        }

        let mut cur_buf = self.current_buffer.lock().unwrap();

        // Check current capacity.
        match *cur_buf {
            Some(ref buf) if buf.capacity >= capacity => {
                return Ok(());
            }
            _ => (),
        };

        self.reset_buf(&mut cur_buf, capacity)
    }

    /// Grants access to a new subbuffer and puts `data` in it.
    ///
    /// If no subbuffer is available (because they are still in use by the GPU), a new buffer will
    /// automatically be allocated.
    ///
    /// > **Note**: You can think of it like a `Vec`. If you insert an element and the `Vec` is not
    /// > large enough, a new chunk of memory is automatically allocated.
    pub fn from_data(
        &self,
        data: T,
    ) -> Result<Arc<CpuBufferPoolSubbuffer<T>>, AllocationCreationError> {
        Ok(Arc::new(CpuBufferPoolSubbuffer {
            chunk: self.chunk_impl([data].into_iter())?,
        }))
    }

    /// Grants access to a new subbuffer and puts all elements of `iter` in it.
    ///
    /// If no subbuffer is available (because they are still in use by the GPU), a new buffer will
    /// automatically be allocated.
    ///
    /// > **Note**: You can think of it like a `Vec`. If you insert elements and the `Vec` is not
    /// > large enough, a new chunk of memory is automatically allocated.
    ///
    /// # Panic
    ///
    /// Panics if the length of the iterator didn't match the actual number of elements.
    pub fn from_iter<I>(
        &self,
        iter: I,
    ) -> Result<Arc<CpuBufferPoolChunk<T>>, AllocationCreationError>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        self.chunk_impl(iter.into_iter()).map(Arc::new)
    }

    fn chunk_impl(
        &self,
        data: impl ExactSizeIterator<Item = T>,
    ) -> Result<CpuBufferPoolChunk<T>, AllocationCreationError> {
        let mut mutex = self.current_buffer.lock().unwrap();

        let data = match self.try_next_impl(&mut mutex, data) {
            Ok(n) => return Ok(n),
            Err(d) => d,
        };

        let next_capacity = match *mutex {
            Some(ref b) if (data.len() as DeviceSize) < b.capacity => 2 * b.capacity,
            _ => 2 * data.len().max(1) as DeviceSize,
        };

        self.reset_buf(&mut mutex, next_capacity)?;

        match self.try_next_impl(&mut mutex, data) {
            Ok(n) => Ok(n),
            Err(_) => unreachable!(),
        }
    }

    /// Grants access to a new subbuffer and puts `data` in it.
    ///
    /// Returns `None` if no subbuffer is available.
    ///
    /// A `CpuBufferPool` is always empty the first time you use it, so you shouldn't use
    /// `try_next` the first time you use it.
    pub fn try_next(&self, data: T) -> Option<Arc<CpuBufferPoolSubbuffer<T>>> {
        let mut mutex = self.current_buffer.lock().unwrap();
        self.try_next_impl(&mut mutex, [data])
            .map(|c| Arc::new(CpuBufferPoolSubbuffer { chunk: c }))
            .ok()
    }

    // Creates a new buffer and sets it as current. The capacity is in number of elements.
    //
    // `cur_buf_mutex` must be an active lock of `self.current_buffer`.
    fn reset_buf(
        &self,
        cur_buf_mutex: &mut MutexGuard<'_, Option<Arc<ActualBuffer>>>,
        capacity: DeviceSize,
    ) -> Result<(), AllocationCreationError> {
        let size = match (size_of::<T>() as DeviceSize).checked_mul(capacity) {
            Some(s) => s,
            None => {
                return Err(AllocationCreationError::VulkanError(
                    VulkanError::OutOfDeviceMemory,
                ))
            }
        };

        let raw_buffer = RawBuffer::new(
            self.device().clone(),
            BufferCreateInfo {
                size,
                usage: self.buffer_usage,
                ..Default::default()
            },
        )
        .map_err(|err| match err {
            BufferError::AllocError(err) => err,
            // We don't use sparse-binding, therefore the other errors can't happen.
            _ => unreachable!(),
        })?;
        let requirements = *raw_buffer.memory_requirements();
        let create_info = AllocationCreateInfo {
            requirements,
            allocation_type: AllocationType::Linear,
            usage: self.memory_usage,
            allocate_preference: MemoryAllocatePreference::Unknown,
            dedicated_allocation: Some(DedicatedAllocation::Buffer(&raw_buffer)),
            ..Default::default()
        };

        match unsafe { self.allocator.allocate_unchecked(create_info) } {
            Ok(mut alloc) => {
                debug_assert!(alloc.offset() % requirements.alignment == 0);
                debug_assert!(alloc.size() == requirements.size);
                alloc.shrink(size);
                let inner = unsafe {
                    Arc::new(
                        raw_buffer
                            .bind_memory_unchecked(alloc)
                            .map_err(|(err, _, _)| err)?,
                    )
                };

                **cur_buf_mutex = Some(Arc::new(ActualBuffer {
                    inner,
                    chunks_in_use: Mutex::new(vec![]),
                    next_index: AtomicU64::new(0),
                    capacity,
                }));

                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    // Tries to lock a subbuffer from the current buffer.
    //
    // `cur_buf_mutex` must be an active lock of `self.current_buffer`.
    //
    // Returns `data` wrapped inside an `Err` if there is no slot available in the current buffer.
    //
    // # Panic
    //
    // Panics if the length of the iterator didn't match the actual number of element.
    fn try_next_impl<I>(
        &self,
        cur_buf_mutex: &mut MutexGuard<'_, Option<Arc<ActualBuffer>>>,
        data: I,
    ) -> Result<CpuBufferPoolChunk<T>, I::IntoIter>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut data = data.into_iter();

        // Grab the current buffer. Return `Err` if the pool wasn't "initialized" yet.
        let current_buffer = match cur_buf_mutex.clone() {
            Some(b) => b,
            None => return Err(data),
        };

        let mut chunks_in_use = current_buffer.chunks_in_use.lock().unwrap();
        debug_assert!(!chunks_in_use.iter().any(|c| c.len == 0));

        // Number of elements requested by the user.
        let requested_len = data.len() as DeviceSize;

        // We special case when 0 elements are requested. Polluting the list of allocated chunks
        // with chunks of length 0 means that we will have troubles deallocating.
        if requested_len == 0 {
            assert!(
                data.next().is_none(),
                "Expected iterator passed to CpuBufferPool::chunk to be empty"
            );
            return Ok(CpuBufferPoolChunk {
                // TODO: remove .clone() once non-lexical borrows land
                buffer: current_buffer.clone(),
                index: 0,
                align_offset: 0,
                requested_len: 0,
                marker: PhantomData,
            });
        }

        let allocation = match current_buffer.inner.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        // Find a suitable offset and len, or returns if none available.
        let (index, occupied_len, align_offset) = {
            let (tentative_index, tentative_len, tentative_align_offset) = {
                // Since the only place that touches `next_index` is this code, and since we
                // own a mutex lock to the buffer, it means that `next_index` can't be accessed
                // concurrently.
                // TODO: ^ eventually should be put inside the mutex
                let idx = current_buffer.next_index.load(Ordering::SeqCst);

                // Find the required alignment in bytes.
                let align_uniform = if self.buffer_usage.uniform_buffer {
                    self.device()
                        .physical_device()
                        .properties()
                        .min_uniform_buffer_offset_alignment
                } else {
                    1
                };
                let align_storage = if self.buffer_usage.storage_buffer {
                    self.device()
                        .physical_device()
                        .properties()
                        .min_storage_buffer_offset_alignment
                } else {
                    1
                };
                let mut align_bytes = align_uniform
                    .max(align_storage)
                    .max(align_of::<T>() as DeviceSize);
                if let Some(atom_size) = allocation.atom_size() {
                    align_bytes = DeviceSize::max(align_bytes, atom_size.get());
                }

                let tentative_align_offset = (align_bytes
                    - ((idx * size_of::<T>() as DeviceSize) % align_bytes))
                    % align_bytes;
                let additional_len = if tentative_align_offset == 0 {
                    0
                } else {
                    1 + (tentative_align_offset - 1) / size_of::<T>() as DeviceSize
                };

                (idx, requested_len + additional_len, tentative_align_offset)
            };

            // Find out whether any chunk in use overlaps this range.
            if tentative_index + tentative_len <= current_buffer.capacity
                && !chunks_in_use.iter().any(|c| {
                    (c.index >= tentative_index && c.index < tentative_index + tentative_len)
                        || (c.index <= tentative_index && c.index + c.len > tentative_index)
                })
            {
                (tentative_index, tentative_len, tentative_align_offset)
            } else {
                // Impossible to allocate at `tentative_index`. Let's try 0 instead.
                if requested_len <= current_buffer.capacity
                    && !chunks_in_use.iter().any(|c| c.index < requested_len)
                {
                    (0, requested_len, 0)
                } else {
                    // Buffer is full. Return.
                    return Err(data);
                }
            }
        };

        // Write `data` in the memory.
        unsafe {
            let mut range = (index * size_of::<T>() as DeviceSize + align_offset)
                ..((index + requested_len) * size_of::<T>() as DeviceSize + align_offset);

            let bytes = allocation.write(range.clone()).unwrap();
            let mapping = <[T]>::from_bytes_mut(bytes).unwrap();

            let mut written = 0;
            for (o, i) in mapping.iter_mut().zip(data) {
                ptr::write(o, i);
                written += 1;
            }

            if let Some(atom_size) = allocation.atom_size() {
                range.end =
                    DeviceSize::min(align_up(range.end, atom_size.get()), allocation.size());
                allocation.flush_range(range).unwrap();
            }

            assert_eq!(
                written, requested_len,
                "Iterator passed to CpuBufferPool::chunk has a mismatch between reported \
                length and actual number of elements"
            );
        }

        // Mark the chunk as in use.
        current_buffer
            .next_index
            .store(index + occupied_len, Ordering::SeqCst);
        chunks_in_use.push(ActualBufferChunk {
            index,
            len: occupied_len,
            num_cpu_accesses: 1,
        });

        Ok(CpuBufferPoolChunk {
            // TODO: remove .clone() once non-lexical borrows land
            buffer: current_buffer.clone(),
            index,
            align_offset,
            requested_len,
            marker: PhantomData,
        })
    }
}

// Can't automatically derive `Clone`, otherwise the compiler adds a `T: Clone` requirement.
impl<T, A> Clone for CpuBufferPool<T, A>
where
    [T]: BufferContents,
    A: MemoryAllocator + ?Sized,
{
    fn clone(&self) -> Self {
        let buf = self.current_buffer.lock().unwrap();

        CpuBufferPool {
            allocator: self.allocator.clone(),
            current_buffer: Mutex::new(buf.clone()),
            buffer_usage: self.buffer_usage,
            memory_usage: self.memory_usage,
            marker: PhantomData,
        }
    }
}

unsafe impl<T, A> DeviceOwned for CpuBufferPool<T, A>
where
    [T]: BufferContents,
    A: MemoryAllocator + ?Sized,
{
    fn device(&self) -> &Arc<Device> {
        self.allocator.device()
    }
}

impl<T> Clone for CpuBufferPoolChunk<T>
where
    [T]: BufferContents,
{
    fn clone(&self) -> CpuBufferPoolChunk<T> {
        let mut chunks_in_use_lock = self.buffer.chunks_in_use.lock().unwrap();
        let chunk = chunks_in_use_lock
            .iter_mut()
            .find(|c| c.index == self.index)
            .unwrap();

        debug_assert!(chunk.num_cpu_accesses >= 1);
        chunk.num_cpu_accesses = chunk
            .num_cpu_accesses
            .checked_add(1)
            .expect("Overflow in CPU accesses");

        CpuBufferPoolChunk {
            buffer: self.buffer.clone(),
            index: self.index,
            align_offset: self.align_offset,
            requested_len: self.requested_len,
            marker: PhantomData,
        }
    }
}

unsafe impl<T> BufferAccess for CpuBufferPoolChunk<T>
where
    T: Send + Sync,
    [T]: BufferContents,
{
    fn inner(&self) -> BufferInner<'_> {
        BufferInner {
            buffer: &self.buffer.inner,
            offset: self.index * size_of::<T>() as DeviceSize + self.align_offset,
        }
    }

    fn size(&self) -> DeviceSize {
        self.requested_len * size_of::<T>() as DeviceSize
    }
}

impl<T> BufferAccessObject for Arc<CpuBufferPoolChunk<T>>
where
    T: Send + Sync,
    [T]: BufferContents,
{
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess> {
        self.clone()
    }
}

impl<T> Drop for CpuBufferPoolChunk<T>
where
    [T]: BufferContents,
{
    fn drop(&mut self) {
        // If `requested_len` is 0, then no entry was added in the chunks.
        if self.requested_len == 0 {
            return;
        }

        let mut chunks_in_use_lock = self.buffer.chunks_in_use.lock().unwrap();
        let chunk_num = chunks_in_use_lock
            .iter_mut()
            .position(|c| c.index == self.index)
            .unwrap();

        if chunks_in_use_lock[chunk_num].num_cpu_accesses >= 2 {
            chunks_in_use_lock[chunk_num].num_cpu_accesses -= 1;
        } else {
            chunks_in_use_lock.remove(chunk_num);
        }
    }
}

unsafe impl<T> TypedBufferAccess for CpuBufferPoolChunk<T>
where
    T: Send + Sync,
    [T]: BufferContents,
{
    type Content = [T];
}

unsafe impl<T> DeviceOwned for CpuBufferPoolChunk<T>
where
    [T]: BufferContents,
{
    fn device(&self) -> &Arc<Device> {
        self.buffer.inner.device()
    }
}

impl<T> PartialEq for CpuBufferPoolChunk<T>
where
    T: Send + Sync,
    [T]: BufferContents,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T> Eq for CpuBufferPoolChunk<T>
where
    T: Send + Sync,
    [T]: BufferContents,
{
}

impl<T> Hash for CpuBufferPoolChunk<T>
where
    T: Send + Sync,
    [T]: BufferContents,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

impl<T> Clone for CpuBufferPoolSubbuffer<T>
where
    [T]: BufferContents,
{
    fn clone(&self) -> CpuBufferPoolSubbuffer<T> {
        CpuBufferPoolSubbuffer {
            chunk: self.chunk.clone(),
        }
    }
}

unsafe impl<T> BufferAccess for CpuBufferPoolSubbuffer<T>
where
    T: Send + Sync,
    [T]: BufferContents,
{
    fn inner(&self) -> BufferInner<'_> {
        self.chunk.inner()
    }

    fn size(&self) -> DeviceSize {
        self.chunk.size()
    }
}

impl<T> BufferAccessObject for Arc<CpuBufferPoolSubbuffer<T>>
where
    T: Send + Sync,
    [T]: BufferContents,
{
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess> {
        self.clone()
    }
}

unsafe impl<T> TypedBufferAccess for CpuBufferPoolSubbuffer<T>
where
    T: BufferContents,
    [T]: BufferContents,
{
    type Content = T;
}

unsafe impl<T> DeviceOwned for CpuBufferPoolSubbuffer<T>
where
    [T]: BufferContents,
{
    fn device(&self) -> &Arc<Device> {
        self.chunk.buffer.inner.device()
    }
}

impl<T> PartialEq for CpuBufferPoolSubbuffer<T>
where
    T: Send + Sync,
    [T]: BufferContents,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T> Eq for CpuBufferPoolSubbuffer<T>
where
    T: Send + Sync,
    [T]: BufferContents,
{
}

impl<T> Hash for CpuBufferPoolSubbuffer<T>
where
    T: Send + Sync,
    [T]: BufferContents,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn basic_create() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));
        let _ = CpuBufferPool::<u8>::upload(memory_allocator);
    }

    #[test]
    fn reserve() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let pool = CpuBufferPool::<u8>::upload(memory_allocator);
        assert_eq!(pool.capacity(), 0);

        pool.reserve(83).unwrap();
        assert_eq!(pool.capacity(), 83);
    }

    #[test]
    fn capacity_increase() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let pool = CpuBufferPool::upload(memory_allocator);
        assert_eq!(pool.capacity(), 0);

        pool.from_data(12).unwrap();
        let first_cap = pool.capacity();
        assert!(first_cap >= 1);

        for _ in 0..first_cap + 5 {
            mem::forget(pool.from_data(12).unwrap());
        }

        assert!(pool.capacity() > first_cap);
    }

    #[test]
    fn reuse_subbuffers() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let pool = CpuBufferPool::upload(memory_allocator);
        assert_eq!(pool.capacity(), 0);

        let mut capacity = None;
        for _ in 0..64 {
            pool.from_data(12).unwrap();

            let new_cap = pool.capacity();
            assert!(new_cap >= 1);
            match capacity {
                None => capacity = Some(new_cap),
                Some(c) => assert_eq!(c, new_cap),
            }
        }
    }

    #[test]
    fn chunk_loopback() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let pool = CpuBufferPool::<u8>::upload(memory_allocator);
        pool.reserve(5).unwrap();

        let a = pool.from_iter(vec![0, 0]).unwrap();
        let b = pool.from_iter(vec![0, 0]).unwrap();
        assert_eq!(b.index, 2);
        drop(a);

        let c = pool.from_iter(vec![0, 0]).unwrap();
        assert_eq!(c.index, 0);

        assert_eq!(pool.capacity(), 5);
    }

    #[test]
    fn chunk_0_elems_doesnt_pollute() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let pool = CpuBufferPool::<u8>::upload(memory_allocator);

        let _ = pool.from_iter(vec![]).unwrap();
        let _ = pool.from_iter(vec![0, 0]).unwrap();
    }
}
