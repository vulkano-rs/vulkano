// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::hash::Hash;
use std::hash::Hasher;
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use buffer::BufferUsage;
use buffer::sys::BufferCreationError;
use buffer::sys::SparseLevel;
use buffer::sys::UnsafeBuffer;
use buffer::traits::BufferAccess;
use buffer::traits::BufferInner;
use buffer::traits::TypedBufferAccess;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use memory::DedicatedAlloc;
use memory::DeviceMemoryAllocError;
use memory::pool::AllocFromRequirementsFilter;
use memory::pool::AllocLayout;
use memory::pool::MappingRequirement;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::PotentialDedicatedAllocation;
use memory::pool::StdMemoryPool;
use sync::AccessError;
use sync::Sharing;

use OomError;

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
/// use vulkano::command_buffer::CommandBuffer;
/// use vulkano::sync::GpuFuture;
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
///
/// // Create the ring buffer.
/// let buffer = CpuBufferPool::upload(device.clone());
///
/// for n in 0 .. 25u32 {
///     // Each loop grabs a new entry from that ring buffer and stores ` data` in it.
///     let data: [f32; 4] = [1.0, 0.5, n as f32 / 24.0, 0.0];
///     let sub_buffer = buffer.next(data).unwrap();
///
///     // You can then use `sub_buffer` as if it was an entirely separate buffer.
///     AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
///         .unwrap()
///         // For the sake of the example we just call `update_buffer` on the buffer, even though
///         // it is pointless to do that.
///         .update_buffer(sub_buffer.clone(), [0.2, 0.3, 0.4, 0.5])
///         .unwrap()
///         .build().unwrap()
///         .execute(queue.clone())
///         .unwrap()
///         .then_signal_fence_and_flush()
///         .unwrap();
/// }
/// ```
///
pub struct CpuBufferPool<T, A = Arc<StdMemoryPool>>
    where A: MemoryPool
{
    // The device of the pool.
    device: Arc<Device>,

    // The memory pool to use for allocations.
    pool: A,

    // Current buffer from which elements are grabbed.
    current_buffer: Mutex<Option<Arc<ActualBuffer<A>>>>,

    // Buffer usage.
    usage: BufferUsage,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

// One buffer of the pool.
struct ActualBuffer<A>
    where A: MemoryPool
{
    // Inner content.
    inner: UnsafeBuffer,

    // The memory held by the buffer.
    memory: PotentialDedicatedAllocation<A::Alloc>,

    // List of the chunks that are reserved.
    chunks_in_use: Mutex<Vec<ActualBufferChunk>>,

    // The index of the chunk that should be available next for the ring buffer.
    next_index: AtomicUsize,

    // Number of elements in the buffer.
    capacity: usize,
}

// Access pattern of one subbuffer.
#[derive(Debug)]
struct ActualBufferChunk {
    // First element number within the actual buffer.
    index: usize,

    // Number of occupied elements within the actual buffer.
    len: usize,

    // Number of `CpuBufferPoolSubbuffer` objects that point to this subbuffer.
    num_cpu_accesses: usize,

    // Number of `CpuBufferPoolSubbuffer` objects that point to this subbuffer and that have been
    // GPU-locked.
    num_gpu_accesses: usize,
}

/// A subbuffer allocated from a `CpuBufferPool`.
///
/// When this object is destroyed, the subbuffer is automatically reclaimed by the pool.
pub struct CpuBufferPoolChunk<T, A>
    where A: MemoryPool
{
    buffer: Arc<ActualBuffer<A>>,

    // Index of the subbuffer within `buffer`. In number of elements.
    index: usize,

    // Number of bytes to add to `index * mem::size_of::<T>()` to obtain the start of the data in
    // the buffer. Necessary for alignment purposes.
    align_offset: usize,

    // Size of the subbuffer in number of elements, as requested by the user.
    // If this is 0, then no entry was added to `chunks_in_use`.
    requested_len: usize,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

/// A subbuffer allocated from a `CpuBufferPool`.
///
/// When this object is destroyed, the subbuffer is automatically reclaimed by the pool.
pub struct CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    // This struct is just a wrapper around `CpuBufferPoolChunk`.
    chunk: CpuBufferPoolChunk<T, A>,
}

impl<T> CpuBufferPool<T> {
    /// Builds a `CpuBufferPool`.
    #[inline]
    pub fn new(device: Arc<Device>, usage: BufferUsage) -> CpuBufferPool<T> {
        let pool = Device::standard_pool(&device);

        CpuBufferPool {
            device: device,
            pool: pool,
            current_buffer: Mutex::new(None),
            usage: usage.clone(),
            marker: PhantomData,
        }
    }

    /// Builds a `CpuBufferPool` meant for simple uploads.
    ///
    /// Shortcut for a pool that can only be used as transfer source and with exclusive queue
    /// family accesses.
    #[inline]
    pub fn upload(device: Arc<Device>) -> CpuBufferPool<T> {
        CpuBufferPool::new(device, BufferUsage::transfer_source())
    }

    /// Builds a `CpuBufferPool` meant for simple downloads.
    ///
    /// Shortcut for a pool that can only be used as transfer destination and with exclusive queue
    /// family accesses.
    #[inline]
    pub fn download(device: Arc<Device>) -> CpuBufferPool<T> {
        CpuBufferPool::new(device, BufferUsage::transfer_destination())
    }

    /// Builds a `CpuBufferPool` meant for usage as a uniform buffer.
    ///
    /// Shortcut for a pool that can only be used as uniform buffer and with exclusive queue
    /// family accesses.
    #[inline]
    pub fn uniform_buffer(device: Arc<Device>) -> CpuBufferPool<T> {
        CpuBufferPool::new(device, BufferUsage::uniform_buffer())
    }

    /// Builds a `CpuBufferPool` meant for usage as a vertex buffer.
    ///
    /// Shortcut for a pool that can only be used as vertex buffer and with exclusive queue
    /// family accesses.
    #[inline]
    pub fn vertex_buffer(device: Arc<Device>) -> CpuBufferPool<T> {
        CpuBufferPool::new(device, BufferUsage::vertex_buffer())
    }

    /// Builds a `CpuBufferPool` meant for usage as a indirect buffer.
    ///
    /// Shortcut for a pool that can only be used as indirect buffer and with exclusive queue
    /// family accesses.
    #[inline]
    pub fn indirect_buffer(device: Arc<Device>) -> CpuBufferPool<T> {
        CpuBufferPool::new(device, BufferUsage::indirect_buffer())
    }
}

impl<T, A> CpuBufferPool<T, A>
    where A: MemoryPool
{
    /// Returns the current capacity of the pool, in number of elements.
    pub fn capacity(&self) -> usize {
        match *self.current_buffer.lock().unwrap() {
            None => 0,
            Some(ref buf) => buf.capacity,
        }
    }

    /// Makes sure that the capacity is at least `capacity`. Allocates memory if it is not the
    /// case.
    ///
    /// Since this can involve a memory allocation, an `OomError` can happen.
    pub fn reserve(&self, capacity: usize) -> Result<(), DeviceMemoryAllocError> {
        let mut cur_buf = self.current_buffer.lock().unwrap();

        // Check current capacity.
        match *cur_buf {
            Some(ref buf) if buf.capacity >= capacity => {
                return Ok(());
            },
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
    #[inline]
    pub fn next(&self, data: T) -> Result<CpuBufferPoolSubbuffer<T, A>, DeviceMemoryAllocError> {
        Ok(CpuBufferPoolSubbuffer { chunk: self.chunk(iter::once(data))? })
    }

    /// Grants access to a new subbuffer and puts `data` in it.
    ///
    /// If no subbuffer is available (because they are still in use by the GPU), a new buffer will
    /// automatically be allocated.
    ///
    /// > **Note**: You can think of it like a `Vec`. If you insert elements and the `Vec` is not
    /// > large enough, a new chunk of memory is automatically allocated.
    ///
    /// # Panic
    ///
    /// Panics if the length of the iterator didn't match the actual number of element.
    ///
    pub fn chunk<I>(&self, data: I) -> Result<CpuBufferPoolChunk<T, A>, DeviceMemoryAllocError>
        where I: IntoIterator<Item = T>,
              I::IntoIter: ExactSizeIterator
    {
        let data = data.into_iter();

        let mut mutex = self.current_buffer.lock().unwrap();

        let data = match self.try_next_impl(&mut mutex, data) {
            Ok(n) => return Ok(n),
            Err(d) => d,
        };

        let next_capacity = match *mutex {
            Some(ref b) if data.len() < b.capacity => 2 * b.capacity,
            _ => 2 * data.len(),
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
    #[inline]
    pub fn try_next(&self, data: T) -> Option<CpuBufferPoolSubbuffer<T, A>> {
        let mut mutex = self.current_buffer.lock().unwrap();
        self.try_next_impl(&mut mutex, iter::once(data))
            .map(|c| CpuBufferPoolSubbuffer { chunk: c })
            .ok()
    }

    // Creates a new buffer and sets it as current. The capacity is in number of elements.
    //
    // `cur_buf_mutex` must be an active lock of `self.current_buffer`.
    fn reset_buf(&self, cur_buf_mutex: &mut MutexGuard<Option<Arc<ActualBuffer<A>>>>,
                 capacity: usize)
                 -> Result<(), DeviceMemoryAllocError> {
        unsafe {
            let (buffer, mem_reqs) = {
                let size_bytes = match mem::size_of::<T>().checked_mul(capacity) {
                    Some(s) => s,
                    None =>
                        return Err(DeviceMemoryAllocError::OomError(OomError::OutOfDeviceMemory)),
                };

                match UnsafeBuffer::new(self.device.clone(),
                                          size_bytes,
                                          self.usage,
                                          Sharing::Exclusive::<iter::Empty<_>>,
                                          SparseLevel::none()) {
                    Ok(b) => b,
                    Err(BufferCreationError::AllocError(err)) => return Err(err),
                    Err(_) => unreachable!(),        // We don't use sparse binding, therefore the other
                    // errors can't happen
                }
            };

            let mem = MemoryPool::alloc_from_requirements(&self.pool,
                                        &mem_reqs,
                                        AllocLayout::Linear,
                                        MappingRequirement::Map,
                                        DedicatedAlloc::Buffer(&buffer),
                                        |_| AllocFromRequirementsFilter::Allowed)?;
            debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
            debug_assert!(mem.mapped_memory().is_some());
            buffer.bind_memory(mem.memory(), mem.offset())?;

            **cur_buf_mutex = Some(Arc::new(ActualBuffer {
                                                inner: buffer,
                                                memory: mem,
                                                chunks_in_use: Mutex::new(vec![]),
                                                next_index: AtomicUsize::new(0),
                                                capacity: capacity,
                                            }));

            Ok(())
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
    //
    fn try_next_impl<I>(&self, cur_buf_mutex: &mut MutexGuard<Option<Arc<ActualBuffer<A>>>>,
                        mut data: I)
                        -> Result<CpuBufferPoolChunk<T, A>, I>
        where I: ExactSizeIterator<Item = T>
    {
        // Grab the current buffer. Return `Err` if the pool wasn't "initialized" yet.
        let current_buffer = match cur_buf_mutex.clone() {
            Some(b) => b,
            None => return Err(data),
        };

        let mut chunks_in_use = current_buffer.chunks_in_use.lock().unwrap();
        debug_assert!(!chunks_in_use.iter().any(|c| c.len == 0));

        // Number of elements requested by the user.
        let requested_len = data.len();

        // We special case when 0 elements are requested. Polluting the list of allocated chunks
        // with chunks of length 0 means that we will have troubles deallocating.
        if requested_len == 0 {
            assert!(data.next().is_none(),
                    "Expected iterator passed to CpuBufferPool::chunk to be empty");
            return Ok(CpuBufferPoolChunk {
                          // TODO: remove .clone() once non-lexical borrows land
                          buffer: current_buffer.clone(),
                          index: 0,
                          align_offset: 0,
                          requested_len: 0,
                          marker: PhantomData,
                      });
        }

        // Find a suitable offset and len, or returns if none available.
        let (index, occupied_len, align_offset) = {
            let (tentative_index, tentative_len, tentative_align_offset) = {
                // Since the only place that touches `next_index` is this code, and since we
                // own a mutex lock to the buffer, it means that `next_index` can't be accessed
                // concurrently.
                // TODO: ^ eventually should be put inside the mutex
                let idx = current_buffer.next_index.load(Ordering::SeqCst);

                // Find the required alignment in bytes.
                let align_bytes = cmp::max(if self.usage.uniform_buffer {
                                               self.device()
                                                   .physical_device()
                                                   .limits()
                                                   .min_uniform_buffer_offset_alignment() as
                                                   usize
                                           } else {
                                               1
                                           },
                                           if self.usage.storage_buffer {
                                               self.device()
                                                   .physical_device()
                                                   .limits()
                                                   .min_storage_buffer_offset_alignment() as
                                                   usize
                                           } else {
                                               1
                                           });

                let tentative_align_offset =
                    (align_bytes - ((idx * mem::size_of::<T>()) % align_bytes)) % align_bytes;
                let additional_len = if tentative_align_offset == 0 {
                    0
                } else {
                    1 + (tentative_align_offset - 1) / mem::size_of::<T>()
                };

                (idx, requested_len + additional_len, tentative_align_offset)
            };

            // Find out whether any chunk in use overlaps this range.
            if tentative_index + tentative_len <= current_buffer.capacity &&
                !chunks_in_use.iter().any(|c| {
                                              (c.index >= tentative_index &&
                                                   c.index < tentative_index + tentative_len) ||
                                                  (c.index <= tentative_index &&
                                                       c.index + c.len > tentative_index)
                                          })
            {
                (tentative_index, tentative_len, tentative_align_offset)
            } else {
                // Impossible to allocate at `tentative_index`. Let's try 0 instead.
                if requested_len <= current_buffer.capacity &&
                    !chunks_in_use.iter().any(|c| c.index < requested_len)
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
            let mem_off = current_buffer.memory.offset();
            let range_start = index * mem::size_of::<T>() + align_offset + mem_off;
            let range_end = (index + requested_len) * mem::size_of::<T>() + align_offset + mem_off;
            let mut mapping = current_buffer
                .memory
                .mapped_memory()
                .unwrap()
                .read_write::<[T]>(range_start .. range_end);

            let mut written = 0;
            for (o, i) in mapping.iter_mut().zip(data) {
                ptr::write(o, i);
                written += 1;
            }
            assert_eq!(written,
                       requested_len,
                       "Iterator passed to CpuBufferPool::chunk has a mismatch between reported \
                        length and actual number of elements");
        }

        // Mark the chunk as in use.
        current_buffer
            .next_index
            .store(index + occupied_len, Ordering::SeqCst);
        chunks_in_use.push(ActualBufferChunk {
                               index,
                               len: occupied_len,
                               num_cpu_accesses: 1,
                               num_gpu_accesses: 0,
                           });

        Ok(CpuBufferPoolChunk {
               // TODO: remove .clone() once non-lexical borrows land
               buffer: current_buffer.clone(),
               index: index,
               align_offset,
               requested_len,
               marker: PhantomData,
           })
    }
}

// Can't automatically derive `Clone`, otherwise the compiler adds a `T: Clone` requirement.
impl<T, A> Clone for CpuBufferPool<T, A>
    where A: MemoryPool + Clone
{
    fn clone(&self) -> Self {
        let buf = self.current_buffer.lock().unwrap();

        CpuBufferPool {
            device: self.device.clone(),
            pool: self.pool.clone(),
            current_buffer: Mutex::new(buf.clone()),
            usage: self.usage.clone(),
            marker: PhantomData,
        }
    }
}

unsafe impl<T, A> DeviceOwned for CpuBufferPool<T, A>
    where A: MemoryPool
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl<T, A> Clone for CpuBufferPoolChunk<T, A>
    where A: MemoryPool
{
    fn clone(&self) -> CpuBufferPoolChunk<T, A> {
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

unsafe impl<T, A> BufferAccess for CpuBufferPoolChunk<T, A>
    where A: MemoryPool
{
    #[inline]
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.buffer.inner,
            offset: self.index * mem::size_of::<T>() + self.align_offset,
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.requested_len * mem::size_of::<T>()
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
        (
            self.buffer.inner.key(),
            // ensure the special cased empty buffers don't collide with a regular buffer starting at 0
            if self.requested_len == 0 { usize::max_value() } else { self.index }
        )
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, _: &Queue) -> Result<(), AccessError> {
        if self.requested_len == 0 {
            return Ok(());
        }

        let mut chunks_in_use_lock = self.buffer.chunks_in_use.lock().unwrap();
        let chunk = chunks_in_use_lock
            .iter_mut()
            .find(|c| c.index == self.index)
            .unwrap();

        if chunk.num_gpu_accesses != 0 {
            return Err(AccessError::AlreadyInUse);
        }

        chunk.num_gpu_accesses = 1;
        Ok(())
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        if self.requested_len == 0 {
            return;
        }

        let mut chunks_in_use_lock = self.buffer.chunks_in_use.lock().unwrap();
        let chunk = chunks_in_use_lock
            .iter_mut()
            .find(|c| c.index == self.index)
            .unwrap();

        debug_assert!(chunk.num_gpu_accesses >= 1);
        chunk.num_gpu_accesses = chunk
            .num_gpu_accesses
            .checked_add(1)
            .expect("Overflow in GPU usages");
    }

    #[inline]
    unsafe fn unlock(&self) {
        if self.requested_len == 0 {
            return;
        }

        let mut chunks_in_use_lock = self.buffer.chunks_in_use.lock().unwrap();
        let chunk = chunks_in_use_lock
            .iter_mut()
            .find(|c| c.index == self.index)
            .unwrap();

        debug_assert!(chunk.num_gpu_accesses >= 1);
        chunk.num_gpu_accesses -= 1;
    }
}

impl<T, A> Drop for CpuBufferPoolChunk<T, A>
    where A: MemoryPool
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
            debug_assert_eq!(chunks_in_use_lock[chunk_num].num_gpu_accesses, 0);
            chunks_in_use_lock.remove(chunk_num);
        }
    }
}

unsafe impl<T, A> TypedBufferAccess for CpuBufferPoolChunk<T, A>
    where A: MemoryPool
{
    type Content = [T];
}

unsafe impl<T, A> DeviceOwned for CpuBufferPoolChunk<T, A>
    where A: MemoryPool
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.inner.device()
    }
}

impl<T, A> PartialEq for CpuBufferPoolChunk<T, A>
    where A: MemoryPool
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T, A> Eq for CpuBufferPoolChunk<T, A>
    where A: MemoryPool
{}

impl<T, A> Hash for CpuBufferPoolChunk<T, A>
    where A: MemoryPool
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

impl<T, A> Clone for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    fn clone(&self) -> CpuBufferPoolSubbuffer<T, A> {
        CpuBufferPoolSubbuffer { chunk: self.chunk.clone() }
    }
}

unsafe impl<T, A> BufferAccess for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    #[inline]
    fn inner(&self) -> BufferInner {
        self.chunk.inner()
    }

    #[inline]
    fn size(&self) -> usize {
        self.chunk.size()
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
        self.chunk.conflict_key()
    }

    #[inline]
    fn try_gpu_lock(&self, e: bool, q: &Queue) -> Result<(), AccessError> {
        self.chunk.try_gpu_lock(e, q)
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        self.chunk.increase_gpu_lock()
    }

    #[inline]
    unsafe fn unlock(&self) {
        self.chunk.unlock()
    }
}

unsafe impl<T, A> TypedBufferAccess for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    type Content = T;
}

unsafe impl<T, A> DeviceOwned for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.chunk.buffer.inner.device()
    }
}

impl<T, A> PartialEq for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T, A> Eq for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{}

impl<T, A> Hash for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use buffer::CpuBufferPool;
    use std::mem;

    #[test]
    fn basic_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = CpuBufferPool::<u8>::upload(device);
    }

    #[test]
    fn reserve() {
        let (device, _) = gfx_dev_and_queue!();

        let pool = CpuBufferPool::<u8>::upload(device);
        assert_eq!(pool.capacity(), 0);

        pool.reserve(83).unwrap();
        assert_eq!(pool.capacity(), 83);
    }

    #[test]
    fn capacity_increase() {
        let (device, _) = gfx_dev_and_queue!();

        let pool = CpuBufferPool::upload(device);
        assert_eq!(pool.capacity(), 0);

        pool.next(12).unwrap();
        let first_cap = pool.capacity();
        assert!(first_cap >= 1);

        for _ in 0 .. first_cap + 5 {
            mem::forget(pool.next(12).unwrap());
        }

        assert!(pool.capacity() > first_cap);
    }

    #[test]
    fn reuse_subbuffers() {
        let (device, _) = gfx_dev_and_queue!();

        let pool = CpuBufferPool::upload(device);
        assert_eq!(pool.capacity(), 0);

        let mut capacity = None;
        for _ in 0 .. 64 {
            pool.next(12).unwrap();

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

        let pool = CpuBufferPool::<u8>::upload(device);
        pool.reserve(5).unwrap();

        let a = pool.chunk(vec![0, 0]).unwrap();
        let b = pool.chunk(vec![0, 0]).unwrap();
        assert_eq!(b.index, 2);
        drop(a);

        let c = pool.chunk(vec![0, 0]).unwrap();
        assert_eq!(c.index, 0);

        assert_eq!(pool.capacity(), 5);
    }

    #[test]
    fn chunk_0_elems_doesnt_pollute() {
        let (device, _) = gfx_dev_and_queue!();

        let pool = CpuBufferPool::<u8>::upload(device);

        let _ = pool.chunk(vec![]).unwrap();
        let _ = pool.chunk(vec![0, 0]).unwrap();
    }
}
