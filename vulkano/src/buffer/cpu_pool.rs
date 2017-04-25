// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use smallvec::SmallVec;

use buffer::sys::BufferCreationError;
use buffer::sys::SparseLevel;
use buffer::sys::UnsafeBuffer;
use buffer::sys::Usage;
use buffer::traits::BufferAccess;
use buffer::traits::BufferInner;
use buffer::traits::Buffer;
use buffer::traits::TypedBufferAccess;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use instance::QueueFamily;
use memory::pool::AllocLayout;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdMemoryPool;
use sync::Sharing;

use OomError;

/// Buffer from which "sub-buffers" of fixed size can be individually allocated.
///
/// This buffer is especially suitable when you want to upload or download some data at each frame.
///
/// # Usage
///
/// A `CpuBufferPool` is a bit similar to a `Vec`. You start by creating an empty pool, then you
/// grab elements from the pool and use them, and if the pool is full it will automatically grow
/// in size.
///
/// But contrary to a `Vec`, elements automatically free themselves when they are dropped (ie.
/// usually when they are no longer in use by the GPU).
///
/// # Arc-like
///
/// The `CpuBufferPool` struct internally contains an `Arc`. You can clone the `CpuBufferPool` for
/// a cheap cost, and all the clones will share the same underlying buffer.
///
pub struct CpuBufferPool<T: ?Sized, A = Arc<StdMemoryPool>> where A: MemoryPool {
    // The device of the pool.
    device: Arc<Device>,

    // The memory pool to use for allocations.
    pool: A,

    // Current buffer from which subbuffers are grabbed.
    current_buffer: Mutex<Option<Arc<ActualBuffer<A>>>>,

    // Size in bytes of one subbuffer.
    one_size: usize,

    // Buffer usage.
    usage: Usage,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

// One buffer of the pool.
struct ActualBuffer<A> where A: MemoryPool {
    // Inner content.
    inner: UnsafeBuffer,

    // The memory held by the buffer.
    memory: A::Alloc,

    // Access pattern of the subbuffers.
    subbuffers: Vec<ActualBufferSubbuffer>,

    // The subbuffer that should be available next.
    next_subbuffer: AtomicUsize,

    // Number of subbuffers in the buffer.
    capacity: usize,
}

// Access pattern of one subbuffer.
#[derive(Debug)]
struct ActualBufferSubbuffer {
    // Number of `CpuBufferPoolSubbuffer` objects that point to this subbuffer.
    num_cpu_accesses: AtomicUsize,

    // Number of `CpuBufferPoolSubbuffer` objects that point to this subbuffer and that have been
    // GPU-locked.
    num_gpu_accesses: AtomicUsize,
}

/// A subbuffer allocated from a `CpuBufferPool`.
///
/// When this object is destroyed, the subbuffer is automatically reclaimed by the pool.
pub struct CpuBufferPoolSubbuffer<T: ?Sized, A> where A: MemoryPool {
    buffer: Arc<ActualBuffer<A>>,

    // Index of the subbuffer within `buffer`.
    subbuffer_index: usize,

    // Size in bytes of the subbuffer.
    size: usize,

    // Whether this subbuffer was locked on the GPU.
    // If true, then num_gpu_accesses must be decreased.
    gpu_locked: AtomicBool,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

impl<T> CpuBufferPool<T> {
    #[inline]
    pub fn new<'a, I>(device: Arc<Device>, usage: &Usage, queue_families: I)
                      -> CpuBufferPool<T>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            CpuBufferPool::raw(device, mem::size_of::<T>(), usage, queue_families)
        }
    }

    /// Builds a `CpuBufferPool` meant for simple uploads.
    ///
    /// Shortcut for a pool that can only be used as transfer sources and with exclusive queue
    /// family accesses.
    #[inline]
    pub fn upload(device: Arc<Device>) -> CpuBufferPool<T> {
        CpuBufferPool::new(device, &Usage::transfer_source(), iter::empty())
    }
}

impl<T> CpuBufferPool<[T]> {
    #[inline]
    pub fn array<'a, I>(device: Arc<Device>, len: usize, usage: &Usage, queue_families: I)
                      -> CpuBufferPool<[T]>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            CpuBufferPool::raw(device, mem::size_of::<T>() * len, usage, queue_families)
        }
    }
}

impl<T: ?Sized> CpuBufferPool<T> {
    pub unsafe fn raw<'a, I>(device: Arc<Device>, one_size: usize,
                             usage: &Usage, queue_families: I) -> CpuBufferPool<T>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        let queue_families = queue_families.into_iter().map(|f| f.id())
                                           .collect::<SmallVec<[u32; 4]>>();

        let pool = Device::standard_pool(&device);

        CpuBufferPool {
            device: device,
            pool: pool,
            current_buffer: Mutex::new(None),
            one_size: one_size,
            usage: usage.clone(),
            queue_families: queue_families,
            marker: PhantomData,
        }
    }

    /// Returns the current capacity of the pool.
    pub fn capacity(&self) -> usize {
        match *self.current_buffer.lock().unwrap() {
            None => 0,
            Some(ref buf) => buf.capacity,
        }
    }
}

impl<T, A> CpuBufferPool<T, A> where A: MemoryPool, T: 'static {
    /// Sets the capacity to `capacity`, or does nothing if the capacity is already higher.
    ///
    /// Since this can involve a memory allocation, an `OomError` can happen.
    pub fn reserve(&self, capacity: usize) -> Result<(), OomError> {
        let mut cur_buf = self.current_buffer.lock().unwrap();

        // Check current capacity.
        match *cur_buf {
            Some(ref buf) if buf.capacity >= capacity => {
                return Ok(())
            },
            _ => ()
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
    pub fn next(&self, data: T) -> CpuBufferPoolSubbuffer<T, A> {
        let mut mutex = self.current_buffer.lock().unwrap();

        let data = match self.try_next_impl(&mut mutex, data) {
            Ok(n) => return n,
            Err(d) => d,
        };

        let next_capacity = match *mutex {
            Some(ref b) => b.capacity * 2,
            None => 3,
        };

        self.reset_buf(&mut mutex, next_capacity).unwrap();        /* FIXME: error */

        match self.try_next_impl(&mut mutex, data) {
            Ok(n) => n,
            Err(_) => unreachable!()
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
        self.try_next_impl(&mut mutex, data).ok()
    }

    // Creates a new buffer and sets it as current.
    fn reset_buf(&self, cur_buf_mutex: &mut MutexGuard<Option<Arc<ActualBuffer<A>>>>, capacity: usize) -> Result<(), OomError> {
        unsafe {
            let (buffer, mem_reqs) = {
                let sharing = if self.queue_families.len() >= 2 {
                    Sharing::Concurrent(self.queue_families.iter().cloned())
                } else {
                    Sharing::Exclusive
                };

                let total_size = match self.one_size.checked_mul(capacity) {
                    Some(s) => s,
                    None => return Err(OomError::OutOfDeviceMemory),
                };

                match UnsafeBuffer::new(&self.device, total_size, &self.usage, sharing, SparseLevel::none()) {
                    Ok(b) => b,
                    Err(BufferCreationError::OomError(err)) => return Err(err),
                    Err(_) => unreachable!()        // We don't use sparse binding, therefore the other
                                                    // errors can't happen
                }
            };

            let mem_ty = self.device.physical_device().memory_types()
                            .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                            .filter(|t| t.is_host_visible())
                            .next().unwrap();    // Vk specs guarantee that this can't fail

            let mem = try!(MemoryPool::alloc(&self.pool, mem_ty,
                                            mem_reqs.size, mem_reqs.alignment, AllocLayout::Linear));
            debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
            debug_assert!(mem.mapped_memory().is_some());
            try!(buffer.bind_memory(mem.memory(), mem.offset()));

            **cur_buf_mutex = Some(Arc::new(ActualBuffer {
                inner: buffer,
                memory: mem,
                subbuffers: {
                    let mut v = Vec::with_capacity(capacity);
                    for _ in 0 .. capacity {
                        v.push(ActualBufferSubbuffer {
                            num_cpu_accesses: AtomicUsize::new(0),
                            num_gpu_accesses: AtomicUsize::new(0),
                         });
                    }
                    v
                },
                capacity: capacity,
                next_subbuffer: AtomicUsize::new(0),
            }));

            Ok(())
        }
    }

    // Tries to lock a subbuffer from the current buffer.
    fn try_next_impl(&self, cur_buf_mutex: &mut MutexGuard<Option<Arc<ActualBuffer<A>>>>, data: T)
                     -> Result<CpuBufferPoolSubbuffer<T, A>, T>
    {
        // Grab the current buffer. Return `Err` if the pool wasn't "initialized" yet.
        let current_buffer = match cur_buf_mutex.clone() {
            Some(b) => b,
            None => return Err(data)
        };

        // Grab the next subbuffer to use.
        let next_subbuffer = {
            // Since the only place that touches `next_subbuffer` is this code, and since we own a
            // mutex lock to the buffer, it means that `next_subbuffer` can't be accessed
            // concurrently.
            let val = current_buffer.next_subbuffer.fetch_add(1, Ordering::Relaxed);
            // TODO: handle overflows?
            // TODO: rewrite this in a proper way by holding an intermediary struct in the mutex instead of the Arc directly
            val % current_buffer.capacity
        };

        // Check if subbuffer is already taken. If so, the pool is full.
        if current_buffer.subbuffers[next_subbuffer].num_cpu_accesses.compare_and_swap(0, 1, Ordering::SeqCst) != 0 {
            return Err(data);
        }

        // Reset num_gpu_accesses.
        current_buffer.subbuffers[next_subbuffer].num_gpu_accesses.store(0, Ordering::SeqCst);

        // Write `data` in the memory.
        unsafe {
            let range = (next_subbuffer * self.one_size) .. ((next_subbuffer + 1) * self.one_size);
            let mut mapping = current_buffer.memory.mapped_memory().unwrap().read_write(range);
            *mapping = data;
        }

        Ok(CpuBufferPoolSubbuffer {
            buffer: current_buffer,
            subbuffer_index: next_subbuffer,
            gpu_locked: AtomicBool::new(false),
            size: self.one_size,
            marker: PhantomData,
        })
    }
}

// Can't automatically derive `Clone`, otherwise the compiler adds a `T: Clone` requirement.
impl<T: ?Sized, A> Clone for CpuBufferPool<T, A> where A: MemoryPool + Clone {
    fn clone(&self) -> Self {
        let buf = self.current_buffer.lock().unwrap();

        CpuBufferPool {
            device: self.device.clone(),
            pool: self.pool.clone(),
            current_buffer: Mutex::new(buf.clone()),
            one_size: self.one_size,
            usage: self.usage.clone(),
            queue_families: self.queue_families.clone(),
            marker: PhantomData,
        }
    }
}

unsafe impl<T: ?Sized, A> DeviceOwned for CpuBufferPool<T, A>
    where A: MemoryPool
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<T: ?Sized, A> Buffer for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    type Access = Self;

    #[inline]
    fn access(self) -> Self {
        self
    }

    #[inline]
    fn size(&self) -> usize {
        self.size
    }
}

impl<T: ?Sized, A> Clone for CpuBufferPoolSubbuffer<T, A> where A: MemoryPool {
    fn clone(&self) -> CpuBufferPoolSubbuffer<T, A> {
        let old_val = self.buffer.subbuffers[self.subbuffer_index].num_cpu_accesses.fetch_add(1, Ordering::SeqCst);
        debug_assert!(old_val >= 1);

        CpuBufferPoolSubbuffer {
            buffer: self.buffer.clone(),
            subbuffer_index: self.subbuffer_index,
            gpu_locked: AtomicBool::new(false),
            size: self.size,
            marker: PhantomData,
        }
    }
}

unsafe impl<T: ?Sized, A> BufferAccess for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    #[inline]
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.buffer.inner,
            offset: self.subbuffer_index * self.size,
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn conflict_key(&self, self_offset: usize, self_size: usize) -> u64 {
        self.buffer.inner.key() + self.subbuffer_index as u64
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, _: &Queue) -> bool {
        let in_use = &self.buffer.subbuffers[self.subbuffer_index].num_gpu_accesses;
        if in_use.compare_and_swap(0, 1, Ordering::SeqCst) != 0 {
            return false;
        }

        let was_locked = self.gpu_locked.swap(true, Ordering::SeqCst);
        debug_assert!(!was_locked);
        true
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        let was_locked = self.gpu_locked.swap(true, Ordering::SeqCst);
        debug_assert!(!was_locked);

        let in_use = &self.buffer.subbuffers[self.subbuffer_index];
        let num_usages = in_use.num_gpu_accesses.fetch_add(1, Ordering::SeqCst);
        debug_assert!(num_usages >= 1);
        debug_assert!(num_usages <= in_use.num_cpu_accesses.load(Ordering::SeqCst));
    }
}

unsafe impl<T: ?Sized, A> TypedBufferAccess for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool, T: 'static + Copy + Clone
{
    type Content = T;
}

unsafe impl<T: ?Sized, A> DeviceOwned for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.inner.device()
    }
}

impl<T: ?Sized, A> Drop for CpuBufferPoolSubbuffer<T, A>
    where A: MemoryPool
{
    #[inline]
    fn drop(&mut self) {
        let in_use = &self.buffer.subbuffers[self.subbuffer_index];
        let prev_val = in_use.num_cpu_accesses.fetch_sub(1, Ordering::SeqCst);
        debug_assert!(prev_val >= 1);

        if self.gpu_locked.load(Ordering::SeqCst) {
            let was_in_use = in_use.num_gpu_accesses.fetch_sub(1, Ordering::SeqCst);
            debug_assert!(was_in_use >= 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem;
    use buffer::CpuBufferPool;

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

        pool.next(12);
        let first_cap = pool.capacity();
        assert!(first_cap >= 1);

        for _ in 0 .. first_cap + 5 {
            mem::forget(pool.next(12));
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
            pool.next(12);

            let new_cap = pool.capacity();
            assert!(new_cap >= 1);
            match capacity {
                None => capacity = Some(new_cap),
                Some(c) => assert_eq!(c, new_cap),
            }
        }
    }
}
