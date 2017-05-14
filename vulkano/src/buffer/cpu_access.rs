// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
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

use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;
use std::sync::TryLockError;
use smallvec::SmallVec;

use buffer::sys::BufferCreationError;
use buffer::sys::SparseLevel;
use buffer::sys::UnsafeBuffer;
use buffer::BufferUsage;
use buffer::traits::BufferAccess;
use buffer::traits::BufferInner;
use buffer::traits::Buffer;
use buffer::traits::TypedBuffer;
use buffer::traits::TypedBufferAccess;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use instance::QueueFamily;
use memory::Content;
use memory::CpuAccess as MemCpuAccess;
use memory::pool::AllocLayout;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdMemoryPoolAlloc;
use sync::AccessError;
use sync::Sharing;

use OomError;

/// Buffer whose content is accessible by the CPU.
#[derive(Debug)]
pub struct CpuAccessibleBuffer<T: ?Sized, A = StdMemoryPoolAlloc> {
    // Inner content.
    inner: UnsafeBuffer,

    // The memory held by the buffer.
    memory: A,

    // Access pattern of the buffer. Can be read-locked for a shared CPU access, or write-locked
    // for either a write CPU access or a GPU access.
    access: RwLock<()>,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

impl<T> CpuAccessibleBuffer<T> {
    /// Deprecated. Use `from_data` instead.
    #[deprecated]
    #[inline]
    pub fn new<'a, I>(device: Arc<Device>, usage: BufferUsage, queue_families: I)
                      -> Result<Arc<CpuAccessibleBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            CpuAccessibleBuffer::raw(device, mem::size_of::<T>(), usage, queue_families)
        }
    }

    /// Builds a new buffer with some data in it. Only allowed for sized data.
    pub fn from_data<'a, I>(device: Arc<Device>, usage: BufferUsage, queue_families: I, data: T)
                            -> Result<Arc<CpuAccessibleBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>,
              T: 'static,      // TODO: think about this bound
    {
        unsafe {
            let uninitialized = try!(
                CpuAccessibleBuffer::raw(device, mem::size_of::<T>(), usage, queue_families)
            );

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
    // TODO: take Arc<Device> by value
    #[inline]
    pub unsafe fn uninitialized<'a, I>(device: Arc<Device>, usage: BufferUsage, queue_families: I)
                                       -> Result<Arc<CpuAccessibleBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        CpuAccessibleBuffer::raw(device, mem::size_of::<T>(), usage, queue_families)
    }
}

impl<T> CpuAccessibleBuffer<[T]> {
    /// Builds a new buffer that contains an array `T`. The initial data comes from an iterator
    /// that produces that list of Ts.
    pub fn from_iter<'a, I, Q>(device: Arc<Device>, usage: BufferUsage, queue_families: Q, data: I)
                               -> Result<Arc<CpuAccessibleBuffer<[T]>>, OomError>
        where I: ExactSizeIterator<Item = T>,
              T: 'static,      // TODO: think about this bound
              Q: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            let uninitialized = try!(
                CpuAccessibleBuffer::uninitialized_array(device, data.len(), usage, queue_families)
            );

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

    /// Deprecated. Use `uninitialized_array` or `from_iter` instead.
    // TODO: remove
    // TODO: take Arc<Device> by value
    #[inline]
    #[deprecated]
    pub fn array<'a, I>(device: Arc<Device>, len: usize, usage: BufferUsage, queue_families: I)
                      -> Result<Arc<CpuAccessibleBuffer<[T]>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            CpuAccessibleBuffer::uninitialized_array(device, len, usage, queue_families)
        }
    }

    /// Builds a new buffer. Can be used for arrays.
    // TODO: take Arc<Device> by value
    #[inline]
    pub unsafe fn uninitialized_array<'a, I>(device: Arc<Device>, len: usize, usage: BufferUsage,
                                             queue_families: I)
                                             -> Result<Arc<CpuAccessibleBuffer<[T]>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        CpuAccessibleBuffer::raw(device, len * mem::size_of::<T>(), usage, queue_families)
    }
}

impl<T: ?Sized> CpuAccessibleBuffer<T> {
    /// Builds a new buffer without checking the size.
    ///
    /// # Safety
    ///
    /// You must ensure that the size that you pass is correct for `T`.
    ///
    pub unsafe fn raw<'a, I>(device: Arc<Device>, size: usize, usage: BufferUsage, queue_families: I)
                             -> Result<Arc<CpuAccessibleBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        let queue_families = queue_families.into_iter().map(|f| f.id())
                                           .collect::<SmallVec<[u32; 4]>>();

        let (buffer, mem_reqs) = {
            let sharing = if queue_families.len() >= 2 {
                Sharing::Concurrent(queue_families.iter().cloned())
            } else {
                Sharing::Exclusive
            };

            match UnsafeBuffer::new(device.clone(), size, usage, sharing, SparseLevel::none()) {
                Ok(b) => b,
                Err(BufferCreationError::OomError(err)) => return Err(err),
                Err(_) => unreachable!()        // We don't use sparse binding, therefore the other
                                                // errors can't happen
            }
        };

        let mem_ty = device.physical_device().memory_types()
                           .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                           .filter(|t| t.is_host_visible())
                           .next().unwrap();    // Vk specs guarantee that this can't fail

        let mem = try!(MemoryPool::alloc(&Device::standard_pool(&device), mem_ty,
                                         mem_reqs.size, mem_reqs.alignment, AllocLayout::Linear));
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        debug_assert!(mem.mapped_memory().is_some());
        try!(buffer.bind_memory(mem.memory(), mem.offset()));

        Ok(Arc::new(CpuAccessibleBuffer {
            inner: buffer,
            memory: mem,
            access: RwLock::new(()),
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
        self.queue_families.iter().map(|&num| {
            self.device().physical_device().queue_family_by_id(num).unwrap()
        }).collect()
    }
}

impl<T: ?Sized, A> CpuAccessibleBuffer<T, A>
    where A: MemoryPoolAlloc,
          T: Content + 'static,      // TODO: think about this bound
{
    /// Locks the buffer in order to write its content.
    ///
    /// If the buffer is currently in use by the GPU, this function will block until either the
    /// buffer is available or the timeout is reached. A value of `0` for the timeout is valid and
    /// means that the function should never block.
    ///
    /// After this function successfully locks the buffer, any attempt to submit a command buffer
    /// that uses it will block until you unlock it.
    #[inline]
    pub fn read(&self) -> Result<ReadLock<T>, TryLockError<RwLockReadGuard<()>>> {
        let lock = try!(self.access.try_read());

        let offset = self.memory.offset();
        let range = offset .. offset + self.inner.size();

        Ok(ReadLock {
            inner: unsafe { self.memory.mapped_memory().unwrap().read_write(range) },
            lock: lock,
        })
    }

    /// Locks the buffer in order to write its content.
    ///
    /// If the buffer is currently in use by the GPU, this function will block until either the
    /// buffer is available or the timeout is reached. A value of `0` for the timeout is valid and
    /// means that the function should never block.
    ///
    /// After this function successfully locks the buffer, any attempt to submit a command buffer
    /// that uses it will block until you unlock it.
    #[inline]
    pub fn write(&self) -> Result<WriteLock<T>, TryLockError<RwLockWriteGuard<()>>> {
        let lock = try!(self.access.try_write());

        let offset = self.memory.offset();
        let range = offset .. offset + self.inner.size();

        Ok(WriteLock {
            inner: unsafe { self.memory.mapped_memory().unwrap().read_write(range) },
            lock: lock,
        })
    }
}

unsafe impl<T: ?Sized, A> Buffer for Arc<CpuAccessibleBuffer<T, A>> {
    type Access = CpuAccessibleBufferAccess<T, A>;

    #[inline]
    fn access(self) -> CpuAccessibleBufferAccess<T, A> {
        CpuAccessibleBufferAccess {
            buffer: self
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }
}

unsafe impl<T: ?Sized, A> TypedBuffer for Arc<CpuAccessibleBuffer<T, A>> {
    type Content = T;
}

unsafe impl<T: ?Sized, A> DeviceOwned for CpuAccessibleBuffer<T, A> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

/// Access to a `CpuAccessibleBuffer`.
#[derive(Debug)]
pub struct CpuAccessibleBufferAccess<T: ?Sized, A> {
    buffer: Arc<CpuAccessibleBuffer<T, A>>,
}

impl<T: ?Sized, A> Clone for CpuAccessibleBufferAccess<T, A> {
    #[inline]
    fn clone(&self) -> CpuAccessibleBufferAccess<T, A> {
        CpuAccessibleBufferAccess {
            buffer: self.buffer.clone(),
        }
    }
}

unsafe impl<T: ?Sized, A> BufferAccess for CpuAccessibleBufferAccess<T, A> {
    #[inline]
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.buffer.inner,
            offset: 0,
        }
    }

    #[inline]
    fn conflict_key(&self, self_offset: usize, self_size: usize) -> u64 {
        self.buffer.inner.key()
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive_access: bool, queue: &Queue) -> Result<(), AccessError> {
        Ok(())       // FIXME:
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        // FIXME:
    }
}

unsafe impl<T: ?Sized, A> TypedBufferAccess for CpuAccessibleBufferAccess<T, A> {
    type Content = T;
}

unsafe impl<T: ?Sized, A> Buffer for CpuAccessibleBufferAccess<T, A> {
    type Access = Self;

    #[inline]
    fn access(self) -> Self {
        self
    }

    #[inline]
    fn size(&self) -> usize {
        self.buffer.inner.size()
    }
}

unsafe impl<T: ?Sized, A> TypedBuffer for CpuAccessibleBufferAccess<T, A> {
    type Content = T;
}

unsafe impl<T: ?Sized, A> DeviceOwned for CpuAccessibleBufferAccess<T, A> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

/// Object that can be used to read or write the content of a `CpuAccessBuffer`.
///
/// Note that this object holds a rwlock read guard on the chunk. If another thread tries to access
/// this buffer's content or tries to submit a GPU command that uses this buffer, it will block.
pub struct ReadLock<'a, T: ?Sized + 'a> {
    inner: MemCpuAccess<'a, T>,
    lock: RwLockReadGuard<'a, ()>,
}

impl<'a, T: ?Sized + 'a> ReadLock<'a, T> {
    /// Makes a new `ReadLock` to access a sub-part of the current `ReadLock`.
    #[inline]
    pub fn map<U: ?Sized + 'a, F>(self, f: F) -> ReadLock<'a, U>
        where F: FnOnce(&mut T) -> &mut U
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

/// Object that can be used to read or write the content of a `CpuAccessBuffer`.
///
/// Note that this object holds a rwlock write guard on the chunk. If another thread tries to access
/// this buffer's content or tries to submit a GPU command that uses this buffer, it will block.
pub struct WriteLock<'a, T: ?Sized + 'a> {
    inner: MemCpuAccess<'a, T>,
    lock: RwLockWriteGuard<'a, ()>,
}

impl<'a, T: ?Sized + 'a> WriteLock<'a, T> {
    /// Makes a new `WriteLock` to access a sub-part of the current `WriteLock`.
    #[inline]
    pub fn map<U: ?Sized + 'a, F>(self, f: F) -> WriteLock<'a, U>
        where F: FnOnce(&mut T) -> &mut U
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

#[cfg(test)]
mod tests {
    use buffer::{CpuAccessibleBuffer, BufferUsage};

    #[test]
    fn create_empty_buffer() {
        let (device, queue) = gfx_dev_and_queue!();

        const EMPTY: [i32; 0] = [];

        let _ = CpuAccessibleBuffer::from_data(device, BufferUsage::all(), Some(queue.family()), EMPTY.iter());
    }
}
