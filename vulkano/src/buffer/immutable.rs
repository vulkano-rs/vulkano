// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Buffer that is written once then read for as long as it is alive.
//!
//! Use this buffer when you have data that you never modify.
//!
//! Only the first ever command buffer that uses this buffer can write to it (for example by
//! copying from another buffer). Any subsequent command buffer **must** only read from the buffer,
//! or a panic will happen.
//!
//! The buffer will be stored in device-local memory if possible
//!

use smallvec::SmallVec;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::mem;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use buffer::sys::BufferCreationError;
use buffer::sys::SparseLevel;
use buffer::sys::UnsafeBuffer;
use buffer::traits::BufferAccess;
use buffer::traits::BufferInner;
use buffer::traits::TypedBufferAccess;
use buffer::BufferUsage;
use buffer::CpuAccessibleBuffer;
use command_buffer::AutoCommandBuffer;
use command_buffer::AutoCommandBufferBuilder;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferExecFuture;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use instance::QueueFamily;
use memory::pool::AllocFromRequirementsFilter;
use memory::pool::AllocLayout;
use memory::pool::MappingRequirement;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::PotentialDedicatedAllocation;
use memory::pool::StdMemoryPoolAlloc;
use memory::DedicatedAlloc;
use memory::DeviceMemoryAllocError;
use sync::AccessError;
use sync::NowFuture;
use sync::Sharing;

/// Buffer that is written once then read for as long as it is alive.
// TODO: implement Debug
pub struct ImmutableBuffer<T: ?Sized, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>> {
    // Inner content.
    inner: UnsafeBuffer,

    // Memory allocated for the buffer.
    memory: A,

    // True if the `ImmutableBufferInitialization` object was used by the GPU then dropped.
    // This means that the `ImmutableBuffer` can be used as much as we want without any restriction.
    initialized: AtomicBool,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    // Necessary to have the appropriate template parameter.
    marker: PhantomData<Box<T>>,
}

// TODO: make this prettier
type ImmutableBufferFromBufferFuture = CommandBufferExecFuture<NowFuture, AutoCommandBuffer>;

impl<T: ?Sized> ImmutableBuffer<T> {
    /// Builds an `ImmutableBuffer` from some data.
    ///
    /// This function builds a memory-mapped intermediate buffer, writes the data to it, builds a
    /// command buffer that copies from this intermediate buffer to the final buffer, and finally
    /// submits the command buffer as a future.
    ///
    /// This function returns two objects: the newly-created buffer, and a future representing
    /// the initial upload operation. In order to be allowed to use the `ImmutableBuffer`, you must
    /// either submit your operation after this future, or execute this future and wait for it to
    /// be finished before submitting your own operation.
    pub fn from_data(
        data: T,
        usage: BufferUsage,
        queue: Arc<Queue>,
    ) -> Result<(Arc<ImmutableBuffer<T>>, ImmutableBufferFromBufferFuture), DeviceMemoryAllocError>
    where
        T: 'static + Send + Sync + Sized,
    {
        let source = CpuAccessibleBuffer::from_data(
            queue.device().clone(),
            BufferUsage::transfer_source(),
            false,
            data,
        )?;
        ImmutableBuffer::from_buffer(source, usage, queue)
    }

    /// Builds an `ImmutableBuffer` that copies its data from another buffer.
    ///
    /// This function returns two objects: the newly-created buffer, and a future representing
    /// the initial upload operation. In order to be allowed to use the `ImmutableBuffer`, you must
    /// either submit your operation after this future, or execute this future and wait for it to
    /// be finished before submitting your own operation.
    pub fn from_buffer<B>(
        source: B,
        usage: BufferUsage,
        queue: Arc<Queue>,
    ) -> Result<(Arc<ImmutableBuffer<T>>, ImmutableBufferFromBufferFuture), DeviceMemoryAllocError>
    where
        B: BufferAccess + TypedBufferAccess<Content = T> + 'static + Clone + Send + Sync,
        T: 'static + Send + Sync,
    {
        unsafe {
            // We automatically set `transfer_destination` to true in order to avoid annoying errors.
            let actual_usage = BufferUsage {
                transfer_destination: true,
                ..usage
            };

            let (buffer, init) = ImmutableBuffer::raw(
                source.device().clone(),
                source.size(),
                actual_usage,
                source.device().active_queue_families(),
            )?;

            let mut cbb = AutoCommandBufferBuilder::new(source.device().clone(), queue.family())?;
            cbb.copy_buffer(source, init).unwrap(); // TODO: return error?
            let cb = cbb.build().unwrap(); // TODO: return OomError

            let future = match cb.execute(queue) {
                Ok(f) => f,
                Err(_) => unreachable!(),
            };

            Ok((buffer, future))
        }
    }
}

impl<T> ImmutableBuffer<T> {
    /// Builds a new buffer with uninitialized data. Only allowed for sized data.
    ///
    /// Returns two things: the buffer, and a special access that should be used for the initial
    /// upload to the buffer.
    ///
    /// You will get an error if you try to use the buffer before using the initial upload access.
    /// However this function doesn't check whether you actually used this initial upload to fill
    /// the buffer like you're supposed to do.
    ///
    /// You will also get an error if you try to get exclusive access to the final buffer.
    ///
    /// # Safety
    ///
    /// - The `ImmutableBufferInitialization` should be used to fill the buffer with some initial
    ///   data, otherwise the content is undefined.
    ///
    #[inline]
    pub unsafe fn uninitialized(
        device: Arc<Device>,
        usage: BufferUsage,
    ) -> Result<(Arc<ImmutableBuffer<T>>, ImmutableBufferInitialization<T>), DeviceMemoryAllocError>
    {
        ImmutableBuffer::raw(
            device.clone(),
            mem::size_of::<T>(),
            usage,
            device.active_queue_families(),
        )
    }
}

impl<T> ImmutableBuffer<[T]> {
    pub fn from_iter<D>(
        data: D,
        usage: BufferUsage,
        queue: Arc<Queue>,
    ) -> Result<(Arc<ImmutableBuffer<[T]>>, ImmutableBufferFromBufferFuture), DeviceMemoryAllocError>
    where
        D: ExactSizeIterator<Item = T>,
        T: 'static + Send + Sync + Sized,
    {
        let source = CpuAccessibleBuffer::from_iter(
            queue.device().clone(),
            BufferUsage::transfer_source(),
            false,
            data,
        )?;
        ImmutableBuffer::from_buffer(source, usage, queue)
    }

    /// Builds a new buffer with uninitialized data. Can be used for arrays.
    ///
    /// Returns two things: the buffer, and a special access that should be used for the initial
    /// upload to the buffer.
    ///
    /// You will get an error if you try to use the buffer before using the initial upload access.
    /// However this function doesn't check whether you actually used this initial upload to fill
    /// the buffer like you're supposed to do.
    ///
    /// You will also get an error if you try to get exclusive access to the final buffer.
    ///
    /// # Safety
    ///
    /// - The `ImmutableBufferInitialization` should be used to fill the buffer with some initial
    ///   data, otherwise the content is undefined.
    ///
    #[inline]
    pub unsafe fn uninitialized_array(
        device: Arc<Device>,
        len: usize,
        usage: BufferUsage,
    ) -> Result<
        (
            Arc<ImmutableBuffer<[T]>>,
            ImmutableBufferInitialization<[T]>,
        ),
        DeviceMemoryAllocError,
    > {
        ImmutableBuffer::raw(
            device.clone(),
            len * mem::size_of::<T>(),
            usage,
            device.active_queue_families(),
        )
    }
}

impl<T: ?Sized> ImmutableBuffer<T> {
    /// Builds a new buffer without checking the size and granting free access for the initial
    /// upload.
    ///
    /// Returns two things: the buffer, and a special access that should be used for the initial
    /// upload to the buffer.
    /// You will get an error if you try to use the buffer before using the initial upload access.
    /// However this function doesn't check whether you used this initial upload to fill the buffer.
    /// You will also get an error if you try to get exclusive access to the final buffer.
    ///
    /// # Safety
    ///
    /// - You must ensure that the size that you pass is correct for `T`.
    /// - The `ImmutableBufferInitialization` should be used to fill the buffer with some initial
    ///   data.
    ///
    #[inline]
    pub unsafe fn raw<'a, I>(
        device: Arc<Device>,
        size: usize,
        usage: BufferUsage,
        queue_families: I,
    ) -> Result<(Arc<ImmutableBuffer<T>>, ImmutableBufferInitialization<T>), DeviceMemoryAllocError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        let queue_families = queue_families.into_iter().map(|f| f.id()).collect();
        ImmutableBuffer::raw_impl(device, size, usage, queue_families)
    }

    // Internal implementation of `raw`. This is separated from `raw` so that it doesn't need to be
    // inlined.
    unsafe fn raw_impl(
        device: Arc<Device>,
        size: usize,
        usage: BufferUsage,
        queue_families: SmallVec<[u32; 4]>,
    ) -> Result<(Arc<ImmutableBuffer<T>>, ImmutableBufferInitialization<T>), DeviceMemoryAllocError>
    {
        let (buffer, mem_reqs) = {
            let sharing = if queue_families.len() >= 2 {
                Sharing::Concurrent(queue_families.iter().cloned())
            } else {
                Sharing::Exclusive
            };

            match UnsafeBuffer::new(device.clone(), size, usage, sharing, SparseLevel::none()) {
                Ok(b) => b,
                Err(BufferCreationError::AllocError(err)) => return Err(err),
                Err(_) => unreachable!(), // We don't use sparse binding, therefore the other
                                          // errors can't happen
            }
        };

        let mem = MemoryPool::alloc_from_requirements(
            &Device::standard_pool(&device),
            &mem_reqs,
            AllocLayout::Linear,
            MappingRequirement::DoNotMap,
            DedicatedAlloc::Buffer(&buffer),
            |t| {
                if t.is_device_local() {
                    AllocFromRequirementsFilter::Preferred
                } else {
                    AllocFromRequirementsFilter::Allowed
                }
            },
        )?;
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        buffer.bind_memory(mem.memory(), mem.offset())?;

        let final_buf = Arc::new(ImmutableBuffer {
            inner: buffer,
            memory: mem,
            queue_families: queue_families,
            initialized: AtomicBool::new(false),
            marker: PhantomData,
        });

        let initialization = ImmutableBufferInitialization {
            buffer: final_buf.clone(),
            used: Arc::new(AtomicBool::new(false)),
        };

        Ok((final_buf, initialization))
    }
}

impl<T: ?Sized, A> ImmutableBuffer<T, A> {
    /// Returns the device used to create this buffer.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }

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

unsafe impl<T: ?Sized, A> BufferAccess for ImmutableBuffer<T, A> {
    #[inline]
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.inner,
            offset: 0,
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
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
        (self.inner.key(), 0)
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive_access: bool, _: &Queue) -> Result<(), AccessError> {
        if exclusive_access {
            return Err(AccessError::ExclusiveDenied);
        }

        if !self.initialized.load(Ordering::Relaxed) {
            return Err(AccessError::BufferNotInitialized);
        }

        Ok(())
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {}

    #[inline]
    unsafe fn unlock(&self) {}
}

unsafe impl<T: ?Sized, A> TypedBufferAccess for ImmutableBuffer<T, A> {
    type Content = T;
}

unsafe impl<T: ?Sized, A> DeviceOwned for ImmutableBuffer<T, A> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl<T: ?Sized, A> PartialEq for ImmutableBuffer<T, A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T: ?Sized, A> Eq for ImmutableBuffer<T, A> {}

impl<T: ?Sized, A> Hash for ImmutableBuffer<T, A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

/// Access to the immutable buffer that can be used for the initial upload.
//#[derive(Debug)]      // TODO:
pub struct ImmutableBufferInitialization<
    T: ?Sized,
    A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>,
> {
    buffer: Arc<ImmutableBuffer<T, A>>,
    used: Arc<AtomicBool>,
}

unsafe impl<T: ?Sized, A> BufferAccess for ImmutableBufferInitialization<T, A> {
    #[inline]
    fn inner(&self) -> BufferInner {
        self.buffer.inner()
    }

    #[inline]
    fn size(&self) -> usize {
        self.buffer.size()
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
        (self.buffer.inner.key(), 0)
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, _: &Queue) -> Result<(), AccessError> {
        if self.buffer.initialized.load(Ordering::Relaxed) {
            return Err(AccessError::AlreadyInUse);
        }

        if !self.used.compare_and_swap(false, true, Ordering::Relaxed) {
            Ok(())
        } else {
            Err(AccessError::AlreadyInUse)
        }
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        debug_assert!(self.used.load(Ordering::Relaxed));
    }

    #[inline]
    unsafe fn unlock(&self) {
        self.buffer.initialized.store(true, Ordering::Relaxed);
    }
}

unsafe impl<T: ?Sized, A> TypedBufferAccess for ImmutableBufferInitialization<T, A> {
    type Content = T;
}

unsafe impl<T: ?Sized, A> DeviceOwned for ImmutableBufferInitialization<T, A> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.inner.device()
    }
}

impl<T: ?Sized, A> Clone for ImmutableBufferInitialization<T, A> {
    #[inline]
    fn clone(&self) -> ImmutableBufferInitialization<T, A> {
        ImmutableBufferInitialization {
            buffer: self.buffer.clone(),
            used: self.used.clone(),
        }
    }
}

impl<T: ?Sized, A> PartialEq for ImmutableBufferInitialization<T, A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T: ?Sized, A> Eq for ImmutableBufferInitialization<T, A> {}

impl<T: ?Sized, A> Hash for ImmutableBufferInitialization<T, A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use buffer::cpu_access::CpuAccessibleBuffer;
    use buffer::immutable::ImmutableBuffer;
    use buffer::BufferUsage;
    use command_buffer::AutoCommandBufferBuilder;
    use command_buffer::CommandBuffer;
    use sync::GpuFuture;

    #[test]
    fn from_data_working() {
        let (device, queue) = gfx_dev_and_queue!();

        let (buffer, _) =
            ImmutableBuffer::from_data(12u32, BufferUsage::all(), queue.clone()).unwrap();

        let destination =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, 0).unwrap();

        let mut cbb = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
        cbb.copy_buffer(buffer, destination.clone()).unwrap();
        let _ = cbb
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        let destination_content = destination.read().unwrap();
        assert_eq!(*destination_content, 12);
    }

    #[test]
    fn from_iter_working() {
        let (device, queue) = gfx_dev_and_queue!();

        let (buffer, _) = ImmutableBuffer::from_iter(
            (0..512u32).map(|n| n * 2),
            BufferUsage::all(),
            queue.clone(),
        )
        .unwrap();

        let destination = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            (0..512).map(|_| 0u32),
        )
        .unwrap();

        let mut cbb = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
        cbb.copy_buffer(buffer, destination.clone()).unwrap();
        let _ = cbb
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        let destination_content = destination.read().unwrap();
        for (n, &v) in destination_content.iter().enumerate() {
            assert_eq!(n * 2, v as usize);
        }
    }

    #[test]
    fn writing_forbidden() {
        let (device, queue) = gfx_dev_and_queue!();

        let (buffer, _) =
            ImmutableBuffer::from_data(12u32, BufferUsage::all(), queue.clone()).unwrap();

        assert_should_panic!({
            // TODO: check Result error instead of panicking
            let mut cbb = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
            cbb.fill_buffer(buffer, 50).unwrap();
            let _ = cbb
                .build()
                .unwrap()
                .execute(queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap();
        });
    }

    #[test]
    fn read_uninitialized_forbidden() {
        let (device, queue) = gfx_dev_and_queue!();

        let (buffer, _) = unsafe {
            ImmutableBuffer::<u32>::uninitialized(device.clone(), BufferUsage::all()).unwrap()
        };

        let source =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, 0).unwrap();

        assert_should_panic!({
            // TODO: check Result error instead of panicking
            let mut cbb = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
            cbb.copy_buffer(source, buffer).unwrap();
            let _ = cbb
                .build()
                .unwrap()
                .execute(queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap();
        });
    }

    #[test]
    fn init_then_read_same_cb() {
        let (device, queue) = gfx_dev_and_queue!();

        let (buffer, init) = unsafe {
            ImmutableBuffer::<u32>::uninitialized(device.clone(), BufferUsage::all()).unwrap()
        };

        let source =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, 0).unwrap();

        let mut cbb = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
        cbb.copy_buffer(source.clone(), init)
            .unwrap()
            .copy_buffer(buffer, source.clone())
            .unwrap();
        let _ = cbb
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
    }

    #[test]
    #[ignore] // TODO: doesn't work because the submit sync layer isn't properly implemented
    fn init_then_read_same_future() {
        let (device, queue) = gfx_dev_and_queue!();

        let (buffer, init) = unsafe {
            ImmutableBuffer::<u32>::uninitialized(device.clone(), BufferUsage::all()).unwrap()
        };

        let source =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, 0).unwrap();

        let mut cbb = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
        cbb.copy_buffer(source.clone(), init).unwrap();
        let cb1 = cbb.build().unwrap();

        let mut cbb = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
        cbb.copy_buffer(buffer, source.clone()).unwrap();
        let cb2 = cbb.build().unwrap();

        let _ = cb1
            .execute(queue.clone())
            .unwrap()
            .then_execute(queue.clone(), cb2)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
    }

    #[test]
    fn create_buffer_zero_size_data() {
        let (device, queue) = gfx_dev_and_queue!();

        let _ = ImmutableBuffer::from_data((), BufferUsage::all(), queue.clone());
    }

    // TODO: write tons of tests that try to exploit loopholes
    // this isn't possible yet because checks aren't correctly implemented yet
}
