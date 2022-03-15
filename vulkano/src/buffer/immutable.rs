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

use super::{
    sys::UnsafeBuffer, BufferAccess, BufferAccessObject, BufferContents, BufferInner, BufferUsage,
    CpuAccessibleBuffer,
};
use crate::{
    buffer::{sys::UnsafeBufferCreateInfo, BufferCreationError, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage,
        PrimaryAutoCommandBuffer, PrimaryCommandBuffer,
    },
    device::{physical::QueueFamily, Device, DeviceOwned, Queue},
    memory::{
        pool::{
            AllocFromRequirementsFilter, AllocLayout, MappingRequirement, MemoryPoolAlloc,
            PotentialDedicatedAllocation, StdMemoryPoolAlloc,
        },
        DedicatedAllocation, DeviceMemoryAllocationError, MemoryPool,
    },
    sync::{NowFuture, Sharing},
    DeviceSize,
};
use smallvec::SmallVec;
use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::size_of,
    sync::Arc,
};

/// Buffer that is written once then read for as long as it is alive.
#[derive(Debug)]
pub struct ImmutableBuffer<T, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>>
where
    T: BufferContents + ?Sized,
{
    // Inner content.
    inner: UnsafeBuffer,

    // Memory allocated for the buffer.
    memory: A,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    // Necessary to have the appropriate template parameter.
    marker: PhantomData<Box<T>>,
}

// TODO: make this prettier
type ImmutableBufferFromBufferFuture = CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>;

impl<T> ImmutableBuffer<T>
where
    T: BufferContents + ?Sized,
{
    /// Builds an `ImmutableBuffer` that copies its data from another buffer.
    ///
    /// This function returns two objects: the newly-created buffer, and a future representing
    /// the initial upload operation. In order to be allowed to use the `ImmutableBuffer`, you must
    /// either submit your operation after this future, or execute this future and wait for it to
    /// be finished before submitting your own operation.
    pub fn from_buffer<B>(
        source: Arc<B>,
        usage: BufferUsage,
        queue: Arc<Queue>,
    ) -> Result<
        (Arc<ImmutableBuffer<T>>, ImmutableBufferFromBufferFuture),
        DeviceMemoryAllocationError,
    >
    where
        B: TypedBufferAccess<Content = T> + 'static,
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

            let mut cbb = AutoCommandBufferBuilder::primary(
                source.device().clone(),
                queue.family(),
                CommandBufferUsage::MultipleSubmit,
            )?;
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

impl<T> ImmutableBuffer<T>
where
    T: BufferContents,
{
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
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    pub fn from_data(
        data: T,
        usage: BufferUsage,
        queue: Arc<Queue>,
    ) -> Result<
        (Arc<ImmutableBuffer<T>>, ImmutableBufferFromBufferFuture),
        DeviceMemoryAllocationError,
    > {
        let source = CpuAccessibleBuffer::from_data(
            queue.device().clone(),
            BufferUsage::transfer_source(),
            false,
            data,
        )?;
        ImmutableBuffer::from_buffer(source, usage, queue)
    }

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
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    #[inline]
    pub unsafe fn uninitialized(
        device: Arc<Device>,
        usage: BufferUsage,
    ) -> Result<
        (
            Arc<ImmutableBuffer<T>>,
            Arc<ImmutableBufferInitialization<T>>,
        ),
        DeviceMemoryAllocationError,
    > {
        ImmutableBuffer::raw(
            device.clone(),
            size_of::<T>() as DeviceSize,
            usage,
            device.active_queue_families(),
        )
    }
}

impl<T> ImmutableBuffer<[T]>
where
    [T]: BufferContents,
{
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `data` is empty.
    pub fn from_iter<D>(
        data: D,
        usage: BufferUsage,
        queue: Arc<Queue>,
    ) -> Result<
        (Arc<ImmutableBuffer<[T]>>, ImmutableBufferFromBufferFuture),
        DeviceMemoryAllocationError,
    >
    where
        D: IntoIterator<Item = T>,
        D::IntoIter: ExactSizeIterator,
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
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `len` is zero.
    #[inline]
    pub unsafe fn uninitialized_array(
        device: Arc<Device>,
        len: DeviceSize,
        usage: BufferUsage,
    ) -> Result<
        (
            Arc<ImmutableBuffer<[T]>>,
            Arc<ImmutableBufferInitialization<[T]>>,
        ),
        DeviceMemoryAllocationError,
    > {
        ImmutableBuffer::raw(
            device.clone(),
            len * size_of::<T>() as DeviceSize,
            usage,
            device.active_queue_families(),
        )
    }
}

impl<T> ImmutableBuffer<T>
where
    T: BufferContents + ?Sized,
{
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
    /// # Panics
    ///
    /// - Panics if `size` is zero.
    #[inline]
    pub unsafe fn raw<'a, I>(
        device: Arc<Device>,
        size: DeviceSize,
        usage: BufferUsage,
        queue_families: I,
    ) -> Result<
        (
            Arc<ImmutableBuffer<T>>,
            Arc<ImmutableBufferInitialization<T>>,
        ),
        DeviceMemoryAllocationError,
    >
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
        size: DeviceSize,
        usage: BufferUsage,
        queue_families: SmallVec<[u32; 4]>,
    ) -> Result<
        (
            Arc<ImmutableBuffer<T>>,
            Arc<ImmutableBufferInitialization<T>>,
        ),
        DeviceMemoryAllocationError,
    > {
        let buffer = match UnsafeBuffer::new(
            device.clone(),
            UnsafeBufferCreateInfo {
                sharing: if queue_families.len() >= 2 {
                    Sharing::Concurrent(queue_families.clone())
                } else {
                    Sharing::Exclusive
                },
                size,
                usage,
                ..Default::default()
            },
        ) {
            Ok(b) => b,
            Err(BufferCreationError::AllocError(err)) => return Err(err),
            Err(_) => unreachable!(), // We don't use sparse binding, therefore the other
                                      // errors can't happen
        };
        let mem_reqs = buffer.memory_requirements();

        let mem = MemoryPool::alloc_from_requirements(
            &Device::standard_pool(&device),
            &mem_reqs,
            AllocLayout::Linear,
            MappingRequirement::DoNotMap,
            Some(DedicatedAllocation::Buffer(&buffer)),
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
            marker: PhantomData,
        });

        let initialization = Arc::new(ImmutableBufferInitialization {
            buffer: final_buf.clone(),
        });

        Ok((final_buf, initialization))
    }
}

impl<T, A> ImmutableBuffer<T, A>
where
    T: BufferContents + ?Sized,
{
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

unsafe impl<T, A> BufferAccess for ImmutableBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    #[inline]
    fn inner(&self) -> BufferInner {
        BufferInner {
            buffer: &self.inner,
            offset: 0,
        }
    }

    #[inline]
    fn size(&self) -> DeviceSize {
        self.inner.size()
    }

    #[inline]
    fn conflict_key(&self) -> (u64, u64) {
        (self.inner.key(), 0)
    }
}

impl<T, A> BufferAccessObject for Arc<ImmutableBuffer<T, A>>
where
    T: BufferContents + ?Sized,
    A: Send + Sync + 'static,
{
    #[inline]
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess> {
        self.clone()
    }
}

unsafe impl<T, A> TypedBufferAccess for ImmutableBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    type Content = T;
}

unsafe impl<T, A> DeviceOwned for ImmutableBuffer<T, A>
where
    T: BufferContents + ?Sized,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl<T, A> PartialEq for ImmutableBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T, A> Eq for ImmutableBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
}

impl<T, A> Hash for ImmutableBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

/// Access to the immutable buffer that can be used for the initial upload.
#[derive(Debug)]
pub struct ImmutableBufferInitialization<T, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>>
where
    T: BufferContents + ?Sized,
{
    buffer: Arc<ImmutableBuffer<T, A>>,
}

unsafe impl<T, A> BufferAccess for ImmutableBufferInitialization<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    #[inline]
    fn inner(&self) -> BufferInner {
        self.buffer.inner()
    }

    #[inline]
    fn size(&self) -> DeviceSize {
        self.buffer.size()
    }

    #[inline]
    fn conflict_key(&self) -> (u64, u64) {
        (self.buffer.inner.key(), 0)
    }
}

impl<T, A> BufferAccessObject for Arc<ImmutableBufferInitialization<T, A>>
where
    T: BufferContents + ?Sized,
    A: Send + Sync + 'static,
{
    #[inline]
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess> {
        self.clone()
    }
}

unsafe impl<T, A> TypedBufferAccess for ImmutableBufferInitialization<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    type Content = T;
}

unsafe impl<T, A> DeviceOwned for ImmutableBufferInitialization<T, A>
where
    T: BufferContents + ?Sized,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.inner.device()
    }
}

impl<T, A> Clone for ImmutableBufferInitialization<T, A>
where
    T: BufferContents + ?Sized,
{
    #[inline]
    fn clone(&self) -> ImmutableBufferInitialization<T, A> {
        ImmutableBufferInitialization {
            buffer: self.buffer.clone(),
        }
    }
}

impl<T, A> PartialEq for ImmutableBufferInitialization<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T, A> Eq for ImmutableBufferInitialization<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
}

impl<T, A> Hash for ImmutableBufferInitialization<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use crate::buffer::cpu_access::CpuAccessibleBuffer;
    use crate::buffer::immutable::ImmutableBuffer;
    use crate::buffer::BufferUsage;
    use crate::command_buffer::AutoCommandBufferBuilder;
    use crate::command_buffer::CommandBufferUsage;
    use crate::command_buffer::PrimaryCommandBuffer;
    use crate::sync::GpuFuture;

    #[test]
    fn from_data_working() {
        let (device, queue) = gfx_dev_and_queue!();

        let (buffer, _) =
            ImmutableBuffer::from_data(12u32, BufferUsage::all(), queue.clone()).unwrap();

        let destination =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, 0).unwrap();

        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
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

        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
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
    fn init_then_read_same_cb() {
        let (device, queue) = gfx_dev_and_queue!();

        let (buffer, init) = unsafe {
            ImmutableBuffer::<u32>::uninitialized(device.clone(), BufferUsage::all()).unwrap()
        };

        let source =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, 0).unwrap();

        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
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

        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
        cbb.copy_buffer(source.clone(), init).unwrap();
        let cb1 = cbb.build().unwrap();

        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
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
    #[allow(unused)]
    fn create_buffer_zero_size_data() {
        let (device, queue) = gfx_dev_and_queue!();

        assert_should_panic!({
            ImmutableBuffer::from_data((), BufferUsage::all(), queue.clone()).unwrap();
        });
    }

    // TODO: write tons of tests that try to exploit loopholes
    // this isn't possible yet because checks aren't correctly implemented yet
}
