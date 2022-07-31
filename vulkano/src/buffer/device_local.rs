// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Buffer whose content is read-written by the GPU only.
//!
//! Each access from the CPU or from the GPU locks the whole buffer for either reading or writing.
//! You can read the buffer multiple times simultaneously from multiple queues. Trying to read and
//! write simultaneously, or write and write simultaneously will block with a semaphore.
//!

use super::{
    sys::{UnsafeBuffer, UnsafeBufferCreateInfo},
    BufferAccess, BufferAccessObject, BufferContents, BufferCreationError, BufferInner,
    BufferUsage, CpuAccessibleBuffer, TypedBufferAccess,
};
use crate::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferBeginError, CommandBufferExecFuture,
        CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer, PrimaryCommandBuffer,
    },
    device::{physical::QueueFamily, Device, DeviceOwned, Queue},
    memory::{
        pool::{
            alloc_dedicated_with_exportable_fd, AllocFromRequirementsFilter, AllocLayout,
            MappingRequirement, MemoryPoolAlloc, PotentialDedicatedAllocation, StdMemoryPoolAlloc,
        },
        DedicatedAllocation, DeviceMemoryAllocationError, DeviceMemoryExportError,
        ExternalMemoryHandleType, MemoryPool, MemoryRequirements,
    },
    sync::{NowFuture, Sharing},
    DeviceSize,
};
use core::fmt;
use smallvec::SmallVec;
use std::{
    error,
    fs::File,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::size_of,
    sync::Arc,
};

/// Buffer whose content is in device-local memory.
///
/// This buffer type is useful in order to store intermediary data. For example you execute a
/// compute shader that writes to this buffer, then read the content of the buffer in a following
/// compute or graphics pipeline.
///
/// The `DeviceLocalBuffer` will be in device-local memory, unless the device doesn't provide any
/// device-local memory.
///
/// # Usage
///
/// Since a `DeviceLocalBuffer` can only be directly accessed by the GPU, data cannot be transfered between
/// the host process and the buffer alone. One must use additional buffers which are accessible to the CPU as
/// staging areas, then use command buffers to execute the necessary data transfers.
///
/// Despite this, if one knows in advance that a buffer will not need to be frequently accessed by the host,
/// then there may be significant performance gains by using a `DeviceLocalBuffer` over a buffer type which
/// allows host access.
///
/// # Example
///
/// The following example outlines the general strategy one may take when initializing a `DeviceLocalBuffer`.
///
/// ```
/// use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
/// use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBuffer};
/// use vulkano::sync::GpuFuture;
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
///
/// // Simple iterator to construct test data.
/// let data = (0..10_000).map(|i| i as f32);
///
/// // Create a CPU accessible buffer initialized with the data.
/// let temporary_accessible_buffer = CpuAccessibleBuffer::from_iter(
///     device.clone(),
///     BufferUsage::transfer_src(), // Specify this buffer will be used as a transfer source.
///     false,
///     data,
/// )
/// .unwrap();
///
/// // Create a buffer array on the GPU with enough space for `10_000` floats.
/// let device_local_buffer = DeviceLocalBuffer::<[f32]>::array(
///     device.clone(),
///     10_000 as vulkano::DeviceSize,
///     BufferUsage::storage_buffer() | BufferUsage::transfer_dst(), // Specify use as a storage buffer and transfer destination.
///     device.active_queue_families(),
/// )
/// .unwrap();
///
/// // Create a one-time command to copy between the buffers.
/// let mut cbb = AutoCommandBufferBuilder::primary(
///     device.clone(),
///     queue.family(),
///     CommandBufferUsage::OneTimeSubmit,
/// )
/// .unwrap();
/// cbb.copy_buffer(CopyBufferInfo::buffers(
///     temporary_accessible_buffer,
///     device_local_buffer.clone(),
/// ))
/// .unwrap();
/// let cb = cbb.build().unwrap();
///
/// // Execute copy command and wait for completion before proceeding.
/// cb.execute(queue.clone())
/// .unwrap()
/// .then_signal_fence_and_flush()
/// .unwrap()
/// .wait(None /* timeout */)
/// .unwrap()
/// ```
///
#[derive(Debug)]
pub struct DeviceLocalBuffer<T, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>>
where
    T: BufferContents + ?Sized,
{
    // Inner content.
    inner: Arc<UnsafeBuffer>,

    // The memory held by the buffer.
    memory: A,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    // Necessary to make it compile.
    marker: PhantomData<Box<T>>,
}

#[derive(Debug, Copy, Clone)]
enum GpuAccess {
    None,
    NonExclusive { num: u32 },
    Exclusive { num: u32 },
}

impl<T> DeviceLocalBuffer<T>
where
    T: BufferContents,
{
    /// Builds a new buffer. Only allowed for sized data.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    #[inline]
    pub fn new<'a, I>(
        device: Arc<Device>,
        usage: BufferUsage,
        queue_families: I,
    ) -> Result<Arc<DeviceLocalBuffer<T>>, DeviceMemoryAllocationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        unsafe {
            DeviceLocalBuffer::raw(device, size_of::<T>() as DeviceSize, usage, queue_families)
        }
    }
}

// TODO: make this prettier
type DeviceLocalBufferFromBufferFuture =
    CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>;

impl<T> DeviceLocalBuffer<T>
where
    T: BufferContents + ?Sized,
{
    /// Builds a `DeviceLocalBuffer` that copies its data from another buffer.
    ///
    /// This function returns two objects: the newly-created buffer, and a future representing
    /// the initial upload operation. In order to be allowed to use the `DeviceLocalBuffer`, you must
    /// either submit your operation after this future, or execute this future and wait for it to
    /// be finished before submitting your own operation.
    pub fn from_buffer<B>(
        source: Arc<B>,
        usage: BufferUsage,
        queue: Arc<Queue>,
    ) -> Result<
        (Arc<DeviceLocalBuffer<T>>, DeviceLocalBufferFromBufferFuture),
        DeviceLocalBufferCreationError,
    >
    where
        B: TypedBufferAccess<Content = T> + 'static,
    {
        unsafe {
            // We automatically set `transfer_dst` to true in order to avoid annoying errors.
            let actual_usage = BufferUsage {
                transfer_dst: true,
                ..usage
            };

            let buffer = DeviceLocalBuffer::raw(
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
            cbb.copy_buffer(CopyBufferInfo::buffers(source, buffer.clone()))
                .unwrap(); // TODO: return error?
            let cb = cbb.build().unwrap(); // TODO: return OomError

            let future = match cb.execute(queue) {
                Ok(f) => f,
                Err(_) => unreachable!(),
            };

            Ok((buffer, future))
        }
    }
}

impl<T> DeviceLocalBuffer<T>
where
    T: BufferContents,
{
    /// Builds an `DeviceLocalBuffer` from some data.
    ///
    /// This function builds a memory-mapped intermediate buffer, writes the data to it, builds a
    /// command buffer that copies from this intermediate buffer to the final buffer, and finally
    /// submits the command buffer as a future.
    ///
    /// This function returns two objects: the newly-created buffer, and a future representing
    /// the initial upload operation. In order to be allowed to use the `DeviceLocalBuffer`, you must
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
        (Arc<DeviceLocalBuffer<T>>, DeviceLocalBufferFromBufferFuture),
        DeviceLocalBufferCreationError,
    > {
        let source = CpuAccessibleBuffer::from_data(
            queue.device().clone(),
            BufferUsage::transfer_src(),
            false,
            data,
        )?;
        DeviceLocalBuffer::from_buffer(source, usage, queue)
    }
}

impl<T> DeviceLocalBuffer<[T]>
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
        (
            Arc<DeviceLocalBuffer<[T]>>,
            DeviceLocalBufferFromBufferFuture,
        ),
        DeviceLocalBufferCreationError,
    >
    where
        D: IntoIterator<Item = T>,
        D::IntoIter: ExactSizeIterator,
    {
        let source = CpuAccessibleBuffer::from_iter(
            queue.device().clone(),
            BufferUsage::transfer_src(),
            false,
            data,
        )?;
        DeviceLocalBuffer::from_buffer(source, usage, queue)
    }
}

impl<T> DeviceLocalBuffer<[T]>
where
    [T]: BufferContents,
{
    /// Builds a new buffer. Can be used for arrays.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `len` is zero.
    #[inline]
    pub fn array<'a, I>(
        device: Arc<Device>,
        len: DeviceSize,
        usage: BufferUsage,
        queue_families: I,
    ) -> Result<Arc<DeviceLocalBuffer<[T]>>, DeviceMemoryAllocationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        unsafe {
            DeviceLocalBuffer::raw(
                device,
                len * size_of::<T>() as DeviceSize,
                usage,
                queue_families,
            )
        }
    }
}

impl<T> DeviceLocalBuffer<T>
where
    T: BufferContents + ?Sized,
{
    /// Builds a new buffer without checking the size.
    ///
    /// # Safety
    ///
    /// - You must ensure that the size that you pass is correct for `T`.
    ///
    /// # Panics
    ///
    /// - Panics if `size` is zero.
    pub unsafe fn raw<'a, I>(
        device: Arc<Device>,
        size: DeviceSize,
        usage: BufferUsage,
        queue_families: I,
    ) -> Result<Arc<DeviceLocalBuffer<T>>, DeviceMemoryAllocationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        let queue_families = queue_families
            .into_iter()
            .map(|f| f.id())
            .collect::<SmallVec<[u32; 4]>>();

        let (buffer, mem_reqs) = Self::build_buffer(&device, size, usage, &queue_families)?;

        let memory = MemoryPool::alloc_from_requirements(
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
        debug_assert!((memory.offset() % mem_reqs.alignment) == 0);
        buffer.bind_memory(memory.memory(), memory.offset())?;

        Ok(Arc::new(DeviceLocalBuffer {
            inner: buffer,
            memory,
            queue_families,
            marker: PhantomData,
        }))
    }

    /// Same as `raw` but with exportable fd option for the allocated memory on Linux/BSD
    ///
    /// # Panics
    ///
    /// - Panics if `size` is zero.
    pub unsafe fn raw_with_exportable_fd<'a, I>(
        device: Arc<Device>,
        size: DeviceSize,
        usage: BufferUsage,
        queue_families: I,
    ) -> Result<Arc<DeviceLocalBuffer<T>>, DeviceMemoryAllocationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        assert!(device.enabled_extensions().khr_external_memory_fd);
        assert!(device.enabled_extensions().khr_external_memory);

        let queue_families = queue_families
            .into_iter()
            .map(|f| f.id())
            .collect::<SmallVec<[u32; 4]>>();

        let (buffer, mem_reqs) = Self::build_buffer(&device, size, usage, &queue_families)?;

        let memory = alloc_dedicated_with_exportable_fd(
            device.clone(),
            &mem_reqs,
            AllocLayout::Linear,
            MappingRequirement::DoNotMap,
            DedicatedAllocation::Buffer(&buffer),
            |t| {
                if t.is_device_local() {
                    AllocFromRequirementsFilter::Preferred
                } else {
                    AllocFromRequirementsFilter::Allowed
                }
            },
        )?;
        let mem_offset = memory.offset();
        debug_assert!((mem_offset % mem_reqs.alignment) == 0);
        buffer.bind_memory(memory.memory(), mem_offset)?;

        Ok(Arc::new(DeviceLocalBuffer {
            inner: buffer,
            memory,
            queue_families,
            marker: PhantomData,
        }))
    }

    unsafe fn build_buffer(
        device: &Arc<Device>,
        size: DeviceSize,
        usage: BufferUsage,
        queue_families: &SmallVec<[u32; 4]>,
    ) -> Result<(Arc<UnsafeBuffer>, MemoryRequirements), DeviceMemoryAllocationError> {
        let buffer = {
            match UnsafeBuffer::new(
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
            }
        };
        let mem_reqs = buffer.memory_requirements();
        Ok((buffer, mem_reqs))
    }

    /// Exports posix file descriptor for the allocated memory
    /// requires `khr_external_memory_fd` and `khr_external_memory` extensions to be loaded.
    /// Only works on Linux/BSD.
    pub fn export_posix_fd(&self) -> Result<File, DeviceMemoryExportError> {
        self.memory
            .memory()
            .export_fd(ExternalMemoryHandleType::OpaqueFd)
    }
}

impl<T, A> DeviceLocalBuffer<T, A>
where
    T: BufferContents + ?Sized,
{
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

unsafe impl<T, A> DeviceOwned for DeviceLocalBuffer<T, A>
where
    T: BufferContents + ?Sized,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<T, A> BufferAccess for DeviceLocalBuffer<T, A>
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
}

impl<T, A> BufferAccessObject for Arc<DeviceLocalBuffer<T, A>>
where
    T: BufferContents + ?Sized,
    A: Send + Sync + 'static,
{
    #[inline]
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess> {
        self.clone()
    }
}

unsafe impl<T, A> TypedBufferAccess for DeviceLocalBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    type Content = T;
}

impl<T, A> PartialEq for DeviceLocalBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T, A> Eq for DeviceLocalBuffer<T, A>
where
    T: BufferContents + ?Sized,
    A: Send + Sync,
{
}

impl<T, A> Hash for DeviceLocalBuffer<T, A>
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

#[derive(Clone, Debug)]
pub enum DeviceLocalBufferCreationError {
    DeviceMemoryAllocationError(DeviceMemoryAllocationError),
    CommandBufferBeginError(CommandBufferBeginError),
}

impl error::Error for DeviceLocalBufferCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::DeviceMemoryAllocationError(err) => Some(err),
            Self::CommandBufferBeginError(err) => Some(err),
        }
    }
}

impl fmt::Display for DeviceLocalBufferCreationError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceMemoryAllocationError(err) => err.fmt(f),
            Self::CommandBufferBeginError(err) => err.fmt(f),
        }
    }
}

impl From<DeviceMemoryAllocationError> for DeviceLocalBufferCreationError {
    #[inline]
    fn from(e: DeviceMemoryAllocationError) -> Self {
        Self::DeviceMemoryAllocationError(e)
    }
}

impl From<CommandBufferBeginError> for DeviceLocalBufferCreationError {
    #[inline]
    fn from(e: CommandBufferBeginError) -> Self {
        Self::CommandBufferBeginError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::GpuFuture;

    #[test]
    fn from_data_working() {
        let (device, queue) = gfx_dev_and_queue!();

        let (buffer, _) =
            DeviceLocalBuffer::from_data(12u32, BufferUsage::all(), queue.clone()).unwrap();

        let destination =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, 0).unwrap();

        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
        cbb.copy_buffer(CopyBufferInfo::buffers(buffer, destination.clone()))
            .unwrap();
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

        let (buffer, _) = DeviceLocalBuffer::from_iter(
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
        cbb.copy_buffer(CopyBufferInfo::buffers(buffer, destination.clone()))
            .unwrap();
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
    #[allow(unused)]
    fn create_buffer_zero_size_data() {
        let (device, queue) = gfx_dev_and_queue!();

        assert_should_panic!({
            DeviceLocalBuffer::from_data((), BufferUsage::all(), queue.clone()).unwrap();
        });
    }

    // TODO: write tons of tests that try to exploit loopholes
    // this isn't possible yet because checks aren't correctly implemented yet
}
