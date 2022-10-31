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
    sys::{Buffer, BufferCreateInfo, BufferMemory, RawBuffer},
    BufferAccess, BufferAccessObject, BufferContents, BufferInner, BufferUsage,
    CpuAccessibleBuffer, TypedBufferAccess,
};
use crate::{
    buffer::{BufferError, ExternalBufferInfo},
    command_buffer::{allocator::CommandBufferAllocator, AutoCommandBufferBuilder, CopyBufferInfo},
    device::{Device, DeviceOwned},
    memory::{
        allocator::{
            AllocationCreateInfo, AllocationCreationError, AllocationType,
            MemoryAllocatePreference, MemoryAllocator, MemoryUsage,
        },
        DedicatedAllocation, DeviceMemoryError, ExternalMemoryHandleType,
        ExternalMemoryHandleTypes,
    },
    sync::Sharing,
    DeviceSize,
};
use smallvec::SmallVec;
use std::{
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
/// Since a `DeviceLocalBuffer` can only be directly accessed by the GPU, data cannot be transfered
/// between the host process and the buffer alone. One must use additional buffers which are
/// accessible to the CPU as staging areas, then use command buffers to execute the necessary data
/// transfers.
///
/// Despite this, if one knows in advance that a buffer will not need to be frequently accessed by
/// the host, then there may be significant performance gains by using a `DeviceLocalBuffer` over a
/// buffer type which allows host access.
///
/// # Examples
///
/// The following example outlines the general strategy one may take when initializing a
/// `DeviceLocalBuffer`.
///
/// ```
/// use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
/// use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract};
/// use vulkano::sync::GpuFuture;
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
/// # let memory_allocator: vulkano::memory::allocator::StandardMemoryAllocator = return;
/// # let command_buffer_allocator: vulkano::command_buffer::allocator::StandardCommandBufferAllocator = return;
///
/// // Simple iterator to construct test data.
/// let data = (0..10_000).map(|i| i as f32);
///
/// // Create a CPU accessible buffer initialized with the data.
/// let temporary_accessible_buffer = CpuAccessibleBuffer::from_iter(
///     &memory_allocator,
///     BufferUsage { transfer_src: true, ..BufferUsage::empty() }, // Specify this buffer will be used as a transfer source.
///     false,
///     data,
/// )
/// .unwrap();
///
/// // Create a buffer array on the GPU with enough space for `10_000` floats.
/// let device_local_buffer = DeviceLocalBuffer::<[f32]>::array(
///     &memory_allocator,
///     10_000 as vulkano::DeviceSize,
///     BufferUsage {
///         storage_buffer: true,
///         transfer_dst: true,
///         ..BufferUsage::empty()
///     }, // Specify use as a storage buffer and transfer destination.
///     device.active_queue_family_indices().iter().copied(),
/// )
/// .unwrap();
///
/// // Create a one-time command to copy between the buffers.
/// let mut cbb = AutoCommandBufferBuilder::primary(
///     &command_buffer_allocator,
///     queue.queue_family_index(),
///     CommandBufferUsage::OneTimeSubmit,
/// )
/// .unwrap();
/// cbb.copy_buffer(CopyBufferInfo::buffers(
///         temporary_accessible_buffer,
///         device_local_buffer.clone(),
///     ))
///     .unwrap();
/// let cb = cbb.build().unwrap();
///
/// // Execute copy command and wait for completion before proceeding.
/// cb.execute(queue.clone())
///     .unwrap()
///     .then_signal_fence_and_flush()
///     .unwrap()
///     .wait(None /* timeout */)
///     .unwrap()
/// ```
#[derive(Debug)]
pub struct DeviceLocalBuffer<T>
where
    T: BufferContents + ?Sized,
{
    inner: Arc<Buffer>,
    marker: PhantomData<Box<T>>,
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
    pub fn new(
        allocator: &(impl MemoryAllocator + ?Sized),
        usage: BufferUsage,
        queue_family_indices: impl IntoIterator<Item = u32>,
    ) -> Result<Arc<DeviceLocalBuffer<T>>, AllocationCreationError> {
        unsafe {
            DeviceLocalBuffer::raw(
                allocator,
                size_of::<T>() as DeviceSize,
                usage,
                queue_family_indices,
            )
        }
    }
}

impl<T> DeviceLocalBuffer<T>
where
    T: BufferContents + ?Sized,
{
    /// Builds a `DeviceLocalBuffer` that copies its data from another buffer.
    ///
    /// This is a convenience function, equivalent to calling [`new`](DeviceLocalBuffer::new) with
    /// the queue family index of `command_buffer_builder`, then recording a `copy_buffer` command
    /// to `command_buffer_builder`.
    ///
    /// `command_buffer_builder` can then be used to record other commands, built, and executed as
    /// normal. If it is not executed, the buffer contents will be left undefined.
    pub fn from_buffer<B, L, A>(
        allocator: &(impl MemoryAllocator + ?Sized),
        source: Arc<B>,
        usage: BufferUsage,
        command_buffer_builder: &mut AutoCommandBufferBuilder<L, A>,
    ) -> Result<Arc<DeviceLocalBuffer<T>>, AllocationCreationError>
    where
        B: TypedBufferAccess<Content = T> + 'static,
        A: CommandBufferAllocator,
    {
        unsafe {
            // We automatically set `transfer_dst` to true in order to avoid annoying errors.
            let actual_usage = usage | BufferUsage::TRANSFER_DST;

            let buffer = DeviceLocalBuffer::raw(
                allocator,
                source.size(),
                actual_usage,
                source
                    .device()
                    .active_queue_family_indices()
                    .iter()
                    .copied(),
            )?;

            command_buffer_builder
                .copy_buffer(CopyBufferInfo::buffers(source, buffer.clone()))
                .unwrap(); // TODO: return error?

            Ok(buffer)
        }
    }
}

impl<T> DeviceLocalBuffer<T>
where
    T: BufferContents,
{
    /// Builds a `DeviceLocalBuffer` from some data.
    ///
    /// This is a convenience function, equivalent to creating a `CpuAccessibleBuffer`, writing
    /// `data` to it, then calling [`from_buffer`](DeviceLocalBuffer::from_buffer) to copy the data
    /// over.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    pub fn from_data<L, A>(
        allocator: &(impl MemoryAllocator + ?Sized),
        data: T,
        usage: BufferUsage,
        command_buffer_builder: &mut AutoCommandBufferBuilder<L, A>,
    ) -> Result<Arc<DeviceLocalBuffer<T>>, AllocationCreationError>
    where
        A: CommandBufferAllocator,
    {
        let source =
            CpuAccessibleBuffer::from_data(allocator, BufferUsage::TRANSFER_SRC, false, data)?;
        DeviceLocalBuffer::from_buffer(allocator, source, usage, command_buffer_builder)
    }
}

impl<T> DeviceLocalBuffer<[T]>
where
    [T]: BufferContents,
{
    /// Builds a `DeviceLocalBuffer` from an iterator of data.
    ///
    /// This is a convenience function, equivalent to creating a `CpuAccessibleBuffer`, writing
    /// `iter` to it, then calling [`from_buffer`](DeviceLocalBuffer::from_buffer) to copy the data
    /// over.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `data` is empty.
    pub fn from_iter<D, L, A>(
        allocator: &(impl MemoryAllocator + ?Sized),
        data: D,
        usage: BufferUsage,
        command_buffer_builder: &mut AutoCommandBufferBuilder<L, A>,
    ) -> Result<Arc<DeviceLocalBuffer<[T]>>, AllocationCreationError>
    where
        D: IntoIterator<Item = T>,
        D::IntoIter: ExactSizeIterator,
        A: CommandBufferAllocator,
    {
        let source =
            CpuAccessibleBuffer::from_iter(allocator, BufferUsage::TRANSFER_SRC, false, data)?;
        DeviceLocalBuffer::from_buffer(allocator, source, usage, command_buffer_builder)
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
    pub fn array(
        allocator: &(impl MemoryAllocator + ?Sized),
        len: DeviceSize,
        usage: BufferUsage,
        queue_family_indices: impl IntoIterator<Item = u32>,
    ) -> Result<Arc<DeviceLocalBuffer<[T]>>, AllocationCreationError> {
        unsafe {
            DeviceLocalBuffer::raw(
                allocator,
                len * size_of::<T>() as DeviceSize,
                usage,
                queue_family_indices,
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
    pub unsafe fn raw(
        allocator: &(impl MemoryAllocator + ?Sized),
        size: DeviceSize,
        usage: BufferUsage,
        queue_family_indices: impl IntoIterator<Item = u32>,
    ) -> Result<Arc<DeviceLocalBuffer<T>>, AllocationCreationError> {
        let queue_family_indices: SmallVec<[_; 4]> = queue_family_indices.into_iter().collect();

        let raw_buffer = RawBuffer::new(
            allocator.device().clone(),
            BufferCreateInfo {
                sharing: if queue_family_indices.len() >= 2 {
                    Sharing::Concurrent(queue_family_indices)
                } else {
                    Sharing::Exclusive
                },
                size,
                usage,
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
            usage: MemoryUsage::GpuOnly,
            allocate_preference: MemoryAllocatePreference::Unknown,
            dedicated_allocation: Some(DedicatedAllocation::Buffer(&raw_buffer)),
            ..Default::default()
        };

        match allocator.allocate_unchecked(create_info) {
            Ok(alloc) => {
                debug_assert!(alloc.offset() % requirements.alignment == 0);
                debug_assert!(alloc.size() == requirements.size);
                let inner = Arc::new(
                    raw_buffer
                        .bind_memory_unchecked(alloc)
                        .map_err(|(err, _, _)| err)?,
                );

                Ok(Arc::new(DeviceLocalBuffer {
                    inner,
                    marker: PhantomData,
                }))
            }
            Err(err) => Err(err),
        }
    }

    /// Same as `raw` but with exportable fd option for the allocated memory on Linux/BSD
    ///
    /// # Panics
    ///
    /// - Panics if `size` is zero.
    pub unsafe fn raw_with_exportable_fd(
        allocator: &(impl MemoryAllocator + ?Sized),
        size: DeviceSize,
        usage: BufferUsage,
        queue_family_indices: impl IntoIterator<Item = u32>,
    ) -> Result<Arc<DeviceLocalBuffer<T>>, AllocationCreationError> {
        let enabled_extensions = allocator.device().enabled_extensions();
        assert!(enabled_extensions.khr_external_memory_fd);
        assert!(enabled_extensions.khr_external_memory);

        let queue_family_indices: SmallVec<[_; 4]> = queue_family_indices.into_iter().collect();

        let external_memory_properties = allocator
            .device()
            .physical_device()
            .external_buffer_properties(ExternalBufferInfo {
                usage,
                ..ExternalBufferInfo::handle_type(ExternalMemoryHandleType::OpaqueFd)
            })
            .unwrap()
            .external_memory_properties;
        // VUID-VkExportMemoryAllocateInfo-handleTypes-00656
        assert!(external_memory_properties.exportable);

        // VUID-VkMemoryAllocateInfo-pNext-00639
        // Guaranteed because we always create a dedicated allocation

        let external_memory_handle_types = ExternalMemoryHandleTypes::OPAQUE_FD;
        let raw_buffer = RawBuffer::new(
            allocator.device().clone(),
            BufferCreateInfo {
                sharing: if queue_family_indices.len() >= 2 {
                    Sharing::Concurrent(queue_family_indices)
                } else {
                    Sharing::Exclusive
                },
                size,
                usage,
                external_memory_handle_types,
                ..Default::default()
            },
        )
        .map_err(|err| match err {
            BufferError::AllocError(err) => err,
            // We don't use sparse-binding, therefore the other errors can't happen.
            _ => unreachable!(),
        })?;
        let requirements = raw_buffer.memory_requirements();
        let memory_type_index = allocator
            .find_memory_type_index(requirements.memory_type_bits, MemoryUsage::GpuOnly.into())
            .expect("failed to find a suitable memory type");

        let memory_properties = allocator.device().physical_device().memory_properties();
        let heap_index = memory_properties.memory_types[memory_type_index as usize].heap_index;
        // VUID-vkAllocateMemory-pAllocateInfo-01713
        assert!(size <= memory_properties.memory_heaps[heap_index as usize].size);

        match allocator.allocate_dedicated_unchecked(
            memory_type_index,
            requirements.size,
            Some(DedicatedAllocation::Buffer(&raw_buffer)),
            external_memory_handle_types,
        ) {
            Ok(alloc) => {
                debug_assert!(alloc.offset() % requirements.alignment == 0);
                debug_assert!(alloc.size() == requirements.size);
                let inner = Arc::new(
                    raw_buffer
                        .bind_memory_unchecked(alloc)
                        .map_err(|(err, _, _)| err)?,
                );

                Ok(Arc::new(DeviceLocalBuffer {
                    inner,
                    marker: PhantomData,
                }))
            }
            Err(err) => Err(err),
        }
    }

    /// Exports posix file descriptor for the allocated memory
    /// requires `khr_external_memory_fd` and `khr_external_memory` extensions to be loaded.
    /// Only works on Linux/BSD.
    pub fn export_posix_fd(&self) -> Result<File, DeviceMemoryError> {
        let allocation = match self.inner.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        allocation
            .device_memory()
            .export_fd(ExternalMemoryHandleType::OpaqueFd)
    }
}

unsafe impl<T> DeviceOwned for DeviceLocalBuffer<T>
where
    T: BufferContents + ?Sized,
{
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<T> BufferAccess for DeviceLocalBuffer<T>
where
    T: BufferContents + ?Sized,
{
    fn inner(&self) -> BufferInner<'_> {
        BufferInner {
            buffer: &self.inner,
            offset: 0,
        }
    }

    fn size(&self) -> DeviceSize {
        self.inner.size()
    }
}

impl<T> BufferAccessObject for Arc<DeviceLocalBuffer<T>>
where
    T: BufferContents + ?Sized,
{
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess> {
        self.clone()
    }
}

unsafe impl<T> TypedBufferAccess for DeviceLocalBuffer<T>
where
    T: BufferContents + ?Sized,
{
    type Content = T;
}

impl<T> PartialEq for DeviceLocalBuffer<T>
where
    T: BufferContents + ?Sized,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl<T> Eq for DeviceLocalBuffer<T> where T: BufferContents + ?Sized {}

impl<T> Hash for DeviceLocalBuffer<T>
where
    T: BufferContents + ?Sized,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        command_buffer::{
            allocator::StandardCommandBufferAllocator, CommandBufferUsage,
            PrimaryCommandBufferAbstract,
        },
        memory::allocator::StandardMemoryAllocator,
        sync::GpuFuture,
    };

    #[test]
    fn from_data_working() {
        let (device, queue) = gfx_dev_and_queue!();

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        let buffer = DeviceLocalBuffer::from_data(
            &memory_allocator,
            12u32,
            BufferUsage::TRANSFER_SRC,
            &mut command_buffer_builder,
        )
        .unwrap();

        let destination =
            CpuAccessibleBuffer::from_data(&memory_allocator, BufferUsage::TRANSFER_DST, false, 0)
                .unwrap();

        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(buffer, destination.clone()))
            .unwrap();
        let _ = command_buffer_builder
            .build()
            .unwrap()
            .execute(queue)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        let destination_content = destination.read().unwrap();
        assert_eq!(*destination_content, 12);
    }

    #[test]
    fn from_iter_working() {
        let (device, queue) = gfx_dev_and_queue!();

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        let allocator = StandardMemoryAllocator::new_default(device);

        let buffer = DeviceLocalBuffer::from_iter(
            &allocator,
            (0..512u32).map(|n| n * 2),
            BufferUsage::TRANSFER_SRC,
            &mut command_buffer_builder,
        )
        .unwrap();

        let destination = CpuAccessibleBuffer::from_iter(
            &allocator,
            BufferUsage::TRANSFER_DST,
            false,
            (0..512).map(|_| 0u32),
        )
        .unwrap();

        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(buffer, destination.clone()))
            .unwrap();
        let _ = command_buffer_builder
            .build()
            .unwrap()
            .execute(queue)
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

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        let allocator = StandardMemoryAllocator::new_default(device);

        assert_should_panic!({
            DeviceLocalBuffer::from_data(
                &allocator,
                (),
                BufferUsage::TRANSFER_DST,
                &mut command_buffer_builder,
            )
            .unwrap();
        });
    }

    // TODO: write tons of tests that try to exploit loopholes
    // this isn't possible yet because checks aren't correctly implemented yet
}
