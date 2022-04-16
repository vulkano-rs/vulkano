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

use super::{
    sys::{UnsafeBuffer, UnsafeBufferCreateInfo},
    BufferAccess, BufferAccessObject, BufferContents, BufferCreationError, BufferInner,
    BufferUsage, TypedBufferAccess,
};
use crate::{
    device::{physical::QueueFamily, Device, DeviceOwned},
    memory::{
        pool::{
            alloc_dedicated_with_exportable_fd, AllocFromRequirementsFilter, AllocLayout,
            MappingRequirement, MemoryPoolAlloc, PotentialDedicatedAllocation, StdMemoryPoolAlloc,
        },
        DedicatedAllocation, DeviceMemoryAllocationError, DeviceMemoryExportError,
        ExternalMemoryHandleType, MemoryPool, MemoryRequirements,
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
