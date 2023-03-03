// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Location in memory that contains data.
//!
//! A Vulkan buffer is very similar to a buffer that you would use in programming languages in
//! general, in the sense that it is a location in memory that contains data. The difference
//! between a Vulkan buffer and a regular buffer is that the content of a Vulkan buffer is
//! accessible from the GPU.
//!
//! Vulkano does not perform any specific marshalling of buffer data. The representation of the
//! buffer in memory is identical between the CPU and GPU. Because the Rust compiler is allowed to
//! reorder struct fields at will by default when using `#[repr(Rust)]`, it is advised to mark each
//! struct requiring imput assembly as `#[repr(C)]`. This forces Rust to follow the standard C
//! procedure. Each element is laid out in memory in the order of declaration and aligned to a
//! multiple of their alignment.
//!
//! # Multiple levels of abstraction
//!
//! - The low-level implementation of a buffer is [`RawBuffer`], which corresponds directly to a
//!   `VkBuffer`, and as such doesn't hold onto any memory.
//! - [`Buffer`] is a `RawBuffer` with memory bound to it, and with state tracking.
//! - [`Subbuffer`] is what you will use most of the time, as it is what all the APIs expect. It is
//!   reference to a portion of a `Buffer`. `Subbuffer` also has a type parameter, which is a hint
//!   for how the data in the portion of the buffer is going to be interpreted.
//!
//! # `Subbuffer` allocation
//!
//! There are two ways to get a `Subbuffer`:
//!
//! - By using the functions on `Buffer`, which create a new buffer and memory allocation each
//!   time, and give you a `Subbuffer` that has an entire `Buffer` dedicated to it.
//! - By using the [`SubbufferAllocator`], which creates `Subbuffer`s by suballocating existing
//!   `Buffer`s such that the `Buffer`s can keep being reused.
//!
//! Which of these you should choose depends on the use case. For example, if you need to upload
//! data to the device each frame, then you should use `SubbufferAllocator`. Same goes for if you
//! need to download data very frequently, or if you need to allocate a lot of intermediary buffers
//! that are only accessed by the device. On the other hand, if you need to upload some data just
//! once, or you can keep reusing the same buffer (because its size is unchanging) it's best to
//! use a dedicated `Buffer` for that.
//!
//! # Memory usage
//!
//! When allocating memory for a buffer, you have to specify a *memory usage*. This tells the
//! memory allocator what memory type it should pick for the allocation.
//!
//! - [`MemoryUsage::GpuOnly`] will allocate a buffer that's usually located in device-local
//!   memory and whose content can't be directly accessed by your application. Accessing this
//!   buffer from the device is generally faster compared to accessing a buffer that's located in
//!   host-visible memory.
//! - [`MemoryUsage::Upload`] and [`MemoryUsage::Download`] both allocate from a host-visible
//!   memory type, which means the buffer can be accessed directly from the host. Buffers allocated
//!   with these memory usages are needed to get data to and from the device.
//!
//! Take for example a buffer that is under constant access by the device but you need to read its
//! content on the host from time to time, it may be a good idea to use a device-local buffer as
//! the main buffer and a host-visible buffer for when you need to read it. Then whenever you need
//! to read the main buffer, ask the device to copy from the device-local buffer to the
//! host-visible buffer, and read the host-visible buffer instead.
//!
//! # Buffer usage
//!
//! When you create a buffer, you have to specify its *usage*. In other words, you have to
//! specify the way it is going to be used. Trying to use a buffer in a way that wasn't specified
//! when you created it will result in a runtime error.
//!
//! You can use buffers for the following purposes:
//!
//! - Can contain arbitrary data that can be transferred from/to other buffers and images.
//! - Can be read and modified from a shader.
//! - Can be used as a source of vertices and indices.
//! - Can be used as a source of list of models for draw indirect commands.
//!
//! Accessing a buffer from a shader can be done in the following ways:
//!
//! - As a uniform buffer. Uniform buffers are read-only.
//! - As a storage buffer. Storage buffers can be read and written.
//! - As a uniform texel buffer. Contrary to a uniform buffer, the data is interpreted by the GPU
//!   and can be for example normalized.
//! - As a storage texel buffer. Additionally, some data formats can be modified with atomic
//!   operations.
//!
//! Using uniform/storage texel buffers requires creating a *buffer view*. See [the `view` module]
//! for how to create a buffer view.
//!
//! See also [the `shader` module documentation] for information about how buffer contents need to
//! be laid out in accordance with the shader interface.
//!
//! [`RawBuffer`]: self::sys::RawBuffer
//! [`SubbufferAllocator`]: self::allocator::SubbufferAllocator
//! [the `view` module]: self::view
//! [the `shader` module documentation]: crate::shader

pub use self::{subbuffer::Subbuffer, usage::BufferUsage};
use self::{
    subbuffer::{ReadLockError, WriteLockError},
    sys::{BufferCreateInfo, RawBuffer},
};
use crate::{
    device::{Device, DeviceOwned},
    macros::vulkan_bitflags,
    memory::{
        allocator::{
            AllocationCreateInfo, AllocationCreationError, AllocationType, DeviceAlignment,
            DeviceLayout, MemoryAlloc, MemoryAllocatePreference, MemoryAllocator, MemoryUsage,
        },
        DedicatedAllocation, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        ExternalMemoryProperties, MemoryRequirements,
    },
    range_map::RangeMap,
    sync::{future::AccessError, CurrentAccess, Sharing},
    DeviceSize, NonZeroDeviceSize, RequirementNotMet, RequiresOneOf, Version, VulkanError,
    VulkanObject,
};
use bytemuck::{Pod, PodCastError};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
use std::{
    alloc::Layout,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    mem::{size_of, size_of_val},
    ops::Range,
    ptr,
    sync::Arc,
};

pub mod allocator;
pub mod subbuffer;
pub mod sys;
mod usage;
pub mod view;

/// A storage for raw bytes.
///
/// Unlike [`RawBuffer`], a `Buffer` has memory backing it, and can be used normally.
///
/// See [the module-level documentation] for more information about buffers.
///
/// # Examples
///
/// Sometimes, you need a buffer that is rarely accessed by the host. To get the best performance
/// in this case, one should use a buffer in device-local memory, that is inaccessible from the
/// host. As such, to initialize or otherwise access such a buffer, we need a *staging buffer*.
///
/// The following example outlines the general strategy one may take when initializing a
/// device-local buffer.
///
/// ```
/// use vulkano::{
///     buffer::{BufferUsage, Buffer, BufferAllocateInfo},
///     command_buffer::{
///         AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
///         PrimaryCommandBufferAbstract,
///     },
///     memory::allocator::MemoryUsage,
///     sync::GpuFuture,
///     DeviceSize,
/// };
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
/// # let memory_allocator: vulkano::memory::allocator::StandardMemoryAllocator = return;
/// # let command_buffer_allocator: vulkano::command_buffer::allocator::StandardCommandBufferAllocator = return;
/// // Simple iterator to construct test data.
/// let data = (0..10_000).map(|i| i as f32);
///
/// // Create a host-accessible buffer initialized with the data.
/// let temporary_accessible_buffer = Buffer::from_iter(
///     &memory_allocator,
///     BufferAllocateInfo {
///         // Specify that this buffer will be used as a transfer source.
///         buffer_usage: BufferUsage::TRANSFER_SRC,
///         // Specify use for upload to the device.
///         memory_usage: MemoryUsage::Upload,
///         ..Default::default()
///     },
///     data,
/// )
/// .unwrap();
///
/// // Create a buffer in device-local with enough space for a slice of `10_000` floats.
/// let device_local_buffer = Buffer::new_slice::<f32>(
///     &memory_allocator,
///     BufferAllocateInfo {
///         // Specify use as a storage buffer and transfer destination.
///         buffer_usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
///         // Specify use by the device only.
///         memory_usage: MemoryUsage::GpuOnly,
///         ..Default::default()
///     },
///     10_000 as DeviceSize,
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
/// // Execute the copy command and wait for completion before proceeding.
/// cb.execute(queue.clone())
///     .unwrap()
///     .then_signal_fence_and_flush()
///     .unwrap()
///     .wait(None /* timeout */)
///     .unwrap()
/// ```
///
/// [the module-level documentation]: self
#[derive(Debug)]
pub struct Buffer {
    inner: RawBuffer,
    memory: BufferMemory,
    state: Mutex<BufferState>,
}

/// The type of backing memory that a buffer can have.
#[derive(Debug)]
pub enum BufferMemory {
    /// The buffer is backed by normal memory, bound with [`bind_memory`].
    ///
    /// [`bind_memory`]: RawBuffer::bind_memory
    Normal(MemoryAlloc),

    /// The buffer is backed by sparse memory, bound with [`bind_sparse`].
    ///
    /// [`bind_sparse`]: crate::device::QueueGuard::bind_sparse
    Sparse,
}

impl Buffer {
    /// Creates a new `Buffer` and writes `data` in it. Returns a [`Subbuffer`] spanning the whole
    /// buffer.
    ///
    /// This only works with memory types that are host-visible. If you want to upload data to a
    /// buffer allocated in device-local memory, you will need to create a staging buffer and copy
    /// the contents over.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `T` has an alignment greater than `64`.
    pub fn from_data<T>(
        allocator: &(impl MemoryAllocator + ?Sized),
        allocate_info: BufferAllocateInfo,
        data: T,
    ) -> Result<Subbuffer<T>, BufferError>
    where
        T: BufferContents,
    {
        let buffer = Buffer::new_sized(allocator, allocate_info)?;

        unsafe { ptr::write(&mut *buffer.write()?, data) };

        Ok(buffer)
    }

    /// Creates a new `Buffer` and writes all elements of `iter` in it. Returns a [`Subbuffer`]
    /// spanning the whole buffer.
    ///
    /// This only works with memory types that are host-visible. If you want to upload data to a
    /// buffer allocated in device-local memory, you will need to create a staging buffer and copy
    /// the contents over.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `T` has an alignment greater than `64`.
    /// - Panics if `iter` is empty.
    pub fn from_iter<T, I>(
        allocator: &(impl MemoryAllocator + ?Sized),
        allocate_info: BufferAllocateInfo,
        iter: I,
    ) -> Result<Subbuffer<[T]>, BufferError>
    where
        [T]: BufferContents,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        let buffer = Buffer::new_slice(allocator, allocate_info, iter.len().try_into().unwrap())?;

        for (o, i) in buffer.write()?.iter_mut().zip(iter) {
            unsafe { ptr::write(o, i) };
        }

        Ok(buffer)
    }

    /// Creates a new uninitialized `Buffer` for sized data. Returns a [`Subbuffer`] spanning the
    /// whole buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `T` has an alignment greater than `64`.
    pub fn new_sized<T>(
        allocator: &(impl MemoryAllocator + ?Sized),
        allocate_info: BufferAllocateInfo,
    ) -> Result<Subbuffer<T>, BufferError> {
        let layout = Layout::new::<T>()
            .try_into()
            .expect("can't allocate memory for zero-sized types");

        Buffer::new(allocator, allocate_info, layout).map(Subbuffer::from_buffer)
    }

    /// Creates a new uninitialized `Buffer` for a slice. Returns a [`Subbuffer`] spanning the
    /// whole buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `T` has zero size.
    /// - Panics if `T` has an alignment greater than `64`.
    /// - Panics if `len` is zero.
    pub fn new_slice<T>(
        allocator: &(impl MemoryAllocator + ?Sized),
        allocate_info: BufferAllocateInfo,
        len: DeviceSize,
    ) -> Result<Subbuffer<[T]>, BufferError> {
        let layout = Layout::array::<T>(len.try_into().unwrap())
            .unwrap()
            .try_into()
            .expect("can't allocate memory for zero-sized types");

        Buffer::new(allocator, allocate_info, layout).map(Subbuffer::from_buffer)
    }

    /// Creates a new uninitialized `Buffer` with the given `layout`.
    ///
    /// # Panics
    ///
    /// - Panics if `layout.alignment()` is greater than 64.
    pub fn new(
        allocator: &(impl MemoryAllocator + ?Sized),
        allocate_info: BufferAllocateInfo,
        layout: DeviceLayout,
    ) -> Result<Arc<Self>, BufferError> {
        assert!(layout.alignment().as_devicesize() <= 64);
        // TODO: Enable once sparse binding materializes
        // assert!(!allocate_info.flags.contains(BufferCreateFlags::SPARSE_BINDING));

        let raw_buffer = RawBuffer::new(
            allocator.device().clone(),
            BufferCreateInfo {
                flags: allocate_info.flags,
                sharing: allocate_info.sharing,
                size: layout.size(),
                usage: allocate_info.buffer_usage,
                external_memory_handle_types: allocate_info.external_memory_handle_types,
                _ne: crate::NonExhaustive(()),
            },
        )?;
        let mut requirements = *raw_buffer.memory_requirements();
        requirements.layout = requirements.layout.align_to(layout.alignment()).unwrap();
        let create_info = AllocationCreateInfo {
            requirements,
            allocation_type: AllocationType::Linear,
            usage: allocate_info.memory_usage,
            dedicated_allocation: Some(DedicatedAllocation::Buffer(&raw_buffer)),
            allocate_preference: allocate_info.allocate_preference,
            _ne: crate::NonExhaustive(()),
        };

        let mut allocation = unsafe { allocator.allocate_unchecked(create_info) }?;
        debug_assert!(allocation.offset() % requirements.layout.alignment().as_nonzero() == 0);
        debug_assert!(allocation.size() == requirements.layout.size());

        // The implementation might require a larger size than we wanted. With this it is easier to
        // invalidate and flush the whole buffer. It does not affect the allocation in any way.
        allocation.shrink(layout.size());

        unsafe { raw_buffer.bind_memory_unchecked(allocation) }
            .map(Arc::new)
            .map_err(|(err, _, _)| err.into())
    }

    fn from_raw(inner: RawBuffer, memory: BufferMemory) -> Self {
        let state = Mutex::new(BufferState::new(inner.size()));

        Buffer {
            inner,
            memory,
            state,
        }
    }

    /// Returns the type of memory that is backing this buffer.
    #[inline]
    pub fn memory(&self) -> &BufferMemory {
        &self.memory
    }

    /// Returns the memory requirements for this buffer.
    #[inline]
    pub fn memory_requirements(&self) -> &MemoryRequirements {
        self.inner.memory_requirements()
    }

    /// Returns the flags the buffer was created with.
    #[inline]
    pub fn flags(&self) -> BufferCreateFlags {
        self.inner.flags()
    }

    /// Returns the size of the buffer in bytes.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.inner.size()
    }

    /// Returns the usage the buffer was created with.
    #[inline]
    pub fn usage(&self) -> BufferUsage {
        self.inner.usage()
    }

    /// Returns the sharing the buffer was created with.
    #[inline]
    pub fn sharing(&self) -> &Sharing<SmallVec<[u32; 4]>> {
        self.inner.sharing()
    }

    /// Returns the external memory handle types that are supported with this buffer.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.inner.external_memory_handle_types()
    }

    /// Returns the device address for this buffer.
    // TODO: Caching?
    pub fn device_address(&self) -> Result<NonZeroDeviceSize, BufferError> {
        let device = self.device();

        // VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324
        if !device.enabled_features().buffer_device_address {
            return Err(BufferError::RequirementNotMet {
                required_for: "`Buffer::device_address`",
                requires_one_of: RequiresOneOf {
                    features: &["buffer_device_address"],
                    ..Default::default()
                },
            });
        }

        // VUID-VkBufferDeviceAddressInfo-buffer-02601
        if !self.usage().intersects(BufferUsage::SHADER_DEVICE_ADDRESS) {
            return Err(BufferError::BufferMissingUsage);
        }

        let info = ash::vk::BufferDeviceAddressInfo {
            buffer: self.handle(),
            ..Default::default()
        };
        let fns = device.fns();
        let f = if device.api_version() >= Version::V1_2 {
            fns.v1_2.get_buffer_device_address
        } else if device.enabled_extensions().khr_buffer_device_address {
            fns.khr_buffer_device_address.get_buffer_device_address_khr
        } else {
            fns.ext_buffer_device_address.get_buffer_device_address_ext
        };
        let ptr = unsafe { f(device.handle(), &info) };

        Ok(NonZeroDeviceSize::new(ptr).unwrap())
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, BufferState> {
        self.state.lock()
    }
}

unsafe impl VulkanObject for Buffer {
    type Handle = ash::vk::Buffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for Buffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl PartialEq for Buffer {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

/// Parameters to create a new [`RawBuffer`] and allocate and bind memory to it.
#[derive(Clone, Debug)]
pub struct BufferAllocateInfo {
    /// Flags to enable.
    ///
    /// The default value is [`BufferCreateFlags::empty()`].
    pub flags: BufferCreateFlags,

    /// Whether the buffer can be shared across multiple queues, or is limited to a single queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub sharing: Sharing<SmallVec<[u32; 4]>>,

    /// How the buffer is going to be used.
    ///
    /// The default value is [`BufferUsage::empty()`], which must be overridden.
    pub buffer_usage: BufferUsage,

    /// The external memory handle types that are going to be used with the buffer.
    ///
    /// If this value is not empty, then the device API version must be at least 1.1, or the
    /// [`khr_external_memory`] extension must be enabled on the device.
    ///
    /// The default value is [`ExternalMemoryHandleTypes::empty()`].
    ///
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    pub external_memory_handle_types: ExternalMemoryHandleTypes,

    /// The memory usage to use for the allocation.
    ///
    /// If this is set to [`MemoryUsage::GpuOnly`], then the buffer may need to be initialized
    /// using a staging buffer. The exception is some integrated GPUs and laptop GPUs, which do not
    /// have memory types that are not host-visible. With [`MemoryUsage::Upload`] and
    /// [`MemoryUsage::Download`], a staging buffer is never needed.
    ///
    /// The default value is [`MemoryUsage::Upload`].
    pub memory_usage: MemoryUsage,

    /// The memory allocate preference to use for the allocation.
    ///
    /// The default value is [`MemoryAllocatePreference::Unknown`].
    pub allocate_preference: MemoryAllocatePreference,

    pub _ne: crate::NonExhaustive,
}

impl Default for BufferAllocateInfo {
    #[inline]
    fn default() -> Self {
        BufferAllocateInfo {
            flags: BufferCreateFlags::empty(),
            sharing: Sharing::Exclusive,
            buffer_usage: BufferUsage::empty(),
            external_memory_handle_types: ExternalMemoryHandleTypes::empty(),
            memory_usage: MemoryUsage::Upload,
            allocate_preference: MemoryAllocatePreference::Unknown,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The current state of a buffer.
#[derive(Debug)]
pub(crate) struct BufferState {
    ranges: RangeMap<DeviceSize, BufferRangeState>,
}

impl BufferState {
    fn new(size: DeviceSize) -> Self {
        BufferState {
            ranges: [(
                0..size,
                BufferRangeState {
                    current_access: CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads: 0,
                    },
                },
            )]
            .into_iter()
            .collect(),
        }
    }

    pub(crate) fn check_cpu_read(&self, range: Range<DeviceSize>) -> Result<(), ReadLockError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::CpuExclusive { .. } => return Err(ReadLockError::CpuWriteLocked),
                CurrentAccess::GpuExclusive { .. } => return Err(ReadLockError::GpuWriteLocked),
                CurrentAccess::Shared { .. } => (),
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn cpu_read_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::Shared { cpu_reads, .. } => {
                    *cpu_reads += 1;
                }
                _ => unreachable!("Buffer is being written by the CPU or GPU"),
            }
        }
    }

    pub(crate) unsafe fn cpu_read_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::Shared { cpu_reads, .. } => *cpu_reads -= 1,
                _ => unreachable!("Buffer was not locked for CPU read"),
            }
        }
    }

    pub(crate) fn check_cpu_write(&self, range: Range<DeviceSize>) -> Result<(), WriteLockError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::CpuExclusive => return Err(WriteLockError::CpuLocked),
                CurrentAccess::GpuExclusive { .. } => return Err(WriteLockError::GpuLocked),
                CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads: 0,
                } => (),
                CurrentAccess::Shared { cpu_reads, .. } if *cpu_reads > 0 => {
                    return Err(WriteLockError::CpuLocked)
                }
                CurrentAccess::Shared { .. } => return Err(WriteLockError::GpuLocked),
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn cpu_write_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            state.current_access = CurrentAccess::CpuExclusive;
        }
    }

    pub(crate) unsafe fn cpu_write_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::CpuExclusive => {
                    state.current_access = CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads: 0,
                    }
                }
                _ => unreachable!("Buffer was not locked for CPU write"),
            }
        }
    }

    pub(crate) fn check_gpu_read(&self, range: Range<DeviceSize>) -> Result<(), AccessError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::Shared { .. } => (),
                _ => return Err(AccessError::AlreadyInUse),
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn gpu_read_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_reads, .. }
                | CurrentAccess::Shared { gpu_reads, .. } => *gpu_reads += 1,
                _ => unreachable!("Buffer is being written by the CPU"),
            }
        }
    }

    pub(crate) unsafe fn gpu_read_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_reads, .. } => *gpu_reads -= 1,
                CurrentAccess::Shared { gpu_reads, .. } => *gpu_reads -= 1,
                _ => unreachable!("Buffer was not locked for GPU read"),
            }
        }
    }

    pub(crate) fn check_gpu_write(&self, range: Range<DeviceSize>) -> Result<(), AccessError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads: 0,
                } => (),
                _ => return Err(AccessError::AlreadyInUse),
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn gpu_write_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_writes, .. } => *gpu_writes += 1,
                &mut CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads,
                } => {
                    state.current_access = CurrentAccess::GpuExclusive {
                        gpu_reads,
                        gpu_writes: 1,
                    }
                }
                _ => unreachable!("Buffer is being accessed by the CPU"),
            }
        }
    }

    pub(crate) unsafe fn gpu_write_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                &mut CurrentAccess::GpuExclusive {
                    gpu_reads,
                    gpu_writes: 1,
                } => {
                    state.current_access = CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads,
                    }
                }
                CurrentAccess::GpuExclusive { gpu_writes, .. } => *gpu_writes -= 1,
                _ => unreachable!("Buffer was not locked for GPU write"),
            }
        }
    }
}

/// The current state of a specific range of bytes in a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BufferRangeState {
    current_access: CurrentAccess,
}

/// Error that can happen in buffer functions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BufferError {
    VulkanError(VulkanError),

    /// Allocating memory failed.
    AllocError(AllocationCreationError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The buffer is missing the `SHADER_DEVICE_ADDRESS` usage.
    BufferMissingUsage,

    /// The memory was created dedicated to a resource, but not to this buffer.
    DedicatedAllocationMismatch,

    /// A dedicated allocation is required for this buffer, but one was not provided.
    DedicatedAllocationRequired,

    /// The host is already using this buffer in a way that is incompatible with the
    /// requested access.
    InUseByHost,

    /// The device is already using this buffer in a way that is incompatible with the
    /// requested access.
    InUseByDevice,

    /// The specified size exceeded the value of the `max_buffer_size` limit.
    MaxBufferSizeExceeded {
        size: DeviceSize,
        max: DeviceSize,
    },

    /// The offset of the allocation does not have the required alignment.
    MemoryAllocationNotAligned {
        allocation_offset: DeviceSize,
        required_alignment: DeviceAlignment,
    },

    /// The size of the allocation is smaller than what is required.
    MemoryAllocationTooSmall {
        allocation_size: DeviceSize,
        required_size: DeviceSize,
    },

    /// The buffer was created with the `SHADER_DEVICE_ADDRESS` usage, but the memory does not
    /// support this usage.
    MemoryBufferDeviceAddressNotSupported,

    /// The memory was created with export handle types, but none of these handle types were
    /// enabled on the buffer.
    MemoryExternalHandleTypesDisjoint {
        buffer_handle_types: ExternalMemoryHandleTypes,
        memory_export_handle_types: ExternalMemoryHandleTypes,
    },

    /// The memory was created with an import, but the import's handle type was not enabled on
    /// the buffer.
    MemoryImportedHandleTypeNotEnabled {
        buffer_handle_types: ExternalMemoryHandleTypes,
        memory_imported_handle_type: ExternalMemoryHandleType,
    },

    /// The memory backing this buffer is not visible to the host.
    MemoryNotHostVisible,

    /// The protection of buffer and memory are not equal.
    MemoryProtectedMismatch {
        buffer_protected: bool,
        memory_protected: bool,
    },

    /// The provided memory type is not one of the allowed memory types that can be bound to this
    /// buffer.
    MemoryTypeNotAllowed {
        provided_memory_type_index: u32,
        allowed_memory_type_bits: u32,
    },

    /// The sharing mode was set to `Concurrent`, but one of the specified queue family indices was
    /// out of range.
    SharingQueueFamilyIndexOutOfRange {
        queue_family_index: u32,
        queue_family_count: u32,
    },
}

impl Error for BufferError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::VulkanError(err) => Some(err),
            Self::AllocError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for BufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::VulkanError(_) => write!(f, "a runtime error occurred"),
            Self::AllocError(_) => write!(f, "allocating memory failed"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::BufferMissingUsage => {
                write!(f, "the buffer is missing the `SHADER_DEVICE_ADDRESS` usage")
            }
            Self::DedicatedAllocationMismatch => write!(
                f,
                "the memory was created dedicated to a resource, but not to this buffer",
            ),
            Self::DedicatedAllocationRequired => write!(
                f,
                "a dedicated allocation is required for this buffer, but one was not provided"
            ),
            Self::InUseByHost => write!(
                f,
                "the host is already using this buffer in a way that is incompatible with the \
                requested access",
            ),
            Self::InUseByDevice => write!(
                f,
                "the device is already using this buffer in a way that is incompatible with the \
                requested access"
            ),
            Self::MaxBufferSizeExceeded { .. } => write!(
                f,
                "the specified size exceeded the value of the `max_buffer_size` limit",
            ),
            Self::MemoryAllocationNotAligned {
                allocation_offset,
                required_alignment,
            } => write!(
                f,
                "the offset of the allocation ({}) does not have the required alignment ({:?})",
                allocation_offset, required_alignment,
            ),
            Self::MemoryAllocationTooSmall {
                allocation_size,
                required_size,
            } => write!(
                f,
                "the size of the allocation ({}) is smaller than what is required ({})",
                allocation_size, required_size,
            ),
            Self::MemoryBufferDeviceAddressNotSupported => write!(
                f,
                "the buffer was created with the `SHADER_DEVICE_ADDRESS` usage, but the memory \
                does not support this usage",
            ),
            Self::MemoryExternalHandleTypesDisjoint { .. } => write!(
                f,
                "the memory was created with export handle types, but none of these handle types \
                were enabled on the buffer",
            ),
            Self::MemoryImportedHandleTypeNotEnabled { .. } => write!(
                f,
                "the memory was created with an import, but the import's handle type was not \
                enabled on the buffer",
            ),
            Self::MemoryNotHostVisible => write!(
                f,
                "the memory backing this buffer is not visible to the host",
            ),
            Self::MemoryProtectedMismatch {
                buffer_protected,
                memory_protected,
            } => write!(
                f,
                "the protection of buffer ({}) and memory ({}) are not equal",
                buffer_protected, memory_protected,
            ),
            Self::MemoryTypeNotAllowed {
                provided_memory_type_index,
                allowed_memory_type_bits,
            } => write!(
                f,
                "the provided memory type ({}) is not one of the allowed memory types (",
                provided_memory_type_index,
            )
            .and_then(|_| {
                let mut first = true;

                for i in (0..size_of_val(allowed_memory_type_bits))
                    .filter(|i| allowed_memory_type_bits & (1 << i) != 0)
                {
                    if first {
                        write!(f, "{}", i)?;
                        first = false;
                    } else {
                        write!(f, ", {}", i)?;
                    }
                }

                Ok(())
            })
            .and_then(|_| write!(f, ") that can be bound to this buffer")),
            Self::SharingQueueFamilyIndexOutOfRange { .. } => write!(
                f,
                "the sharing mode was set to `Concurrent`, but one of the specified queue family \
                indices was out of range",
            ),
        }
    }
}

impl From<VulkanError> for BufferError {
    fn from(err: VulkanError) -> Self {
        Self::VulkanError(err)
    }
}

impl From<AllocationCreationError> for BufferError {
    fn from(err: AllocationCreationError) -> Self {
        Self::AllocError(err)
    }
}

impl From<RequirementNotMet> for BufferError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

impl From<ReadLockError> for BufferError {
    fn from(err: ReadLockError) -> Self {
        match err {
            ReadLockError::CpuWriteLocked => Self::InUseByHost,
            ReadLockError::GpuWriteLocked => Self::InUseByDevice,
        }
    }
}

impl From<WriteLockError> for BufferError {
    fn from(err: WriteLockError) -> Self {
        match err {
            WriteLockError::CpuLocked => Self::InUseByHost,
            WriteLockError::GpuLocked => Self::InUseByDevice,
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags to be set when creating a buffer.
    BufferCreateFlags = BufferCreateFlags(u32);

    /* TODO: enable
    /// The buffer will be backed by sparse memory binding (through queue commands) instead of
    /// regular binding (through [`bind_memory`]).
    ///
    /// The [`sparse_binding`] feature must be enabled on the device.
    ///
    /// [`bind_memory`]: sys::RawBuffer::bind_memory
    /// [`sparse_binding`]: crate::device::Features::sparse_binding
    SPARSE_BINDING = SPARSE_BINDING,*/

    /* TODO: enable
    /// The buffer can be used without being fully resident in memory at the time of use.
    ///
    /// This requires the `sparse_binding` flag as well.
    ///
    /// The [`sparse_residency_buffer`] feature must be enabled on the device.
    ///
    /// [`sparse_residency_buffer`]: crate::device::Features::sparse_residency_buffer
    SPARSE_RESIDENCY = SPARSE_RESIDENCY,*/

    /* TODO: enable
    /// The buffer's memory can alias with another buffer or a different part of the same buffer.
    ///
    /// This requires the `sparse_binding` flag as well.
    ///
    /// The [`sparse_residency_aliased`] feature must be enabled on the device.
    ///
    /// [`sparse_residency_aliased`]: crate::device::Features::sparse_residency_aliased
    SPARSE_ALIASED = SPARSE_ALIASED,*/

    /* TODO: enable
    /// The buffer is protected, and can only be used in combination with protected memory and other
    /// protected objects.
    ///
    /// The device API version must be at least 1.1.
    PROTECTED = PROTECTED {
        api_version: V1_1,
    },*/

    /* TODO: enable
    /// The buffer's device address can be saved and reused on a subsequent run.
    ///
    /// The device API version must be at least 1.2, or either the [`khr_buffer_device_address`] or
    /// [`ext_buffer_device_address`] extension must be enabled on the device.
    DEVICE_ADDRESS_CAPTURE_REPLAY = DEVICE_ADDRESS_CAPTURE_REPLAY {
        api_version: V1_2,
        device_extensions: [khr_buffer_device_address, ext_buffer_device_address],
    },*/
}

/// Trait for types of data that can be put in a buffer. These can be safely transmuted to and from
/// a slice of bytes.
pub unsafe trait BufferContents: Send + Sync + 'static {
    /// Converts an immutable reference to `Self` to an immutable byte slice.
    fn as_bytes(&self) -> &[u8];

    /// Converts a mutable reference to `Self` to a mutable byte slice.
    fn as_bytes_mut(&mut self) -> &mut [u8];

    /// Converts an immutable byte slice into an immutable reference to `Self`.
    fn from_bytes(bytes: &[u8]) -> Result<&Self, PodCastError>;

    /// Converts a mutable byte slice into a mutable reference to `Self`.
    fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut Self, PodCastError>;

    /// Returns the size of an element of the type.
    fn size_of_element() -> DeviceSize;
}

unsafe impl<T> BufferContents for T
where
    T: Pod + Send + Sync,
{
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        bytemuck::bytes_of_mut(self)
    }

    fn from_bytes(bytes: &[u8]) -> Result<&T, PodCastError> {
        bytemuck::try_from_bytes(bytes)
    }

    fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut T, PodCastError> {
        bytemuck::try_from_bytes_mut(bytes)
    }

    fn size_of_element() -> DeviceSize {
        1
    }
}

unsafe impl<T> BufferContents for [T]
where
    T: Pod + Send + Sync,
{
    fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(self)
    }

    fn from_bytes(bytes: &[u8]) -> Result<&[T], PodCastError> {
        bytemuck::try_cast_slice(bytes)
    }

    fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut [T], PodCastError> {
        bytemuck::try_cast_slice_mut(bytes)
    }

    fn size_of_element() -> DeviceSize {
        size_of::<T>() as DeviceSize
    }
}

/// The buffer configuration to query in
/// [`PhysicalDevice::external_buffer_properties`](crate::device::physical::PhysicalDevice::external_buffer_properties).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExternalBufferInfo {
    /// The external handle type that will be used with the buffer.
    pub handle_type: ExternalMemoryHandleType,

    /// The usage that the buffer will have.
    pub usage: BufferUsage,

    /// The sparse binding parameters that will be used.
    pub sparse: Option<BufferCreateFlags>,

    pub _ne: crate::NonExhaustive,
}

impl ExternalBufferInfo {
    /// Returns an `ExternalBufferInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalMemoryHandleType) -> Self {
        Self {
            handle_type,
            usage: BufferUsage::empty(),
            sparse: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The external memory properties supported for buffers with a given configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExternalBufferProperties {
    /// The properties for external memory.
    pub external_memory_properties: ExternalMemoryProperties,
}
