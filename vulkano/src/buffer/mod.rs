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
//!   a reference to a portion of a `Buffer`. `Subbuffer` also has a type parameter, which is a
//!   hint for how the data in the portion of the buffer is going to be interpreted.
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
//! - [`MemoryUsage::DeviceOnly`] will allocate a buffer that's usually located in device-local
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
//! [`MemoryUsage::DeviceOnly`]: crate::memory::allocator::MemoryUsage::DeviceOnly
//! [`MemoryUsage::Upload`]: crate::memory::allocator::MemoryUsage::Upload
//! [`MemoryUsage::Download`]: crate::memory::allocator::MemoryUsage::Download
//! [the `view` module]: self::view
//! [the `shader` module documentation]: crate::shader

use self::sys::RawBuffer;
pub use self::{subbuffer::*, sys::*, usage::*};
use crate::{
    device::{physical::PhysicalDevice, Device, DeviceOwned},
    macros::{vulkan_bitflags, vulkan_enum},
    memory::{
        allocator::{
            AllocationCreateInfo, AllocationType, DeviceLayout, MemoryAlloc, MemoryAllocator,
            MemoryAllocatorError,
        },
        is_aligned, DedicatedAllocation, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        ExternalMemoryProperties, MemoryRequirements,
    },
    range_map::RangeMap,
    sync::{future::AccessError, AccessConflict, CurrentAccess, Sharing},
    DeviceSize, NonZeroDeviceSize, Requires, RequiresAllOf, RequiresOneOf, RuntimeError,
    ValidationError, Version, VulkanError, VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
use std::{
    error::Error,
    fmt::{Display, Formatter},
    hash::{Hash, Hasher},
    ops::Range,
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
///     buffer::{BufferUsage, Buffer, BufferCreateInfo},
///     command_buffer::{
///         AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
///         PrimaryCommandBufferAbstract,
///     },
///     memory::allocator::{AllocationCreateInfo, MemoryUsage},
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
///     BufferCreateInfo {
///         // Specify that this buffer will be used as a transfer source.
///         usage: BufferUsage::TRANSFER_SRC,
///         ..Default::default()
///     },
///     AllocationCreateInfo {
///         // Specify use for upload to the device.
///         usage: MemoryUsage::Upload,
///         ..Default::default()
///     },
///     data,
/// )
/// .unwrap();
///
/// // Create a buffer in device-local with enough space for a slice of `10_000` floats.
/// let device_local_buffer = Buffer::new_slice::<f32>(
///     &memory_allocator,
///     BufferCreateInfo {
///         // Specify use as a storage buffer and transfer destination.
///         usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
///         ..Default::default()
///     },
///     AllocationCreateInfo {
///         // Specify use by the device only.
///         usage: MemoryUsage::DeviceOnly,
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
    /// > **Note**: This only works with memory types that are host-visible. If you want to upload
    /// > data to a buffer allocated in device-local memory, you will need to create a staging
    /// > buffer and copy the contents over.
    ///
    /// # Panics
    ///
    /// - Panics if `buffer_info.size` is not zero.
    /// - Panics if the chosen memory type is not host-visible.
    pub fn from_data<T>(
        allocator: &(impl MemoryAllocator + ?Sized),
        buffer_info: BufferCreateInfo,
        allocation_info: AllocationCreateInfo,
        data: T,
    ) -> Result<Subbuffer<T>, BufferAllocateError>
    where
        T: BufferContents,
    {
        let buffer = Buffer::new_sized(allocator, buffer_info, allocation_info)?;

        {
            let mut write_guard = buffer
                .write()
                .expect("the buffer is somehow in use before we returned it to the user");
            *write_guard = data;
        }

        Ok(buffer)
    }

    /// Creates a new `Buffer` and writes all elements of `iter` in it. Returns a [`Subbuffer`]
    /// spanning the whole buffer.
    ///
    /// > **Note**: This only works with memory types that are host-visible. If you want to upload
    /// > data to a buffer allocated in device-local memory, you will need to create a staging
    /// > buffer and copy the contents over.
    ///
    /// # Panics
    ///
    /// - Panics if `buffer_info.size` is not zero.
    /// - Panics if the chosen memory type is not host-visible.
    /// - Panics if `iter` is empty.
    pub fn from_iter<T, I>(
        allocator: &(impl MemoryAllocator + ?Sized),
        buffer_info: BufferCreateInfo,
        allocation_info: AllocationCreateInfo,
        iter: I,
    ) -> Result<Subbuffer<[T]>, BufferAllocateError>
    where
        T: BufferContents,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        let buffer = Buffer::new_slice(
            allocator,
            buffer_info,
            allocation_info,
            iter.len().try_into().unwrap(),
        )?;

        {
            let mut write_guard = buffer
                .write()
                .expect("the buffer is somehow in use before we returned it to the user");

            for (o, i) in write_guard.iter_mut().zip(iter) {
                *o = i;
            }
        }

        Ok(buffer)
    }

    /// Creates a new uninitialized `Buffer` for sized data. Returns a [`Subbuffer`] spanning the
    /// whole buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `buffer_info.size` is not zero.
    pub fn new_sized<T>(
        allocator: &(impl MemoryAllocator + ?Sized),
        buffer_info: BufferCreateInfo,
        allocation_info: AllocationCreateInfo,
    ) -> Result<Subbuffer<T>, BufferAllocateError>
    where
        T: BufferContents,
    {
        let layout = T::LAYOUT.unwrap_sized();
        let buffer = Subbuffer::new(Buffer::new(
            allocator,
            buffer_info,
            allocation_info,
            layout,
        )?);

        Ok(unsafe { buffer.reinterpret_unchecked() })
    }

    /// Creates a new uninitialized `Buffer` for a slice. Returns a [`Subbuffer`] spanning the
    /// whole buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `buffer_info.size` is not zero.
    /// - Panics if `len` is zero.
    pub fn new_slice<T>(
        allocator: &(impl MemoryAllocator + ?Sized),
        buffer_info: BufferCreateInfo,
        allocation_info: AllocationCreateInfo,
        len: DeviceSize,
    ) -> Result<Subbuffer<[T]>, BufferAllocateError>
    where
        T: BufferContents,
    {
        Buffer::new_unsized(allocator, buffer_info, allocation_info, len)
    }

    /// Creates a new uninitialized `Buffer` for unsized data. Returns a [`Subbuffer`] spanning the
    /// whole buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `buffer_info.size` is not zero.
    /// - Panics if `len` is zero.
    pub fn new_unsized<T>(
        allocator: &(impl MemoryAllocator + ?Sized),
        buffer_info: BufferCreateInfo,
        allocation_info: AllocationCreateInfo,
        len: DeviceSize,
    ) -> Result<Subbuffer<T>, BufferAllocateError>
    where
        T: BufferContents + ?Sized,
    {
        let len = NonZeroDeviceSize::new(len).expect("empty slices are not valid buffer contents");
        let layout = T::LAYOUT.layout_for_len(len).unwrap();
        let buffer = Subbuffer::new(Buffer::new(
            allocator,
            buffer_info,
            allocation_info,
            layout,
        )?);

        Ok(unsafe { buffer.reinterpret_unchecked() })
    }

    /// Creates a new uninitialized `Buffer` with the given `layout`.
    ///
    /// # Panics
    ///
    /// - Panics if `buffer_info.size` is not zero.
    /// - Panics if `layout.alignment()` is greater than 64.
    pub fn new(
        allocator: &(impl MemoryAllocator + ?Sized),
        mut buffer_info: BufferCreateInfo,
        allocation_info: AllocationCreateInfo,
        layout: DeviceLayout,
    ) -> Result<Arc<Self>, BufferAllocateError> {
        assert!(layout.alignment().as_devicesize() <= 64);
        // TODO: Enable once sparse binding materializes
        // assert!(!allocate_info.flags.contains(BufferCreateFlags::SPARSE_BINDING));

        assert!(
            buffer_info.size == 0,
            "`Buffer::new*` functions set the `buffer_info.size` field themselves, you should not \
            set it yourself",
        );

        buffer_info.size = layout.size();

        let raw_buffer = RawBuffer::new(allocator.device().clone(), buffer_info)
            .map_err(BufferAllocateError::CreateBuffer)?;
        let mut requirements = *raw_buffer.memory_requirements();
        requirements.layout = requirements.layout.align_to(layout.alignment()).unwrap();

        let mut allocation = unsafe {
            allocator
                .allocate_unchecked(
                    requirements,
                    AllocationType::Linear,
                    allocation_info,
                    Some(DedicatedAllocation::Buffer(&raw_buffer)),
                )
                .map_err(BufferAllocateError::AllocateMemory)?
        };
        debug_assert!(is_aligned(
            allocation.offset(),
            requirements.layout.alignment(),
        ));
        debug_assert!(allocation.size() == requirements.layout.size());

        // The implementation might require a larger size than we wanted. With this it is easier to
        // invalidate and flush the whole buffer. It does not affect the allocation in any way.
        allocation.shrink(layout.size());

        unsafe { raw_buffer.bind_memory_unchecked(allocation) }
            .map(Arc::new)
            .map_err(|(err, _, _)| BufferAllocateError::BindMemory(err))
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
    pub fn device_address(&self) -> Result<NonZeroDeviceSize, ValidationError> {
        self.validate_device_address()?;

        unsafe { Ok(self.device_address_unchecked()) }
    }

    fn validate_device_address(&self) -> Result<(), ValidationError> {
        let device = self.device();

        if !device.enabled_features().buffer_device_address {
            return Err(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "buffer_device_address",
                )])]),
                vuids: &["VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324"],
                ..Default::default()
            });
        }

        if !self.usage().intersects(BufferUsage::SHADER_DEVICE_ADDRESS) {
            return Err(ValidationError {
                context: "self.usage()".into(),
                problem: "does not contain `BufferUsage::SHADER_DEVICE_ADDRESS`".into(),
                vuids: &["VUID-VkBufferDeviceAddressInfo-buffer-02601"],
                ..Default::default()
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn device_address_unchecked(&self) -> NonZeroDeviceSize {
        let device = self.device();

        let info_vk = ash::vk::BufferDeviceAddressInfo {
            buffer: self.handle(),
            ..Default::default()
        };

        let ptr = {
            let fns = device.fns();
            let f = if device.api_version() >= Version::V1_2 {
                fns.v1_2.get_buffer_device_address
            } else if device.enabled_extensions().khr_buffer_device_address {
                fns.khr_buffer_device_address.get_buffer_device_address_khr
            } else {
                fns.ext_buffer_device_address.get_buffer_device_address_ext
            };
            f(device.handle(), &info_vk)
        };

        NonZeroDeviceSize::new(ptr).unwrap()
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

/// Error that can happen when allocating a new buffer.
#[derive(Clone, Debug)]
pub enum BufferAllocateError {
    CreateBuffer(VulkanError),
    AllocateMemory(MemoryAllocatorError),
    BindMemory(RuntimeError),
}

impl Error for BufferAllocateError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CreateBuffer(err) => Some(err),
            Self::AllocateMemory(err) => Some(err),
            Self::BindMemory(err) => Some(err),
        }
    }
}

impl Display for BufferAllocateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateBuffer(_) => write!(f, "creating the buffer failed"),
            Self::AllocateMemory(_) => write!(f, "allocating memory for the buffer failed"),
            Self::BindMemory(_) => write!(f, "binding memory to the buffer failed"),
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

    pub(crate) fn check_cpu_read(&self, range: Range<DeviceSize>) -> Result<(), AccessConflict> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::CpuExclusive { .. } => return Err(AccessConflict::HostWrite),
                CurrentAccess::GpuExclusive { .. } => return Err(AccessConflict::DeviceWrite),
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

    pub(crate) fn check_cpu_write(&self, range: Range<DeviceSize>) -> Result<(), AccessConflict> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::CpuExclusive => return Err(AccessConflict::HostWrite),
                CurrentAccess::GpuExclusive { .. } => return Err(AccessConflict::DeviceWrite),
                CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads: 0,
                } => (),
                CurrentAccess::Shared { cpu_reads, .. } if *cpu_reads > 0 => {
                    return Err(AccessConflict::HostRead)
                }
                CurrentAccess::Shared { .. } => return Err(AccessConflict::DeviceRead),
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

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a buffer.
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
    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),*/

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

    pub(crate) fn validate(&self, physical_device: &PhysicalDevice) -> Result<(), ValidationError> {
        let &Self {
            handle_type,
            usage,
            sparse: _,
            _ne: _,
        } = self;

        usage
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "usage".into(),
                vuids: &["VUID-VkPhysicalDeviceExternalBufferInfo-usage-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if usage.is_empty() {
            return Err(ValidationError {
                context: "usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkPhysicalDeviceExternalBufferInfo-usage-requiredbitmask"],
                ..Default::default()
            });
        }

        handle_type
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "handle_type".into(),
                vuids: &["VUID-VkPhysicalDeviceExternalBufferInfo-handleType-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        Ok(())
    }
}

/// The external memory properties supported for buffers with a given configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExternalBufferProperties {
    /// The properties for external memory.
    pub external_memory_properties: ExternalMemoryProperties,
}

vulkan_enum! {
    #[non_exhaustive]

    /// An enumeration of all valid index types.
    IndexType = IndexType(i32);

    /// Indices are 8-bit unsigned integers.
    U8 = UINT8_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_index_type_uint8)]),
    ]),

    /// Indices are 16-bit unsigned integers.
    U16 = UINT16,

    /// Indices are 32-bit unsigned integers.
    U32 = UINT32,
}

impl IndexType {
    /// Returns the size in bytes of indices of this type.
    #[inline]
    pub fn size(self) -> DeviceSize {
        match self {
            IndexType::U8 => 1,
            IndexType::U16 => 2,
            IndexType::U32 => 4,
        }
    }
}

/// A buffer holding index values, which index into buffers holding vertex data.
#[derive(Clone, Debug)]
pub enum IndexBuffer {
    /// An index buffer containing unsigned 8-bit indices.
    ///
    /// The [`index_type_uint8`] feature must be enabled on the device.
    ///
    /// [`index_type_uint8`]: crate::device::Features::index_type_uint8
    U8(Subbuffer<[u8]>),

    /// An index buffer containing unsigned 16-bit indices.
    U16(Subbuffer<[u16]>),

    /// An index buffer containing unsigned 32-bit indices.
    U32(Subbuffer<[u32]>),
}

impl IndexBuffer {
    #[inline]
    pub fn index_type(&self) -> IndexType {
        match self {
            Self::U8(_) => IndexType::U8,
            Self::U16(_) => IndexType::U16,
            Self::U32(_) => IndexType::U32,
        }
    }

    #[inline]
    pub(crate) fn as_bytes(&self) -> &Subbuffer<[u8]> {
        match self {
            IndexBuffer::U8(buffer) => buffer.as_bytes(),
            IndexBuffer::U16(buffer) => buffer.as_bytes(),
            IndexBuffer::U32(buffer) => buffer.as_bytes(),
        }
    }
}

impl From<Subbuffer<[u8]>> for IndexBuffer {
    #[inline]
    fn from(value: Subbuffer<[u8]>) -> Self {
        Self::U8(value)
    }
}

impl From<Subbuffer<[u16]>> for IndexBuffer {
    #[inline]
    fn from(value: Subbuffer<[u16]>) -> Self {
        Self::U16(value)
    }
}

impl From<Subbuffer<[u32]>> for IndexBuffer {
    #[inline]
    fn from(value: Subbuffer<[u32]>) -> Self {
        Self::U32(value)
    }
}
