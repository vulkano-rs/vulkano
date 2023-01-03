// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low level implementation of buffers.
//!
//! This module contains low-level wrappers around the Vulkan buffer types. All
//! other buffer types of this library, and all custom buffer types
//! that you create must wrap around the types in this module.

use super::{
    cpu_access::{ReadLockError, WriteLockError},
    BufferContents, BufferCreateFlags, BufferUsage,
};
use crate::{
    device::{Device, DeviceOwned},
    memory::{
        allocator::{
            AllocationCreationError, AllocationType, DeviceAlignment, DeviceLayout, MemoryAlloc,
        },
        DedicatedTo, ExternalMemoryHandleType, ExternalMemoryHandleTypes, MemoryAllocateFlags,
        MemoryPropertyFlags, MemoryRequirements,
    },
    range_map::RangeMap,
    sync::{future::AccessError, CurrentAccess, Sharing},
    DeviceSize, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    mem::{size_of_val, MaybeUninit},
    num::NonZeroU64,
    ops::{Deref, DerefMut, Range},
    ptr,
    sync::Arc,
};

/// A raw buffer, with no memory backing it.
///
/// This is the basic buffer type, a direct translation of a `VkBuffer` object, but it is mostly
/// useless in this form. After creating a raw buffer, you must call `bind_memory` to make a
/// complete buffer object.
#[derive(Debug)]
pub struct RawBuffer {
    handle: ash::vk::Buffer,
    device: Arc<Device>,
    id: NonZeroU64,

    flags: BufferCreateFlags,
    size: DeviceSize,
    usage: BufferUsage,
    sharing: Sharing<SmallVec<[u32; 4]>>,
    external_memory_handle_types: ExternalMemoryHandleTypes,

    memory_requirements: MemoryRequirements,
}

impl RawBuffer {
    /// Creates a new `RawBuffer`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.sharing` is [`Concurrent`](Sharing::Concurrent) with less than 2
    ///   items.
    /// - Panics if `create_info.size` is zero.
    /// - Panics if `create_info.usage` is empty.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        mut create_info: BufferCreateInfo,
    ) -> Result<Self, BufferError> {
        match &mut create_info.sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                // VUID-VkBufferCreateInfo-sharingMode-01419
                queue_family_indices.sort_unstable();
                queue_family_indices.dedup();
            }
        }

        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(device: &Device, create_info: &BufferCreateInfo) -> Result<(), BufferError> {
        let &BufferCreateInfo {
            flags,
            ref sharing,
            size,
            usage,
            external_memory_handle_types,
            _ne: _,
        } = create_info;

        // VUID-VkBufferCreateInfo-flags-parameter
        flags.validate_device(device)?;

        // VUID-VkBufferCreateInfo-usage-parameter
        usage.validate_device(device)?;

        // VUID-VkBufferCreateInfo-usage-requiredbitmask
        assert!(!usage.is_empty());

        // VUID-VkBufferCreateInfo-size-00912
        assert!(size != 0);

        /* Enable when sparse binding is properly handled
        if let Some(sparse_level) = sparse {
            // VUID-VkBufferCreateInfo-flags-00915
            if !device.enabled_features().sparse_binding {
                return Err(BufferError::RequirementNotMet {
                    required_for: "`create_info.sparse` is `Some`",
                    requires_one_of: RequiresOneOf {
                        features: &["sparse_binding"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkBufferCreateInfo-flags-00916
            if sparse_level.sparse_residency && !device.enabled_features().sparse_residency_buffer {
                return Err(BufferError::RequirementNotMet {
                    required_for: "`create_info.sparse` is `Some(sparse_level)`, where \
                        `sparse_level` contains `BufferCreateFlags::SPARSE_RESIDENCY`",
                    requires_one_of: RequiresOneOf {
                        features: &["sparse_residency_buffer"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkBufferCreateInfo-flags-00917
            if sparse_level.sparse_aliased && !device.enabled_features().sparse_residency_aliased {
                return Err(BufferError::RequirementNotMet {
                    required_for: "`create_info.sparse` is `Some(sparse_level)`, where \
                        `sparse_level` contains `BufferCreateFlags::SPARSE_ALIASED`",
                    requires_one_of: RequiresOneOf {
                        features: &["sparse_residency_aliased"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkBufferCreateInfo-flags-00918
        }
        */

        match sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                // VUID-VkBufferCreateInfo-sharingMode-00914
                assert!(queue_family_indices.len() >= 2);

                for &queue_family_index in queue_family_indices.iter() {
                    // VUID-VkBufferCreateInfo-sharingMode-01419
                    if queue_family_index
                        >= device.physical_device().queue_family_properties().len() as u32
                    {
                        return Err(BufferError::SharingQueueFamilyIndexOutOfRange {
                            queue_family_index,
                            queue_family_count: device
                                .physical_device()
                                .queue_family_properties()
                                .len() as u32,
                        });
                    }
                }
            }
        }

        if let Some(max_buffer_size) = device.physical_device().properties().max_buffer_size {
            // VUID-VkBufferCreateInfo-size-06409
            if size > max_buffer_size {
                return Err(BufferError::MaxBufferSizeExceeded {
                    size,
                    max: max_buffer_size,
                });
            }
        }

        if !external_memory_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_memory)
            {
                return Err(BufferError::RequirementNotMet {
                    required_for: "`create_info.external_memory_handle_types` is not empty",
                    requires_one_of: RequiresOneOf {
                        api_version: Some(Version::V1_1),
                        device_extensions: &["khr_external_memory"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkExternalMemoryBufferCreateInfo-handleTypes-parameter
            external_memory_handle_types.validate_device(device)?;

            // VUID-VkBufferCreateInfo-pNext-00920
            // TODO:
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: BufferCreateInfo,
    ) -> Result<Self, VulkanError> {
        let &BufferCreateInfo {
            flags,
            ref sharing,
            size,
            usage,
            external_memory_handle_types,
            _ne: _,
        } = &create_info;

        let (sharing_mode, queue_family_index_count, p_queue_family_indices) = match sharing {
            Sharing::Exclusive => (ash::vk::SharingMode::EXCLUSIVE, 0, &[] as _),
            Sharing::Concurrent(queue_family_indices) => (
                ash::vk::SharingMode::CONCURRENT,
                queue_family_indices.len() as u32,
                queue_family_indices.as_ptr(),
            ),
        };

        let mut create_info_vk = ash::vk::BufferCreateInfo {
            flags: flags.into(),
            size,
            usage: usage.into(),
            sharing_mode,
            queue_family_index_count,
            p_queue_family_indices,
            ..Default::default()
        };
        let mut external_memory_info_vk = None;

        if !external_memory_handle_types.is_empty() {
            let _ = external_memory_info_vk.insert(ash::vk::ExternalMemoryBufferCreateInfo {
                handle_types: external_memory_handle_types.into(),
                ..Default::default()
            });
        }

        if let Some(next) = external_memory_info_vk.as_mut() {
            next.p_next = create_info_vk.p_next;
            create_info_vk.p_next = next as *const _ as *const _;
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_buffer)(
                device.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(device, handle, create_info))
    }

    /// Creates a new `RawBuffer` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `handle` must refer to a buffer that has not yet had memory bound to it.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Buffer,
        create_info: BufferCreateInfo,
    ) -> Self {
        let BufferCreateInfo {
            flags,
            size,
            usage,
            sharing,
            external_memory_handle_types,
            _ne: _,
        } = create_info;

        let mut memory_requirements = Self::get_memory_requirements(&device, handle);

        debug_assert!(memory_requirements.layout.size() >= size);
        debug_assert!(memory_requirements.memory_type_bits != 0);

        // We have to manually enforce some additional requirements for some buffer types.
        let properties = device.physical_device().properties();
        if usage.intersects(BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER) {
            memory_requirements.layout = memory_requirements
                .layout
                .align_to(
                    DeviceAlignment::new(properties.min_texel_buffer_offset_alignment).unwrap(),
                )
                .unwrap();
        }

        if usage.intersects(BufferUsage::STORAGE_BUFFER) {
            memory_requirements.layout = memory_requirements
                .layout
                .align_to(
                    DeviceAlignment::new(properties.min_storage_buffer_offset_alignment).unwrap(),
                )
                .unwrap();
        }

        if usage.intersects(BufferUsage::UNIFORM_BUFFER) {
            memory_requirements.layout = memory_requirements
                .layout
                .align_to(
                    DeviceAlignment::new(properties.min_uniform_buffer_offset_alignment).unwrap(),
                )
                .unwrap();
        }

        RawBuffer {
            handle,
            device,
            id: Self::next_id(),
            flags,
            size,
            usage,
            sharing,
            external_memory_handle_types,
            memory_requirements,
        }
    }

    fn get_memory_requirements(device: &Device, handle: ash::vk::Buffer) -> MemoryRequirements {
        let info_vk = ash::vk::BufferMemoryRequirementsInfo2 {
            buffer: handle,
            ..Default::default()
        };

        let mut memory_requirements2_vk = ash::vk::MemoryRequirements2::default();
        let mut memory_dedicated_requirements_vk = None;

        if device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation
        {
            debug_assert!(
                device.api_version() >= Version::V1_1
                    || device.enabled_extensions().khr_get_memory_requirements2
            );

            let next = memory_dedicated_requirements_vk
                .insert(ash::vk::MemoryDedicatedRequirements::default());

            next.p_next = memory_requirements2_vk.p_next;
            memory_requirements2_vk.p_next = next as *mut _ as *mut _;
        }

        unsafe {
            let fns = device.fns();

            if device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_get_memory_requirements2
            {
                if device.api_version() >= Version::V1_1 {
                    (fns.v1_1.get_buffer_memory_requirements2)(
                        device.handle(),
                        &info_vk,
                        &mut memory_requirements2_vk,
                    );
                } else {
                    (fns.khr_get_memory_requirements2
                        .get_buffer_memory_requirements2_khr)(
                        device.handle(),
                        &info_vk,
                        &mut memory_requirements2_vk,
                    );
                }
            } else {
                (fns.v1_0.get_buffer_memory_requirements)(
                    device.handle(),
                    handle,
                    &mut memory_requirements2_vk.memory_requirements,
                );
            }
        }

        MemoryRequirements {
            layout: DeviceLayout::from_size_alignment(
                memory_requirements2_vk.memory_requirements.size,
                memory_requirements2_vk.memory_requirements.alignment,
            )
            .unwrap(),
            memory_type_bits: memory_requirements2_vk.memory_requirements.memory_type_bits,
            prefers_dedicated_allocation: memory_dedicated_requirements_vk
                .map_or(false, |dreqs| dreqs.prefers_dedicated_allocation != 0),
            requires_dedicated_allocation: memory_dedicated_requirements_vk
                .map_or(false, |dreqs| dreqs.requires_dedicated_allocation != 0),
        }
    }

    pub(crate) fn id(&self) -> NonZeroU64 {
        self.id
    }

    /// Binds device memory to this buffer.
    pub fn bind_memory(
        self,
        allocation: MemoryAlloc,
    ) -> Result<Buffer, (BufferError, RawBuffer, MemoryAlloc)> {
        if let Err(err) = self.validate_bind_memory(&allocation) {
            return Err((err, self, allocation));
        }

        unsafe { self.bind_memory_unchecked(allocation) }
            .map_err(|(err, buffer, allocation)| (err.into(), buffer, allocation))
    }

    fn validate_bind_memory(&self, allocation: &MemoryAlloc) -> Result<(), BufferError> {
        assert_ne!(allocation.allocation_type(), AllocationType::NonLinear);

        let memory_requirements = &self.memory_requirements;
        let memory = allocation.device_memory();
        let memory_offset = allocation.offset();
        let memory_type = &self
            .device
            .physical_device()
            .memory_properties()
            .memory_types[memory.memory_type_index() as usize];

        // VUID-VkBindBufferMemoryInfo-commonparent
        assert_eq!(self.device(), memory.device());

        // VUID-VkBindBufferMemoryInfo-buffer-07459
        // Ensured by taking ownership of `RawBuffer`.

        // VUID-VkBindBufferMemoryInfo-buffer-01030
        // Currently ensured by not having sparse binding flags, but this needs to be checked once
        // those are enabled.

        // VUID-VkBindBufferMemoryInfo-memoryOffset-01031
        // Assume that `allocation` was created correctly.

        // VUID-VkBindBufferMemoryInfo-memory-01035
        if memory_requirements.memory_type_bits & (1 << memory.memory_type_index()) == 0 {
            return Err(BufferError::MemoryTypeNotAllowed {
                provided_memory_type_index: memory.memory_type_index(),
                allowed_memory_type_bits: memory_requirements.memory_type_bits,
            });
        }

        // VUID-VkBindBufferMemoryInfo-memoryOffset-01036
        if memory_offset % memory_requirements.layout.alignment().as_nonzero() != 0 {
            return Err(BufferError::MemoryAllocationNotAligned {
                allocation_offset: memory_offset,
                required_alignment: memory_requirements.layout.alignment(),
            });
        }

        // VUID-VkBindBufferMemoryInfo-size-01037
        if allocation.size() < memory_requirements.layout.size() {
            return Err(BufferError::MemoryAllocationTooSmall {
                allocation_size: allocation.size(),
                required_size: memory_requirements.layout.size(),
            });
        }

        if let Some(dedicated_to) = memory.dedicated_to() {
            // VUID-VkBindBufferMemoryInfo-memory-01508
            match dedicated_to {
                DedicatedTo::Buffer(id) if id == self.id => {}
                _ => return Err(BufferError::DedicatedAllocationMismatch),
            }
            debug_assert!(memory_offset == 0); // This should be ensured by the allocator
        } else {
            // VUID-VkBindBufferMemoryInfo-buffer-01444
            if memory_requirements.requires_dedicated_allocation {
                return Err(BufferError::DedicatedAllocationRequired);
            }
        }

        // VUID-VkBindBufferMemoryInfo-None-01899
        if memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::PROTECTED)
        {
            return Err(BufferError::MemoryProtectedMismatch {
                buffer_protected: false,
                memory_protected: true,
            });
        }

        // VUID-VkBindBufferMemoryInfo-memory-02726
        if !memory.export_handle_types().is_empty()
            && !memory
                .export_handle_types()
                .intersects(self.external_memory_handle_types)
        {
            return Err(BufferError::MemoryExternalHandleTypesDisjoint {
                buffer_handle_types: self.external_memory_handle_types,
                memory_export_handle_types: memory.export_handle_types(),
            });
        }

        if let Some(handle_type) = memory.imported_handle_type() {
            // VUID-VkBindBufferMemoryInfo-memory-02985
            if !ExternalMemoryHandleTypes::from(handle_type)
                .intersects(self.external_memory_handle_types)
            {
                return Err(BufferError::MemoryImportedHandleTypeNotEnabled {
                    buffer_handle_types: self.external_memory_handle_types,
                    memory_imported_handle_type: handle_type,
                });
            }
        }

        // VUID-VkBindBufferMemoryInfo-bufferDeviceAddress-03339
        if !self.device.enabled_extensions().ext_buffer_device_address
            && self.usage.intersects(BufferUsage::SHADER_DEVICE_ADDRESS)
            && !memory
                .flags()
                .intersects(MemoryAllocateFlags::DEVICE_ADDRESS)
        {
            return Err(BufferError::MemoryBufferDeviceAddressNotSupported);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_memory_unchecked(
        self,
        allocation: MemoryAlloc,
    ) -> Result<Buffer, (VulkanError, RawBuffer, MemoryAlloc)> {
        let memory = allocation.device_memory();
        let memory_offset = allocation.offset();

        let fns = self.device.fns();

        let result = if self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_bind_memory2
        {
            let bind_infos_vk = [ash::vk::BindBufferMemoryInfo {
                buffer: self.handle,
                memory: memory.handle(),
                memory_offset,
                ..Default::default()
            }];

            if self.device.api_version() >= Version::V1_1 {
                (fns.v1_1.bind_buffer_memory2)(
                    self.device.handle(),
                    bind_infos_vk.len() as u32,
                    bind_infos_vk.as_ptr(),
                )
            } else {
                (fns.khr_bind_memory2.bind_buffer_memory2_khr)(
                    self.device.handle(),
                    bind_infos_vk.len() as u32,
                    bind_infos_vk.as_ptr(),
                )
            }
        } else {
            (fns.v1_0.bind_buffer_memory)(
                self.device.handle(),
                self.handle,
                memory.handle(),
                memory_offset,
            )
        }
        .result();

        if let Err(err) = result {
            return Err((VulkanError::from(err), self, allocation));
        }

        Ok(Buffer::from_raw(self, BufferMemory::Normal(allocation)))
    }

    /// Returns the memory requirements for this buffer.
    pub fn memory_requirements(&self) -> &MemoryRequirements {
        &self.memory_requirements
    }

    /// Returns the flags the buffer was created with.
    #[inline]
    pub fn flags(&self) -> BufferCreateFlags {
        self.flags
    }

    /// Returns the size of the buffer in bytes.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    /// Returns the usage the buffer was created with.
    #[inline]
    pub fn usage(&self) -> &BufferUsage {
        &self.usage
    }

    /// Returns the sharing the buffer was created with.
    #[inline]
    pub fn sharing(&self) -> &Sharing<SmallVec<[u32; 4]>> {
        &self.sharing
    }

    /// Returns the external memory handle types that are supported with this buffer.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.external_memory_handle_types
    }
}

impl Drop for RawBuffer {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_buffer)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for RawBuffer {
    type Handle = ash::vk::Buffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for RawBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

crate::impl_id_counter!(RawBuffer);

/// Parameters to create a new `Buffer`.
#[derive(Clone, Debug)]
pub struct BufferCreateInfo {
    /// Flags to enable.
    ///
    /// The default value is [`BufferCreateFlags::empty()`].
    pub flags: BufferCreateFlags,

    /// Whether the buffer can be shared across multiple queues, or is limited to a single queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub sharing: Sharing<SmallVec<[u32; 4]>>,

    /// The size in bytes of the buffer.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: DeviceSize,

    /// How the buffer is going to be used.
    ///
    /// The default value is [`BufferUsage::empty()`], which must be overridden.
    pub usage: BufferUsage,

    /// The external memory handle types that are going to be used with the buffer.
    ///
    /// If this value is not empty, then the device API version must be at least 1.1, or the
    /// [`khr_external_memory`] extension must be enabled on the device.
    ///
    /// The default value is [`ExternalMemoryHandleTypes::empty()`].
    ///
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    pub external_memory_handle_types: ExternalMemoryHandleTypes,

    pub _ne: crate::NonExhaustive,
}

impl Default for BufferCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: BufferCreateFlags::empty(),
            sharing: Sharing::Exclusive,
            size: 0,
            usage: BufferUsage::empty(),
            external_memory_handle_types: ExternalMemoryHandleTypes::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A storage for raw bytes.
///
/// Unlike [`RawBuffer`], a `Buffer` has memory backing it, and can be used normally.
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
    fn from_raw(inner: RawBuffer, memory: BufferMemory) -> Self {
        let state = Mutex::new(BufferState::new(inner.size));

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
        &self.inner.memory_requirements
    }

    /// Returns the flags the buffer was created with.
    #[inline]
    pub fn flags(&self) -> BufferCreateFlags {
        self.inner.flags
    }

    /// Returns the size of the buffer in bytes.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.inner.size
    }

    /// Returns the usage the buffer was created with.
    #[inline]
    pub fn usage(&self) -> &BufferUsage {
        &self.inner.usage
    }

    /// Returns the sharing the buffer was created with.
    #[inline]
    pub fn sharing(&self) -> &Sharing<SmallVec<[u32; 4]>> {
        &self.inner.sharing
    }

    /// Returns the external memory handle types that are supported with this buffer.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.inner.external_memory_handle_types
    }

    /// Locks the buffer in order to read its content from the host.
    ///
    /// If the buffer is currently used in exclusive mode by the device, this function will return
    /// an error. Similarly if you called `write()` on the buffer and haven't dropped the lock,
    /// this function will return an error as well.
    ///
    /// After this function successfully locks the buffer, any attempt to submit a command buffer
    /// that uses it in exclusive mode will fail. You can still submit this buffer for non-exclusive
    /// accesses (ie. reads).
    pub fn read(&self, range: Range<DeviceSize>) -> Result<BufferReadGuard<'_, [u8]>, BufferError> {
        assert!(!range.is_empty() && range.end <= self.inner.size);

        let allocation = match &self.memory {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => todo!("`Buffer::read` doesn't support sparse binding yet"),
        };

        if allocation.mapped_ptr().is_none() {
            return Err(BufferError::MemoryNotHostVisible);
        }

        let mut state = self.state();

        unsafe {
            state.check_cpu_read(range.clone())?;
            state.cpu_read_lock(range.clone());
        }

        let data = unsafe {
            // If there are other read locks being held at this point, they also called
            // `invalidate_range` when locking. The GPU can't write data while the CPU holds a read
            // lock, so there will be no new data and this call will do nothing.
            // TODO: probably still more efficient to call it only if we're the first to acquire a
            // read lock, but the number of CPU locks isn't currently tracked anywhere.
            allocation.invalidate_range(0..self.size()).unwrap();
            allocation.mapped_slice().unwrap()
        };

        Ok(BufferReadGuard {
            buffer: self,
            range,
            data,
        })
    }

    /// Locks the buffer in order to write its content from the host.
    ///
    /// If the buffer is currently in use by the device, this function will return an error.
    /// Similarly if you called `read()` on the buffer and haven't dropped the lock, this function
    /// will return an error as well.
    ///
    /// After this function successfully locks the buffer, any attempt to submit a command buffer
    /// that uses it and any attempt to call `read()` will return an error.
    pub fn write(&self, range: Range<DeviceSize>) -> Result<BufferWriteGuard<'_>, BufferError> {
        assert!(!range.is_empty() && range.end <= self.inner.size);

        let allocation = match &self.memory {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => todo!("`Buffer::write` doesn't support sparse binding yet"),
        };

        if allocation.mapped_ptr().is_none() {
            return Err(BufferError::MemoryNotHostVisible);
        }

        let mut state = self.state();

        unsafe {
            state.check_cpu_write(range.clone())?;
            state.cpu_write_lock(range.clone());
        }

        let data = unsafe {
            allocation.invalidate_range(0..self.size()).unwrap();
            allocation.write(0..self.size()).unwrap()
        };

        Ok(BufferWriteGuard {
            buffer: self,
            range,
            data,
        })
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, BufferState> {
        self.state.lock()
    }
}

unsafe impl VulkanObject for Buffer {
    type Handle = ash::vk::Buffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle
    }
}

unsafe impl DeviceOwned for Buffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.inner.device
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

/// RAII structure used to release the CPU read access of a buffer when dropped.
///
/// This structure is created by the [`read`] method on [`Buffer`].
///
/// [`read`]: Buffer::read
#[derive(Debug)]
pub struct BufferReadGuard<'a, T>
where
    T: BufferContents + ?Sized,
{
    buffer: &'a Buffer,
    range: Range<DeviceSize>,
    data: &'a T,
}

impl<'a, T> Drop for BufferReadGuard<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    fn drop(&mut self) {
        unsafe {
            let mut state = self.buffer.state();
            state.cpu_read_unlock(self.range.clone());
        }
    }
}

impl<'a, T> Deref for BufferReadGuard<'a, T>
where
    T: BufferContents + ?Sized + 'a,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

/// RAII structure used to release the CPU write access of a buffer when dropped.
///
/// This structure is created by the [`write`] method on [`Buffer`].
///
/// [`write`]: Buffer::write
#[derive(Debug)]
pub struct BufferWriteGuard<'a> {
    buffer: &'a Buffer,
    range: Range<DeviceSize>,
    data: &'a mut [u8],
}

impl<'a> Drop for BufferWriteGuard<'a> {
    fn drop(&mut self) {
        let allocation = match &self.buffer.memory {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => unreachable!(),
        };

        unsafe {
            allocation.flush_range(0..self.buffer.size()).unwrap();

            let mut state = self.buffer.state();
            state.cpu_write_unlock(self.range.clone());
        }
    }
}

impl<'a> Deref for BufferWriteGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a> DerefMut for BufferWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
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

    /// The buffer was created with the `shader_device_address` usage, but the memory does not
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
            BufferError::VulkanError(err) => Some(err),
            BufferError::AllocError(err) => Some(err),
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
                "the buffer was created with the `shader_device_address` usage, but the memory \
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
    fn from(err: VulkanError) -> BufferError {
        match err {
            VulkanError::OutOfHostMemory => BufferError::AllocError(
                AllocationCreationError::VulkanError(VulkanError::OutOfHostMemory),
            ),
            VulkanError::OutOfDeviceMemory => BufferError::AllocError(
                AllocationCreationError::VulkanError(VulkanError::OutOfDeviceMemory),
            ),
            _ => panic!("unexpected error: {:?}", err),
        }
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

#[cfg(test)]
mod tests {
    use super::{BufferCreateInfo, BufferUsage, RawBuffer};
    use crate::device::{Device, DeviceOwned};

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let buf = RawBuffer::new(
            device.clone(),
            BufferCreateInfo {
                size: 128,
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
        )
        .unwrap();
        let reqs = buf.memory_requirements();

        assert!(reqs.layout.size() >= 128);
        assert_eq!(buf.size(), 128);
        assert_eq!(&**buf.device() as *const Device, &*device as *const Device);
    }

    /* Re-enable when sparse binding is properly implemented
    #[test]
    fn missing_feature_sparse_binding() {
        let (device, _) = gfx_dev_and_queue!();
        match RawBuffer::new(
            device,
            BufferCreateInfo {
                size: 128,
                sparse: Some(BufferCreateFlags::empty()),
                usage: BufferUsage::transfer_dst,
                ..Default::default()
            },
        ) {
            Err(BufferError::RequirementNotMet {
                requires_one_of: RequiresOneOf { features, .. },
                ..
            }) if features.contains(&"sparse_binding") => (),
            _ => panic!(),
        }
    }

    #[test]
    fn missing_feature_sparse_residency() {
        let (device, _) = gfx_dev_and_queue!(sparse_binding);
        match RawBuffer::new(
            device,
            BufferCreateInfo {
                size: 128,
                sparse: Some(BufferCreateFlags {
                    sparse_residency: true,
                    sparse_aliased: false,
                    ..Default::default()
                }),
                usage: BufferUsage::transfer_dst,
                ..Default::default()
            },
        ) {
            Err(BufferError::RequirementNotMet {
                requires_one_of: RequiresOneOf { features, .. },
                ..
            }) if features.contains(&"sparse_residency_buffer") => (),
            _ => panic!(),
        }
    }

    #[test]
    fn missing_feature_sparse_aliased() {
        let (device, _) = gfx_dev_and_queue!(sparse_binding);
        match RawBuffer::new(
            device,
            BufferCreateInfo {
                size: 128,
                sparse: Some(BufferCreateFlags {
                    sparse_residency: false,
                    sparse_aliased: true,
                    ..Default::default()
                }),
                usage: BufferUsage::transfer_dst,
                ..Default::default()
            },
        ) {
            Err(BufferError::RequirementNotMet {
                requires_one_of: RequiresOneOf { features, .. },
                ..
            }) if features.contains(&"sparse_residency_aliased") => (),
            _ => panic!(),
        }
    }
    */

    #[test]
    fn create_empty_buffer() {
        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            RawBuffer::new(
                device,
                BufferCreateInfo {
                    size: 0,
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
            )
        });
    }
}
