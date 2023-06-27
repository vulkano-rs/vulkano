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

use super::{Buffer, BufferCreateFlags, BufferError, BufferMemory, BufferUsage};
use crate::{
    device::{Device, DeviceOwned},
    macros::impl_id_counter,
    memory::{
        allocator::{AllocationType, DeviceLayout, MemoryAlloc},
        is_aligned, DedicatedTo, ExternalMemoryHandleTypes, MemoryAllocateFlags,
        MemoryPropertyFlags, MemoryRequirements,
    },
    sync::Sharing,
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, RuntimeError, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::{mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

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
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature("sparse_binding")])]),
                });
            }

            // VUID-VkBufferCreateInfo-flags-00916
            if sparse_level.sparse_residency && !device.enabled_features().sparse_residency_buffer {
                return Err(BufferError::RequirementNotMet {
                    required_for: "`create_info.sparse` is `Some(sparse_level)`, where \
                        `sparse_level` contains `BufferCreateFlags::SPARSE_RESIDENCY`",
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature("sparse_residency_buffer")])]),
                });
            }

            // VUID-VkBufferCreateInfo-flags-00917
            if sparse_level.sparse_aliased && !device.enabled_features().sparse_residency_aliased {
                return Err(BufferError::RequirementNotMet {
                    required_for: "`create_info.sparse` is `Some(sparse_level)`, where \
                        `sparse_level` contains `BufferCreateFlags::SPARSE_ALIASED`",
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature("sparse_residency_aliased")])]),
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
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_external_memory")]),
                    ]),
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
    ) -> Result<Self, RuntimeError> {
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
            .map_err(RuntimeError::from)?;
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
                .align_to(properties.min_texel_buffer_offset_alignment)
                .unwrap();
        }

        if usage.intersects(BufferUsage::STORAGE_BUFFER) {
            memory_requirements.layout = memory_requirements
                .layout
                .align_to(properties.min_storage_buffer_offset_alignment)
                .unwrap();
        }

        if usage.intersects(BufferUsage::UNIFORM_BUFFER) {
            memory_requirements.layout = memory_requirements
                .layout
                .align_to(properties.min_uniform_buffer_offset_alignment)
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
        if !is_aligned(memory_offset, memory_requirements.layout.alignment()) {
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
    ) -> Result<Buffer, (RuntimeError, RawBuffer, MemoryAlloc)> {
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
            return Err((RuntimeError::from(err), self, allocation));
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
    pub fn usage(&self) -> BufferUsage {
        self.usage
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

impl_id_counter!(RawBuffer);

/// Parameters to create a new [`Buffer`].
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
    /// When using the [`Buffer`] constructors, you must leave this at `0`. They fill this field
    /// based on the data type of the contents and the other parameters you provide, and then pass
    /// this create-info to [`RawBuffer::new`]. You must override the default when constructing
    /// [`RawBuffer`] directly.
    ///
    /// The default value is `0`.
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
