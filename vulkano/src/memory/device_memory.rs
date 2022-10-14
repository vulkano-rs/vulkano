// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::DedicatedAllocation;
use crate::{
    device::{Device, DeviceOwned},
    macros::{vulkan_bitflags, vulkan_enum},
    DeviceSize, OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use std::{
    error::Error,
    ffi::c_void,
    fmt::{Display, Error as FmtError, Formatter},
    fs::File,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops::Range,
    ptr, slice,
    sync::{atomic::Ordering, Arc},
};

/// Represents memory that has been allocated from the device.
///
/// The destructor of `DeviceMemory` automatically frees the memory.
///
/// # Examples
///
/// ```
/// use vulkano::memory::{DeviceMemory, MemoryAllocateInfo};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let memory_type_index = 0;
///
/// // Allocates 1KB of memory.
/// let memory = DeviceMemory::allocate(
///     device.clone(),
///     MemoryAllocateInfo {
///         allocation_size: 1024,
///         memory_type_index,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// ```
#[derive(Debug)]
pub struct DeviceMemory {
    handle: ash::vk::DeviceMemory,
    device: Arc<Device>,

    allocation_size: DeviceSize,
    memory_type_index: u32,
    export_handle_types: ExternalMemoryHandleTypes,
    flags: MemoryAllocateFlags,
}

impl DeviceMemory {
    /// Allocates a block of memory from the device.
    ///
    /// Some platforms may have a limit on the maximum size of a single allocation. For example,
    /// certain systems may fail to create allocations with a size greater than or equal to 4GB.
    ///
    /// # Panics
    ///
    /// - Panics if `allocate_info.allocation_size` is 0.
    /// - Panics if `allocate_info.dedicated_allocation` is `Some` and the contained buffer or
    ///   image does not belong to `device`.
    pub fn allocate(
        device: Arc<Device>,
        mut allocate_info: MemoryAllocateInfo<'_>,
    ) -> Result<Self, DeviceMemoryError> {
        Self::validate(&device, &mut allocate_info, None)?;

        unsafe { Self::allocate_unchecked(device, allocate_info, None) }.map_err(Into::into)
    }

    /// Creates a new `DeviceMemory` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `allocate_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::DeviceMemory,
        allocate_info: MemoryAllocateInfo<'_>,
    ) -> Self {
        let MemoryAllocateInfo {
            allocation_size,
            memory_type_index,
            dedicated_allocation: _,
            export_handle_types,
            flags,
            _ne: _,
        } = allocate_info;

        DeviceMemory {
            handle,
            device,
            allocation_size,
            memory_type_index,
            export_handle_types,
            flags,
        }
    }

    /// Imports a block of memory from an external source.
    ///
    /// # Safety
    ///
    /// - See the documentation of the variants of [`MemoryImportInfo`].
    ///
    /// # Panics
    ///
    /// - Panics if `allocate_info.allocation_size` is 0.
    /// - Panics if `allocate_info.dedicated_allocation` is `Some` and the contained buffer or
    ///   image does not belong to `device`.
    pub unsafe fn import(
        device: Arc<Device>,
        mut allocate_info: MemoryAllocateInfo<'_>,
        import_info: MemoryImportInfo,
    ) -> Result<Self, DeviceMemoryError> {
        Self::validate(&device, &mut allocate_info, Some(&import_info))?;

        Self::allocate_unchecked(device, allocate_info, Some(import_info)).map_err(Into::into)
    }

    #[inline(never)]
    fn validate(
        device: &Device,
        allocate_info: &mut MemoryAllocateInfo<'_>,
        import_info: Option<&MemoryImportInfo>,
    ) -> Result<(), DeviceMemoryError> {
        let &mut MemoryAllocateInfo {
            allocation_size,
            memory_type_index,
            ref mut dedicated_allocation,
            export_handle_types,
            flags,
            _ne: _,
        } = allocate_info;

        if !(device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation)
        {
            // Fall back instead of erroring out
            *dedicated_allocation = None;
        }

        let memory_properties = device.physical_device().memory_properties();

        // VUID-vkAllocateMemory-pAllocateInfo-01714
        let memory_type = memory_properties
            .memory_types
            .get(memory_type_index as usize)
            .ok_or(DeviceMemoryError::MemoryTypeIndexOutOfRange {
                memory_type_index,
                memory_type_count: memory_properties.memory_types.len() as u32,
            })?;

        // VUID-VkMemoryAllocateInfo-memoryTypeIndex-01872
        if memory_type.property_flags.protected && !device.enabled_features().protected_memory {
            return Err(DeviceMemoryError::RequirementNotMet {
                required_for: "`allocate_info.memory_type_index` refers to a memory type where `property_flags.protected` is set",
                requires_one_of: RequiresOneOf {
                    features: &["protected_memory"],
                    ..Default::default()
                },
            });
        }

        // VUID-VkMemoryAllocateInfo-pNext-01874
        assert!(allocation_size != 0);

        // VUID-vkAllocateMemory-pAllocateInfo-01713
        let heap_size = memory_properties.memory_heaps[memory_type.heap_index as usize].size;
        if heap_size != 0 && allocation_size > heap_size {
            return Err(DeviceMemoryError::MemoryTypeHeapSizeExceeded {
                allocation_size,
                heap_size,
            });
        }

        if let Some(dedicated_allocation) = dedicated_allocation {
            match dedicated_allocation {
                DedicatedAllocation::Buffer(buffer) => {
                    // VUID-VkMemoryDedicatedAllocateInfo-commonparent
                    assert_eq!(device, buffer.device().as_ref());

                    let required_size = buffer.memory_requirements().size;

                    // VUID-VkMemoryDedicatedAllocateInfo-buffer-02965
                    if allocation_size != required_size {
                        return Err(DeviceMemoryError::DedicatedAllocationSizeMismatch {
                            allocation_size,
                            required_size,
                        });
                    }
                }
                DedicatedAllocation::Image(image) => {
                    // VUID-VkMemoryDedicatedAllocateInfo-commonparent
                    assert_eq!(device, image.device().as_ref());

                    let required_size = image.memory_requirements().size;

                    // VUID-VkMemoryDedicatedAllocateInfo-image-02964
                    if allocation_size != required_size {
                        return Err(DeviceMemoryError::DedicatedAllocationSizeMismatch {
                            allocation_size,
                            required_size,
                        });
                    }
                }
            }
        }

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_memory)
            {
                return Err(DeviceMemoryError::RequirementNotMet {
                    required_for: "`allocate_info.export_handle_types` is not empty",
                    requires_one_of: RequiresOneOf {
                        api_version: Some(Version::V1_1),
                        device_extensions: &["khr_external_memory"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkExportMemoryAllocateInfo-handleTypes-parameter
            export_handle_types.validate_device(device)?;

            // VUID-VkMemoryAllocateInfo-pNext-00639
            // VUID-VkExportMemoryAllocateInfo-handleTypes-00656
            // TODO: how do you fullfill this when you don't know the image or buffer parameters?
            // Does exporting memory require specifying these parameters up front, and does it tie
            // the allocation to only images or buffers of that type?
        }

        if let Some(import_info) = import_info {
            match *import_info {
                MemoryImportInfo::Fd {
                    #[cfg(unix)]
                    handle_type,
                    #[cfg(not(unix))]
                        handle_type: _,
                    file: _,
                } => {
                    if !device.enabled_extensions().khr_external_memory_fd {
                        return Err(DeviceMemoryError::RequirementNotMet {
                            required_for:
                                "`allocate_info.import_info` is `Some(MemoryImportInfo::Fd)`",
                            requires_one_of: RequiresOneOf {
                                device_extensions: &["khr_external_memory_fd"],
                                ..Default::default()
                            },
                        });
                    }

                    #[cfg(not(unix))]
                    unreachable!(
                        "`khr_external_memory_fd` was somehow enabled on a non-Unix system"
                    );

                    #[cfg(unix)]
                    {
                        // VUID-VkImportMemoryFdInfoKHR-handleType-parameter
                        handle_type.validate_device(device)?;

                        // VUID-VkImportMemoryFdInfoKHR-handleType-00669
                        match handle_type {
                            ExternalMemoryHandleType::OpaqueFd => {
                                // VUID-VkMemoryAllocateInfo-allocationSize-01742
                                // Can't validate, must be ensured by user

                                // VUID-VkMemoryDedicatedAllocateInfo-buffer-01879
                                // Can't validate, must be ensured by user

                                // VUID-VkMemoryDedicatedAllocateInfo-image-01878
                                // Can't validate, must be ensured by user
                            }
                            ExternalMemoryHandleType::DmaBuf => {}
                            _ => {
                                return Err(DeviceMemoryError::ImportFdHandleTypeNotSupported {
                                    handle_type,
                                })
                            }
                        }

                        // VUID-VkMemoryAllocateInfo-memoryTypeIndex-00648
                        // Can't validate, must be ensured by user
                    }
                }
                MemoryImportInfo::Win32 {
                    #[cfg(windows)]
                    handle_type,
                    #[cfg(not(windows))]
                        handle_type: _,
                    handle: _,
                } => {
                    if !device.enabled_extensions().khr_external_memory_win32 {
                        return Err(DeviceMemoryError::RequirementNotMet {
                            required_for:
                                "`allocate_info.import_info` is `Some(MemoryImportInfo::Win32)`",
                            requires_one_of: RequiresOneOf {
                                device_extensions: &["khr_external_memory_win32"],
                                ..Default::default()
                            },
                        });
                    }

                    #[cfg(not(windows))]
                    unreachable!(
                        "`khr_external_memory_win32` was somehow enabled on a non-Windows system"
                    );

                    #[cfg(windows)]
                    {
                        // VUID-VkImportMemoryWin32HandleInfoKHR-handleType-parameter
                        handle_type.validate_device(device)?;

                        // VUID-VkImportMemoryWin32HandleInfoKHR-handleType-00660
                        match handle_type {
                            ExternalMemoryHandleType::OpaqueWin32
                            | ExternalMemoryHandleType::OpaqueWin32Kmt => {
                                // VUID-VkMemoryAllocateInfo-allocationSize-01742
                                // Can't validate, must be ensured by user

                                // VUID-VkMemoryDedicatedAllocateInfo-buffer-01879
                                // Can't validate, must be ensured by user

                                // VUID-VkMemoryDedicatedAllocateInfo-image-01878
                                // Can't validate, must be ensured by user
                            }
                            _ => {
                                return Err(DeviceMemoryError::ImportWin32HandleTypeNotSupported {
                                    handle_type,
                                })
                            }
                        }

                        // VUID-VkMemoryAllocateInfo-memoryTypeIndex-00645
                        // Can't validate, must be ensured by user
                    }
                }
            }
        }

        if !flags.is_empty()
            && device.physical_device().api_version() < Version::V1_1
            && !device.enabled_extensions().khr_device_group
        {
            return Err(DeviceMemoryError::RequirementNotMet {
                required_for: "`allocate_info.flags` is not empty",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_1),
                    device_extensions: &["khr_device_group"],
                    ..Default::default()
                },
            });
        }

        if flags.device_address {
            // VUID-VkMemoryAllocateInfo-flags-03331
            if !device.enabled_features().buffer_device_address {
                return Err(DeviceMemoryError::RequirementNotMet {
                    required_for: "`allocate_info.flags.device_address` is `true`",
                    requires_one_of: RequiresOneOf {
                        features: &["buffer_device_address"],
                        ..Default::default()
                    },
                });
            }

            if device.enabled_extensions().ext_buffer_device_address {
                return Err(DeviceMemoryError::RequirementNotMet {
                    required_for: "`allocate_info.flags.device_address` is `true`",
                    requires_one_of: RequiresOneOf {
                        api_version: Some(Version::V1_2),
                        device_extensions: &["khr_buffer_device_address"],
                        ..Default::default()
                    },
                });
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline(never)]
    pub unsafe fn allocate_unchecked(
        device: Arc<Device>,
        allocate_info: MemoryAllocateInfo<'_>,
        import_info: Option<MemoryImportInfo>,
    ) -> Result<Self, VulkanError> {
        let MemoryAllocateInfo {
            allocation_size,
            memory_type_index,
            dedicated_allocation,
            export_handle_types,
            flags,
            _ne: _,
        } = allocate_info;

        let mut allocate_info = ash::vk::MemoryAllocateInfo::builder()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type_index);

        // VUID-VkMemoryDedicatedAllocateInfo-image-01432
        let mut dedicated_allocate_info =
            dedicated_allocation.map(|dedicated_allocation| match dedicated_allocation {
                DedicatedAllocation::Buffer(buffer) => ash::vk::MemoryDedicatedAllocateInfo {
                    buffer: buffer.internal_object(),
                    ..Default::default()
                },
                DedicatedAllocation::Image(image) => ash::vk::MemoryDedicatedAllocateInfo {
                    image: image.internal_object(),
                    ..Default::default()
                },
            });

        if let Some(info) = dedicated_allocate_info.as_mut() {
            allocate_info = allocate_info.push_next(info);
        }

        let mut export_allocate_info = if !export_handle_types.is_empty() {
            Some(ash::vk::ExportMemoryAllocateInfo {
                handle_types: export_handle_types.into(),
                ..Default::default()
            })
        } else {
            None
        };

        if let Some(info) = export_allocate_info.as_mut() {
            allocate_info = allocate_info.push_next(info);
        }

        #[cfg(unix)]
        let mut import_fd_info = match import_info {
            Some(MemoryImportInfo::Fd { handle_type, file }) => {
                use std::os::unix::io::IntoRawFd;

                Some(ash::vk::ImportMemoryFdInfoKHR {
                    handle_type: handle_type.into(),
                    fd: file.into_raw_fd(),
                    ..Default::default()
                })
            }
            _ => None,
        };

        #[cfg(unix)]
        if let Some(info) = import_fd_info.as_mut() {
            allocate_info = allocate_info.push_next(info);
        }

        #[cfg(windows)]
        let mut import_win32_handle_info = match import_info {
            Some(MemoryImportInfo::Win32 {
                handle_type,
                handle,
            }) => Some(ash::vk::ImportMemoryWin32HandleInfoKHR {
                handle_type: handle_type.into(),
                handle,
                ..Default::default()
            }),
            _ => None,
        };

        #[cfg(windows)]
        if let Some(info) = import_win32_handle_info.as_mut() {
            allocate_info = allocate_info.push_next(info);
        }

        let mut flags_info = ash::vk::MemoryAllocateFlagsInfo {
            flags: flags.into(),
            ..Default::default()
        };

        if !flags.is_empty() {
            allocate_info = allocate_info.push_next(&mut flags_info);
        }

        // VUID-vkAllocateMemory-maxMemoryAllocationCount-04101
        let max_allocations = device
            .physical_device()
            .properties()
            .max_memory_allocation_count;
        device
            .allocation_count
            .fetch_update(Ordering::Acquire, Ordering::Relaxed, move |count| {
                (count < max_allocations).then_some(count + 1)
            })
            .map_err(|_| VulkanError::TooManyObjects)?;

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.allocate_memory)(
                device.internal_object(),
                &allocate_info.build(),
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(|e| {
                device.allocation_count.fetch_sub(1, Ordering::Release);
                VulkanError::from(e)
            })?;

            output.assume_init()
        };

        Ok(DeviceMemory {
            handle,
            device,
            allocation_size,
            memory_type_index,
            export_handle_types,
            flags,
        })
    }

    /// Returns the index of the memory type that this memory was allocated from.
    #[inline]
    pub fn memory_type_index(&self) -> u32 {
        self.memory_type_index
    }

    /// Returns the size in bytes of the memory allocation.
    #[inline]
    pub fn allocation_size(&self) -> DeviceSize {
        self.allocation_size
    }

    /// Returns the handle types that can be exported from the memory allocation.
    #[inline]
    pub fn export_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.export_handle_types
    }

    /// Returns the flags the memory was allocated with.
    #[inline]
    pub fn flags(&self) -> MemoryAllocateFlags {
        self.flags
    }

    /// Retrieves the amount of lazily-allocated memory that is currently commited to this
    /// memory object.
    ///
    /// The device may change this value at any time, and the returned value may be
    /// already out-of-date.
    ///
    /// `self` must have been allocated from a memory type that has the
    /// [`lazily_allocated`](crate::memory::MemoryPropertyFlags::lazily_allocated) flag set.
    #[inline]
    pub fn commitment(&self) -> Result<DeviceSize, DeviceMemoryError> {
        self.validate_commitment()?;

        unsafe { Ok(self.commitment_unchecked()) }
    }

    fn validate_commitment(&self) -> Result<(), DeviceMemoryError> {
        let memory_type = &self
            .device
            .physical_device()
            .memory_properties()
            .memory_types[self.memory_type_index as usize];

        // VUID-vkGetDeviceMemoryCommitment-memory-00690
        if !memory_type.property_flags.lazily_allocated {
            return Err(DeviceMemoryError::NotLazilyAllocated);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn commitment_unchecked(&self) -> DeviceSize {
        let mut output: DeviceSize = 0;

        let fns = self.device.fns();
        (fns.v1_0.get_device_memory_commitment)(
            self.device.internal_object(),
            self.handle,
            &mut output,
        );

        output
    }

    /// Exports the device memory into a Unix file descriptor. The caller owns the returned `File`.
    ///
    /// # Panics
    ///
    /// - Panics if the user requests an invalid handle type for this device memory object.
    #[inline]
    pub fn export_fd(
        &self,
        handle_type: ExternalMemoryHandleType,
    ) -> Result<std::fs::File, DeviceMemoryError> {
        // VUID-VkMemoryGetFdInfoKHR-handleType-parameter
        handle_type.validate_device(&self.device)?;

        // VUID-VkMemoryGetFdInfoKHR-handleType-00672
        if !matches!(
            handle_type,
            ExternalMemoryHandleType::OpaqueFd | ExternalMemoryHandleType::DmaBuf
        ) {
            return Err(DeviceMemoryError::HandleTypeNotSupported { handle_type });
        }

        // VUID-VkMemoryGetFdInfoKHR-handleType-00671
        if !ash::vk::ExternalMemoryHandleTypeFlags::from(self.export_handle_types)
            .intersects(ash::vk::ExternalMemoryHandleTypeFlags::from(handle_type))
        {
            return Err(DeviceMemoryError::HandleTypeNotSupported { handle_type });
        }

        debug_assert!(self.device().enabled_extensions().khr_external_memory_fd);

        #[cfg(not(unix))]
        unreachable!("`khr_external_memory_fd` was somehow enabled on a non-Unix system");

        #[cfg(unix)]
        {
            use std::os::unix::io::FromRawFd;

            let fd = unsafe {
                let fns = self.device.fns();
                let info = ash::vk::MemoryGetFdInfoKHR {
                    memory: self.handle,
                    handle_type: handle_type.into(),
                    ..Default::default()
                };

                let mut output = MaybeUninit::uninit();
                (fns.khr_external_memory_fd.get_memory_fd_khr)(
                    self.device.internal_object(),
                    &info,
                    output.as_mut_ptr(),
                )
                .result()
                .map_err(VulkanError::from)?;
                output.assume_init()
            };

            let file = unsafe { std::fs::File::from_raw_fd(fd) };

            Ok(file)
        }
    }
}

impl Drop for DeviceMemory {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.free_memory)(self.device.internal_object(), self.handle, ptr::null());
            self.device.allocation_count.fetch_sub(1, Ordering::Release);
        }
    }
}

unsafe impl VulkanObject for DeviceMemory {
    type Object = ash::vk::DeviceMemory;

    #[inline]
    fn internal_object(&self) -> ash::vk::DeviceMemory {
        self.handle
    }
}

unsafe impl DeviceOwned for DeviceMemory {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for DeviceMemory {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for DeviceMemory {}

impl Hash for DeviceMemory {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device.hash(state);
    }
}

/// Parameters to allocate a new `DeviceMemory`.
#[derive(Clone, Debug)]
pub struct MemoryAllocateInfo<'d> {
    /// The number of bytes to allocate.
    ///
    /// The default value is `0`, which must be overridden.
    pub allocation_size: DeviceSize,

    /// The index of the memory type that should be allocated.
    ///
    /// The default value is [`u32::MAX`], which must be overridden.
    pub memory_type_index: u32,

    /// Allocates memory for a specific buffer or image.
    ///
    /// This value is silently ignored (treated as `None`) if the device API version is less than
    /// 1.1 and the
    /// [`khr_dedicated_allocation`](crate::device::DeviceExtensions::khr_dedicated_allocation)
    /// extension is not enabled on the device.
    pub dedicated_allocation: Option<DedicatedAllocation<'d>>,

    /// The handle types that can be exported from the allocated memory.
    pub export_handle_types: ExternalMemoryHandleTypes,

    /// Additional flags for the memory allocation.
    ///
    /// If not empty, the device API version must be at least 1.1, or the
    /// [`khr_device_group`](crate::device::DeviceExtensions::khr_device_group) extension must be
    /// enabled on the device.
    ///
    /// The default value is [`MemoryAllocateFlags::empty()`].
    pub flags: MemoryAllocateFlags,

    pub _ne: crate::NonExhaustive,
}

impl Default for MemoryAllocateInfo<'static> {
    #[inline]
    fn default() -> Self {
        Self {
            allocation_size: 0,
            memory_type_index: u32::MAX,
            dedicated_allocation: None,
            export_handle_types: ExternalMemoryHandleTypes::empty(),
            flags: MemoryAllocateFlags::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl<'d> MemoryAllocateInfo<'d> {
    /// Returns a `MemoryAllocateInfo` with the specified `dedicated_allocation`.
    #[inline]
    pub fn dedicated_allocation(dedicated_allocation: DedicatedAllocation<'d>) -> Self {
        Self {
            allocation_size: 0,
            memory_type_index: u32::MAX,
            dedicated_allocation: Some(dedicated_allocation),
            export_handle_types: ExternalMemoryHandleTypes::empty(),
            flags: MemoryAllocateFlags::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to import memory from an external source.
#[derive(Debug)]
#[non_exhaustive]
pub enum MemoryImportInfo {
    /// Import memory from a Unix file descriptor.
    ///
    /// `handle_type` must be either [`ExternalMemoryHandleType::OpaqueFd`] or
    /// [`ExternalMemoryHandleType::DmaBuf`].
    ///
    /// # Safety
    ///
    /// - `file` must be a valid Unix file descriptor.
    /// - Vulkan will take ownership of `file`, and once the memory is imported, you must not
    ///   perform any operations on `file` nor on any of its clones/duplicates.
    /// - If `file` was created by the Vulkan API, and `handle_type` is
    ///   [`ExternalMemoryHandleType::OpaqueFd`]:
    ///   - [`MemoryAllocateInfo::allocation_size`] and [`MemoryAllocateInfo::memory_type_index`]
    ///     must match those of the original memory allocation.
    ///   - If the original memory allocation used [`MemoryAllocateInfo::dedicated_allocation`],
    ///     the imported one must also use it, and the associated buffer or image must be defined
    ///     identically to the original.
    /// - If `file` was not created by the Vulkan API, then
    ///   [`MemoryAllocateInfo::memory_type_index`] must be one of the memory types returned by
    ///   [`Device::memory_fd_properties`].
    Fd {
        handle_type: ExternalMemoryHandleType,
        file: File,
    },

    /// Import memory from a Windows handle.
    ///
    /// `handle_type` must be either [`ExternalMemoryHandleType::OpaqueWin32`] or
    /// [`ExternalMemoryHandleType::OpaqueWin32Kmt`].
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Windows handle.
    /// - Vulkan will not take ownership of `handle`.
    /// - If `handle_type` is [`ExternalMemoryHandleType::OpaqueWin32`], it owns a reference
    ///   to the underlying resource and must eventually be closed by the caller.
    /// - If `handle_type` is [`ExternalMemoryHandleType::OpaqueWin32Kmt`], it does not own a
    ///   reference to the underlying resource.
    /// - `handle` must be created by the Vulkan API.
    /// - [`MemoryAllocateInfo::allocation_size`] and [`MemoryAllocateInfo::memory_type_index`]
    ///   must match those of the original memory allocation.
    /// - If the original memory allocation used [`MemoryAllocateInfo::dedicated_allocation`],
    ///   the imported one must also use it, and the associated buffer or image must be defined
    ///   identically to the original.
    Win32 {
        handle_type: ExternalMemoryHandleType,
        handle: ash::vk::HANDLE,
    },
}

vulkan_enum! {
    /// Describes a handle type used for Vulkan external memory apis.  This is **not** just a
    /// suggestion.  Check out vkExternalMemoryHandleTypeFlagBits in the Vulkan spec.
    ///
    /// If you specify an handle type that doesnt make sense (for example, using a dma-buf handle
    /// type on Windows) when using this handle, a panic will happen.
    #[non_exhaustive]
    ExternalMemoryHandleType = ExternalMemoryHandleTypeFlags(u32);

    // TODO: document
    OpaqueFd = OPAQUE_FD,

    // TODO: document
    OpaqueWin32 = OPAQUE_WIN32,

    // TODO: document
    OpaqueWin32Kmt = OPAQUE_WIN32_KMT,

    // TODO: document
    D3D11Texture = D3D11_TEXTURE,

    // TODO: document
    D3D11TextureKmt = D3D11_TEXTURE_KMT,

    // TODO: document
    D3D12Heap = D3D12_HEAP,

    // TODO: document
    D3D12Resource = D3D12_RESOURCE,

    // TODO: document
    DmaBuf = DMA_BUF_EXT {
        device_extensions: [ext_external_memory_dma_buf],
    },

    // TODO: document
    AndroidHardwareBuffer = ANDROID_HARDWARE_BUFFER_ANDROID {
        device_extensions: [android_external_memory_android_hardware_buffer],
    },

    // TODO: document
    HostAllocation = HOST_ALLOCATION_EXT {
        device_extensions: [ext_external_memory_host],
    },

    // TODO: document
    HostMappedForeignMemory = HOST_MAPPED_FOREIGN_MEMORY_EXT {
        device_extensions: [ext_external_memory_host],
    },

    // TODO: document
    ZirconVmo = ZIRCON_VMO_FUCHSIA {
        device_extensions: [fuchsia_external_memory],
    },

    // TODO: document
    RdmaAddress = RDMA_ADDRESS_NV {
        device_extensions: [nv_external_memory_rdma],
    },
}

vulkan_bitflags! {
    /// A mask of multiple handle types.
    #[non_exhaustive]
    ExternalMemoryHandleTypes = ExternalMemoryHandleTypeFlags(u32);

    // TODO: document
    opaque_fd = OPAQUE_FD,

    // TODO: document
    opaque_win32 = OPAQUE_WIN32,

    // TODO: document
    opaque_win32_kmt = OPAQUE_WIN32_KMT,

    // TODO: document
    d3d11_texture = D3D11_TEXTURE,

    // TODO: document
    d3d11_texture_kmt = D3D11_TEXTURE_KMT,

    // TODO: document
    d3d12_heap = D3D12_HEAP,

    // TODO: document
    d3d12_resource = D3D12_RESOURCE,

    // TODO: document
    dma_buf = DMA_BUF_EXT {
        device_extensions: [ext_external_memory_dma_buf],
    },

    // TODO: document
    android_hardware_buffer = ANDROID_HARDWARE_BUFFER_ANDROID {
        device_extensions: [android_external_memory_android_hardware_buffer],
    },

    // TODO: document
    host_allocation = HOST_ALLOCATION_EXT {
        device_extensions: [ext_external_memory_host],
    },

    // TODO: document
    host_mapped_foreign_memory = HOST_MAPPED_FOREIGN_MEMORY_EXT {
        device_extensions: [ext_external_memory_host],
    },

    // TODO: document
    zircon_vmo = ZIRCON_VMO_FUCHSIA {
        device_extensions: [fuchsia_external_memory],
    },

    // TODO: document
    rdma_address = RDMA_ADDRESS_NV {
        device_extensions: [nv_external_memory_rdma],
    },
}

impl ExternalMemoryHandleTypes {
    /// Returns an iterator of `ExternalMemoryHandleType` enum values, representing the fields that
    /// are set in `self`.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ExternalMemoryHandleType> {
        let ExternalMemoryHandleTypes {
            opaque_fd,
            opaque_win32,
            opaque_win32_kmt,
            d3d11_texture,
            d3d11_texture_kmt,
            d3d12_heap,
            d3d12_resource,
            dma_buf,
            android_hardware_buffer,
            host_allocation,
            host_mapped_foreign_memory,
            zircon_vmo,
            rdma_address,
            _ne: _,
        } = *self;

        [
            opaque_fd.then_some(ExternalMemoryHandleType::OpaqueFd),
            opaque_win32.then_some(ExternalMemoryHandleType::OpaqueWin32),
            opaque_win32_kmt.then_some(ExternalMemoryHandleType::OpaqueWin32Kmt),
            d3d11_texture.then_some(ExternalMemoryHandleType::D3D11Texture),
            d3d11_texture_kmt.then_some(ExternalMemoryHandleType::D3D11TextureKmt),
            d3d12_heap.then_some(ExternalMemoryHandleType::D3D12Heap),
            d3d12_resource.then_some(ExternalMemoryHandleType::D3D12Resource),
            dma_buf.then_some(ExternalMemoryHandleType::DmaBuf),
            android_hardware_buffer.then_some(ExternalMemoryHandleType::AndroidHardwareBuffer),
            host_allocation.then_some(ExternalMemoryHandleType::HostAllocation),
            host_mapped_foreign_memory.then_some(ExternalMemoryHandleType::HostMappedForeignMemory),
            zircon_vmo.then_some(ExternalMemoryHandleType::HostMappedForeignMemory),
            rdma_address.then_some(ExternalMemoryHandleType::HostMappedForeignMemory),
        ]
        .into_iter()
        .flatten()
    }
}

vulkan_bitflags! {
    /// A mask specifying flags for device memory allocation.
    #[non_exhaustive]
    MemoryAllocateFlags = MemoryAllocateFlags(u32);

    // TODO: implement
    // device_mask = DEVICE_MASK,

    /// Specifies that the allocated device memory can be bound to a buffer created with the
    /// [`shader_device_address`] usage. This requires that the [`buffer_device_address`] feature
    /// is enabled on the device and the [`ext_buffer_device_address`] extension is not enabled on
    /// the device.
    ///
    /// [`shader_device_address`]: crate::buffer::BufferUsage::shader_device_address
    /// [`buffer_device_address`]: crate::device::Features::buffer_device_address
    /// [`ext_buffer_device_address`]: crate::device::DeviceExtensions::ext_buffer_device_address
    device_address = DEVICE_ADDRESS,

    // TODO: implement
    // device_address_capture_replay = DEVICE_ADDRESS_CAPTURE_REPLAY,
}

/// Error type returned by functions related to `DeviceMemory`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceMemoryError {
    /// Not enough memory available.
    OomError(OomError),

    /// The maximum number of allocations has been exceeded.
    TooManyObjects,

    /// An error occurred when mapping the memory.
    MemoryMapError(MemoryMapError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// `dedicated_allocation` was `Some`, but the provided `allocation_size`  was different from
    /// the required size of the buffer or image.
    DedicatedAllocationSizeMismatch {
        allocation_size: DeviceSize,
        required_size: DeviceSize,
    },

    /// The requested export handle type is not supported for this operation, or was not provided in
    /// `export_handle_types` when allocating the memory.
    HandleTypeNotSupported {
        handle_type: ExternalMemoryHandleType,
    },

    /// The provided `MemoryImportInfo::Fd::handle_type` is not supported for file descriptors.
    ImportFdHandleTypeNotSupported {
        handle_type: ExternalMemoryHandleType,
    },

    /// The provided `MemoryImportInfo::Win32::handle_type` is not supported.
    ImportWin32HandleTypeNotSupported {
        handle_type: ExternalMemoryHandleType,
    },

    /// The provided `allocation_size` was greater than the memory type's heap size.
    MemoryTypeHeapSizeExceeded {
        allocation_size: DeviceSize,
        heap_size: DeviceSize,
    },

    /// The provided `memory_type_index` was not less than the number of memory types in the
    /// physical device.
    MemoryTypeIndexOutOfRange {
        memory_type_index: u32,
        memory_type_count: u32,
    },

    /// The memory type from which this memory was allocated does not have the
    /// [`lazily_allocated`](crate::memory::MemoryPropertyFlags::lazily_allocated) flag set.
    NotLazilyAllocated,

    /// Spec violation, containing the Valid Usage ID (VUID) from the Vulkan spec.
    // TODO: Remove
    SpecViolation(u32),

    /// An implicit violation that's convered in the Vulkan spec.
    // TODO: Remove
    ImplicitSpecViolation(&'static str),
}

impl Error for DeviceMemoryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            Self::MemoryMapError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for DeviceMemoryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::TooManyObjects => {
                write!(f, "the maximum number of allocations has been exceeded")
            }
            Self::MemoryMapError(_) => write!(f, "error occurred when mapping the memory"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::DedicatedAllocationSizeMismatch {
                allocation_size,
                required_size,
            } => write!(
                f,
                "`dedicated_allocation` was `Some`, but the provided `allocation_size` ({}) was \
                different from the required size of the buffer or image ({})",
                allocation_size, required_size,
            ),
            Self::HandleTypeNotSupported { handle_type } => write!(
                f,
                "the requested export handle type ({:?}) is not supported for this operation, or \
                was not provided in `export_handle_types` when allocating the memory",
                handle_type,
            ),
            Self::ImportFdHandleTypeNotSupported { handle_type } => write!(
                f,
                "the provided `MemoryImportInfo::Fd::handle_type` ({:?}) is not supported for file \
                descriptors",
                handle_type,
            ),
            Self::ImportWin32HandleTypeNotSupported { handle_type } => write!(
                f,
                "the provided `MemoryImportInfo::Win32::handle_type` ({:?}) is not supported",
                handle_type,
            ),
            Self::MemoryTypeHeapSizeExceeded {
                allocation_size,
                heap_size,
            } => write!(
                f,
                "the provided `allocation_size` ({}) was greater than the memory type's heap size \
                ({})",
                allocation_size, heap_size,
            ),
            Self::MemoryTypeIndexOutOfRange {
                memory_type_index,
                memory_type_count,
            } => write!(
                f,
                "the provided `memory_type_index` ({}) was not less than the number of memory \
                types in the physical device ({})",
                memory_type_index, memory_type_count,
            ),
            Self::NotLazilyAllocated => write!(
                f,
                "the memory type from which this memory was allocated does not have the \
                `lazily_allocated` flag set",
            ),

            Self::SpecViolation(u) => {
                write!(f, "valid usage ID check {} failed", u)
            }
            Self::ImplicitSpecViolation(e) => {
                write!(f, "Implicit spec violation failed {}", e)
            }
        }
    }
}

impl From<VulkanError> for DeviceMemoryError {
    fn from(err: VulkanError) -> Self {
        match err {
            e @ VulkanError::OutOfHostMemory | e @ VulkanError::OutOfDeviceMemory => {
                Self::OomError(e.into())
            }
            VulkanError::TooManyObjects => Self::TooManyObjects,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for DeviceMemoryError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<MemoryMapError> for DeviceMemoryError {
    fn from(err: MemoryMapError) -> Self {
        Self::MemoryMapError(err)
    }
}

impl From<RequirementNotMet> for DeviceMemoryError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

/// Represents device memory that has been mapped in a CPU-accessible space.
///
/// In order to access the contents of the allocated memory, you can use the `read` and `write`
/// methods.
///
/// # Examples
///
/// ```
/// use vulkano::memory::{DeviceMemory, MappedDeviceMemory, MemoryAllocateInfo};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// // The memory type must be mappable.
/// let memory_type_index = device
///     .physical_device()
///     .memory_properties()
///     .memory_types
///     .iter()
///     .position(|t| t.property_flags.host_visible)
///     .map(|i| i as u32)
///     .unwrap(); // Vk specs guarantee that this can't fail
///
/// // Allocates 1KB of memory.
/// let memory = DeviceMemory::allocate(
///     device.clone(),
///     MemoryAllocateInfo {
///         allocation_size: 1024,
///         memory_type_index,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// let mapped_memory = MappedDeviceMemory::new(memory, 0..1024).unwrap();
///
/// // Get access to the content.
/// // Note that this is very unsafe because the access is unsynchronized.
/// unsafe {
///     let content = mapped_memory.write(0..1024).unwrap();
///     content[12] = 54;
/// }
/// ```
#[derive(Debug)]
pub struct MappedDeviceMemory {
    memory: DeviceMemory,
    pointer: *mut c_void, // points to `range.start`
    range: Range<DeviceSize>,

    atom_size: DeviceSize,
    coherent: bool,
}

// Note that `MappedDeviceMemory` doesn't implement `Drop`, as we don't need to unmap memory before
// freeing it.
//
// Vulkan specs, documentation of `vkFreeMemory`:
// > If a memory object is mapped at the time it is freed, it is implicitly unmapped.

impl MappedDeviceMemory {
    /// Maps a range of memory to be accessed by the CPU.
    ///
    /// `memory` must be allocated from host-visible memory.
    ///
    /// `range` is specified in bytes relative to the start of the memory allocation, and must fall
    /// within the range of the allocation (`0..allocation_size`). If `memory` was not allocated
    /// from host-coherent memory, then the start and end of `range` must be a multiple of the
    /// [`non_coherent_atom_size`](crate::device::Properties::non_coherent_atom_size) device
    /// property, but `range.end` can also the memory's `allocation_size`.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    pub fn new(memory: DeviceMemory, range: Range<DeviceSize>) -> Result<Self, MemoryMapError> {
        // VUID-vkMapMemory-size-00680
        assert!(!range.is_empty());

        // VUID-vkMapMemory-memory-00678
        // Guaranteed because we take ownership of `memory`, no other mapping can exist.

        // VUID-vkMapMemory-offset-00679
        // VUID-vkMapMemory-size-00681
        if range.end > memory.allocation_size {
            return Err(MemoryMapError::OutOfRange {
                provided_range: range,
                allowed_range: 0..memory.allocation_size,
            });
        }

        let device = memory.device();
        let memory_type = &device.physical_device().memory_properties().memory_types
            [memory.memory_type_index() as usize];

        // VUID-vkMapMemory-memory-00682
        if !memory_type.property_flags.host_visible {
            return Err(MemoryMapError::NotHostVisible);
        }

        let coherent = memory_type.property_flags.host_coherent;
        let atom_size = device.physical_device().properties().non_coherent_atom_size;

        // Not required for merely mapping, but without this check the user can end up with
        // parts of the mapped memory at the start and end that they're not able to
        // invalidate/flush, which is probably unintended.
        if !coherent
            && (range.start % atom_size != 0
                || (range.end % atom_size != 0 && range.end != memory.allocation_size))
        {
            return Err(MemoryMapError::RangeNotAlignedToAtomSize { range, atom_size });
        }

        let pointer = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.map_memory)(
                device.internal_object(),
                memory.handle,
                range.start,
                range.end - range.start,
                ash::vk::MemoryMapFlags::empty(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(MappedDeviceMemory {
            memory,
            pointer,
            range,

            atom_size,
            coherent,
        })
    }

    /// Unmaps the memory. It will no longer be accessible from the CPU.
    #[inline]
    pub fn unmap(self) -> DeviceMemory {
        unsafe {
            let device = self.memory.device();
            let fns = device.fns();
            (fns.v1_0.unmap_memory)(device.internal_object(), self.memory.handle);
        }

        self.memory
    }

    /// Invalidates the host (CPU) cache for a range of mapped memory.
    ///
    /// If the mapped memory is not host-coherent, you must call this function before the memory is
    /// read by the host, if the device previously wrote to the memory. It has no effect if the
    /// mapped memory is host-coherent.
    ///
    /// `range` is specified in bytes relative to the start of the memory allocation, and must fall
    /// within the range of the memory mapping given to `new`. If the memory was not allocated
    /// from host-coherent memory, then the start and end of `range` must be a multiple of the
    /// [`non_coherent_atom_size`](crate::device::Properties::non_coherent_atom_size) device
    /// property, but `range.end` can also equal the memory's `allocation_size`.
    ///
    /// # Safety
    ///
    /// - If there are memory writes by the GPU that have not been propagated into the CPU cache,
    ///   then there must not be any references in Rust code to the specified `range` of the memory.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    #[inline]
    pub unsafe fn invalidate_range(&self, range: Range<DeviceSize>) -> Result<(), MemoryMapError> {
        if self.coherent {
            return Ok(());
        }

        self.check_range(range.clone())?;

        // VUID-VkMappedMemoryRange-memory-00684
        // Guaranteed because `self` owns the memory and it's mapped during our lifetime.

        let range = ash::vk::MappedMemoryRange {
            memory: self.memory.internal_object(),
            offset: range.start,
            size: range.end - range.start,
            ..Default::default()
        };

        let fns = self.memory.device().fns();
        (fns.v1_0.invalidate_mapped_memory_ranges)(
            self.memory.device().internal_object(),
            1,
            &range,
        )
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Flushes the host (CPU) cache for a range of mapped memory.
    ///
    /// If the mapped memory is not host-coherent, you must call this function after writing to the
    /// memory, if the device is going to read the memory. It has no effect if the
    /// mapped memory is host-coherent.
    ///
    /// `range` is specified in bytes relative to the start of the memory allocation, and must fall
    /// within the range of the memory mapping given to `map`. If the memory was not allocated
    /// from host-coherent memory, then the start and end of `range` must be a multiple of the
    /// [`non_coherent_atom_size`](crate::device::Properties::non_coherent_atom_size) device
    /// property, but `range.end` can also equal the memory's `allocation_size`.
    ///
    /// # Safety
    ///
    /// - There must be no operations pending or executing in a GPU queue, that access the specified
    ///   `range` of the memory.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    #[inline]
    pub unsafe fn flush_range(&self, range: Range<DeviceSize>) -> Result<(), MemoryMapError> {
        self.check_range(range.clone())?;

        if self.coherent {
            return Ok(());
        }

        // VUID-VkMappedMemoryRange-memory-00684
        // Guaranteed because `self` owns the memory and it's mapped during our lifetime.

        let range = ash::vk::MappedMemoryRange {
            memory: self.memory.internal_object(),
            offset: range.start,
            size: range.end - range.start,
            ..Default::default()
        };

        let fns = self.device().fns();
        (fns.v1_0.flush_mapped_memory_ranges)(self.memory.device().internal_object(), 1, &range)
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Returns a reference to bytes in the mapped memory.
    ///
    /// `range` is specified in bytes relative to the start of the memory allocation, and must fall
    /// within the range of the memory mapping given to `map`. If the memory was not allocated
    /// from host-coherent memory, then the start and end of `range` must be a multiple of the
    /// [`non_coherent_atom_size`](crate::device::Properties::non_coherent_atom_size) device
    /// property, but `range.end` can also equal the memory's `allocation_size`.
    ///
    /// # Safety
    ///
    /// - While the returned reference exists, there must not be any mutable references in Rust code
    ///   to the same memory.
    /// - While the returned reference exists, there must be no operations pending or executing in
    ///   a GPU queue, that write to the same memory.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    #[inline]
    pub unsafe fn read(&self, range: Range<DeviceSize>) -> Result<&[u8], MemoryMapError> {
        self.check_range(range.clone())?;

        let bytes = slice::from_raw_parts(
            self.pointer.add((range.start - self.range.start) as usize) as *const u8,
            (range.end - range.start) as usize,
        );

        Ok(bytes)
    }

    /// Returns a mutable reference to bytes in the mapped memory.
    ///
    /// `range` is specified in bytes relative to the start of the memory allocation, and must fall
    /// within the range of the memory mapping given to `map`. If the memory was not allocated
    /// from host-coherent memory, then the start and end of `range` must be a multiple of the
    /// [`non_coherent_atom_size`](crate::device::Properties::non_coherent_atom_size) device
    /// property, but `range.end` can also equal the memory's `allocation_size`.
    ///
    /// # Safety
    ///
    /// - While the returned reference exists, there must not be any other references in Rust code
    ///   to the same memory.
    /// - While the returned reference exists, there must be no operations pending or executing in
    ///   a GPU queue, that access the same memory.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    #[inline]
    pub unsafe fn write(&self, range: Range<DeviceSize>) -> Result<&mut [u8], MemoryMapError> {
        self.check_range(range.clone())?;

        let bytes = slice::from_raw_parts_mut(
            self.pointer.add((range.start - self.range.start) as usize) as *mut u8,
            (range.end - range.start) as usize,
        );

        Ok(bytes)
    }

    #[inline]
    fn check_range(&self, range: Range<DeviceSize>) -> Result<(), MemoryMapError> {
        assert!(!range.is_empty());

        // VUID-VkMappedMemoryRange-size-00685
        if range.start < self.range.start || range.end > self.range.end {
            return Err(MemoryMapError::OutOfRange {
                provided_range: range,
                allowed_range: self.range.clone(),
            });
        }

        if !self.coherent {
            // VUID-VkMappedMemoryRange-offset-00687
            // VUID-VkMappedMemoryRange-size-01390
            if range.start % self.atom_size != 0
                || (range.end % self.atom_size != 0 && range.end != self.memory.allocation_size)
            {
                return Err(MemoryMapError::RangeNotAlignedToAtomSize {
                    range,
                    atom_size: self.atom_size,
                });
            }
        }

        Ok(())
    }
}

impl AsRef<DeviceMemory> for MappedDeviceMemory {
    #[inline]
    fn as_ref(&self) -> &DeviceMemory {
        &self.memory
    }
}

impl AsMut<DeviceMemory> for MappedDeviceMemory {
    #[inline]
    fn as_mut(&mut self) -> &mut DeviceMemory {
        &mut self.memory
    }
}

unsafe impl DeviceOwned for MappedDeviceMemory {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.memory.device()
    }
}

unsafe impl Send for MappedDeviceMemory {}
unsafe impl Sync for MappedDeviceMemory {}

/// Error type returned by functions related to `DeviceMemory`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryMapError {
    /// Not enough memory available.
    OomError(OomError),

    /// Memory map failed.
    MemoryMapFailed,

    /// Tried to map memory whose type is not host-visible.
    NotHostVisible,

    /// The specified `range` is not contained within the allocated or mapped memory range.
    OutOfRange {
        provided_range: Range<DeviceSize>,
        allowed_range: Range<DeviceSize>,
    },

    /// The memory is not host-coherent, and the specified `range` bounds are not a multiple of the
    /// [`non_coherent_atom_size`](crate::device::Properties::non_coherent_atom_size) device
    /// property.
    RangeNotAlignedToAtomSize {
        range: Range<DeviceSize>,
        atom_size: DeviceSize,
    },
}

impl Error for MemoryMapError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for MemoryMapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::MemoryMapFailed => write!(f, "memory map failed"),
            Self::NotHostVisible => {
                write!(f, "tried to map memory whose type is not host-visible")
            }
            Self::OutOfRange {
                provided_range,
                allowed_range,
            } => write!(
                f,
                "the specified `range` ({:?}) was not contained within the allocated or mapped \
                memory range ({:?})",
                provided_range, allowed_range,
            ),
            Self::RangeNotAlignedToAtomSize { range, atom_size } => write!(
                f,
                "the memory is not host-coherent, and the specified `range` bounds ({:?}) are not \
                a multiple of the `non_coherent_atom_size` device property ({})",
                range, atom_size,
            ),
        }
    }
}

impl From<VulkanError> for MemoryMapError {
    fn from(err: VulkanError) -> Self {
        match err {
            e @ VulkanError::OutOfHostMemory | e @ VulkanError::OutOfDeviceMemory => {
                Self::OomError(e.into())
            }
            VulkanError::MemoryMapFailed => Self::MemoryMapFailed,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for MemoryMapError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::MemoryAllocateInfo;
    use crate::{
        memory::{DeviceMemory, DeviceMemoryError},
        OomError,
    };

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = DeviceMemory::allocate(
            device,
            MemoryAllocateInfo {
                allocation_size: 256,
                memory_type_index: 0,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn zero_size() {
        let (device, _) = gfx_dev_and_queue!();
        assert_should_panic!({
            let _ = DeviceMemory::allocate(
                device.clone(),
                MemoryAllocateInfo {
                    allocation_size: 0,
                    memory_type_index: 0,
                    ..Default::default()
                },
            )
            .unwrap();
        });
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn oom_single() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_type_index = device
            .physical_device()
            .memory_properties()
            .memory_types
            .iter()
            .enumerate()
            .find_map(|(i, m)| (!m.property_flags.lazily_allocated).then_some(i as u32))
            .unwrap();

        match DeviceMemory::allocate(
            device,
            MemoryAllocateInfo {
                allocation_size: 0xffffffffffffffff,
                memory_type_index,
                ..Default::default()
            },
        ) {
            Err(DeviceMemoryError::MemoryTypeHeapSizeExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    #[ignore] // TODO: test fails for now on Mesa+Intel
    fn oom_multi() {
        let (device, _) = gfx_dev_and_queue!();
        let (memory_type_index, memory_type) = device
            .physical_device()
            .memory_properties()
            .memory_types
            .iter()
            .enumerate()
            .find_map(|(i, m)| (!m.property_flags.lazily_allocated).then_some((i as u32, m)))
            .unwrap();
        let heap_size = device.physical_device().memory_properties().memory_heaps
            [memory_type.heap_index as usize]
            .size;

        let mut allocs = Vec::new();

        for _ in 0..4 {
            match DeviceMemory::allocate(
                device.clone(),
                MemoryAllocateInfo {
                    allocation_size: heap_size / 3,
                    memory_type_index,
                    ..Default::default()
                },
            ) {
                Err(DeviceMemoryError::OomError(OomError::OutOfDeviceMemory)) => return, // test succeeded
                Ok(a) => allocs.push(a),
                _ => (),
            }
        }

        panic!()
    }

    #[test]
    fn allocation_count() {
        let (device, _) = gfx_dev_and_queue!();
        assert_eq!(device.allocation_count(), 0);
        let _mem1 = DeviceMemory::allocate(
            device.clone(),
            MemoryAllocateInfo {
                allocation_size: 256,
                memory_type_index: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(device.allocation_count(), 1);
        {
            let _mem2 = DeviceMemory::allocate(
                device.clone(),
                MemoryAllocateInfo {
                    allocation_size: 256,
                    memory_type_index: 0,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(device.allocation_count(), 2);
        }
        assert_eq!(device.allocation_count(), 1);
    }
}
