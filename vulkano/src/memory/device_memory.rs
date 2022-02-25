// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{Content, DedicatedAllocation};
use crate::{
    check_errors,
    device::{physical::MemoryType, Device, DeviceOwned},
    DeviceSize, Error, OomError, Version, VulkanObject,
};
use std::{
    error,
    ffi::c_void,
    fmt,
    fs::File,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops::{BitOr, Deref, DerefMut, Range},
    ptr,
    sync::{Arc, Mutex},
};

/// Represents memory that has been allocated from the device.
///
/// The destructor of `DeviceMemory` automatically frees the memory.
///
/// # Example
///
/// ```
/// use vulkano::memory::{DeviceMemory, MemoryAllocateInfo};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let memory_type = device.physical_device().memory_types().next().unwrap();
///
/// // Allocates 1KB of memory.
/// let memory = DeviceMemory::allocate(
///     device.clone(),
///     MemoryAllocateInfo {
///         allocation_size: 1024,
///         memory_type_index: memory_type.id(),
///         ..Default::default()
///     },
/// ).unwrap();
/// ```
#[derive(Debug)]
pub struct DeviceMemory {
    handle: ash::vk::DeviceMemory,
    device: Arc<Device>,

    allocation_size: DeviceSize,
    memory_type_index: u32,
    export_handle_types: ExternalMemoryHandleTypes,

    mapped: Mutex<bool>,
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
        mut allocate_info: MemoryAllocateInfo,
    ) -> Result<Self, DeviceMemoryAllocationError> {
        Self::validate(&device, &mut allocate_info, None)?;
        let handle = unsafe { Self::create(&device, &allocate_info, None)? };

        let MemoryAllocateInfo {
            allocation_size,
            memory_type_index,
            dedicated_allocation,
            export_handle_types,
            _ne: _,
        } = allocate_info;

        Ok(DeviceMemory {
            handle,
            device,

            allocation_size,
            memory_type_index,
            export_handle_types,

            mapped: Mutex::new(false),
        })
    }

    /// Allocates a block of memory from the device, and maps it to host memory.
    ///
    /// # Panics
    ///
    /// - Panics if `allocate_info.allocation_size` is 0.
    /// - Panics if `allocate_info.dedicated_allocation` is `Some` and the contained buffer or
    ///   image does not belong to `device`.
    /// - Panics if the memory type is not host-visible.
    pub fn allocate_and_map(
        device: Arc<Device>,
        allocate_info: MemoryAllocateInfo,
    ) -> Result<MappedDeviceMemory, DeviceMemoryAllocationError> {
        let device_memory = Self::allocate(device, allocate_info)?;
        Self::map_allocation(device_memory.device().clone(), device_memory)
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
        mut allocate_info: MemoryAllocateInfo,
        mut import_info: MemoryImportInfo,
    ) -> Result<Self, DeviceMemoryAllocationError> {
        Self::validate(&device, &mut allocate_info, Some(&mut import_info))?;
        let handle = Self::create(&device, &allocate_info, Some(import_info))?;

        let MemoryAllocateInfo {
            allocation_size,
            memory_type_index,
            dedicated_allocation,
            export_handle_types,
            _ne: _,
        } = allocate_info;

        Ok(DeviceMemory {
            handle,
            device,

            allocation_size,
            memory_type_index,
            export_handle_types,

            mapped: Mutex::new(false),
        })
    }

    /// Imports a block of memory from an external source, and maps it to host memory.
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
    /// - Panics if the memory type is not host-visible.
    pub unsafe fn import_and_map(
        device: Arc<Device>,
        allocate_info: MemoryAllocateInfo,
        import_info: MemoryImportInfo,
    ) -> Result<MappedDeviceMemory, DeviceMemoryAllocationError> {
        let device_memory = Self::import(device, allocate_info, import_info)?;
        Self::map_allocation(device_memory.device().clone(), device_memory)
    }

    fn validate(
        device: &Device,
        allocate_info: &mut MemoryAllocateInfo,
        import_info: Option<&mut MemoryImportInfo>,
    ) -> Result<(), DeviceMemoryAllocationError> {
        let &mut MemoryAllocateInfo {
            allocation_size,
            memory_type_index,
            ref mut dedicated_allocation,
            export_handle_types,
            _ne: _,
        } = allocate_info;

        if !(device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation)
        {
            // Fall back instead of erroring out
            *dedicated_allocation = None;
        }

        // VUID-vkAllocateMemory-pAllocateInfo-01714
        let memory_type = device
            .physical_device()
            .memory_type_by_id(memory_type_index)
            .ok_or_else(|| DeviceMemoryAllocationError::MemoryTypeIndexOutOfRange {
                memory_type_index,
                memory_type_count: device.physical_device().memory_types().len() as u32,
            })?;

        // VUID-VkMemoryAllocateInfo-memoryTypeIndex-01872
        if memory_type.is_protected() && !device.enabled_features().protected_memory {
            return Err(DeviceMemoryAllocationError::FeatureNotEnabled {
                feature: "protected_memory",
                reason: "selected memory type is protected",
            });
        }

        // VUID-VkMemoryAllocateInfo-pNext-01874
        assert!(allocation_size != 0);

        // VUID-vkAllocateMemory-pAllocateInfo-01713
        let heap_size = memory_type.heap().size();
        if heap_size != 0 && allocation_size > heap_size {
            return Err(DeviceMemoryAllocationError::MemoryTypeHeapSizeExceeded {
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
                        return Err(
                            DeviceMemoryAllocationError::DedicatedAllocationSizeMismatch {
                                allocation_size,
                                required_size,
                            },
                        );
                    }
                }
                DedicatedAllocation::Image(image) => {
                    // VUID-VkMemoryDedicatedAllocateInfo-commonparent
                    assert_eq!(device, image.device().as_ref());

                    let required_size = image.memory_requirements().size;

                    // VUID-VkMemoryDedicatedAllocateInfo-image-02964
                    if allocation_size != required_size {
                        return Err(
                            DeviceMemoryAllocationError::DedicatedAllocationSizeMismatch {
                                allocation_size,
                                required_size,
                            },
                        );
                    }
                }
            }
        }

        // VUID-VkMemoryAllocateInfo-pNext-00639
        // VUID-VkExportMemoryAllocateInfo-handleTypes-00656
        // TODO: how do you fullfill this when you don't know the image or buffer parameters?
        // Does exporting memory require specifying these parameters up front, and does it tie the
        // allocation to only images or buffers of that type?

        if export_handle_types.opaque_fd && !device.enabled_extensions().khr_external_memory_fd {
            return Err(DeviceMemoryAllocationError::ExtensionNotEnabled {
                extension: "khr_external_memory_fd",
                reason: "`export_handle_types.opaque_fd` was set",
            });
        }

        if export_handle_types.dma_buf && !device.enabled_extensions().ext_external_memory_dma_buf {
            return Err(DeviceMemoryAllocationError::ExtensionNotEnabled {
                extension: "ext_external_memory_dma_buf",
                reason: "`export_handle_types.dma_buf` was set",
            });
        }

        if let Some(import_info) = import_info {
            match import_info {
                &mut MemoryImportInfo::Fd {
                    handle_type,
                    ref file,
                } => {
                    if !device.enabled_extensions().khr_external_memory_fd {
                        return Err(DeviceMemoryAllocationError::ExtensionNotEnabled {
                            extension: "khr_external_memory_fd",
                            reason: "`import_info` was `MemoryImportInfo::Fd`",
                        });
                    }

                    #[cfg(not(unix))]
                    unreachable!(
                        "`khr_external_memory_fd` was somehow enabled on a non-Unix system"
                    );

                    #[cfg(unix)]
                    {
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
                            ExternalMemoryHandleType::DmaBuf => {
                                if !device.enabled_extensions().ext_external_memory_dma_buf {
                                    return Err(DeviceMemoryAllocationError::ExtensionNotEnabled {
                                    extension: "ext_external_memory_dma_buf",
                                    reason: "`import_info` was `MemoryImportInfo::Fd` and `handle_type` was `ExternalMemoryHandleType::DmaBuf`"
                                });
                                }
                            }
                            _ => {
                                return Err(
                                    DeviceMemoryAllocationError::ImportFdHandleTypeNotSupported {
                                        handle_type,
                                    },
                                )
                            }
                        }

                        // VUID-VkMemoryAllocateInfo-memoryTypeIndex-00648
                        // Can't validate, must be ensured by user
                    }
                }
            }
        }

        Ok(())
    }

    unsafe fn create(
        device: &Device,
        allocate_info: &MemoryAllocateInfo,
        import_info: Option<MemoryImportInfo>,
    ) -> Result<ash::vk::DeviceMemory, DeviceMemoryAllocationError> {
        let &MemoryAllocateInfo {
            allocation_size,
            memory_type_index,
            dedicated_allocation,
            export_handle_types,
            _ne: _,
        } = allocate_info;

        let mut allocate_info = ash::vk::MemoryAllocateInfo::builder()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type_index);

        // VUID-VkMemoryDedicatedAllocateInfo-image-01432
        let mut dedicated_allocate_info = if let Some(dedicated_allocation) = dedicated_allocation {
            Some(match dedicated_allocation {
                DedicatedAllocation::Buffer(buffer) => ash::vk::MemoryDedicatedAllocateInfo {
                    buffer: buffer.internal_object(),
                    ..Default::default()
                },
                DedicatedAllocation::Image(image) => ash::vk::MemoryDedicatedAllocateInfo {
                    image: image.internal_object(),
                    ..Default::default()
                },
            })
        } else {
            None
        };

        if let Some(info) = dedicated_allocate_info.as_mut() {
            allocate_info = allocate_info.push_next(info);
        }

        let mut export_allocate_info = if export_handle_types != ExternalMemoryHandleTypes::none() {
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

        let mut allocation_count = device.allocation_count().lock().expect("Poisoned mutex");

        // VUID-vkAllocateMemory-maxMemoryAllocationCount-04101
        // This is technically validation, but it must be atomic with the `allocate_memory` call.
        if *allocation_count
            >= device
                .physical_device()
                .properties()
                .max_memory_allocation_count
        {
            return Err(DeviceMemoryAllocationError::TooManyObjects);
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.allocate_memory(
                device.internal_object(),
                &allocate_info.build(),
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        *allocation_count += 1;

        Ok(handle)
    }

    fn map_allocation(
        device: Arc<Device>,
        mem: DeviceMemory,
    ) -> Result<MappedDeviceMemory, DeviceMemoryAllocationError> {
        let fns = device.fns();
        let coherent = mem.memory_type().is_host_coherent();
        let ptr = unsafe {
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.map_memory(
                device.internal_object(),
                mem.handle,
                0,
                mem.allocation_size,
                ash::vk::MemoryMapFlags::empty(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(MappedDeviceMemory {
            memory: mem,
            pointer: ptr,
            coherent,
        })
    }

    /// Returns the memory type that this memory was allocated from.
    #[inline]
    pub fn memory_type(&self) -> MemoryType {
        self.device
            .physical_device()
            .memory_type_by_id(self.memory_type_index)
            .unwrap()
    }

    /// Returns the size in bytes of the memory allocation.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.allocation_size
    }

    /// Exports the device memory into a Unix file descriptor. The caller owns the returned `File`.
    ///
    /// # Panic
    ///
    /// - Panics if the user requests an invalid handle type for this device memory object.
    #[inline]
    pub fn export_fd(
        &self,
        handle_type: ExternalMemoryHandleType,
    ) -> Result<std::fs::File, DeviceMemoryExportError> {
        // VUID-VkMemoryGetFdInfoKHR-handleType-00672
        if !matches!(
            handle_type,
            ExternalMemoryHandleType::OpaqueFd | ExternalMemoryHandleType::DmaBuf
        ) {
            return Err(DeviceMemoryExportError::HandleTypeNotSupported { handle_type });
        }

        // VUID-VkMemoryGetFdInfoKHR-handleType-00671
        if !ash::vk::ExternalMemoryHandleTypeFlags::from(self.export_handle_types)
            .intersects(ash::vk::ExternalMemoryHandleTypeFlags::from(handle_type))
        {
            return Err(DeviceMemoryExportError::HandleTypeNotSupported { handle_type });
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
                check_errors(fns.khr_external_memory_fd.get_memory_fd_khr(
                    self.device.internal_object(),
                    &info,
                    output.as_mut_ptr(),
                ))?;
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
            fns.v1_0
                .free_memory(self.device.internal_object(), self.handle, ptr::null());
            let mut allocation_count = self
                .device
                .allocation_count()
                .lock()
                .expect("Poisoned mutex");
            *allocation_count -= 1;
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
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device.hash(state);
    }
}

/// Error type returned by functions related to `DeviceMemory`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeviceMemoryAllocationError {
    /// Not enough memory available.
    OomError(OomError),

    /// The maximum number of allocations has been exceeded.
    TooManyObjects,

    /// Memory map failed.
    MemoryMapFailed,

    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },
    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
    },

    /// `dedicated_allocation` was `Some`, but the provided `allocation_size`  was different from
    /// the required size of the buffer or image.
    DedicatedAllocationSizeMismatch {
        allocation_size: DeviceSize,
        required_size: DeviceSize,
    },

    /// The provided `MemoryImportInfo::Fd::handle_type` is not supported for file descriptors.
    ImportFdHandleTypeNotSupported {
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

    /// Spec violation, containing the Valid Usage ID (VUID) from the Vulkan spec.
    // TODO: Remove
    SpecViolation(u32),

    /// An implicit violation that's convered in the Vulkan spec.
    // TODO: Remove
    ImplicitSpecViolation(&'static str),
}

impl error::Error for DeviceMemoryAllocationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for DeviceMemoryAllocationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::OomError(_) => write!(fmt, "not enough memory available"),
            Self::TooManyObjects => {
                write!(fmt, "the maximum number of allocations has been exceeded")
            }
            Self::MemoryMapFailed => write!(fmt, "memory map failed"),
            Self::ExtensionNotEnabled { extension, reason } => write!(
                fmt,
                "the extension {} must be enabled: {}",
                extension, reason
            ),
            Self::FeatureNotEnabled { feature, reason } => {
                write!(fmt, "the feature {} must be enabled: {}", feature, reason)
            }
            Self::DedicatedAllocationSizeMismatch { allocation_size, required_size } => write!(
                fmt,
                "`dedicated_allocation` was `Some`, but the provided `allocation_size` ({}) was different from the required size of the buffer or image ({})",
                allocation_size, required_size,
            ),
            Self::ImportFdHandleTypeNotSupported { handle_type } => write!(
                fmt,
                "the provided `MemoryImportInfo::Fd::handle_type` ({:?}) is not supported for file descriptors",
                handle_type,
            ),
            Self::MemoryTypeHeapSizeExceeded { allocation_size, heap_size } => write!(
                fmt,
                "the provided `allocation_size` ({}) was greater than the memory type's heap size ({})",
                allocation_size, heap_size,
            ),
            Self::MemoryTypeIndexOutOfRange { memory_type_index, memory_type_count } => write!(
                fmt,
                "the provided `memory_type_index` ({}) was not less than the number of memory types in the physical device ({})",
                memory_type_index, memory_type_count,
            ),

            Self::SpecViolation(u) => {
                write!(fmt, "valid usage ID check {} failed", u)
            }
            Self::ImplicitSpecViolation(e) => {
                write!(fmt, "Implicit spec violation failed {}", e)
            }
        }
    }
}

impl From<Error> for DeviceMemoryAllocationError {
    #[inline]
    fn from(err: Error) -> Self {
        match err {
            e @ Error::OutOfHostMemory | e @ Error::OutOfDeviceMemory => Self::OomError(e.into()),
            Error::TooManyObjects => Self::TooManyObjects,
            Error::MemoryMapFailed => Self::MemoryMapFailed,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for DeviceMemoryAllocationError {
    #[inline]
    fn from(err: OomError) -> Self {
        Self::OomError(err)
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

    pub _ne: crate::NonExhaustive,
}

impl Default for MemoryAllocateInfo<'static> {
    #[inline]
    fn default() -> Self {
        Self {
            allocation_size: 0,
            memory_type_index: u32::MAX,
            dedicated_allocation: None,
            export_handle_types: ExternalMemoryHandleTypes::none(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl<'d> MemoryAllocateInfo<'d> {
    /// Returns a `MemoryAllocateInfo` with the specified `dedicated_allocation`.
    pub fn dedicated_allocation(dedicated_allocation: DedicatedAllocation<'d>) -> Self {
        Self {
            allocation_size: 0,
            memory_type_index: u32::MAX,
            dedicated_allocation: Some(dedicated_allocation),
            export_handle_types: ExternalMemoryHandleTypes::none(),
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
}

/// Describes a handle type used for Vulkan external memory apis.  This is **not** just a
/// suggestion.  Check out vkExternalMemoryHandleTypeFlagBits in the Vulkan spec.
///
/// If you specify an handle type that doesnt make sense (for example, using a dma-buf handle type
/// on Windows) when using this handle, a panic will happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ExternalMemoryHandleType {
    OpaqueFd = ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD.as_raw(),
    OpaqueWin32 = ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32.as_raw(),
    OpaqueWin32Kmt = ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT.as_raw(),
    D3D11Texture = ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE.as_raw(),
    D3D11TextureKmt = ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT.as_raw(),
    D3D12Heap = ash::vk::ExternalMemoryHandleTypeFlags::D3D12_HEAP.as_raw(),
    D3D12Resource = ash::vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE.as_raw(),
    DmaBuf = ash::vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT.as_raw(),
    AndroidHardwareBuffer =
        ash::vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID.as_raw(),
    HostAllocation = ash::vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT.as_raw(),
    HostMappedForeignMemory =
        ash::vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT.as_raw(),
}

impl From<ExternalMemoryHandleType> for ash::vk::ExternalMemoryHandleTypeFlags {
    fn from(val: ExternalMemoryHandleType) -> Self {
        Self::from_raw(val as u32)
    }
}

/// A mask of multiple handle types.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExternalMemoryHandleTypes {
    pub opaque_fd: bool,
    pub opaque_win32: bool,
    pub opaque_win32_kmt: bool,
    pub d3d11_texture: bool,
    pub d3d11_texture_kmt: bool,
    pub d3d12_heap: bool,
    pub d3d12_resource: bool,
    pub dma_buf: bool,
    pub android_hardware_buffer: bool,
    pub host_allocation: bool,
    pub host_mapped_foreign_memory: bool,
}

impl ExternalMemoryHandleTypes {
    /// Builds a `ExternalMemoryHandleTypes` with all values set to false. Useful as a default value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::memory::ExternalMemoryHandleTypes as ExternalMemoryHandleTypes;
    ///
    /// let _handle_type = ExternalMemoryHandleTypes {
    ///     opaque_fd: true,
    ///     .. ExternalMemoryHandleTypes::none()
    /// };
    /// ```
    #[inline]
    pub fn none() -> Self {
        ExternalMemoryHandleTypes {
            opaque_fd: false,
            opaque_win32: false,
            opaque_win32_kmt: false,
            d3d11_texture: false,
            d3d11_texture_kmt: false,
            d3d12_heap: false,
            d3d12_resource: false,
            dma_buf: false,
            android_hardware_buffer: false,
            host_allocation: false,
            host_mapped_foreign_memory: false,
        }
    }

    /// Builds an `ExternalMemoryHandleTypes` for a posix file descriptor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::memory::ExternalMemoryHandleTypes as ExternalMemoryHandleTypes;
    ///
    /// let _handle_type = ExternalMemoryHandleTypes::posix();
    /// ```
    #[inline]
    pub fn posix() -> ExternalMemoryHandleTypes {
        ExternalMemoryHandleTypes {
            opaque_fd: true,
            ..ExternalMemoryHandleTypes::none()
        }
    }

    /// Returns whether any of the fields are set.
    #[inline]
    pub fn is_empty(&self) -> bool {
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
        } = *self;

        !(opaque_fd
            || opaque_win32
            || opaque_win32_kmt
            || d3d11_texture
            || d3d11_texture_kmt
            || d3d12_heap
            || d3d12_resource
            || dma_buf
            || android_hardware_buffer
            || host_allocation
            || host_mapped_foreign_memory)
    }

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
        } = *self;

        [
            opaque_fd.then(|| ExternalMemoryHandleType::OpaqueFd),
            opaque_win32.then(|| ExternalMemoryHandleType::OpaqueWin32),
            opaque_win32_kmt.then(|| ExternalMemoryHandleType::OpaqueWin32Kmt),
            d3d11_texture.then(|| ExternalMemoryHandleType::D3D11Texture),
            d3d11_texture_kmt.then(|| ExternalMemoryHandleType::D3D11TextureKmt),
            d3d12_heap.then(|| ExternalMemoryHandleType::D3D12Heap),
            d3d12_resource.then(|| ExternalMemoryHandleType::D3D12Resource),
            dma_buf.then(|| ExternalMemoryHandleType::DmaBuf),
            android_hardware_buffer.then(|| ExternalMemoryHandleType::AndroidHardwareBuffer),
            host_allocation.then(|| ExternalMemoryHandleType::HostAllocation),
            host_mapped_foreign_memory.then(|| ExternalMemoryHandleType::HostMappedForeignMemory),
        ]
        .into_iter()
        .flatten()
    }
}

impl From<ExternalMemoryHandleTypes> for ash::vk::ExternalMemoryHandleTypeFlags {
    #[inline]
    fn from(val: ExternalMemoryHandleTypes) -> Self {
        let mut result = ash::vk::ExternalMemoryHandleTypeFlags::empty();
        if val.opaque_fd {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        }
        if val.opaque_win32 {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;
        }
        if val.opaque_win32_kmt {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT;
        }
        if val.d3d11_texture {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE;
        }
        if val.d3d11_texture_kmt {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT;
        }
        if val.d3d12_heap {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::D3D12_HEAP;
        }
        if val.d3d12_resource {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE;
        }
        if val.dma_buf {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        }
        if val.android_hardware_buffer {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID;
        }
        if val.host_allocation {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        }
        if val.host_mapped_foreign_memory {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT
        }
        result
    }
}

impl From<ash::vk::ExternalMemoryHandleTypeFlags> for ExternalMemoryHandleTypes {
    fn from(val: ash::vk::ExternalMemoryHandleTypeFlags) -> Self {
        ExternalMemoryHandleTypes {
            opaque_fd: !(val & ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD).is_empty(),
            opaque_win32: !(val & ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32).is_empty(),
            opaque_win32_kmt: !(val & ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT)
                .is_empty(),
            d3d11_texture: !(val & ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE)
                .is_empty(),
            d3d11_texture_kmt: !(val & ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT)
                .is_empty(),
            d3d12_heap: !(val & ash::vk::ExternalMemoryHandleTypeFlags::D3D12_HEAP).is_empty(),
            d3d12_resource: !(val & ash::vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE)
                .is_empty(),
            dma_buf: !(val & ash::vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT).is_empty(),
            android_hardware_buffer: !(val
                & ash::vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID)
                .is_empty(),
            host_allocation: !(val & ash::vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                .is_empty(),
            host_mapped_foreign_memory: !(val
                & ash::vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT)
                .is_empty(),
        }
    }
}

impl BitOr for ExternalMemoryHandleTypes {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        ExternalMemoryHandleTypes {
            opaque_fd: self.opaque_fd || rhs.opaque_fd,
            opaque_win32: self.opaque_win32 || rhs.opaque_win32,
            opaque_win32_kmt: self.opaque_win32_kmt || rhs.opaque_win32_kmt,
            d3d11_texture: self.d3d11_texture || rhs.d3d11_texture,
            d3d11_texture_kmt: self.d3d11_texture_kmt || rhs.d3d11_texture_kmt,
            d3d12_heap: self.d3d12_heap || rhs.d3d12_heap,
            d3d12_resource: self.d3d12_resource || rhs.d3d12_resource,
            dma_buf: self.dma_buf || rhs.dma_buf,
            android_hardware_buffer: self.android_hardware_buffer || rhs.android_hardware_buffer,
            host_allocation: self.host_allocation || rhs.host_allocation,
            host_mapped_foreign_memory: self.host_mapped_foreign_memory
                || rhs.host_mapped_foreign_memory,
        }
    }
}

/// Error type returned by functions related to `DeviceMemory`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeviceMemoryExportError {
    /// Not enough memory available.
    OomError(OomError),

    /// The maximum number of allocations has been exceeded.
    TooManyObjects,

    /// The requested export handle type is not supported for this operation, or was not provided in
    /// `export_handle_types` when allocating the memory.
    HandleTypeNotSupported {
        handle_type: ExternalMemoryHandleType,
    },
}

impl error::Error for DeviceMemoryExportError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for DeviceMemoryExportError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::OomError(_) => write!(fmt, "not enough memory available"),
            Self::TooManyObjects => {
                write!(fmt, "the maximum number of allocations has been exceeded")
            }
            Self::HandleTypeNotSupported {
                handle_type,
            } => write!(
                fmt,
                "the requested export handle type ({:?}) is not supported for this operation, or was not provided in `export_handle_types` when allocating the memory",
                handle_type,
            ),
        }
    }
}

impl From<Error> for DeviceMemoryExportError {
    #[inline]
    fn from(err: Error) -> Self {
        match err {
            e @ Error::OutOfHostMemory | e @ Error::OutOfDeviceMemory => Self::OomError(e.into()),
            Error::TooManyObjects => Self::TooManyObjects,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for DeviceMemoryExportError {
    #[inline]
    fn from(err: OomError) -> DeviceMemoryExportError {
        Self::OomError(err)
    }
}

/// Represents memory that has been allocated and mapped in CPU accessible space.
///
/// Can be obtained with `DeviceMemory::alloc_and_map`. The function will panic if the memory type
/// is not host-accessible.
///
/// In order to access the content of the allocated memory, you can use the `read_write` method.
/// This method returns a guard object that derefs to the content.
///
/// # Example
///
/// ```
/// use vulkano::memory::{DeviceMemory, MemoryAllocateInfo};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// // The memory type must be mappable.
/// let memory_type = device.physical_device().memory_types()
///                     .filter(|t| t.is_host_visible())
///                     .next().unwrap();    // Vk specs guarantee that this can't fail
///
/// // Allocates 1KB of memory.
/// let memory = DeviceMemory::allocate_and_map(
///     device.clone(),
///     MemoryAllocateInfo {
///         allocation_size: 1024,
///         memory_type_index: memory_type.id(),
///         ..Default::default()
///     },
/// ).unwrap();
///
/// // Get access to the content. Note that this is very unsafe for two reasons: 1) the content is
/// // uninitialized, and 2) the access is unsynchronized.
/// unsafe {
///     let mut content = memory.read_write::<[u8]>(0 .. 1024);
///     content[12] = 54;       // `content` derefs to a `&[u8]` or a `&mut [u8]`
/// }
/// ```
pub struct MappedDeviceMemory {
    memory: DeviceMemory,
    pointer: *mut c_void,
    coherent: bool,
}

// Note that `MappedDeviceMemory` doesn't implement `Drop`, as we don't need to unmap memory before
// freeing it.
//
// Vulkan specs, documentation of `vkFreeMemory`:
// > If a memory object is mapped at the time it is freed, it is implicitly unmapped.
//

impl MappedDeviceMemory {
    /// Unmaps the memory. It will no longer be accessible from the CPU.
    pub fn unmap(self) -> DeviceMemory {
        unsafe {
            let device = self.memory.device();
            let fns = device.fns();
            fns.v1_0
                .unmap_memory(device.internal_object(), self.memory.handle);
        }

        self.memory
    }

    /// Gives access to the content of the memory.
    ///
    /// This function takes care of calling `vkInvalidateMappedMemoryRanges` and
    /// `vkFlushMappedMemoryRanges` on the given range. You are therefore encouraged to use the
    /// smallest range as possible, and to not call this function multiple times in a row for
    /// several small changes.
    ///
    /// # Safety
    ///
    /// - Type safety is not checked. You must ensure that `T` corresponds to the content of the
    ///   buffer.
    /// - Accesses are not synchronized. Synchronization must be handled outside of
    ///   the `MappedDeviceMemory`.
    ///
    #[inline]
    pub unsafe fn read_write<T: ?Sized>(&self, range: Range<DeviceSize>) -> CpuAccess<T>
    where
        T: Content,
    {
        let fns = self.memory.device().fns();
        let pointer = T::ref_from_ptr(
            (self.pointer as usize + range.start as usize) as *mut _,
            (range.end - range.start) as usize,
        )
        .unwrap(); // TODO: error

        if !self.coherent {
            let range = ash::vk::MappedMemoryRange {
                memory: self.memory.internal_object(),
                offset: range.start,
                size: range.end - range.start,
                ..Default::default()
            };

            // TODO: return result instead?
            check_errors(fns.v1_0.invalidate_mapped_memory_ranges(
                self.memory.device().internal_object(),
                1,
                &range,
            ))
            .unwrap();
        }

        CpuAccess {
            pointer: pointer,
            mem: self,
            coherent: self.coherent,
            range,
        }
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

impl fmt::Debug for MappedDeviceMemory {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_tuple("MappedDeviceMemory")
            .field(&self.memory)
            .finish()
    }
}

unsafe impl Send for DeviceMemoryMapping {}
unsafe impl Sync for DeviceMemoryMapping {}

/// Represents memory mapped in CPU accessible space.
///
/// Takes an additional reference on the underlying device memory and device.
pub struct DeviceMemoryMapping {
    memory: Arc<DeviceMemory>,
    pointer: *mut c_void,
    coherent: bool,
}

impl DeviceMemoryMapping {
    /// Creates a new `DeviceMemoryMapping` object given the previously allocated `device` and `memory`.
    pub fn new(
        memory: Arc<DeviceMemory>,
        offset: DeviceSize,
        size: DeviceSize,
        flags: u32,
    ) -> Result<DeviceMemoryMapping, DeviceMemoryAllocationError> {
        // VUID-vkMapMemory-memory-00678: "memory must not be currently host mapped".
        let mut mapped = memory.mapped.lock().expect("Poisoned mutex");

        if *mapped {
            return Err(DeviceMemoryAllocationError::SpecViolation(678));
        }

        // VUID-vkMapMemory-offset-00679: "offset must be less than the size of memory"
        if size != ash::vk::WHOLE_SIZE && offset >= memory.size() {
            return Err(DeviceMemoryAllocationError::SpecViolation(679));
        }

        // VUID-vkMapMemory-size-00680: "If size is not equal to VK_WHOLE_SIZE, size must be
        // greater than 0".
        if size != ash::vk::WHOLE_SIZE && size == 0 {
            return Err(DeviceMemoryAllocationError::SpecViolation(680));
        }

        // VUID-vkMapMemory-size-00681: "If size is not equal to VK_WHOLE_SIZE, size must be less
        // than or equal to the size of the memory minus offset".
        if size != ash::vk::WHOLE_SIZE && size > memory.size() - offset {
            return Err(DeviceMemoryAllocationError::SpecViolation(681));
        }

        // VUID-vkMapMemory-memory-00682: "memory must have been created with a memory type
        // that reports VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT"
        let coherent = memory.memory_type().is_host_coherent();
        if !coherent {
            return Err(DeviceMemoryAllocationError::SpecViolation(682));
        }

        // VUID-vkMapMemory-memory-00683: "memory must not have been allocated with multiple instances".
        // Confused about this one, so not implemented.

        // VUID-vkMapMemory-flags-zerobitmask: "flags must be 0".
        if flags != 0 {
            return Err(DeviceMemoryAllocationError::ImplicitSpecViolation(
                "VUID-vkMapMemory-flags-zerobitmask",
            ));
        }

        // VUID-vkMapMemory-device-parameter, VUID-vkMapMemory-memory-parameter and
        // VUID-vkMapMemory-ppData-parameter satisfied via Vulkano internally.

        let fns = memory.device().fns();
        let ptr = unsafe {
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.map_memory(
                memory.device().internal_object(),
                memory.handle,
                0,
                memory.allocation_size,
                ash::vk::MemoryMapFlags::empty(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        *mapped = true;
        std::mem::drop(mapped);

        Ok(DeviceMemoryMapping {
            memory,
            pointer: ptr,
            coherent,
        })
    }

    /// Returns the raw pointer associated with the `DeviceMemoryMapping`.
    ///
    /// # Safety
    ///
    /// The caller of this function must ensure that the use of the raw pointer does not outlive
    /// the associated `DeviceMemoryMapping`.
    pub unsafe fn as_ptr(&self) -> *mut u8 {
        self.pointer as *mut u8
    }
}

impl Drop for DeviceMemoryMapping {
    #[inline]
    fn drop(&mut self) {
        let mut mapped = self.memory.mapped.lock().expect("Poisoned mutex");

        unsafe {
            let device = self.memory.device();
            let fns = device.fns();
            fns.v1_0
                .unmap_memory(device.internal_object(), self.memory.handle);
        }

        *mapped = false;
    }
}

/// Object that can be used to read or write the content of a `MappedDeviceMemory`.
///
/// This object derefs to the content, just like a `MutexGuard` for example.
pub struct CpuAccess<'a, T: ?Sized + 'a> {
    pointer: *mut T,
    mem: &'a MappedDeviceMemory,
    coherent: bool,
    range: Range<DeviceSize>,
}

impl<'a, T: ?Sized + 'a> CpuAccess<'a, T> {
    /// Builds a new `CpuAccess` to access a sub-part of the current `CpuAccess`.
    ///
    /// This function is unstable. Don't use it directly.
    // TODO: unsafe?
    // TODO: decide what to do with this
    #[doc(hidden)]
    #[inline]
    pub fn map<U: ?Sized + 'a, F>(self, f: F) -> CpuAccess<'a, U>
    where
        F: FnOnce(*mut T) -> *mut U,
    {
        CpuAccess {
            pointer: f(self.pointer),
            mem: self.mem,
            coherent: self.coherent,
            range: self.range.clone(), // TODO: ?
        }
    }
}

unsafe impl<'a, T: ?Sized + 'a> Send for CpuAccess<'a, T> {}
unsafe impl<'a, T: ?Sized + 'a> Sync for CpuAccess<'a, T> {}

impl<'a, T: ?Sized + 'a> Deref for CpuAccess<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for CpuAccess<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer }
    }
}

impl<'a, T: ?Sized + 'a> Drop for CpuAccess<'a, T> {
    #[inline]
    fn drop(&mut self) {
        // If the memory doesn't have the `coherent` flag, we need to flush the data.
        if !self.coherent {
            let fns = self.mem.as_ref().device().fns();

            let range = ash::vk::MappedMemoryRange {
                memory: self.mem.as_ref().internal_object(),
                offset: self.range.start,
                size: self.range.end - self.range.start,
                ..Default::default()
            };

            unsafe {
                check_errors(fns.v1_0.flush_mapped_memory_ranges(
                    self.mem.as_ref().device().internal_object(),
                    1,
                    &range,
                ))
                .unwrap();
            }
        }
    }
}

#[repr(C)]
pub struct BaseOutStructure {
    pub s_type: i32,
    pub p_next: *mut BaseOutStructure,
}

pub(crate) unsafe fn ptr_chain_iter<T>(ptr: &mut T) -> impl Iterator<Item = *mut BaseOutStructure> {
    let ptr: *mut BaseOutStructure = ptr as *mut T as _;
    (0..).scan(ptr, |p_ptr, _| {
        if p_ptr.is_null() {
            return None;
        }
        let n_ptr = (**p_ptr).p_next as *mut BaseOutStructure;
        let old = *p_ptr;
        *p_ptr = n_ptr;
        Some(old)
    })
}

pub unsafe trait ExtendsMemoryAllocateInfo {}
unsafe impl ExtendsMemoryAllocateInfo for ash::vk::MemoryDedicatedAllocateInfoKHR {}
unsafe impl ExtendsMemoryAllocateInfo for ash::vk::ExportMemoryAllocateInfo {}
unsafe impl ExtendsMemoryAllocateInfo for ash::vk::ImportMemoryFdInfoKHR {}

#[cfg(test)]
mod tests {
    use super::MemoryAllocateInfo;
    use crate::memory::DeviceMemory;
    use crate::memory::DeviceMemoryAllocationError;
    use crate::OomError;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_type = device.physical_device().memory_types().next().unwrap();
        let _ = DeviceMemory::allocate(
            device.clone(),
            MemoryAllocateInfo {
                allocation_size: 256,
                memory_type_index: memory_type.id(),
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn zero_size() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_type = device.physical_device().memory_types().next().unwrap();
        assert_should_panic!({
            let _ = DeviceMemory::allocate(
                device.clone(),
                MemoryAllocateInfo {
                    allocation_size: 0,
                    memory_type_index: memory_type.id(),
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
        let memory_type = device
            .physical_device()
            .memory_types()
            .filter(|m| !m.is_lazily_allocated())
            .next()
            .unwrap();

        match DeviceMemory::allocate(
            device.clone(),
            MemoryAllocateInfo {
                allocation_size: 0xffffffffffffffff,
                memory_type_index: memory_type.id(),
                ..Default::default()
            },
        ) {
            Err(DeviceMemoryAllocationError::MemoryTypeHeapSizeExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    #[ignore] // TODO: test fails for now on Mesa+Intel
    fn oom_multi() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_type = device
            .physical_device()
            .memory_types()
            .filter(|m| !m.is_lazily_allocated())
            .next()
            .unwrap();
        let heap_size = memory_type.heap().size();

        let mut allocs = Vec::new();

        for _ in 0..4 {
            match DeviceMemory::allocate(
                device.clone(),
                MemoryAllocateInfo {
                    allocation_size: heap_size / 3,
                    memory_type_index: memory_type.id(),
                    ..Default::default()
                },
            ) {
                Err(DeviceMemoryAllocationError::OomError(OomError::OutOfDeviceMemory)) => return, // test succeeded
                Ok(a) => allocs.push(a),
                _ => (),
            }
        }

        panic!()
    }

    #[test]
    fn allocation_count() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_type = device.physical_device().memory_types().next().unwrap();
        assert_eq!(*device.allocation_count().lock().unwrap(), 0);
        let mem1 = DeviceMemory::allocate(
            device.clone(),
            MemoryAllocateInfo {
                allocation_size: 256,
                memory_type_index: memory_type.id(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(*device.allocation_count().lock().unwrap(), 1);
        {
            let mem2 = DeviceMemory::allocate(
                device.clone(),
                MemoryAllocateInfo {
                    allocation_size: 256,
                    memory_type_index: memory_type.id(),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(*device.allocation_count().lock().unwrap(), 2);
        }
        assert_eq!(*device.allocation_count().lock().unwrap(), 1);
    }
}
