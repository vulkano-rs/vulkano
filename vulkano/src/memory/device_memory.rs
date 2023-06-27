// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{DedicatedAllocation, DedicatedTo, DeviceAlignment};
use crate::{
    device::{Device, DeviceOwned},
    macros::{impl_id_counter, vulkan_bitflags, vulkan_bitflags_enum},
    memory::{is_aligned, MemoryPropertyFlags},
    DebugWrapper, DeviceSize, Requires, RequiresAllOf, RequiresOneOf, RuntimeError,
    ValidationError, Version, VulkanError, VulkanObject,
};
use std::{
    ffi::c_void,
    fs::File,
    mem::MaybeUninit,
    num::NonZeroU64,
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
    device: DebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    allocation_size: DeviceSize,
    memory_type_index: u32,
    dedicated_to: Option<DedicatedTo>,
    export_handle_types: ExternalMemoryHandleTypes,
    imported_handle_type: Option<ExternalMemoryHandleType>,
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
    #[inline]
    pub fn allocate(
        device: Arc<Device>,
        mut allocate_info: MemoryAllocateInfo<'_>,
    ) -> Result<Self, VulkanError> {
        if !(device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation)
        {
            // Fall back instead of erroring out
            allocate_info.dedicated_allocation = None;
        }

        Self::validate_allocate(&device, &allocate_info, None)?;

        unsafe { Ok(Self::allocate_unchecked(device, allocate_info, None)?) }
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
    #[inline]
    pub unsafe fn import(
        device: Arc<Device>,
        mut allocate_info: MemoryAllocateInfo<'_>,
        import_info: MemoryImportInfo,
    ) -> Result<Self, VulkanError> {
        if !(device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation)
        {
            // Fall back instead of erroring out
            allocate_info.dedicated_allocation = None;
        }

        Self::validate_allocate(&device, &allocate_info, Some(&import_info))?;

        Ok(Self::allocate_unchecked(
            device,
            allocate_info,
            Some(import_info),
        )?)
    }

    #[inline(never)]
    fn validate_allocate(
        device: &Device,
        allocate_info: &MemoryAllocateInfo<'_>,
        import_info: Option<&MemoryImportInfo>,
    ) -> Result<(), ValidationError> {
        allocate_info
            .validate(device)
            .map_err(|err| err.add_context("allocate_info"))?;

        if let Some(import_info) = import_info {
            import_info
                .validate(device)
                .map_err(|err| err.add_context("import_info"))?;
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline(never)]
    pub unsafe fn allocate_unchecked(
        device: Arc<Device>,
        mut allocate_info: MemoryAllocateInfo<'_>,
        import_info: Option<MemoryImportInfo>,
    ) -> Result<Self, RuntimeError> {
        if !(device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation)
        {
            // Fall back instead of erroring out
            allocate_info.dedicated_allocation = None;
        }

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
                    buffer: buffer.handle(),
                    ..Default::default()
                },
                DedicatedAllocation::Image(image) => ash::vk::MemoryDedicatedAllocateInfo {
                    image: image.handle(),
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

        let imported_handle_type = import_info.as_ref().map(|import_info| match import_info {
            MemoryImportInfo::Fd { handle_type, .. } => *handle_type,
            MemoryImportInfo::Win32 { handle_type, .. } => *handle_type,
        });

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
            .map_err(|_| RuntimeError::TooManyObjects)?;

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.allocate_memory)(
                device.handle(),
                &allocate_info.build(),
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(|e| {
                device.allocation_count.fetch_sub(1, Ordering::Release);
                RuntimeError::from(e)
            })?;

            output.assume_init()
        };

        Ok(DeviceMemory {
            handle,
            device: DebugWrapper(device),
            id: Self::next_id(),
            allocation_size,
            memory_type_index,
            dedicated_to: dedicated_allocation.map(Into::into),
            export_handle_types,
            imported_handle_type,
            flags,
        })
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
            dedicated_allocation,
            export_handle_types,
            flags,
            _ne: _,
        } = allocate_info;

        DeviceMemory {
            handle,
            device: DebugWrapper(device),
            id: Self::next_id(),
            allocation_size,
            memory_type_index,
            dedicated_to: dedicated_allocation.map(Into::into),
            export_handle_types,
            imported_handle_type: None,
            flags,
        }
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

    /// Returns `true` if the memory is a [dedicated] to a resource.
    ///
    /// [dedicated]: MemoryAllocateInfo#structfield.dedicated_allocation
    #[inline]
    pub fn is_dedicated(&self) -> bool {
        self.dedicated_to.is_some()
    }

    pub(crate) fn dedicated_to(&self) -> Option<DedicatedTo> {
        self.dedicated_to
    }

    /// Returns the handle types that can be exported from the memory allocation.
    #[inline]
    pub fn export_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.export_handle_types
    }

    /// Returns the handle type that the memory allocation was imported from, if any.
    #[inline]
    pub fn imported_handle_type(&self) -> Option<ExternalMemoryHandleType> {
        self.imported_handle_type
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
    /// `self` must have been allocated from a memory type that has the [`LAZILY_ALLOCATED`] flag
    /// set.
    ///
    /// [`LAZILY_ALLOCATED`]: crate::memory::MemoryPropertyFlags::LAZILY_ALLOCATED
    #[inline]
    pub fn commitment(&self) -> Result<DeviceSize, ValidationError> {
        self.validate_commitment()?;

        unsafe { Ok(self.commitment_unchecked()) }
    }

    fn validate_commitment(&self) -> Result<(), ValidationError> {
        let memory_type = &self
            .device
            .physical_device()
            .memory_properties()
            .memory_types[self.memory_type_index as usize];

        if !memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::LAZILY_ALLOCATED)
        {
            return Err(ValidationError {
                problem: "the `property_flags` of the memory type does not contain the \
                    `MemoryPropertyFlags::LAZILY_ALLOCATED` flag"
                    .into(),
                vuids: &["VUID-vkGetDeviceMemoryCommitment-memory-00690"],
                ..Default::default()
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn commitment_unchecked(&self) -> DeviceSize {
        let mut output: DeviceSize = 0;

        let fns = self.device.fns();
        (fns.v1_0.get_device_memory_commitment)(self.device.handle(), self.handle, &mut output);

        output
    }

    /// Exports the device memory into a Unix file descriptor. The caller owns the returned `File`.
    ///
    /// # Panics
    ///
    /// - Panics if the user requests an invalid handle type for this device memory object.
    #[inline]
    pub fn export_fd(&self, handle_type: ExternalMemoryHandleType) -> Result<File, VulkanError> {
        self.validate_export_fd(handle_type)?;

        unsafe { Ok(self.export_fd_unchecked(handle_type)?) }
    }

    fn validate_export_fd(
        &self,
        handle_type: ExternalMemoryHandleType,
    ) -> Result<(), ValidationError> {
        handle_type
            .validate_device(&self.device)
            .map_err(|err| ValidationError {
                context: "handle_type".into(),
                vuids: &["VUID-VkMemoryGetFdInfoKHR-handleType-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if !matches!(
            handle_type,
            ExternalMemoryHandleType::OpaqueFd | ExternalMemoryHandleType::DmaBuf
        ) {
            return Err(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalMemoryHandleType::OpaqueFd` or \
                    `ExternalMemoryHandleType::DmaBuf`"
                    .into(),
                vuids: &["VUID-VkMemoryGetFdInfoKHR-handleType-00672"],
                ..Default::default()
            });
        }

        if !self.export_handle_types.contains_enum(handle_type) {
            return Err(ValidationError {
                context: "handle_type".into(),
                problem: "is not contained in this memory's `export_handle_types`".into(),
                vuids: &["VUID-VkMemoryGetFdInfoKHR-handleType-00671"],
                ..Default::default()
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn export_fd_unchecked(
        &self,
        handle_type: ExternalMemoryHandleType,
    ) -> Result<File, RuntimeError> {
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
                    self.device.handle(),
                    &info,
                    output.as_mut_ptr(),
                )
                .result()
                .map_err(RuntimeError::from)?;
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
            (fns.v1_0.free_memory)(self.device.handle(), self.handle, ptr::null());
            self.device.allocation_count.fetch_sub(1, Ordering::Release);
        }
    }
}

unsafe impl VulkanObject for DeviceMemory {
    type Handle = ash::vk::DeviceMemory;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for DeviceMemory {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(DeviceMemory);

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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            allocation_size,
            memory_type_index,
            ref dedicated_allocation,
            export_handle_types,
            flags,
            _ne: _,
        } = self;

        let memory_properties = device.physical_device().memory_properties();
        let memory_type = memory_properties
            .memory_types
            .get(memory_type_index as usize)
            .ok_or(ValidationError {
                context: "memory_type_index".into(),
                problem: "is not less than the number of memory types in the device".into(),
                vuids: &["VUID-vkAllocateMemory-pAllocateInfo-01714"],
                ..Default::default()
            })?;
        let memory_heap = &memory_properties.memory_heaps[memory_type.heap_index as usize];

        if memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::PROTECTED)
            && !device.enabled_features().protected_memory
        {
            return Err(ValidationError {
                context: "memory_type_index".into(),
                problem: "refers to a memory type where `property_flags` contains \
                    `MemoryPropertyFlags::PROTECTED`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "protected_memory",
                )])]),
                vuids: &["VUID-VkMemoryAllocateInfo-memoryTypeIndex-01872"],
            });
        }

        if memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::DEVICE_COHERENT)
            && !device.enabled_features().device_coherent_memory
        {
            return Err(ValidationError {
                context: "memory_type_index".into(),
                problem: "refers to a memory type where `property_flags` contains \
                    `MemoryPropertyFlags::DEVICE_COHERENT`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "device_coherent_memory",
                )])]),
                vuids: &["VUID-vkAllocateMemory-deviceCoherentMemory-02790"],
            });
        }

        if allocation_size == 0 {
            return Err(ValidationError {
                context: "allocation_size".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkMemoryAllocateInfo-pNext-01874"],
                ..Default::default()
            });
        }

        if memory_heap.size != 0 && allocation_size > memory_heap.size {
            return Err(ValidationError {
                context: "allocation_size".into(),
                problem: "is greater than the size of the memory heap".into(),
                vuids: &["VUID-vkAllocateMemory-pAllocateInfo-01713"],
                ..Default::default()
            });
        }

        if let Some(dedicated_allocation) = dedicated_allocation {
            match dedicated_allocation {
                DedicatedAllocation::Buffer(buffer) => {
                    // VUID-VkMemoryDedicatedAllocateInfo-commonparent
                    assert_eq!(device, buffer.device().as_ref());

                    let required_size = buffer.memory_requirements().layout.size();

                    if allocation_size != required_size {
                        return Err(ValidationError {
                            problem: "`allocation_size` does not equal the size required for the \
                                buffer specified in `dedicated_allocation`"
                                .into(),
                            vuids: &["VUID-VkMemoryDedicatedAllocateInfo-buffer-02965"],
                            ..Default::default()
                        });
                    }
                }
                DedicatedAllocation::Image(image) => {
                    // VUID-VkMemoryDedicatedAllocateInfo-commonparent
                    assert_eq!(device, image.device().as_ref());

                    let required_size = image.memory_requirements()[0].layout.size();

                    if allocation_size != required_size {
                        return Err(ValidationError {
                            problem: "`allocation_size` does not equal the size required for the \
                                image specified in `dedicated_allocation`"
                                .into(),
                            vuids: &["VUID-VkMemoryDedicatedAllocateInfo-image-02964"],
                            ..Default::default()
                        });
                    }
                }
            }
        }

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_memory)
            {
                return Err(ValidationError {
                    context: "export_handle_types".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_external_memory")]),
                    ]),
                    ..Default::default()
                });
            }

            export_handle_types
                .validate_device(device)
                .map_err(|err| ValidationError {
                    context: "export_handle_types".into(),
                    vuids: &["VUID-VkExportMemoryAllocateInfo-handleTypes-parameter"],
                    ..ValidationError::from_requirement(err)
                })?;

            // VUID-VkMemoryAllocateInfo-pNext-00639
            // VUID-VkExportMemoryAllocateInfo-handleTypes-00656
            // TODO: how do you fullfill this when you don't know the image or buffer parameters?
            // Does exporting memory require specifying these parameters up front, and does it tie
            // the allocation to only images or buffers of that type?
        }

        if !flags.is_empty() {
            if !(device.physical_device().api_version() >= Version::V1_1
                || device.enabled_extensions().khr_device_group)
            {
                return Err(ValidationError {
                    context: "flags".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_device_group")]),
                    ]),
                    ..Default::default()
                });
            }

            if flags.intersects(MemoryAllocateFlags::DEVICE_ADDRESS) {
                if !((device.api_version() >= Version::V1_2
                    || device.enabled_extensions().khr_buffer_device_address)
                    && device.enabled_features().buffer_device_address)
                {
                    return Err(ValidationError {
                        context: "flags".into(),
                        problem: "contains `MemoryAllocateFlags::DEVICE_ADDRESS`".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[
                                Requires::APIVersion(Version::V1_2),
                                Requires::Feature("buffer_device_address"),
                            ]),
                            RequiresAllOf(&[
                                Requires::DeviceExtension("khr_buffer_device_address"),
                                Requires::Feature("buffer_device_address"),
                            ]),
                        ]),
                        vuids: &["VUID-VkMemoryAllocateInfo-flags-03331"],
                    });
                }
            }
        }

        Ok(())
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

impl MemoryImportInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        match self {
            MemoryImportInfo::Fd {
                #[cfg(unix)]
                handle_type,
                #[cfg(not(unix))]
                    handle_type: _,
                file: _,
            } => {
                if !device.enabled_extensions().khr_external_memory_fd {
                    return Err(ValidationError {
                        problem: "is `MemoryImportInfo::Fd`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceExtension("khr_external_memory_fd"),
                        ])]),
                        ..Default::default()
                    });
                }

                #[cfg(not(unix))]
                unreachable!("`khr_external_memory_fd` was somehow enabled on a non-Unix system");

                #[cfg(unix)]
                {
                    handle_type
                        .validate_device(device)
                        .map_err(|err| ValidationError {
                            context: "handle_type".into(),
                            vuids: &["VUID-VkImportMemoryFdInfoKHR-handleType-parameter"],
                            ..ValidationError::from_requirement(err)
                        })?;

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
                            return Err(ValidationError {
                                context: "handle_type".into(),
                                problem: "is not `ExternalMemoryHandleType::OpaqueFd` or \
                                    `ExternalMemoryHandleType::DmaBuf`"
                                    .into(),
                                vuids: &["VUID-VkImportMemoryFdInfoKHR-handleType-00669"],
                                ..Default::default()
                            });
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
                    return Err(ValidationError {
                        problem: "is `MemoryImportInfo::Win32`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceExtension("khr_external_memory_win32"),
                        ])]),
                        ..Default::default()
                    });
                }

                #[cfg(not(windows))]
                unreachable!(
                    "`khr_external_memory_win32` was somehow enabled on a non-Windows system"
                );

                #[cfg(windows)]
                {
                    handle_type
                        .validate_device(device)
                        .map_err(|err| ValidationError {
                            context: "handle_type".into(),
                            vuids: &["VUID-VkImportMemoryWin32HandleInfoKHR-handleType-parameter"],
                            ..ValidationError::from_requirement(err)
                        })?;

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
                            return Err(ValidationError {
                                context: "handle_type".into(),
                                problem: "is not `ExternalMemoryHandleType::OpaqueWin32` or \
                                    `ExternalMemoryHandleType::OpaqueWin32Kmt`"
                                    .into(),
                                vuids: &["VUID-VkImportMemoryWin32HandleInfoKHR-handleType-00660"],
                                ..Default::default()
                            });
                        }
                    }

                    // VUID-VkMemoryAllocateInfo-memoryTypeIndex-00645
                    // Can't validate, must be ensured by user
                }
            }
        }

        Ok(())
    }
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`ExternalMemoryHandleType`] values.
    ExternalMemoryHandleTypes,

    /// A handle type used to export or import memory to/from an external source.
    ExternalMemoryHandleType,

    = ExternalMemoryHandleTypeFlags(u32);

    /// A POSIX file descriptor handle that is only usable with Vulkan and compatible APIs.
    OPAQUE_FD, OpaqueFd = OPAQUE_FD,

    /// A Windows NT handle that is only usable with Vulkan and compatible APIs.
    OPAQUE_WIN32, OpaqueWin32 = OPAQUE_WIN32,

    /// A Windows global share handle that is only usable with Vulkan and compatible APIs.
    OPAQUE_WIN32_KMT, OpaqueWin32Kmt = OPAQUE_WIN32_KMT,

    /// A Windows NT handle that refers to a Direct3D 10 or 11 texture resource.
    D3D11_TEXTURE, D3D11Texture = D3D11_TEXTURE,

    /// A Windows global share handle that refers to a Direct3D 10 or 11 texture resource.
    D3D11_TEXTURE_KMT, D3D11TextureKmt = D3D11_TEXTURE_KMT,

    /// A Windows NT handle that refers to a Direct3D 12 heap resource.
    D3D12_HEAP, D3D12Heap = D3D12_HEAP,

    /// A Windows NT handle that refers to a Direct3D 12 committed resource.
    D3D12_RESOURCE, D3D12Resource = D3D12_RESOURCE,

    /// A POSIX file descriptor handle that refers to a Linux dma-buf.
    DMA_BUF, DmaBuf = DMA_BUF_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_external_memory_dma_buf)]),
    ]),

    /// A handle for an Android `AHardwareBuffer` object.
    ANDROID_HARDWARE_BUFFER, AndroidHardwareBuffer = ANDROID_HARDWARE_BUFFER_ANDROID
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(android_external_memory_android_hardware_buffer)]),
    ]),

    /// A pointer to memory that was allocated by the host.
    HOST_ALLOCATION, HostAllocation = HOST_ALLOCATION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_external_memory_host)]),
    ]),

    /// A pointer to a memory mapping on the host that maps non-host memory.
    HOST_MAPPED_FOREIGN_MEMORY, HostMappedForeignMemory = HOST_MAPPED_FOREIGN_MEMORY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_external_memory_host)]),
    ]),

    /// A Zircon handle to a virtual memory object.
    ZIRCON_VMO, ZirconVmo = ZIRCON_VMO_FUCHSIA
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(fuchsia_external_memory)]),
    ]),

    /// A Remote Direct Memory Address handle to an allocation that is accessible by remote devices.
    RDMA_ADDRESS, RdmaAddress = RDMA_ADDRESS_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_external_memory_rdma)]),
    ]),
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// A mask specifying flags for device memory allocation.
    MemoryAllocateFlags = MemoryAllocateFlags(u32);

    /* TODO: enable
    DEVICE_MASK = DEVICE_MASK,*/

    /// Specifies that the allocated device memory can be bound to a buffer created with the
    /// [`SHADER_DEVICE_ADDRESS`] usage. This requires that the [`buffer_device_address`] feature
    /// is enabled on the device and the [`ext_buffer_device_address`] extension is not enabled on
    /// the device.
    ///
    /// [`SHADER_DEVICE_ADDRESS`]: crate::buffer::BufferUsage::SHADER_DEVICE_ADDRESS
    /// [`buffer_device_address`]: crate::device::Features::buffer_device_address
    /// [`ext_buffer_device_address`]: crate::device::DeviceExtensions::ext_buffer_device_address
    DEVICE_ADDRESS = DEVICE_ADDRESS,

    /* TODO: enable
    DEVICE_ADDRESS_CAPTURE_REPLAY = DEVICE_ADDRESS_CAPTURE_REPLAY,*/
}

/// Represents device memory that has been mapped in a CPU-accessible space.
///
/// In order to access the contents of the allocated memory, you can use the `read` and `write`
/// methods.
///
/// # Examples
///
/// ```
/// use vulkano::memory::{DeviceMemory, MappedDeviceMemory, MemoryAllocateInfo, MemoryPropertyFlags};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// // The memory type must be mappable.
/// let memory_type_index = device
///     .physical_device()
///     .memory_properties()
///     .memory_types
///     .iter()
///     .position(|t| t.property_flags.intersects(MemoryPropertyFlags::HOST_VISIBLE))
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

    atom_size: DeviceAlignment,
    is_coherent: bool,
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
    #[inline]
    pub fn new(memory: DeviceMemory, range: Range<DeviceSize>) -> Result<Self, VulkanError> {
        Self::validate_new(&memory, range.clone())?;

        unsafe { Ok(Self::new_unchecked(memory, range)?) }
    }

    fn validate_new(
        memory: &DeviceMemory,
        range: Range<DeviceSize>,
    ) -> Result<(), ValidationError> {
        if range.is_empty() {
            return Err(ValidationError {
                context: "range".into(),
                problem: "is empty".into(),
                vuids: &["VUID-vkMapMemory-size-00680"],
                ..Default::default()
            });
        }

        let device = memory.device();
        let memory_type = &device.physical_device().memory_properties().memory_types
            [memory.memory_type_index() as usize];

        if !memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::HOST_VISIBLE)
        {
            return Err(ValidationError {
                context: "memory".into(),
                problem: "has a memory type whose `property_flags` does not contain \
                    `MemoryPropertyFlags::HOST_VISIBLE`"
                    .into(),
                vuids: &["VUID-vkMapMemory-memory-00682"],
                ..Default::default()
            });
        }

        // VUID-vkMapMemory-memory-00678
        // Guaranteed because we take ownership of `memory`, no other mapping can exist.

        if range.end > memory.allocation_size {
            return Err(ValidationError {
                problem: "`range.end` is greater than `memory.allocation_size()`".into(),
                vuids: &[
                    "VUID-vkMapMemory-offset-00679",
                    "VUID-vkMapMemory-size-00681",
                ],
                ..Default::default()
            });
        }

        let is_coherent = memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::HOST_COHERENT);
        let atom_size = device.physical_device().properties().non_coherent_atom_size;

        // Not required for merely mapping, but without this check the user can end up with
        // parts of the mapped memory at the start and end that they're not able to
        // invalidate/flush, which is probably unintended.
        if !is_coherent
            && (!is_aligned(range.start, atom_size)
                || (!is_aligned(range.end, atom_size) && range.end != memory.allocation_size))
        {
            return Err(ValidationError {
                problem: "`memory` has a memory type whose `property_flags` does not contain \
                    `MemoryPropertyFlags::HOST_COHERENT`, and `range.start` and/or `range.end` \
                    are not aligned to the `non_coherent_atom_size` device property"
                    .into(),
                ..Default::default()
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        memory: DeviceMemory,
        range: Range<DeviceSize>,
    ) -> Result<Self, RuntimeError> {
        let device = memory.device();

        let pointer = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.map_memory)(
                device.handle(),
                memory.handle,
                range.start,
                range.end - range.start,
                ash::vk::MemoryMapFlags::empty(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        let atom_size = device.physical_device().properties().non_coherent_atom_size;
        let memory_type = &device.physical_device().memory_properties().memory_types
            [memory.memory_type_index() as usize];
        let is_coherent = memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::HOST_COHERENT);

        Ok(MappedDeviceMemory {
            memory,
            pointer,
            range,
            atom_size,
            is_coherent,
        })
    }

    /// Unmaps the memory. It will no longer be accessible from the CPU.
    #[inline]
    pub fn unmap(self) -> DeviceMemory {
        unsafe {
            let device = self.memory.device();
            let fns = device.fns();
            (fns.v1_0.unmap_memory)(device.handle(), self.memory.handle);
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
    pub unsafe fn invalidate_range(&self, range: Range<DeviceSize>) -> Result<(), VulkanError> {
        self.validate_range(range.clone())?;

        Ok(self.invalidate_range_unchecked(range)?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn invalidate_range_unchecked(
        &self,
        range: Range<DeviceSize>,
    ) -> Result<(), RuntimeError> {
        if self.is_coherent {
            return Ok(());
        }

        let range = ash::vk::MappedMemoryRange {
            memory: self.memory.handle(),
            offset: range.start,
            size: range.end - range.start,
            ..Default::default()
        };

        let fns = self.memory.device().fns();
        (fns.v1_0.invalidate_mapped_memory_ranges)(self.memory.device().handle(), 1, &range)
            .result()
            .map_err(RuntimeError::from)?;

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
    pub unsafe fn flush_range(&self, range: Range<DeviceSize>) -> Result<(), VulkanError> {
        self.validate_range(range.clone())?;

        Ok(self.flush_range_unchecked(range)?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn flush_range_unchecked(
        &self,
        range: Range<DeviceSize>,
    ) -> Result<(), RuntimeError> {
        if self.is_coherent {
            return Ok(());
        }

        let range = ash::vk::MappedMemoryRange {
            memory: self.memory.handle(),
            offset: range.start,
            size: range.end - range.start,
            ..Default::default()
        };

        let fns = self.device().fns();
        (fns.v1_0.flush_mapped_memory_ranges)(self.memory.device().handle(), 1, &range)
            .result()
            .map_err(RuntimeError::from)?;

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
    pub unsafe fn read(&self, range: Range<DeviceSize>) -> Result<&[u8], ValidationError> {
        self.validate_range(range.clone())?;

        Ok(self.read_unchecked(range))
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn read_unchecked(&self, range: Range<DeviceSize>) -> &[u8] {
        slice::from_raw_parts(
            self.pointer.add((range.start - self.range.start) as usize) as *const u8,
            (range.end - range.start) as usize,
        )
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
    pub unsafe fn write(&self, range: Range<DeviceSize>) -> Result<&mut [u8], ValidationError> {
        self.validate_range(range.clone())?;

        Ok(self.write_unchecked(range))
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn write_unchecked(&self, range: Range<DeviceSize>) -> &mut [u8] {
        slice::from_raw_parts_mut(
            self.pointer.add((range.start - self.range.start) as usize) as *mut u8,
            (range.end - range.start) as usize,
        )
    }

    #[inline]
    fn validate_range(&self, range: Range<DeviceSize>) -> Result<(), ValidationError> {
        // VUID-VkMappedMemoryRange-memory-00684
        // Guaranteed because `self` owns the memory and it's mapped during our lifetime.

        if range.is_empty() {
            return Err(ValidationError {
                context: "range".into(),
                problem: "is empty".into(),
                ..Default::default()
            });
        }

        if range.start < self.range.start || range.end > self.range.end {
            return Err(ValidationError {
                context: "range".into(),
                problem: "is not within the mapped range of this mapped device memory".into(),
                vuids: &["VUID-VkMappedMemoryRange-size-00685"],
                ..Default::default()
            });
        }

        if !self.is_coherent {
            if !is_aligned(range.start, self.atom_size)
                || (!is_aligned(range.end, self.atom_size)
                    && range.end != self.memory.allocation_size)
            {
                return Err(ValidationError {
                    problem: "this mapped device memory has a memory type whose `property_flags` \
                        does not contain `MemoryPropertyFlags::HOST_COHERENT`, and \
                        `range.start` and/or `range.end` are not aligned to the \
                        `non_coherent_atom_size` device property"
                        .into(),
                    vuids: &[
                        "VUID-VkMappedMemoryRange-offset-00687",
                        "VUID-VkMappedMemoryRange-size-01390",
                    ],
                    ..Default::default()
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

#[cfg(test)]
mod tests {
    use super::MemoryAllocateInfo;
    use crate::memory::{DeviceMemory, MemoryPropertyFlags};

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
            .find_map(|(i, m)| {
                (!m.property_flags
                    .intersects(MemoryPropertyFlags::LAZILY_ALLOCATED))
                .then_some(i as u32)
            })
            .unwrap();

        match DeviceMemory::allocate(
            device,
            MemoryAllocateInfo {
                allocation_size: 0xffffffffffffffff,
                memory_type_index,
                ..Default::default()
            },
        ) {
            Err(_) => (),
            Ok(_) => panic!(),
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
            .find_map(|(i, m)| {
                (!m.property_flags
                    .intersects(MemoryPropertyFlags::LAZILY_ALLOCATED))
                .then_some((i as u32, m))
            })
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
                Err(_) => return, // test succeeded
                Ok(a) => allocs.push(a),
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
