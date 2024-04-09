use super::{DedicatedAllocation, DedicatedTo, DeviceAlignment};
use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_bitflags_enum},
    memory::{is_aligned, MemoryPropertyFlags},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version,
    VulkanError, VulkanObject,
};
use std::{
    ffi::c_void,
    fs::File,
    mem::MaybeUninit,
    num::NonZeroU64,
    ops::Range,
    ptr::{self, NonNull},
    slice,
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
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    allocation_size: DeviceSize,
    memory_type_index: u32,
    dedicated_to: Option<DedicatedTo>,
    export_handle_types: ExternalMemoryHandleTypes,
    imported_handle_type: Option<ExternalMemoryHandleType>,
    flags: MemoryAllocateFlags,

    mapping_state: Option<MappingState>,
    atom_size: DeviceAlignment,
    is_coherent: bool,
}

impl DeviceMemory {
    /// Allocates a block of memory from the device.
    ///
    /// Some platforms may have a limit on the maximum size of a single allocation. For example,
    /// certain systems may fail to create allocations with a size greater than or equal to 4GB.
    ///
    /// # Panics
    ///
    /// - Panics if `allocate_info.dedicated_allocation` is `Some` and the contained buffer or
    ///   image does not belong to `device`.
    #[inline]
    pub fn allocate(
        device: Arc<Device>,
        mut allocate_info: MemoryAllocateInfo<'_>,
    ) -> Result<Self, Validated<VulkanError>> {
        if !(device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation)
        {
            // Fall back instead of erroring out.
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
    /// - Panics if `allocate_info.dedicated_allocation` is `Some` and the contained buffer or
    ///   image does not belong to `device`.
    #[inline]
    pub unsafe fn import(
        device: Arc<Device>,
        mut allocate_info: MemoryAllocateInfo<'_>,
        import_info: MemoryImportInfo,
    ) -> Result<Self, Validated<VulkanError>> {
        if !(device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation)
        {
            // Fall back instead of erroring out.
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
    ) -> Result<(), Box<ValidationError>> {
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
    ) -> Result<Self, VulkanError> {
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

        let mut allocate_info_vk = ash::vk::MemoryAllocateInfo {
            allocation_size,
            memory_type_index,
            ..Default::default()
        };

        let mut dedicated_allocate_info_vk = None;
        let mut export_allocate_info_vk = None;
        let mut import_fd_info_vk = None;
        let mut import_win32_handle_info_vk = None;
        let mut flags_info_vk = None;

        // VUID-VkMemoryDedicatedAllocateInfo-image-01432
        if let Some(dedicated_allocation) = dedicated_allocation {
            let next = dedicated_allocate_info_vk.insert(match dedicated_allocation {
                DedicatedAllocation::Buffer(buffer) => ash::vk::MemoryDedicatedAllocateInfo {
                    buffer: buffer.handle(),
                    ..Default::default()
                },
                DedicatedAllocation::Image(image) => ash::vk::MemoryDedicatedAllocateInfo {
                    image: image.handle(),
                    ..Default::default()
                },
            });

            next.p_next = allocate_info_vk.p_next;
            allocate_info_vk.p_next = <*const _>::cast(next);
        }

        if !export_handle_types.is_empty() {
            let next = export_allocate_info_vk.insert(ash::vk::ExportMemoryAllocateInfo {
                handle_types: export_handle_types.into(),
                ..Default::default()
            });

            next.p_next = allocate_info_vk.p_next;
            allocate_info_vk.p_next = <*const _>::cast(next);
        }

        let imported_handle_type = import_info.as_ref().map(|import_info| match import_info {
            MemoryImportInfo::Fd { handle_type, .. } => *handle_type,
            MemoryImportInfo::Win32 { handle_type, .. } => *handle_type,
        });

        if let Some(import_info) = import_info {
            match import_info {
                MemoryImportInfo::Fd { handle_type, file } => {
                    #[cfg(unix)]
                    let fd = {
                        use std::os::fd::IntoRawFd;
                        file.into_raw_fd()
                    };

                    #[cfg(not(unix))]
                    let fd = {
                        let _ = file;
                        -1
                    };

                    let next = import_fd_info_vk.insert(ash::vk::ImportMemoryFdInfoKHR {
                        handle_type: handle_type.into(),
                        fd,
                        ..Default::default()
                    });

                    next.p_next = allocate_info_vk.p_next;
                    allocate_info_vk.p_next = <*const _>::cast(next);
                }
                MemoryImportInfo::Win32 {
                    handle_type,
                    handle,
                } => {
                    let next = import_win32_handle_info_vk.insert(
                        ash::vk::ImportMemoryWin32HandleInfoKHR {
                            handle_type: handle_type.into(),
                            handle,
                            ..Default::default()
                        },
                    );

                    next.p_next = allocate_info_vk.p_next;
                    allocate_info_vk.p_next = <*const _>::cast(next);
                }
            }
        }

        if !flags.is_empty() {
            let next = flags_info_vk.insert(ash::vk::MemoryAllocateFlagsInfo {
                flags: flags.into(),
                ..Default::default()
            });

            next.p_next = allocate_info_vk.p_next;
            allocate_info_vk.p_next = <*const _>::cast(next);
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
                device.handle(),
                &allocate_info_vk,
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

        let atom_size = device.physical_device().properties().non_coherent_atom_size;

        let is_coherent = device.physical_device().memory_properties().memory_types
            [memory_type_index as usize]
            .property_flags
            .intersects(MemoryPropertyFlags::HOST_COHERENT);

        Ok(DeviceMemory {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            allocation_size,
            memory_type_index,
            dedicated_to: dedicated_allocation.map(Into::into),
            export_handle_types,
            imported_handle_type,
            flags,

            mapping_state: None,
            atom_size,
            is_coherent,
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

        let atom_size = device.physical_device().properties().non_coherent_atom_size;

        let is_coherent = device.physical_device().memory_properties().memory_types
            [memory_type_index as usize]
            .property_flags
            .intersects(MemoryPropertyFlags::HOST_COHERENT);

        DeviceMemory {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            allocation_size,
            memory_type_index,
            dedicated_to: dedicated_allocation.map(Into::into),
            export_handle_types,
            imported_handle_type: None,
            flags,

            mapping_state: None,
            atom_size,
            is_coherent,
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

    /// Returns the current mapping state, or [`None`] if the memory is not currently host-mapped.
    #[inline]
    pub fn mapping_state(&self) -> Option<&MappingState> {
        self.mapping_state.as_ref()
    }

    pub(crate) fn atom_size(&self) -> DeviceAlignment {
        self.atom_size
    }

    pub(crate) fn is_coherent(&self) -> bool {
        self.is_coherent
    }

    /// Maps a range of memory to be accessed by the host.
    ///
    /// `self` must not be host-mapped already and must be allocated from host-visible memory.
    #[inline]
    pub fn map(&mut self, map_info: MemoryMapInfo) -> Result<(), Validated<VulkanError>> {
        self.validate_map(&map_info)?;

        unsafe { Ok(self.map_unchecked(map_info)?) }
    }

    fn validate_map(&self, map_info: &MemoryMapInfo) -> Result<(), Box<ValidationError>> {
        if self.mapping_state.is_some() {
            return Err(Box::new(ValidationError {
                problem: "this device memory is already host-mapped".into(),
                vuids: &["VUID-vkMapMemory-memory-00678"],
                ..Default::default()
            }));
        }

        map_info
            .validate(self)
            .map_err(|err| err.add_context("map_info"))?;

        let memory_type = &self
            .device()
            .physical_device()
            .memory_properties()
            .memory_types[self.memory_type_index() as usize];

        if !memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::HOST_VISIBLE)
        {
            return Err(Box::new(ValidationError {
                problem: "`self.memory_type_index()` refers to a memory type whose \
                    `property_flags` does not contain `MemoryPropertyFlags::HOST_VISIBLE`"
                    .into(),
                vuids: &["VUID-vkMapMemory-memory-00682"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn map_unchecked(&mut self, map_info: MemoryMapInfo) -> Result<(), VulkanError> {
        let MemoryMapInfo {
            offset,
            size,
            placed_address,
            _ne: _,
        } = map_info;

        let device = self.device();

        let ptr = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();

            if device.enabled_extensions().khr_map_memory2 {
                let map_info_vk = ash::vk::MemoryMapInfoKHR {
                    flags: {
                        if placed_address.is_some() {
                            ash::vk::MemoryMapFlags::PLACED_EXT
                        } else {
                            ash::vk::MemoryMapFlags::empty()
                        }
                    },
                    memory: self.handle(),
                    offset,
                    size,
                    ..Default::default()
                };

                let mut map_placed_info_ext = ash::vk::MemoryMapPlacedInfoEXT::default();
                let map_info_vk = if let Some(placed_address) = placed_address {
                    map_placed_info_ext.p_placed_address = placed_address;

                    map_info_vk.push_next(&mut map_placed_info_ext)
                } else {
                    map_info_vk
                };

                (fns.khr_map_memory2.map_memory2_khr)(
                    device.handle(),
                    &map_info_vk,
                    output.as_mut_ptr(),
                )
                .result()
                .map_err(VulkanError::from)?;
            } else {
                (fns.v1_0.map_memory)(
                    device.handle(),
                    self.handle,
                    offset,
                    size,
                    ash::vk::MemoryMapFlags::empty(),
                    output.as_mut_ptr(),
                )
                .result()
                .map_err(VulkanError::from)?;
            }

            output.assume_init()
        };

        let ptr = NonNull::new(ptr).unwrap();
        let range = offset..offset + size;
        self.mapping_state = Some(MappingState { ptr, range });

        Ok(())
    }

    /// Unmaps the memory. It will no longer be accessible from the host.
    ///
    /// `self` must be currently host-mapped.
    //
    // NOTE(Marc): The `&mut` here is more than just because we need to mutate the struct.
    // `vkMapMemory` and `vkUnmapMemory` must be externally synchronized, but more importantly, if
    // we allowed unmapping through a shared reference, it would be possible to unmap a resource
    // that's currently being read or written by the host elsewhere, requiring even more locking on
    // each host access.
    #[inline]
    pub fn unmap(&mut self, unmap_info: MemoryUnmapInfo) -> Result<(), Validated<VulkanError>> {
        self.validate_unmap(&unmap_info)?;

        unsafe { self.unmap_unchecked(unmap_info) }?;

        Ok(())
    }

    fn validate_unmap(&self, unmap_info: &MemoryUnmapInfo) -> Result<(), Box<ValidationError>> {
        if self.mapping_state.is_none() {
            return Err(Box::new(ValidationError {
                problem: "this device memory is not currently host-mapped".into(),
                vuids: &["VUID-vkUnmapMemory-memory-00689"],
                ..Default::default()
            }));
        }

        unmap_info
            .validate(self)
            .map_err(|err| err.add_context("unmap_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn unmap_unchecked(
        &mut self,
        unmap_info: MemoryUnmapInfo,
    ) -> Result<(), VulkanError> {
        let MemoryUnmapInfo { _ne: _ } = unmap_info;

        let device = self.device();
        let fns = device.fns();

        if device.enabled_extensions().khr_map_memory2 {
            let unmap_info_vk = ash::vk::MemoryUnmapInfoKHR {
                flags: ash::vk::MemoryUnmapFlagsKHR::empty(),
                memory: self.handle(),
                ..Default::default()
            };

            (fns.khr_map_memory2.unmap_memory2_khr)(device.handle(), &unmap_info_vk)
                .result()
                .map_err(VulkanError::from)?;
        } else {
            (fns.v1_0.unmap_memory)(device.handle(), self.handle);
        }

        self.mapping_state = None;

        Ok(())
    }

    /// Invalidates the host cache for a range of mapped memory.
    ///
    /// If the device memory is not [host-coherent], you must call this function before the memory
    /// is read by the host, if the device previously wrote to the memory. It has no effect if the
    /// memory is host-coherent.
    ///
    /// # Safety
    ///
    /// - If there are memory writes by the device that have not been propagated into the host
    ///   cache, then there must not be any references in Rust code to any portion of the specified
    ///   `memory_range`.
    ///
    /// [host-coherent]: MemoryPropertyFlags::HOST_COHERENT
    /// [`map`]: Self::map
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    #[inline]
    pub unsafe fn invalidate_range(
        &self,
        memory_range: MappedMemoryRange,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_memory_range(&memory_range)?;

        Ok(self.invalidate_range_unchecked(memory_range)?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn invalidate_range_unchecked(
        &self,
        memory_range: MappedMemoryRange,
    ) -> Result<(), VulkanError> {
        if self.is_coherent {
            return Ok(());
        }

        let MappedMemoryRange {
            offset,
            size,
            _ne: _,
        } = memory_range;

        let memory_range_vk = ash::vk::MappedMemoryRange {
            memory: self.handle(),
            offset,
            size,
            ..Default::default()
        };

        let fns = self.device().fns();
        (fns.v1_0.invalidate_mapped_memory_ranges)(self.device().handle(), 1, &memory_range_vk)
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Flushes the host cache for a range of mapped memory.
    ///
    /// If the device memory is not [host-coherent], you must call this function after writing to
    /// the memory, if the device is going to read the memory. It has no effect if the memory is
    /// host-coherent.
    ///
    /// # Safety
    ///
    /// - There must be no operations pending or executing in a device queue, that access the
    ///   specified `memory_range`.
    ///
    /// [host-coherent]: MemoryPropertyFlags::HOST_COHERENT
    /// [`map`]: Self::map
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    #[inline]
    pub unsafe fn flush_range(
        &self,
        memory_range: MappedMemoryRange,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_memory_range(&memory_range)?;

        Ok(self.flush_range_unchecked(memory_range)?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn flush_range_unchecked(
        &self,
        memory_range: MappedMemoryRange,
    ) -> Result<(), VulkanError> {
        if self.is_coherent {
            return Ok(());
        }

        let MappedMemoryRange {
            offset,
            size,
            _ne: _,
        } = memory_range;

        let memory_range_vk = ash::vk::MappedMemoryRange {
            memory: self.handle(),
            offset,
            size,
            ..Default::default()
        };

        let fns = self.device().fns();
        (fns.v1_0.flush_mapped_memory_ranges)(self.device().handle(), 1, &memory_range_vk)
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    // NOTE(Marc): We are validating the parameters regardless of whether the memory is
    // non-coherent on purpose, to catch potential bugs arising because the code isn't tested on
    // such hardware.
    fn validate_memory_range(
        &self,
        memory_range: &MappedMemoryRange,
    ) -> Result<(), Box<ValidationError>> {
        memory_range
            .validate(self)
            .map_err(|err| err.add_context("memory_range"))?;

        Ok(())
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
    /// [`LAZILY_ALLOCATED`]: MemoryPropertyFlags::LAZILY_ALLOCATED
    #[inline]
    pub fn commitment(&self) -> Result<DeviceSize, Box<ValidationError>> {
        self.validate_commitment()?;

        unsafe { Ok(self.commitment_unchecked()) }
    }

    fn validate_commitment(&self) -> Result<(), Box<ValidationError>> {
        let memory_type = &self
            .device
            .physical_device()
            .memory_properties()
            .memory_types[self.memory_type_index as usize];

        if !memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::LAZILY_ALLOCATED)
        {
            return Err(Box::new(ValidationError {
                problem: "the `property_flags` of the memory type does not contain the \
                    `MemoryPropertyFlags::LAZILY_ALLOCATED` flag"
                    .into(),
                vuids: &["VUID-vkGetDeviceMemoryCommitment-memory-00690"],
                ..Default::default()
            }));
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
    pub fn export_fd(
        &self,
        handle_type: ExternalMemoryHandleType,
    ) -> Result<File, Validated<VulkanError>> {
        self.validate_export_fd(handle_type)?;

        unsafe { Ok(self.export_fd_unchecked(handle_type)?) }
    }

    fn validate_export_fd(
        &self,
        handle_type: ExternalMemoryHandleType,
    ) -> Result<(), Box<ValidationError>> {
        handle_type.validate_device(&self.device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkMemoryGetFdInfoKHR-handleType-parameter"])
        })?;

        if !matches!(
            handle_type,
            ExternalMemoryHandleType::OpaqueFd | ExternalMemoryHandleType::DmaBuf
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalMemoryHandleType::OpaqueFd` or \
                    `ExternalMemoryHandleType::DmaBuf`"
                    .into(),
                vuids: &["VUID-VkMemoryGetFdInfoKHR-handleType-00672"],
                ..Default::default()
            }));
        }

        if !self.export_handle_types.contains_enum(handle_type) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not contained in this memory's `export_handle_types`".into(),
                vuids: &["VUID-VkMemoryGetFdInfoKHR-handleType-00671"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn export_fd_unchecked(
        &self,
        handle_type: ExternalMemoryHandleType,
    ) -> Result<File, VulkanError> {
        let info_vk = ash::vk::MemoryGetFdInfoKHR {
            memory: self.handle,
            handle_type: handle_type.into(),
            ..Default::default()
        };

        let fns = self.device.fns();
        let mut output = MaybeUninit::uninit();
        (fns.khr_external_memory_fd.get_memory_fd_khr)(
            self.device.handle(),
            &info_vk,
            output.as_mut_ptr(),
        )
        .result()
        .map_err(VulkanError::from)?;

        #[cfg(unix)]
        {
            use std::os::unix::io::FromRawFd;
            Ok(File::from_raw_fd(output.assume_init()))
        }

        #[cfg(not(unix))]
        {
            let _ = output;
            unreachable!("`khr_external_memory_fd` was somehow enabled on a non-Unix system");
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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
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
            .ok_or_else(|| {
                Box::new(ValidationError {
                    context: "memory_type_index".into(),
                    problem: "is not less than the number of memory types in the device".into(),
                    vuids: &["VUID-vkAllocateMemory-pAllocateInfo-01714"],
                    ..Default::default()
                })
            })?;
        let memory_heap = &memory_properties.memory_heaps[memory_type.heap_index as usize];

        if memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::PROTECTED)
            && !device.enabled_features().protected_memory
        {
            return Err(Box::new(ValidationError {
                context: "memory_type_index".into(),
                problem: "refers to a memory type where `property_flags` contains \
                    `MemoryPropertyFlags::PROTECTED`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "protected_memory",
                )])]),
                vuids: &["VUID-VkMemoryAllocateInfo-memoryTypeIndex-01872"],
            }));
        }

        if memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::DEVICE_COHERENT)
            && !device.enabled_features().device_coherent_memory
        {
            return Err(Box::new(ValidationError {
                context: "memory_type_index".into(),
                problem: "refers to a memory type where `property_flags` contains \
                    `MemoryPropertyFlags::DEVICE_COHERENT`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "device_coherent_memory",
                )])]),
                vuids: &["VUID-vkAllocateMemory-deviceCoherentMemory-02790"],
            }));
        }

        if allocation_size == 0 {
            return Err(Box::new(ValidationError {
                context: "allocation_size".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkMemoryAllocateInfo-pNext-01874"],
                ..Default::default()
            }));
        }

        if memory_heap.size != 0 && allocation_size > memory_heap.size {
            return Err(Box::new(ValidationError {
                context: "allocation_size".into(),
                problem: "is greater than the size of the memory heap".into(),
                vuids: &["VUID-vkAllocateMemory-pAllocateInfo-01713"],
                ..Default::default()
            }));
        }

        if let Some(dedicated_allocation) = dedicated_allocation {
            match dedicated_allocation {
                DedicatedAllocation::Buffer(buffer) => {
                    // VUID-VkMemoryDedicatedAllocateInfo-commonparent
                    assert_eq!(device, buffer.device().as_ref());

                    let required_size = buffer.memory_requirements().layout.size();

                    if allocation_size != required_size {
                        return Err(Box::new(ValidationError {
                            problem: "`allocation_size` does not equal the size required for the \
                                buffer specified in `dedicated_allocation`"
                                .into(),
                            vuids: &["VUID-VkMemoryDedicatedAllocateInfo-buffer-02965"],
                            ..Default::default()
                        }));
                    }
                }
                DedicatedAllocation::Image(image) => {
                    // VUID-VkMemoryDedicatedAllocateInfo-commonparent
                    assert_eq!(device, image.device().as_ref());

                    let required_size = image.memory_requirements()[0].layout.size();

                    if allocation_size != required_size {
                        return Err(Box::new(ValidationError {
                            problem: "`allocation_size` does not equal the size required for the \
                                image specified in `dedicated_allocation`"
                                .into(),
                            vuids: &["VUID-VkMemoryDedicatedAllocateInfo-image-02964"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_memory)
            {
                return Err(Box::new(ValidationError {
                    context: "export_handle_types".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_external_memory")]),
                    ]),
                    ..Default::default()
                }));
            }

            export_handle_types.validate_device(device).map_err(|err| {
                err.add_context("export_handle_types")
                    .set_vuids(&["VUID-VkExportMemoryAllocateInfo-handleTypes-parameter"])
            })?;

            // VUID-VkMemoryAllocateInfo-pNext-00639
            // VUID-VkExportMemoryAllocateInfo-handleTypes-00656
            // Impossible to validate here, instead this is validated in `RawBuffer::bind_memory`
            // and `RawImage::bind_memory`.
        }

        if !flags.is_empty() {
            if !(device.physical_device().api_version() >= Version::V1_1
                || device.enabled_extensions().khr_device_group)
            {
                return Err(Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_device_group")]),
                    ]),
                    ..Default::default()
                }));
            }

            if flags.intersects(MemoryAllocateFlags::DEVICE_ADDRESS) {
                if !((device.api_version() >= Version::V1_2
                    || device.enabled_extensions().khr_buffer_device_address)
                    && device.enabled_features().buffer_device_address)
                {
                    return Err(Box::new(ValidationError {
                        context: "flags".into(),
                        problem: "contains `MemoryAllocateFlags::DEVICE_ADDRESS`".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[
                                Requires::APIVersion(Version::V1_2),
                                Requires::DeviceFeature("buffer_device_address"),
                            ]),
                            RequiresAllOf(&[
                                Requires::DeviceExtension("khr_buffer_device_address"),
                                Requires::DeviceFeature("buffer_device_address"),
                            ]),
                        ]),
                        vuids: &["VUID-VkMemoryAllocateInfo-flags-03331"],
                    }));
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
    /// - If `handle_type` is [`ExternalMemoryHandleType::OpaqueWin32`], it owns a reference to the
    ///   underlying resource and must eventually be closed by the caller.
    /// - If `handle_type` is [`ExternalMemoryHandleType::OpaqueWin32Kmt`], it does not own a
    ///   reference to the underlying resource.
    /// - `handle` must be created by the Vulkan API.
    /// - [`MemoryAllocateInfo::allocation_size`] and [`MemoryAllocateInfo::memory_type_index`]
    ///   must match those of the original memory allocation.
    /// - If the original memory allocation used [`MemoryAllocateInfo::dedicated_allocation`], the
    ///   imported one must also use it, and the associated buffer or image must be defined
    ///   identically to the original.
    Win32 {
        handle_type: ExternalMemoryHandleType,
        handle: ash::vk::HANDLE,
    },
}

impl MemoryImportInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        match self {
            MemoryImportInfo::Fd {
                handle_type,
                file: _,
            } => {
                if !device.enabled_extensions().khr_external_memory_fd {
                    return Err(Box::new(ValidationError {
                        problem: "is `MemoryImportInfo::Fd`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceExtension("khr_external_memory_fd"),
                        ])]),
                        ..Default::default()
                    }));
                }

                handle_type.validate_device(device).map_err(|err| {
                    err.add_context("handle_type")
                        .set_vuids(&["VUID-VkImportMemoryFdInfoKHR-handleType-parameter"])
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
                        return Err(Box::new(ValidationError {
                            context: "handle_type".into(),
                            problem: "is not `ExternalMemoryHandleType::OpaqueFd` or \
                                `ExternalMemoryHandleType::DmaBuf`"
                                .into(),
                            vuids: &["VUID-VkImportMemoryFdInfoKHR-handleType-00669"],
                            ..Default::default()
                        }));
                    }
                }

                // VUID-VkMemoryAllocateInfo-memoryTypeIndex-00648
                // Can't validate, must be ensured by user
            }
            MemoryImportInfo::Win32 {
                handle_type,
                handle: _,
            } => {
                if !device.enabled_extensions().khr_external_memory_win32 {
                    return Err(Box::new(ValidationError {
                        problem: "is `MemoryImportInfo::Win32`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceExtension("khr_external_memory_win32"),
                        ])]),
                        ..Default::default()
                    }));
                }

                handle_type.validate_device(device).map_err(|err| {
                    err.add_context("handle_type")
                        .set_vuids(&["VUID-VkImportMemoryWin32HandleInfoKHR-handleType-parameter"])
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
                        return Err(Box::new(ValidationError {
                            context: "handle_type".into(),
                            problem: "is not `ExternalMemoryHandleType::OpaqueWin32` or \
                                `ExternalMemoryHandleType::OpaqueWin32Kmt`"
                                .into(),
                            vuids: &["VUID-VkImportMemoryWin32HandleInfoKHR-handleType-00660"],
                            ..Default::default()
                        }));
                    }
                }

                // VUID-VkMemoryAllocateInfo-memoryTypeIndex-00645
                // Can't validate, must be ensured by user
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

    /// Flags specifying additional properties of a device memory allocation.
    MemoryAllocateFlags = MemoryAllocateFlags(u32);

    /* TODO: enable
    DEVICE_MASK = DEVICE_MASK,*/

    /// Specifies that the allocated device memory can be bound to a buffer created with the
    /// [`SHADER_DEVICE_ADDRESS`] usage. This requires that the [`buffer_device_address`] feature
    /// is enabled on the device and the [`ext_buffer_device_address`] extension is not enabled on
    /// the device.
    ///
    /// [`SHADER_DEVICE_ADDRESS`]: crate::buffer::BufferUsage::SHADER_DEVICE_ADDRESS
    /// [`buffer_device_address`]: crate::device::DeviceFeatures::buffer_device_address
    /// [`ext_buffer_device_address`]: crate::device::DeviceExtensions::ext_buffer_device_address
    DEVICE_ADDRESS = DEVICE_ADDRESS,

    /* TODO: enable
    DEVICE_ADDRESS_CAPTURE_REPLAY = DEVICE_ADDRESS_CAPTURE_REPLAY,*/
}

/// Parameters of a memory map operation.
#[derive(Debug)]
pub struct MemoryMapInfo {
    /// The offset (in bytes) from the beginning of the `DeviceMemory`, where the mapping starts.
    ///
    /// Must be less than the [`allocation_size`] of the device memory. If the the memory was not
    /// allocated from [host-coherent] memory, then this must be a multiple of the
    /// [`non_coherent_atom_size`] device property.
    ///
    /// The default value is `0`.
    ///
    /// [`allocation_size`]: DeviceMemory::allocation_size
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    pub offset: DeviceSize,

    /// The size (in bytes) of the mapping.
    ///
    /// Must be less than or equal to the [`allocation_size`] of the device memory minus `offset`.
    /// If the the memory was not allocated from [host-coherent] memory, then this must be a
    /// multiple of the [`non_coherent_atom_size`] device property, or be equal to the allocation
    /// size minus `offset`.
    ///
    /// The default value is `0`, which must be overridden.
    ///
    /// [`allocation_size`]: DeviceMemory::allocation_size
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    pub size: DeviceSize,

    /// The address in host memory to map to.
    ///
    /// Requires [`DeviceExtensions::ext_map_memory_placed`] and
    /// [`DeviceFeatures::memory_map_placed`] to be enabled.
    ///
    /// Must align with [`DeviceProperties::min_placed_memory_map_alignment`].
    ///
    /// [`DeviceExtensions::ext_map_memory_placed`]: crate::device::DeviceExtensions::ext_map_memory_placed
    /// [`DeviceFeatures::memory_map_placed`]: crate::device::DeviceFeatures::memory_map_placed
    /// [`DeviceProperties::min_placed_memory_map_alignment`]: crate::device::DeviceProperties::min_placed_memory_map_alignment
    pub placed_address: Option<*mut c_void>,

    pub _ne: crate::NonExhaustive,
}

impl MemoryMapInfo {
    pub(crate) fn validate(&self, memory: &DeviceMemory) -> Result<(), Box<ValidationError>> {
        let &Self {
            offset,
            size,
            placed_address,
            _ne: _,
        } = self;

        if !(offset < memory.allocation_size()) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not less than `self.allocation_size()`".into(),
                vuids: &["VUID-vkMapMemory-offset-00679"],
                ..Default::default()
            }));
        }

        if size == 0 {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is zero".into(),
                vuids: &["VUID-vkMapMemory-size-00680"],
                ..Default::default()
            }));
        }

        if !(size <= memory.allocation_size() - offset) && size != DeviceSize::MAX {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is not less than or equal to `self.allocation_size()` minus `offset` or \
                    equal to VK_WHOLE_SIZE"
                    .into(),
                vuids: &["VUID-vkMapMemory-size-00681"],
                ..Default::default()
            }));
        }

        if let Some(placed_address) = placed_address {
            if !memory.device.enabled_extensions().ext_map_memory_placed {
                return Err(Box::new(ValidationError {
                    context: "placed_address".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_map_memory_placed",
                    )])]),
                    ..Default::default()
                }));
            }

            let features = memory.device.enabled_features();
            if !features.memory_map_placed {
                return Err(Box::new(ValidationError {
                    context: "placed_address".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "memory_map_placed",
                    )])]),
                    vuids: &["VUID-VkMemoryMapInfoKHR-flags-09569"],
                }));
            }

            // SAFETY:
            // min_placed_memory_map_alignment is always provided when the device extension
            // ext_map_memory_placed is available.
            let min_placed_memory_map_alignment = memory
                .device
                .physical_device()
                .properties()
                .min_placed_memory_map_alignment
                .unwrap();

            if !is_aligned(
                placed_address as DeviceSize,
                min_placed_memory_map_alignment,
            ) {
                return Err(Box::new(ValidationError {
                    context: "placed_address".into(),
                    problem: "must be aligned to an integer multiple of \
                        `min_placed_memory_map_alignment` device property"
                        .into(),
                    vuids: &["VUID-VkMemoryMapPlacedInfoEXT-pPlacedAddress-09577"],
                    ..Default::default()
                }));
            }

            if features.memory_map_range_placed {
                if !is_aligned(offset, min_placed_memory_map_alignment) {
                    return Err(Box::new(ValidationError {
                        context: "offset".into(),
                        problem: "must be aligned to an integer multiple of \
                            `min_placed_memory_map_alignment` device property"
                            .into(),
                        vuids: &["VUID-VkMemoryMapInfoKHR-flags-09573"],
                        ..Default::default()
                    }));
                }

                if !is_aligned(size, min_placed_memory_map_alignment) && size != DeviceSize::MAX {
                    return Err(Box::new(ValidationError {
                        context: "size".into(),
                        problem: "must be aligned to an integer multiple of \
                            `min_placed_memory_map_alignment` device property or must be \
                            VK_WHOLE_SIZE"
                            .into(),
                        vuids: &["VUID-VkMemoryMapInfoKHR-flags-09574"],
                        ..Default::default()
                    }));
                }
            } else {
                if offset != 0 {
                    return Err(Box::new(ValidationError {
                        context: "offset".into(),
                        problem: "must be zero".into(),
                        vuids: &["VUID-VkMemoryMapInfoKHR-flags-09571"],
                        ..Default::default()
                    }));
                }

                if size != DeviceSize::MAX {
                    return Err(Box::new(ValidationError {
                        context: "size".into(),
                        problem: "must be VK_WHOLE_SIZE".into(),
                        vuids: &["VUID-VkMemoryMapInfoKHR-flags-09572"],
                        ..Default::default()
                    }));
                }
            }
        }

        let atom_size = memory.atom_size();

        // Not required for merely mapping, but without this check the user can end up with
        // parts of the mapped memory at the start and end that they're not able to
        // invalidate/flush, which is probably unintended.
        //
        // NOTE(Marc): We also rely on this for soundness, because it is easier and more optimal to
        // not have to worry about whether a range of mapped memory is still in bounds of the
        // mapped memory after being aligned to the non-coherent atom size.
        if !memory.is_coherent
            && (!is_aligned(offset, atom_size)
                || (!is_aligned(size, atom_size) && offset + size != memory.allocation_size()))
        {
            return Err(Box::new(ValidationError {
                problem: "`self.memory_type_index()` refers to a memory type whose \
                    `property_flags` does not contain `MemoryPropertyFlags::HOST_COHERENT`, and \
                    `offset` and/or `size` are not aligned to the `non_coherent_atom_size` device \
                    property"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }
}

impl Default for MemoryMapInfo {
    #[inline]
    fn default() -> Self {
        MemoryMapInfo {
            offset: 0,
            size: 0,
            placed_address: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters of a memory unmap operation.
#[derive(Debug)]
pub struct MemoryUnmapInfo {
    pub _ne: crate::NonExhaustive,
}

impl MemoryUnmapInfo {
    pub(crate) fn validate(&self, _memory: &DeviceMemory) -> Result<(), Box<ValidationError>> {
        let &Self { _ne: _ } = self;

        Ok(())
    }
}

impl Default for MemoryUnmapInfo {
    #[inline]
    fn default() -> Self {
        MemoryUnmapInfo {
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Represents the currently host-mapped region of a [`DeviceMemory`] block.
#[derive(Debug)]
pub struct MappingState {
    ptr: NonNull<c_void>,
    range: Range<DeviceSize>,
}

// It is safe to share `ptr` between threads because the user would have to use unsafe code
// themself to get UB in the first place.
unsafe impl Send for MappingState {}
unsafe impl Sync for MappingState {}

impl MappingState {
    /// Returns the pointer to the start of the mapped memory. Meaning that the pointer is already
    /// offset by the [`offset`].
    ///
    /// [`offset`]: Self::offset
    #[inline]
    pub fn ptr(&self) -> NonNull<c_void> {
        self.ptr
    }

    /// Returns the offset given to [`DeviceMemory::map`].
    #[inline]
    pub fn offset(&self) -> DeviceSize {
        self.range.start
    }

    /// Returns the size given to [`DeviceMemory::map`].
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.range.end - self.range.start
    }

    /// Returns a pointer to a slice of the mapped memory. Returns `None` if out of bounds.
    ///
    /// `range` is specified in bytes relative to the start of the memory allocation, and must fall
    /// within the range of the memory mapping given to [`DeviceMemory::map`].
    ///
    /// This function is safe in the sense that the returned pointer is guaranteed to be within
    /// bounds of the mapped memory, however dereferencing the pointer isn't:
    ///
    /// - Normal Rust aliasing rules apply: if you create a mutable reference out of the pointer,
    ///   you must ensure that no other references exist in Rust to any portion of the same memory.
    /// - While a reference created from the pointer exists, there must be no operations pending or
    ///   executing in any queue on the device, that write to any portion of the same memory.
    /// - While a mutable reference created from the pointer exists, there must be no operations
    ///   pending or executing in any queue on the device, that read from any portion of the same
    ///   memory.
    #[inline]
    pub fn slice(&self, range: Range<DeviceSize>) -> Option<NonNull<[u8]>> {
        if self.range.start <= range.start
            && range.start <= range.end
            && range.end <= self.range.end
        {
            // SAFETY: We checked that the range is within the currently mapped range.
            Some(unsafe { self.slice_unchecked(range) })
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// - `range` must be within the currently mapped range.
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn slice_unchecked(&self, range: Range<DeviceSize>) -> NonNull<[u8]> {
        let ptr = self.ptr.as_ptr();

        // SAFETY: The caller must guarantee that `range` is within the currently mapped range,
        // which means that the offset pointer and length must denote a slice that's contained
        // within the allocated (mapped) object.
        let ptr = ptr.add((range.start - self.range.start) as usize);
        let len = (range.end - range.start) as usize;

        let ptr = ptr::slice_from_raw_parts_mut(<*mut c_void>::cast::<u8>(ptr), len);

        // SAFETY: The original pointer was non-null, and the caller must guarantee that `range`
        // is within the currently mapped range, which means that the offset couldn't have wrapped
        // around the address space.
        NonNull::new_unchecked(ptr)
    }
}

/// Represents a range of host-mapped [`DeviceMemory`] to be invalidated or flushed.
///
/// Must be contained within the currently mapped range of the device memory.
#[derive(Debug)]
pub struct MappedMemoryRange {
    /// The offset (in bytes) from the beginning of the allocation, where the range starts.
    ///
    /// Must be a multiple of the [`non_coherent_atom_size`] device property.
    ///
    /// The default value is `0`.
    ///
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    pub offset: DeviceSize,

    /// The size (in bytes) of the range.
    ///
    /// Must be a multiple of the [`non_coherent_atom_size`] device property, or be equal to the
    /// allocation size minus `offset`.
    ///
    /// The default value is `0`.
    ///
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    pub size: DeviceSize,

    pub _ne: crate::NonExhaustive,
}

impl MappedMemoryRange {
    pub(crate) fn validate(&self, memory: &DeviceMemory) -> Result<(), Box<ValidationError>> {
        let &Self {
            offset,
            size,
            _ne: _,
        } = self;

        if let Some(state) = &memory.mapping_state {
            if !(state.range.start <= offset && size <= state.range.end - offset) {
                return Err(Box::new(ValidationError {
                    problem: "is not contained within the mapped range of this device memory"
                        .into(),
                    vuids: &["VUID-VkMappedMemoryRange-size-00685"],
                    ..Default::default()
                }));
            }
        } else {
            return Err(Box::new(ValidationError {
                problem: "this device memory is not currently host-mapped".into(),
                vuids: &["VUID-VkMappedMemoryRange-memory-00684"],
                ..Default::default()
            }));
        }

        if !is_aligned(offset, memory.atom_size()) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not aligned to the `non_coherent_atom_size` device property".into(),
                vuids: &["VUID-VkMappedMemoryRange-offset-00687"],
                ..Default::default()
            }));
        }

        if !(is_aligned(size, memory.atom_size()) || size == memory.allocation_size() - offset) {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is not aligned to the `non_coherent_atom_size` device property nor \
                    equal to `self.allocation_size()` minus `offset`"
                    .into(),
                vuids: &["VUID-VkMappedMemoryRange-size-01390"],
                ..Default::default()
            }));
        }

        Ok(())
    }
}

impl Default for MappedMemoryRange {
    #[inline]
    fn default() -> Self {
        MappedMemoryRange {
            offset: 0,
            size: 0,
            _ne: crate::NonExhaustive(()),
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
/// use vulkano::memory::{
///     DeviceMemory, MappedDeviceMemory, MemoryAllocateInfo, MemoryPropertyFlags,
/// };
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// // The memory type must be mappable.
/// let memory_type_index = device
///     .physical_device()
///     .memory_properties()
///     .memory_types
///     .iter()
///     .position(|t| {
///         t.property_flags
///             .intersects(MemoryPropertyFlags::HOST_VISIBLE)
///     })
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
#[deprecated(
    since = "0.34.0",
    note = "use the methods provided directly on `DeviceMemory` instead"
)]
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

#[allow(deprecated)]
impl MappedDeviceMemory {
    /// Maps a range of memory to be accessed by the CPU.
    ///
    /// `memory` must be allocated from host-visible memory.
    ///
    /// `range` is specified in bytes relative to the start of the memory allocation, and must fall
    /// within the range of the allocation (`0..allocation_size`). If `memory` was not allocated
    /// from host-coherent memory, then the start and end of `range` must be a multiple of the
    /// [`non_coherent_atom_size`](crate::device::DeviceProperties::non_coherent_atom_size) device
    /// property, but `range.end` can also the memory's `allocation_size`.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    #[inline]
    pub fn new(
        memory: DeviceMemory,
        range: Range<DeviceSize>,
    ) -> Result<Self, Validated<VulkanError>> {
        Self::validate_new(&memory, range.clone())?;

        unsafe { Ok(Self::new_unchecked(memory, range)?) }
    }

    fn validate_new(
        memory: &DeviceMemory,
        range: Range<DeviceSize>,
    ) -> Result<(), Box<ValidationError>> {
        if range.is_empty() {
            return Err(Box::new(ValidationError {
                context: "range".into(),
                problem: "is empty".into(),
                vuids: &["VUID-vkMapMemory-size-00680"],
                ..Default::default()
            }));
        }

        let device = memory.device();
        let memory_type = &device.physical_device().memory_properties().memory_types
            [memory.memory_type_index() as usize];

        if !memory_type
            .property_flags
            .intersects(MemoryPropertyFlags::HOST_VISIBLE)
        {
            return Err(Box::new(ValidationError {
                context: "memory".into(),
                problem: "has a memory type whose `property_flags` does not contain \
                    `MemoryPropertyFlags::HOST_VISIBLE`"
                    .into(),
                vuids: &["VUID-vkMapMemory-memory-00682"],
                ..Default::default()
            }));
        }

        if memory.mapping_state().is_some() {
            return Err(Box::new(ValidationError {
                context: "memory".into(),
                problem: "is already host-mapped".into(),
                vuids: &["VUID-vkMapMemory-memory-00678"],
                ..Default::default()
            }));
        }

        if range.end > memory.allocation_size {
            return Err(Box::new(ValidationError {
                problem: "`range.end` is greater than `memory.allocation_size()`".into(),
                vuids: &[
                    "VUID-vkMapMemory-offset-00679",
                    "VUID-vkMapMemory-size-00681",
                ],
                ..Default::default()
            }));
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
            return Err(Box::new(ValidationError {
                problem: "`memory` has a memory type whose `property_flags` does not contain \
                    `MemoryPropertyFlags::HOST_COHERENT`, and `range.start` and/or `range.end` \
                    are not aligned to the `non_coherent_atom_size` device property"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        memory: DeviceMemory,
        range: Range<DeviceSize>,
    ) -> Result<Self, VulkanError> {
        // Sanity check: this would lead to UB when calculating pointer offsets.
        assert!(range.end - range.start <= isize::MAX.try_into().unwrap());

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
            .map_err(VulkanError::from)?;
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
    /// [`non_coherent_atom_size`](crate::device::DeviceProperties::non_coherent_atom_size) device
    /// property, but `range.end` can also equal the memory's `allocation_size`.
    ///
    /// # Safety
    ///
    /// - If there are memory writes by the GPU that have not been propagated into the CPU cache,
    ///   then there must not be any references in Rust code to the specified `range` of the
    ///   memory.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    #[inline]
    pub unsafe fn invalidate_range(
        &self,
        range: Range<DeviceSize>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_range(range.clone())?;

        Ok(self.invalidate_range_unchecked(range)?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn invalidate_range_unchecked(
        &self,
        range: Range<DeviceSize>,
    ) -> Result<(), VulkanError> {
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
    /// [`non_coherent_atom_size`](crate::device::DeviceProperties::non_coherent_atom_size) device
    /// property, but `range.end` can also equal the memory's `allocation_size`.
    ///
    /// # Safety
    ///
    /// - There must be no operations pending or executing in a GPU queue, that access the
    ///   specified `range` of the memory.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    #[inline]
    pub unsafe fn flush_range(
        &self,
        range: Range<DeviceSize>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_range(range.clone())?;

        Ok(self.flush_range_unchecked(range)?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn flush_range_unchecked(
        &self,
        range: Range<DeviceSize>,
    ) -> Result<(), VulkanError> {
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
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Returns a reference to bytes in the mapped memory.
    ///
    /// `range` is specified in bytes relative to the start of the memory allocation, and must fall
    /// within the range of the memory mapping given to `map`. If the memory was not allocated
    /// from host-coherent memory, then the start and end of `range` must be a multiple of the
    /// [`non_coherent_atom_size`](crate::device::DeviceProperties::non_coherent_atom_size) device
    /// property, but `range.end` can also equal the memory's `allocation_size`.
    ///
    /// # Safety
    ///
    /// - While the returned reference exists, there must not be any mutable references in Rust
    ///   code to the same memory.
    /// - While the returned reference exists, there must be no operations pending or executing in
    ///   a GPU queue, that write to the same memory.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    #[inline]
    pub unsafe fn read(&self, range: Range<DeviceSize>) -> Result<&[u8], Box<ValidationError>> {
        self.validate_range(range.clone())?;

        Ok(self.read_unchecked(range))
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn read_unchecked(&self, range: Range<DeviceSize>) -> &[u8] {
        slice::from_raw_parts(
            self.pointer
                .add((range.start - self.range.start).try_into().unwrap())
                .cast(),
            (range.end - range.start) as usize,
        )
    }

    /// Returns a mutable reference to bytes in the mapped memory.
    ///
    /// `range` is specified in bytes relative to the start of the memory allocation, and must fall
    /// within the range of the memory mapping given to `map`. If the memory was not allocated
    /// from host-coherent memory, then the start and end of `range` must be a multiple of the
    /// [`non_coherent_atom_size`](crate::device::DeviceProperties::non_coherent_atom_size) device
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
    pub unsafe fn write(
        &self,
        range: Range<DeviceSize>,
    ) -> Result<&mut [u8], Box<ValidationError>> {
        self.validate_range(range.clone())?;

        Ok(self.write_unchecked(range))
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn write_unchecked(&self, range: Range<DeviceSize>) -> &mut [u8] {
        slice::from_raw_parts_mut(
            self.pointer
                .add((range.start - self.range.start).try_into().unwrap())
                .cast::<u8>(),
            (range.end - range.start).try_into().unwrap(),
        )
    }

    #[inline]
    fn validate_range(&self, range: Range<DeviceSize>) -> Result<(), Box<ValidationError>> {
        // VUID-VkMappedMemoryRange-memory-00684
        // Guaranteed because `self` owns the memory and it's mapped during our lifetime.

        if range.is_empty() {
            return Err(Box::new(ValidationError {
                context: "range".into(),
                problem: "is empty".into(),
                ..Default::default()
            }));
        }

        if range.start < self.range.start || range.end > self.range.end {
            return Err(Box::new(ValidationError {
                context: "range".into(),
                problem: "is not within the mapped range of this mapped device memory".into(),
                vuids: &["VUID-VkMappedMemoryRange-size-00685"],
                ..Default::default()
            }));
        }

        if !self.is_coherent {
            if !is_aligned(range.start, self.atom_size)
                || (!is_aligned(range.end, self.atom_size)
                    && range.end != self.memory.allocation_size)
            {
                return Err(Box::new(ValidationError {
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
                }));
            }
        }

        Ok(())
    }
}

#[allow(deprecated)]
impl AsRef<DeviceMemory> for MappedDeviceMemory {
    #[inline]
    fn as_ref(&self) -> &DeviceMemory {
        &self.memory
    }
}

#[allow(deprecated)]
impl AsMut<DeviceMemory> for MappedDeviceMemory {
    #[inline]
    fn as_mut(&mut self) -> &mut DeviceMemory {
        &mut self.memory
    }
}

#[allow(deprecated)]
unsafe impl DeviceOwned for MappedDeviceMemory {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.memory.device()
    }
}

#[allow(deprecated)]
unsafe impl Send for MappedDeviceMemory {}
#[allow(deprecated)]
unsafe impl Sync for MappedDeviceMemory {}

#[cfg(test)]
mod tests {
    use super::MemoryAllocateInfo;
    use crate::memory::{DeviceMemory, MemoryMapInfo, MemoryPropertyFlags};
    use ash::vk::DeviceSize;
    use std::ptr;

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

    #[test]
    #[cfg(unix)]
    fn map_placed() {
        let (device, _) = gfx_dev_and_queue!(memory_map_placed; ext_map_memory_placed);

        let memory_type_index = {
            let physical_device = device.physical_device();
            let memory_properties = physical_device.memory_properties();
            let (idx, _) = memory_properties
                .memory_types
                .iter()
                .enumerate()
                .find(|(_idx, it)| {
                    it.property_flags.contains(
                        MemoryPropertyFlags::HOST_COHERENT
                            | MemoryPropertyFlags::HOST_VISIBLE
                            | MemoryPropertyFlags::DEVICE_LOCAL,
                    )
                })
                .unwrap();

            idx as u32
        };

        let mut memory = DeviceMemory::allocate(
            device.clone(),
            MemoryAllocateInfo {
                allocation_size: 16 * 1024,
                memory_type_index,
                ..Default::default()
            },
        )
        .unwrap();

        let address = unsafe {
            let address = libc::mmap(
                ptr::null_mut(),
                16 * 1024,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if address as i64 == -1 {
                panic!("failed to map memory")
            }

            address
        };

        memory
            .map(MemoryMapInfo {
                offset: 0,
                size: DeviceSize::MAX,
                placed_address: Some(address),
                ..Default::default()
            })
            .unwrap();
    }
}
