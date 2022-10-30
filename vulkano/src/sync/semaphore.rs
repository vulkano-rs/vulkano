// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    device::{Device, DeviceOwned, Queue},
    macros::{vulkan_bitflags, vulkan_enum},
    OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
#[cfg(unix)]
use std::fs::File;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::MaybeUninit,
    num::NonZeroU64,
    ptr,
    sync::{Arc, Weak},
};

/// Used to provide synchronization between command buffers during their execution.
///
/// It is similar to a fence, except that it is purely on the GPU side. The CPU can't query a
/// semaphore's status or wait for it to be signaled.
#[derive(Debug)]
pub struct Semaphore {
    handle: ash::vk::Semaphore,
    device: Arc<Device>,
    id: NonZeroU64,
    must_put_in_pool: bool,

    export_handle_types: ExternalSemaphoreHandleTypes,

    state: Mutex<SemaphoreState>,
}

impl Semaphore {
    /// Creates a new `Semaphore`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        create_info: SemaphoreCreateInfo,
    ) -> Result<Semaphore, SemaphoreError> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        create_info: &SemaphoreCreateInfo,
    ) -> Result<(), SemaphoreError> {
        let &SemaphoreCreateInfo {
            export_handle_types,
            _ne: _,
        } = create_info;

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_semaphore)
            {
                return Err(SemaphoreError::RequirementNotMet {
                    required_for: "`create_info.export_handle_types` is not empty",
                    requires_one_of: RequiresOneOf {
                        api_version: Some(Version::V1_1),
                        device_extensions: &["khr_external_semaphore"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkExportSemaphoreCreateInfo-handleTypes-parameter
            export_handle_types.validate_device(device)?;

            // VUID-VkExportSemaphoreCreateInfo-handleTypes-01124
            for handle_type in export_handle_types.into_iter() {
                let external_semaphore_properties = unsafe {
                    device
                        .physical_device()
                        .external_semaphore_properties_unchecked(
                            ExternalSemaphoreInfo::handle_type(handle_type),
                        )
                };

                if !external_semaphore_properties.exportable {
                    return Err(SemaphoreError::HandleTypeNotExportable { handle_type });
                }

                if !external_semaphore_properties
                    .compatible_handle_types
                    .contains(&export_handle_types)
                {
                    return Err(SemaphoreError::ExportHandleTypesNotCompatible);
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: SemaphoreCreateInfo,
    ) -> Result<Semaphore, VulkanError> {
        let SemaphoreCreateInfo {
            export_handle_types,
            _ne: _,
        } = create_info;

        let mut create_info_vk = ash::vk::SemaphoreCreateInfo {
            flags: ash::vk::SemaphoreCreateFlags::empty(),
            ..Default::default()
        };
        let mut export_semaphore_create_info_vk = None;

        if !export_handle_types.is_empty() {
            let _ = export_semaphore_create_info_vk.insert(ash::vk::ExportSemaphoreCreateInfo {
                handle_types: export_handle_types.into(),
                ..Default::default()
            });
        };

        if let Some(info) = export_semaphore_create_info_vk.as_mut() {
            info.p_next = create_info_vk.p_next;
            create_info_vk.p_next = info as *const _ as *const _;
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_semaphore)(
                device.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Semaphore {
            handle,
            device,
            id: Self::next_id(),
            must_put_in_pool: false,
            export_handle_types,
            state: Mutex::new(Default::default()),
        })
    }

    /// Takes a semaphore from the vulkano-provided semaphore pool.
    /// If the pool is empty, a new semaphore will be allocated.
    /// Upon `drop`, the semaphore is put back into the pool.
    ///
    /// For most applications, using the pool should be preferred,
    /// in order to avoid creating new semaphores every frame.
    #[inline]
    pub fn from_pool(device: Arc<Device>) -> Result<Semaphore, SemaphoreError> {
        let handle = device.semaphore_pool().lock().pop();
        let semaphore = match handle {
            Some(handle) => Semaphore {
                handle,
                device,
                id: Self::next_id(),
                must_put_in_pool: true,
                export_handle_types: ExternalSemaphoreHandleTypes::empty(),
                state: Mutex::new(Default::default()),
            },
            None => {
                // Pool is empty, alloc new semaphore
                let mut semaphore = Semaphore::new(device, Default::default())?;
                semaphore.must_put_in_pool = true;
                semaphore
            }
        };

        Ok(semaphore)
    }

    /// Creates a new `Semaphore` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Semaphore,
        create_info: SemaphoreCreateInfo,
    ) -> Semaphore {
        let SemaphoreCreateInfo {
            export_handle_types,
            _ne: _,
        } = create_info;

        Semaphore {
            handle,
            device,
            id: Self::next_id(),
            must_put_in_pool: false,
            export_handle_types,
            state: Mutex::new(Default::default()),
        }
    }

    /// Exports the semaphore into a POSIX file descriptor. The caller owns the returned `File`.
    #[cfg(unix)]
    #[inline]
    pub fn export_fd(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<File, SemaphoreError> {
        let mut state = self.state.lock();
        self.validate_export_fd(handle_type, &state)?;

        unsafe { Ok(self.export_fd_unchecked_locked(handle_type, &mut state)?) }
    }

    #[cfg(unix)]
    fn validate_export_fd(
        &self,
        handle_type: ExternalSemaphoreHandleType,
        state: &SemaphoreState,
    ) -> Result<(), SemaphoreError> {
        if !self.device.enabled_extensions().khr_external_semaphore_fd {
            return Err(SemaphoreError::RequirementNotMet {
                required_for: "`export_fd`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_external_semaphore_fd"],
                    ..Default::default()
                },
            });
        }

        // VUID-VkSemaphoreGetFdInfoKHR-handleType-parameter
        handle_type.validate_device(&self.device)?;

        // VUID-VkSemaphoreGetFdInfoKHR-handleType-01132
        if !self.export_handle_types.intersects(&handle_type.into()) {
            return Err(SemaphoreError::HandleTypeNotEnabled);
        }

        // VUID-VkSemaphoreGetFdInfoKHR-semaphore-01133
        if let Some(imported_handle_type) = state.current_import {
            match imported_handle_type {
                ImportType::SwapchainAcquire => {
                    return Err(SemaphoreError::ImportedForSwapchainAcquire)
                }
                ImportType::ExternalSemaphore(imported_handle_type) => {
                    let external_semaphore_properties = unsafe {
                        self.device
                            .physical_device()
                            .external_semaphore_properties_unchecked(
                                ExternalSemaphoreInfo::handle_type(handle_type),
                            )
                    };

                    if !external_semaphore_properties
                        .export_from_imported_handle_types
                        .intersects(&imported_handle_type.into())
                    {
                        return Err(SemaphoreError::ExportFromImportedNotSupported {
                            imported_handle_type,
                        });
                    }
                }
            }
        }

        if handle_type.has_copy_transference() {
            // VUID-VkSemaphoreGetFdInfoKHR-handleType-01134
            if state.is_wait_pending() {
                return Err(SemaphoreError::QueueIsWaiting);
            }

            // VUID-VkSemaphoreGetFdInfoKHR-handleType-01135
            // VUID-VkSemaphoreGetFdInfoKHR-handleType-03254
            if !(state.is_signaled().unwrap_or(false) || state.is_signal_pending()) {
                return Err(SemaphoreError::HandleTypeCopyNotSignaled);
            }
        }

        // VUID-VkSemaphoreGetFdInfoKHR-handleType-01136
        if !matches!(
            handle_type,
            ExternalSemaphoreHandleType::OpaqueFd | ExternalSemaphoreHandleType::SyncFd
        ) {
            return Err(SemaphoreError::HandleTypeNotFd);
        }

        Ok(())
    }

    #[cfg(unix)]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn export_fd_unchecked(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<File, VulkanError> {
        let mut state = self.state.lock();
        self.export_fd_unchecked_locked(handle_type, &mut state)
    }

    #[cfg(unix)]
    unsafe fn export_fd_unchecked_locked(
        &self,
        handle_type: ExternalSemaphoreHandleType,
        state: &mut SemaphoreState,
    ) -> Result<File, VulkanError> {
        use std::os::unix::io::FromRawFd;

        let info = ash::vk::SemaphoreGetFdInfoKHR {
            semaphore: self.handle,
            handle_type: handle_type.into(),
            ..Default::default()
        };

        let mut output = MaybeUninit::uninit();
        let fns = self.device.fns();
        (fns.khr_external_semaphore_fd.get_semaphore_fd_khr)(
            self.device.handle(),
            &info,
            output.as_mut_ptr(),
        )
        .result()
        .map_err(VulkanError::from)?;

        state.export(handle_type);

        Ok(File::from_raw_fd(output.assume_init()))
    }

    /// Exports the semaphore into a Win32 handle.
    ///
    /// The [`khr_external_semaphore_win32`](crate::device::DeviceExtensions::khr_external_semaphore_win32)
    /// extension must be enabled on the device.
    #[cfg(windows)]
    #[inline]
    pub fn export_win32_handle(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<*mut std::ffi::c_void, SemaphoreError> {
        let mut state = self.state.lock();
        self.validate_export_win32_handle(handle_type, &state)?;

        unsafe { Ok(self.export_win32_handle_unchecked_locked(handle_type, &mut state)?) }
    }

    #[cfg(windows)]
    fn validate_export_win32_handle(
        &self,
        handle_type: ExternalSemaphoreHandleType,
        state: &SemaphoreState,
    ) -> Result<(), SemaphoreError> {
        if !self
            .device
            .enabled_extensions()
            .khr_external_semaphore_win32
        {
            return Err(SemaphoreError::RequirementNotMet {
                required_for: "`export_win32_handle`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_external_semaphore_win32"],
                    ..Default::default()
                },
            });
        }

        // VUID-VkSemaphoreGetWin32HandleInfoKHR-handleType-parameter
        handle_type.validate_device(&self.device)?;

        // VUID-VkSemaphoreGetWin32HandleInfoKHR-handleType-01126
        if !self.export_handle_types.intersects(&handle_type.into()) {
            return Err(SemaphoreError::HandleTypeNotEnabled);
        }

        // VUID-VkSemaphoreGetWin32HandleInfoKHR-handleType-01127
        if matches!(
            handle_type,
            ExternalSemaphoreHandleType::OpaqueWin32 | ExternalSemaphoreHandleType::D3D12Fence
        ) && state.is_exported(handle_type)
        {
            return Err(SemaphoreError::AlreadyExported);
        }

        // VUID-VkSemaphoreGetWin32HandleInfoKHR-semaphore-01128
        if let Some(imported_handle_type) = state.current_import {
            match imported_handle_type {
                ImportType::SwapchainAcquire => {
                    return Err(SemaphoreError::ImportedForSwapchainAcquire)
                }
                ImportType::ExternalSemaphore(imported_handle_type) => {
                    let external_semaphore_properties = unsafe {
                        self.device
                            .physical_device()
                            .external_semaphore_properties_unchecked(
                                ExternalSemaphoreInfo::handle_type(handle_type),
                            )
                    };

                    if !external_semaphore_properties
                        .export_from_imported_handle_types
                        .intersects(&imported_handle_type.into())
                    {
                        return Err(SemaphoreError::ExportFromImportedNotSupported {
                            imported_handle_type,
                        });
                    }
                }
            }
        }

        if handle_type.has_copy_transference() {
            // VUID-VkSemaphoreGetWin32HandleInfoKHR-handleType-01129
            if state.is_wait_pending() {
                return Err(SemaphoreError::QueueIsWaiting);
            }

            // VUID-VkSemaphoreGetWin32HandleInfoKHR-handleType-01130
            if !(state.is_signaled().unwrap_or(false) || state.is_signal_pending()) {
                return Err(SemaphoreError::HandleTypeCopyNotSignaled);
            }
        }

        // VUID-VkSemaphoreGetWin32HandleInfoKHR-handleType-01131
        if !matches!(
            handle_type,
            ExternalSemaphoreHandleType::OpaqueWin32
                | ExternalSemaphoreHandleType::OpaqueWin32Kmt
                | ExternalSemaphoreHandleType::D3D12Fence
        ) {
            return Err(SemaphoreError::HandleTypeNotWin32);
        }

        Ok(())
    }

    #[cfg(windows)]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn export_win32_handle_unchecked(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<*mut std::ffi::c_void, VulkanError> {
        let mut state = self.state.lock();
        self.export_win32_handle_unchecked_locked(handle_type, &mut state)
    }

    #[cfg(windows)]
    unsafe fn export_win32_handle_unchecked_locked(
        &self,
        handle_type: ExternalSemaphoreHandleType,
        state: &mut SemaphoreState,
    ) -> Result<*mut std::ffi::c_void, VulkanError> {
        let info_vk = ash::vk::SemaphoreGetWin32HandleInfoKHR {
            semaphore: self.handle,
            handle_type: handle_type.into(),
            ..Default::default()
        };

        let mut output = MaybeUninit::uninit();
        let fns = self.device.fns();
        (fns.khr_external_semaphore_win32
            .get_semaphore_win32_handle_khr)(
            self.device.handle(), &info_vk, output.as_mut_ptr()
        )
        .result()
        .map_err(VulkanError::from)?;

        state.export(handle_type);

        Ok(output.assume_init())
    }

    /// Exports the semaphore into a Zircon event handle.
    #[cfg(target_os = "fuchsia")]
    #[inline]
    pub fn export_zircon_handle(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<ash::vk::zx_handle_t, SemaphoreError> {
        let mut state = self.state.lock();
        self.validate_export_zircon_handle(handle_type, &state)?;

        unsafe { Ok(self.export_zircon_handle_unchecked_locked(handle_type, &mut state)?) }
    }

    #[cfg(target_os = "fuchsia")]
    fn validate_export_zircon_handle(
        &self,
        handle_type: ExternalSemaphoreHandleType,
        state: &SemaphoreState,
    ) -> Result<(), SemaphoreError> {
        if !self.device.enabled_extensions().fuchsia_external_semaphore {
            return Err(SemaphoreError::RequirementNotMet {
                required_for: "`export_zircon_handle`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["fuchsia_external_semaphore"],
                    ..Default::default()
                },
            });
        }

        // VUID-VkSemaphoreGetZirconHandleInfoFUCHSIA-handleType-parameter
        handle_type.validate_device(&self.device)?;

        // VUID-VkSemaphoreGetZirconHandleInfoFUCHSIA-handleType-04758
        if !self.export_handle_types.intersects(&handle_type.into()) {
            return Err(SemaphoreError::HandleTypeNotEnabled);
        }

        // VUID-VkSemaphoreGetZirconHandleInfoFUCHSIA-semaphore-04759
        if let Some(imported_handle_type) = state.current_import {
            match imported_handle_type {
                ImportType::SwapchainAcquire => {
                    return Err(SemaphoreError::ImportedForSwapchainAcquire)
                }
                ImportType::ExternalSemaphore(imported_handle_type) => {
                    let external_semaphore_properties = unsafe {
                        self.device
                            .physical_device()
                            .external_semaphore_properties_unchecked(
                                ExternalSemaphoreInfo::handle_type(handle_type),
                            )
                    };

                    if !external_semaphore_properties
                        .export_from_imported_handle_types
                        .intersects(&imported_handle_type.into())
                    {
                        return Err(SemaphoreError::ExportFromImportedNotSupported {
                            imported_handle_type,
                        });
                    }
                }
            }
        }

        if handle_type.has_copy_transference() {
            // VUID-VkSemaphoreGetZirconHandleInfoFUCHSIA-handleType-04760
            if state.is_wait_pending() {
                return Err(SemaphoreError::QueueIsWaiting);
            }

            // VUID-VkSemaphoreGetZirconHandleInfoFUCHSIA-handleType-04761
            if !(state.is_signaled().unwrap_or(false) || state.is_signal_pending()) {
                return Err(SemaphoreError::HandleTypeCopyNotSignaled);
            }
        }

        // VUID-VkSemaphoreGetZirconHandleInfoFUCHSIA-handleType-04762
        if !matches!(handle_type, ExternalSemaphoreHandleType::ZirconEvent) {
            return Err(SemaphoreError::HandleTypeNotZircon);
        }

        Ok(())
    }

    #[cfg(target_os = "fuchsia")]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn export_zircon_handle_unchecked(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<ash::vk::zx_handle_t, VulkanError> {
        let mut state = self.state.lock();
        self.export_zircon_handle_unchecked_locked(handle_type, &mut state)
    }

    #[cfg(target_os = "fuchsia")]
    unsafe fn export_zircon_handle_unchecked_locked(
        &self,
        handle_type: ExternalSemaphoreHandleType,
        state: &mut SemaphoreState,
    ) -> Result<ash::vk::zx_handle_t, VulkanError> {
        let info = ash::vk::SemaphoreGetZirconHandleInfoFUCHSIA {
            semaphore: self.handle,
            handle_type: handle_type.into(),
            ..Default::default()
        };

        let mut output = MaybeUninit::uninit();
        let fns = self.device.fns();
        (fns.fuchsia_external_semaphore
            .get_semaphore_zircon_handle_fuchsia)(
            self.device.handle(), &info, output.as_mut_ptr()
        )
        .result()
        .map_err(VulkanError::from)?;

        state.export(handle_type);

        Ok(output.assume_init())
    }

    /// Imports a semaphore from a POSIX file descriptor.
    ///
    /// The [`khr_external_semaphore_fd`](crate::device::DeviceExtensions::khr_external_semaphore_fd)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - If in `import_semaphore_fd_info`, `handle_type` is `ExternalHandleType::OpaqueFd`,
    ///   then `file` must represent a binary semaphore that was exported from Vulkan or a
    ///   compatible API, with a driver and device UUID equal to those of the device that owns
    ///   `self`.
    #[cfg(unix)]
    #[inline]
    pub unsafe fn import_fd(
        &self,
        import_semaphore_fd_info: ImportSemaphoreFdInfo,
    ) -> Result<(), SemaphoreError> {
        let mut state = self.state.lock();
        self.validate_import_fd(&import_semaphore_fd_info, &state)?;

        Ok(self.import_fd_unchecked_locked(import_semaphore_fd_info, &mut state)?)
    }

    #[cfg(unix)]
    fn validate_import_fd(
        &self,
        import_semaphore_fd_info: &ImportSemaphoreFdInfo,
        state: &SemaphoreState,
    ) -> Result<(), SemaphoreError> {
        if !self.device.enabled_extensions().khr_external_semaphore_fd {
            return Err(SemaphoreError::RequirementNotMet {
                required_for: "`import_fd`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_external_semaphore_fd"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkImportSemaphoreFdKHR-semaphore-01142
        if state.is_in_queue() {
            return Err(SemaphoreError::InQueue);
        }

        let &ImportSemaphoreFdInfo {
            flags,
            handle_type,
            file: _,
            _ne: _,
        } = import_semaphore_fd_info;

        // VUID-VkImportSemaphoreFdInfoKHR-flags-parameter
        flags.validate_device(&self.device)?;

        // VUID-VkImportSemaphoreFdInfoKHR-handleType-parameter
        handle_type.validate_device(&self.device)?;

        // VUID-VkImportSemaphoreFdInfoKHR-handleType-01143
        if !matches!(
            handle_type,
            ExternalSemaphoreHandleType::OpaqueFd | ExternalSemaphoreHandleType::SyncFd
        ) {
            return Err(SemaphoreError::HandleTypeNotFd);
        }

        // VUID-VkImportSemaphoreFdInfoKHR-fd-01544
        // VUID-VkImportSemaphoreFdInfoKHR-handleType-03263
        // Can't validate, therefore unsafe

        // VUID-VkImportSemaphoreFdInfoKHR-handleType-07307
        if handle_type.has_copy_transference() && !flags.temporary {
            return Err(SemaphoreError::HandletypeCopyNotTemporary);
        }

        Ok(())
    }

    #[cfg(unix)]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn import_fd_unchecked(
        &self,
        import_semaphore_fd_info: ImportSemaphoreFdInfo,
    ) -> Result<(), VulkanError> {
        let mut state = self.state.lock();
        self.import_fd_unchecked_locked(import_semaphore_fd_info, &mut state)
    }

    #[cfg(unix)]
    unsafe fn import_fd_unchecked_locked(
        &self,
        import_semaphore_fd_info: ImportSemaphoreFdInfo,
        state: &mut SemaphoreState,
    ) -> Result<(), VulkanError> {
        use std::os::unix::io::IntoRawFd;

        let ImportSemaphoreFdInfo {
            flags,
            handle_type,
            file,
            _ne: _,
        } = import_semaphore_fd_info;

        let info_vk = ash::vk::ImportSemaphoreFdInfoKHR {
            semaphore: self.handle,
            flags: flags.into(),
            handle_type: handle_type.into(),
            fd: file.map_or(-1, |file| file.into_raw_fd()),
            ..Default::default()
        };

        let fns = self.device.fns();
        (fns.khr_external_semaphore_fd.import_semaphore_fd_khr)(self.device.handle(), &info_vk)
            .result()
            .map_err(VulkanError::from)?;

        state.import(handle_type, flags.temporary);

        Ok(())
    }

    /// Imports a semaphore from a Win32 handle.
    ///
    /// The [`khr_external_semaphore_win32`](crate::device::DeviceExtensions::khr_external_semaphore_win32)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - In `import_semaphore_win32_handle_info`, `handle` must represent a binary semaphore that
    ///   was exported from Vulkan or a compatible API, with a driver and device UUID equal to
    ///   those of the device that owns `self`.
    #[cfg(windows)]
    #[inline]
    pub unsafe fn import_win32_handle(
        &self,
        import_semaphore_win32_handle_info: ImportSemaphoreWin32HandleInfo,
    ) -> Result<(), SemaphoreError> {
        let mut state = self.state.lock();
        self.validate_import_win32_handle(&import_semaphore_win32_handle_info, &state)?;

        Ok(self
            .import_win32_handle_unchecked_locked(import_semaphore_win32_handle_info, &mut state)?)
    }

    #[cfg(windows)]
    fn validate_import_win32_handle(
        &self,
        import_semaphore_win32_handle_info: &ImportSemaphoreWin32HandleInfo,
        state: &SemaphoreState,
    ) -> Result<(), SemaphoreError> {
        if !self
            .device
            .enabled_extensions()
            .khr_external_semaphore_win32
        {
            return Err(SemaphoreError::RequirementNotMet {
                required_for: "`import_win32_handle`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_external_semaphore_win32"],
                    ..Default::default()
                },
            });
        }

        // VUID?
        if state.is_in_queue() {
            return Err(SemaphoreError::InQueue);
        }

        let &ImportSemaphoreWin32HandleInfo {
            flags,
            handle_type,
            handle: _,
            _ne: _,
        } = import_semaphore_win32_handle_info;

        // VUID-VkImportSemaphoreWin32HandleInfoKHR-flags-parameter
        flags.validate_device(&self.device)?;

        // VUID-VkImportSemaphoreWin32HandleInfoKHR-handleType-01140
        handle_type.validate_device(&self.device)?;

        // VUID-VkImportSemaphoreWin32HandleInfoKHR-handleType-01140
        if !matches!(
            handle_type,
            ExternalSemaphoreHandleType::OpaqueWin32
                | ExternalSemaphoreHandleType::OpaqueWin32Kmt
                | ExternalSemaphoreHandleType::D3D12Fence
        ) {
            return Err(SemaphoreError::HandleTypeNotWin32);
        }

        // VUID-VkImportSemaphoreWin32HandleInfoKHR-handle-01542
        // Can't validate, therefore unsafe

        // VUID?
        if handle_type.has_copy_transference() && !flags.temporary {
            return Err(SemaphoreError::HandletypeCopyNotTemporary);
        }

        Ok(())
    }

    #[cfg(windows)]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn import_win32_handle_unchecked(
        &self,
        import_semaphore_win32_handle_info: ImportSemaphoreWin32HandleInfo,
    ) -> Result<(), VulkanError> {
        let mut state = self.state.lock();
        self.import_win32_handle_unchecked_locked(import_semaphore_win32_handle_info, &mut state)
    }

    #[cfg(windows)]
    unsafe fn import_win32_handle_unchecked_locked(
        &self,
        import_semaphore_win32_handle_info: ImportSemaphoreWin32HandleInfo,
        state: &mut SemaphoreState,
    ) -> Result<(), VulkanError> {
        let ImportSemaphoreWin32HandleInfo {
            flags,
            handle_type,
            handle,
            _ne: _,
        } = import_semaphore_win32_handle_info;

        let info_vk = ash::vk::ImportSemaphoreWin32HandleInfoKHR {
            semaphore: self.handle,
            flags: flags.into(),
            handle_type: handle_type.into(),
            handle,
            name: ptr::null(), // TODO: support?
            ..Default::default()
        };

        let fns = self.device.fns();
        (fns.khr_external_semaphore_win32
            .import_semaphore_win32_handle_khr)(self.device.handle(), &info_vk)
        .result()
        .map_err(VulkanError::from)?;

        state.import(handle_type, flags.temporary);

        Ok(())
    }

    /// Imports a semaphore from a Zircon event handle.
    ///
    /// The [`fuchsia_external_semaphore`](crate::device::DeviceExtensions::fuchsia_external_semaphore)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - In `import_semaphore_zircon_handle_info`, `zircon_handle` must have `ZX_RIGHTS_BASIC` and
    ///   `ZX_RIGHTS_SIGNAL`.
    #[cfg(target_os = "fuchsia")]
    #[inline]
    pub unsafe fn import_zircon_handle(
        &self,
        import_semaphore_zircon_handle_info: ImportSemaphoreZirconHandleInfo,
    ) -> Result<(), SemaphoreError> {
        let mut state = self.state.lock();
        self.validate_import_zircon_handle(&import_semaphore_zircon_handle_info, &state)?;

        Ok(self.import_zircon_handle_unchecked_locked(
            import_semaphore_zircon_handle_info,
            &mut state,
        )?)
    }

    #[cfg(target_os = "fuchsia")]
    fn validate_import_zircon_handle(
        &self,
        import_semaphore_zircon_handle_info: &ImportSemaphoreZirconHandleInfo,
        state: &SemaphoreState,
    ) -> Result<(), SemaphoreError> {
        if !self.device.enabled_extensions().fuchsia_external_semaphore {
            return Err(SemaphoreError::RequirementNotMet {
                required_for: "`import_zircon_handle`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["fuchsia_external_semaphore"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkImportSemaphoreZirconHandleFUCHSIA-semaphore-04764
        if state.is_in_queue() {
            return Err(SemaphoreError::InQueue);
        }

        let &ImportSemaphoreZirconHandleInfo {
            flags,
            handle_type,
            zircon_handle: _,
            _ne: _,
        } = import_semaphore_zircon_handle_info;

        // VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-flags-parameter
        flags.validate_device(&self.device)?;

        // VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-handleType-parameter
        handle_type.validate_device(&self.device)?;

        // VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-handleType-04765
        if !matches!(handle_type, ExternalSemaphoreHandleType::ZirconEvent) {
            return Err(SemaphoreError::HandleTypeNotFd);
        }

        // VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-zirconHandle-04766
        // VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-zirconHandle-04767
        // Can't validate, therefore unsafe

        if handle_type.has_copy_transference() && !flags.temporary {
            return Err(SemaphoreError::HandletypeCopyNotTemporary);
        }

        Ok(())
    }

    #[cfg(target_os = "fuchsia")]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn import_zircon_handle_unchecked(
        &self,
        import_semaphore_zircon_handle_info: ImportSemaphoreZirconHandleInfo,
    ) -> Result<(), VulkanError> {
        let mut state = self.state.lock();
        self.import_zircon_handle_unchecked_locked(import_semaphore_zircon_handle_info, &mut state)
    }

    #[cfg(target_os = "fuchsia")]
    unsafe fn import_zircon_handle_unchecked_locked(
        &self,
        import_semaphore_zircon_handle_info: ImportSemaphoreZirconHandleInfo,
        state: &mut SemaphoreState,
    ) -> Result<(), VulkanError> {
        let ImportSemaphoreZirconHandleInfo {
            flags,
            handle_type,
            zircon_handle,
            _ne: _,
        } = import_semaphore_zircon_handle_info;

        let info_vk = ash::vk::ImportSemaphoreZirconHandleInfoFUCHSIA {
            semaphore: self.handle,
            flags: flags.into(),
            handle_type: handle_type.into(),
            zircon_handle,
            ..Default::default()
        };

        let fns = self.device.fns();
        (fns.fuchsia_external_semaphore
            .import_semaphore_zircon_handle_fuchsia)(self.device.handle(), &info_vk)
        .result()
        .map_err(VulkanError::from)?;

        state.import(handle_type, flags.temporary);

        Ok(())
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, SemaphoreState> {
        self.state.lock()
    }
}

impl Drop for Semaphore {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.must_put_in_pool {
                let raw_sem = self.handle;
                self.device.semaphore_pool().lock().push(raw_sem);
            } else {
                let fns = self.device.fns();
                (fns.v1_0.destroy_semaphore)(self.device.handle(), self.handle, ptr::null());
            }
        }
    }
}

unsafe impl VulkanObject for Semaphore {
    type Handle = ash::vk::Semaphore;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Semaphore {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

crate::impl_id_counter!(Semaphore);

#[derive(Debug, Default)]
pub(crate) struct SemaphoreState {
    is_signaled: bool,
    pending_signal: Option<SignalType>,
    pending_wait: Option<Weak<Queue>>,

    reference_exported: bool,
    exported_handle_types: ExternalSemaphoreHandleTypes,
    current_import: Option<ImportType>,
    permanent_import: Option<ExternalSemaphoreHandleType>,
}

impl SemaphoreState {
    /// If the semaphore does not have a pending operation and has no external references,
    /// returns the current status.
    #[inline]
    fn is_signaled(&self) -> Option<bool> {
        // If any of these is true, we can't be certain of the status.
        if self.pending_signal.is_some()
            || self.pending_wait.is_some()
            || self.has_external_reference()
        {
            None
        } else {
            Some(self.is_signaled)
        }
    }

    #[inline]
    fn is_signal_pending(&self) -> bool {
        self.pending_signal.is_some()
    }

    #[inline]
    fn is_wait_pending(&self) -> bool {
        self.pending_wait.is_some()
    }

    #[inline]
    fn is_in_queue(&self) -> bool {
        matches!(self.pending_signal, Some(SignalType::Queue(_))) || self.pending_wait.is_some()
    }

    /// Returns whether there are any potential external references to the semaphore payload.
    /// That is, the semaphore has been exported by reference transference, or imported.
    #[inline]
    fn has_external_reference(&self) -> bool {
        self.reference_exported || self.current_import.is_some()
    }

    #[allow(dead_code)]
    #[inline]
    fn is_exported(&self, handle_type: ExternalSemaphoreHandleType) -> bool {
        self.exported_handle_types.intersects(&handle_type.into())
    }

    #[inline]
    pub(crate) unsafe fn add_queue_signal(&mut self, queue: &Arc<Queue>) {
        self.pending_signal = Some(SignalType::Queue(Arc::downgrade(queue)));
    }

    #[inline]
    pub(crate) unsafe fn add_queue_wait(&mut self, queue: &Arc<Queue>) {
        self.pending_wait = Some(Arc::downgrade(queue));
    }

    /// Called when a queue is unlocking resources.
    #[inline]
    pub(crate) unsafe fn set_signal_finished(&mut self) {
        self.pending_signal = None;
        self.is_signaled = true;
    }

    /// Called when a queue is unlocking resources.
    #[inline]
    pub(crate) unsafe fn set_wait_finished(&mut self) {
        self.pending_wait = None;
        self.current_import = self.permanent_import.map(Into::into);
        self.is_signaled = false;
    }

    #[allow(dead_code)]
    #[inline]
    unsafe fn export(&mut self, handle_type: ExternalSemaphoreHandleType) {
        self.exported_handle_types |= handle_type.into();

        if handle_type.has_copy_transference() {
            self.current_import = self.permanent_import.map(Into::into);
            self.is_signaled = false;
        } else {
            self.reference_exported = true;
        }
    }

    #[allow(dead_code)]
    #[inline]
    unsafe fn import(&mut self, handle_type: ExternalSemaphoreHandleType, temporary: bool) {
        self.current_import = Some(handle_type.into());

        if !temporary {
            self.permanent_import = Some(handle_type);
        }
    }

    #[inline]
    pub(crate) unsafe fn swapchain_acquire(&mut self) {
        self.pending_signal = Some(SignalType::SwapchainAcquire);
        self.current_import = Some(ImportType::SwapchainAcquire);
    }
}

#[derive(Clone, Debug)]
enum SignalType {
    Queue(Weak<Queue>),
    SwapchainAcquire,
}

#[derive(Clone, Copy, Debug)]
enum ImportType {
    SwapchainAcquire,
    ExternalSemaphore(ExternalSemaphoreHandleType),
}

impl From<ExternalSemaphoreHandleType> for ImportType {
    #[inline]
    fn from(handle_type: ExternalSemaphoreHandleType) -> Self {
        Self::ExternalSemaphore(handle_type)
    }
}

/// Parameters to create a new `Semaphore`.
#[derive(Clone, Debug)]
pub struct SemaphoreCreateInfo {
    /// The handle types that can be exported from the semaphore.
    ///
    /// The default value is [`ExternalSemaphoreHandleTypes::empty()`].
    pub export_handle_types: ExternalSemaphoreHandleTypes,

    pub _ne: crate::NonExhaustive,
}

impl Default for SemaphoreCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            export_handle_types: ExternalSemaphoreHandleTypes::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

vulkan_enum! {
    /// The handle type used for Vulkan external semaphore APIs.
    #[non_exhaustive]
    ExternalSemaphoreHandleType = ExternalSemaphoreHandleTypeFlags(u32);

    /// A POSIX file descriptor handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    OpaqueFd = OPAQUE_FD,

    /// A Windows NT handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    OpaqueWin32 = OPAQUE_WIN32,

    /// A Windows global share handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    OpaqueWin32Kmt = OPAQUE_WIN32_KMT,

    /// A Windows NT handle that refers to a Direct3D 11 or 12 fence.
    ///
    /// This handle type has *reference transference*.
    D3D12Fence = D3D12_FENCE,

    /// A POSIX file descriptor handle to a Linux Sync File or Android Fence object.
    ///
    /// This handle type has *copy transference*.
    SyncFd = SYNC_FD,

    /// A handle to a Zircon event object.
    ///
    /// This handle type has *reference transference*.
    ///
    /// The
    /// [`fuchsia_external_semaphore`](crate::device::DeviceExtensions::fuchsia_external_semaphore)
    /// extension must be enabled on the device.
    ZirconEvent = ZIRCON_EVENT_FUCHSIA {
        device_extensions: [fuchsia_external_semaphore],
    },
}

impl ExternalSemaphoreHandleType {
    /// Returns whether the given handle type has *copy transference* rather than *reference
    /// transference*.
    ///
    /// Imports of handles with copy transference must always be temporary. Exports of such
    /// handles must only occur if no queue is waiting on the semaphore, and only if the semaphore
    /// is already signaled, or if there is a semaphore signal operation pending in a queue.
    #[inline]
    pub fn has_copy_transference(&self) -> bool {
        // As defined by
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-semaphore-handletypes-win32
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-semaphore-handletypes-fd
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-semaphore-handletypes-fuchsia
        matches!(self, Self::SyncFd)
    }
}

vulkan_bitflags! {
    /// A mask of multiple external semaphore handle types.
    #[non_exhaustive]
    ExternalSemaphoreHandleTypes = ExternalSemaphoreHandleTypeFlags(u32);

    /// A POSIX file descriptor handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    opaque_fd = OPAQUE_FD,

    /// A Windows NT handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    opaque_win32 = OPAQUE_WIN32,

    /// A Windows global share handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    opaque_win32_kmt = OPAQUE_WIN32_KMT,

    /// A Windows NT handle that refers to a Direct3D 11 or 12 fence.
    ///
    /// This handle type has *reference transference*.
    d3d12_fence = D3D12_FENCE,

    /// A POSIX file descriptor handle to a Linux Sync File or Android Fence object.
    ///
    /// This handle type has *copy transference*.
    sync_fd = SYNC_FD,

    /// A handle to a Zircon event object.
    ///
    /// This handle type has *reference transference*.
    ///
    /// The
    /// [`fuchsia_external_semaphore`](crate::device::DeviceExtensions::fuchsia_external_semaphore)
    /// extension must be enabled on the device.
    zircon_event = ZIRCON_EVENT_FUCHSIA {
        device_extensions: [fuchsia_external_semaphore],
    },
}

impl From<ExternalSemaphoreHandleType> for ExternalSemaphoreHandleTypes {
    #[inline]
    fn from(val: ExternalSemaphoreHandleType) -> Self {
        let mut result = Self::empty();

        match val {
            ExternalSemaphoreHandleType::OpaqueFd => result.opaque_fd = true,
            ExternalSemaphoreHandleType::OpaqueWin32 => result.opaque_win32 = true,
            ExternalSemaphoreHandleType::OpaqueWin32Kmt => result.opaque_win32_kmt = true,
            ExternalSemaphoreHandleType::D3D12Fence => result.d3d12_fence = true,
            ExternalSemaphoreHandleType::SyncFd => result.sync_fd = true,
            ExternalSemaphoreHandleType::ZirconEvent => result.zircon_event = true,
        }

        result
    }
}

impl ExternalSemaphoreHandleTypes {
    fn into_iter(self) -> impl IntoIterator<Item = ExternalSemaphoreHandleType> {
        let Self {
            opaque_fd,
            opaque_win32,
            opaque_win32_kmt,
            d3d12_fence,
            sync_fd,
            zircon_event,
            _ne: _,
        } = self;

        [
            opaque_fd.then_some(ExternalSemaphoreHandleType::OpaqueFd),
            opaque_win32.then_some(ExternalSemaphoreHandleType::OpaqueWin32),
            opaque_win32_kmt.then_some(ExternalSemaphoreHandleType::OpaqueWin32Kmt),
            d3d12_fence.then_some(ExternalSemaphoreHandleType::D3D12Fence),
            sync_fd.then_some(ExternalSemaphoreHandleType::SyncFd),
            zircon_event.then_some(ExternalSemaphoreHandleType::ZirconEvent),
        ]
        .into_iter()
        .flatten()
    }
}

vulkan_bitflags! {
    /// Additional parameters for a semaphore payload import.
    #[non_exhaustive]
    SemaphoreImportFlags = SemaphoreImportFlags(u32);

    /// The semaphore payload will be imported only temporarily, regardless of the permanence of the
    /// imported handle type.
    temporary = TEMPORARY,
}

#[cfg(unix)]
#[derive(Debug)]
pub struct ImportSemaphoreFdInfo {
    /// Additional parameters for the import operation.
    ///
    /// If `handle_type` has *copy transference*, this must include the `temporary` flag.
    ///
    /// The default value is [`SemaphoreImportFlags::empty()`].
    pub flags: SemaphoreImportFlags,

    /// The handle type of `file`.
    ///
    /// There is no default value.
    pub handle_type: ExternalSemaphoreHandleType,

    /// The file to import the semaphore from.
    ///
    /// If `handle_type` is `ExternalSemaphoreHandleType::SyncFd`, then `file` can be `None`.
    /// Instead of an imported file descriptor, a dummy file descriptor `-1` is used,
    /// which represents a semaphore that is always signaled.
    ///
    /// The default value is `None`, which must be overridden if `handle_type` is not
    /// `ExternalSemaphoreHandleType::SyncFd`.
    pub file: Option<File>,

    pub _ne: crate::NonExhaustive,
}

#[cfg(unix)]
impl ImportSemaphoreFdInfo {
    /// Returns an `ImportSemaphoreFdInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalSemaphoreHandleType) -> Self {
        Self {
            flags: SemaphoreImportFlags::empty(),
            handle_type,
            file: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[cfg(windows)]
#[derive(Debug)]
pub struct ImportSemaphoreWin32HandleInfo {
    /// Additional parameters for the import operation.
    ///
    /// If `handle_type` has *copy transference*, this must include the `temporary` flag.
    ///
    /// The default value is [`SemaphoreImportFlags::empty()`].
    pub flags: SemaphoreImportFlags,

    /// The handle type of `handle`.
    ///
    /// There is no default value.
    pub handle_type: ExternalSemaphoreHandleType,

    /// The handle to import the semaphore from.
    ///
    /// The default value is `null`, which must be overridden.
    pub handle: *mut std::ffi::c_void,

    pub _ne: crate::NonExhaustive,
}

#[cfg(windows)]
impl ImportSemaphoreWin32HandleInfo {
    /// Returns an `ImportSemaphoreWin32HandleInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalSemaphoreHandleType) -> Self {
        Self {
            flags: SemaphoreImportFlags::empty(),
            handle_type,
            handle: ptr::null_mut(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[cfg(target_os = "fuchsia")]
#[derive(Debug)]
pub struct ImportSemaphoreZirconHandleInfo {
    /// Additional parameters for the import operation.
    ///
    /// If `handle_type` has *copy transference*, this must include the `temporary` flag.
    ///
    /// The default value is [`SemaphoreImportFlags::empty()`].
    pub flags: SemaphoreImportFlags,

    /// The handle type of `handle`.
    ///
    /// There is no default value.
    pub handle_type: ExternalSemaphoreHandleType,

    /// The handle to import the semaphore from.
    ///
    /// The default value is `ZX_HANDLE_INVALID`, which must be overridden.
    pub zircon_handle: ash::vk::zx_handle_t,

    pub _ne: crate::NonExhaustive,
}

#[cfg(target_os = "fuchsia")]
impl ImportSemaphoreZirconHandleInfo {
    /// Returns an `ImportSemaphoreZirconHandleInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalSemaphoreHandleType) -> Self {
        Self {
            flags: SemaphoreImportFlags::empty(),
            handle_type,
            zircon_handle: 0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The semaphore configuration to query in
/// [`PhysicalDevice::external_semaphore_properties`](crate::device::physical::PhysicalDevice::external_semaphore_properties).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExternalSemaphoreInfo {
    /// The external handle type that will be used with the semaphore.
    pub handle_type: ExternalSemaphoreHandleType,

    pub _ne: crate::NonExhaustive,
}

impl ExternalSemaphoreInfo {
    /// Returns an `ExternalSemaphoreInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalSemaphoreHandleType) -> Self {
        Self {
            handle_type,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The properties for exporting or importing external handles, when a semaphore is created
/// with a specific configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExternalSemaphoreProperties {
    /// Whether a handle can be exported to an external source with the queried
    /// external handle type.
    pub exportable: bool,

    /// Whether a handle can be imported from an external source with the queried
    /// external handle type.
    pub importable: bool,

    /// Which external handle types can be re-exported after the queried external handle type has
    /// been imported.
    pub export_from_imported_handle_types: ExternalSemaphoreHandleTypes,

    /// Which external handle types can be enabled along with the queried external handle type
    /// when creating the semaphore.
    pub compatible_handle_types: ExternalSemaphoreHandleTypes,
}

/// Error that can be returned from operations on a semaphore.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SemaphoreError {
    /// Not enough memory available.
    OomError(OomError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The provided handle type does not permit more than one export,
    /// and a handle of this type was already exported previously.
    AlreadyExported,

    /// The provided handle type cannot be exported from the current import handle type.
    ExportFromImportedNotSupported {
        imported_handle_type: ExternalSemaphoreHandleType,
    },

    /// One of the export handle types is not compatible with the other provided handles.
    ExportHandleTypesNotCompatible,

    /// A handle type with copy transference was provided, but the semaphore is not signaled and
    /// there is no pending queue operation that will signal it.
    HandleTypeCopyNotSignaled,

    /// A handle type with copy transference was provided,
    /// but the `temporary` import flag was not set.
    HandletypeCopyNotTemporary,

    /// The provided export handle type was not set in `export_handle_types` when creating the
    /// semaphore.
    HandleTypeNotEnabled,

    /// Exporting is not supported for the provided handle type.
    HandleTypeNotExportable {
        handle_type: ExternalSemaphoreHandleType,
    },

    /// The provided handle type is not a POSIX file descriptor handle.
    HandleTypeNotFd,

    /// The provided handle type is not a Win32 handle.
    HandleTypeNotWin32,

    /// The provided handle type is not a Zircon event handle.
    HandleTypeNotZircon,

    /// The semaphore currently has a temporary import for a swapchain acquire operation.
    ImportedForSwapchainAcquire,

    /// The semaphore is currently in use by a queue.
    InQueue,

    /// A queue is currently waiting on the semaphore.
    QueueIsWaiting,
}

impl Error for SemaphoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for SemaphoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),

            Self::AlreadyExported => write!(
                f,
                "the provided handle type does not permit more than one export, and a handle of \
                this type was already exported previously",
            ),
            Self::ExportFromImportedNotSupported {
                imported_handle_type,
            } => write!(
                f,
                "the provided handle type cannot be exported from the current imported handle type \
                {:?}",
                imported_handle_type,
            ),
            Self::ExportHandleTypesNotCompatible => write!(
                f,
                "one of the export handle types is not compatible with the other provided handles",
            ),
            Self::HandleTypeCopyNotSignaled => write!(
                f,
                "a handle type with copy transference was provided, but the semaphore is not \
                signaled and there is no pending queue operation that will signal it",
            ),
            Self::HandletypeCopyNotTemporary => write!(
                f,
                "a handle type with copy transference was provided, but the `temporary` \
                import flag was not set",
            ),
            Self::HandleTypeNotEnabled => write!(
                f,
                "the provided export handle type was not set in `export_handle_types` when \
                creating the semaphore",
            ),
            Self::HandleTypeNotExportable { handle_type } => write!(
                f,
                "exporting is not supported for handles of type {:?}",
                handle_type,
            ),
            Self::HandleTypeNotFd => write!(
                f,
                "the provided handle type is not a POSIX file descriptor handle",
            ),
            Self::HandleTypeNotWin32 => {
                write!(f, "the provided handle type is not a Win32 handle")
            }
            Self::HandleTypeNotZircon => {
                write!(f, "the provided handle type is not a Zircon event handle")
            }
            Self::ImportedForSwapchainAcquire => write!(
                f,
                "the semaphore currently has a temporary import for a swapchain acquire operation",
            ),
            Self::InQueue => write!(f, "the semaphore is currently in use by a queue"),
            Self::QueueIsWaiting => write!(f, "a queue is currently waiting on the semaphore"),
        }
    }
}

impl From<VulkanError> for SemaphoreError {
    fn from(err: VulkanError) -> Self {
        match err {
            e @ VulkanError::OutOfHostMemory | e @ VulkanError::OutOfDeviceMemory => {
                Self::OomError(e.into())
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for SemaphoreError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<RequirementNotMet> for SemaphoreError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExternalSemaphoreHandleType;
    use crate::{
        device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
        instance::{Instance, InstanceCreateInfo, InstanceExtensions},
        sync::{ExternalSemaphoreHandleTypes, Semaphore, SemaphoreCreateInfo},
        VulkanLibrary, VulkanObject,
    };

    #[test]
    fn semaphore_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = Semaphore::new(device, Default::default());
    }

    #[test]
    fn semaphore_pool() {
        let (device, _) = gfx_dev_and_queue!();

        assert_eq!(device.semaphore_pool().lock().len(), 0);
        let sem1_internal_obj = {
            let sem = Semaphore::from_pool(device.clone()).unwrap();
            assert_eq!(device.semaphore_pool().lock().len(), 0);
            sem.handle()
        };

        assert_eq!(device.semaphore_pool().lock().len(), 1);
        let sem2 = Semaphore::from_pool(device.clone()).unwrap();
        assert_eq!(device.semaphore_pool().lock().len(), 0);
        assert_eq!(sem2.handle(), sem1_internal_obj);
    }

    #[test]
    #[cfg(unix)]
    fn semaphore_export_fd() {
        let library = match VulkanLibrary::new() {
            Ok(x) => x,
            Err(_) => return,
        };

        let instance = match Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: InstanceExtensions {
                    khr_get_physical_device_properties2: true,
                    khr_external_semaphore_capabilities: true,
                    ..InstanceExtensions::empty()
                },
                ..Default::default()
            },
        ) {
            Ok(x) => x,
            Err(_) => return,
        };

        let physical_device = match instance.enumerate_physical_devices() {
            Ok(mut x) => x.next().unwrap(),
            Err(_) => return,
        };

        let (device, _) = match Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions {
                    khr_external_semaphore: true,
                    khr_external_semaphore_fd: true,
                    ..DeviceExtensions::empty()
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                ..Default::default()
            },
        ) {
            Ok(x) => x,
            Err(_) => return,
        };

        let sem = Semaphore::new(
            device,
            SemaphoreCreateInfo {
                export_handle_types: ExternalSemaphoreHandleTypes {
                    opaque_fd: true,
                    ..ExternalSemaphoreHandleTypes::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let _fd = sem
            .export_fd(ExternalSemaphoreHandleType::OpaqueFd)
            .unwrap();
    }
}
