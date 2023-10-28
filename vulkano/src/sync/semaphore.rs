// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A semaphore provides synchronization between multiple queues, with non-command buffer
//! commands on the same queue, or between the device and an external source.
//!
//! A semaphore has two states: **signaled** and **unsignaled**.
//! Only the device can perform operations on a semaphore,
//! the host cannot perform any operations on it.
//!
//! Two operations can be performed on a semaphore:
//! - A **semaphore signal operation** will put the semaphore into the signaled state.
//! - A **semaphore wait operation** will block execution of the operation is associated with,
//!   as long as the semaphore is in the unsignaled state. Once the semaphore is in the signaled
//!   state, the semaphore is put back in the unsignaled state and execution continues.
//!
//! Semaphore signals and waits must always occur in pairs: one signal operation is paired with one
//! wait operation. If a semaphore is signaled without waiting for it, it stays in the signaled
//! state until it is waited for, or destroyed.
//!
//! # Safety
//!
//! - When a semaphore signal operation is executed on the device,
//!   the semaphore must be in the unsignaled state.
//!   In other words, the same semaphore cannot be signalled by multiple commands;
//!   there must always be a wait operation in between them.
//! - There must never be more than one semaphore wait operation executing on the same semaphore
//!   at the same time.
//! - When a semaphore wait operation is queued as part of a command,
//!   the semaphore must already be in the signaled state, or
//!   the signal operation that it waits for must have been queued previously
//!   (as part of a previous command, or an earlier batch within the same command).

use crate::{
    device::{physical::PhysicalDevice, Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_bitflags_enum},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use std::{fs::File, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

/// Used to provide synchronization between command buffers during their execution.
///
/// It is similar to a fence, except that it is purely on the GPU side. The CPU can't query a
/// semaphore's status or wait for it to be signaled.
#[derive(Debug)]
pub struct Semaphore {
    handle: ash::vk::Semaphore,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    export_handle_types: ExternalSemaphoreHandleTypes,

    must_put_in_pool: bool,
}

impl Semaphore {
    /// Creates a new `Semaphore`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        create_info: SemaphoreCreateInfo,
    ) -> Result<Semaphore, Validated<VulkanError>> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        create_info: &SemaphoreCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
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

        Ok(Self::from_handle(device, handle, create_info))
    }

    /// Takes a semaphore from the vulkano-provided semaphore pool.
    /// If the pool is empty, a new semaphore will be allocated.
    /// Upon `drop`, the semaphore is put back into the pool.
    ///
    /// For most applications, using the pool should be preferred,
    /// in order to avoid creating new semaphores every frame.
    #[inline]
    pub fn from_pool(device: Arc<Device>) -> Result<Semaphore, VulkanError> {
        let handle = device.semaphore_pool().lock().pop();
        let semaphore = match handle {
            Some(handle) => Semaphore {
                handle,
                device: InstanceOwnedDebugWrapper(device),
                id: Self::next_id(),

                export_handle_types: ExternalSemaphoreHandleTypes::empty(),

                must_put_in_pool: true,
            },
            None => {
                // Pool is empty, alloc new semaphore
                let mut semaphore =
                    unsafe { Semaphore::new_unchecked(device, Default::default())? };
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
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            export_handle_types,

            must_put_in_pool: false,
        }
    }

    /// Returns the handle types that can be exported from the semaphore.
    #[inline]
    pub fn export_handle_types(&self) -> ExternalSemaphoreHandleTypes {
        self.export_handle_types
    }

    /// Exports the semaphore into a POSIX file descriptor. The caller owns the returned `File`.
    ///
    /// # Safety
    ///
    /// - If `handle_type` has copy transference, then the semaphore must be signaled, or a signal
    ///   operation on the semaphore must be pending, and no wait operations must be pending.
    /// - The semaphore must not currently have an imported payload from a swapchain acquire
    ///   operation.
    /// - If the semaphore has an imported payload, its handle type must allow re-exporting as
    ///   `handle_type`, as returned by [`PhysicalDevice::external_semaphore_properties`].
    #[inline]
    pub unsafe fn export_fd(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<File, Validated<VulkanError>> {
        self.validate_export_fd(handle_type)?;

        unsafe { Ok(self.export_fd_unchecked(handle_type)?) }
    }

    fn validate_export_fd(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().khr_external_semaphore_fd {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_semaphore_fd",
                )])]),
                ..Default::default()
            }));
        }

        handle_type.validate_device(&self.device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkSemaphoreGetFdInfoKHR-handleType-parameter"])
        })?;

        if !matches!(
            handle_type,
            ExternalSemaphoreHandleType::OpaqueFd | ExternalSemaphoreHandleType::SyncFd
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalSemaphoreHandleType::OpaqueFd` or \
                    `ExternalSemaphoreHandleType::SyncFd`"
                    .into(),
                vuids: &["VUID-VkSemaphoreGetFdInfoKHR-handleType-01136"],
                ..Default::default()
            }));
        }

        if !self.export_handle_types.intersects(handle_type.into()) {
            return Err(Box::new(ValidationError {
                problem: "`self.export_handle_types()` does not contain `handle_type`".into(),
                vuids: &["VUID-VkSemaphoreGetFdInfoKHR-handleType-01132"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn export_fd_unchecked(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<File, VulkanError> {
        let info_vk = ash::vk::SemaphoreGetFdInfoKHR {
            semaphore: self.handle,
            handle_type: handle_type.into(),
            ..Default::default()
        };

        let mut output = MaybeUninit::uninit();
        let fns = self.device.fns();
        (fns.khr_external_semaphore_fd.get_semaphore_fd_khr)(
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
            unreachable!("`khr_external_semaphore_fd` was somehow enabled on a non-Unix system");
        }
    }

    /// Exports the semaphore into a Win32 handle.
    ///
    /// The [`khr_external_semaphore_win32`](crate::device::DeviceExtensions::khr_external_semaphore_win32)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - If `handle_type` has copy transference, then the semaphore must be signaled, or a signal
    ///   operation on the semaphore must be pending, and no wait operations must be pending.
    /// - The semaphore must not currently have an imported payload from a swapchain acquire
    ///   operation.
    /// - If the semaphore has an imported payload, its handle type must allow re-exporting as
    ///   `handle_type`, as returned by [`PhysicalDevice::external_semaphore_properties`].
    /// - If `handle_type` is `ExternalSemaphoreHandleType::OpaqueWin32` or
    ///   `ExternalSemaphoreHandleType::D3D12Fence`, then a handle of this type must not have been
    ///   already exported from this semaphore.
    #[inline]
    pub fn export_win32_handle(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<*mut std::ffi::c_void, Validated<VulkanError>> {
        self.validate_export_win32_handle(handle_type)?;

        unsafe { Ok(self.export_win32_handle_unchecked(handle_type)?) }
    }

    fn validate_export_win32_handle(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .device
            .enabled_extensions()
            .khr_external_semaphore_win32
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_semaphore_win32",
                )])]),
                ..Default::default()
            }));
        }

        handle_type.validate_device(&self.device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkSemaphoreGetWin32HandleInfoKHR-handleType-parameter"])
        })?;

        if !matches!(
            handle_type,
            ExternalSemaphoreHandleType::OpaqueWin32
                | ExternalSemaphoreHandleType::OpaqueWin32Kmt
                | ExternalSemaphoreHandleType::D3D12Fence
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalSemaphoreHandleType::OpaqueWin32`, \
                    `ExternalSemaphoreHandleType::OpaqueWin32Kmt` or \
                    `ExternalSemaphoreHandleType::D3D12Fence`"
                    .into(),
                vuids: &["VUID-VkSemaphoreGetWin32HandleInfoKHR-handleType-01131"],
                ..Default::default()
            }));
        }

        if !self.export_handle_types.intersects(handle_type.into()) {
            return Err(Box::new(ValidationError {
                problem: "`self.export_handle_types()` does not contain `handle_type`".into(),
                vuids: &["VUID-VkSemaphoreGetWin32HandleInfoKHR-handleType-01126"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn export_win32_handle_unchecked(
        &self,
        handle_type: ExternalSemaphoreHandleType,
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

        Ok(output.assume_init())
    }

    /// Exports the semaphore into a Zircon event handle.
    ///
    /// # Safety
    ///
    /// - If `handle_type` has copy transference, then the semaphore must be signaled, or a signal
    ///   operation on the semaphore must be pending, and no wait operations must be pending.
    /// - The semaphore must not currently have an imported payload from a swapchain acquire
    ///   operation.
    /// - If the semaphore has an imported payload, its handle type must allow re-exporting as
    ///   `handle_type`, as returned by [`PhysicalDevice::external_semaphore_properties`].
    #[inline]
    pub unsafe fn export_zircon_handle(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<ash::vk::zx_handle_t, Validated<VulkanError>> {
        self.validate_export_zircon_handle(handle_type)?;

        unsafe { Ok(self.export_zircon_handle_unchecked(handle_type)?) }
    }

    fn validate_export_zircon_handle(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().fuchsia_external_semaphore {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "fuchsia_external_semaphore",
                )])]),
                ..Default::default()
            }));
        }

        handle_type.validate_device(&self.device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkSemaphoreGetZirconHandleInfoFUCHSIA-handleType-parameter"])
        })?;

        if !matches!(handle_type, ExternalSemaphoreHandleType::ZirconEvent) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalSemaphoreHandleType::ZirconEvent`".into(),
                vuids: &["VUID-VkSemaphoreGetZirconHandleInfoFUCHSIA-handleType-04762"],
                ..Default::default()
            }));
        }

        if !self.export_handle_types.intersects(handle_type.into()) {
            return Err(Box::new(ValidationError {
                problem: "`self.export_handle_types()` does not contain `handle_type`".into(),
                vuids: &["VUID-VkSemaphoreGetZirconHandleInfoFUCHSIA-handleType-04758"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn export_zircon_handle_unchecked(
        &self,
        handle_type: ExternalSemaphoreHandleType,
    ) -> Result<ash::vk::zx_handle_t, VulkanError> {
        let info_vk = ash::vk::SemaphoreGetZirconHandleInfoFUCHSIA {
            semaphore: self.handle,
            handle_type: handle_type.into(),
            ..Default::default()
        };

        let mut output = MaybeUninit::uninit();
        let fns = self.device.fns();
        (fns.fuchsia_external_semaphore
            .get_semaphore_zircon_handle_fuchsia)(
            self.device.handle(),
            &info_vk,
            output.as_mut_ptr(),
        )
        .result()
        .map_err(VulkanError::from)?;

        Ok(output.assume_init())
    }

    /// Imports a semaphore from a POSIX file descriptor.
    ///
    /// The [`khr_external_semaphore_fd`](crate::device::DeviceExtensions::khr_external_semaphore_fd)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - The semaphore must not be in use by the device.
    /// - If in `import_semaphore_fd_info`, `handle_type` is `ExternalHandleType::OpaqueFd`,
    ///   then `file` must represent a binary semaphore that was exported from Vulkan or a
    ///   compatible API, with a driver and device UUID equal to those of the device that owns
    ///   `self`.
    #[inline]
    pub unsafe fn import_fd(
        &self,
        import_semaphore_fd_info: ImportSemaphoreFdInfo,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_import_fd(&import_semaphore_fd_info)?;

        Ok(self.import_fd_unchecked(import_semaphore_fd_info)?)
    }

    fn validate_import_fd(
        &self,
        import_semaphore_fd_info: &ImportSemaphoreFdInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().khr_external_semaphore_fd {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_semaphore_fd",
                )])]),
                ..Default::default()
            }));
        }

        import_semaphore_fd_info
            .validate(&self.device)
            .map_err(|err| err.add_context("import_semaphore_fd_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn import_fd_unchecked(
        &self,
        import_semaphore_fd_info: ImportSemaphoreFdInfo,
    ) -> Result<(), VulkanError> {
        let ImportSemaphoreFdInfo {
            flags,
            handle_type,
            file,
            _ne: _,
        } = import_semaphore_fd_info;

        #[cfg(unix)]
        let fd = {
            use std::os::fd::IntoRawFd;
            file.map_or(-1, |file| file.into_raw_fd())
        };

        #[cfg(not(unix))]
        let fd = {
            let _ = file;
            -1
        };

        let info_vk = ash::vk::ImportSemaphoreFdInfoKHR {
            semaphore: self.handle,
            flags: flags.into(),
            handle_type: handle_type.into(),
            fd,
            ..Default::default()
        };

        let fns = self.device.fns();
        (fns.khr_external_semaphore_fd.import_semaphore_fd_khr)(self.device.handle(), &info_vk)
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Imports a semaphore from a Win32 handle.
    ///
    /// The [`khr_external_semaphore_win32`](crate::device::DeviceExtensions::khr_external_semaphore_win32)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - The semaphore must not be in use by the device.
    /// - In `import_semaphore_win32_handle_info`, `handle` must represent a binary semaphore that
    ///   was exported from Vulkan or a compatible API, with a driver and device UUID equal to
    ///   those of the device that owns `self`.
    #[inline]
    pub unsafe fn import_win32_handle(
        &self,
        import_semaphore_win32_handle_info: ImportSemaphoreWin32HandleInfo,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_import_win32_handle(&import_semaphore_win32_handle_info)?;

        Ok(self.import_win32_handle_unchecked(import_semaphore_win32_handle_info)?)
    }

    fn validate_import_win32_handle(
        &self,
        import_semaphore_win32_handle_info: &ImportSemaphoreWin32HandleInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .device
            .enabled_extensions()
            .khr_external_semaphore_win32
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_semaphore_win32",
                )])]),
                ..Default::default()
            }));
        }

        import_semaphore_win32_handle_info
            .validate(&self.device)
            .map_err(|err| err.add_context("import_semaphore_win32_handle_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn import_win32_handle_unchecked(
        &self,
        import_semaphore_win32_handle_info: ImportSemaphoreWin32HandleInfo,
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

        Ok(())
    }

    /// Imports a semaphore from a Zircon event handle.
    ///
    /// The [`fuchsia_external_semaphore`](crate::device::DeviceExtensions::fuchsia_external_semaphore)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - The semaphore must not be in use by the device.
    /// - In `import_semaphore_zircon_handle_info`, `zircon_handle` must have `ZX_RIGHTS_BASIC` and
    ///   `ZX_RIGHTS_SIGNAL`.
    #[inline]
    pub unsafe fn import_zircon_handle(
        &self,
        import_semaphore_zircon_handle_info: ImportSemaphoreZirconHandleInfo,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_import_zircon_handle(&import_semaphore_zircon_handle_info)?;

        Ok(self.import_zircon_handle_unchecked(import_semaphore_zircon_handle_info)?)
    }

    fn validate_import_zircon_handle(
        &self,
        import_semaphore_zircon_handle_info: &ImportSemaphoreZirconHandleInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().fuchsia_external_semaphore {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "fuchsia_external_semaphore",
                )])]),
                ..Default::default()
            }));
        }

        import_semaphore_zircon_handle_info
            .validate(&self.device)
            .map_err(|err| err.add_context("import_semaphore_zircon_handle_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn import_zircon_handle_unchecked(
        &self,
        import_semaphore_zircon_handle_info: ImportSemaphoreZirconHandleInfo,
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

        Ok(())
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

impl_id_counter!(Semaphore);

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

impl SemaphoreCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            export_handle_types,
            _ne: _,
        } = self;

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_semaphore)
            {
                return Err(Box::new(ValidationError {
                    context: "export_handle_types".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_external_semaphore")]),
                    ]),
                    ..Default::default()
                }));
            }

            export_handle_types.validate_device(device).map_err(|err| {
                err.add_context("export_handle_types")
                    .set_vuids(&["VUID-VkExportSemaphoreCreateInfo-handleTypes-parameter"])
            })?;

            for handle_type in export_handle_types.into_iter() {
                let external_semaphore_properties = unsafe {
                    device
                        .physical_device()
                        .external_semaphore_properties_unchecked(
                            ExternalSemaphoreInfo::handle_type(handle_type),
                        )
                };

                if !external_semaphore_properties.exportable {
                    return Err(Box::new(ValidationError {
                        context: "export_handle_types".into(),
                        problem: format!(
                            "the handle type `ExternalSemaphoreHandleTypes::{:?}` is not \
                            exportable, as returned by \
                            `PhysicalDevice::external_semaphore_properties`",
                            ExternalSemaphoreHandleTypes::from(handle_type)
                        )
                        .into(),
                        vuids: &["VUID-VkExportSemaphoreCreateInfo-handleTypes-01124"],
                        ..Default::default()
                    }));
                }

                if !external_semaphore_properties
                    .compatible_handle_types
                    .contains(export_handle_types)
                {
                    return Err(Box::new(ValidationError {
                        context: "export_handle_types".into(),
                        problem: format!(
                            "the handle type `ExternalSemaphoreHandleTypes::{:?}` is not \
                            compatible with the other specified handle types, as returned by \
                            `PhysicalDevice::external_semaphore_properties`",
                            ExternalSemaphoreHandleTypes::from(handle_type)
                        )
                        .into(),
                        vuids: &["VUID-VkExportSemaphoreCreateInfo-handleTypes-01124"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`ExternalSemaphoreHandleType`] values.
    ExternalSemaphoreHandleTypes,

    /// The handle type used to export or import semaphores to/from an external source.
    ExternalSemaphoreHandleType impl {
        /// Returns whether the given handle type has *copy transference* rather than *reference
        /// transference*.
        ///
        /// Imports of handles with copy transference must always be temporary. Exports of such
        /// handles must only occur if no queue is waiting on the semaphore, and only if the semaphore
        /// is already signaled, or if there is a semaphore signal operation pending in a queue.
        #[inline]
        pub fn has_copy_transference(self) -> bool {
            // As defined by
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-semaphore-handletypes-win32
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-semaphore-handletypes-fd
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-semaphore-handletypes-fuchsia
            matches!(self, Self::SyncFd)
        }
    },

    = ExternalSemaphoreHandleTypeFlags(u32);

    /// A POSIX file descriptor handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    OPAQUE_FD, OpaqueFd = OPAQUE_FD,

    /// A Windows NT handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    OPAQUE_WIN32, OpaqueWin32 = OPAQUE_WIN32,

    /// A Windows global share handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    OPAQUE_WIN32_KMT, OpaqueWin32Kmt = OPAQUE_WIN32_KMT,

    /// A Windows NT handle that refers to a Direct3D 11 or 12 fence.
    ///
    /// This handle type has *reference transference*.
    D3D12_FENCE, D3D12Fence = D3D12_FENCE,

    /// A POSIX file descriptor handle to a Linux Sync File or Android Fence object.
    ///
    /// This handle type has *copy transference*.
    SYNC_FD, SyncFd = SYNC_FD,

    /// A handle to a Zircon event object.
    ///
    /// This handle type has *reference transference*.
    ///
    /// The [`fuchsia_external_semaphore`] extension must be enabled on the device.
    ///
    /// [`fuchsia_external_semaphore`]: crate::device::DeviceExtensions::fuchsia_external_semaphore
    ZIRCON_EVENT, ZirconEvent = ZIRCON_EVENT_FUCHSIA
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(fuchsia_external_semaphore)]),
    ]),
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Additional parameters for a semaphore payload import.
    SemaphoreImportFlags = SemaphoreImportFlags(u32);

    /// The semaphore payload will be imported only temporarily, regardless of the permanence of the
    /// imported handle type.
    TEMPORARY = TEMPORARY,
}

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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            handle_type,
            file: _,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkImportSemaphoreFdInfoKHR-flags-parameter"])
        })?;

        handle_type.validate_device(device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkImportSemaphoreFdInfoKHR-handleType-parameter"])
        })?;

        if !matches!(
            handle_type,
            ExternalSemaphoreHandleType::OpaqueFd | ExternalSemaphoreHandleType::SyncFd
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalSemaphoreHandleType::OpaqueFd` or \
                    `ExternalSemaphoreHandleType::SyncFd`"
                    .into(),
                vuids: &["VUID-VkImportSemaphoreFdInfoKHR-handleType-01143"],
                ..Default::default()
            }));
        }

        // VUID-VkImportSemaphoreFdInfoKHR-fd-01544
        // VUID-VkImportSemaphoreFdInfoKHR-handleType-03263
        // Can't validate, therefore unsafe

        if handle_type.has_copy_transference() && !flags.intersects(SemaphoreImportFlags::TEMPORARY)
        {
            return Err(Box::new(ValidationError {
                problem: "`handle_type` has copy transference, but \
                    `flags` does not contain `SemaphoreImportFlags::TEMPORARY`"
                    .into(),
                vuids: &["VUID-VkImportSemaphoreFdInfoKHR-handleType-07307"],
                ..Default::default()
            }));
        }

        Ok(())
    }
}

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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            handle_type,
            handle: _,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkImportSemaphoreWin32HandleInfoKHR-flags-parameter"])
        })?;

        handle_type.validate_device(device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkImportSemaphoreWin32HandleInfoKHR-handleType-01140"])
        })?;

        if !matches!(
            handle_type,
            ExternalSemaphoreHandleType::OpaqueWin32
                | ExternalSemaphoreHandleType::OpaqueWin32Kmt
                | ExternalSemaphoreHandleType::D3D12Fence
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalSemaphoreHandleType::OpaqueWin32`, \
                    `ExternalSemaphoreHandleType::OpaqueWin32Kmt` or \
                    `ExternalSemaphoreHandleType::D3D12Fence`"
                    .into(),
                vuids: &["VUID-VkImportSemaphoreWin32HandleInfoKHR-handleType-01140"],
                ..Default::default()
            }));
        }

        // VUID-VkImportSemaphoreWin32HandleInfoKHR-handle-01542
        // Can't validate, therefore unsafe

        if handle_type.has_copy_transference() && !flags.intersects(SemaphoreImportFlags::TEMPORARY)
        {
            return Err(Box::new(ValidationError {
                problem: "`handle_type` has copy transference, but \
                    `flags` does not contain `SemaphoreImportFlags::TEMPORARY`"
                    .into(),
                // vuids?
                ..Default::default()
            }));
        }

        Ok(())
    }
}

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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            handle_type,
            zircon_handle: _,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-flags-parameter"])
        })?;

        handle_type.validate_device(device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-handleType-parameter"])
        })?;

        if !matches!(handle_type, ExternalSemaphoreHandleType::ZirconEvent) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalSemaphoreHandleType::ZirconEvent`".into(),
                vuids: &["VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-handleType-04765"],
                ..Default::default()
            }));
        }

        // VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-zirconHandle-04766
        // VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-zirconHandle-04767
        // Can't validate, therefore unsafe

        if handle_type.has_copy_transference() && !flags.intersects(SemaphoreImportFlags::TEMPORARY)
        {
            return Err(Box::new(ValidationError {
                problem: "`handle_type` has copy transference, but \
                    `flags` does not contain `SemaphoreImportFlags::TEMPORARY`"
                    .into(),
                // vuids?
                ..Default::default()
            }));
        }

        Ok(())
    }
}

/// The semaphore configuration to query in
/// [`PhysicalDevice::external_semaphore_properties`](crate::device::physical::PhysicalDevice::external_semaphore_properties).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExternalSemaphoreInfo {
    /// The external handle type that will be used with the semaphore.
    ///
    /// There is no default value.
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

    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            handle_type,
            _ne: _,
        } = self;

        handle_type
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("handle_type")
                    .set_vuids(&["VUID-VkPhysicalDeviceExternalSemaphoreInfo-handleType-parameter"])
            })?;

        Ok(())
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

#[cfg(test)]
mod tests {
    use crate::{
        device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
        instance::{Instance, InstanceCreateInfo, InstanceExtensions},
        sync::semaphore::{
            ExternalSemaphoreHandleType, ExternalSemaphoreHandleTypes, Semaphore,
            SemaphoreCreateInfo,
        },
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
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                enabled_extensions: DeviceExtensions {
                    khr_external_semaphore: true,
                    khr_external_semaphore_fd: true,
                    ..DeviceExtensions::empty()
                },
                ..Default::default()
            },
        ) {
            Ok(x) => x,
            Err(_) => return,
        };

        let sem = Semaphore::new(
            device,
            SemaphoreCreateInfo {
                export_handle_types: ExternalSemaphoreHandleTypes::OPAQUE_FD,
                ..Default::default()
            },
        )
        .unwrap();
        let _fd = unsafe {
            sem.export_fd(ExternalSemaphoreHandleType::OpaqueFd)
                .unwrap()
        };
    }
}
