// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A fence provides synchronization between the device and the host, or between an external source
//! and the host.

use crate::{
    device::{Device, DeviceOwned, Queue},
    macros::{impl_id_counter, vulkan_bitflags, vulkan_bitflags_enum},
    OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
#[cfg(unix)]
use std::fs::File;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    future::Future,
    mem::MaybeUninit,
    num::NonZeroU64,
    pin::Pin,
    ptr,
    sync::{Arc, Weak},
    task::{Context, Poll},
    time::Duration,
};

/// A two-state synchronization primitive that is signalled by the device and waited on by the host.
///
/// # Queue-to-host synchronization
///
/// The primary use of a fence is to know when execution of a queue has reached a particular point.
/// When adding a command to a queue, a fence can be provided with the command, to be signaled
/// when the operation finishes. You can check for a fence's current status by calling
/// `is_signaled`, `wait` or `await` on it. If the fence is found to be signaled, that means that
/// the queue has completed the operation that is associated with the fence, and all operations that
/// were submitted before it have been completed as well.
///
/// When a queue command accesses a resource, it must be kept alive until the queue command has
/// finished executing, and you may not be allowed to perform certain other operations (or even any)
/// while the resource is in use. By calling `is_signaled`, `wait` or `await`, the queue will be
/// notified when the fence is signaled, so that all resources of the associated queue operation and
/// preceding operations can be released.
///
/// Because of this, it is highly recommended to call `is_signaled`, `wait` or `await` on your fences.
/// Otherwise, the queue will hold onto resources indefinitely (using up memory)
/// and resource locks will not be released, which may cause errors when submitting future
/// queue operations. It is not strictly necessary to wait for *every* fence, as a fence
/// that was signaled later in the queue will automatically clean up resources associated with
/// earlier fences too.
#[derive(Debug)]
pub struct Fence {
    handle: ash::vk::Fence,
    device: Arc<Device>,
    id: NonZeroU64,
    must_put_in_pool: bool,

    export_handle_types: ExternalFenceHandleTypes,

    state: Mutex<FenceState>,
}

impl Fence {
    /// Creates a new `Fence`.
    #[inline]
    pub fn new(device: Arc<Device>, create_info: FenceCreateInfo) -> Result<Fence, FenceError> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(device: &Device, create_info: &FenceCreateInfo) -> Result<(), FenceError> {
        let &FenceCreateInfo {
            signaled: _,
            export_handle_types,
            _ne: _,
        } = create_info;

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_fence)
            {
                return Err(FenceError::RequirementNotMet {
                    required_for: "`create_info.export_handle_types` is not empty",
                    requires_one_of: RequiresOneOf {
                        api_version: Some(Version::V1_1),
                        device_extensions: &["khr_external_fence"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkExportFenceCreateInfo-handleTypes-01446
            export_handle_types.validate_device(device)?;

            // VUID-VkExportFenceCreateInfo-handleTypes-01446
            for handle_type in export_handle_types.into_iter() {
                let external_fence_properties = unsafe {
                    device
                        .physical_device()
                        .external_fence_properties_unchecked(ExternalFenceInfo::handle_type(
                            handle_type,
                        ))
                };

                if !external_fence_properties.exportable {
                    return Err(FenceError::HandleTypeNotExportable { handle_type });
                }

                if !external_fence_properties
                    .compatible_handle_types
                    .contains(export_handle_types)
                {
                    return Err(FenceError::ExportHandleTypesNotCompatible);
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: FenceCreateInfo,
    ) -> Result<Fence, VulkanError> {
        let FenceCreateInfo {
            signaled,
            export_handle_types,
            _ne: _,
        } = create_info;

        let mut flags = ash::vk::FenceCreateFlags::empty();

        if signaled {
            flags |= ash::vk::FenceCreateFlags::SIGNALED;
        }

        let mut create_info_vk = ash::vk::FenceCreateInfo {
            flags,
            ..Default::default()
        };
        let mut export_fence_create_info_vk = None;

        if !export_handle_types.is_empty() {
            let _ = export_fence_create_info_vk.insert(ash::vk::ExportFenceCreateInfo {
                handle_types: export_handle_types.into(),
                ..Default::default()
            });
        }

        if let Some(info) = export_fence_create_info_vk.as_mut() {
            info.p_next = create_info_vk.p_next;
            create_info_vk.p_next = info as *const _ as *const _;
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_fence)(
                device.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;

            output.assume_init()
        };

        Ok(Fence {
            handle,
            device,
            id: Self::next_id(),
            must_put_in_pool: false,
            export_handle_types,
            state: Mutex::new(FenceState {
                is_signaled: signaled,
                ..Default::default()
            }),
        })
    }

    /// Takes a fence from the vulkano-provided fence pool.
    /// If the pool is empty, a new fence will be created.
    /// Upon `drop`, the fence is put back into the pool.
    ///
    /// For most applications, using the fence pool should be preferred,
    /// in order to avoid creating new fences every frame.
    #[inline]
    pub fn from_pool(device: Arc<Device>) -> Result<Fence, FenceError> {
        let handle = device.fence_pool().lock().pop();
        let fence = match handle {
            Some(handle) => {
                unsafe {
                    // Make sure the fence isn't signaled
                    let fns = device.fns();
                    (fns.v1_0.reset_fences)(device.handle(), 1, &handle)
                        .result()
                        .map_err(VulkanError::from)?;
                }

                Fence {
                    handle,
                    device,
                    id: Self::next_id(),
                    must_put_in_pool: true,
                    export_handle_types: ExternalFenceHandleTypes::empty(),
                    state: Mutex::new(Default::default()),
                }
            }
            None => {
                // Pool is empty, alloc new fence
                let mut fence = Fence::new(device, FenceCreateInfo::default())?;
                fence.must_put_in_pool = true;
                fence
            }
        };

        Ok(fence)
    }

    /// Creates a new `Fence` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Fence,
        create_info: FenceCreateInfo,
    ) -> Fence {
        let FenceCreateInfo {
            signaled,
            export_handle_types,
            _ne: _,
        } = create_info;

        Fence {
            handle,
            device,
            id: Self::next_id(),
            must_put_in_pool: false,
            export_handle_types,
            state: Mutex::new(FenceState {
                is_signaled: signaled,
                ..Default::default()
            }),
        }
    }

    /// Returns true if the fence is signaled.
    #[inline]
    pub fn is_signaled(&self) -> Result<bool, OomError> {
        let queue_to_signal = {
            let mut state = self.state();

            // If the fence is already signaled, or it's unsignaled but there's no queue that
            // could signal it, return the currently known value.
            if let Some(is_signaled) = state.is_signaled() {
                return Ok(is_signaled);
            }

            // We must ask Vulkan for the state.
            let result = unsafe {
                let fns = self.device.fns();
                (fns.v1_0.get_fence_status)(self.device.handle(), self.handle)
            };

            match result {
                ash::vk::Result::SUCCESS => unsafe { state.set_signaled() },
                ash::vk::Result::NOT_READY => return Ok(false),
                err => return Err(VulkanError::from(err).into()),
            }
        };

        // If we have a queue that we need to signal our status to,
        // do so now after the state lock is dropped, to avoid deadlocks.
        if let Some(queue) = queue_to_signal {
            unsafe {
                queue.with(|mut q| q.fence_signaled(self));
            }
        }

        Ok(true)
    }

    /// Waits until the fence is signaled, or at least until the timeout duration has elapsed.
    ///
    /// Returns `Ok` if the fence is now signaled. Returns `Err` if the timeout was reached instead.
    ///
    /// If you pass a duration of 0, then the function will return without blocking.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), FenceError> {
        let queue_to_signal = {
            let mut state = self.state.lock();

            // If the fence is already signaled, we don't need to wait.
            if state.is_signaled().unwrap_or(false) {
                return Ok(());
            }

            let timeout_ns = timeout.map_or(u64::MAX, |timeout| {
                timeout
                    .as_secs()
                    .saturating_mul(1_000_000_000)
                    .saturating_add(timeout.subsec_nanos() as u64)
            });

            let result = unsafe {
                let fns = self.device.fns();
                (fns.v1_0.wait_for_fences)(
                    self.device.handle(),
                    1,
                    &self.handle,
                    ash::vk::TRUE,
                    timeout_ns,
                )
            };

            match result {
                ash::vk::Result::SUCCESS => unsafe { state.set_signaled() },
                ash::vk::Result::TIMEOUT => return Err(FenceError::Timeout),
                err => return Err(VulkanError::from(err).into()),
            }
        };

        // If we have a queue that we need to signal our status to,
        // do so now after the state lock is dropped, to avoid deadlocks.
        if let Some(queue) = queue_to_signal {
            unsafe {
                queue.with(|mut q| q.fence_signaled(self));
            }
        }

        Ok(())
    }

    /// Waits for multiple fences at once.
    ///
    /// # Panics
    ///
    /// - Panics if not all fences belong to the same device.
    pub fn multi_wait<'a>(
        fences: impl IntoIterator<Item = &'a Fence>,
        timeout: Option<Duration>,
    ) -> Result<(), FenceError> {
        let fences: SmallVec<[_; 8]> = fences.into_iter().collect();
        Self::validate_multi_wait(&fences, timeout)?;

        unsafe { Self::multi_wait_unchecked(fences, timeout) }
    }

    fn validate_multi_wait(
        fences: &[&Fence],
        _timeout: Option<Duration>,
    ) -> Result<(), FenceError> {
        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;

        for fence in fences {
            // VUID-vkWaitForFences-pFences-parent
            assert_eq!(device, &fence.device);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn multi_wait_unchecked<'a>(
        fences: impl IntoIterator<Item = &'a Fence>,
        timeout: Option<Duration>,
    ) -> Result<(), FenceError> {
        let queues_to_signal: SmallVec<[_; 8]> = {
            let iter = fences.into_iter();
            let mut fences_vk: SmallVec<[_; 8]> = SmallVec::new();
            let mut fences: SmallVec<[_; 8]> = SmallVec::new();
            let mut states: SmallVec<[_; 8]> = SmallVec::new();

            for fence in iter {
                let state = fence.state.lock();

                // Skip the fences that are already signaled.
                if !state.is_signaled().unwrap_or(false) {
                    fences_vk.push(fence.handle);
                    fences.push(fence);
                    states.push(state);
                }
            }

            // VUID-vkWaitForFences-fenceCount-arraylength
            // If there are no fences, or all the fences are signaled, we don't need to wait.
            if fences_vk.is_empty() {
                return Ok(());
            }

            let device = &fences[0].device;
            let timeout_ns = timeout.map_or(u64::MAX, |timeout| {
                timeout
                    .as_secs()
                    .saturating_mul(1_000_000_000)
                    .saturating_add(timeout.subsec_nanos() as u64)
            });

            let result = {
                let fns = device.fns();
                (fns.v1_0.wait_for_fences)(
                    device.handle(),
                    fences_vk.len() as u32,
                    fences_vk.as_ptr(),
                    ash::vk::TRUE, // TODO: let the user choose false here?
                    timeout_ns,
                )
            };

            match result {
                ash::vk::Result::SUCCESS => fences
                    .into_iter()
                    .zip(&mut states)
                    .filter_map(|(fence, state)| state.set_signaled().map(|state| (state, fence)))
                    .collect(),
                ash::vk::Result::TIMEOUT => return Err(FenceError::Timeout),
                err => return Err(VulkanError::from(err).into()),
            }
        };

        // If we have queues that we need to signal our status to,
        // do so now after the state locks are dropped, to avoid deadlocks.
        for (queue, fence) in queues_to_signal {
            queue.with(|mut q| q.fence_signaled(fence));
        }

        Ok(())
    }

    /// Resets the fence.
    ///
    /// The fence must not be in use by a queue operation.
    #[inline]
    pub fn reset(&self) -> Result<(), FenceError> {
        let mut state = self.state.lock();
        self.validate_reset(&state)?;

        unsafe { Ok(self.reset_unchecked_locked(&mut state)?) }
    }

    fn validate_reset(&self, state: &FenceState) -> Result<(), FenceError> {
        // VUID-vkResetFences-pFences-01123
        if state.is_in_queue() {
            return Err(FenceError::InQueue);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn reset_unchecked(&self) -> Result<(), VulkanError> {
        let mut state = self.state.lock();

        self.reset_unchecked_locked(&mut state)
    }

    unsafe fn reset_unchecked_locked(&self, state: &mut FenceState) -> Result<(), VulkanError> {
        let fns = self.device.fns();
        (fns.v1_0.reset_fences)(self.device.handle(), 1, &self.handle)
            .result()
            .map_err(VulkanError::from)?;

        state.reset();

        Ok(())
    }

    /// Resets multiple fences at once.
    ///
    /// The fences must not be in use by a queue operation.
    ///
    /// # Panics
    ///
    /// - Panics if not all fences belong to the same device.
    pub fn multi_reset<'a>(fences: impl IntoIterator<Item = &'a Fence>) -> Result<(), FenceError> {
        let (fences, mut states): (SmallVec<[_; 8]>, SmallVec<[_; 8]>) = fences
            .into_iter()
            .map(|fence| {
                let state = fence.state.lock();
                (fence, state)
            })
            .unzip();
        Self::validate_multi_reset(&fences, &states)?;

        unsafe { Ok(Self::multi_reset_unchecked_locked(&fences, &mut states)?) }
    }

    fn validate_multi_reset(
        fences: &[&Fence],
        states: &[MutexGuard<'_, FenceState>],
    ) -> Result<(), FenceError> {
        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;

        for (fence, state) in fences.iter().zip(states) {
            // VUID-vkResetFences-pFences-parent
            assert_eq!(device, &fence.device);

            // VUID-vkResetFences-pFences-01123
            if state.is_in_queue() {
                return Err(FenceError::InQueue);
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn multi_reset_unchecked<'a>(
        fences: impl IntoIterator<Item = &'a Fence>,
    ) -> Result<(), VulkanError> {
        let (fences, mut states): (SmallVec<[_; 8]>, SmallVec<[_; 8]>) = fences
            .into_iter()
            .map(|fence| {
                let state = fence.state.lock();
                (fence, state)
            })
            .unzip();

        Self::multi_reset_unchecked_locked(&fences, &mut states)
    }

    unsafe fn multi_reset_unchecked_locked(
        fences: &[&Fence],
        states: &mut [MutexGuard<'_, FenceState>],
    ) -> Result<(), VulkanError> {
        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;
        let fences_vk: SmallVec<[_; 8]> = fences.iter().map(|fence| fence.handle).collect();

        let fns = device.fns();
        (fns.v1_0.reset_fences)(device.handle(), fences_vk.len() as u32, fences_vk.as_ptr())
            .result()
            .map_err(VulkanError::from)?;

        for state in states {
            state.reset();
        }

        Ok(())
    }

    /// Exports the fence into a POSIX file descriptor. The caller owns the returned `File`.
    ///
    /// The [`khr_external_fence_fd`](crate::device::DeviceExtensions::khr_external_fence_fd)
    /// extension must be enabled on the device.
    #[cfg(unix)]
    #[inline]
    pub fn export_fd(&self, handle_type: ExternalFenceHandleType) -> Result<File, FenceError> {
        let mut state = self.state.lock();
        self.validate_export_fd(handle_type, &state)?;

        unsafe { Ok(self.export_fd_unchecked_locked(handle_type, &mut state)?) }
    }

    #[cfg(unix)]
    fn validate_export_fd(
        &self,
        handle_type: ExternalFenceHandleType,
        state: &FenceState,
    ) -> Result<(), FenceError> {
        if !self.device.enabled_extensions().khr_external_fence_fd {
            return Err(FenceError::RequirementNotMet {
                required_for: "`Fence::export_fd`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_external_fence_fd"],
                    ..Default::default()
                },
            });
        }

        // VUID-VkFenceGetFdInfoKHR-handleType-parameter
        handle_type.validate_device(&self.device)?;

        // VUID-VkFenceGetFdInfoKHR-handleType-01453
        if !self.export_handle_types.intersects(handle_type.into()) {
            return Err(FenceError::HandleTypeNotEnabled);
        }

        // VUID-VkFenceGetFdInfoKHR-handleType-01454
        if handle_type.has_copy_transference()
            && !(state.is_signaled().unwrap_or(false) || state.is_in_queue())
        {
            return Err(FenceError::HandleTypeCopyNotSignaled);
        }

        // VUID-VkFenceGetFdInfoKHR-fence-01455
        if let Some(imported_handle_type) = state.current_import {
            match imported_handle_type {
                ImportType::SwapchainAcquire => {
                    return Err(FenceError::ImportedForSwapchainAcquire)
                }
                ImportType::ExternalFence(imported_handle_type) => {
                    let external_fence_properties = unsafe {
                        self.device
                            .physical_device()
                            .external_fence_properties_unchecked(ExternalFenceInfo::handle_type(
                                handle_type,
                            ))
                    };

                    if !external_fence_properties
                        .export_from_imported_handle_types
                        .intersects(imported_handle_type.into())
                    {
                        return Err(FenceError::ExportFromImportedNotSupported {
                            imported_handle_type,
                        });
                    }
                }
            }
        }

        // VUID-VkFenceGetFdInfoKHR-handleType-01456
        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueFd | ExternalFenceHandleType::SyncFd
        ) {
            return Err(FenceError::HandleTypeNotFd);
        }

        Ok(())
    }

    #[cfg(unix)]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn export_fd_unchecked(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<File, VulkanError> {
        let mut state = self.state.lock();
        self.export_fd_unchecked_locked(handle_type, &mut state)
    }

    #[cfg(unix)]
    unsafe fn export_fd_unchecked_locked(
        &self,
        handle_type: ExternalFenceHandleType,
        state: &mut FenceState,
    ) -> Result<File, VulkanError> {
        use std::os::unix::io::FromRawFd;

        let info_vk = ash::vk::FenceGetFdInfoKHR {
            fence: self.handle,
            handle_type: handle_type.into(),
            ..Default::default()
        };

        let mut output = MaybeUninit::uninit();
        let fns = self.device.fns();
        (fns.khr_external_fence_fd.get_fence_fd_khr)(
            self.device.handle(),
            &info_vk,
            output.as_mut_ptr(),
        )
        .result()
        .map_err(VulkanError::from)?;

        state.export(handle_type);

        Ok(File::from_raw_fd(output.assume_init()))
    }

    /// Exports the fence into a Win32 handle.
    ///
    /// The [`khr_external_fence_win32`](crate::device::DeviceExtensions::khr_external_fence_win32)
    /// extension must be enabled on the device.
    #[cfg(windows)]
    #[inline]
    pub fn export_win32_handle(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<*mut std::ffi::c_void, FenceError> {
        let mut state = self.state.lock();
        self.validate_export_win32_handle(handle_type, &state)?;

        unsafe { Ok(self.export_win32_handle_unchecked_locked(handle_type, &mut state)?) }
    }

    #[cfg(windows)]
    fn validate_export_win32_handle(
        &self,
        handle_type: ExternalFenceHandleType,
        state: &FenceState,
    ) -> Result<(), FenceError> {
        if !self.device.enabled_extensions().khr_external_fence_win32 {
            return Err(FenceError::RequirementNotMet {
                required_for: "`Fence::export_win32_handle`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_external_fence_win32"],
                    ..Default::default()
                },
            });
        }

        // VUID-VkFenceGetWin32HandleInfoKHR-handleType-parameter
        handle_type.validate_device(&self.device)?;

        // VUID-VkFenceGetWin32HandleInfoKHR-handleType-01448
        if !self.export_handle_types.intersects(handle_type.into()) {
            return Err(FenceError::HandleTypeNotEnabled);
        }

        // VUID-VkFenceGetWin32HandleInfoKHR-handleType-01449
        if matches!(handle_type, ExternalFenceHandleType::OpaqueWin32)
            && state.is_exported(handle_type)
        {
            return Err(FenceError::AlreadyExported);
        }

        // VUID-VkFenceGetWin32HandleInfoKHR-handleType-01451
        if handle_type.has_copy_transference()
            && !(state.is_signaled().unwrap_or(false) || state.is_in_queue())
        {
            return Err(FenceError::HandleTypeCopyNotSignaled);
        }

        // VUID-VkFenceGetWin32HandleInfoKHR-fence-01450
        if let Some(imported_handle_type) = state.current_import {
            match imported_handle_type {
                ImportType::SwapchainAcquire => {
                    return Err(FenceError::ImportedForSwapchainAcquire)
                }
                ImportType::ExternalFence(imported_handle_type) => {
                    let external_fence_properties = unsafe {
                        self.device
                            .physical_device()
                            .external_fence_properties_unchecked(ExternalFenceInfo::handle_type(
                                handle_type,
                            ))
                    };

                    if !external_fence_properties
                        .export_from_imported_handle_types
                        .intersects(imported_handle_type.into())
                    {
                        return Err(FenceError::ExportFromImportedNotSupported {
                            imported_handle_type,
                        });
                    }
                }
            }
        }

        // VUID-VkFenceGetWin32HandleInfoKHR-handleType-01452
        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueWin32 | ExternalFenceHandleType::OpaqueWin32Kmt
        ) {
            return Err(FenceError::HandleTypeNotWin32);
        }

        Ok(())
    }

    #[cfg(windows)]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn export_win32_handle_unchecked(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<*mut std::ffi::c_void, VulkanError> {
        let mut state = self.state.lock();
        self.export_win32_handle_unchecked_locked(handle_type, &mut state)
    }

    #[cfg(windows)]
    unsafe fn export_win32_handle_unchecked_locked(
        &self,
        handle_type: ExternalFenceHandleType,
        state: &mut FenceState,
    ) -> Result<*mut std::ffi::c_void, VulkanError> {
        let info_vk = ash::vk::FenceGetWin32HandleInfoKHR {
            fence: self.handle,
            handle_type: handle_type.into(),
            ..Default::default()
        };

        let mut output = MaybeUninit::uninit();
        let fns = self.device.fns();
        (fns.khr_external_fence_win32.get_fence_win32_handle_khr)(
            self.device.handle(),
            &info_vk,
            output.as_mut_ptr(),
        )
        .result()
        .map_err(VulkanError::from)?;

        state.export(handle_type);

        Ok(output.assume_init())
    }

    /// Imports a fence from a POSIX file descriptor.
    ///
    /// The [`khr_external_fence_fd`](crate::device::DeviceExtensions::khr_external_fence_fd)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - If in `import_fence_fd_info`, `handle_type` is `ExternalHandleType::OpaqueFd`,
    ///   then `file` must represent a fence that was exported from Vulkan or a compatible API,
    ///   with a driver and device UUID equal to those of the device that owns `self`.
    #[cfg(unix)]
    #[inline]
    pub unsafe fn import_fd(
        &self,
        import_fence_fd_info: ImportFenceFdInfo,
    ) -> Result<(), FenceError> {
        let mut state = self.state.lock();
        self.validate_import_fd(&import_fence_fd_info, &state)?;

        Ok(self.import_fd_unchecked_locked(import_fence_fd_info, &mut state)?)
    }

    #[cfg(unix)]
    fn validate_import_fd(
        &self,
        import_fence_fd_info: &ImportFenceFdInfo,
        state: &FenceState,
    ) -> Result<(), FenceError> {
        if !self.device.enabled_extensions().khr_external_fence_fd {
            return Err(FenceError::RequirementNotMet {
                required_for: "`Fence::import_fd`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_external_fence_fd"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkImportFenceFdKHR-fence-01463
        if state.is_in_queue() {
            return Err(FenceError::InQueue);
        }

        let &ImportFenceFdInfo {
            flags,
            handle_type,
            file: _,
            _ne: _,
        } = import_fence_fd_info;

        // VUID-VkImportFenceFdInfoKHR-flags-parameter
        flags.validate_device(&self.device)?;

        // VUID-VkImportFenceFdInfoKHR-handleType-parameter
        handle_type.validate_device(&self.device)?;

        // VUID-VkImportFenceFdInfoKHR-handleType-01464
        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueFd | ExternalFenceHandleType::SyncFd
        ) {
            return Err(FenceError::HandleTypeNotFd);
        }

        // VUID-VkImportFenceFdInfoKHR-fd-01541
        // Can't validate, therefore unsafe

        // VUID-VkImportFenceFdInfoKHR-handleType-07306
        if handle_type.has_copy_transference() && !flags.intersects(FenceImportFlags::TEMPORARY) {
            return Err(FenceError::HandletypeCopyNotTemporary);
        }

        Ok(())
    }

    #[cfg(unix)]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn import_fd_unchecked(
        &self,
        import_fence_fd_info: ImportFenceFdInfo,
    ) -> Result<(), VulkanError> {
        let mut state = self.state.lock();
        self.import_fd_unchecked_locked(import_fence_fd_info, &mut state)
    }

    #[cfg(unix)]
    unsafe fn import_fd_unchecked_locked(
        &self,
        import_fence_fd_info: ImportFenceFdInfo,
        state: &mut FenceState,
    ) -> Result<(), VulkanError> {
        use std::os::unix::io::IntoRawFd;

        let ImportFenceFdInfo {
            flags,
            handle_type,
            file,
            _ne: _,
        } = import_fence_fd_info;

        let info_vk = ash::vk::ImportFenceFdInfoKHR {
            fence: self.handle,
            flags: flags.into(),
            handle_type: handle_type.into(),
            fd: file.map_or(-1, |file| file.into_raw_fd()),
            ..Default::default()
        };

        let fns = self.device.fns();
        (fns.khr_external_fence_fd.import_fence_fd_khr)(self.device.handle(), &info_vk)
            .result()
            .map_err(VulkanError::from)?;

        state.import(handle_type, flags.intersects(FenceImportFlags::TEMPORARY));

        Ok(())
    }

    /// Imports a fence from a Win32 handle.
    ///
    /// The [`khr_external_fence_win32`](crate::device::DeviceExtensions::khr_external_fence_win32)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - In `import_fence_win32_handle_info`, `handle` must represent a fence that was exported
    ///   from Vulkan or a compatible API, with a driver and device UUID equal to those of the
    ///   device that owns `self`.
    #[cfg(windows)]
    #[inline]
    pub unsafe fn import_win32_handle(
        &self,
        import_fence_win32_handle_info: ImportFenceWin32HandleInfo,
    ) -> Result<(), FenceError> {
        let mut state = self.state.lock();
        self.validate_import_win32_handle(&import_fence_win32_handle_info, &state)?;

        Ok(self.import_win32_handle_unchecked_locked(import_fence_win32_handle_info, &mut state)?)
    }

    #[cfg(windows)]
    fn validate_import_win32_handle(
        &self,
        import_fence_win32_handle_info: &ImportFenceWin32HandleInfo,
        state: &FenceState,
    ) -> Result<(), FenceError> {
        if !self.device.enabled_extensions().khr_external_fence_win32 {
            return Err(FenceError::RequirementNotMet {
                required_for: "`Fence::import_win32_handle`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_external_fence_win32"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkImportFenceWin32HandleKHR-fence-04448
        if state.is_in_queue() {
            return Err(FenceError::InQueue);
        }

        let &ImportFenceWin32HandleInfo {
            flags,
            handle_type,
            handle: _,
            _ne: _,
        } = import_fence_win32_handle_info;

        // VUID-VkImportFenceWin32HandleInfoKHR-flags-parameter
        flags.validate_device(&self.device)?;

        // VUID-VkImportFenceWin32HandleInfoKHR-handleType-01457
        handle_type.validate_device(&self.device)?;

        // VUID-VkImportFenceWin32HandleInfoKHR-handleType-01457
        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueWin32 | ExternalFenceHandleType::OpaqueWin32Kmt
        ) {
            return Err(FenceError::HandleTypeNotWin32);
        }

        // VUID-VkImportFenceWin32HandleInfoKHR-handle-01539
        // Can't validate, therefore unsafe

        // VUID?
        if handle_type.has_copy_transference() && !flags.intersects(FenceImportFlags::TEMPORARY) {
            return Err(FenceError::HandletypeCopyNotTemporary);
        }

        Ok(())
    }

    #[cfg(windows)]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn import_win32_handle_unchecked(
        &self,
        import_fence_win32_handle_info: ImportFenceWin32HandleInfo,
    ) -> Result<(), VulkanError> {
        let mut state = self.state.lock();
        self.import_win32_handle_unchecked_locked(import_fence_win32_handle_info, &mut state)
    }

    #[cfg(windows)]
    unsafe fn import_win32_handle_unchecked_locked(
        &self,
        import_fence_win32_handle_info: ImportFenceWin32HandleInfo,
        state: &mut FenceState,
    ) -> Result<(), VulkanError> {
        let ImportFenceWin32HandleInfo {
            flags,
            handle_type,
            handle,
            _ne: _,
        } = import_fence_win32_handle_info;

        let info_vk = ash::vk::ImportFenceWin32HandleInfoKHR {
            fence: self.handle,
            flags: flags.into(),
            handle_type: handle_type.into(),
            handle,
            name: ptr::null(), // TODO: support?
            ..Default::default()
        };

        let fns = self.device.fns();
        (fns.khr_external_fence_win32.import_fence_win32_handle_khr)(
            self.device.handle(),
            &info_vk,
        )
        .result()
        .map_err(VulkanError::from)?;

        state.import(handle_type, flags.intersects(FenceImportFlags::TEMPORARY));

        Ok(())
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, FenceState> {
        self.state.lock()
    }

    // Shared by Fence and FenceSignalFuture
    pub(crate) fn poll_impl(&self, cx: &mut Context<'_>) -> Poll<Result<(), OomError>> {
        // Vulkan only allows polling of the fence status, so we have to use a spin future.
        // This is still better than blocking in async applications, since a smart-enough async engine
        // can choose to run some other tasks between probing this one.

        // Check if we are done without blocking
        match self.is_signaled() {
            Err(e) => return Poll::Ready(Err(e)),
            Ok(signalled) => {
                if signalled {
                    return Poll::Ready(Ok(()));
                }
            }
        }

        // Otherwise spin
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

impl Drop for Fence {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.must_put_in_pool {
                let raw_fence = self.handle;
                self.device.fence_pool().lock().push(raw_fence);
            } else {
                let fns = self.device.fns();
                (fns.v1_0.destroy_fence)(self.device.handle(), self.handle, ptr::null());
            }
        }
    }
}

impl Future for Fence {
    type Output = Result<(), OomError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.poll_impl(cx)
    }
}

unsafe impl VulkanObject for Fence {
    type Handle = ash::vk::Fence;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Fence {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(Fence);

#[derive(Debug, Default)]
pub(crate) struct FenceState {
    is_signaled: bool,
    pending_signal: Option<Weak<Queue>>,

    reference_exported: bool,
    exported_handle_types: ExternalFenceHandleTypes,
    current_import: Option<ImportType>,
    permanent_import: Option<ExternalFenceHandleType>,
}

impl FenceState {
    /// If the fence is not in a queue and has no external references, returns the current status.
    #[inline]
    fn is_signaled(&self) -> Option<bool> {
        // If either of these is true, we can't be certain of the status.
        if self.is_in_queue() || self.has_external_reference() {
            None
        } else {
            Some(self.is_signaled)
        }
    }

    #[inline]
    fn is_in_queue(&self) -> bool {
        self.pending_signal.is_some()
    }

    /// Returns whether there are any potential external references to the fence payload.
    /// That is, the fence has been exported by reference transference, or imported.
    #[inline]
    fn has_external_reference(&self) -> bool {
        self.reference_exported || self.current_import.is_some()
    }

    #[allow(dead_code)]
    #[inline]
    fn is_exported(&self, handle_type: ExternalFenceHandleType) -> bool {
        self.exported_handle_types.intersects(handle_type.into())
    }

    #[inline]
    pub(crate) unsafe fn add_queue_signal(&mut self, queue: &Arc<Queue>) {
        self.pending_signal = Some(Arc::downgrade(queue));
    }

    /// Called when a fence first discovers that it is signaled.
    /// Returns the queue that should be informed about it.
    #[inline]
    unsafe fn set_signaled(&mut self) -> Option<Arc<Queue>> {
        self.is_signaled = true;

        // Fences with external references can't be used to determine queue completion.
        if self.has_external_reference() {
            self.pending_signal = None;
            None
        } else {
            self.pending_signal.take().and_then(|queue| queue.upgrade())
        }
    }

    /// Called when a queue is unlocking resources.
    #[inline]
    pub(crate) unsafe fn set_signal_finished(&mut self) {
        self.is_signaled = true;
        self.pending_signal = None;
    }

    #[inline]
    unsafe fn reset(&mut self) {
        debug_assert!(!self.is_in_queue());
        self.current_import = self.permanent_import.map(Into::into);
        self.is_signaled = false;
    }

    #[allow(dead_code)]
    #[inline]
    unsafe fn export(&mut self, handle_type: ExternalFenceHandleType) {
        self.exported_handle_types |= handle_type.into();

        if handle_type.has_copy_transference() {
            self.reset();
        } else {
            self.reference_exported = true;
        }
    }

    #[allow(dead_code)]
    #[inline]
    unsafe fn import(&mut self, handle_type: ExternalFenceHandleType, temporary: bool) {
        debug_assert!(!self.is_in_queue());
        self.current_import = Some(handle_type.into());

        if !temporary {
            self.permanent_import = Some(handle_type);
        }
    }

    #[inline]
    pub(crate) unsafe fn import_swapchain_acquire(&mut self) {
        debug_assert!(!self.is_in_queue());
        self.current_import = Some(ImportType::SwapchainAcquire);
    }
}

#[derive(Clone, Copy, Debug)]
enum ImportType {
    SwapchainAcquire,
    ExternalFence(ExternalFenceHandleType),
}

impl From<ExternalFenceHandleType> for ImportType {
    #[inline]
    fn from(handle_type: ExternalFenceHandleType) -> Self {
        Self::ExternalFence(handle_type)
    }
}

/// Parameters to create a new `Fence`.
#[derive(Clone, Debug)]
pub struct FenceCreateInfo {
    /// Whether the fence should be created in the signaled state.
    ///
    /// The default value is `false`.
    pub signaled: bool,

    /// The handle types that can be exported from the fence.
    pub export_handle_types: ExternalFenceHandleTypes,

    pub _ne: crate::NonExhaustive,
}

impl Default for FenceCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            signaled: false,
            export_handle_types: ExternalFenceHandleTypes::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

vulkan_bitflags_enum! {
    #[non_exhaustive]
    /// A set of [`ExternalFenceHandleType`] values.
    ExternalFenceHandleTypes,

    /// The handle type used to export or import fences to/from an external source.
    ExternalFenceHandleType impl {
        /// Returns whether the given handle type has *copy transference* rather than *reference
        /// transference*.
        ///
        /// Imports of handles with copy transference must always be temporary. Exports of such
        /// handles must only occur if the fence is already signaled, or if there is a fence signal
        /// operation pending in a queue.
        #[inline]
        pub fn has_copy_transference(self) -> bool {
            // As defined by
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-fence-handletypes-win32
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-fence-handletypes-fd
            matches!(self, Self::SyncFd)
        }
    },

    = ExternalFenceHandleTypeFlags(u32);

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

    /// A POSIX file descriptor handle to a Linux Sync File or Android Fence object.
    ///
    /// This handle type has *copy transference*.
    SYNC_FD, SyncFd = SYNC_FD,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Additional parameters for a fence payload import.
    FenceImportFlags = FenceImportFlags(u32);

    /// The fence payload will be imported only temporarily, regardless of the permanence of the
    /// imported handle type.
    TEMPORARY = TEMPORARY,
}

#[cfg(unix)]
#[derive(Debug)]
pub struct ImportFenceFdInfo {
    /// Additional parameters for the import operation.
    ///
    /// If `handle_type` has *copy transference*, this must include the `temporary` flag.
    ///
    /// The default value is [`FenceImportFlags::empty()`].
    pub flags: FenceImportFlags,

    /// The handle type of `file`.
    ///
    /// There is no default value.
    pub handle_type: ExternalFenceHandleType,

    /// The file to import the fence from.
    ///
    /// If `handle_type` is `ExternalFenceHandleType::SyncFd`, then `file` can be `None`.
    /// Instead of an imported file descriptor, a dummy file descriptor `-1` is used,
    /// which represents a fence that is always signaled.
    ///
    /// The default value is `None`, which must be overridden if `handle_type` is not
    /// `ExternalFenceHandleType::SyncFd`.
    pub file: Option<File>,

    pub _ne: crate::NonExhaustive,
}

#[cfg(unix)]
impl ImportFenceFdInfo {
    /// Returns an `ImportFenceFdInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalFenceHandleType) -> Self {
        Self {
            flags: FenceImportFlags::empty(),
            handle_type,
            file: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[cfg(windows)]
#[derive(Debug)]
pub struct ImportFenceWin32HandleInfo {
    /// Additional parameters for the import operation.
    ///
    /// If `handle_type` has *copy transference*, this must include the `temporary` flag.
    ///
    /// The default value is [`FenceImportFlags::empty()`].
    pub flags: FenceImportFlags,

    /// The handle type of `handle`.
    ///
    /// There is no default value.
    pub handle_type: ExternalFenceHandleType,

    /// The file to import the fence from.
    ///
    /// The default value is `null`, which must be overridden.
    pub handle: *mut std::ffi::c_void,

    pub _ne: crate::NonExhaustive,
}

#[cfg(windows)]
impl ImportFenceWin32HandleInfo {
    /// Returns an `ImportFenceWin32HandleInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalFenceHandleType) -> Self {
        Self {
            flags: FenceImportFlags::empty(),
            handle_type,
            handle: ptr::null_mut(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The fence configuration to query in
/// [`PhysicalDevice::external_fence_properties`](crate::device::physical::PhysicalDevice::external_fence_properties).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExternalFenceInfo {
    /// The external handle type that will be used with the fence.
    pub handle_type: ExternalFenceHandleType,

    pub _ne: crate::NonExhaustive,
}

impl ExternalFenceInfo {
    /// Returns an `ExternalFenceInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalFenceHandleType) -> Self {
        Self {
            handle_type,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The properties for exporting or importing external handles, when a fence is created
/// with a specific configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExternalFenceProperties {
    /// Whether a handle can be exported to an external source with the queried
    /// external handle type.
    pub exportable: bool,

    /// Whether a handle can be imported from an external source with the queried
    /// external handle type.
    pub importable: bool,

    /// Which external handle types can be re-exported after the queried external handle type has
    /// been imported.
    pub export_from_imported_handle_types: ExternalFenceHandleTypes,

    /// Which external handle types can be enabled along with the queried external handle type
    /// when creating the fence.
    pub compatible_handle_types: ExternalFenceHandleTypes,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FenceError {
    /// Not enough memory available.
    OomError(OomError),

    /// The device has been lost.
    DeviceLost,

    /// The specified timeout wasn't long enough.
    Timeout,

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The provided handle type does not permit more than one export,
    /// and a handle of this type was already exported previously.
    AlreadyExported,

    /// The provided handle type cannot be exported from the current import handle type.
    ExportFromImportedNotSupported {
        imported_handle_type: ExternalFenceHandleType,
    },

    /// One of the export handle types is not compatible with the other provided handles.
    ExportHandleTypesNotCompatible,

    /// A handle type with copy transference was provided, but the fence is not signaled and there
    /// is no pending queue operation that will signal it.
    HandleTypeCopyNotSignaled,

    /// A handle type with copy transference was provided,
    /// but the `temporary` import flag was not set.
    HandletypeCopyNotTemporary,

    /// The provided export handle type was not set in `export_handle_types` when creating the
    /// fence.
    HandleTypeNotEnabled,

    /// Exporting is not supported for the provided handle type.
    HandleTypeNotExportable {
        handle_type: ExternalFenceHandleType,
    },

    /// The provided handle type is not a POSIX file descriptor handle.
    HandleTypeNotFd,

    /// The provided handle type is not a Win32 handle.
    HandleTypeNotWin32,

    /// The fence currently has a temporary import for a swapchain acquire operation.
    ImportedForSwapchainAcquire,

    /// The fence is currently in use by a queue.
    InQueue,
}

impl Error for FenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for FenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::DeviceLost => write!(f, "the device was lost"),
            Self::Timeout => write!(f, "the timeout has been reached"),
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
                "a handle type with copy transference was provided, but the fence is not signaled \
                and there is no pending queue operation that will signal it",
            ),
            Self::HandletypeCopyNotTemporary => write!(
                f,
                "a handle type with copy transference was provided, but the `temporary` \
                import flag was not set",
            ),
            Self::HandleTypeNotEnabled => write!(
                f,
                "the provided export handle type was not set in `export_handle_types` when \
                creating the fence",
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
            Self::ImportedForSwapchainAcquire => write!(
                f,
                "the fence currently has a temporary import for a swapchain acquire operation",
            ),
            Self::InQueue => write!(f, "the fence is currently in use by a queue"),
        }
    }
}

impl From<VulkanError> for FenceError {
    fn from(err: VulkanError) -> Self {
        match err {
            e @ VulkanError::OutOfHostMemory | e @ VulkanError::OutOfDeviceMemory => {
                Self::OomError(e.into())
            }
            VulkanError::DeviceLost => Self::DeviceLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for FenceError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<RequirementNotMet> for FenceError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        sync::fence::{Fence, FenceCreateInfo},
        VulkanObject,
    };
    use std::time::Duration;

    #[test]
    fn fence_create() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(device, Default::default()).unwrap();
        assert!(!fence.is_signaled().unwrap());
    }

    #[test]
    fn fence_create_signaled() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(
            device,
            FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(fence.is_signaled().unwrap());
    }

    #[test]
    fn fence_signaled_wait() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(
            device,
            FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )
        .unwrap();
        fence.wait(Some(Duration::new(0, 10))).unwrap();
    }

    #[test]
    fn fence_reset() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(
            device,
            FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )
        .unwrap();
        fence.reset().unwrap();
        assert!(!fence.is_signaled().unwrap());
    }

    #[test]
    fn multiwait_different_devices() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            let fence1 = Fence::new(
                device1.clone(),
                FenceCreateInfo {
                    signaled: true,
                    ..Default::default()
                },
            )
            .unwrap();
            let fence2 = Fence::new(
                device2.clone(),
                FenceCreateInfo {
                    signaled: true,
                    ..Default::default()
                },
            )
            .unwrap();

            let _ = Fence::multi_wait(
                [&fence1, &fence2].iter().cloned(),
                Some(Duration::new(0, 10)),
            );
        });
    }

    #[test]
    fn multireset_different_devices() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            let fence1 = Fence::new(
                device1.clone(),
                FenceCreateInfo {
                    signaled: true,
                    ..Default::default()
                },
            )
            .unwrap();
            let fence2 = Fence::new(
                device2.clone(),
                FenceCreateInfo {
                    signaled: true,
                    ..Default::default()
                },
            )
            .unwrap();

            let _ = Fence::multi_reset([&fence1, &fence2]);
        });
    }

    #[test]
    fn fence_pool() {
        let (device, _) = gfx_dev_and_queue!();

        assert_eq!(device.fence_pool().lock().len(), 0);
        let fence1_internal_obj = {
            let fence = Fence::from_pool(device.clone()).unwrap();
            assert_eq!(device.fence_pool().lock().len(), 0);
            fence.handle()
        };

        assert_eq!(device.fence_pool().lock().len(), 1);
        let fence2 = Fence::from_pool(device.clone()).unwrap();
        assert_eq!(device.fence_pool().lock().len(), 0);
        assert_eq!(fence2.handle(), fence1_internal_obj);
    }
}
