//! A fence provides synchronization between the device and the host, or between an external source
//! and the host.
//!
//! A fence has two states: **signaled** and **unsignaled**.
//!
//! The device can only perform one operation on a fence:
//! - A **fence signal operation** will put the fence into the signaled state.
//!
//! The host can poll a fence's status, wait for it to become signaled, or reset the fence back
//! to the unsignaled state.
//!
//! # Queue-to-host synchronization
//!
//! The primary use of a fence is to know when a queue operation has completed executing.
//! When adding a command to a queue, a fence can be provided with the command, to be signaled
//! when the operation finishes. You can check for a fence's current status by calling
//! `is_signaled`, `wait` or `await` on it. If the fence is found to be signaled, that means that
//! the queue has completed the operation that is associated with the fence, and all operations
//! that happened-before it have been completed as well.
//!
//! # Safety
//!
//! - There must never be more than one fence signal operation queued at any given time.
//! - The fence must be unsignaled at the time the function (for example [`submit`]) is called.
//!
//! [`submit`]: crate::device::QueueGuard::submit

use crate::{
    device::{physical::PhysicalDevice, Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_bitflags_enum},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use smallvec::SmallVec;
use std::fs::File;
use std::{
    future::Future,
    mem::MaybeUninit,
    num::NonZeroU64,
    pin::Pin,
    ptr,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

/// A two-state synchronization primitive that is signalled by the device and waited on by the host.
#[derive(Debug)]
pub struct Fence {
    handle: ash::vk::Fence,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    flags: FenceCreateFlags,
    export_handle_types: ExternalFenceHandleTypes,

    must_put_in_pool: bool,
}

impl Fence {
    /// Creates a new `Fence`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        create_info: FenceCreateInfo,
    ) -> Result<Fence, Validated<VulkanError>> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        create_info: &FenceCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: FenceCreateInfo,
    ) -> Result<Fence, VulkanError> {
        let FenceCreateInfo {
            flags,
            export_handle_types,
            _ne: _,
        } = create_info;

        let mut create_info_vk = ash::vk::FenceCreateInfo {
            flags: flags.into(),
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
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            flags,
            export_handle_types,

            must_put_in_pool: false,
        })
    }

    /// Takes a fence from the vulkano-provided fence pool.
    /// If the pool is empty, a new fence will be created.
    /// Upon `drop`, the fence is put back into the pool.
    ///
    /// For most applications, using the fence pool should be preferred,
    /// in order to avoid creating new fences every frame.
    #[inline]
    pub fn from_pool(device: Arc<Device>) -> Result<Fence, VulkanError> {
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
                    device: InstanceOwnedDebugWrapper(device),
                    id: Self::next_id(),

                    flags: FenceCreateFlags::empty(),
                    export_handle_types: ExternalFenceHandleTypes::empty(),

                    must_put_in_pool: true,
                }
            }
            None => {
                // Pool is empty, alloc new fence
                let mut fence =
                    unsafe { Fence::new_unchecked(device, FenceCreateInfo::default())? };
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
            flags,
            export_handle_types,
            _ne: _,
        } = create_info;

        Fence {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            flags,
            export_handle_types,

            must_put_in_pool: false,
        }
    }

    /// Returns the flags that the fence was created with.
    #[inline]
    pub fn flags(&self) -> FenceCreateFlags {
        self.flags
    }

    /// Returns the handle types that can be exported from the fence.
    #[inline]
    pub fn export_handle_types(&self) -> ExternalFenceHandleTypes {
        self.export_handle_types
    }

    /// Returns true if the fence is signaled.
    #[inline]
    pub fn is_signaled(&self) -> Result<bool, VulkanError> {
        let result = unsafe {
            let fns = self.device.fns();
            (fns.v1_0.get_fence_status)(self.device.handle(), self.handle)
        };

        match result {
            ash::vk::Result::SUCCESS => Ok(true),
            ash::vk::Result::NOT_READY => Ok(false),
            err => Err(VulkanError::from(err)),
        }
    }

    /// Waits until the fence is signaled, or at least until the timeout duration has elapsed.
    ///
    /// If you pass a duration of 0, then the function will return without blocking.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), VulkanError> {
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
            ash::vk::Result::SUCCESS => Ok(()),
            err => Err(VulkanError::from(err)),
        }
    }

    /// Waits for multiple fences at once.
    ///
    /// # Panics
    ///
    /// - Panics if not all fences belong to the same device.
    pub fn multi_wait<'a>(
        fences: impl IntoIterator<Item = &'a Fence>,
        timeout: Option<Duration>,
    ) -> Result<(), Validated<VulkanError>> {
        let fences: SmallVec<[_; 8]> = fences.into_iter().collect();
        Self::validate_multi_wait(&fences, timeout)?;

        unsafe { Ok(Self::multi_wait_unchecked(fences, timeout)?) }
    }

    fn validate_multi_wait(
        fences: &[&Fence],
        _timeout: Option<Duration>,
    ) -> Result<(), Box<ValidationError>> {
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
    ) -> Result<(), VulkanError> {
        let iter = fences.into_iter();
        let mut fences_vk: SmallVec<[_; 8]> = SmallVec::new();
        let mut fences: SmallVec<[_; 8]> = SmallVec::new();

        for fence in iter {
            fences_vk.push(fence.handle);
            fences.push(fence);
        }

        // VUID-vkWaitForFences-fenceCount-arraylength
        // If there are no fences, we don't need to wait.
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
            ash::vk::Result::SUCCESS => Ok(()),
            err => Err(VulkanError::from(err)),
        }
    }

    /// Resets the fence.
    ///
    /// # Safety
    ///
    /// - The fence must not be in use by the device.
    #[inline]
    pub unsafe fn reset(&self) -> Result<(), Validated<VulkanError>> {
        self.validate_reset()?;

        unsafe { Ok(self.reset_unchecked()?) }
    }

    fn validate_reset(&self) -> Result<(), Box<ValidationError>> {
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_unchecked(&self) -> Result<(), VulkanError> {
        let fns = self.device.fns();
        (fns.v1_0.reset_fences)(self.device.handle(), 1, &self.handle)
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Resets multiple fences at once.
    ///
    /// # Safety
    ///
    /// - The elements of `fences` must not be in use by the device.
    pub unsafe fn multi_reset<'a>(
        fences: impl IntoIterator<Item = &'a Fence>,
    ) -> Result<(), Validated<VulkanError>> {
        let fences: SmallVec<[_; 8]> = fences.into_iter().collect();
        Self::validate_multi_reset(&fences)?;

        unsafe { Ok(Self::multi_reset_unchecked(fences)?) }
    }

    fn validate_multi_reset(fences: &[&Fence]) -> Result<(), Box<ValidationError>> {
        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;

        for fence in fences {
            // VUID-vkResetFences-pFences-parent
            assert_eq!(device, &fence.device);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn multi_reset_unchecked<'a>(
        fences: impl IntoIterator<Item = &'a Fence>,
    ) -> Result<(), VulkanError> {
        let fences: SmallVec<[_; 8]> = fences.into_iter().collect();

        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;
        let fences_vk: SmallVec<[_; 8]> = fences.iter().map(|fence| fence.handle).collect();

        let fns = device.fns();
        (fns.v1_0.reset_fences)(device.handle(), fences_vk.len() as u32, fences_vk.as_ptr())
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Exports the fence into a POSIX file descriptor. The caller owns the returned `File`.
    ///
    /// The [`khr_external_fence_fd`](crate::device::DeviceExtensions::khr_external_fence_fd)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - If `handle_type` has copy transference, then the fence must be signaled, or a signal
    ///   operation on the fence must be pending.
    /// - The fence must not currently have an imported payload from a swapchain acquire operation.
    /// - If the fence has an imported payload, its handle type must allow re-exporting as
    ///   `handle_type`, as returned by [`PhysicalDevice::external_fence_properties`].
    #[inline]
    pub unsafe fn export_fd(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<File, Validated<VulkanError>> {
        self.validate_export_fd(handle_type)?;

        unsafe { Ok(self.export_fd_unchecked(handle_type)?) }
    }

    fn validate_export_fd(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().khr_external_fence_fd {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_fence_fd",
                )])]),
                ..Default::default()
            }));
        }

        handle_type.validate_device(&self.device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkFenceGetFdInfoKHR-handleType-parameter"])
        })?;

        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueFd | ExternalFenceHandleType::SyncFd
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalFenceHandleType::OpaqueFd` or \
                    `ExternalFenceHandleType::SyncFd`"
                    .into(),
                vuids: &["VUID-VkFenceGetFdInfoKHR-handleType-01456"],
                ..Default::default()
            }));
        }

        if !self.export_handle_types.intersects(handle_type.into()) {
            return Err(Box::new(ValidationError {
                problem: "`self.export_handle_types()` does not contain `handle_type`".into(),
                vuids: &["VUID-VkFenceGetFdInfoKHR-handleType-01453"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn export_fd_unchecked(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<File, VulkanError> {
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

        #[cfg(unix)]
        {
            use std::os::unix::io::FromRawFd;
            Ok(File::from_raw_fd(output.assume_init()))
        }

        #[cfg(not(unix))]
        {
            let _ = output;
            unreachable!("`khr_external_fence_fd` was somehow enabled on a non-Unix system");
        }
    }

    /// Exports the fence into a Win32 handle.
    ///
    /// The [`khr_external_fence_win32`](crate::device::DeviceExtensions::khr_external_fence_win32)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - If `handle_type` has copy transference, then the fence must be signaled, or a signal
    ///   operation on the fence must be pending.
    /// - The fence must not currently have an imported payload from a swapchain acquire operation.
    /// - If the fence has an imported payload, its handle type must allow re-exporting as
    ///   `handle_type`, as returned by [`PhysicalDevice::external_fence_properties`].
    /// - If `handle_type` is `ExternalFenceHandleType::OpaqueWin32`, then a handle of this type
    ///   must not have been already exported from this fence.
    #[inline]
    pub fn export_win32_handle(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<*mut std::ffi::c_void, Validated<VulkanError>> {
        self.validate_export_win32_handle(handle_type)?;

        unsafe { Ok(self.export_win32_handle_unchecked(handle_type)?) }
    }

    fn validate_export_win32_handle(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().khr_external_fence_win32 {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_fence_win32",
                )])]),
                ..Default::default()
            }));
        }

        handle_type.validate_device(&self.device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkFenceGetWin32HandleInfoKHR-handleType-parameter"])
        })?;

        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueWin32 | ExternalFenceHandleType::OpaqueWin32Kmt
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalFenceHandleType::OpaqueWin32` or \
                    `ExternalFenceHandleType::OpaqueWin32Kmt`"
                    .into(),
                vuids: &["VUID-VkFenceGetWin32HandleInfoKHR-handleType-01452"],
                ..Default::default()
            }));
        }

        if !self.export_handle_types.intersects(handle_type.into()) {
            return Err(Box::new(ValidationError {
                problem: "`self.export_handle_types()` does not contain `handle_type`".into(),
                vuids: &["VUID-VkFenceGetWin32HandleInfoKHR-handleType-01448"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn export_win32_handle_unchecked(
        &self,
        handle_type: ExternalFenceHandleType,
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

        Ok(output.assume_init())
    }

    /// Imports a fence from a POSIX file descriptor.
    ///
    /// The [`khr_external_fence_fd`](crate::device::DeviceExtensions::khr_external_fence_fd)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - The fence must not be in use by the device.
    /// - If in `import_fence_fd_info`, `handle_type` is `ExternalHandleType::OpaqueFd`,
    ///   then `file` must represent a fence that was exported from Vulkan or a compatible API,
    ///   with a driver and device UUID equal to those of the device that owns `self`.
    #[inline]
    pub unsafe fn import_fd(
        &self,
        import_fence_fd_info: ImportFenceFdInfo,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_import_fd(&import_fence_fd_info)?;

        Ok(self.import_fd_unchecked(import_fence_fd_info)?)
    }

    fn validate_import_fd(
        &self,
        import_fence_fd_info: &ImportFenceFdInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().khr_external_fence_fd {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_fence_fd",
                )])]),
                ..Default::default()
            }));
        }

        import_fence_fd_info
            .validate(&self.device)
            .map_err(|err| err.add_context("import_fence_fd_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn import_fd_unchecked(
        &self,
        import_fence_fd_info: ImportFenceFdInfo,
    ) -> Result<(), VulkanError> {
        let ImportFenceFdInfo {
            flags,
            handle_type,
            file,
            _ne: _,
        } = import_fence_fd_info;

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

        let info_vk = ash::vk::ImportFenceFdInfoKHR {
            fence: self.handle,
            flags: flags.into(),
            handle_type: handle_type.into(),
            fd,
            ..Default::default()
        };

        let fns = self.device.fns();
        (fns.khr_external_fence_fd.import_fence_fd_khr)(self.device.handle(), &info_vk)
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Imports a fence from a Win32 handle.
    ///
    /// The [`khr_external_fence_win32`](crate::device::DeviceExtensions::khr_external_fence_win32)
    /// extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - The fence must not be in use by the device.
    /// - In `import_fence_win32_handle_info`, `handle` must represent a fence that was exported
    ///   from Vulkan or a compatible API, with a driver and device UUID equal to those of the
    ///   device that owns `self`.
    #[inline]
    pub unsafe fn import_win32_handle(
        &self,
        import_fence_win32_handle_info: ImportFenceWin32HandleInfo,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_import_win32_handle(&import_fence_win32_handle_info)?;

        Ok(self.import_win32_handle_unchecked(import_fence_win32_handle_info)?)
    }

    fn validate_import_win32_handle(
        &self,
        import_fence_win32_handle_info: &ImportFenceWin32HandleInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().khr_external_fence_win32 {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_fence_win32",
                )])]),
                ..Default::default()
            }));
        }

        import_fence_win32_handle_info
            .validate(&self.device)
            .map_err(|err| err.add_context("import_fence_win32_handle_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn import_win32_handle_unchecked(
        &self,
        import_fence_win32_handle_info: ImportFenceWin32HandleInfo,
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

        Ok(())
    }

    // Shared by Fence and FenceSignalFuture
    pub(crate) fn poll_impl(&self, cx: &mut Context<'_>) -> Poll<Result<(), VulkanError>> {
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
    type Output = Result<(), VulkanError>;

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

/// Parameters to create a new `Fence`.
#[derive(Clone, Debug)]
pub struct FenceCreateInfo {
    /// Additional properties of the fence.
    ///
    /// The default value is empty.
    pub flags: FenceCreateFlags,

    /// The handle types that can be exported from the fence.
    pub export_handle_types: ExternalFenceHandleTypes,

    pub _ne: crate::NonExhaustive,
}

impl Default for FenceCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: FenceCreateFlags::empty(),
            export_handle_types: ExternalFenceHandleTypes::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl FenceCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            export_handle_types,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkFenceCreateInfo-flags-parameter"])
        })?;

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_fence)
            {
                return Err(Box::new(ValidationError {
                    context: "export_handle_types".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_external_fence")]),
                    ]),
                    ..Default::default()
                }));
            }

            export_handle_types.validate_device(device).map_err(|err| {
                err.add_context("export_handle_types")
                    .set_vuids(&["VUID-VkExportFenceCreateInfo-handleTypes-parameter"])
            })?;

            for handle_type in export_handle_types.into_iter() {
                let external_fence_properties = unsafe {
                    device
                        .physical_device()
                        .external_fence_properties_unchecked(ExternalFenceInfo::handle_type(
                            handle_type,
                        ))
                };

                if !external_fence_properties.exportable {
                    return Err(Box::new(ValidationError {
                        context: "export_handle_types".into(),
                        problem: format!(
                            "the handle type `ExternalFenceHandleTypes::{:?}` is not exportable, \
                            as returned by `PhysicalDevice::external_fence_properties`",
                            ExternalFenceHandleTypes::from(handle_type)
                        )
                        .into(),
                        vuids: &["VUID-VkExportFenceCreateInfo-handleTypes-01446"],
                        ..Default::default()
                    }));
                }

                if !external_fence_properties
                    .compatible_handle_types
                    .contains(export_handle_types)
                {
                    return Err(Box::new(ValidationError {
                        context: "export_handle_types".into(),
                        problem: format!(
                            "the handle type `ExternalFenceHandleTypes::{:?}` is not compatible \
                            with the other specified handle types, as returned by \
                            `PhysicalDevice::external_fence_properties`",
                            ExternalFenceHandleTypes::from(handle_type)
                        )
                        .into(),
                        vuids: &["VUID-VkExportFenceCreateInfo-handleTypes-01446"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a fence.
    FenceCreateFlags = FenceCreateFlags(u32);

    /// Creates the fence in the signaled state.
    SIGNALED = SIGNALED,
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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            handle_type,
            file: _,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkImportFenceFdInfoKHR-flags-parameter"])
        })?;

        handle_type.validate_device(device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkImportFenceFdInfoKHR-handleType-parameter"])
        })?;

        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueFd | ExternalFenceHandleType::SyncFd
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalFenceHandleType::OpaqueFd` or \
                    `ExternalFenceHandleType::SyncFd`"
                    .into(),
                vuids: &["VUID-VkImportFenceFdInfoKHR-handleType-01464"],
                ..Default::default()
            }));
        }

        // VUID-VkImportFenceFdInfoKHR-fd-01541
        // Can't validate, therefore unsafe

        if handle_type.has_copy_transference() && !flags.intersects(FenceImportFlags::TEMPORARY) {
            return Err(Box::new(ValidationError {
                problem: "`handle_type` has copy transference, but \
                    `flags` does not contain `FenceImportFlags::TEMPORARY`"
                    .into(),
                vuids: &["VUID-VkImportFenceFdInfoKHR-handleType-07306"],
                ..Default::default()
            }));
        }

        Ok(())
    }
}

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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            handle_type,
            handle: _,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkImportFenceWin32HandleInfoKHR-flags-parameter"])
        })?;

        handle_type.validate_device(device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkImportFenceWin32HandleInfoKHR-handleType-01457"])
        })?;

        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueWin32 | ExternalFenceHandleType::OpaqueWin32Kmt
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalFenceHandleType::OpaqueWin32` or \
                    `ExternalFenceHandleType::OpaqueWin32Kmt`"
                    .into(),
                vuids: &["VUID-VkImportFenceWin32HandleInfoKHR-handleType-01457"],
                ..Default::default()
            }));
        }

        // VUID-VkImportFenceWin32HandleInfoKHR-handle-01539
        // Can't validate, therefore unsafe

        if handle_type.has_copy_transference() && !flags.intersects(FenceImportFlags::TEMPORARY) {
            return Err(Box::new(ValidationError {
                problem: "`handle_type` has copy transference, but \
                    `flags` does not contain `FenceImportFlags::TEMPORARY`"
                    .into(),
                // vuids?
                ..Default::default()
            }));
        }

        Ok(())
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
                    .set_vuids(&["VUID-VkPhysicalDeviceExternalFenceInfo-handleType-parameter"])
            })?;

        Ok(())
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

#[cfg(test)]
mod tests {
    use crate::{
        sync::fence::{Fence, FenceCreateFlags, FenceCreateInfo},
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
                flags: FenceCreateFlags::SIGNALED,
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
                flags: FenceCreateFlags::SIGNALED,
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
                flags: FenceCreateFlags::SIGNALED,
                ..Default::default()
            },
        )
        .unwrap();

        unsafe {
            fence.reset().unwrap();
        }

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
                    flags: FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
            )
            .unwrap();
            let fence2 = Fence::new(
                device2.clone(),
                FenceCreateInfo {
                    flags: FenceCreateFlags::SIGNALED,
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
                    flags: FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
            )
            .unwrap();
            let fence2 = Fence::new(
                device2.clone(),
                FenceCreateInfo {
                    flags: FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
            )
            .unwrap();

            let _ = unsafe { Fence::multi_reset([&fence1, &fence2]) };
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
