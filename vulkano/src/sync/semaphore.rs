//! A semaphore provides synchronization between multiple queues, with non-command buffer
//! commands on the same queue, or between the device and an external source.
//!
//! Semaphores come in two types: **binary** and **timeline** semaphores.
//!
//! # Binary semaphores
//!
//! A binary semaphore has two states: **signaled** and **unsignaled**.
//! Only the device can perform operations on a binary semaphore,
//! the host cannot perform any operations on it.
//!
//! Two operations can be performed on a binary semaphore:
//! - A **semaphore signal operation** will put the semaphore into the signaled state.
//! - A **semaphore wait operation** will block execution of the operation it is associated with,
//!   as long as the semaphore is in the unsignaled state. Once the semaphore is in the signaled
//!   state, the semaphore is put back in the unsignaled state and execution continues.
//!
//! Binary semaphore signals and waits must always occur in pairs: one signal operation is paired
//! with one wait operation. If a binary semaphore is signaled without waiting for it, it stays in
//! the signaled state until it is waited for, or destroyed.
//!
//! # Timeline semaphores
//!
//! Also called *counting semaphore* in literature, its state is an integer counter value.
//! Timeline semaphores cannot be used in swapchain-related commands,
//! binary semaphores and fences must be used.
//!
//! Both the device and the host can perform the same two operations on a timeline semaphore:
//! - A **semaphore signal operation** will set the semaphore counter value to a specified value.
//! - A **semaphore wait operation** will block execution of the operation it is associated with,
//!   as long as the semaphore's counter value is less than a specified threshold value. Once the
//!   semaphore's counter value is equal to or greater than the threshold, execution continues.
//!   Unlike with binary semaphores, waiting does not alter the state of a timeline semaphore, so
//!   multiple operations can wait for the same semaphore value.
//!
//! Additionally, the host can query the current counter value of a timeline semaphore.
//!
//! If the device signals a timeline semaphore, and the host waits for it, then it can be used
//! in ways similar to a [fence], to signal to the host that the device has completed an
//! operation.
//!
//! # Safety
//!
//! For binary semaphores:
//! - When a semaphore signal operation is executed, the semaphore must be in the unsignaled state.
//!   In other words, the same semaphore cannot be signalled by multiple commands; there must
//!   always be a wait operation in between them.
//! - There must never be more than one semaphore wait operation executing on the same semaphore at
//!   the same time.
//! - When a semaphore wait operation is queued as part of a command, the semaphore must already be
//!   in the signaled state, or the signal operation that it waits for must have been queued
//!   previously (as part of a previous command, or an earlier batch within the same command).
//!
//! For timeline semaphores:
//! - When a semaphore signal operation is executed, the new counter value of the semaphore must be
//!   greater than its current value, and less than the value of any pending signal operations on
//!   that semaphore.
//! - If an operation both waits on and signals the same semaphore, the signaled value must be
//!   greater than the waited value.
//! - At any given time, the difference between the current semaphore counter value, and the value
//!   of any outstanding signal or wait operations on that semaphore, must not be greater than the
//!   [`max_timeline_semaphore_value_difference`] device limit.
//!
//! [fence]: crate::sync::fence
//! [`max_timeline_semaphore_value_difference`]: crate::device::DeviceProperties::max_timeline_semaphore_value_difference

use crate::{
    device::{physical::PhysicalDevice, Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_bitflags_enum, vulkan_enum},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use smallvec::SmallVec;
use std::{fs::File, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc, time::Duration};

/// Used to provide synchronization between command buffers during their execution.
///
/// It is similar to a fence, except that it is purely on the GPU side. The CPU can't query a
/// semaphore's status or wait for it to be signaled.
#[derive(Debug)]
pub struct Semaphore {
    handle: ash::vk::Semaphore,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    semaphore_type: SemaphoreType,
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
        let &SemaphoreCreateInfo {
            semaphore_type,
            initial_value,
            export_handle_types,
            _ne: _,
        } = &create_info;

        let mut create_info_vk = ash::vk::SemaphoreCreateInfo {
            flags: ash::vk::SemaphoreCreateFlags::empty(),
            ..Default::default()
        };
        let mut semaphore_type_create_info_vk = None;
        let mut export_semaphore_create_info_vk = None;

        if semaphore_type != SemaphoreType::Binary {
            let next = semaphore_type_create_info_vk.insert(ash::vk::SemaphoreTypeCreateInfo {
                semaphore_type: semaphore_type.into(),
                initial_value,
                ..Default::default()
            });

            next.p_next = create_info_vk.p_next;
            create_info_vk.p_next = <*const _>::cast(next);
        }

        if !export_handle_types.is_empty() {
            let next = export_semaphore_create_info_vk.insert(ash::vk::ExportSemaphoreCreateInfo {
                handle_types: export_handle_types.into(),
                ..Default::default()
            });

            next.p_next = create_info_vk.p_next;
            create_info_vk.p_next = <*const _>::cast(next);
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

                semaphore_type: SemaphoreType::Binary,
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
            semaphore_type,
            initial_value: _,
            export_handle_types,
            _ne: _,
        } = create_info;

        Semaphore {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            semaphore_type,
            export_handle_types,

            must_put_in_pool: false,
        }
    }

    /// Returns the type of the semaphore.
    #[inline]
    pub fn semaphore_type(&self) -> SemaphoreType {
        self.semaphore_type
    }

    /// Returns the handle types that can be exported from the semaphore.
    #[inline]
    pub fn export_handle_types(&self) -> ExternalSemaphoreHandleTypes {
        self.export_handle_types
    }

    /// If `self` is a timeline semaphore, returns the current counter value of the semaphore.
    ///
    /// The returned value may be immediately out of date, if a signal operation on the semaphore
    /// is pending on the device.
    #[inline]
    pub fn counter_value(&self) -> Result<u64, Validated<VulkanError>> {
        self.validate_counter_value()?;

        unsafe { Ok(self.counter_value_unchecked()?) }
    }

    fn validate_counter_value(&self) -> Result<(), Box<ValidationError>> {
        if self.semaphore_type != SemaphoreType::Timeline {
            return Err(Box::new(ValidationError {
                context: "self.semaphore_type()".into(),
                problem: "is not `SemaphoreType::Timeline`".into(),
                vuids: &["VUID-vkGetSemaphoreCounterValue-semaphore-03255"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn counter_value_unchecked(&self) -> Result<u64, VulkanError> {
        let mut output = MaybeUninit::uninit();
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_2 {
            (fns.v1_2.get_semaphore_counter_value)(
                self.device.handle(),
                self.handle,
                output.as_mut_ptr(),
            )
        } else {
            (fns.khr_timeline_semaphore.get_semaphore_counter_value_khr)(
                self.device.handle(),
                self.handle,
                output.as_mut_ptr(),
            )
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(output.assume_init())
    }

    /// If `self` is a timeline semaphore, performs a signal operation on the semaphore, setting
    /// the new counter value to `value`.
    ///
    /// # Safety
    ///
    /// - The safety requirements for semaphores, as detailed in the module documentation, must be
    ///   followed.
    #[inline]
    pub unsafe fn signal(
        &self,
        signal_info: SemaphoreSignalInfo,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_signal(&signal_info)?;

        Ok(self.signal_unchecked(signal_info)?)
    }

    fn validate_signal(
        &self,
        signal_info: &SemaphoreSignalInfo,
    ) -> Result<(), Box<ValidationError>> {
        if self.semaphore_type != SemaphoreType::Timeline {
            return Err(Box::new(ValidationError {
                context: "self.semaphore_type()".into(),
                problem: "is not `SemaphoreType::Timeline`".into(),
                vuids: &["VUID-VkSemaphoreSignalInfo-semaphore-03257"],
                ..Default::default()
            }));
        }

        signal_info.validate(&self.device)?;

        // unsafe
        // VUID-VkSemaphoreSignalInfo-value-03258
        // VUID-VkSemaphoreSignalInfo-value-03259
        // VUID-VkSemaphoreSignalInfo-value-03260

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn signal_unchecked(
        &self,
        signal_info: SemaphoreSignalInfo,
    ) -> Result<(), VulkanError> {
        let &SemaphoreSignalInfo { value, _ne: _ } = &signal_info;

        let signal_info_vk = ash::vk::SemaphoreSignalInfo {
            semaphore: self.handle,
            value,
            ..Default::default()
        };

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_2 {
            (fns.v1_2.signal_semaphore)(self.device.handle(), &signal_info_vk)
        } else {
            (fns.khr_timeline_semaphore.signal_semaphore_khr)(self.device.handle(), &signal_info_vk)
        }
        .result()
        .map_err(VulkanError::from)
    }

    /// If `self` is a timeline semaphore, performs a wait operation on the semaphore, blocking
    /// until the counter value is equal to or greater than `wait_info.value`.
    #[inline]
    pub fn wait(
        &self,
        wait_info: SemaphoreWaitInfo,
        timeout: Option<Duration>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_wait(&wait_info, timeout)?;

        unsafe { Ok(self.wait_unchecked(wait_info, timeout)?) }
    }

    fn validate_wait(
        &self,
        wait_info: &SemaphoreWaitInfo,
        timeout: Option<Duration>,
    ) -> Result<(), Box<ValidationError>> {
        if self.semaphore_type != SemaphoreType::Timeline {
            return Err(Box::new(ValidationError {
                context: "self.semaphore_type()".into(),
                problem: "is not `SemaphoreType::Timeline`".into(),
                vuids: &["VUID-VkSemaphoreWaitInfo-pSemaphores-03256"],
                ..Default::default()
            }));
        }

        wait_info
            .validate(&self.device)
            .map_err(|err| err.add_context("wait_info"))?;

        if let Some(timeout) = timeout {
            if timeout.as_nanos() >= u64::MAX as u128 {
                return Err(Box::new(ValidationError {
                    context: "timeout".into(),
                    problem: "is not less than `u64::MAX` nanoseconds".into(),
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wait_unchecked(
        &self,
        wait_info: SemaphoreWaitInfo,
        timeout: Option<Duration>,
    ) -> Result<(), VulkanError> {
        let &SemaphoreWaitInfo {
            flags,
            value,
            _ne: _,
        } = &wait_info;

        let semaphores_vk = [self.handle];
        let values_vk = [value];

        let wait_info_vk = ash::vk::SemaphoreWaitInfo {
            flags: flags.into(),
            semaphore_count: 1,
            p_semaphores: semaphores_vk.as_ptr(),
            p_values: values_vk.as_ptr(),
            ..Default::default()
        };

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_2 {
            (fns.v1_2.wait_semaphores)(
                self.device.handle(),
                &wait_info_vk,
                timeout.map_or(u64::MAX, |duration| {
                    u64::try_from(duration.as_nanos()).unwrap()
                }),
            )
        } else {
            (fns.khr_timeline_semaphore.wait_semaphores_khr)(
                self.device.handle(),
                &wait_info_vk,
                timeout.map_or(u64::MAX, |duration| {
                    u64::try_from(duration.as_nanos()).unwrap()
                }),
            )
        }
        .result()
        .map_err(VulkanError::from)
    }

    /// Waits for multiple timeline semaphores at once.
    ///
    /// # Panics
    ///
    /// - Panics if not all semaphores belong to the same device.
    #[inline]
    pub fn wait_multiple(
        wait_info: SemaphoreWaitMultipleInfo,
        timeout: Option<Duration>,
    ) -> Result<(), Validated<VulkanError>> {
        Self::validate_wait_multiple(&wait_info, timeout)?;

        unsafe { Ok(Self::wait_multiple_unchecked(wait_info, timeout)?) }
    }

    fn validate_wait_multiple(
        wait_info: &SemaphoreWaitMultipleInfo,
        timeout: Option<Duration>,
    ) -> Result<(), Box<ValidationError>> {
        if let Some(timeout) = timeout {
            if timeout.as_nanos() >= u64::MAX as u128 {
                return Err(Box::new(ValidationError {
                    context: "timeout".into(),
                    problem: "is not less than `u64::MAX` nanoseconds".into(),
                    ..Default::default()
                }));
            }
        }

        if wait_info.semaphores.is_empty() {
            return Ok(());
        }

        let device = &wait_info.semaphores[0].semaphore.device;
        wait_info
            .validate(device)
            .map_err(|err| err.add_context("wait_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wait_multiple_unchecked(
        wait_info: SemaphoreWaitMultipleInfo,
        timeout: Option<Duration>,
    ) -> Result<(), VulkanError> {
        let &SemaphoreWaitMultipleInfo {
            flags,
            ref semaphores,
            _ne: _,
        } = &wait_info;

        if semaphores.is_empty() {
            return Ok(());
        }

        let mut semaphores_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(semaphores.len());
        let mut values_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(semaphores.len());

        for value_info in semaphores {
            let &SemaphoreWaitValueInfo {
                ref semaphore,
                value,
                _ne: _,
            } = value_info;

            semaphores_vk.push(semaphore.handle);
            values_vk.push(value);
        }

        let wait_info_vk = ash::vk::SemaphoreWaitInfo {
            flags: flags.into(),
            semaphore_count: semaphores_vk.len() as u32,
            p_semaphores: semaphores_vk.as_ptr(),
            p_values: values_vk.as_ptr(),
            ..Default::default()
        };

        let device = &semaphores[0].semaphore.device;
        let fns = device.fns();

        if device.api_version() >= Version::V1_2 {
            (fns.v1_2.wait_semaphores)(
                device.handle(),
                &wait_info_vk,
                timeout.map_or(u64::MAX, |duration| {
                    u64::try_from(duration.as_nanos()).unwrap()
                }),
            )
        } else {
            (fns.khr_timeline_semaphore.wait_semaphores_khr)(
                device.handle(),
                &wait_info_vk,
                timeout.map_or(u64::MAX, |duration| {
                    u64::try_from(duration.as_nanos()).unwrap()
                }),
            )
        }
        .result()
        .map_err(VulkanError::from)
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

        if handle_type.has_copy_transference() && self.semaphore_type != SemaphoreType::Binary {
            return Err(Box::new(ValidationError {
                problem: "`handle_type` has copy transference, but \
                    `self.semaphore_type()` is not `SemaphoreType::Binary`"
                    .into(),
                vuids: &["VUID-VkSemaphoreGetFdInfoKHR-handleType-03253"],
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
    ) -> Result<ash::vk::HANDLE, Validated<VulkanError>> {
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
    ) -> Result<ash::vk::HANDLE, VulkanError> {
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

        if self.semaphore_type != SemaphoreType::Binary {
            return Err(Box::new(ValidationError {
                context: "self.semaphore_type()".into(),
                problem: "is not `SemaphoreType::Binary`".into(),
                vuids: &["VUID-VkSemaphoreGetZirconHandleInfoFUCHSIA-semaphore-04763"],
                ..Default::default()
            }));
        }

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
    /// - If in `import_semaphore_fd_info`, `handle_type` is `ExternalHandleType::OpaqueFd`, then
    ///   `file` must represent a binary semaphore that was exported from Vulkan or a compatible
    ///   API, with a driver and device UUID equal to those of the device that owns `self`.
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

        let &ImportSemaphoreFdInfo {
            flags,
            handle_type: _,
            file: _,
            _ne: _,
        } = import_semaphore_fd_info;

        if self.semaphore_type == SemaphoreType::Timeline
            && flags.intersects(SemaphoreImportFlags::TEMPORARY)
        {
            return Err(Box::new(ValidationError {
                problem: "`self.semaphore_type()` is `SemaphoreType::Timeline`, but \
                    `import_semaphore_fd_info.flags` contains \
                    `SemaphoreImportFlags::TEMPORARY`"
                    .into(),
                vuids: &["VUID-VkImportSemaphoreFdInfoKHR-flags-03323"],
                ..Default::default()
            }));
        }

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

        let &ImportSemaphoreWin32HandleInfo {
            flags,
            handle_type: _,
            handle: _,
            _ne: _,
        } = import_semaphore_win32_handle_info;

        if self.semaphore_type == SemaphoreType::Timeline
            && flags.intersects(SemaphoreImportFlags::TEMPORARY)
        {
            return Err(Box::new(ValidationError {
                problem: "`self.semaphore_type()` is `SemaphoreType::Timeline`, but \
                    `import_semaphore_win32_handle_info.flags` contains \
                    `SemaphoreImportFlags::TEMPORARY`"
                    .into(),
                vuids: &["VUID-VkImportSemaphoreWin32HandleInfoKHR-flags-03322"],
                ..Default::default()
            }));
        }

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

        if self.semaphore_type == SemaphoreType::Timeline {
            return Err(Box::new(ValidationError {
                problem: "`self.semaphore_type()` is `SemaphoreType::Timeline`".into(),
                vuids: &["VUID-VkImportSemaphoreZirconHandleInfoFUCHSIA-semaphoreType-04768"],
                ..Default::default()
            }));
        }

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
    /// The type of semaphore to create.
    ///
    /// The default value is [`SemaphoreType::Binary`].
    pub semaphore_type: SemaphoreType,

    /// If `semaphore_type` is [`SemaphoreType::Timeline`],
    /// specifies the counter value that the semaphore has when it is created.
    ///
    /// If `semaphore_type` is [`SemaphoreType::Binary`], then this must be `0`.
    ///
    /// The default value is `0`.
    pub initial_value: u64,

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
            semaphore_type: SemaphoreType::Binary,
            initial_value: 0,
            export_handle_types: ExternalSemaphoreHandleTypes::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SemaphoreCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            semaphore_type,
            initial_value,
            export_handle_types,
            _ne: _,
        } = self;

        semaphore_type.validate_device(device).map_err(|err| {
            err.add_context("semaphore_type")
                .set_vuids(&["VUID-VkSemaphoreTypeCreateInfo-semaphoreType-parameter"])
        })?;

        match semaphore_type {
            SemaphoreType::Binary => {
                if initial_value != 0 {
                    return Err(Box::new(ValidationError {
                        problem: "`semaphore_type` is `SemaphoreType::Binary`, but \
                            `initial_value` is not `0`"
                            .into(),
                        vuids: &["VUID-VkSemaphoreTypeCreateInfo-semaphoreType-03279"],
                        ..Default::default()
                    }));
                }
            }
            SemaphoreType::Timeline => {
                if !device.enabled_features().timeline_semaphore {
                    return Err(Box::new(ValidationError {
                        context: "semaphore_type".into(),
                        problem: "is `SemaphoreType::Timeline`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("timeline_semaphore"),
                        ])]),
                        vuids: &["VUID-VkSemaphoreTypeCreateInfo-timelineSemaphore-03252"],
                    }));
                }
            }
        }

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
                        .external_semaphore_properties_unchecked(ExternalSemaphoreInfo {
                            semaphore_type,
                            initial_value,
                            ..ExternalSemaphoreInfo::handle_type(handle_type)
                        })
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

vulkan_enum! {
    #[non_exhaustive]

    /// The type that a semaphore can have.
    SemaphoreType = SemaphoreType(i32);

    /// A semaphore that can only have two states: unsignaled and signaled.
    /// At any given time, only one pending operation may signal a binary semaphore, and only
    /// one pending operation may wait on it.
    Binary = BINARY,

    /// A semaphore whose state is a monotonically increasing integer. Signaling and waiting
    /// operations have an associated semaphore value: signaling a timeline semaphore sets it to
    /// the associated value, while waiting for a timeline semaphore will wait
    /// until the current semaphore state is greater than or equal to the associated value.
    Timeline = TIMELINE,
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

/// Parameters to signal a timeline semaphore.
#[derive(Clone, Debug)]
pub struct SemaphoreSignalInfo {
    /// The new value to set the semaphore's counter to.
    ///
    /// The default value is `0`.
    pub value: u64,

    pub _ne: crate::NonExhaustive,
}

impl Default for SemaphoreSignalInfo {
    #[inline]
    fn default() -> Self {
        Self {
            value: 0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SemaphoreSignalInfo {
    pub(crate) fn validate(&self, _device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self { value: _, _ne: _ } = self;

        // unsafe
        // VUID-VkSemaphoreSignalInfo-value-03258
        // VUID-VkSemaphoreSignalInfo-value-03259
        // VUID-VkSemaphoreSignalInfo-value-03260

        Ok(())
    }
}

/// Parameters to wait for a single timeline semaphore.
#[derive(Clone, Debug)]
pub struct SemaphoreWaitInfo {
    /// Additional properties of the wait operation.
    ///
    /// The default value is empty.
    pub flags: SemaphoreWaitFlags,

    /// The value to wait for.
    ///
    /// The default value is `0`.
    pub value: u64,

    pub _ne: crate::NonExhaustive,
}

impl Default for SemaphoreWaitInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: SemaphoreWaitFlags::empty(),
            value: 0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SemaphoreWaitInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            value: _,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkSemaphoreWaitInfo-flags-parameter"])
        })?;

        Ok(())
    }
}

/// Parameters to wait for multiple timeline semaphores.
#[derive(Clone, Debug)]
pub struct SemaphoreWaitMultipleInfo {
    /// Additional properties of the wait operation.
    ///
    /// The default value is empty.
    pub flags: SemaphoreWaitFlags,

    /// The semaphores to wait for, and the values to wait for.
    ///
    /// The default value is empty.
    pub semaphores: Vec<SemaphoreWaitValueInfo>,

    pub _ne: crate::NonExhaustive,
}

impl Default for SemaphoreWaitMultipleInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: SemaphoreWaitFlags::empty(),
            semaphores: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SemaphoreWaitMultipleInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            ref semaphores,
            _ne,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkSemaphoreWaitInfo-flags-parameter"])
        })?;

        if semaphores.is_empty() {
            return Ok(());
        }

        for (index, value_info) in semaphores.iter().enumerate() {
            value_info
                .validate(device)
                .map_err(|err| err.add_context(format!("semaphores[{}]", index)))?;
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a semaphore wait operation.
    SemaphoreWaitFlags = SemaphoreWaitFlags(u32);

    /// Wait for at least one of the semaphores to signal the required value,
    /// rather than all of them.
    ANY = ANY,
}

/// A semaphore to wait for, along with the value to wait for.
#[derive(Clone, Debug)]
pub struct SemaphoreWaitValueInfo {
    /// The semaphore to wait for.
    ///
    /// There is no default value.
    pub semaphore: Arc<Semaphore>,

    /// The value to wait for.
    ///
    /// There is no default value.
    pub value: u64,

    pub _ne: crate::NonExhaustive,
}

impl SemaphoreWaitValueInfo {
    /// Returns a `SemaphoreWaitValueInfo` with the specified `semaphore` and `value`.
    #[inline]
    pub fn new(semaphore: Arc<Semaphore>, value: u64) -> Self {
        Self {
            semaphore,
            value,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref semaphore,
            value: _,
            _ne: _,
        } = self;

        assert_eq!(device, semaphore.device.as_ref());

        if semaphore.semaphore_type != SemaphoreType::Timeline {
            return Err(Box::new(ValidationError {
                context: "semaphore.semaphore_type()".into(),
                problem: "is not `SemaphoreType::Timeline`".into(),
                vuids: &["VUID-VkSemaphoreWaitInfo-pSemaphores-03256"],
                ..Default::default()
            }));
        }

        Ok(())
    }
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
    /// The default value is `0`, which must be overridden.
    pub handle: ash::vk::HANDLE,

    pub _ne: crate::NonExhaustive,
}

impl ImportSemaphoreWin32HandleInfo {
    /// Returns an `ImportSemaphoreWin32HandleInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalSemaphoreHandleType) -> Self {
        Self {
            flags: SemaphoreImportFlags::empty(),
            handle_type,
            handle: 0,
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
/// [`PhysicalDevice::external_semaphore_properties`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExternalSemaphoreInfo {
    /// The external handle type that will be used with the semaphore.
    ///
    /// There is no default value.
    pub handle_type: ExternalSemaphoreHandleType,

    /// The type that the semaphore will have.
    ///
    /// The default value is [`SemaphoreType::Binary`].
    pub semaphore_type: SemaphoreType,

    /// The initial value that the semaphore will have.
    ///
    /// The default value is `0`.
    pub initial_value: u64,

    pub _ne: crate::NonExhaustive,
}

impl ExternalSemaphoreInfo {
    /// Returns an `ExternalSemaphoreInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalSemaphoreHandleType) -> Self {
        Self {
            handle_type,
            semaphore_type: SemaphoreType::Binary,
            initial_value: 0,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            handle_type,
            semaphore_type,
            initial_value,
            _ne: _,
        } = self;

        handle_type
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("handle_type")
                    .set_vuids(&["VUID-VkPhysicalDeviceExternalSemaphoreInfo-handleType-parameter"])
            })?;

        semaphore_type
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("semaphore_type")
                    .set_vuids(&["VUID-VkSemaphoreTypeCreateInfo-semaphoreType-parameter"])
            })?;

        match semaphore_type {
            SemaphoreType::Binary => {
                if initial_value != 0 {
                    return Err(Box::new(ValidationError {
                        problem: "`semaphore_type` is `SemaphoreType::Binary`, but \
                                `initial_value` is not `0`"
                            .into(),
                        vuids: &["VUID-VkSemaphoreTypeCreateInfo-semaphoreType-03279"],
                        ..Default::default()
                    }));
                }
            }
            SemaphoreType::Timeline => {
                if !physical_device.supported_features().timeline_semaphore {
                    return Err(Box::new(ValidationError {
                        context: "semaphore_type".into(),
                        problem: "is `SemaphoreType::Timeline`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("timeline_semaphore"),
                        ])]),
                        vuids: &["VUID-VkSemaphoreTypeCreateInfo-timelineSemaphore-03252"],
                    }));
                }
            }
        }

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
        device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFamilyIndex},
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
                    queue_family_index: QueueFamilyIndex(0),
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
