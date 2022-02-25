// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    check_errors,
    device::{Device, DeviceOwned},
    Error, OomError, Version, VulkanObject,
};
use std::{
    fmt,
    fs::File,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops::BitOr,
    ptr,
    sync::Arc,
};

/// Used to provide synchronization between command buffers during their execution.
///
/// It is similar to a fence, except that it is purely on the GPU side. The CPU can't query a
/// semaphore's status or wait for it to be signaled.
#[derive(Debug)]
pub struct Semaphore {
    handle: ash::vk::Semaphore,
    device: Arc<Device>,
    must_put_in_pool: bool,

    export_handle_types: ExternalSemaphoreHandleTypes,
}

impl Semaphore {
    /// Creates a new `Semaphore`.
    pub fn new(
        device: Arc<Device>,
        create_info: SemaphoreCreateInfo,
    ) -> Result<Semaphore, SemaphoreCreationError> {
        let SemaphoreCreateInfo {
            export_handle_types,
            _ne: _,
        } = create_info;
        let instance = device.instance();

        if export_handle_types != ExternalSemaphoreHandleTypes::none() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_semaphore)
            {
                return Err(SemaphoreCreationError::MissingExtension(
                    "khr_external_semaphore",
                ));
            }

            if (export_handle_types.opaque_fd
                || export_handle_types.opaque_win32
                || export_handle_types.opaque_win32_kmt
                || export_handle_types.d3d12_fence
                || export_handle_types.sync_fd)
                && !(instance.api_version() >= Version::V1_1
                    || instance
                        .enabled_extensions()
                        .khr_external_semaphore_capabilities)
            {
                return Err(SemaphoreCreationError::MissingExtension(
                    "khr_external_semaphore_capabilities",
                ));
            }

            // VUID-VkExportSemaphoreCreateInfo-handleTypes-01124
            // TODO: `vkGetPhysicalDeviceExternalSemaphoreProperties` can only be called with one
            // handle type, so which one do we give it?
        }

        let mut create_info = ash::vk::SemaphoreCreateInfo::builder();

        let mut export_semaphore_create_info =
            if export_handle_types != ExternalSemaphoreHandleTypes::none() {
                Some(ash::vk::ExportSemaphoreCreateInfo {
                    handle_types: export_handle_types.into(),
                    ..Default::default()
                })
            } else {
                None
            };

        if let Some(info) = export_semaphore_create_info.as_mut() {
            create_info = create_info.push_next(info);
        }

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_semaphore(
                device.internal_object(),
                &create_info.build(),
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Semaphore {
            device,
            handle,
            must_put_in_pool: false,

            export_handle_types,
        })
    }

    /// Takes a semaphore from the vulkano-provided semaphore pool.
    /// If the pool is empty, a new semaphore will be allocated.
    /// Upon `drop`, the semaphore is put back into the pool.
    ///
    /// For most applications, using the pool should be preferred,
    /// in order to avoid creating new semaphores every frame.
    pub fn from_pool(device: Arc<Device>) -> Result<Semaphore, SemaphoreCreationError> {
        let handle = device.semaphore_pool().lock().unwrap().pop();
        let semaphore = match handle {
            Some(handle) => Semaphore {
                device,
                handle,
                must_put_in_pool: true,

                export_handle_types: ExternalSemaphoreHandleTypes::none(),
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

    /// # Safety
    ///
    /// - The semaphore must not be used, or have been used, to acquire a swapchain image.
    pub unsafe fn export_opaque_fd(&self) -> Result<File, SemaphoreExportError> {
        let fns = self.device.fns();

        // VUID-VkSemaphoreGetFdInfoKHR-handleType-01132
        if !self.export_handle_types.opaque_fd {
            return Err(SemaphoreExportError::HandleTypeNotSupported {
                handle_type: ExternalSemaphoreHandleType::OpaqueFd,
            });
        }

        assert!(self.device.enabled_extensions().khr_external_semaphore);
        assert!(self.device.enabled_extensions().khr_external_semaphore_fd);

        // VUID-VkSemaphoreGetFdInfoKHR-semaphore-01133
        // Can't validate for swapchain.

        #[cfg(not(unix))]
        unreachable!("`khr_external_semaphore_fd` was somehow enabled on a non-Unix system");

        #[cfg(unix)]
        {
            use std::os::unix::io::FromRawFd;

            let fd = {
                let info = ash::vk::SemaphoreGetFdInfoKHR {
                    semaphore: self.handle,
                    handle_type: ash::vk::ExternalSemaphoreHandleTypeFlagsKHR::OPAQUE_FD,
                    ..Default::default()
                };

                let mut output = MaybeUninit::uninit();
                check_errors(fns.khr_external_semaphore_fd.get_semaphore_fd_khr(
                    self.device.internal_object(),
                    &info,
                    output.as_mut_ptr(),
                ))?;
                output.assume_init()
            };
            let file = File::from_raw_fd(fd);
            Ok(file)
        }
    }
}

impl Drop for Semaphore {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.must_put_in_pool {
                let raw_sem = self.handle;
                self.device.semaphore_pool().lock().unwrap().push(raw_sem);
            } else {
                let fns = self.device.fns();
                fns.v1_0
                    .destroy_semaphore(self.device.internal_object(), self.handle, ptr::null());
            }
        }
    }
}

unsafe impl VulkanObject for Semaphore {
    type Object = ash::vk::Semaphore;

    #[inline]
    fn internal_object(&self) -> ash::vk::Semaphore {
        self.handle
    }
}

unsafe impl DeviceOwned for Semaphore {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for Semaphore {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for Semaphore {}

impl Hash for Semaphore {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SemaphoreCreationError {
    /// Not enough memory available.
    OomError(OomError),

    /// An extension is missing.
    MissingExtension(&'static str),
}

impl fmt::Display for SemaphoreCreationError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::OomError(_) => write!(fmt, "not enough memory available"),
            Self::MissingExtension(s) => {
                write!(fmt, "Missing the following extension: {}", s)
            }
        }
    }
}

impl From<Error> for SemaphoreCreationError {
    #[inline]
    fn from(err: Error) -> Self {
        match err {
            e @ Error::OutOfHostMemory | e @ Error::OutOfDeviceMemory => Self::OomError(e.into()),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl std::error::Error for SemaphoreCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl From<OomError> for SemaphoreCreationError {
    #[inline]
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

/// Parameters to create a new `Semaphore`.
#[derive(Clone, Debug)]
pub struct SemaphoreCreateInfo {
    /// The handle types that can be exported from the semaphore.
    ///
    /// The default value is [`ExternalSemaphoreHandleTypes::none()`].
    pub export_handle_types: ExternalSemaphoreHandleTypes,

    pub _ne: crate::NonExhaustive,
}

impl Default for SemaphoreCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            export_handle_types: ExternalSemaphoreHandleTypes::none(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Describes the handle type used for Vulkan external semaphore APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ExternalSemaphoreHandleType {
    OpaqueFd = ash::vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD.as_raw(),
    OpaqueWin32 = ash::vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32.as_raw(),
    OpaqueWin32Kmt = ash::vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32_KMT.as_raw(),
    D3D12Fence = ash::vk::ExternalSemaphoreHandleTypeFlags::D3D12_FENCE.as_raw(),
    SyncFd = ash::vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD.as_raw(),
}

impl From<ExternalSemaphoreHandleType> for ash::vk::ExternalSemaphoreHandleTypeFlags {
    fn from(val: ExternalSemaphoreHandleType) -> Self {
        Self::from_raw(val as u32)
    }
}

/// A mask of multiple handle types.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ExternalSemaphoreHandleTypes {
    pub opaque_fd: bool,
    pub opaque_win32: bool,
    pub opaque_win32_kmt: bool,
    pub d3d12_fence: bool,
    pub sync_fd: bool,
}

impl ExternalSemaphoreHandleTypes {
    /// Builds a `ExternalSemaphoreHandleTypes` with all values set to false. Useful as a default value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::sync::ExternalSemaphoreHandleTypes;
    ///
    /// let _handle_type = ExternalSemaphoreHandleTypes {
    ///     opaque_fd: true,
    ///     .. ExternalSemaphoreHandleTypes::none()
    /// };
    /// ```
    #[inline]
    pub fn none() -> ExternalSemaphoreHandleTypes {
        ExternalSemaphoreHandleTypes {
            opaque_fd: false,
            opaque_win32: false,
            opaque_win32_kmt: false,
            d3d12_fence: false,
            sync_fd: false,
        }
    }

    /// Builds an `ExternalSemaphoreHandleTypes` for a posix file descriptor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::sync::ExternalSemaphoreHandleTypes;
    ///
    /// let _handle_type = ExternalSemaphoreHandleTypes::posix();
    /// ```
    #[inline]
    pub fn posix() -> ExternalSemaphoreHandleTypes {
        ExternalSemaphoreHandleTypes {
            opaque_fd: true,
            ..ExternalSemaphoreHandleTypes::none()
        }
    }
}

impl From<ExternalSemaphoreHandleTypes> for ash::vk::ExternalSemaphoreHandleTypeFlags {
    #[inline]
    fn from(val: ExternalSemaphoreHandleTypes) -> Self {
        let mut result = ash::vk::ExternalSemaphoreHandleTypeFlags::empty();
        if val.opaque_fd {
            result |= ash::vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;
        }
        if val.opaque_win32 {
            result |= ash::vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32;
        }
        if val.opaque_win32_kmt {
            result |= ash::vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32_KMT;
        }
        if val.d3d12_fence {
            result |= ash::vk::ExternalSemaphoreHandleTypeFlags::D3D12_FENCE;
        }
        if val.sync_fd {
            result |= ash::vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD;
        }
        result
    }
}

impl From<ash::vk::ExternalSemaphoreHandleTypeFlags> for ExternalSemaphoreHandleTypes {
    fn from(val: ash::vk::ExternalSemaphoreHandleTypeFlags) -> Self {
        Self {
            opaque_fd: !(val & ash::vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD).is_empty(),
            opaque_win32: !(val & ash::vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32)
                .is_empty(),
            opaque_win32_kmt: !(val & ash::vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32_KMT)
                .is_empty(),
            d3d12_fence: !(val & ash::vk::ExternalSemaphoreHandleTypeFlags::D3D12_FENCE).is_empty(),
            sync_fd: !(val & ash::vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD).is_empty(),
        }
    }
}

impl BitOr for ExternalSemaphoreHandleTypes {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        ExternalSemaphoreHandleTypes {
            opaque_fd: self.opaque_fd || rhs.opaque_fd,
            opaque_win32: self.opaque_win32 || rhs.opaque_win32,
            opaque_win32_kmt: self.opaque_win32_kmt || rhs.opaque_win32_kmt,
            d3d12_fence: self.d3d12_fence || rhs.d3d12_fence,
            sync_fd: self.sync_fd || rhs.sync_fd,
        }
    }
}

/// The semaphore configuration to query in
/// [`PhysicalDevice::external_semaphore_properties`](crate::device::physical::PhysicalDevice::external_semaphore_properties).
#[derive(Clone, Debug)]
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
    /// when creating the buffer or image.
    pub compatible_handle_types: ExternalSemaphoreHandleTypes,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SemaphoreExportError {
    /// Not enough memory available.
    OomError(OomError),

    /// The requested export handle type was not provided in `export_handle_types` when creating the
    /// semaphore.
    HandleTypeNotSupported {
        handle_type: ExternalSemaphoreHandleType,
    },
}

impl fmt::Display for SemaphoreExportError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::OomError(_) => write!(fmt, "not enough memory available"),
            Self::HandleTypeNotSupported { handle_type } => write!(
                fmt,
                "the requested export handle type ({:?}) was not provided in `export_handle_types` when creating the semaphore",
                handle_type,
            ),
        }
    }
}

impl From<Error> for SemaphoreExportError {
    #[inline]
    fn from(err: Error) -> Self {
        match err {
            e @ Error::OutOfHostMemory | e @ Error::OutOfDeviceMemory => Self::OomError(e.into()),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl std::error::Error for SemaphoreExportError {
    #[inline]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl From<OomError> for SemaphoreExportError {
    #[inline]
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

#[cfg(test)]
mod tests {
    use crate::sync::Semaphore;
    use crate::VulkanObject;

    #[test]
    fn semaphore_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = Semaphore::new(device.clone(), Default::default());
    }

    #[test]
    fn semaphore_pool() {
        let (device, _) = gfx_dev_and_queue!();

        assert_eq!(device.semaphore_pool().lock().unwrap().len(), 0);
        let sem1_internal_obj = {
            let sem = Semaphore::from_pool(device.clone()).unwrap();
            assert_eq!(device.semaphore_pool().lock().unwrap().len(), 0);
            sem.internal_object()
        };

        assert_eq!(device.semaphore_pool().lock().unwrap().len(), 1);
        let sem2 = Semaphore::from_pool(device.clone()).unwrap();
        assert_eq!(device.semaphore_pool().lock().unwrap().len(), 0);
        assert_eq!(sem2.internal_object(), sem1_internal_obj);
    }

    #[test]
    fn semaphore_export() {
        use crate::device::physical::PhysicalDevice;
        use crate::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
        use crate::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
        use crate::sync::semaphore::SemaphoreCreateInfo;
        use crate::sync::ExternalSemaphoreHandleTypes;

        let supported_ext = InstanceExtensions::supported_by_core().unwrap();
        if supported_ext.khr_get_display_properties2
            && supported_ext.khr_external_semaphore_capabilities
        {
            let instance = Instance::new(InstanceCreateInfo {
                enabled_extensions: InstanceExtensions {
                    khr_get_physical_device_properties2: true,
                    khr_external_semaphore_capabilities: true,
                    ..InstanceExtensions::none()
                },
                ..Default::default()
            })
            .unwrap();

            let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

            let queue_family = physical.queue_families().next().unwrap();

            let device_ext = DeviceExtensions {
                khr_external_semaphore: true,
                khr_external_semaphore_fd: true,
                ..DeviceExtensions::none()
            };
            let (device, _) = Device::new(
                physical,
                DeviceCreateInfo {
                    enabled_extensions: device_ext,
                    queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                    ..Default::default()
                },
            )
            .unwrap();

            let supported_ext = physical.supported_extensions();
            if supported_ext.khr_external_semaphore && supported_ext.khr_external_semaphore_fd {
                let sem = Semaphore::new(
                    device.clone(),
                    SemaphoreCreateInfo {
                        export_handle_types: ExternalSemaphoreHandleTypes::posix(),
                        ..Default::default()
                    },
                )
                .unwrap();
                let fd = unsafe { sem.export_opaque_fd().unwrap() };
            }
        }
    }
}
