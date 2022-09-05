// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    device::{Device, DeviceOwned},
    macros::{vulkan_bitflags, vulkan_enum, ExtensionNotEnabled},
    OomError, Version, VulkanError, VulkanObject,
};
use std::{
    error::Error,
    fmt,
    fs::File,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
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

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_semaphore)
            {
                return Err(SemaphoreCreationError::ExtensionNotEnabled {
                    extension: "khr_external_semaphore",
                    reason: "export_handle_types was not empty",
                });
            }

            // VUID-VkExportSemaphoreCreateInfo-handleTypes-parameter
            export_handle_types.validate(&device)?;

            // VUID-VkExportSemaphoreCreateInfo-handleTypes-01124
            // TODO: `vkGetPhysicalDeviceExternalSemaphoreProperties` can only be called with one
            // handle type, so which one do we give it?
        }

        let mut create_info = ash::vk::SemaphoreCreateInfo::builder();

        let mut export_semaphore_create_info = if !export_handle_types.is_empty() {
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
            (fns.v1_0.create_semaphore)(
                device.internal_object(),
                &create_info.build(),
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
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
        let handle = device.semaphore_pool().lock().pop();
        let semaphore = match handle {
            Some(handle) => Semaphore {
                device,
                handle,
                must_put_in_pool: true,

                export_handle_types: ExternalSemaphoreHandleTypes::empty(),
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

    /// Creates a new `Semaphore` from an ash-handle
    /// # Safety
    /// The `handle` has to be a valid vulkan object handle and
    /// the `create_info` must match the info used to create said object
    pub unsafe fn from_handle(
        handle: ash::vk::Semaphore,
        create_info: SemaphoreCreateInfo,
        device: Arc<Device>,
    ) -> Semaphore {
        let SemaphoreCreateInfo {
            export_handle_types,
            _ne: _,
        } = create_info;

        Semaphore {
            device,
            handle,
            must_put_in_pool: false,

            export_handle_types,
        }
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
                (fns.khr_external_semaphore_fd.get_semaphore_fd_khr)(
                    self.device.internal_object(),
                    &info,
                    output.as_mut_ptr(),
                )
                .result()
                .map_err(VulkanError::from)?;
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
                self.device.semaphore_pool().lock().push(raw_sem);
            } else {
                let fns = self.device.fns();
                (fns.v1_0.destroy_semaphore)(
                    self.device.internal_object(),
                    self.handle,
                    ptr::null(),
                );
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

    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },
}

impl Error for SemaphoreCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SemaphoreCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::ExtensionNotEnabled { extension, reason } => {
                write!(f, "the extension {} must be enabled: {}", extension, reason)
            }
        }
    }
}

impl From<VulkanError> for SemaphoreCreationError {
    #[inline]
    fn from(err: VulkanError) -> Self {
        match err {
            e @ VulkanError::OutOfHostMemory | e @ VulkanError::OutOfDeviceMemory => {
                Self::OomError(e.into())
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for SemaphoreCreationError {
    #[inline]
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<ExtensionNotEnabled> for SemaphoreCreationError {
    #[inline]
    fn from(err: ExtensionNotEnabled) -> Self {
        Self::ExtensionNotEnabled {
            extension: err.extension,
            reason: err.reason,
        }
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
    /// Describes the handle type used for Vulkan external semaphore APIs.
    #[non_exhaustive]
    ExternalSemaphoreHandleType = ExternalSemaphoreHandleTypeFlags(u32);

    // TODO: document
    OpaqueFd = OPAQUE_FD,

    // TODO: document
    OpaqueWin32 = OPAQUE_WIN32,

    // TODO: document
    OpaqueWin32Kmt = OPAQUE_WIN32_KMT,

    // TODO: document
    D3D12Fence = D3D12_FENCE,

    // TODO: document
    SyncFd = SYNC_FD,

    /*
    // TODO: document
    ZirconEvent = ZIRCON_EVENT_FUCHSIA {
        extensions: [fuchsia_external_semaphore],
    },
     */
}

vulkan_bitflags! {
    /// A mask of multiple handle types.
    #[non_exhaustive]
    ExternalSemaphoreHandleTypes = ExternalSemaphoreHandleTypeFlags(u32);

    // TODO: document
    opaque_fd = OPAQUE_FD,

    // TODO: document
    opaque_win32 = OPAQUE_WIN32,

    // TODO: document
    opaque_win32_kmt = OPAQUE_WIN32_KMT,

    // TODO: document
    d3d12_fence = D3D12_FENCE,

    // TODO: document
    sync_fd = SYNC_FD,

    /*
    // TODO: document
    zircon_event = ZIRCON_EVENT_FUCHSIA {
        extensions: [fuchsia_external_semaphore],
    },
     */
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

impl From<VulkanError> for SemaphoreExportError {
    #[inline]
    fn from(err: VulkanError) -> Self {
        match err {
            e @ VulkanError::OutOfHostMemory | e @ VulkanError::OutOfDeviceMemory => {
                Self::OomError(e.into())
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl Error for SemaphoreExportError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
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
    use crate::{
        device::{
            physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        },
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
            sem.internal_object()
        };

        assert_eq!(device.semaphore_pool().lock().len(), 1);
        let sem2 = Semaphore::from_pool(device.clone()).unwrap();
        assert_eq!(device.semaphore_pool().lock().len(), 0);
        assert_eq!(sem2.internal_object(), sem1_internal_obj);
    }

    #[test]
    fn semaphore_export() {
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

        let physical_device = PhysicalDevice::enumerate(&instance).next().unwrap();
        let queue_family = physical_device.queue_families().next().unwrap();

        let (device, _) = match Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions {
                    khr_external_semaphore: true,
                    khr_external_semaphore_fd: true,
                    ..DeviceExtensions::empty()
                },
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
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
        let _fd = unsafe { sem.export_opaque_fd().unwrap() };
    }
}
