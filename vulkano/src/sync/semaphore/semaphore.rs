// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::Error;
use crate::OomError;
use crate::SafeDeref;
use crate::VulkanObject;
use std::fmt;
#[cfg(target_os = "linux")]
use std::fs::File;
use std::mem::MaybeUninit;
#[cfg(target_os = "linux")]
use std::os::unix::io::FromRawFd;
use std::ptr;
use std::sync::Arc;

use crate::sync::semaphore::ExternalSemaphoreHandleType;

/// Used to provide synchronization between command buffers during their execution.
///
/// It is similar to a fence, except that it is purely on the GPU side. The CPU can't query a
/// semaphore's status or wait for it to be signaled.
#[derive(Debug)]
pub struct Semaphore<D = Arc<Device>>
where
    D: SafeDeref<Target = Device>,
{
    semaphore: ash::vk::Semaphore,
    device: D,
    must_put_in_pool: bool,
}

// TODO: Add support for VkExportSemaphoreWin32HandleInfoKHR
// TODO: Add suport for importable semaphores
pub struct SemaphoreBuilder<D = Arc<Device>>
where
    D: SafeDeref<Target = Device>,
{
    device: D,
    export_info: Option<ash::vk::ExportSemaphoreCreateInfo>,
    create: ash::vk::SemaphoreCreateInfo,
    must_put_in_pool: bool,
}

impl<D> SemaphoreBuilder<D>
where
    D: SafeDeref<Target = Device>,
{
    pub fn new(device: D) -> Self {
        let create = ash::vk::SemaphoreCreateInfo::default();

        Self {
            device,
            export_info: None,
            create,
            must_put_in_pool: false,
        }
    }
    /// Configures the semaphore to be added to the semaphore pool once it is destroyed.
    pub(crate) fn in_pool(mut self) -> Self {
        self.must_put_in_pool = true;
        self
    }

    /// Sets an optional field for exportable allocations in the `SemaphoreBuilder`.
    ///
    /// # Panic
    ///
    /// - Panics if the export info has already been set.
    pub fn export_info(mut self, handle_types: ExternalSemaphoreHandleType) -> Self {
        assert!(self.export_info.is_none());
        let export_info = ash::vk::ExportSemaphoreCreateInfo {
            handle_types: handle_types.into(),
            ..Default::default()
        };

        self.export_info = Some(export_info);
        self.create.p_next = unsafe { std::mem::transmute(&export_info) };

        self
    }

    pub fn build(self) -> Result<Semaphore<D>, SemaphoreError> {
        if self.export_info.is_some()
            && !self
                .device
                .instance()
                .loaded_extensions()
                .khr_external_semaphore_capabilities
        {
            Err(SemaphoreError::MissingExtension(
                "khr_external_semaphore_capabilities",
            ))
        } else {
            let semaphore = unsafe {
                let fns = self.device.fns();
                let mut output = MaybeUninit::uninit();
                check_errors(fns.v1_0.create_semaphore(
                    self.device.internal_object(),
                    &self.create,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                output.assume_init()
            };

            Ok(Semaphore {
                device: self.device,
                semaphore,
                must_put_in_pool: self.must_put_in_pool,
            })
        }
    }
}

impl<D> Semaphore<D>
where
    D: SafeDeref<Target = Device>,
{
    /// Takes a semaphore from the vulkano-provided semaphore pool.
    /// If the pool is empty, a new semaphore will be allocated.
    /// Upon `drop`, the semaphore is put back into the pool.
    ///
    /// For most applications, using the pool should be preferred,
    /// in order to avoid creating new semaphores every frame.
    pub fn from_pool(device: D) -> Result<Semaphore<D>, SemaphoreError> {
        let maybe_raw_sem = device.semaphore_pool().lock().unwrap().pop();
        match maybe_raw_sem {
            Some(raw_sem) => Ok(Semaphore {
                device,
                semaphore: raw_sem,
                must_put_in_pool: true,
            }),
            None => {
                // Pool is empty, alloc new semaphore
                SemaphoreBuilder::new(device).in_pool().build()
            }
        }
    }

    /// Builds a new semaphore.
    #[inline]
    pub fn alloc(device: D) -> Result<Semaphore<D>, SemaphoreError> {
        SemaphoreBuilder::new(device).build()
    }

    /// Same as `alloc`, but allows exportable opaque file descriptor on Linux
    #[inline]
    #[cfg(target_os = "linux")]
    pub fn alloc_with_exportable_fd(device: D) -> Result<Semaphore<D>, SemaphoreError> {
        SemaphoreBuilder::new(device)
            .export_info(ExternalSemaphoreHandleType::posix())
            .build()
    }

    #[cfg(target_os = "linux")]
    pub fn export_opaque_fd(&self) -> Result<File, SemaphoreError> {
        let fns = self.device.fns();

        assert!(self.device.loaded_extensions().khr_external_semaphore);
        assert!(self.device.loaded_extensions().khr_external_semaphore_fd);

        let fd = unsafe {
            let info = ash::vk::SemaphoreGetFdInfoKHR {
                semaphore: self.semaphore,
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
        let file = unsafe { File::from_raw_fd(fd) };
        Ok(file)
    }
}

unsafe impl DeviceOwned for Semaphore {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<D> VulkanObject for Semaphore<D>
where
    D: SafeDeref<Target = Device>,
{
    type Object = ash::vk::Semaphore;

    #[inline]
    fn internal_object(&self) -> ash::vk::Semaphore {
        self.semaphore
    }
}

impl<D> Drop for Semaphore<D>
where
    D: SafeDeref<Target = Device>,
{
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.must_put_in_pool {
                let raw_sem = self.semaphore;
                self.device.semaphore_pool().lock().unwrap().push(raw_sem);
            } else {
                let fns = self.device.fns();
                fns.v1_0.destroy_semaphore(
                    self.device.internal_object(),
                    self.semaphore,
                    ptr::null(),
                );
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SemaphoreError {
    /// Not enough memory available.
    OomError(OomError),
    /// An extensions is missing.
    MissingExtension(&'static str),
}

impl fmt::Display for SemaphoreError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            SemaphoreError::OomError(_) => write!(fmt, "not enough memory available"),
            SemaphoreError::MissingExtension(s) => {
                write!(fmt, "Missing the following extension: {}", s)
            }
        }
    }
}

impl From<Error> for SemaphoreError {
    #[inline]
    fn from(err: Error) -> SemaphoreError {
        match err {
            e @ Error::OutOfHostMemory | e @ Error::OutOfDeviceMemory => {
                SemaphoreError::OomError(e.into())
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl std::error::Error for SemaphoreError {
    #[inline]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            SemaphoreError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl From<OomError> for SemaphoreError {
    #[inline]
    fn from(err: OomError) -> SemaphoreError {
        SemaphoreError::OomError(err)
    }
}

#[cfg(test)]
mod tests {

    use crate::device::{Device, DeviceExtensions};
    use crate::instance::{Instance, InstanceExtensions, PhysicalDevice};
    use crate::VulkanObject;
    use crate::{sync::Semaphore, Version};

    #[test]
    fn semaphore_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = Semaphore::alloc(device.clone());
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
    #[cfg(target_os = "linux")]
    fn semaphore_export() {
        let supported_ext = InstanceExtensions::supported_by_core().unwrap();
        if supported_ext.khr_get_display_properties2
            && supported_ext.khr_external_semaphore_capabilities
        {
            let instance = Instance::new(
                None,
                Version::V1_1,
                &InstanceExtensions {
                    khr_get_physical_device_properties2: true,
                    khr_external_semaphore_capabilities: true,
                    ..InstanceExtensions::none()
                },
                None,
            )
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
                physical.supported_features(),
                &device_ext,
                [(queue_family, 0.5)].iter().cloned(),
            )
            .unwrap();

            let supported_ext = DeviceExtensions::supported_by_device(physical.clone());
            if supported_ext.khr_external_semaphore && supported_ext.khr_external_semaphore_fd {
                let sem = Semaphore::alloc_with_exportable_fd(device.clone()).unwrap();
                let fd = sem.export_opaque_fd().unwrap();
            }
        }
    }
}
