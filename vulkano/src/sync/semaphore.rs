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
use crate::OomError;
use crate::SafeDeref;
use crate::VulkanObject;
#[cfg(target_os = "linux")]
use std::fs::File;
use std::mem::MaybeUninit;
#[cfg(target_os = "linux")]
use std::os::unix::io::FromRawFd;
use std::ptr;
use std::sync::Arc;

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
    pub fn from_pool(device: D) -> Result<Semaphore<D>, OomError> {
        let maybe_raw_sem = device.semaphore_pool().lock().unwrap().pop();
        match maybe_raw_sem {
            Some(raw_sem) => Ok(Semaphore {
                device: device,
                semaphore: raw_sem,
                must_put_in_pool: true,
            }),
            None => {
                // Pool is empty, alloc new semaphore
                Semaphore::alloc_impl(device, true, false)
            }
        }
    }

    /// Builds a new semaphore.
    #[inline]
    pub fn alloc(device: D) -> Result<Semaphore<D>, OomError> {
        Semaphore::alloc_impl(device, false, false)
    }

    /// Same as `alloc`, but allows exportable opaque file descriptor on Linux
    #[inline]
    #[cfg(target_os = "linux")]
    pub fn alloc_with_exportable_fd(device: D) -> Result<Semaphore<D>, OomError> {
        Semaphore::alloc_impl(device, false, true)
    }

    fn alloc_impl(
        device: D,
        must_put_in_pool: bool,
        exportable_fd: bool,
    ) -> Result<Semaphore<D>, OomError> {
        let semaphore = unsafe {
            // since the creation is constant, we use a `static` instead of a struct on the stack

            let export_semaphore_info = ash::vk::ExportSemaphoreCreateInfoKHR {
                handle_types: ash::vk::ExternalSemaphoreHandleTypeFlagsKHR::OPAQUE_FD,
                ..Default::default()
            };

            let mut infos = ash::vk::SemaphoreCreateInfo {
                flags: ash::vk::SemaphoreCreateFlags::empty(),
                ..Default::default()
            };

            #[cfg(target_os = "linux")]
            {
                infos.p_next = std::mem::transmute(&export_semaphore_info);
            }

            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_semaphore(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Semaphore {
            device: device,
            semaphore: semaphore,
            must_put_in_pool: must_put_in_pool,
        })
    }

    #[cfg(target_os = "linux")]
    pub fn export_opaque_fd(&self) -> Result<File, OomError> {
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

#[cfg(test)]
mod tests {
    use crate::sync::Semaphore;
    use crate::VulkanObject;

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
        let (device, _) = gfx_dev_and_queue!();
        let sem = Semaphore::alloc_with_exportable_fd(device.clone()).unwrap();
        let fd = sem.export_opaque_fd().unwrap();
    }
}
