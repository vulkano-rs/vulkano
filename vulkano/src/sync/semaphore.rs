// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::ptr;
use std::sync::Arc;

use OomError;
use SafeDeref;
use VulkanObject;
use check_errors;
use device::Device;
use device::DeviceOwned;
use vk;

/// Used to provide synchronization between command buffers during their execution.
///
/// It is similar to a fence, except that it is purely on the GPU side. The CPU can't query a
/// semaphore's status or wait for it to be signaled.
#[derive(Debug)]
pub struct Semaphore<D = Arc<Device>>
    where D: SafeDeref<Target = Device>
{
    semaphore: vk::Semaphore,
    device: D,
    must_put_in_pool: bool,
}

impl<D> Semaphore<D>
    where D: SafeDeref<Target = Device>
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
            Some(raw_sem) => {
                Ok(Semaphore {
                       device: device,
                       semaphore: raw_sem,
                       must_put_in_pool: true,
                   })
            },
            None => {
                // Pool is empty, alloc new semaphore
                Semaphore::alloc_impl(device, true)
            },
        }
    }

    /// Builds a new semaphore.
    #[inline]
    pub fn alloc(device: D) -> Result<Semaphore<D>, OomError> {
        Semaphore::alloc_impl(device, false)
    }

    fn alloc_impl(device: D, must_put_in_pool: bool) -> Result<Semaphore<D>, OomError> {
        let semaphore = unsafe {
            // since the creation is constant, we use a `static` instead of a struct on the stack
            static mut INFOS: vk::SemaphoreCreateInfo = vk::SemaphoreCreateInfo {
                sType: vk::STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                pNext: 0 as *const _, // ptr::null()
                flags: 0, // reserved
            };

            let vk = device.pointers();
            let mut output = mem::uninitialized();
            check_errors(vk.CreateSemaphore(device.internal_object(),
                                            &INFOS,
                                            ptr::null(),
                                            &mut output))?;
            output
        };

        Ok(Semaphore {
               device: device,
               semaphore: semaphore,
               must_put_in_pool: must_put_in_pool,
           })
    }
}

unsafe impl DeviceOwned for Semaphore {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<D> VulkanObject for Semaphore<D>
    where D: SafeDeref<Target = Device>
{
    type Object = vk::Semaphore;

    #[inline]
    fn internal_object(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl<D> Drop for Semaphore<D>
    where D: SafeDeref<Target = Device>
{
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.must_put_in_pool {
                let raw_sem = self.semaphore;
                self.device.semaphore_pool().lock().unwrap().push(raw_sem);
            } else {
                let vk = self.device.pointers();
                vk.DestroySemaphore(self.device.internal_object(), self.semaphore, ptr::null());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use VulkanObject;
    use sync::Semaphore;

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
}
