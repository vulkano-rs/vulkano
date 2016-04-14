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

use device::Device;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Used to provide synchronization between command buffers during their execution.
/// 
/// It is similar to a fence, except that it is purely on the GPU side. The CPU can't query a
/// semaphore's status or wait for it to be signaled.
#[derive(Debug)]
pub struct Semaphore {
    device: Arc<Device>,
    semaphore: vk::Semaphore,
}

impl Semaphore {
    /// Builds a new semaphore.
    #[inline]
    pub fn new(device: &Arc<Device>) -> Result<Arc<Semaphore>, OomError> {
        let vk = device.pointers();

        let semaphore = unsafe {
            // since the creation is constant, we use a `static` instead of a struct on the stack
            static mut INFOS: vk::SemaphoreCreateInfo = vk::SemaphoreCreateInfo {
                sType: vk::STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                pNext: 0 as *const _,   // ptr::null()
                flags: 0,   // reserved
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateSemaphore(device.internal_object(), &INFOS,
                                                 ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(Semaphore {
            device: device.clone(),
            semaphore: semaphore,
        }))
    }
}

unsafe impl VulkanObject for Semaphore {
    type Object = vk::Semaphore;

    #[inline]
    fn internal_object(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for Semaphore {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroySemaphore(self.device.internal_object(), self.semaphore, ptr::null());
        }
    }
}

#[cfg(test)]
mod tests {
    use sync::Semaphore;

    #[test]
    fn semaphore_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = Semaphore::new(&device).unwrap();
    }
}
