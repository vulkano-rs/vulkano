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
use std::sync::Mutex;
use std::sync::MutexGuard;

use device::Device;

use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Pool from which descriptor sets are allocated from.
///
/// A pool has a maximum number of descriptor sets and a maximum number of descriptors (one value
/// per descriptor type) it can allocate.
pub struct DescriptorPool {
    pool: Mutex<vk::DescriptorPool>,
    device: Arc<Device>,
}

impl DescriptorPool {
    /// See the docs of new().
    // FIXME: capacity of the pool
    pub fn raw(device: &Arc<Device>) -> Result<DescriptorPool, OomError> {
        let vk = device.pointers();

        // FIXME: arbitrary
        let pool_sizes = vec![
            vk::DescriptorPoolSize {
                ty: vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount: 10,
            },
            vk::DescriptorPoolSize {
                ty: vk::DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 10,
            },
            vk::DescriptorPoolSize {
                ty: vk::DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount: 10,
            },
            vk::DescriptorPoolSize {
                ty: vk::DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                descriptorCount: 10,
            },
            vk::DescriptorPoolSize {
                ty: vk::DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                descriptorCount: 10,
            },
            vk::DescriptorPoolSize {
                ty: vk::DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount: 10,
            },
        ];

        let pool = unsafe {
            let infos = vk::DescriptorPoolCreateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: vk::DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,   // TODO:
                maxSets: 100,       // TODO: let user choose
                poolSizeCount: pool_sizes.len() as u32,
                pPoolSizes: pool_sizes.as_ptr(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateDescriptorPool(device.internal_object(), &infos,
                                                      ptr::null(), &mut output)));
            output
        };

        Ok(DescriptorPool {
            pool: Mutex::new(pool),
            device: device.clone(),
        })
    }
    
    /// Initializes a new pool.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    ///
    // FIXME: capacity of the pool
    #[inline]
    pub fn new(device: &Arc<Device>) -> Arc<DescriptorPool> {
        Arc::new(DescriptorPool::raw(device).unwrap())
    }

    /// Returns the device this pool was created from.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl SynchronizedVulkanObject for DescriptorPool {
    type Object = vk::DescriptorPool;

    #[inline]
    fn internal_object_guard(&self) -> MutexGuard<vk::DescriptorPool> {
        self.pool.lock().unwrap()
    }
}

impl Drop for DescriptorPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            let pool = self.pool.lock().unwrap();
            vk.DestroyDescriptorPool(self.device.internal_object(), *pool, ptr::null());
        }
    }
}

#[cfg(test)]
mod tests {
    use descriptor::descriptor_set::DescriptorPool;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = DescriptorPool::new(&device);
    }

    #[test]
    fn device() {
        let (device, _) = gfx_dev_and_queue!();
        let pool = DescriptorPool::new(&device);
        assert_eq!(&**pool.device() as *const _, &*device as *const _);
    }
}
