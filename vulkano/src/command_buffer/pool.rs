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

use instance::QueueFamily;

use device::Device;
use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// A pool from which command buffers are created from.
pub struct CommandBufferPool {
    pool: Mutex<vk::CommandPool>,
    device: Arc<Device>,
    queue_family_index: u32,
}

impl CommandBufferPool {
    /// Creates a new pool.
    ///
    /// The command buffers created with this pool can only be executed on queues of the given
    /// family.
    ///
    /// # Panic
    ///
    /// Panicks if the queue family doesn't belong to the same physical device as `device`.
    ///
    #[inline]
    pub fn raw(device: &Arc<Device>, queue_family: &QueueFamily)
               -> Result<CommandBufferPool, OomError>
    {
        assert_eq!(device.physical_device().internal_object(),
                   queue_family.physical_device().internal_object());

        let vk = device.pointers();

        let pool = unsafe {
            let infos = vk::CommandPoolCreateInfo {
                sType: vk::STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,       // TODO: 
                queueFamilyIndex: queue_family.id(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateCommandPool(device.internal_object(), &infos,
                                                   ptr::null(), &mut output)));
            output
        };

        Ok(CommandBufferPool {
            pool: Mutex::new(pool),
            device: device.clone(),
            queue_family_index: queue_family.id(),
        })
    }
    
    /// Creates a new pool.
    ///
    /// The command buffers created with this pool can only be executed on queues of the given
    /// family.
    ///
    /// # Panic
    ///
    /// - Panicks if the queue family doesn't belong to the same physical device as `device`.
    /// - Panicks if the device or host ran out of memory.
    ///
    #[inline]
    pub fn new(device: &Arc<Device>, queue_family: &QueueFamily)
               -> Arc<CommandBufferPool>
    {
        Arc::new(CommandBufferPool::raw(device, queue_family).unwrap())
    }

    /// Returns the device this command pool was created with.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the queue family on which command buffers of this pool can be executed.
    #[inline]
    pub fn queue_family(&self) -> QueueFamily {
        self.device.physical_device().queue_family_by_id(self.queue_family_index).unwrap()
    }
}

unsafe impl SynchronizedVulkanObject for CommandBufferPool {
    type Object = vk::CommandPool;

    #[inline]
    fn internal_object_guard(&self) -> MutexGuard<vk::CommandPool> {
        self.pool.lock().unwrap()
    }
}

impl Drop for CommandBufferPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            let pool = self.pool.lock().unwrap();
            vk.DestroyCommandPool(self.device.internal_object(), *pool, ptr::null());
        }
    }
}
