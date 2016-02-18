use std::mem;
use std::ptr;
use std::sync::Arc;

use instance::QueueFamily;

use device::Device;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// A pool from which command buffers are created from.
pub struct CommandBufferPool {
    device: Arc<Device>,
    pool: vk::CommandPool,
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
    pub fn new(device: &Arc<Device>, queue_family: &QueueFamily)
               -> Result<Arc<CommandBufferPool>, OomError>
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

        Ok(Arc::new(CommandBufferPool {
            device: device.clone(),
            pool: pool,
            queue_family_index: queue_family.id(),
        }))
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

impl VulkanObject for CommandBufferPool {
    type Object = vk::CommandPool;

    #[inline]
    fn internal_object(&self) -> vk::CommandPool {
        self.pool
    }
}

impl Drop for CommandBufferPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyCommandPool(self.device.internal_object(), self.pool, ptr::null());
        }
    }
}
