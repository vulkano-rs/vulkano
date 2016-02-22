use std::mem;
use std::option::IntoIter as OptionIntoIter;
use std::ptr;
use std::sync::Arc;

use buffer::BufferResource;
use device::Device;

use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Pool from which descriptor sets are allocated from.
pub struct DescriptorPool {
    pool: vk::DescriptorPool,
    device: Arc<Device>,
}

impl DescriptorPool {
    pub fn new(device: &Arc<Device>) -> Result<Arc<DescriptorPool>, OomError> {
        let vk = device.pointers();

        // FIXME: arbitrary
        let pool_sizes = vec![
            vk::DescriptorPoolSize {
                ty: vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount: 10,
            }
        ];

        let pool = unsafe {
            let infos = vk::DescriptorPoolCreateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // TODO:
                maxSets: 100,       // TODO: let user choose
                poolSizeCount: pool_sizes.len() as u32,
                pPoolSizes: pool_sizes.as_ptr(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateDescriptorPool(device.internal_object(), &infos,
                                                      ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(DescriptorPool {
            pool: pool,
            device: device.clone(),
        }))
    }

    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for DescriptorPool {
    type Object = vk::DescriptorPool;

    #[inline]
    fn internal_object(&self) -> vk::DescriptorPool {
        self.pool
    }
}

impl Drop for DescriptorPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyDescriptorPool(self.device.internal_object(), self.pool, ptr::null());
        }
    }
}
