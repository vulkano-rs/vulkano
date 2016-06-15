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
use smallvec::SmallVec;

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
    capacity: PoolCapacity,
    free_individual_sets: bool,
}

impl DescriptorPool {
    /// See the docs of new().
    // FIXME: capacity of the pool
    pub fn raw(device: &Arc<Device>, free_individual_sets: bool, capacity: PoolCapacity)
               -> Result<DescriptorPool, OomError>
    {
        let vk = device.pointers();

        assert!(capacity.sets >= 1, "Descriptor pool capacity must not be 0");

        let mut pool_sizes: SmallVec<[_; 11]> = SmallVec::new();

        macro_rules! cap {
            ($ident:ident => $vk:expr) => (
                if capacity.$ident >= 1 {
                    pool_sizes.push(vk::DescriptorPoolSize {
                        ty: $vk,
                        descriptorCount: capacity.$ident,
                    });
                }
            );
        }

        cap!(sampler => vk::DESCRIPTOR_TYPE_SAMPLER);
        cap!(combined_image_sampler => vk::DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        cap!(sampled_image => vk::DESCRIPTOR_TYPE_SAMPLED_IMAGE);
        cap!(storage_image => vk::DESCRIPTOR_TYPE_STORAGE_IMAGE);
        cap!(input_attachment => vk::DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        cap!(uniform_buffer => vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        cap!(storage_buffer => vk::DESCRIPTOR_TYPE_STORAGE_BUFFER);
        cap!(uniform_buffer_dynamic => vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
        cap!(storage_buffer_dynamic => vk::DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
        cap!(uniform_texel_buffer => vk::DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER);
        cap!(storage_texel_buffer => vk::DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);

        assert!(!pool_sizes.is_empty(), "The number of descriptors in a descriptor pool can't \
                                         be 0");

        let pool = unsafe {
            let infos = vk::DescriptorPoolCreateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: if free_individual_sets {
                    vk::DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
                } else {
                    0
                },
                maxSets: capacity.sets,
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
            capacity: capacity,
            free_individual_sets: free_individual_sets,
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
    pub fn new(device: &Arc<Device>, free_individual_sets: bool, capacity: PoolCapacity)
               -> Arc<DescriptorPool>
    {
        Arc::new(DescriptorPool::raw(device, free_individual_sets, capacity).unwrap())
    }

    /// Returns the device this pool was created from.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns true if this pool was configured so that it's possible to free individual
    /// descriptor sets.
    #[inline]
    pub fn free_individual_sets(&self) -> bool {
        self.free_individual_sets
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PoolCapacity {
    pub sets: u32,
    pub sampler: u32,
    pub combined_image_sampler: u32,
    pub sampled_image: u32,
    pub storage_image: u32,
    pub input_attachment: u32,
    pub uniform_buffer: u32,
    pub storage_buffer: u32,
    pub uniform_buffer_dynamic: u32,
    pub storage_buffer_dynamic: u32,
    pub uniform_texel_buffer: u32,
    pub storage_texel_buffer: u32,
}

#[cfg(test)]
mod tests {
    use descriptor::descriptor_set::DescriptorPool;
    use descriptor::descriptor_set::PoolCapacity;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let _pool = DescriptorPool::new(&device, false, PoolCapacity {
            sets: 10,
            sampler: 1,
            combined_image_sampler: 1,
            sampled_image: 1,
            storage_image: 1,
            input_attachment: 1,
            uniform_buffer: 1,
            storage_buffer: 1,
            uniform_buffer_dynamic: 1,
            storage_buffer_dynamic: 1,
            uniform_texel_buffer: 1,
            storage_texel_buffer: 1,
        });
    }

    #[test]
    fn device() {
        let (device, _) = gfx_dev_and_queue!();
        let pool = DescriptorPool::new(&device, false, PoolCapacity {
            sets: 10,
            sampler: 1,
            combined_image_sampler: 1,
            sampled_image: 1,
            storage_image: 1,
            input_attachment: 1,
            uniform_buffer: 1,
            storage_buffer: 1,
            uniform_buffer_dynamic: 1,
            storage_buffer_dynamic: 1,
            uniform_texel_buffer: 1,
            storage_texel_buffer: 1,
        });
        assert_eq!(&**pool.device() as *const _, &*device as *const _);
    }

    #[test]
    #[should_panic = "Descriptor pool capacity must not be 0"]
    fn zero_set() {
        let (device, _) = gfx_dev_and_queue!();
        let _pool = DescriptorPool::new(&device, false, PoolCapacity {
            sets: 0,
            sampler: 1,
            combined_image_sampler: 1,
            sampled_image: 1,
            storage_image: 1,
            input_attachment: 1,
            uniform_buffer: 1,
            storage_buffer: 1,
            uniform_buffer_dynamic: 1,
            storage_buffer_dynamic: 1,
            uniform_texel_buffer: 1,
            storage_texel_buffer: 1,
        });
    }

    #[test]
    #[should_panic = "The number of descriptors in a descriptor pool can't be 0"]
    fn zero_descrs() {
        let (device, _) = gfx_dev_and_queue!();
        let _pool = DescriptorPool::new(&device, false, PoolCapacity {
            sets: 10,
            sampler: 0,
            combined_image_sampler: 0,
            sampled_image: 0,
            storage_image: 0,
            input_attachment: 0,
            uniform_buffer: 0,
            storage_buffer: 0,
            uniform_buffer_dynamic: 0,
            storage_buffer_dynamic: 0,
            uniform_texel_buffer: 0,
            storage_texel_buffer: 0,
        });
    }
}
