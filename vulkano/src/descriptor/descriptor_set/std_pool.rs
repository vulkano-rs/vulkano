// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use std::sync::Mutex;

use device::Device;
use device::DeviceOwned;
use descriptor::descriptor_set::DescriptorsCount;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::DescriptorPoolAlloc;
use descriptor::descriptor_set::DescriptorPoolAllocError;
use descriptor::descriptor_set::UnsafeDescriptorPool;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use OomError;

/// Standard implementation of a descriptor pool.
///
/// Whenever a set is allocated, this implementation will try to find a pool that has some space
/// for it. If there is one, allocate from it. If there is none, create a new pool whose capacity
/// is 40 sets and 40 times the requested descriptors. This number is arbitrary.
pub struct StdDescriptorPool {
    device: Arc<Device>,
    pools: Mutex<Vec<Arc<Mutex<Pool>>>>,
}

struct Pool {
    pool: UnsafeDescriptorPool,
    remaining_capacity: DescriptorsCount,
    remaining_sets_count: u32,
}

impl StdDescriptorPool {
    /// Builds a new `StdDescriptorPool`.
    pub fn new(device: Arc<Device>) -> StdDescriptorPool {
        StdDescriptorPool {
            device: device,
            pools: Mutex::new(Vec::new()),
        }
    }
}

/// A descriptor set allocated from a `StdDescriptorPool`.
pub struct StdDescriptorPoolAlloc {
    pool: Arc<Mutex<Pool>>,
    // The set. Inside an option so that we can extract it in the destructor.
    set: Option<UnsafeDescriptorSet>,
    // We need to keep track of this count in order to add it back to the capacity when freeing.
    descriptors: DescriptorsCount,
}

unsafe impl DescriptorPool for Arc<StdDescriptorPool> {
    type Alloc = StdDescriptorPoolAlloc;

    // TODO: eventually use a lock-free algorithm?
    fn alloc(&self, layout: &UnsafeDescriptorSetLayout)
             -> Result<StdDescriptorPoolAlloc, OomError>
    {
        let mut pools = self.pools.lock().unwrap();

        // Try find an existing pool with some free space.
        for pool_arc in pools.iter_mut() {
            let mut pool = pool_arc.lock().unwrap();

            if pool.remaining_sets_count == 0 {
                continue;
            }

            if !(pool.remaining_capacity >= *layout.descriptors_count()) {
                continue;
            }

            // Note that we decrease these values *before* trying to allocate from the pool.
            // If allocating from the pool results in an error, we just ignore it. In order to
            // avoid trying the same failing pool every time, we "pollute" it by reducing the
            // available space.
            pool.remaining_sets_count -= 1;
            pool.remaining_capacity -= *layout.descriptors_count();

            let alloc = unsafe {
                match pool.pool.alloc(Some(layout)) {
                    Ok(mut sets) => sets.next().unwrap(),
                    // An error can happen if we're out of memory, or if the pool is fragmented.
                    // We handle these errors by just ignoring this pool and trying the next ones.
                    Err(_) => continue,
                }
            };

            return Ok(StdDescriptorPoolAlloc {
                pool: pool_arc.clone(),
                set: Some(alloc),
                descriptors: *layout.descriptors_count(),
            });
        }

        // No existing pool can be used. Create a new one.
        // We use an arbitrary number of 40 sets and 40 times the requested descriptors.
        let count = layout.descriptors_count().clone() * 40;
        // Failure to allocate a new pool results in an error for the whole function because
        // there's no way we can recover from that.
        let mut new_pool = try!(UnsafeDescriptorPool::new(self.device.clone(), &count, 40, true));

        let alloc = unsafe {
            match new_pool.alloc(Some(layout)) {
                Ok(mut sets) => sets.next().unwrap(),
                Err(DescriptorPoolAllocError::OutOfHostMemory) => {
                    return Err(OomError::OutOfHostMemory);
                },
                Err(DescriptorPoolAllocError::OutOfDeviceMemory) => {
                    return Err(OomError::OutOfDeviceMemory);
                },
                // A fragmented pool error can't happen at the first ever allocation.
                Err(DescriptorPoolAllocError::FragmentedPool) => unreachable!(),
            }
        };

        let pool_obj = Arc::new(Mutex::new(Pool {
            pool: new_pool,
            remaining_capacity: count - *layout.descriptors_count(),
            remaining_sets_count: 40 - 1,
        }));

        pools.push(pool_obj.clone());

        Ok(StdDescriptorPoolAlloc {
            pool: pool_obj,
            set: Some(alloc),
            descriptors: *layout.descriptors_count(),
        })
    }
}

unsafe impl DeviceOwned for StdDescriptorPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl DescriptorPoolAlloc for StdDescriptorPoolAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.set.as_ref().unwrap()
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet {
        self.set.as_mut().unwrap()
    }
}

impl Drop for StdDescriptorPoolAlloc {
    // This is the destructor of a single allocation (not of the whole pool).
    fn drop(&mut self) {
        unsafe {
            let mut pool = self.pool.lock().unwrap();
            pool.pool.free(self.set.take()).unwrap();
            // Add back the capacity only after freeing, in case of a panic during the free.
            pool.remaining_sets_count += 1;
            pool.remaining_capacity += self.descriptors;
        }
    }
}
