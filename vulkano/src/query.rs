// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! This module provides support for query pools.
//! 
//! In Vulkan, queries are not created individually. Instead you manipulate **query pools**, which
//! represent a collection of queries. Whenever you use a query, you have to specify both the query
//! pool and the slot id within that query pool.

use std::mem;
use std::ptr;
use std::sync::Arc;

use device::Device;

use check_errors;
use OomError;
use VulkanObject;
use VulkanPointers;
use vk;

pub struct OcclusionQueriesPool {
    pool: vk::QueryPool,
    num_slots: u32,
    device: Arc<Device>,
}

impl OcclusionQueriesPool {
    /// Builds a new query pool.
    pub fn new(device: &Arc<Device>, num_slots: u32)
               -> Result<Arc<OcclusionQueriesPool>, OomError>
    {
        let vk = device.pointers();

        let pool = unsafe {
            let infos = vk::QueryPoolCreateInfo {
                sType: vk::STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                queryType: vk::QUERY_TYPE_OCCLUSION,
                queryCount: num_slots,
                pipelineStatistics: 0,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateQueryPool(device.internal_object(), &infos,
                                                 ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(OcclusionQueriesPool {
            pool: pool,
            num_slots: num_slots,
            device: device.clone(),
        }))
    }

    /// Returns the number of slots of that query pool.
    #[inline]
    pub fn num_slots(&self) -> u32 {
        self.num_slots
    }

    /// Returns the device that was used to create this pool.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Drop for OcclusionQueriesPool {
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyQueryPool(self.device.internal_object(), self.pool, ptr::null());
        }
    }
}

#[cfg(test)]
mod tests {
    use query::OcclusionQueriesPool;

    #[test]
    fn occlusion_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = OcclusionQueriesPool::new(&device, 256).unwrap();
    }
}
