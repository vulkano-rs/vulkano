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

use vk;

use device::Device;

macro_rules! query_pool {
    ($name:ident, $query_type:expr) => {
        pub struct $name {
            device: Arc<Device>,
            pool: VkQueryPool,
            num_slots: u32,
        }

        impl $name {
            /// Builds a new query pool.
            pub fn new(device: Arc<Device>, num_slots: u32) -> Arc<$name> {
                let create_infos = VkQueryPoolCreateInfo {
                    sType: vk::QUERY_POOL_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,
                    queryType: $query_type,
                    num_slots: num_slots,
                    pipelineStatistics: 0,      // TODO: 
                };

                let mut output = mem::uninitialized();
                vkCreateQueryPool(device.internal_object(), &create_infos, ptr::null(), &mut output);
            }

            /// Returns the number of slots of that query pool.
            #[inline]
            pub fn num_slots(&self) -> u32 {
                self.num_slots
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    vkDestroyQueryPool
                }
            }
        }
    };
}

query_pool!(OcclusionQueriesPool, vk::QUERY_TYPE_OCCLUSION);
