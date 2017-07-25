// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use fnv::FnvHashMap;
use std::cmp;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;
use std::thread;

use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolAlloc;
use command_buffer::pool::CommandPoolBuilderAlloc;
use command_buffer::pool::UnsafeCommandPool;
use command_buffer::pool::UnsafeCommandPoolAlloc;
use instance::QueueFamily;

use OomError;
use VulkanObject;
use device::Device;
use device::DeviceOwned;

/// Standard implementation of a command pool.
///
/// Will use one Vulkan pool per thread in order to avoid locking. Will try to reuse command
/// buffers. Command buffers can't be moved between threads during the building process, but
/// finished command buffers can.
pub struct StandardCommandPool {
    // The device.
    device: Arc<Device>,

    // Identifier of the queue family.
    queue_family: u32,

    // For each thread, we store thread-specific info.
    per_thread: Mutex<FnvHashMap<thread::ThreadId, Weak<Mutex<StandardCommandPoolPerThread>>>>,
}

unsafe impl Send for StandardCommandPool {
}
unsafe impl Sync for StandardCommandPool {
}

struct StandardCommandPoolPerThread {
    // The Vulkan pool of this thread.
    pool: UnsafeCommandPool,
    // List of existing primary command buffers that are available for reuse.
    available_primary_command_buffers: Vec<UnsafeCommandPoolAlloc>,
    // List of existing secondary command buffers that are available for reuse.
    available_secondary_command_buffers: Vec<UnsafeCommandPoolAlloc>,
}

impl StandardCommandPool {
    /// Builds a new pool.
    ///
    /// # Panic
    ///
    /// - Panics if the device and the queue family don't belong to the same physical device.
    ///
    pub fn new(device: Arc<Device>, queue_family: QueueFamily) -> StandardCommandPool {
        assert_eq!(device.physical_device().internal_object(),
                   queue_family.physical_device().internal_object());

        StandardCommandPool {
            device: device,
            queue_family: queue_family.id(),
            per_thread: Mutex::new(Default::default()),
        }
    }
}

unsafe impl CommandPool for Arc<StandardCommandPool> {
    type Iter = Box<Iterator<Item = StandardCommandPoolBuilder>>; // TODO: meh for Box
    type Builder = StandardCommandPoolBuilder;
    type Alloc = StandardCommandPoolAlloc;

    fn alloc(&self, secondary: bool, count: u32) -> Result<Self::Iter, OomError> {
        // Find the correct `StandardCommandPoolPerThread` structure.
        let mut hashmap = self.per_thread.lock().unwrap();
        //hashmap.retain(|_, w| w.upgrade().is_some());     // TODO: unstable     // TODO: meh for iterating everything every time

        // TODO: this hashmap lookup can probably be optimized
        let curr_thread_id = thread::current().id();
        let per_thread = hashmap.get(&curr_thread_id).and_then(|p| p.upgrade());
        let per_thread = match per_thread {
            Some(pt) => pt,
            None => {
                let new_pool =
                    UnsafeCommandPool::new(self.device.clone(), self.queue_family(), false, true)?;
                let pt = Arc::new(Mutex::new(StandardCommandPoolPerThread {
                                                 pool: new_pool,
                                                 available_primary_command_buffers: Vec::new(),
                                                 available_secondary_command_buffers: Vec::new(),
                                             }));

                hashmap.insert(curr_thread_id, Arc::downgrade(&pt));
                pt
            },
        };

        let mut pt_lock = per_thread.lock().unwrap();

        // Build an iterator to pick from already-existing command buffers.
        let (num_from_existing, from_existing) = {
            // Which list of already-existing command buffers we are going to pick CBs from.
            let mut existing = if secondary {
                &mut pt_lock.available_secondary_command_buffers
            } else {
                &mut pt_lock.available_primary_command_buffers
            };
            let num_from_existing = cmp::min(count as usize, existing.len());
            let from_existing = existing
                .drain(0 .. num_from_existing)
                .collect::<Vec<_>>()
                .into_iter();
            (num_from_existing, from_existing)
        };

        // Build an iterator to construct the missing command buffers from the Vulkan pool.
        let num_new = count as usize - num_from_existing;
        debug_assert!(num_new <= count as usize); // Check overflows.
        let newly_allocated = pt_lock.pool.alloc_command_buffers(secondary, num_new)?;

        // Returning them as a chain.
        let device = self.device.clone();
        let queue_family_id = self.queue_family;
        let per_thread = per_thread.clone();
        let final_iter = from_existing
            .chain(newly_allocated)
            .map(move |cmd| {
                StandardCommandPoolBuilder {
                    cmd: Some(cmd),
                    pool: per_thread.clone(),
                    secondary: secondary,
                    device: device.clone(),
                    queue_family_id: queue_family_id,
                    dummy_avoid_send_sync: PhantomData,
                }
            })
            .collect::<Vec<_>>();

        Ok(Box::new(final_iter.into_iter()))
    }

    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.device
            .physical_device()
            .queue_family_by_id(self.queue_family)
            .unwrap()
    }
}

unsafe impl DeviceOwned for StandardCommandPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

pub struct StandardCommandPoolBuilder {
    cmd: Option<UnsafeCommandPoolAlloc>,
    pool: Arc<Mutex<StandardCommandPoolPerThread>>,
    secondary: bool,
    device: Arc<Device>,
    queue_family_id: u32,
    dummy_avoid_send_sync: PhantomData<*const u8>,
}

unsafe impl CommandPoolBuilderAlloc for StandardCommandPoolBuilder {
    type Alloc = StandardCommandPoolAlloc;

    #[inline]
    fn inner(&self) -> &UnsafeCommandPoolAlloc {
        self.cmd.as_ref().unwrap()
    }

    #[inline]
    fn into_alloc(mut self) -> Self::Alloc {
        StandardCommandPoolAlloc {
            cmd: Some(self.cmd.take().unwrap()),
            pool: self.pool.clone(),
            secondary: self.secondary,
            device: self.device.clone(),
            queue_family_id: self.queue_family_id,
        }
    }

    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.device
            .physical_device()
            .queue_family_by_id(self.queue_family_id)
            .unwrap()
    }
}

unsafe impl DeviceOwned for StandardCommandPoolBuilder {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Drop for StandardCommandPoolBuilder {
    fn drop(&mut self) {
        if let Some(cmd) = self.cmd.take() {
            let mut pool = self.pool.lock().unwrap();

            if self.secondary {
                pool.available_secondary_command_buffers.push(cmd);
            } else {
                pool.available_primary_command_buffers.push(cmd);
            }
        }
    }
}

pub struct StandardCommandPoolAlloc {
    cmd: Option<UnsafeCommandPoolAlloc>,
    pool: Arc<Mutex<StandardCommandPoolPerThread>>,
    secondary: bool,
    device: Arc<Device>,
    queue_family_id: u32,
}

unsafe impl Send for StandardCommandPoolAlloc {
}
unsafe impl Sync for StandardCommandPoolAlloc {
}

unsafe impl CommandPoolAlloc for StandardCommandPoolAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeCommandPoolAlloc {
        self.cmd.as_ref().unwrap()
    }

    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.device
            .physical_device()
            .queue_family_by_id(self.queue_family_id)
            .unwrap()
    }
}

unsafe impl DeviceOwned for StandardCommandPoolAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Drop for StandardCommandPoolAlloc {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();

        if self.secondary {
            pool.available_secondary_command_buffers
                .push(self.cmd.take().unwrap());
        } else {
            pool.available_primary_command_buffers
                .push(self.cmd.take().unwrap());
        }
    }
}
