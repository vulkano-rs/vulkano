// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use fnv::FnvHashMap;
use std::collections::hash_map::Entry;
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
        // TODO: meh for iterating everything every time
        hashmap.retain(|_, w| w.upgrade().is_some());

        // Get an appropriate `Arc<Mutex<StandardCommandPoolPerThread>>`.
        let per_thread = match hashmap.entry(thread::current().id()) {
            Entry::Occupied(mut entry) => {
                if let Some(entry) = entry.get().upgrade() {
                    entry
                } else {
                    let new_pool =
                        UnsafeCommandPool::new(self.device.clone(), self.queue_family(), false, true)?;
                    let pt = Arc::new(Mutex::new(StandardCommandPoolPerThread {
                                                    pool: new_pool,
                                                    available_primary_command_buffers: Vec::new(),
                                                    available_secondary_command_buffers: Vec::new(),
                                                }));

                    entry.insert(Arc::downgrade(&pt));
                    pt
                }
            },
            Entry::Vacant(entry) => {
                let new_pool =
                    UnsafeCommandPool::new(self.device.clone(), self.queue_family(), false, true)?;
                let pt = Arc::new(Mutex::new(StandardCommandPoolPerThread {
                                                 pool: new_pool,
                                                 available_primary_command_buffers: Vec::new(),
                                                 available_secondary_command_buffers: Vec::new(),
                                             }));

                entry.insert(Arc::downgrade(&pt));
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
        let per_thread = per_thread.clone();
        let final_iter = from_existing
            .chain(newly_allocated)
            .map(move |cmd| {
                StandardCommandPoolBuilder {
                    inner: StandardCommandPoolAlloc {
                        cmd: Some(cmd),
                        pool: per_thread.clone(),
                        secondary: secondary,
                        device: device.clone(),
                    },
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

/// Command buffer allocated from a `StandardCommandPool` and that is currently being built.
pub struct StandardCommandPoolBuilder {
    // The only difference between a `StandardCommandPoolBuilder` and a `StandardCommandPoolAlloc`
    // is that the former must not implement `Send` and `Sync`. Therefore we just share the structs.
    inner: StandardCommandPoolAlloc,
    // Unimplemented `Send` and `Sync` from the builder.
    dummy_avoid_send_sync: PhantomData<*const u8>,
}

unsafe impl CommandPoolBuilderAlloc for StandardCommandPoolBuilder {
    type Alloc = StandardCommandPoolAlloc;

    #[inline]
    fn inner(&self) -> &UnsafeCommandPoolAlloc {
        self.inner.inner()
    }

    #[inline]
    fn into_alloc(self) -> Self::Alloc {
        self.inner
    }

    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.inner.queue_family()
    }
}

unsafe impl DeviceOwned for StandardCommandPoolBuilder {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

/// Command buffer allocated from a `StandardCommandPool`.
pub struct StandardCommandPoolAlloc {
    // The actual command buffer. Must always be `Some`. Value extracted in the destructor.
    cmd: Option<UnsafeCommandPoolAlloc>,
    // We hold a reference to the command pool for our destructor.
    pool: Arc<Mutex<StandardCommandPoolPerThread>>,
    // True if secondary command buffer.
    secondary: bool,
    // The device we belong to. Necessary because of the `DeviceOwned` trait implementation.
    device: Arc<Device>,
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
        let queue_family_id = self.pool.lock().unwrap().pool.queue_family().id();

        self.device
            .physical_device()
            .queue_family_by_id(queue_family_id)
            .unwrap()
    }
}

unsafe impl DeviceOwned for StandardCommandPoolAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        // Note that we could grab the device from `self.pool`. Unfortunately this requires a mutex
        // lock, so it isn't compatible with the API of `DeviceOwned`.
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
