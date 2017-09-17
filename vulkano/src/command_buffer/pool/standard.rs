// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crossbeam::sync::MsQueue;
use fnv::FnvHashMap;
use std::collections::hash_map::Entry;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;
use std::thread;
use std::vec::IntoIter as VecIntoIter;

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
/// It is guaranteed that the allocated command buffers keep the `Arc<StandardCommandPool>` alive.
/// This is desirable so that we can store a `Weak<StandardCommandPool>`.
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
    per_thread: Mutex<FnvHashMap<thread::ThreadId, Weak<StandardCommandPoolPerThread>>>,
}

unsafe impl Send for StandardCommandPool {
}
unsafe impl Sync for StandardCommandPool {
}

struct StandardCommandPoolPerThread {
    // The Vulkan pool of this thread.
    pool: Mutex<UnsafeCommandPool>,
    // List of existing primary command buffers that are available for reuse.
    available_primary_command_buffers: MsQueue<UnsafeCommandPoolAlloc>,
    // List of existing secondary command buffers that are available for reuse.
    available_secondary_command_buffers: MsQueue<UnsafeCommandPoolAlloc>,
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
    type Iter = VecIntoIter<StandardCommandPoolBuilder>;
    type Builder = StandardCommandPoolBuilder;
    type Alloc = StandardCommandPoolAlloc;

    fn alloc(&self, secondary: bool, count: u32) -> Result<Self::Iter, OomError> {
        // Find the correct `StandardCommandPoolPerThread` structure.
        let mut hashmap = self.per_thread.lock().unwrap();
        // TODO: meh for iterating everything every time
        hashmap.retain(|_, w| w.upgrade().is_some());

        // Get an appropriate `Arc<StandardCommandPoolPerThread>`.
        let per_thread = match hashmap.entry(thread::current().id()) {
            Entry::Occupied(entry) => {
                // The `unwrap()` can't fail, since we retained only valid members earlier.
                entry.get().upgrade().unwrap()
            },
            Entry::Vacant(entry) => {
                let new_pool =
                    UnsafeCommandPool::new(self.device.clone(), self.queue_family(), false, true)?;
                let pt = Arc::new(StandardCommandPoolPerThread {
                                      pool: Mutex::new(new_pool),
                                      available_primary_command_buffers: MsQueue::new(),
                                      available_secondary_command_buffers: MsQueue::new(),
                                  });

                entry.insert(Arc::downgrade(&pt));
                pt
            },
        };

        // The final output.
        let mut output = Vec::with_capacity(count as usize);

        // First, pick from already-existing command buffers.
        {
            let existing = if secondary {
                &per_thread.available_secondary_command_buffers
            } else {
                &per_thread.available_primary_command_buffers
            };

            for _ in 0 .. count as usize {
                if let Some(cmd) = existing.try_pop() {
                    output.push(StandardCommandPoolBuilder {
                                    inner: StandardCommandPoolAlloc {
                                        cmd: ManuallyDrop::new(cmd),
                                        pool: per_thread.clone(),
                                        pool_parent: self.clone(),
                                        secondary: secondary,
                                        device: self.device.clone(),
                                    },
                                    dummy_avoid_send_sync: PhantomData,
                                });
                } else {
                    break;
                }
            }
        };

        // Then allocate the rest.
        if output.len() < count as usize {
            let pool_lock = per_thread.pool.lock().unwrap();
            let num_new = count as usize - output.len();

            for cmd in pool_lock.alloc_command_buffers(secondary, num_new)? {
                output.push(StandardCommandPoolBuilder {
                                inner: StandardCommandPoolAlloc {
                                    cmd: ManuallyDrop::new(cmd),
                                    pool: per_thread.clone(),
                                    pool_parent: self.clone(),
                                    secondary: secondary,
                                    device: self.device.clone(),
                                },
                                dummy_avoid_send_sync: PhantomData,
                            });
            }
        }

        // Final output.
        Ok(output.into_iter())
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
    // The actual command buffer. Extracted in the `Drop` implementation.
    cmd: ManuallyDrop<UnsafeCommandPoolAlloc>,
    // We hold a reference to the command pool for our destructor.
    pool: Arc<StandardCommandPoolPerThread>,
    // Keep alive the `StandardCommandPool`, otherwise it would be destroyed.
    pool_parent: Arc<StandardCommandPool>,
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
        &*self.cmd
    }

    #[inline]
    fn queue_family(&self) -> QueueFamily {
        let queue_family_id = self.pool.pool.lock().unwrap().queue_family().id();

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
        // Safe because `self.cmd` is wrapped in a `ManuallyDrop`.
        let cmd: UnsafeCommandPoolAlloc = unsafe { ptr::read(&*self.cmd) };

        if self.secondary {
            self.pool.available_secondary_command_buffers.push(cmd);
        } else {
            self.pool.available_primary_command_buffers.push(cmd);
        }
    }
}

#[cfg(test)]
mod tests {
    use VulkanObject;
    use command_buffer::pool::CommandPool;
    use command_buffer::pool::CommandPoolBuilderAlloc;
    use command_buffer::pool::StandardCommandPool;
    use device::Device;
    use std::sync::Arc;

    #[test]
    fn reuse_command_buffers() {
        let (device, _) = gfx_dev_and_queue!();
        let queue_family = device.physical_device().queue_families().next().unwrap();

        let pool = Device::standard_command_pool(&device, queue_family);

        let cb = pool.alloc(false, 1).unwrap().next().unwrap();
        let raw = cb.inner().internal_object();
        drop(cb);

        let cb2 = pool.alloc(false, 1).unwrap().next().unwrap();
        assert_eq!(raw, cb2.inner().internal_object());
    }

    #[test]
    fn pool_kept_alive_by_allocs() {
        let (device, queue) = gfx_dev_and_queue!();

        let pool = Arc::new(StandardCommandPool::new(device, queue.family()));
        let pool_weak = Arc::downgrade(&pool);

        let cb = pool.alloc(false, 1).unwrap().next().unwrap();
        drop(pool);
        assert!(pool_weak.upgrade().is_some());

        drop(cb);
        assert!(pool_weak.upgrade().is_none());
    }
}
