// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::collections::HashMap;
use std::iter::Chain;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::vec::IntoIter as VecIntoIter;

use command_buffer::pool::AllocatedCommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolFinished;
use command_buffer::pool::UnsafeCommandPool;
use command_buffer::pool::UnsafeCommandPoolAllocIter;
use instance::QueueFamily;

use device::Device;
use OomError;
use VulkanObject;

// Since the stdlib doesn't have a "thread ID" yet, we store a `Box<u8>` for each thread and the
// value of the pointer will be used as a thread id.
thread_local!(static THREAD_ID: Box<u8> = Box::new(0));
#[inline]
fn curr_thread_id() -> usize { THREAD_ID.with(|data| &**data as *const u8 as usize) }

/// Standard implementation of a command pool.
///
/// Will use one pool per thread in order to avoid locking. Will try to reuse command buffers.
/// Locking is required only when allocating/freeing command buffers.
pub struct StandardCommandPool {
    device: Arc<Device>,
    queue_family: u32,
    per_thread: Mutex<HashMap<usize, StandardCommandPoolPerThread>>,

    // Dummy marker in order to not implement `Send` and `Sync`.
    //
    // Since `StandardCommandPool` isn't Send/Sync, then the command buffers that use this pool
    // won't be Send/Sync either, which means that we don't need to lock the pool while the CB
    // is being built.
    //
    // However `StandardCommandPoolFinished` *is* Send/Sync because the only operation that can
    // be called on `StandardCommandPoolFinished` is freeing, and freeing does actually lock.
    dummy_avoid_send_sync: PhantomData<*const u8>,
}

impl StandardCommandPool {
    /// Builds a new pool.
    ///
    /// # Panic
    ///
    /// - Panicks if the device and the queue family don't belong to the same physical device.
    ///
    pub fn new(device: &Arc<Device>, queue_family: QueueFamily) -> StandardCommandPool {
        assert_eq!(device.physical_device().internal_object(),
                   queue_family.physical_device().internal_object());

        StandardCommandPool {
            device: device.clone(),
            queue_family: queue_family.id(),
            per_thread: Mutex::new(HashMap::new()),
            dummy_avoid_send_sync: PhantomData,
        }
    }
}

struct StandardCommandPoolPerThread {
    pool: UnsafeCommandPool,
    available_primary_command_buffers: Vec<AllocatedCommandBuffer>,
    available_secondary_command_buffers: Vec<AllocatedCommandBuffer>,
}

unsafe impl CommandPool for Arc<StandardCommandPool> {
    type Iter = Chain<VecIntoIter<AllocatedCommandBuffer>, UnsafeCommandPoolAllocIter>;
    type Lock = ();
    type Finished = StandardCommandPoolFinished;

    fn alloc(&self, secondary: bool, count: u32) -> Result<Self::Iter, OomError> {
        let mut per_thread = self.per_thread.lock().unwrap();
        let mut per_thread = per_thread.entry(curr_thread_id())
                                     .or_insert_with(|| {
                                         StandardCommandPoolPerThread {
                                             pool: UnsafeCommandPool::new(&self.device, self.queue_family(), false, true).unwrap(),     // FIXME: return error instead
                                             available_primary_command_buffers: Vec::new(),
                                             available_secondary_command_buffers: Vec::new(),
                                         }
                                      });

        let mut existing = if secondary { &mut per_thread.available_secondary_command_buffers }
                           else { &mut per_thread.available_primary_command_buffers };

        let num_from_existing = cmp::min(count as usize, existing.len());
        let from_existing = existing.drain(0 .. num_from_existing).collect::<Vec<_>>().into_iter();

        let num_new = count as usize - num_from_existing;
        debug_assert!(num_new <= count as usize);        // Check overflows.
        let newly_allocated = try!(per_thread.pool.alloc_command_buffers(secondary, num_new));

        Ok(from_existing.chain(newly_allocated))
    }

    unsafe fn free<I>(&self, secondary: bool, command_buffers: I)
        where I: Iterator<Item = AllocatedCommandBuffer>
    {
        let mut per_thread = self.per_thread.lock().unwrap();
        let mut per_thread = per_thread.get_mut(&curr_thread_id()).unwrap();

        if secondary {
            for cb in command_buffers {
                per_thread.available_secondary_command_buffers.push(cb);
            }
        } else {
            for cb in command_buffers {
                per_thread.available_primary_command_buffers.push(cb);
            }
        }
    }

    #[inline]
    fn finish(self) -> Self::Finished {
        StandardCommandPoolFinished {
            pool: self,
            thread_id: curr_thread_id(),
        }
    }

    #[inline]
    fn lock(&self) -> Self::Lock {
        ()
    }

    #[inline]
    fn can_reset_invidual_command_buffers(&self) -> bool {
        true
    }

    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }

    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.device.physical_device().queue_family_by_id(self.queue_family).unwrap()
    }
}

pub struct StandardCommandPoolFinished {
    pool: Arc<StandardCommandPool>,
    thread_id: usize,
}

unsafe impl CommandPoolFinished for StandardCommandPoolFinished {
    unsafe fn free<I>(&self, secondary: bool, command_buffers: I)
        where I: Iterator<Item = AllocatedCommandBuffer>
    {
        let mut per_thread = self.pool.per_thread.lock().unwrap();
        let mut per_thread = per_thread.get_mut(&curr_thread_id()).unwrap();

        if secondary {
            for cb in command_buffers {
                per_thread.available_secondary_command_buffers.push(cb);
            }
        } else {
            for cb in command_buffers {
                per_thread.available_primary_command_buffers.push(cb);
            }
        }
    }

    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.pool.device()
    }

    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.pool.queue_family()
    }
}

// See `StandardCommandPool` for comments about this.
unsafe impl Send for StandardCommandPoolFinished {}
unsafe impl Sync for StandardCommandPoolFinished {}
