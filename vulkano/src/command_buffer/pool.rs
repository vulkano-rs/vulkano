// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! In the Vulkan API, command buffers must be allocated from *command pools*.
//! 
//! A command pool holds and manages the memory of one or more command buffers. If you destroy a
//! command pool, all of its command buffers are automatically destroyed.
//! 
//! In vulkano, creating a command buffer requires passing an implementation of the `CommandPool`
//! trait. By default vulkano will use the `StandardCommandPool` struct, but you can implement
//! this trait yourself by wrapping around the `UnsafeCommandPool` type.

use std::cmp;
use std::collections::HashMap;
use std::iter::Chain;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::vec::IntoIter as VecIntoIter;
use smallvec::SmallVec;

use instance::QueueFamily;

use device::Device;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Types that manage the memory of command buffers.
pub unsafe trait CommandPool {
    /// See `alloc()`.
    type Iter: Iterator<Item = AllocatedCommandBuffer>;
    /// See `lock()`.
    type Lock;
    /// See `finish()`.
    type Finished: CommandPoolFinished;

    /// Allocates command buffers from this pool.
    fn alloc(&self, secondary: bool, count: u32) -> Result<Self::Iter, OomError>;

    /// Frees command buffers from this pool.
    ///
    /// # Safety
    ///
    /// - The command buffers must have been allocated from this pool.
    /// - `secondary` must have the same value as what was passed to `alloc`.
    ///
    unsafe fn free<I>(&self, secondary: bool, command_buffers: I)
        where I: Iterator<Item = AllocatedCommandBuffer>;

    /// Once a command buffer has finished being built, it should call this method in order to
    /// produce a `Finished` object.
    ///
    /// The `Finished` object must hold the pool alive.
    ///
    /// The point of this object is to change the Send/Sync strategy after a command buffer has
    /// finished being built compared to before.
    fn finish(self) -> Self::Finished;

    /// Before any command buffer allocated from this pool can be modified, the pool itself must
    /// be locked by calling this method.
    ///
    /// All the operations are atomic at the thread level, so the point of this lock is to
    /// prevent the pool from being accessed from multiple threads in parallel.
    fn lock(&self) -> Self::Lock;

    /// Returns true if command buffers can be reset individually. In other words, if the pool
    /// was created with `reset_cb` set to true.
    fn can_reset_invidual_command_buffers(&self) -> bool;

    /// Returns the device used to create this pool.
    fn device(&self) -> &Arc<Device>;

    /// Returns the queue family that this pool targets.
    fn queue_family(&self) -> QueueFamily;
}

/// See `CommandPool::finish()`.
pub unsafe trait CommandPoolFinished {
    /// Frees command buffers.
    ///
    /// # Safety
    ///
    /// - The command buffers must have been allocated from this pool.
    /// - `secondary` must have the same value as what was passed to `alloc`.
    ///
    unsafe fn free<I>(&self, secondary: bool, command_buffers: I)
        where I: Iterator<Item = AllocatedCommandBuffer>;

    /// Returns the device used to create this pool.
    fn device(&self) -> &Arc<Device>;

    /// Returns the queue family that this pool targets.
    fn queue_family(&self) -> QueueFamily;
}

/// Opaque type that represents a command buffer allocated from a pool.
pub struct AllocatedCommandBuffer(vk::CommandBuffer);

impl From<vk::CommandBuffer> for AllocatedCommandBuffer {
    #[inline]
    fn from(cmd: vk::CommandBuffer) -> AllocatedCommandBuffer {
        AllocatedCommandBuffer(cmd)
    }
}

unsafe impl VulkanObject for AllocatedCommandBuffer {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.0
    }
}

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

/// Low-level implementation of a command pool.
pub struct UnsafeCommandPool {
    pool: vk::CommandPool,
    device: Arc<Device>,
    queue_family_index: u32,

    // We don't want `UnsafeCommandPool` to implement Sync, since the Vulkan command pool isn't
    // thread safe.
    //
    // This marker unimplements both Send and Sync, but we reimplement Send manually right under.
    dummy_avoid_sync: PhantomData<*const u8>,
}

unsafe impl Send for UnsafeCommandPool {}

impl UnsafeCommandPool {
    /// Creates a new pool.
    ///
    /// The command buffers created with this pool can only be executed on queues of the given
    /// family.
    ///
    /// Setting `transient` to true is a hint to the implementation that the command buffers will
    /// be short-lived.
    /// Setting `reset_cb` to true means that command buffers can be reset individually.
    ///
    /// # Panic
    ///
    /// - Panicks if the queue family doesn't belong to the same physical device as `device`.
    ///
    pub fn new(device: &Arc<Device>, queue_family: QueueFamily, transient: bool,
               reset_cb: bool) -> Result<UnsafeCommandPool, OomError>
    {
        assert_eq!(device.physical_device().internal_object(),
                   queue_family.physical_device().internal_object());

        let vk = device.pointers();

        let flags = {
            let flag1 = if transient { vk::COMMAND_POOL_CREATE_TRANSIENT_BIT } else { 0 };
            let flag2 = if reset_cb { vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT }
                        else { 0 };
            flag1 | flag2
        };

        let pool = unsafe {
            let infos = vk::CommandPoolCreateInfo {
                sType: vk::STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: flags,
                queueFamilyIndex: queue_family.id(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateCommandPool(device.internal_object(), &infos,
                                                   ptr::null(), &mut output)));
            output
        };

        Ok(UnsafeCommandPool {
            pool: pool,
            device: device.clone(),
            queue_family_index: queue_family.id(),
            dummy_avoid_sync: PhantomData,
        })
    }

    /// Resets the pool, which resets all the command buffers that were allocated from it.
    ///
    /// # Safety
    ///
    /// The command buffers allocated from this pool jump to the initial state.
    ///
    #[inline]
    pub unsafe fn reset(&self, release_resources: bool) -> Result<(), OomError> {
        let flags = if release_resources { vk::COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT }
                    else { 0 };

        let vk = self.device.pointers();
        try!(check_errors(vk.ResetCommandPool(self.device.internal_object(), self.pool, flags)));
        Ok(())
    }

    /// Allocates `count` command buffers.
    ///
    /// If `secondary` is true, allocates secondary command buffers. Otherwise, allocates primary
    /// command buffers.
    pub fn alloc_command_buffers(&self, secondary: bool, count: usize)
                                 -> Result<UnsafeCommandPoolAllocIter, OomError>
    {
        if count == 0 {
            return Ok(UnsafeCommandPoolAllocIter(None));
        }

        let infos = vk::CommandBufferAllocateInfo {
            sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: self.pool,
            level: if secondary { vk::COMMAND_BUFFER_LEVEL_SECONDARY }
                   else { vk::COMMAND_BUFFER_LEVEL_PRIMARY },
            commandBufferCount: count as u32,
        };

        unsafe {
            let vk = self.device.pointers();
            let mut out = Vec::with_capacity(count);
            try!(check_errors(vk.AllocateCommandBuffers(self.device.internal_object(), &infos,
                                                        out.as_mut_ptr())));

            out.set_len(count);

            Ok(UnsafeCommandPoolAllocIter(Some(out.into_iter())))
        }
    }

    /// Frees individual command buffers.
    ///
    /// # Safety
    ///
    /// The command buffers must have been allocated from this pool.
    ///
    pub unsafe fn free_command_buffers<I>(&self, command_buffers: I)
        where I: Iterator<Item = AllocatedCommandBuffer>
    {
        let command_buffers: SmallVec<[_; 4]> = command_buffers.map(|cb| cb.0).collect();
        let vk = self.device.pointers();
        vk.FreeCommandBuffers(self.device.internal_object(), self.pool,
                              command_buffers.len() as u32, command_buffers.as_ptr())
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

unsafe impl VulkanObject for UnsafeCommandPool {
    type Object = vk::CommandPool;

    #[inline]
    fn internal_object(&self) -> vk::CommandPool {
        self.pool
    }
}

impl Drop for UnsafeCommandPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyCommandPool(self.device.internal_object(), self.pool, ptr::null());
        }
    }
}

/// Iterator for newly-allocated command buffers.
pub struct UnsafeCommandPoolAllocIter(Option<VecIntoIter<vk::CommandBuffer>>);

impl Iterator for UnsafeCommandPoolAllocIter {
    type Item = AllocatedCommandBuffer;

    #[inline]
    fn next(&mut self) -> Option<AllocatedCommandBuffer> {
        self.0.as_mut().and_then(|i| i.next()).map(|cb| AllocatedCommandBuffer(cb))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.as_ref().map(|i| i.size_hint()).unwrap_or((0, Some(0)))
    }
}

impl ExactSizeIterator for UnsafeCommandPoolAllocIter {}
