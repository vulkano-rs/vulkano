// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;
use smallvec::SmallVec;

use command_buffer::pool::AllocatedCommandBuffer;
use instance::QueueFamily;

use device::Device;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

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
    /// - Panics if the queue family doesn't belong to the same physical device as `device`.
    ///
    pub fn new(device: &Arc<Device>, queue_family: QueueFamily, transient: bool, reset_cb: bool) -> Result<UnsafeCommandPool, OomError> {
        assert_eq!(device.physical_device().internal_object(),
                   queue_family.physical_device().internal_object());

        let vk = device.pointers();

        let flags = {
            let flag1 = if transient {
                vk::COMMAND_POOL_CREATE_TRANSIENT_BIT
            } else {
                0
            };
            let flag2 = if reset_cb {
                vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
            } else {
                0
            };
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
            try!(check_errors(vk.CreateCommandPool(device.internal_object(), &infos, ptr::null(), &mut output)));
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
        let flags = if release_resources {
            vk::COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT
        } else {
            0
        };

        let vk = self.device.pointers();
        try!(check_errors(vk.ResetCommandPool(self.device.internal_object(), self.pool, flags)));
        Ok(())
    }

    /// Allocates `count` command buffers.
    ///
    /// If `secondary` is true, allocates secondary command buffers. Otherwise, allocates primary
    /// command buffers.
    pub fn alloc_command_buffers(&self, secondary: bool, count: usize) -> Result<UnsafeCommandPoolAllocIter, OomError> {
        if count == 0 {
            return Ok(UnsafeCommandPoolAllocIter(None));
        }

        let infos = vk::CommandBufferAllocateInfo {
            sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: self.pool,
            level: if secondary {
                vk::COMMAND_BUFFER_LEVEL_SECONDARY
            } else {
                vk::COMMAND_BUFFER_LEVEL_PRIMARY
            },
            commandBufferCount: count as u32,
        };

        unsafe {
            let vk = self.device.pointers();
            let mut out = Vec::with_capacity(count);
            try!(check_errors(vk.AllocateCommandBuffers(self.device.internal_object(), &infos, out.as_mut_ptr())));

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
        vk.FreeCommandBuffers(self.device.internal_object(),
                              self.pool,
                              command_buffers.len() as u32,
                              command_buffers.as_ptr())
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
