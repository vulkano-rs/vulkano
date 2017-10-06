// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

use instance::QueueFamily;

use Error;
use OomError;
use VulkanObject;
use check_errors;
use device::Device;
use device::DeviceOwned;
use vk;

/// Low-level implementation of a command pool.
///
/// A command pool is always tied to a specific queue family. Command buffers allocated from a pool
/// can only be executed on the corresponding queue familiy.
///
/// This struct doesn't implement the `Sync` trait because Vulkan command pools are not thread
/// safe. In other words, you can only use a pool from one thread at a time.
#[derive(Debug)]
pub struct UnsafeCommandPool {
    pool: vk::CommandPool,
    device: Arc<Device>,

    // Index of the associated queue family in the physical device.
    queue_family_index: u32,

    // We don't want `UnsafeCommandPool` to implement Sync.
    // This marker unimplements both Send and Sync, but we reimplement Send manually right under.
    dummy_avoid_sync: PhantomData<*const u8>,
}

unsafe impl Send for UnsafeCommandPool {
}

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
    pub fn new(device: Arc<Device>, queue_family: QueueFamily, transient: bool, reset_cb: bool)
               -> Result<UnsafeCommandPool, OomError> {
        assert_eq!(device.physical_device().internal_object(),
                   queue_family.physical_device().internal_object(),
                   "Device doesn't match physical device when creating a command pool");

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
            check_errors(vk.CreateCommandPool(device.internal_object(),
                                              &infos,
                                              ptr::null(),
                                              &mut output))?;
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
    /// If `release_resources` is true, it is a hint to the implementation that it should free all
    /// the memory internally allocated for this pool.
    ///
    /// # Safety
    ///
    /// The command buffers allocated from this pool jump to the initial state.
    ///
    pub unsafe fn reset(&self, release_resources: bool) -> Result<(), OomError> {
        let flags = if release_resources {
            vk::COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT
        } else {
            0
        };

        let vk = self.device.pointers();
        check_errors(vk.ResetCommandPool(self.device.internal_object(), self.pool, flags))?;
        Ok(())
    }

    /// Trims a command pool, which recycles unused internal memory from the command pool back to
    /// the system.
    ///
    /// Command buffers allocated from the pool are not affected by trimming.
    ///
    /// This function is supported only if the `VK_KHR_maintenance1` extension was enabled at
    /// device creation. Otherwise an error is returned.
    /// Since this operation is purely an optimization it is legitimate to call this function and
    /// simply ignore any possible error.
    pub fn trim(&self) -> Result<(), CommandPoolTrimError> {
        unsafe {
            if !self.device.loaded_extensions().khr_maintenance1 {
                return Err(CommandPoolTrimError::Maintenance1ExtensionNotEnabled);
            }

            let vk = self.device.pointers();
            vk.TrimCommandPoolKHR(self.device.internal_object(),
                                  self.pool,
                                  0 /* reserved */);
            Ok(())
        }
    }

    /// Allocates `count` command buffers.
    ///
    /// If `secondary` is true, allocates secondary command buffers. Otherwise, allocates primary
    /// command buffers.
    pub fn alloc_command_buffers(&self, secondary: bool, count: usize)
                                 -> Result<UnsafeCommandPoolAllocIter, OomError> {
        if count == 0 {
            return Ok(UnsafeCommandPoolAllocIter { list: vec![].into_iter() });
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
            check_errors(vk.AllocateCommandBuffers(self.device.internal_object(),
                                                   &infos,
                                                   out.as_mut_ptr()))?;

            out.set_len(count);

            Ok(UnsafeCommandPoolAllocIter { list: out.into_iter() })
        }
    }

    /// Frees individual command buffers.
    ///
    /// # Safety
    ///
    /// The command buffers must have been allocated from this pool. They must not be in use.
    ///
    pub unsafe fn free_command_buffers<I>(&self, command_buffers: I)
        where I: Iterator<Item = UnsafeCommandPoolAlloc>
    {
        let command_buffers: SmallVec<[_; 4]> = command_buffers.map(|cb| cb.0).collect();
        let vk = self.device.pointers();
        vk.FreeCommandBuffers(self.device.internal_object(),
                              self.pool,
                              command_buffers.len() as u32,
                              command_buffers.as_ptr())
    }

    /// Returns the queue family on which command buffers of this pool can be executed.
    #[inline]
    pub fn queue_family(&self) -> QueueFamily {
        self.device
            .physical_device()
            .queue_family_by_id(self.queue_family_index)
            .unwrap()
    }
}

unsafe impl DeviceOwned for UnsafeCommandPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
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

/// Opaque type that represents a command buffer allocated from a pool.
pub struct UnsafeCommandPoolAlloc(vk::CommandBuffer);

unsafe impl VulkanObject for UnsafeCommandPoolAlloc {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.0
    }
}

/// Iterator for newly-allocated command buffers.
#[derive(Debug)]
pub struct UnsafeCommandPoolAllocIter {
    list: VecIntoIter<vk::CommandBuffer>,
}

impl Iterator for UnsafeCommandPoolAllocIter {
    type Item = UnsafeCommandPoolAlloc;

    #[inline]
    fn next(&mut self) -> Option<UnsafeCommandPoolAlloc> {
        self.list.next().map(|cb| UnsafeCommandPoolAlloc(cb))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.list.size_hint()
    }
}

impl ExactSizeIterator for UnsafeCommandPoolAllocIter {
}

/// Error that can happen when trimming command pools.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CommandPoolTrimError {
    /// The `KHR_maintenance1` extension was not enabled.
    Maintenance1ExtensionNotEnabled,
}

impl error::Error for CommandPoolTrimError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CommandPoolTrimError::Maintenance1ExtensionNotEnabled =>
                "the `KHR_maintenance1` extension was not enabled",
        }
    }
}

impl fmt::Display for CommandPoolTrimError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for CommandPoolTrimError {
    #[inline]
    fn from(err: Error) -> CommandPoolTrimError {
        panic!("unexpected error: {:?}", err)
    }
}

#[cfg(test)]
mod tests {
    use command_buffer::pool::CommandPoolTrimError;
    use command_buffer::pool::UnsafeCommandPool;

    #[test]
    fn basic_create() {
        let (device, queue) = gfx_dev_and_queue!();
        let _ = UnsafeCommandPool::new(device, queue.family(), false, false).unwrap();
    }

    #[test]
    fn queue_family_getter() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = UnsafeCommandPool::new(device, queue.family(), false, false).unwrap();
        assert_eq!(pool.queue_family().id(), queue.family().id());
    }

    #[test]
    fn panic_if_not_match_family() {
        let (device, _) = gfx_dev_and_queue!();
        let (_, queue) = gfx_dev_and_queue!();

        assert_should_panic!("Device doesn't match physical device when creating a command pool", {
            let _ = UnsafeCommandPool::new(device, queue.family(), false, false);
        });
    }

    #[test]
    fn check_maintenance_when_trim() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = UnsafeCommandPool::new(device, queue.family(), false, false).unwrap();

        match pool.trim() {
            Err(CommandPoolTrimError::Maintenance1ExtensionNotEnabled) => (),
            _ => panic!(),
        }
    }

    // TODO: test that trim works if VK_KHR_maintenance1 if enabled ; the test macro doesn't
    //       support enabling extensions yet

    #[test]
    fn basic_alloc() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = UnsafeCommandPool::new(device, queue.family(), false, false).unwrap();
        let iter = pool.alloc_command_buffers(false, 12).unwrap();
        assert_eq!(iter.count(), 12);
    }
}
