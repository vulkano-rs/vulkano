// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::instance::QueueFamily;
use crate::Error;
use crate::OomError;
use crate::Version;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

/// Low-level implementation of a command pool.
///
/// A command pool is always tied to a specific queue family. Command buffers allocated from a pool
/// can only be executed on the corresponding queue family.
///
/// This struct doesn't implement the `Sync` trait because Vulkan command pools are not thread
/// safe. In other words, you can only use a pool from one thread at a time.
#[derive(Debug)]
pub struct UnsafeCommandPool {
    pool: ash::vk::CommandPool,
    device: Arc<Device>,

    // Index of the associated queue family in the physical device.
    queue_family_index: u32,

    // We don't want `UnsafeCommandPool` to implement Sync.
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
    pub fn new(
        device: Arc<Device>,
        queue_family: QueueFamily,
        transient: bool,
        reset_cb: bool,
    ) -> Result<UnsafeCommandPool, OomError> {
        assert_eq!(
            device.physical_device().internal_object(),
            queue_family.physical_device().internal_object(),
            "Device doesn't match physical device when creating a command pool"
        );

        let fns = device.fns();

        let flags = {
            let flag1 = if transient {
                ash::vk::CommandPoolCreateFlags::TRANSIENT
            } else {
                ash::vk::CommandPoolCreateFlags::empty()
            };
            let flag2 = if reset_cb {
                ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
            } else {
                ash::vk::CommandPoolCreateFlags::empty()
            };
            flag1 | flag2
        };

        let pool = unsafe {
            let infos = ash::vk::CommandPoolCreateInfo {
                flags: flags,
                queue_family_index: queue_family.id(),
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_command_pool(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
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
            ash::vk::CommandPoolResetFlags::RELEASE_RESOURCES
        } else {
            ash::vk::CommandPoolResetFlags::empty()
        };

        let fns = self.device.fns();
        check_errors(
            fns.v1_0
                .reset_command_pool(self.device.internal_object(), self.pool, flags),
        )?;
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
            if !(self.device.api_version() >= Version::V1_1
                || self.device.loaded_extensions().khr_maintenance1)
            {
                return Err(CommandPoolTrimError::Maintenance1ExtensionNotEnabled);
            }

            let fns = self.device.fns();

            if self.device.api_version() >= Version::V1_1 {
                fns.v1_1.trim_command_pool(
                    self.device.internal_object(),
                    self.pool,
                    ash::vk::CommandPoolTrimFlags::empty(),
                );
            } else {
                fns.khr_maintenance1.trim_command_pool_khr(
                    self.device.internal_object(),
                    self.pool,
                    ash::vk::CommandPoolTrimFlagsKHR::empty(),
                );
            }

            Ok(())
        }
    }

    /// Allocates `count` command buffers.
    ///
    /// If `secondary` is true, allocates secondary command buffers. Otherwise, allocates primary
    /// command buffers.
    pub fn alloc_command_buffers(
        &self,
        secondary: bool,
        count: usize,
    ) -> Result<UnsafeCommandPoolAllocIter, OomError> {
        if count == 0 {
            return Ok(UnsafeCommandPoolAllocIter {
                device: self.device.clone(),
                list: vec![].into_iter(),
            });
        }

        let infos = ash::vk::CommandBufferAllocateInfo {
            command_pool: self.pool,
            level: if secondary {
                ash::vk::CommandBufferLevel::SECONDARY
            } else {
                ash::vk::CommandBufferLevel::PRIMARY
            },
            command_buffer_count: count as u32,
            ..Default::default()
        };

        unsafe {
            let fns = self.device.fns();
            let mut out = Vec::with_capacity(count);
            check_errors(fns.v1_0.allocate_command_buffers(
                self.device.internal_object(),
                &infos,
                out.as_mut_ptr(),
            ))?;

            out.set_len(count);

            Ok(UnsafeCommandPoolAllocIter {
                device: self.device.clone(),
                list: out.into_iter(),
            })
        }
    }

    /// Frees individual command buffers.
    ///
    /// # Safety
    ///
    /// The command buffers must have been allocated from this pool. They must not be in use.
    ///
    pub unsafe fn free_command_buffers<I>(&self, command_buffers: I)
    where
        I: Iterator<Item = UnsafeCommandPoolAlloc>,
    {
        let command_buffers: SmallVec<[_; 4]> =
            command_buffers.map(|cb| cb.command_buffer).collect();
        let fns = self.device.fns();
        fns.v1_0.free_command_buffers(
            self.device.internal_object(),
            self.pool,
            command_buffers.len() as u32,
            command_buffers.as_ptr(),
        )
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
    type Object = ash::vk::CommandPool;

    #[inline]
    fn internal_object(&self) -> ash::vk::CommandPool {
        self.pool
    }
}

impl Drop for UnsafeCommandPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_command_pool(self.device.internal_object(), self.pool, ptr::null());
        }
    }
}

/// Opaque type that represents a command buffer allocated from a pool.
pub struct UnsafeCommandPoolAlloc {
    command_buffer: ash::vk::CommandBuffer,
    device: Arc<Device>,
}

unsafe impl DeviceOwned for UnsafeCommandPoolAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for UnsafeCommandPoolAlloc {
    type Object = ash::vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> ash::vk::CommandBuffer {
        self.command_buffer
    }
}

/// Iterator for newly-allocated command buffers.
#[derive(Debug)]
pub struct UnsafeCommandPoolAllocIter {
    device: Arc<Device>,
    list: VecIntoIter<ash::vk::CommandBuffer>,
}

impl Iterator for UnsafeCommandPoolAllocIter {
    type Item = UnsafeCommandPoolAlloc;

    #[inline]
    fn next(&mut self) -> Option<UnsafeCommandPoolAlloc> {
        self.list
            .next()
            .map(|command_buffer| UnsafeCommandPoolAlloc {
                command_buffer,
                device: self.device.clone(),
            })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.list.size_hint()
    }
}

impl ExactSizeIterator for UnsafeCommandPoolAllocIter {}

/// Error that can happen when trimming command pools.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CommandPoolTrimError {
    /// The `KHR_maintenance1` extension was not enabled.
    Maintenance1ExtensionNotEnabled,
}

impl error::Error for CommandPoolTrimError {}

impl fmt::Display for CommandPoolTrimError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CommandPoolTrimError::Maintenance1ExtensionNotEnabled => {
                    "the `KHR_maintenance1` extension was not enabled"
                }
            }
        )
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
    use crate::command_buffer::pool::CommandPoolTrimError;
    use crate::command_buffer::pool::UnsafeCommandPool;
    use crate::Version;

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

        assert_should_panic!(
            "Device doesn't match physical device when creating a command pool",
            {
                let _ = UnsafeCommandPool::new(device, queue.family(), false, false);
            }
        );
    }

    #[test]
    fn check_maintenance_when_trim() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = UnsafeCommandPool::new(device.clone(), queue.family(), false, false).unwrap();

        if device.api_version() >= Version::V1_1 {
            match pool.trim() {
                Err(CommandPoolTrimError::Maintenance1ExtensionNotEnabled) => panic!(),
                _ => (),
            }
        } else {
            match pool.trim() {
                Err(CommandPoolTrimError::Maintenance1ExtensionNotEnabled) => (),
                _ => panic!(),
            }
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
