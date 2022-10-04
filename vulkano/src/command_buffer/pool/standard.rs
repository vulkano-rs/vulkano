use super::{
    sys::{CommandBufferAllocateInfo, UnsafeCommandPoolCreateInfo, UnsafeCommandPoolCreationError},
    CommandPool, CommandPoolAlloc, CommandPoolBuilderAlloc, UnsafeCommandPool,
    UnsafeCommandPoolAlloc,
};
use crate::{
    command_buffer::CommandBufferLevel,
    device::{Device, DeviceOwned},
    OomError,
};
use crossbeam_queue::SegQueue;
use std::{marker::PhantomData, mem::ManuallyDrop, ptr, sync::Arc, vec::IntoIter as VecIntoIter};

// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

/// Standard implementation of a command pool.
///
/// A thread can have as many `Arc<StandardCommandPool>`s as needed, but none of them can escape the
/// thread they were created on. This is done so that there are no locks involved when creating
/// command buffers. Command buffers can't be moved between threads during the building process, but
/// finished command buffers can. When a command buffer is dropped, it is returned back to the pool
/// for reuse.
#[derive(Debug)]
pub struct StandardCommandPool {
    // The Vulkan pool specific to a device's queue family.
    inner: UnsafeCommandPool,
    // List of existing primary command buffers that are available for reuse.
    available_primary_command_buffers: SegQueue<UnsafeCommandPoolAlloc>,
    // List of existing secondary command buffers that are available for reuse.
    available_secondary_command_buffers: SegQueue<UnsafeCommandPoolAlloc>,
}

impl StandardCommandPool {
    /// Builds a new pool.
    ///
    /// # Panics
    ///
    /// - Panics if the device and the queue family don't belong to the same physical device.
    pub fn new(
        device: Arc<Device>,
        queue_family_index: u32,
    ) -> Result<StandardCommandPool, OomError> {
        assert!(
            queue_family_index < device.physical_device().queue_family_properties().len() as u32
        );

        let inner = UnsafeCommandPool::new(
            device,
            UnsafeCommandPoolCreateInfo {
                queue_family_index,
                reset_command_buffer: true,
                ..Default::default()
            },
        )
        .map_err(|err| match err {
            UnsafeCommandPoolCreationError::OomError(err) => err,
            _ => panic!("Unexpected error: {}", err),
        })?;

        Ok(StandardCommandPool {
            inner,
            available_primary_command_buffers: Default::default(),
            available_secondary_command_buffers: Default::default(),
        })
    }
}

unsafe impl CommandPool for Arc<StandardCommandPool> {
    type Iter = VecIntoIter<StandardCommandPoolBuilder>;
    type Builder = StandardCommandPoolBuilder;
    type Alloc = StandardCommandPoolAlloc;

    #[inline]
    fn allocate(
        &self,
        level: CommandBufferLevel,
        mut command_buffer_count: u32,
    ) -> Result<Self::Iter, OomError> {
        // The final output.
        let mut output = Vec::with_capacity(command_buffer_count as usize);

        // First, pick from already-existing command buffers.
        {
            let existing = match level {
                CommandBufferLevel::Primary => &self.available_primary_command_buffers,
                CommandBufferLevel::Secondary => &self.available_secondary_command_buffers,
            };

            for _ in 0..command_buffer_count as usize {
                if let Some(cmd) = existing.pop() {
                    output.push(StandardCommandPoolBuilder {
                        inner: StandardCommandPoolAlloc {
                            cmd: ManuallyDrop::new(cmd),
                            pool: self.clone(),
                        },
                        dummy_avoid_send_sync: PhantomData,
                    });
                } else {
                    break;
                }
            }
        }

        // Then allocate the rest.
        if output.len() < command_buffer_count as usize {
            command_buffer_count -= output.len() as u32;

            for cmd in self
                .inner
                .allocate_command_buffers(CommandBufferAllocateInfo {
                    level,
                    command_buffer_count,
                    ..Default::default()
                })?
            {
                output.push(StandardCommandPoolBuilder {
                    inner: StandardCommandPoolAlloc {
                        cmd: ManuallyDrop::new(cmd),
                        pool: self.clone(),
                    },
                    dummy_avoid_send_sync: PhantomData,
                });
            }
        }

        // Final output.
        Ok(output.into_iter())
    }

    #[inline]
    fn queue_family_index(&self) -> u32 {
        self.inner.queue_family_index()
    }
}

unsafe impl DeviceOwned for StandardCommandPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

/// Command buffer allocated from a `StandardCommandPool` that is currently being built.
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
    fn queue_family_index(&self) -> u32 {
        self.inner.queue_family_index()
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
    pool: Arc<StandardCommandPool>,
}

unsafe impl Send for StandardCommandPoolAlloc {}
unsafe impl Sync for StandardCommandPoolAlloc {}

unsafe impl CommandPoolAlloc for StandardCommandPoolAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeCommandPoolAlloc {
        &*self.cmd
    }

    #[inline]
    fn queue_family_index(&self) -> u32 {
        self.pool.queue_family_index()
    }
}

unsafe impl DeviceOwned for StandardCommandPoolAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.pool.device()
    }
}

impl Drop for StandardCommandPoolAlloc {
    #[inline]
    fn drop(&mut self) {
        // Safe because `self.cmd` is wrapped in a `ManuallyDrop`.
        let cmd: UnsafeCommandPoolAlloc = unsafe { ptr::read(&*self.cmd) };

        match cmd.level() {
            CommandBufferLevel::Primary => self.pool.available_primary_command_buffers.push(cmd),
            CommandBufferLevel::Secondary => {
                self.pool.available_secondary_command_buffers.push(cmd)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        command_buffer::{
            pool::{CommandPool, CommandPoolBuilderAlloc},
            CommandBufferLevel,
        },
        VulkanObject,
    };
    use std::{sync::Arc, thread};

    #[test]
    fn reuse_command_buffers() {
        let (device, queue) = gfx_dev_and_queue!();

        device
            .with_standard_command_pool(queue.queue_family_index(), |pool| {
                let cb = pool
                    .allocate(CommandBufferLevel::Primary, 1)
                    .unwrap()
                    .next()
                    .unwrap();
                let raw = cb.inner().internal_object();
                drop(cb);

                let cb2 = pool
                    .allocate(CommandBufferLevel::Primary, 1)
                    .unwrap()
                    .next()
                    .unwrap();
                assert_eq!(raw, cb2.inner().internal_object());
            })
            .unwrap();
    }

    #[test]
    fn pool_kept_alive_by_thread() {
        let (device, queue) = gfx_dev_and_queue!();

        let thread = thread::spawn({
            let (device, queue) = (device, queue);
            move || {
                device
                    .with_standard_command_pool(queue.queue_family_index(), |pool| {
                        pool.allocate(CommandBufferLevel::Primary, 1)
                            .unwrap()
                            .next()
                            .unwrap()
                            .inner
                    })
                    .unwrap()
            }
        });

        // The thread-local storage should drop its reference to the pool here
        let cb = thread.join().unwrap();

        let pool_weak = Arc::downgrade(&cb.pool);
        drop(cb);
        assert!(pool_weak.upgrade().is_none());
    }
}
