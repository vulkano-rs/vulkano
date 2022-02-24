// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    check_errors,
    command_buffer::CommandBufferLevel,
    device::{physical::QueueFamily, Device, DeviceOwned},
    Error, OomError, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    error, fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::MaybeUninit,
    ptr,
    sync::Arc,
};

/// Low-level implementation of a command pool.
///
/// A command pool is always tied to a specific queue family. Command buffers allocated from a pool
/// can only be executed on the corresponding queue family.
///
/// This struct doesn't implement the `Sync` trait because Vulkan command pools are not thread
/// safe. In other words, you can only use a pool from one thread at a time.
#[derive(Debug)]
pub struct UnsafeCommandPool {
    handle: ash::vk::CommandPool,
    device: Arc<Device>,
    // We don't want `UnsafeCommandPool` to implement Sync.
    // This marker unimplements both Send and Sync, but we reimplement Send manually right under.
    dummy_avoid_sync: PhantomData<*const u8>,

    queue_family_index: u32,
    transient: bool,
    reset_command_buffer: bool,
}

unsafe impl Send for UnsafeCommandPool {}

impl UnsafeCommandPool {
    /// Creates a new `UnsafeCommandPool`.
    pub fn new(
        device: Arc<Device>,
        mut create_info: UnsafeCommandPoolCreateInfo,
    ) -> Result<UnsafeCommandPool, UnsafeCommandPoolCreationError> {
        Self::validate(&device, &mut create_info)?;
        let handle = unsafe { Self::create(&device, &create_info)? };

        let UnsafeCommandPoolCreateInfo {
            queue_family_index,
            transient,
            reset_command_buffer,
            _ne: _,
        } = create_info;

        Ok(UnsafeCommandPool {
            handle,
            device,
            dummy_avoid_sync: PhantomData,

            queue_family_index,
            transient,
            reset_command_buffer,
        })
    }

    fn validate(
        device: &Device,
        create_info: &mut UnsafeCommandPoolCreateInfo,
    ) -> Result<(), UnsafeCommandPoolCreationError> {
        let &mut UnsafeCommandPoolCreateInfo {
            queue_family_index,
            transient,
            reset_command_buffer,
            _ne: _,
        } = create_info;

        // VUID-vkCreateCommandPool-queueFamilyIndex-01937
        if device
            .physical_device()
            .queue_family_by_id(queue_family_index)
            .is_none()
        {
            return Err(UnsafeCommandPoolCreationError::QueueFamilyIndexOutOfRange {
                queue_family_index,
                queue_family_count: device.physical_device().queue_families().len() as u32,
            });
        }

        Ok(())
    }

    unsafe fn create(
        device: &Device,
        create_info: &UnsafeCommandPoolCreateInfo,
    ) -> Result<ash::vk::CommandPool, UnsafeCommandPoolCreationError> {
        let &UnsafeCommandPoolCreateInfo {
            queue_family_index,
            transient,
            reset_command_buffer,
            _ne: _,
        } = create_info;

        let mut flags = ash::vk::CommandPoolCreateFlags::empty();

        if transient {
            flags |= ash::vk::CommandPoolCreateFlags::TRANSIENT;
        }

        if reset_command_buffer {
            flags |= ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;
        }

        let create_info = ash::vk::CommandPoolCreateInfo {
            flags,
            queue_family_index,
            ..Default::default()
        };

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_command_pool(
                device.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(handle)
    }

    /// Resets the pool, which resets all the command buffers that were allocated from it.
    ///
    /// If `release_resources` is true, it is a hint to the implementation that it should free all
    /// the memory internally allocated for this pool.
    ///
    /// # Safety
    ///
    /// - The command buffers allocated from this pool jump to the initial state.
    pub unsafe fn reset(&self, release_resources: bool) -> Result<(), OomError> {
        let flags = if release_resources {
            ash::vk::CommandPoolResetFlags::RELEASE_RESOURCES
        } else {
            ash::vk::CommandPoolResetFlags::empty()
        };

        let fns = self.device.fns();
        check_errors(fns.v1_0.reset_command_pool(
            self.device.internal_object(),
            self.handle,
            flags,
        ))?;
        Ok(())
    }

    /// Allocates command buffers.
    pub fn allocate_command_buffers(
        &self,
        allocate_info: CommandBufferAllocateInfo,
    ) -> Result<impl ExactSizeIterator<Item = UnsafeCommandPoolAlloc>, OomError> {
        let CommandBufferAllocateInfo {
            level,
            command_buffer_count,
            _ne: _,
        } = allocate_info;

        // VUID-vkAllocateCommandBuffers-pAllocateInfo::commandBufferCount-arraylength
        let out = if command_buffer_count == 0 {
            vec![]
        } else {
            let allocate_info = ash::vk::CommandBufferAllocateInfo {
                command_pool: self.handle,
                level: level.into(),
                command_buffer_count,
                ..Default::default()
            };

            unsafe {
                let fns = self.device.fns();
                let mut out = Vec::with_capacity(command_buffer_count as usize);
                check_errors(fns.v1_0.allocate_command_buffers(
                    self.device.internal_object(),
                    &allocate_info,
                    out.as_mut_ptr(),
                ))?;
                out.set_len(command_buffer_count as usize);
                out
            }
        };

        let device = self.device.clone();

        Ok(out
            .into_iter()
            .map(move |command_buffer| UnsafeCommandPoolAlloc {
                handle: command_buffer,
                device: device.clone(),

                level,
            }))
    }

    /// Frees individual command buffers.
    ///
    /// # Safety
    ///
    /// - The `command_buffers` must have been allocated from this pool.
    /// - The `command_buffers` must not be in the pending state.
    pub unsafe fn free_command_buffers<I>(&self, command_buffers: I)
    where
        I: IntoIterator<Item = UnsafeCommandPoolAlloc>,
    {
        let command_buffers: SmallVec<[_; 4]> =
            command_buffers.into_iter().map(|cb| cb.handle).collect();
        let fns = self.device.fns();
        fns.v1_0.free_command_buffers(
            self.device.internal_object(),
            self.handle,
            command_buffers.len() as u32,
            command_buffers.as_ptr(),
        )
    }

    /// Trims a command pool, which recycles unused internal memory from the command pool back to
    /// the system.
    ///
    /// Command buffers allocated from the pool are not affected by trimming.
    ///
    /// This function is supported only if the
    /// [`khr_maintenance1`](crate::device::DeviceExtensions::khr_maintenance1) extension is
    /// enabled on the device. Otherwise an error is returned.
    /// Since this operation is purely an optimization it is legitimate to call this function and
    /// simply ignore any possible error.
    pub fn trim(&self) -> Result<(), CommandPoolTrimError> {
        if !(self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_maintenance1)
        {
            return Err(CommandPoolTrimError::Maintenance1ExtensionNotEnabled);
        }

        unsafe {
            let fns = self.device.fns();

            if self.device.api_version() >= Version::V1_1 {
                fns.v1_1.trim_command_pool(
                    self.device.internal_object(),
                    self.handle,
                    ash::vk::CommandPoolTrimFlags::empty(),
                );
            } else {
                fns.khr_maintenance1.trim_command_pool_khr(
                    self.device.internal_object(),
                    self.handle,
                    ash::vk::CommandPoolTrimFlagsKHR::empty(),
                );
            }

            Ok(())
        }
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

impl Drop for UnsafeCommandPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_command_pool(self.device.internal_object(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for UnsafeCommandPool {
    type Object = ash::vk::CommandPool;

    #[inline]
    fn internal_object(&self) -> ash::vk::CommandPool {
        self.handle
    }
}

unsafe impl DeviceOwned for UnsafeCommandPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for UnsafeCommandPool {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for UnsafeCommandPool {}

impl Hash for UnsafeCommandPool {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Error that can happen when creating an `UnsafeCommandPool`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsafeCommandPoolCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// The provided `queue_family_index` was not less than the number of queue families in the
    /// physical device.
    QueueFamilyIndexOutOfRange {
        queue_family_index: u32,
        queue_family_count: u32,
    },
}

impl error::Error for UnsafeCommandPoolCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for UnsafeCommandPoolCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::OomError(_) => write!(fmt, "not enough memory",),
            Self::QueueFamilyIndexOutOfRange {
                queue_family_index,
                queue_family_count,
            } => write!(
                fmt,
                "the provided `queue_family_index` ({}) was not less than the number of queue families in the physical device ({})",
                queue_family_index, queue_family_count,
            ),
        }
    }
}

impl From<Error> for UnsafeCommandPoolCreationError {
    #[inline]
    fn from(err: Error) -> Self {
        match err {
            err @ Error::OutOfHostMemory => Self::OomError(OomError::from(err)),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// Parameters to create an `UnsafeCommandPool`.
#[derive(Clone, Debug)]
pub struct UnsafeCommandPoolCreateInfo {
    /// The index of the queue family that this pool is created for. All command buffers allocated
    /// from this pool must be submitted on a queue belonging to that family.
    ///
    /// The default value is `u32::MAX`, which must be overridden.
    pub queue_family_index: u32,

    /// A hint to the implementation that the command buffers allocated from this pool will be
    /// short-lived.
    ///
    /// The default value is `false`.
    pub transient: bool,

    /// Whether the command buffers allocated from this pool can be reset individually.
    ///
    /// The default value is `false`.
    pub reset_command_buffer: bool,

    pub _ne: crate::NonExhaustive,
}

impl Default for UnsafeCommandPoolCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            queue_family_index: u32::MAX,
            transient: false,
            reset_command_buffer: false,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to allocate an `UnsafeCommandPoolAlloc`.
#[derive(Clone, Debug)]
pub struct CommandBufferAllocateInfo {
    /// The level of command buffer to allocate.
    ///
    /// The default value is [`CommandBufferLevel::Primary`].
    pub level: CommandBufferLevel,

    /// The number of command buffers to allocate.
    ///
    /// The default value is `1`.
    pub command_buffer_count: u32,

    pub _ne: crate::NonExhaustive,
}

impl Default for CommandBufferAllocateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            level: CommandBufferLevel::Primary,
            command_buffer_count: 1,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Opaque type that represents a command buffer allocated from a pool.
#[derive(Debug)]
pub struct UnsafeCommandPoolAlloc {
    handle: ash::vk::CommandBuffer,
    device: Arc<Device>,
    level: CommandBufferLevel,
}

impl UnsafeCommandPoolAlloc {
    /// Returns the level of the command buffer.
    #[inline]
    pub fn level(&self) -> CommandBufferLevel {
        self.level
    }
}

unsafe impl VulkanObject for UnsafeCommandPoolAlloc {
    type Object = ash::vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> ash::vk::CommandBuffer {
        self.handle
    }
}

unsafe impl DeviceOwned for UnsafeCommandPoolAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for UnsafeCommandPoolAlloc {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for UnsafeCommandPoolAlloc {}

impl Hash for UnsafeCommandPoolAlloc {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

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
    use super::{
        CommandPoolTrimError, UnsafeCommandPool, UnsafeCommandPoolCreateInfo,
        UnsafeCommandPoolCreationError,
    };
    use crate::{
        command_buffer::{pool::sys::CommandBufferAllocateInfo, CommandBufferLevel},
        Version,
    };

    #[test]
    fn basic_create() {
        let (device, queue) = gfx_dev_and_queue!();
        let _ = UnsafeCommandPool::new(
            device,
            UnsafeCommandPoolCreateInfo {
                queue_family_index: queue.family().id(),
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn queue_family_getter() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = UnsafeCommandPool::new(
            device,
            UnsafeCommandPoolCreateInfo {
                queue_family_index: queue.family().id(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(pool.queue_family().id(), queue.family().id());
    }

    #[test]
    fn check_queue_family_too_high() {
        let (device, _) = gfx_dev_and_queue!();

        match UnsafeCommandPool::new(
            device,
            UnsafeCommandPoolCreateInfo {
                ..Default::default()
            },
        ) {
            Err(UnsafeCommandPoolCreationError::QueueFamilyIndexOutOfRange { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn check_maintenance_when_trim() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = UnsafeCommandPool::new(
            device.clone(),
            UnsafeCommandPoolCreateInfo {
                queue_family_index: queue.family().id(),
                ..Default::default()
            },
        )
        .unwrap();

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
        let pool = UnsafeCommandPool::new(
            device,
            UnsafeCommandPoolCreateInfo {
                queue_family_index: queue.family().id(),
                ..Default::default()
            },
        )
        .unwrap();
        let iter = pool
            .allocate_command_buffers(CommandBufferAllocateInfo {
                level: CommandBufferLevel::Primary,
                command_buffer_count: 12,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(iter.count(), 12);
    }
}
