// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    command_buffer::CommandBufferLevel,
    device::{Device, DeviceOwned},
    OomError, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    cell::Cell,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::MaybeUninit,
    ptr,
    sync::Arc,
};

/// Represents a Vulkan command pool.
///
/// A command pool is always tied to a specific queue family. Command buffers allocated from a pool
/// can only be executed on the corresponding queue family.
///
/// This struct doesn't implement the `Sync` trait because Vulkan command pools are not thread
/// safe. In other words, you can only use a pool from one thread at a time.
#[derive(Debug)]
pub struct CommandPool {
    handle: ash::vk::CommandPool,
    device: Arc<Device>,

    queue_family_index: u32,
    _transient: bool,
    _reset_command_buffer: bool,
    // Unimplement `Sync`, as Vulkan command pools are not thread-safe.
    _marker: PhantomData<Cell<ash::vk::CommandPool>>,
}

impl CommandPool {
    /// Creates a new `CommandPool`.
    pub fn new(
        device: Arc<Device>,
        mut create_info: CommandPoolCreateInfo,
    ) -> Result<CommandPool, CommandPoolCreationError> {
        Self::validate(&device, &mut create_info)?;
        let handle = unsafe { Self::create(&device, &create_info)? };

        let CommandPoolCreateInfo {
            queue_family_index,
            transient,
            reset_command_buffer,
            _ne: _,
        } = create_info;

        Ok(CommandPool {
            handle,
            device,
            queue_family_index,
            _transient: transient,
            _reset_command_buffer: reset_command_buffer,
            _marker: PhantomData,
        })
    }

    /// Creates a new `UnsafeCommandPool` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::CommandPool,
        create_info: CommandPoolCreateInfo,
    ) -> CommandPool {
        let CommandPoolCreateInfo {
            queue_family_index,
            transient,
            reset_command_buffer,
            _ne: _,
        } = create_info;

        CommandPool {
            handle,
            device,
            queue_family_index,
            _transient: transient,
            _reset_command_buffer: reset_command_buffer,
            _marker: PhantomData,
        }
    }

    fn validate(
        device: &Device,
        create_info: &mut CommandPoolCreateInfo,
    ) -> Result<(), CommandPoolCreationError> {
        let &mut CommandPoolCreateInfo {
            queue_family_index,
            transient: _,
            reset_command_buffer: _,
            _ne: _,
        } = create_info;

        // VUID-vkCreateCommandPool-queueFamilyIndex-01937
        if queue_family_index >= device.physical_device().queue_family_properties().len() as u32 {
            return Err(CommandPoolCreationError::QueueFamilyIndexOutOfRange {
                queue_family_index,
                queue_family_count: device.physical_device().queue_family_properties().len() as u32,
            });
        }

        Ok(())
    }

    unsafe fn create(
        device: &Device,
        create_info: &CommandPoolCreateInfo,
    ) -> Result<ash::vk::CommandPool, CommandPoolCreationError> {
        let &CommandPoolCreateInfo {
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
            (fns.v1_0.create_command_pool)(
                device.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
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
    #[inline]
    pub unsafe fn reset(&self, release_resources: bool) -> Result<(), OomError> {
        let flags = if release_resources {
            ash::vk::CommandPoolResetFlags::RELEASE_RESOURCES
        } else {
            ash::vk::CommandPoolResetFlags::empty()
        };

        let fns = self.device.fns();
        (fns.v1_0.reset_command_pool)(self.device.handle(), self.handle, flags)
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Allocates command buffers.
    #[inline]
    pub fn allocate_command_buffers(
        &self,
        allocate_info: CommandBufferAllocateInfo,
    ) -> Result<impl ExactSizeIterator<Item = CommandPoolAlloc>, OomError> {
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
                (fns.v1_0.allocate_command_buffers)(
                    self.device.handle(),
                    &allocate_info,
                    out.as_mut_ptr(),
                )
                .result()
                .map_err(VulkanError::from)?;
                out.set_len(command_buffer_count as usize);
                out
            }
        };

        let device = self.device.clone();

        Ok(out.into_iter().map(move |command_buffer| CommandPoolAlloc {
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
    pub unsafe fn free_command_buffers(
        &self,
        command_buffers: impl IntoIterator<Item = CommandPoolAlloc>,
    ) {
        let command_buffers: SmallVec<[_; 4]> =
            command_buffers.into_iter().map(|cb| cb.handle).collect();
        let fns = self.device.fns();
        (fns.v1_0.free_command_buffers)(
            self.device.handle(),
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
    #[inline]
    pub fn trim(&self) -> Result<(), CommandPoolTrimError> {
        if !(self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_maintenance1)
        {
            return Err(CommandPoolTrimError::RequirementNotMet {
                required_for: "`trim`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_1),
                    device_extensions: &["khr_maintenance1"],
                    ..Default::default()
                },
            });
        }

        unsafe {
            let fns = self.device.fns();

            if self.device.api_version() >= Version::V1_1 {
                (fns.v1_1.trim_command_pool)(
                    self.device.handle(),
                    self.handle,
                    ash::vk::CommandPoolTrimFlags::empty(),
                );
            } else {
                (fns.khr_maintenance1.trim_command_pool_khr)(
                    self.device.handle(),
                    self.handle,
                    ash::vk::CommandPoolTrimFlagsKHR::empty(),
                );
            }

            Ok(())
        }
    }

    /// Returns the queue family on which command buffers of this pool can be executed.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
}

impl Drop for CommandPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_command_pool)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for CommandPool {
    type Handle = ash::vk::CommandPool;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for CommandPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for CommandPool {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for CommandPool {}

impl Hash for CommandPool {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Error that can happen when creating a `CommandPool`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandPoolCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// The provided `queue_family_index` was not less than the number of queue families in the
    /// physical device.
    QueueFamilyIndexOutOfRange {
        queue_family_index: u32,
        queue_family_count: u32,
    },
}

impl Error for CommandPoolCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for CommandPoolCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory",),
            Self::QueueFamilyIndexOutOfRange {
                queue_family_index,
                queue_family_count,
            } => write!(
                f,
                "the provided `queue_family_index` ({}) was not less than the number of queue \
                families in the physical device ({})",
                queue_family_index, queue_family_count,
            ),
        }
    }
}

impl From<VulkanError> for CommandPoolCreationError {
    fn from(err: VulkanError) -> Self {
        match err {
            err @ VulkanError::OutOfHostMemory => Self::OomError(OomError::from(err)),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// Parameters to create an `CommandPool`.
#[derive(Clone, Debug)]
pub struct CommandPoolCreateInfo {
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

impl Default for CommandPoolCreateInfo {
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
pub struct CommandPoolAlloc {
    handle: ash::vk::CommandBuffer,
    device: Arc<Device>,
    level: CommandBufferLevel,
}

impl CommandPoolAlloc {
    /// Returns the level of the command buffer.
    #[inline]
    pub fn level(&self) -> CommandBufferLevel {
        self.level
    }
}

unsafe impl VulkanObject for CommandPoolAlloc {
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for CommandPoolAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for CommandPoolAlloc {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for CommandPoolAlloc {}

impl Hash for CommandPoolAlloc {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Error that can happen when trimming command pools.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CommandPoolTrimError {
    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },
}

impl Error for CommandPoolTrimError {}

impl Display for CommandPoolTrimError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
        }
    }
}

impl From<VulkanError> for CommandPoolTrimError {
    fn from(err: VulkanError) -> CommandPoolTrimError {
        panic!("unexpected error: {:?}", err)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CommandPool, CommandPoolCreateInfo, CommandPoolCreationError, CommandPoolTrimError,
    };
    use crate::{
        command_buffer::{pool::CommandBufferAllocateInfo, CommandBufferLevel},
        RequiresOneOf, Version,
    };

    #[test]
    fn basic_create() {
        let (device, queue) = gfx_dev_and_queue!();
        let _ = CommandPool::new(
            device,
            CommandPoolCreateInfo {
                queue_family_index: queue.queue_family_index(),
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn queue_family_getter() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = CommandPool::new(
            device,
            CommandPoolCreateInfo {
                queue_family_index: queue.queue_family_index(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(pool.queue_family_index(), queue.queue_family_index());
    }

    #[test]
    fn check_queue_family_too_high() {
        let (device, _) = gfx_dev_and_queue!();

        match CommandPool::new(
            device,
            CommandPoolCreateInfo {
                ..Default::default()
            },
        ) {
            Err(CommandPoolCreationError::QueueFamilyIndexOutOfRange { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn check_maintenance_when_trim() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = CommandPool::new(
            device.clone(),
            CommandPoolCreateInfo {
                queue_family_index: queue.queue_family_index(),
                ..Default::default()
            },
        )
        .unwrap();

        if device.api_version() >= Version::V1_1 {
            if matches!(
                pool.trim(),
                Err(CommandPoolTrimError::RequirementNotMet {
                    requires_one_of: RequiresOneOf {
                        device_extensions,
                        ..
                    }, ..
                }) if device_extensions.contains(&"khr_maintenance1")
            ) {
                panic!()
            }
        } else {
            if !matches!(
                pool.trim(),
                Err(CommandPoolTrimError::RequirementNotMet {
                    requires_one_of: RequiresOneOf {
                        device_extensions,
                        ..
                    }, ..
                }) if device_extensions.contains(&"khr_maintenance1")
            ) {
                panic!()
            }
        }
    }

    // TODO: test that trim works if VK_KHR_maintenance1 if enabled ; the test macro doesn't
    //       support enabling extensions yet

    #[test]
    fn basic_alloc() {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = CommandPool::new(
            device,
            CommandPoolCreateInfo {
                queue_family_index: queue.queue_family_index(),
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
