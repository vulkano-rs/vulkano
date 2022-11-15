// Copyright (c) 2022 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! The standard command buffer and builder.
//!
//! Using `CommandBufferBuilder`, you must manually record synchronization commands to ensure
//! correct operation.

pub use self::builder::*;
use super::{
    allocator::{CommandBufferAlloc, StandardCommandBufferAlloc},
    CommandBufferExecError, CommandBufferInheritanceInfo, CommandBufferState, CommandBufferUsage,
};
use crate::{
    device::{Device, DeviceOwned},
    VulkanObject,
};
use parking_lot::Mutex;
use std::{
    any::Any,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

mod builder;

/// A command buffer that is finished recording, and can be submitted to a queue.
pub struct PrimaryCommandBuffer<A = StandardCommandBufferAlloc>
where
    A: CommandBufferAlloc,
{
    alloc: A,
    _usage: CommandBufferUsage,
    _resources: Vec<Box<dyn Any + Send + Sync>>,

    _state: Mutex<CommandBufferState>,
}

unsafe impl<A> VulkanObject for PrimaryCommandBuffer<A>
where
    A: CommandBufferAlloc,
{
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.alloc.inner().handle()
    }
}

unsafe impl<A> DeviceOwned for PrimaryCommandBuffer<A>
where
    A: CommandBufferAlloc,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.alloc.device()
    }
}

/// A command buffer that is finished recording, and can be executed within a primary command
/// buffer by calling [`execute_commands`].
///
/// [`execute_commands`]: CommandBufferBuilder::execute_commands
pub struct SecondaryCommandBuffer<A = StandardCommandBufferAlloc>
where
    A: CommandBufferAlloc,
{
    alloc: A,
    inheritance_info: CommandBufferInheritanceInfo,
    usage: CommandBufferUsage,
    _resources: Vec<Box<dyn Any + Send + Sync>>,

    submit_state: SubmitState,
}

unsafe impl<A> VulkanObject for SecondaryCommandBuffer<A>
where
    A: CommandBufferAlloc,
{
    type Handle = ash::vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.alloc.inner().handle()
    }
}

unsafe impl<A> DeviceOwned for SecondaryCommandBuffer<A>
where
    A: CommandBufferAlloc,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.alloc.device()
    }
}

impl<A> SecondaryCommandBuffer<A>
where
    A: CommandBufferAlloc,
{
    #[inline]
    pub fn inheritance_info(&self) -> &CommandBufferInheritanceInfo {
        &self.inheritance_info
    }

    pub fn lock_record(&self) -> Result<(), CommandBufferExecError> {
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                let was_already_submitted = already_submitted.swap(true, Ordering::Acquire);
                if was_already_submitted {
                    return Err(CommandBufferExecError::OneTimeSubmitAlreadySubmitted);
                }
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let already_in_use = in_use.swap(true, Ordering::Acquire);
                if already_in_use {
                    return Err(CommandBufferExecError::ExclusiveAlreadyInUse);
                }
            }
            SubmitState::Concurrent => (),
        };

        Ok(())
    }

    pub unsafe fn unlock(&self) {
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                debug_assert!(already_submitted.load(Ordering::Relaxed));
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let old_val = in_use.swap(false, Ordering::Release);
                debug_assert!(old_val);
            }
            SubmitState::Concurrent => (),
        };
    }
}

/// Whether the command buffer can be submitted.
#[derive(Debug)]
enum SubmitState {
    /// The command buffer was created with the "SimultaneousUse" flag. Can always be submitted at
    /// any time.
    Concurrent,

    /// The command buffer can only be submitted once simultaneously.
    ExclusiveUse {
        /// True if the command buffer is current in use by the GPU.
        in_use: AtomicBool,
    },

    /// The command buffer can only ever be submitted once.
    OneTime {
        /// True if the command buffer has already been submitted once and can be no longer be
        /// submitted.
        already_submitted: AtomicBool,
    },
}
