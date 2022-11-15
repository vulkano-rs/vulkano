// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    CommandBufferInheritanceInfo, CommandBufferResourcesUsage, CommandBufferState,
    CommandBufferUsage, SemaphoreSubmitInfo, SubmitInfo,
};
use crate::{
    buffer::{sys::Buffer, BufferAccess},
    device::{Device, DeviceOwned, Queue},
    image::{sys::Image, ImageAccess, ImageLayout, ImageSubresourceRange},
    swapchain::Swapchain,
    sync::{
        future::{
            now, AccessCheckError, AccessError, FlushError, GpuFuture, NowFuture, SubmitAnyBuilder,
        },
        AccessFlags, PipelineMemoryAccess, PipelineStages,
    },
    DeviceSize, SafeDeref, VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use std::{
    borrow::Cow,
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

pub unsafe trait PrimaryCommandBufferAbstract:
    VulkanObject<Handle = ash::vk::CommandBuffer> + DeviceOwned + Send + Sync
{
    /// Returns the usage of this command buffer.
    fn usage(&self) -> CommandBufferUsage;

    /// Executes this command buffer on a queue.
    ///
    /// This function returns an object that implements the `GpuFuture` trait. See the
    /// documentation of the `sync` module for more information.
    ///
    /// The command buffer is not actually executed until you call `flush()` on the object.
    /// You are encouraged to chain together as many futures as possible before calling `flush()`,
    /// and call `.then_signal_future()` before doing so. Note however that once you called
    /// `execute()` there is no way to cancel the execution, even if you didn't flush yet.
    ///
    /// > **Note**: In the future this function may return `-> impl GpuFuture` instead of a
    /// > concrete type.
    ///
    /// > **Note**: This is just a shortcut for `execute_after(vulkano::sync::now(), queue)`.
    ///
    /// # Panics
    ///
    /// - Panics if the device of the command buffer is not the same as the device of the future.
    #[inline]
    fn execute(
        self,
        queue: Arc<Queue>,
    ) -> Result<CommandBufferExecFuture<NowFuture>, CommandBufferExecError>
    where
        Self: Sized + 'static,
    {
        let device = queue.device().clone();
        self.execute_after(now(device), queue)
    }

    /// Executes the command buffer after an existing future.
    ///
    /// This function returns an object that implements the `GpuFuture` trait. See the
    /// documentation of the `sync` module for more information.
    ///
    /// The command buffer is not actually executed until you call `flush()` on the object.
    /// You are encouraged to chain together as many futures as possible before calling `flush()`,
    /// and call `.then_signal_future()` before doing so. Note however that once you called
    /// `execute()` there is no way to cancel the execution, even if you didn't flush yet.
    ///
    /// > **Note**: In the future this function may return `-> impl GpuFuture` instead of a
    /// > concrete type.
    ///
    /// This function requires the `'static` lifetime to be on the command buffer. This is because
    /// this function returns a `CommandBufferExecFuture` whose job is to lock resources and keep
    /// them alive while they are in use by the GPU. If `'static` wasn't required, you could call
    /// `std::mem::forget` on that object and "unlock" these resources. For more information about
    /// this problem, search the web for "rust thread scoped leakpocalypse".
    ///
    /// # Panics
    ///
    /// - Panics if the device of the command buffer is not the same as the device of the future.
    fn execute_after<F>(
        self,
        future: F,
        queue: Arc<Queue>,
    ) -> Result<CommandBufferExecFuture<F>, CommandBufferExecError>
    where
        Self: Sized + 'static,
        F: GpuFuture,
    {
        assert_eq!(self.device().handle(), future.device().handle());

        if !future.queue_change_allowed() {
            assert!(future.queue().unwrap() == queue);
        }

        Ok(CommandBufferExecFuture {
            previous: future,
            command_buffer: Arc::new(self),
            queue,
            submitted: Mutex::new(false),
            finished: AtomicBool::new(false),
        })
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError>;

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError>;

    #[doc(hidden)]
    fn state(&self) -> MutexGuard<'_, CommandBufferState>;

    #[doc(hidden)]
    fn resources_usage(&self) -> &CommandBufferResourcesUsage;
}

impl Debug for dyn PrimaryCommandBufferAbstract {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        Debug::fmt(&self.handle(), f)
    }
}

unsafe impl<T> PrimaryCommandBufferAbstract for T
where
    T: VulkanObject<Handle = ash::vk::CommandBuffer> + SafeDeref + Send + Sync,
    T::Target: PrimaryCommandBufferAbstract,
{
    fn usage(&self) -> CommandBufferUsage {
        (**self).usage()
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        (**self).check_buffer_access(buffer, range, exclusive, queue)
    }

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        (**self).check_image_access(image, range, exclusive, expected_layout, queue)
    }

    fn state(&self) -> MutexGuard<'_, CommandBufferState> {
        (**self).state()
    }

    fn resources_usage(&self) -> &CommandBufferResourcesUsage {
        (**self).resources_usage()
    }
}

pub unsafe trait SecondaryCommandBufferAbstract:
    VulkanObject<Handle = ash::vk::CommandBuffer> + DeviceOwned + Send + Sync
{
    /// Returns the usage of this command buffer.
    fn usage(&self) -> CommandBufferUsage;

    /// Returns a `CommandBufferInheritance` value describing the properties that the command
    /// buffer inherits from its parent primary command buffer.
    fn inheritance_info(&self) -> &CommandBufferInheritanceInfo;

    /// Checks whether this command buffer is allowed to be recorded to a command buffer,
    /// and if so locks it.
    ///
    /// If you call this function, then you should call `unlock` afterwards.
    fn lock_record(&self) -> Result<(), CommandBufferExecError>;

    /// Unlocks the command buffer. Should be called once for each call to `lock_record`.
    ///
    /// # Safety
    ///
    /// Must not be called if you haven't called `lock_record` before.
    unsafe fn unlock(&self);

    /// Returns the number of buffers accessed by this command buffer.
    fn num_buffers(&self) -> usize;

    /// Returns the `index`th buffer of this command buffer, or `None` if out of range.
    ///
    /// The valid range is between 0 and `num_buffers()`.
    fn buffer(
        &self,
        index: usize,
    ) -> Option<(
        &Arc<dyn BufferAccess>,
        Range<DeviceSize>,
        PipelineMemoryAccess,
    )>;

    /// Returns the number of images accessed by this command buffer.
    fn num_images(&self) -> usize;

    /// Returns the `index`th image of this command buffer, or `None` if out of range.
    ///
    /// The valid range is between 0 and `num_images()`.
    fn image(
        &self,
        index: usize,
    ) -> Option<(
        &Arc<dyn ImageAccess>,
        &ImageSubresourceRange,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
    )>;
}

unsafe impl<T> SecondaryCommandBufferAbstract for T
where
    T: VulkanObject<Handle = ash::vk::CommandBuffer> + SafeDeref + Send + Sync,
    T::Target: SecondaryCommandBufferAbstract,
{
    fn usage(&self) -> CommandBufferUsage {
        (**self).usage()
    }

    fn inheritance_info(&self) -> &CommandBufferInheritanceInfo {
        (**self).inheritance_info()
    }

    fn lock_record(&self) -> Result<(), CommandBufferExecError> {
        (**self).lock_record()
    }

    unsafe fn unlock(&self) {
        (**self).unlock();
    }

    fn num_buffers(&self) -> usize {
        (**self).num_buffers()
    }

    fn buffer(
        &self,
        index: usize,
    ) -> Option<(
        &Arc<dyn BufferAccess>,
        Range<DeviceSize>,
        PipelineMemoryAccess,
    )> {
        (**self).buffer(index)
    }

    fn num_images(&self) -> usize {
        (**self).num_images()
    }

    fn image(
        &self,
        index: usize,
    ) -> Option<(
        &Arc<dyn ImageAccess>,
        &ImageSubresourceRange,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
    )> {
        (**self).image(index)
    }
}

/// Represents a command buffer being executed by the GPU and the moment when the execution
/// finishes.
#[derive(Debug)]
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct CommandBufferExecFuture<F>
where
    F: GpuFuture,
{
    previous: F,
    command_buffer: Arc<dyn PrimaryCommandBufferAbstract>,
    queue: Arc<Queue>,
    // True if the command buffer has already been submitted.
    // If flush is called multiple times, we want to block so that only one flushing is executed.
    // Therefore we use a `Mutex<bool>` and not an `AtomicBool`.
    submitted: Mutex<bool>,
    finished: AtomicBool,
}

impl<F> CommandBufferExecFuture<F>
where
    F: GpuFuture,
{
    // Implementation of `build_submission`. Doesn't check whenever the future was already flushed.
    // You must make sure to not submit same command buffer multiple times.
    unsafe fn build_submission_impl(&self) -> Result<SubmitAnyBuilder, FlushError> {
        Ok(match self.previous.build_submission()? {
            SubmitAnyBuilder::Empty => SubmitAnyBuilder::CommandBuffer(
                SubmitInfo {
                    command_buffers: vec![self.command_buffer.clone()],
                    ..Default::default()
                },
                None,
            ),
            SubmitAnyBuilder::SemaphoresWait(semaphores) => {
                SubmitAnyBuilder::CommandBuffer(
                    SubmitInfo {
                        wait_semaphores: semaphores
                            .into_iter()
                            .map(|semaphore| {
                                SemaphoreSubmitInfo {
                                    // TODO: correct stages ; hard
                                    stages: PipelineStages::ALL_COMMANDS,
                                    ..SemaphoreSubmitInfo::semaphore(semaphore)
                                }
                            })
                            .collect(),
                        command_buffers: vec![self.command_buffer.clone()],
                        ..Default::default()
                    },
                    None,
                )
            }
            SubmitAnyBuilder::CommandBuffer(mut submit_info, fence) => {
                // FIXME: add pipeline barrier
                submit_info
                    .command_buffers
                    .push(self.command_buffer.clone());
                SubmitAnyBuilder::CommandBuffer(submit_info, fence)
            }
            SubmitAnyBuilder::QueuePresent(_) | SubmitAnyBuilder::BindSparse(_, _) => {
                unimplemented!() // TODO:
                                 /*present.submit();     // TODO: wrong
                                 let mut builder = SubmitCommandBufferBuilder::new();
                                 builder.add_command_buffer(self.command_buffer.inner());
                                 SubmitAnyBuilder::CommandBuffer(builder)*/
            }
        })
    }
}

unsafe impl<F> GpuFuture for CommandBufferExecFuture<F>
where
    F: GpuFuture,
{
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        if *self.submitted.lock() {
            return Ok(SubmitAnyBuilder::Empty);
        }

        self.build_submission_impl()
    }

    fn flush(&self) -> Result<(), FlushError> {
        unsafe {
            let mut submitted = self.submitted.lock();
            if *submitted {
                return Ok(());
            }

            match self.build_submission_impl()? {
                SubmitAnyBuilder::Empty => {}
                SubmitAnyBuilder::CommandBuffer(submit_info, fence) => {
                    self.queue.with(|mut q| {
                        q.submit_with_future(submit_info, fence, &self.previous, &self.queue)
                    })?;
                }
                _ => unreachable!(),
            };

            // Only write `true` here in order to try again next time if we failed to submit.
            *submitted = true;
            Ok(())
        }
    }

    unsafe fn signal_finished(&self) {
        self.finished.store(true, Ordering::SeqCst);
        self.previous.signal_finished();
    }

    fn queue_change_allowed(&self) -> bool {
        false
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        Some(self.queue.clone())
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        match self
            .command_buffer
            .check_buffer_access(buffer, range.clone(), exclusive, queue)
        {
            Ok(v) => Ok(v),
            Err(AccessCheckError::Denied(err)) => Err(AccessCheckError::Denied(err)),
            Err(AccessCheckError::Unknown) => self
                .previous
                .check_buffer_access(buffer, range, exclusive, queue),
        }
    }

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        match self.command_buffer.check_image_access(
            image,
            range.clone(),
            exclusive,
            expected_layout,
            queue,
        ) {
            Ok(v) => Ok(v),
            Err(AccessCheckError::Denied(err)) => Err(AccessCheckError::Denied(err)),
            Err(AccessCheckError::Unknown) => {
                self.previous
                    .check_image_access(image, range, exclusive, expected_layout, queue)
            }
        }
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        _before: bool,
    ) -> Result<(), AccessCheckError> {
        self.previous
            .check_swapchain_image_acquired(swapchain, image_index, false)
    }
}

unsafe impl<F> DeviceOwned for CommandBufferExecFuture<F>
where
    F: GpuFuture,
{
    fn device(&self) -> &Arc<Device> {
        self.command_buffer.device()
    }
}

impl<F> Drop for CommandBufferExecFuture<F>
where
    F: GpuFuture,
{
    fn drop(&mut self) {
        unsafe {
            if !*self.finished.get_mut() {
                // TODO: handle errors?
                self.flush().unwrap();
                // Block until the queue finished.
                self.queue.with(|mut q| q.wait_idle()).unwrap();
                self.previous.signal_finished();
            }
        }
    }
}

/// Error that can happen when attempting to execute a command buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandBufferExecError {
    /// Access to a resource has been denied.
    AccessError {
        error: AccessError,
        command_name: Cow<'static, str>,
        command_param: Cow<'static, str>,
        command_offset: usize,
    },

    /// The command buffer or one of the secondary command buffers it executes was created with the
    /// "one time submit" flag, but has already been submitted it the past.
    OneTimeSubmitAlreadySubmitted,

    /// The command buffer or one of the secondary command buffers it executes is already in use by
    /// the GPU and was not created with the "concurrent" flag.
    ExclusiveAlreadyInUse,
    // TODO: missing entries (eg. wrong queue family, secondary command buffer)
}

impl Error for CommandBufferExecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CommandBufferExecError::AccessError { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl Display for CommandBufferExecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
                CommandBufferExecError::AccessError { .. } =>
                    "access to a resource has been denied",
                CommandBufferExecError::OneTimeSubmitAlreadySubmitted => {
                    "the command buffer or one of the secondary command buffers it executes was \
                    created with the \"one time submit\" flag, but has already been submitted in \
                    the past"
                }
                CommandBufferExecError::ExclusiveAlreadyInUse => {
                    "the command buffer or one of the secondary command buffers it executes is \
                    already in use was not created with the \"concurrent\" flag"
                }
            }
        )
    }
}
