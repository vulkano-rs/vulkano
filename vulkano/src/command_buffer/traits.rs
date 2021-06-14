// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::command_buffer::submit::SubmitAnyBuilder;
use crate::command_buffer::submit::SubmitCommandBufferBuilder;
use crate::command_buffer::sys::UnsafeCommandBuffer;
use crate::command_buffer::CommandBufferInheritance;
use crate::command_buffer::ImageUninitializedSafe;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::device::Queue;
use crate::image::ImageAccess;
use crate::image::ImageLayout;
use crate::render_pass::FramebufferAbstract;
use crate::sync::now;
use crate::sync::AccessCheckError;
use crate::sync::AccessError;
use crate::sync::AccessFlags;
use crate::sync::FlushError;
use crate::sync::GpuFuture;
use crate::sync::NowFuture;
use crate::sync::PipelineMemoryAccess;
use crate::sync::PipelineStages;
use crate::SafeDeref;
use crate::VulkanObject;
use std::borrow::Cow;
use std::error;
use std::fmt;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;

pub unsafe trait PrimaryCommandBuffer: DeviceOwned {
    /// Returns the underlying `UnsafeCommandBuffer` of this command buffer.
    fn inner(&self) -> &UnsafeCommandBuffer;

    /// Checks whether this command buffer is allowed to be submitted after the `future` and on
    /// the given queue, and if so locks it.
    ///
    /// If you call this function, then you should call `unlock` afterwards.
    fn lock_submit(
        &self,
        future: &dyn GpuFuture,
        queue: &Queue,
    ) -> Result<(), CommandBufferExecError>;

    /// Unlocks the command buffer. Should be called once for each call to `lock_submit`.
    ///
    /// # Safety
    ///
    /// Must not be called if you haven't called `lock_submit` before.
    unsafe fn unlock(&self);

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
    /// # Panic
    ///
    /// Panics if the device of the command buffer is not the same as the device of the future.
    #[inline]
    fn execute(
        self,
        queue: Arc<Queue>,
    ) -> Result<CommandBufferExecFuture<NowFuture, Self>, CommandBufferExecError>
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
    /// # Panic
    ///
    /// Panics if the device of the command buffer is not the same as the device of the future.
    #[inline]
    fn execute_after<F>(
        self,
        future: F,
        queue: Arc<Queue>,
    ) -> Result<CommandBufferExecFuture<F, Self>, CommandBufferExecError>
    where
        Self: Sized + 'static,
        F: GpuFuture,
    {
        assert_eq!(
            self.device().internal_object(),
            future.device().internal_object()
        );

        if !future.queue_change_allowed() {
            assert!(future.queue().unwrap().is_same(&queue));
        }

        self.lock_submit(&future, &queue)?;

        Ok(CommandBufferExecFuture {
            previous: future,
            command_buffer: self,
            queue,
            submitted: Mutex::new(false),
            finished: AtomicBool::new(false),
        })
    }

    fn check_buffer_access(
        &self,
        buffer: &dyn BufferAccess,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError>;

    fn check_image_access(
        &self,
        image: &dyn ImageAccess,
        layout: ImageLayout,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError>;
}

unsafe impl<T> PrimaryCommandBuffer for T
where
    T: SafeDeref,
    T::Target: PrimaryCommandBuffer,
{
    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer {
        (**self).inner()
    }

    #[inline]
    fn lock_submit(
        &self,
        future: &dyn GpuFuture,
        queue: &Queue,
    ) -> Result<(), CommandBufferExecError> {
        (**self).lock_submit(future, queue)
    }

    #[inline]
    unsafe fn unlock(&self) {
        (**self).unlock();
    }

    #[inline]
    fn check_buffer_access(
        &self,
        buffer: &dyn BufferAccess,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        (**self).check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(
        &self,
        image: &dyn ImageAccess,
        layout: ImageLayout,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        (**self).check_image_access(image, layout, exclusive, queue)
    }
}

pub unsafe trait SecondaryCommandBuffer: DeviceOwned {
    /// Returns the underlying `UnsafeCommandBuffer` of this command buffer.
    fn inner(&self) -> &UnsafeCommandBuffer;

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

    /// Returns a `CommandBufferInheritance` value describing the properties that the command
    /// buffer inherits from its parent primary command buffer.
    fn inheritance(&self) -> CommandBufferInheritance<&dyn FramebufferAbstract>;

    /// Returns the number of buffers accessed by this command buffer.
    fn num_buffers(&self) -> usize;

    /// Returns the `index`th buffer of this command buffer, or `None` if out of range.
    ///
    /// The valid range is between 0 and `num_buffers()`.
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, PipelineMemoryAccess)>;

    /// Returns the number of images accessed by this command buffer.
    fn num_images(&self) -> usize;

    /// Returns the `index`th image of this command buffer, or `None` if out of range.
    ///
    /// The valid range is between 0 and `num_images()`.
    fn image(
        &self,
        index: usize,
    ) -> Option<(
        &dyn ImageAccess,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
        ImageUninitializedSafe,
    )>;
}

unsafe impl<T> SecondaryCommandBuffer for T
where
    T: SafeDeref,
    T::Target: SecondaryCommandBuffer,
{
    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer {
        (**self).inner()
    }

    #[inline]
    fn lock_record(&self) -> Result<(), CommandBufferExecError> {
        (**self).lock_record()
    }

    #[inline]
    unsafe fn unlock(&self) {
        (**self).unlock();
    }

    #[inline]
    fn inheritance(&self) -> CommandBufferInheritance<&dyn FramebufferAbstract> {
        (**self).inheritance()
    }

    #[inline]
    fn num_buffers(&self) -> usize {
        (**self).num_buffers()
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, PipelineMemoryAccess)> {
        (**self).buffer(index)
    }

    #[inline]
    fn num_images(&self) -> usize {
        (**self).num_images()
    }

    #[inline]
    fn image(
        &self,
        index: usize,
    ) -> Option<(
        &dyn ImageAccess,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
        ImageUninitializedSafe,
    )> {
        (**self).image(index)
    }
}

/// Represents a command buffer being executed by the GPU and the moment when the execution
/// finishes.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct CommandBufferExecFuture<F, Cb>
where
    F: GpuFuture,
    Cb: PrimaryCommandBuffer,
{
    previous: F,
    command_buffer: Cb,
    queue: Arc<Queue>,
    // True if the command buffer has already been submitted.
    // If flush is called multiple times, we want to block so that only one flushing is executed.
    // Therefore we use a `Mutex<bool>` and not an `AtomicBool`.
    submitted: Mutex<bool>,
    finished: AtomicBool,
}

unsafe impl<F, Cb> GpuFuture for CommandBufferExecFuture<F, Cb>
where
    F: GpuFuture,
    Cb: PrimaryCommandBuffer,
{
    #[inline]
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        Ok(match self.previous.build_submission()? {
            SubmitAnyBuilder::Empty => {
                let mut builder = SubmitCommandBufferBuilder::new();
                builder.add_command_buffer(self.command_buffer.inner());
                SubmitAnyBuilder::CommandBuffer(builder)
            }
            SubmitAnyBuilder::SemaphoresWait(sem) => {
                let mut builder: SubmitCommandBufferBuilder = sem.into();
                builder.add_command_buffer(self.command_buffer.inner());
                SubmitAnyBuilder::CommandBuffer(builder)
            }
            SubmitAnyBuilder::CommandBuffer(mut builder) => {
                // FIXME: add pipeline barrier
                builder.add_command_buffer(self.command_buffer.inner());
                SubmitAnyBuilder::CommandBuffer(builder)
            }
            SubmitAnyBuilder::QueuePresent(_) | SubmitAnyBuilder::BindSparse(_) => {
                unimplemented!() // TODO:
                                 /*present.submit();     // TODO: wrong
                                 let mut builder = SubmitCommandBufferBuilder::new();
                                 builder.add_command_buffer(self.command_buffer.inner());
                                 SubmitAnyBuilder::CommandBuffer(builder)*/
            }
        })
    }

    #[inline]
    fn flush(&self) -> Result<(), FlushError> {
        unsafe {
            let mut submitted = self.submitted.lock().unwrap();
            if *submitted {
                return Ok(());
            }

            let queue = self.queue.clone();

            match self.build_submission()? {
                SubmitAnyBuilder::Empty => {}
                SubmitAnyBuilder::CommandBuffer(builder) => {
                    builder.submit(&queue)?;
                }
                _ => unreachable!(),
            };

            // Only write `true` here in order to try again next time if we failed to submit.
            *submitted = true;
            Ok(())
        }
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        if self.finished.swap(true, Ordering::SeqCst) == false {
            self.command_buffer.unlock();
        }

        self.previous.signal_finished();
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        false
    }

    #[inline]
    fn queue(&self) -> Option<Arc<Queue>> {
        Some(self.queue.clone())
    }

    #[inline]
    fn check_buffer_access(
        &self,
        buffer: &dyn BufferAccess,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        match self
            .command_buffer
            .check_buffer_access(buffer, exclusive, queue)
        {
            Ok(v) => Ok(v),
            Err(AccessCheckError::Denied(err)) => Err(AccessCheckError::Denied(err)),
            Err(AccessCheckError::Unknown) => {
                self.previous.check_buffer_access(buffer, exclusive, queue)
            }
        }
    }

    #[inline]
    fn check_image_access(
        &self,
        image: &dyn ImageAccess,
        layout: ImageLayout,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        match self
            .command_buffer
            .check_image_access(image, layout, exclusive, queue)
        {
            Ok(v) => Ok(v),
            Err(AccessCheckError::Denied(err)) => Err(AccessCheckError::Denied(err)),
            Err(AccessCheckError::Unknown) => self
                .previous
                .check_image_access(image, layout, exclusive, queue),
        }
    }
}

unsafe impl<F, Cb> DeviceOwned for CommandBufferExecFuture<F, Cb>
where
    F: GpuFuture,
    Cb: PrimaryCommandBuffer,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.command_buffer.device()
    }
}

impl<F, Cb> Drop for CommandBufferExecFuture<F, Cb>
where
    F: GpuFuture,
    Cb: PrimaryCommandBuffer,
{
    fn drop(&mut self) {
        unsafe {
            if !*self.finished.get_mut() {
                // TODO: handle errors?
                self.flush().unwrap();
                // Block until the queue finished.
                self.queue.wait().unwrap();
                self.command_buffer.unlock();
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

impl error::Error for CommandBufferExecError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            CommandBufferExecError::AccessError { ref error, .. } => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for CommandBufferExecError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
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
