// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::borrow::Cow;
use std::error;
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use SafeDeref;
use VulkanObject;
use buffer::BufferAccess;
use command_buffer::submit::SubmitAnyBuilder;
use command_buffer::submit::SubmitCommandBufferBuilder;
use command_buffer::sys::UnsafeCommandBuffer;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use image::ImageLayout;
use sync::AccessCheckError;
use sync::AccessError;
use sync::AccessFlagBits;
use sync::FlushError;
use sync::GpuFuture;
use sync::NowFuture;
use sync::PipelineStages;
use sync::now;

pub unsafe trait CommandBuffer: DeviceOwned {
    /// The command pool of the command buffer.
    type PoolAlloc;

    /// Returns the underlying `UnsafeCommandBuffer` of this command buffer.
    fn inner(&self) -> &UnsafeCommandBuffer<Self::PoolAlloc>;

    /*/// Returns the queue family of the command buffer.
    #[inline]
    fn queue_family(&self) -> QueueFamily
        where Self::PoolAlloc: CommandPoolAlloc
    {
        self.inner().queue_family()
    }*/

    /// Checks whether this command buffer is allowed to be submitted after the `future` and on
    /// the given queue, and if so locks it.
    ///
    /// If you call this function, then you should call `unlock` afterwards.
    fn lock_submit(&self, future: &GpuFuture, queue: &Queue) -> Result<(), CommandBufferExecError>;

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
    fn execute(self, queue: Arc<Queue>)
               -> Result<CommandBufferExecFuture<NowFuture, Self>, CommandBufferExecError>
        where Self: Sized + 'static
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
    fn execute_after<F>(self, future: F, queue: Arc<Queue>)
                        -> Result<CommandBufferExecFuture<F, Self>, CommandBufferExecError>
        where Self: Sized + 'static,
              F: GpuFuture
    {
        assert_eq!(self.device().internal_object(),
                   future.device().internal_object());

        if !future.queue_change_allowed() {
            assert!(future.queue().unwrap().is_same(&queue));
        }

        self.lock_submit(&future, &queue)?;

        Ok(CommandBufferExecFuture {
               previous: future,
               command_buffer: self,
               queue: queue,
               submitted: Mutex::new(false),
               finished: AtomicBool::new(false),
           })
    }

    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>;

    fn check_image_access(&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool,
                          queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>;

    // FIXME: lots of other methods
}

unsafe impl<T> CommandBuffer for T
    where T: SafeDeref,
          T::Target: CommandBuffer
{
    type PoolAlloc = <T::Target as CommandBuffer>::PoolAlloc;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::PoolAlloc> {
        (**self).inner()
    }

    #[inline]
    fn lock_submit(&self, future: &GpuFuture, queue: &Queue) -> Result<(), CommandBufferExecError> {
        (**self).lock_submit(future, queue)
    }

    #[inline]
    unsafe fn unlock(&self) {
        (**self).unlock();
    }

    #[inline]
    fn check_buffer_access(
        &self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
        -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        (**self).check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool,
                          queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        (**self).check_image_access(image, layout, exclusive, queue)
    }
}

/// Represents a command buffer being executed by the GPU and the moment when the execution
/// finishes.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct CommandBufferExecFuture<F, Cb>
    where F: GpuFuture,
          Cb: CommandBuffer
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
    where F: GpuFuture,
          Cb: CommandBuffer
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
               },
               SubmitAnyBuilder::SemaphoresWait(sem) => {
                   let mut builder: SubmitCommandBufferBuilder = sem.into();
                   builder.add_command_buffer(self.command_buffer.inner());
                   SubmitAnyBuilder::CommandBuffer(builder)
               },
               SubmitAnyBuilder::CommandBuffer(mut builder) => {
                   // FIXME: add pipeline barrier
                   builder.add_command_buffer(self.command_buffer.inner());
                   SubmitAnyBuilder::CommandBuffer(builder)
               },
               SubmitAnyBuilder::QueuePresent(_) |
               SubmitAnyBuilder::BindSparse(_) => {
                   unimplemented!() // TODO:
                /*present.submit();     // TODO: wrong
                let mut builder = SubmitCommandBufferBuilder::new();
                builder.add_command_buffer(self.command_buffer.inner());
                SubmitAnyBuilder::CommandBuffer(builder)*/
               },
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
                SubmitAnyBuilder::Empty => {},
                SubmitAnyBuilder::CommandBuffer(builder) => {
                    builder.submit(&queue)?;
                },
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
        &self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
        -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        match self.command_buffer
            .check_buffer_access(buffer, exclusive, queue) {
            Ok(v) => Ok(v),
            Err(AccessCheckError::Denied(err)) => Err(AccessCheckError::Denied(err)),
            Err(AccessCheckError::Unknown) => {
                self.previous.check_buffer_access(buffer, exclusive, queue)
            },
        }
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool,
                          queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        match self.command_buffer
            .check_image_access(image, layout, exclusive, queue) {
            Ok(v) => Ok(v),
            Err(AccessCheckError::Denied(err)) => Err(AccessCheckError::Denied(err)),
            Err(AccessCheckError::Unknown) => {
                self.previous
                    .check_image_access(image, layout, exclusive, queue)
            },
        }
    }
}

unsafe impl<F, Cb> DeviceOwned for CommandBufferExecFuture<F, Cb>
    where F: GpuFuture,
          Cb: CommandBuffer
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.command_buffer.device()
    }
}

impl<F, Cb> Drop for CommandBufferExecFuture<F, Cb>
    where F: GpuFuture,
          Cb: CommandBuffer
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
    fn description(&self) -> &str {
        match *self {
            CommandBufferExecError::AccessError { .. } => {
                "access to a resource has been denied"
            },
            CommandBufferExecError::OneTimeSubmitAlreadySubmitted => {
                "the command buffer or one of the secondary command buffers it executes was \
                 created with the \"one time submit\" flag, but has already been submitted it \
                 the past"
            },
            CommandBufferExecError::ExclusiveAlreadyInUse => {
                "the command buffer or one of the secondary command buffers it executes is \
                 already in use by the GPU and was not created with the \"concurrent\" flag"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            CommandBufferExecError::AccessError { ref error, .. } => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for CommandBufferExecError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
