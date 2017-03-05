// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use buffer::Buffer;
use command_buffer::cb::UnsafeCommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::submit::SubmitAnyBuilder;
use command_buffer::submit::SubmitCommandBufferBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::Image;
use sync::AccessFlagBits;
use sync::DummyFuture;
use sync::GpuFuture;
use sync::PipelineStages;
use SafeDeref;
use VulkanObject;

pub unsafe trait CommandBuffer: DeviceOwned {
    /// The command pool of the command buffer.
    type Pool: CommandPool;

    /// Returns the underlying `UnsafeCommandBuffer` of this command buffer.
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool>;

    /// Checks whether this command buffer is allowed to be submitted after the `future` and on
    /// the given queue.
    ///
    /// **You should not call this function directly**, otherwise any further attempt to submit
    /// will return a runtime error.
    // TODO: better error
    fn submit_check(&self, future: &GpuFuture, queue: &Queue) -> Result<(), Box<error::Error>>;

    /// Executes this command buffer on a queue.
    ///
    /// > **Note**: This is just a shortcut for `execute_after`.
    ///
    /// # Panic
    ///
    /// Panics if the device of the command buffer is not the same as the device of the future.
    #[inline]
    fn execute(self, queue: Arc<Queue>) -> CommandBufferExecFuture<DummyFuture, Self>
        where Self: Sized + 'static
    {
        let device = queue.device().clone();
        self.execute_after(DummyFuture::new(device), queue)
    }

    /// Executes the command buffer after an existing future.
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
    fn execute_after<F>(self, future: F, queue: Arc<Queue>) -> CommandBufferExecFuture<F, Self>
        where Self: Sized + 'static, F: GpuFuture
    {
        assert_eq!(self.device().internal_object(), future.device().internal_object());

        self.submit_check(&future, &queue).expect("Forbidden");     // TODO: error

        if !future.queue_change_allowed() {
            assert!(future.queue().unwrap().is_same(&queue));
        }

        CommandBufferExecFuture {
            previous: future,
            command_buffer: self,
            queue: queue,
            submitted: Mutex::new(false),
            finished: AtomicBool::new(false),
        }
    }

    // FIXME: lots of other methods
}

unsafe impl<T> CommandBuffer for T where T: SafeDeref, T::Target: CommandBuffer {
    type Pool = <T::Target as CommandBuffer>::Pool;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        (**self).inner()
    }

    #[inline]
    fn submit_check(&self, future: &GpuFuture, queue: &Queue) -> Result<(), Box<error::Error>> {
        (**self).submit_check(future, queue)
    }
}

/// Represents a command buffer being executed by the GPU and the moment when the execution
/// finishes.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct CommandBufferExecFuture<F, Cb> where F: GpuFuture, Cb: CommandBuffer {
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
    where F: GpuFuture, Cb: CommandBuffer
{
    #[inline]
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Box<error::Error>> {
        Ok(match try!(self.previous.build_submission()) {
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
            SubmitAnyBuilder::QueuePresent(present) => {
                unimplemented!()        // TODO:
                /*present.submit();     // TODO: wrong
                let mut builder = SubmitCommandBufferBuilder::new();
                builder.add_command_buffer(self.command_buffer.inner());
                SubmitAnyBuilder::CommandBuffer(builder)*/
            },
        })
    }

    #[inline]
    fn flush(&self) -> Result<(), Box<error::Error>> {
        unsafe {
            let mut submitted = self.submitted.lock().unwrap();
            if *submitted {
                return Ok(());
            }

            let queue = self.queue.clone();

            match try!(self.build_submission()) {
                SubmitAnyBuilder::Empty => {},
                SubmitAnyBuilder::CommandBuffer(builder) => {
                    try!(builder.submit(&queue));
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
        self.finished.store(true, Ordering::SeqCst);
        self.previous.signal_finished();
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        false
    }

    #[inline]
    fn queue(&self) -> Option<&Arc<Queue>> {
        debug_assert!(match self.previous.queue() {
            None => true,
            Some(q) => q.is_same(&self.queue)
        });

        Some(&self.queue)
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &Buffer, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        // FIXME: check the command buffer too
        self.previous.check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(&self, image: &Image, exclusive: bool, queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        // FIXME: check the command buffer too
        self.previous.check_image_access(image, exclusive, queue)
    }
}

unsafe impl<F, Cb> DeviceOwned for CommandBufferExecFuture<F, Cb>
    where F: GpuFuture, Cb: CommandBuffer
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.command_buffer.device()
    }
}

impl<F, Cb> Drop for CommandBufferExecFuture<F, Cb> where F: GpuFuture, Cb: CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            if !*self.finished.get_mut() {
                // TODO: handle errors?
                self.flush().unwrap();
                // Block until the queue finished.
                self.queue.wait().unwrap();
                self.previous.signal_finished();
            }
        }
    }
}
