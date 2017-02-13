// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use smallvec::SmallVec;

use command_buffer::cb::UnsafeCommandBuffer;
use command_buffer::pool::CommandPool;
use device::Queue;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

use check_errors;
use vk;
use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use SynchronizedVulkanObject;

/// Prototype for a submission that executes command buffers.
#[derive(Debug)]
pub struct SubmitCommandBufferBuilder<'a> {
    wait_semaphores: SmallVec<[vk::Semaphore; 16]>,
    dest_stages: SmallVec<[vk::PipelineStageFlags; 8]>,
    signal_semaphores: SmallVec<[vk::Semaphore; 16]>,
    command_buffers: SmallVec<[vk::CommandBuffer; 4]>,
    fence: vk::Fence,
    marker: PhantomData<&'a ()>,
}

impl<'a> SubmitCommandBufferBuilder<'a> {
    /// Builds a new empty `SubmitCommandBufferBuilder`.
    #[inline]
    pub fn new() -> SubmitCommandBufferBuilder<'a> {
        SubmitCommandBufferBuilder {
            wait_semaphores: SmallVec::new(),
            dest_stages: SmallVec::new(),
            signal_semaphores: SmallVec::new(),
            command_buffers: SmallVec::new(),
            fence: 0,
            marker: PhantomData,
        }
    }

    /// Returns true if this builder will signal a fence when submitted.
    #[inline]
    pub fn has_fence(&self) -> bool {
        self.fence != 0
    }

    /// Adds an operation that signals a fence after this submission ends.
    ///
    /// If a fence was previously set, it will no longer be signaled.
    #[inline]
    pub unsafe fn set_fence_signal(&mut self, fence: &'a Fence) {
        self.fence = fence.internal_object()
    }

    /// Adds a semaphore to be waited upon before the command buffers are executed.
    ///
    /// Only the given `stages` of the command buffers added afterwards will wait upon
    /// the semaphore. Other stages not included in `stages` can execute before waiting.
    #[inline]
    pub unsafe fn add_wait_semaphore(&mut self, semaphore: &'a Semaphore, stages: PipelineStages) {
        debug_assert!(Into::<vk::PipelineStageFlagBits>::into(stages) != 0);
        self.wait_semaphores.push(semaphore.internal_object());
        self.dest_stages.push(stages.into());
    }

    /// Adds a command buffer that is executed as part of this command.
    ///
    /// The command buffers are submitted in the order in which they are added.
    #[inline]
    pub unsafe fn add_command_buffer<P>(&mut self, command_buffer: &'a UnsafeCommandBuffer<P>)
        where P: CommandPool
    {
        self.command_buffers.push(command_buffer.internal_object());
    }

    /// Returns the number of semaphores to signal.
    #[inline]
    pub fn num_signal_semaphores(&self) -> usize {
        self.signal_semaphores.len()
    }

    /// Adds a semaphore that is going to be signaled at the end of the submission.
    #[inline]
    pub unsafe fn add_signal_semaphore(&mut self, semaphore: &'a Semaphore) {
        self.signal_semaphores.push(semaphore.internal_object());
    }

    /// Submits the command buffer.
    pub fn submit(mut self, queue: &Queue) -> Result<(), SubmitCommandBufferError> {
        unsafe {
            let vk = queue.device().pointers();
            let queue = queue.internal_object_guard();

            debug_assert_eq!(self.wait_semaphores.len(), self.dest_stages.len());

            let batch = vk::SubmitInfo {
                sType: vk::STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: self.wait_semaphores.len() as u32,
                pWaitSemaphores: self.wait_semaphores.as_ptr(),
                pWaitDstStageMask: self.dest_stages.as_ptr(),
                commandBufferCount: self.command_buffers.len() as u32,
                pCommandBuffers: self.command_buffers.as_ptr(),
                signalSemaphoreCount: self.signal_semaphores.len() as u32,
                pSignalSemaphores: self.signal_semaphores.as_ptr(),
            };

            try!(check_errors(vk.QueueSubmit(*queue, 1, &batch, self.fence)));
            Ok(())
        }
    }
}

/// Error that can happen when submitting the prototype.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SubmitCommandBufferError {
    /// Not enough memory.
    OomError(OomError),

    /// The connection to the device has been lost.
    DeviceLost,
}

impl error::Error for SubmitCommandBufferError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            SubmitCommandBufferError::OomError(_) => "not enough memory",
            SubmitCommandBufferError::DeviceLost => "the connection to the device has been lost",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            SubmitCommandBufferError::OomError(ref err) => Some(err),
            _ => None
        }
    }
}

impl fmt::Display for SubmitCommandBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for SubmitCommandBufferError {
    #[inline]
    fn from(err: Error) -> SubmitCommandBufferError {
        match err {
            err @ Error::OutOfHostMemory => SubmitCommandBufferError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => SubmitCommandBufferError::OomError(OomError::from(err)),
            Error::DeviceLost => SubmitCommandBufferError::DeviceLost,
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}
