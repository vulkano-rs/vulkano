// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use device::Queue;
use swapchain::Swapchain;
use sync::Semaphore;

use Error;
use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use check_errors;
use vk;

/// Prototype for a submission that presents a swapchain on the screen.
// TODO: example here
#[derive(Debug)]
pub struct SubmitPresentBuilder<'a> {
    wait_semaphores: SmallVec<[vk::Semaphore; 8]>,
    swapchains: SmallVec<[vk::SwapchainKHR; 4]>,
    image_indices: SmallVec<[u32; 4]>,
    marker: PhantomData<&'a ()>,
}

impl<'a> SubmitPresentBuilder<'a> {
    /// Builds a new empty `SubmitPresentBuilder`.
    #[inline]
    pub fn new() -> SubmitPresentBuilder<'a> {
        SubmitPresentBuilder {
            wait_semaphores: SmallVec::new(),
            swapchains: SmallVec::new(),
            image_indices: SmallVec::new(),
            marker: PhantomData,
        }
    }

    /// Adds a semaphore to be waited upon before the presents are executed.
    ///
    /// # Safety
    ///
    /// - If you submit this builder, the semaphore must be kept alive until you are guaranteed
    ///   that the GPU has presented the swapchains.
    ///
    /// - If you submit this builder, no other queue must be waiting on these semaphores. In other
    ///   words, each semaphore signal can only correspond to one semaphore wait.
    ///
    /// - If you submit this builder, the semaphores must be signaled when the queue execution
    ///   reaches this submission, or there must be one or more submissions in queues that are
    ///   going to signal these semaphores. In other words, you must not block the queue with
    ///   semaphores that can't get signaled.
    ///
    /// - The swapchains and semaphores must all belong to the same device.
    ///
    #[inline]
    pub unsafe fn add_wait_semaphore(&mut self, semaphore: &'a Semaphore) {
        self.wait_semaphores.push(semaphore.internal_object());
    }

    /// Adds an image of a swapchain to be presented.
    ///
    /// # Safety
    ///
    /// - If you submit this builder, the swapchain must be kept alive until you are
    ///   guaranteed that the GPU has finished presenting.
    ///
    /// - The swapchains and semaphores must all belong to the same device.
    ///
    #[inline]
    pub unsafe fn add_swapchain(&mut self, swapchain: &'a Swapchain, image_num: u32) {
        debug_assert!(image_num < swapchain.num_images());
        self.swapchains.push(swapchain.internal_object());
        self.image_indices.push(image_num);
    }

    /// Submits the command. Calls `vkQueuePresentKHR`.
    ///
    /// # Panic
    ///
    /// Panics if no swapchain image has been added to the builder.
    ///
    pub fn submit(self, queue: &Queue) -> Result<(), SubmitPresentError> {
        unsafe {
            debug_assert_eq!(self.swapchains.len(), self.image_indices.len());
            assert!(!self.swapchains.is_empty(),
                    "Tried to submit a present command without any swapchain");

            let vk = queue.device().pointers();
            let queue = queue.internal_object_guard();

            let mut results = vec![mem::uninitialized(); self.swapchains.len()]; // TODO: alloca

            let infos = vk::PresentInfoKHR {
                sType: vk::STRUCTURE_TYPE_PRESENT_INFO_KHR,
                pNext: ptr::null(),
                waitSemaphoreCount: self.wait_semaphores.len() as u32,
                pWaitSemaphores: self.wait_semaphores.as_ptr(),
                swapchainCount: self.swapchains.len() as u32,
                pSwapchains: self.swapchains.as_ptr(),
                pImageIndices: self.image_indices.as_ptr(),
                pResults: results.as_mut_ptr(),
            };

            check_errors(vk.QueuePresentKHR(*queue, &infos))?;

            for result in results {
                // TODO: AMD driver initially didn't write the results ; check that it's been fixed
                //try!(check_errors(result));
            }

            Ok(())
        }
    }
}

/// Error that can happen when submitting the present prototype.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SubmitPresentError {
    /// Not enough memory.
    OomError(OomError),

    /// The connection to the device has been lost.
    DeviceLost,

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,

    /// The surface has changed in a way that makes the swapchain unusable. You must query the
    /// surface's new properties and recreate a new swapchain if you want to continue drawing.
    OutOfDate,
}

impl error::Error for SubmitPresentError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            SubmitPresentError::OomError(_) => "not enough memory",
            SubmitPresentError::DeviceLost => "the connection to the device has been lost",
            SubmitPresentError::SurfaceLost => "the surface of this swapchain is no longer valid",
            SubmitPresentError::OutOfDate => "the swapchain needs to be recreated",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            SubmitPresentError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SubmitPresentError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for SubmitPresentError {
    #[inline]
    fn from(err: Error) -> SubmitPresentError {
        match err {
            err @ Error::OutOfHostMemory => SubmitPresentError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => SubmitPresentError::OomError(OomError::from(err)),
            Error::DeviceLost => SubmitPresentError::DeviceLost,
            Error::SurfaceLost => SubmitPresentError::SurfaceLost,
            Error::OutOfDate => SubmitPresentError::OutOfDate,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_swapchain_added() {
        let (_, queue) = gfx_dev_and_queue!();
        assert_should_panic!("Tried to submit a present command without any swapchain", {
            let _ = SubmitPresentBuilder::new().submit(&queue);
        });
    }
}
