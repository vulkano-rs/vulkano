// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    device::{DeviceOwned, Queue},
    swapchain::{PresentInfo, SwapchainPresentInfo},
    sync::Semaphore,
    OomError, VulkanError,
};
use std::{
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    sync::{atomic::Ordering, Arc},
};

/// Prototype for a submission that presents a swapchain on the screen.
// TODO: example here
#[derive(Debug)]
pub struct SubmitPresentBuilder {
    present_info: PresentInfo,
}

impl SubmitPresentBuilder {
    /// Builds a new empty `SubmitPresentBuilder`.
    #[inline]
    pub fn new() -> SubmitPresentBuilder {
        SubmitPresentBuilder {
            present_info: Default::default(),
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
    pub unsafe fn add_wait_semaphore(&mut self, semaphore: Arc<Semaphore>) {
        self.present_info.wait_semaphores.push(semaphore);
    }

    /// Adds an image of a swapchain to be presented.
    ///
    /// Allows to specify a present region.
    /// Areas outside the present region *can* be ignored by the Vulkan implementation for
    /// optimizations purposes.
    ///
    /// If `VK_KHR_incremental_present` is not enabled, the `present_region` parameter is ignored.
    ///
    /// If `present_id` feature is not enabled, then the `present_id` parameter is ignored.
    ///
    /// # Safety
    ///
    /// - If you submit this builder, the swapchain must be kept alive until you are
    ///   guaranteed that the GPU has finished presenting.
    ///
    /// - The swapchains and semaphores must all belong to the same device.
    #[inline]
    pub unsafe fn add_swapchain(&mut self, mut swapchain_info: SwapchainPresentInfo) {
        let SwapchainPresentInfo {
            swapchain,
            image_index,
            present_id,
            present_regions,
            _ne: _,
        } = &mut swapchain_info;

        debug_assert!((*image_index as u32) < swapchain.image_count());

        if !swapchain.device().enabled_features().present_id {
            *present_id = None;
        }

        if swapchain
            .device()
            .enabled_extensions()
            .khr_incremental_present
        {
            for rectangle in present_regions {
                assert!(rectangle.is_compatible_with(swapchain.as_ref()));
            }
        } else {
            *present_regions = Default::default();
        }

        self.present_info.swapchain_infos.push(swapchain_info);
    }

    /// Submits the command. Calls `vkQueuePresentKHR`.
    ///
    /// # Panic
    ///
    /// Panics if no swapchain image has been added to the builder.
    ///
    pub fn submit(self, queue: &Queue) -> Result<(), SubmitPresentError> {
        unsafe {
            assert!(
                !self.present_info.swapchain_infos.is_empty(),
                "Tried to submit a present command without any swapchain"
            );

            // VUID-VkPresentIdKHR-presentIds-04999
            for swapchain_info in &self.present_info.swapchain_infos {
                if let Some(present_id) = swapchain_info.present_id.map(Into::into) {
                    if swapchain_info
                        .swapchain
                        .prev_present_id()
                        .fetch_max(present_id, Ordering::SeqCst)
                        >= present_id
                    {
                        return Err(SubmitPresentError::PresentIdLessThanOrEqual);
                    }
                }
            }

            let mut queue_guard = queue.lock();
            let results = queue_guard.present_unchecked(self.present_info);

            for result in results {
                result?
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

    /// The swapchain has lost or doesn't have full-screen exclusivity possibly for
    /// implementation-specific reasons outside of the application’s control.
    FullScreenExclusiveModeLost,

    /// The surface has changed in a way that makes the swapchain unusable. You must query the
    /// surface's new properties and recreate a new swapchain if you want to continue drawing.
    OutOfDate,

    /// A non-zero present_id must be greater than any non-zero present_id passed previously
    /// for the same swapchain.
    PresentIdLessThanOrEqual,
}

impl Error for SubmitPresentError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            SubmitPresentError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl Display for SubmitPresentError {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match *self {
                SubmitPresentError::OomError(_) => "not enough memory",
                SubmitPresentError::DeviceLost => "the connection to the device has been lost",
                SubmitPresentError::SurfaceLost =>
                    "the surface of this swapchain is no longer valid",
                SubmitPresentError::OutOfDate => "the swapchain needs to be recreated",
                SubmitPresentError::FullScreenExclusiveModeLost => {
                    "the swapchain no longer has full-screen exclusivity"
                }
                SubmitPresentError::PresentIdLessThanOrEqual => {
                    "present id is less than or equal to previous"
                }
            }
        )
    }
}

impl From<VulkanError> for SubmitPresentError {
    #[inline]
    fn from(err: VulkanError) -> SubmitPresentError {
        match err {
            err @ VulkanError::OutOfHostMemory => SubmitPresentError::OomError(OomError::from(err)),
            err @ VulkanError::OutOfDeviceMemory => {
                SubmitPresentError::OomError(OomError::from(err))
            }
            VulkanError::DeviceLost => SubmitPresentError::DeviceLost,
            VulkanError::SurfaceLost => SubmitPresentError::SurfaceLost,
            VulkanError::OutOfDate => SubmitPresentError::OutOfDate,
            VulkanError::FullScreenExclusiveModeLost => {
                SubmitPresentError::FullScreenExclusiveModeLost
            }
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
