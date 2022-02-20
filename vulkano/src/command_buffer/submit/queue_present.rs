// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;

use crate::device::DeviceOwned;
use crate::device::Queue;
use crate::swapchain::PresentRegion;
use crate::swapchain::Swapchain;
use crate::sync::Semaphore;

use crate::check_errors;
use crate::Error;
use crate::OomError;
use crate::SynchronizedVulkanObject;
use crate::VulkanObject;

/// Prototype for a submission that presents a swapchain on the screen.
// TODO: example here
pub struct SubmitPresentBuilder<'a> {
    wait_semaphores: SmallVec<[ash::vk::Semaphore; 8]>,
    swapchains: SmallVec<[ash::vk::SwapchainKHR; 4]>,
    image_indices: SmallVec<[u32; 4]>,
    present_regions: SmallVec<[ash::vk::PresentRegionKHR; 4]>,
    rect_layers: SmallVec<[ash::vk::RectLayerKHR; 4]>,
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
            present_regions: SmallVec::new(),
            rect_layers: SmallVec::new(),
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
    /// Allows to specify a present region.
    /// Areas outside the present region *can* be ignored by the Vulkan implementation for
    /// optimizations purposes.
    ///
    /// If `VK_KHR_incremental_present` is not enabled, the `present_region` parameter is ignored.
    ///
    /// # Safety
    ///
    /// - If you submit this builder, the swapchain must be kept alive until you are
    ///   guaranteed that the GPU has finished presenting.
    ///
    /// - The swapchains and semaphores must all belong to the same device.
    ///
    #[inline]
    pub unsafe fn add_swapchain<W>(
        &mut self,
        swapchain: &'a Swapchain<W>,
        image_num: u32,
        present_region: Option<&'a PresentRegion>,
    ) {
        debug_assert!(image_num < swapchain.image_count());

        if swapchain
            .device()
            .enabled_extensions()
            .khr_incremental_present
        {
            let vk_present_region = match present_region {
                Some(present_region) => {
                    assert!(present_region.is_compatible_with(swapchain));
                    for rectangle in &present_region.rectangles {
                        self.rect_layers.push(rectangle.into());
                    }
                    ash::vk::PresentRegionKHR {
                        rectangle_count: present_region.rectangles.len() as u32,
                        // Set this to null for now; in submit fill it with self.rect_layers
                        p_rectangles: ptr::null(),
                    }
                }
                None => ash::vk::PresentRegionKHR {
                    rectangle_count: 0,
                    p_rectangles: ptr::null(),
                },
            };
            self.present_regions.push(vk_present_region);
        }

        self.swapchains.push(swapchain.internal_object());
        self.image_indices.push(image_num);
    }

    /// Submits the command. Calls `vkQueuePresentKHR`.
    ///
    /// # Panic
    ///
    /// Panics if no swapchain image has been added to the builder.
    ///
    pub fn submit(mut self, queue: &Queue) -> Result<(), SubmitPresentError> {
        unsafe {
            debug_assert_eq!(self.swapchains.len(), self.image_indices.len());
            assert!(
                !self.swapchains.is_empty(),
                "Tried to submit a present command without any swapchain"
            );

            let present_regions = {
                if !self.present_regions.is_empty() {
                    debug_assert!(queue.device().enabled_extensions().khr_incremental_present);
                    debug_assert_eq!(self.swapchains.len(), self.present_regions.len());
                    let mut current_index = 0;
                    for present_region in &mut self.present_regions {
                        present_region.p_rectangles = self.rect_layers[current_index..].as_ptr();
                        current_index += present_region.rectangle_count as usize;
                    }
                    Some(ash::vk::PresentRegionsKHR {
                        swapchain_count: self.present_regions.len() as u32,
                        p_regions: self.present_regions.as_ptr(),
                        ..Default::default()
                    })
                } else {
                    None
                }
            };

            let mut results = vec![ash::vk::Result::SUCCESS; self.swapchains.len()];

            let fns = queue.device().fns();
            let queue = queue.internal_object_guard();

            let infos = ash::vk::PresentInfoKHR {
                p_next: present_regions
                    .as_ref()
                    .map(|pr| pr as *const ash::vk::PresentRegionsKHR as *const _)
                    .unwrap_or(ptr::null()),
                wait_semaphore_count: self.wait_semaphores.len() as u32,
                p_wait_semaphores: self.wait_semaphores.as_ptr(),
                swapchain_count: self.swapchains.len() as u32,
                p_swapchains: self.swapchains.as_ptr(),
                p_image_indices: self.image_indices.as_ptr(),
                p_results: results.as_mut_ptr(),
                ..Default::default()
            };

            check_errors(fns.khr_swapchain.queue_present_khr(*queue, &infos))?;

            for result in results {
                check_errors(result)?;
            }

            Ok(())
        }
    }
}

impl<'a> fmt::Debug for SubmitPresentBuilder<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_struct("SubmitPresentBuilder")
            .field("wait_semaphores", &self.wait_semaphores)
            .field("swapchains", &self.swapchains)
            .field("image_indices", &self.image_indices)
            .finish()
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
    FullScreenExclusiveLost,

    /// The surface has changed in a way that makes the swapchain unusable. You must query the
    /// surface's new properties and recreate a new swapchain if you want to continue drawing.
    OutOfDate,
}

impl error::Error for SubmitPresentError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            SubmitPresentError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SubmitPresentError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                SubmitPresentError::OomError(_) => "not enough memory",
                SubmitPresentError::DeviceLost => "the connection to the device has been lost",
                SubmitPresentError::SurfaceLost =>
                    "the surface of this swapchain is no longer valid",
                SubmitPresentError::OutOfDate => "the swapchain needs to be recreated",
                SubmitPresentError::FullScreenExclusiveLost => {
                    "the swapchain no longer has full-screen exclusivity"
                }
            }
        )
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
            Error::FullScreenExclusiveLost => SubmitPresentError::FullScreenExclusiveLost,
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
