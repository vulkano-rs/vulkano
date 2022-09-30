// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{AccessCheckError, FlushError, GpuFuture, SubmitAnyBuilder};
use crate::{
    buffer::sys::UnsafeBuffer,
    command_buffer::{SemaphoreSubmitInfo, SubmitInfo},
    device::{Device, DeviceOwned, Queue},
    image::{sys::UnsafeImage, ImageLayout},
    sync::{AccessError, AccessFlags, PipelineStages, Semaphore},
    DeviceSize,
};
use parking_lot::Mutex;
use smallvec::smallvec;
use std::{
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// Builds a new semaphore signal future.
#[inline]
pub fn then_signal_semaphore<F>(future: F) -> SemaphoreSignalFuture<F>
where
    F: GpuFuture,
{
    let device = future.device().clone();

    assert!(future.queue().is_some()); // TODO: document

    SemaphoreSignalFuture {
        previous: future,
        semaphore: Arc::new(Semaphore::from_pool(device).unwrap()),
        wait_submitted: Mutex::new(false),
        finished: AtomicBool::new(false),
    }
}

/// Represents a semaphore being signaled after a previous event.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished \
              processing the submission"]
pub struct SemaphoreSignalFuture<F>
where
    F: GpuFuture,
{
    previous: F,
    semaphore: Arc<Semaphore>,
    // True if the signaling command has already been submitted.
    // If flush is called multiple times, we want to block so that only one flushing is executed.
    // Therefore we use a `Mutex<bool>` and not an `AtomicBool`.
    wait_submitted: Mutex<bool>,
    finished: AtomicBool,
}

unsafe impl<F> GpuFuture for SemaphoreSignalFuture<F>
where
    F: GpuFuture,
{
    #[inline]
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        // Flushing the signaling part, since it must always be submitted before the waiting part.
        self.flush()?;

        let sem = smallvec![self.semaphore.clone()];
        Ok(SubmitAnyBuilder::SemaphoresWait(sem))
    }

    fn flush(&self) -> Result<(), FlushError> {
        unsafe {
            let mut wait_submitted = self.wait_submitted.lock();

            if *wait_submitted {
                return Ok(());
            }

            let queue = self.previous.queue().unwrap();

            match self.previous.build_submission()? {
                SubmitAnyBuilder::Empty => {
                    let mut queue_guard = queue.lock();
                    queue_guard.submit_unchecked(
                        [SubmitInfo {
                            signal_semaphores: vec![SemaphoreSubmitInfo::semaphore(
                                self.semaphore.clone(),
                            )],
                            ..Default::default()
                        }],
                        None,
                    )?;
                }
                SubmitAnyBuilder::SemaphoresWait(semaphores) => {
                    let mut queue_guard = queue.lock();
                    queue_guard.submit_unchecked(
                        [SubmitInfo {
                            wait_semaphores: semaphores
                                .into_iter()
                                .map(|semaphore| {
                                    SemaphoreSubmitInfo {
                                        stages: PipelineStages {
                                            // TODO: correct stages ; hard
                                            all_commands: true,
                                            ..PipelineStages::empty()
                                        },
                                        ..SemaphoreSubmitInfo::semaphore(semaphore)
                                    }
                                })
                                .collect(),
                            signal_semaphores: vec![SemaphoreSubmitInfo::semaphore(
                                self.semaphore.clone(),
                            )],
                            ..Default::default()
                        }],
                        None,
                    )?;
                }
                SubmitAnyBuilder::CommandBuffer(mut submit_info, fence) => {
                    debug_assert!(submit_info.signal_semaphores.is_empty());

                    submit_info
                        .signal_semaphores
                        .push(SemaphoreSubmitInfo::semaphore(self.semaphore.clone()));

                    let mut queue_guard = queue.lock();
                    queue_guard.submit_unchecked([submit_info], fence)?;
                }
                SubmitAnyBuilder::BindSparse(_, _) => {
                    unimplemented!() // TODO: how to do that?
                                     /*debug_assert_eq!(builder.num_signal_semaphores(), 0);
                                     builder.add_signal_semaphore(&self.semaphore);
                                     builder.submit(&queue)?;*/
                }
                SubmitAnyBuilder::QueuePresent(present_info) => {
                    // VUID-VkPresentIdKHR-presentIds-04999
                    for swapchain_info in &present_info.swapchain_infos {
                        if swapchain_info.present_id.map_or(false, |present_id| {
                            !swapchain_info.swapchain.try_claim_present_id(present_id)
                        }) {
                            return Err(FlushError::PresentIdLessThanOrEqual);
                        }

                        match self.previous.check_swapchain_image_acquired(
                            swapchain_info
                                .swapchain
                                .raw_image(swapchain_info.image_index)
                                .unwrap()
                                .image,
                            true,
                        ) {
                            Ok(_) => (),
                            Err(AccessCheckError::Unknown) => {
                                return Err(AccessError::SwapchainImageNotAcquired.into())
                            }
                            Err(AccessCheckError::Denied(e)) => return Err(e.into()),
                        }
                    }

                    let mut queue_guard = queue.lock();
                    queue_guard
                        .present_unchecked(present_info)
                        .map(|r| r.map(|_| ()))
                        .fold(Ok(()), Result::and)?;

                    // FIXME: problematic because if we return an error and flush() is called again, then we'll submit the present twice
                    queue_guard.submit_unchecked(
                        [SubmitInfo {
                            signal_semaphores: vec![SemaphoreSubmitInfo::semaphore(
                                self.semaphore.clone(),
                            )],
                            ..Default::default()
                        }],
                        None,
                    )?;
                }
            };

            // Only write `true` here in order to try again next time if an error occurs.
            *wait_submitted = true;
            Ok(())
        }
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        debug_assert!(*self.wait_submitted.lock());
        self.finished.store(true, Ordering::SeqCst);
        self.previous.signal_finished();
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        true
    }

    #[inline]
    fn queue(&self) -> Option<Arc<Queue>> {
        self.previous.queue()
    }

    #[inline]
    fn check_buffer_access(
        &self,
        buffer: &UnsafeBuffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        self.previous
            .check_buffer_access(buffer, range, exclusive, queue)
            .map(|_| None)
    }

    #[inline]
    fn check_image_access(
        &self,
        image: &UnsafeImage,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        self.previous
            .check_image_access(image, range, exclusive, expected_layout, queue)
            .map(|_| None)
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        image: &UnsafeImage,
        _before: bool,
    ) -> Result<(), AccessCheckError> {
        self.previous.check_swapchain_image_acquired(image, false)
    }
}

unsafe impl<F> DeviceOwned for SemaphoreSignalFuture<F>
where
    F: GpuFuture,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.semaphore.device()
    }
}

impl<F> Drop for SemaphoreSignalFuture<F>
where
    F: GpuFuture,
{
    fn drop(&mut self) {
        unsafe {
            if !*self.finished.get_mut() {
                // TODO: handle errors?
                self.flush().unwrap();
                // Block until the queue finished.
                self.queue().unwrap().lock().wait_idle().unwrap();
                self.previous.signal_finished();
            }
        }
    }
}
