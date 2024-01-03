use super::{queue_present, AccessCheckError, GpuFuture, SubmitAnyBuilder};
use crate::{
    buffer::Buffer,
    command_buffer::{SemaphoreSubmitInfo, SubmitInfo},
    device::{Device, DeviceOwned, Queue},
    image::{Image, ImageLayout},
    swapchain::Swapchain,
    sync::{
        future::{queue_submit, AccessError},
        semaphore::Semaphore,
        PipelineStages,
    },
    DeviceSize, Validated, ValidationError, VulkanError,
};
use parking_lot::Mutex;
use smallvec::smallvec;
use std::{
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

/// Builds a new semaphore signal future.
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
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        // Flushing the signaling part, since it must always be submitted before the waiting part.
        self.flush()?;
        let sem = smallvec![self.semaphore.clone()];

        Ok(SubmitAnyBuilder::SemaphoresWait(sem))
    }

    fn flush(&self) -> Result<(), Validated<VulkanError>> {
        unsafe {
            let mut wait_submitted = self.wait_submitted.lock();

            if *wait_submitted {
                return Ok(());
            }

            let queue = self.previous.queue().unwrap();

            match self.previous.build_submission()? {
                SubmitAnyBuilder::Empty => {
                    queue_submit(
                        &queue,
                        SubmitInfo {
                            signal_semaphores: vec![SemaphoreSubmitInfo::new(
                                self.semaphore.clone(),
                            )],
                            ..Default::default()
                        },
                        None,
                        &self.previous,
                    )?;
                }
                SubmitAnyBuilder::SemaphoresWait(semaphores) => {
                    queue_submit(
                        &queue,
                        SubmitInfo {
                            wait_semaphores: semaphores
                                .into_iter()
                                .map(|semaphore| {
                                    SemaphoreSubmitInfo {
                                        // TODO: correct stages ; hard
                                        stages: PipelineStages::ALL_COMMANDS,
                                        ..SemaphoreSubmitInfo::new(semaphore)
                                    }
                                })
                                .collect(),
                            signal_semaphores: vec![SemaphoreSubmitInfo::new(
                                self.semaphore.clone(),
                            )],
                            ..Default::default()
                        },
                        None,
                        &self.previous,
                    )?;
                }
                SubmitAnyBuilder::CommandBuffer(mut submit_info, fence) => {
                    debug_assert!(submit_info.signal_semaphores.is_empty());

                    submit_info
                        .signal_semaphores
                        .push(SemaphoreSubmitInfo::new(self.semaphore.clone()));

                    queue_submit(&queue, submit_info, fence, &self.previous)?;
                }
                SubmitAnyBuilder::BindSparse(_, _) => {
                    unimplemented!() // TODO: how to do that?
                                     /*debug_assert_eq!(builder.num_signal_semaphores(), 0);
                                     builder.add_signal_semaphore(&self.semaphore);
                                     builder.submit(&queue)?;*/
                }
                SubmitAnyBuilder::QueuePresent(present_info) => {
                    for swapchain_info in &present_info.swapchains {
                        if swapchain_info.present_id.map_or(false, |present_id| {
                            !swapchain_info.swapchain.try_claim_present_id(present_id)
                        }) {
                            return Err(Box::new(ValidationError {
                                problem: "the provided `present_id` was not greater than any \
                                    `present_id` passed previously for the same swapchain"
                                    .into(),
                                vuids: &["VUID-VkPresentIdKHR-presentIds-04999"],
                                ..Default::default()
                            })
                            .into());
                        }

                        match self.previous.check_swapchain_image_acquired(
                            &swapchain_info.swapchain,
                            swapchain_info.image_index,
                            true,
                        ) {
                            Ok(_) => (),
                            Err(AccessCheckError::Unknown) => {
                                return Err(Box::new(ValidationError::from_error(
                                    AccessError::SwapchainImageNotAcquired,
                                ))
                                .into());
                            }
                            Err(AccessCheckError::Denied(err)) => {
                                return Err(Box::new(ValidationError::from_error(err)).into());
                            }
                        }
                    }

                    queue_present(&queue, present_info)?
                        .map(|r| r.map(|_| ()))
                        .fold(Ok(()), Result::and)?;

                    // FIXME: problematic because if we return an error and flush() is called again,
                    // then we'll submit the present twice
                    queue_submit(
                        &queue,
                        SubmitInfo {
                            signal_semaphores: vec![SemaphoreSubmitInfo::new(
                                self.semaphore.clone(),
                            )],
                            ..Default::default()
                        },
                        None,
                        &self.previous,
                    )?;
                }
            };

            // Only write `true` here in order to try again next time if an error occurs.
            *wait_submitted = true;
            Ok(())
        }
    }

    unsafe fn signal_finished(&self) {
        debug_assert!(*self.wait_submitted.lock());
        self.finished.store(true, Ordering::SeqCst);
        self.previous.signal_finished();
    }

    fn queue_change_allowed(&self) -> bool {
        true
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        self.previous.queue()
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        self.previous
            .check_buffer_access(buffer, range, exclusive, queue)
    }

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        self.previous
            .check_image_access(image, range, exclusive, expected_layout, queue)
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

unsafe impl<F> DeviceOwned for SemaphoreSignalFuture<F>
where
    F: GpuFuture,
{
    fn device(&self) -> &Arc<Device> {
        self.semaphore.device()
    }
}

impl<F> Drop for SemaphoreSignalFuture<F>
where
    F: GpuFuture,
{
    fn drop(&mut self) {
        if !*self.finished.get_mut() && !thread::panicking() {
            // TODO: handle errors?
            self.flush().unwrap();
            // Block until the queue finished.
            self.queue().unwrap().with(|mut q| q.wait_idle()).unwrap();

            unsafe {
                self.signal_finished();
            }
        }
    }
}
