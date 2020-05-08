// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::BufferAccess;
use command_buffer::submit::SubmitAnyBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use image::ImageLayout;
use sync::AccessCheckError;
use sync::AccessFlagBits;
use sync::FlushError;
use sync::GpuFuture;
use sync::PipelineStages;

use VulkanObject;

/// Joins two futures together.
// TODO: handle errors
#[inline]
pub fn join<F, S>(first: F, second: S) -> JoinFuture<F, S>
    where F: GpuFuture,
          S: GpuFuture
{
    assert_eq!(first.device().internal_object(),
               second.device().internal_object());

    if !first.queue_change_allowed() && !second.queue_change_allowed() {
        assert!(first.queue().unwrap().is_same(&second.queue().unwrap()));
    }

    JoinFuture {
        first: first,
        second: second,
    }
}

/// Two futures joined into one.
#[must_use]
pub struct JoinFuture<A, B> {
    first: A,
    second: B,
}

unsafe impl<A, B> DeviceOwned for JoinFuture<A, B>
    where A: DeviceOwned,
          B: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        let device = self.first.device();
        debug_assert_eq!(self.second.device().internal_object(),
                         device.internal_object());
        device
    }
}

unsafe impl<A, B> GpuFuture for JoinFuture<A, B>
    where A: GpuFuture,
          B: GpuFuture
{
    #[inline]
    fn cleanup_finished(&mut self) {
        self.first.cleanup_finished();
        self.second.cleanup_finished();
    }

    #[inline]
    fn flush(&self) -> Result<(), FlushError> {
        // Since each future remembers whether it has been flushed, there's no safety issue here
        // if we call this function multiple times.
        self.first.flush()?;
        self.second.flush()?;
        Ok(())
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        // TODO: review this function
        let first = self.first.build_submission()?;
        let second = self.second.build_submission()?;

        // In some cases below we have to submit previous command buffers already, this s done by flushing previous.
        // Since the implementation should remember being flushed it's safe to call build_submission multiple times
        Ok(match (first, second) {
               (SubmitAnyBuilder::Empty, b) => b,
               (a, SubmitAnyBuilder::Empty) => a,
               (SubmitAnyBuilder::SemaphoresWait(mut a), SubmitAnyBuilder::SemaphoresWait(b)) => {
                   a.merge(b);
                   SubmitAnyBuilder::SemaphoresWait(a)
               },
               (SubmitAnyBuilder::SemaphoresWait(a), SubmitAnyBuilder::CommandBuffer(b)) => {
                   self.second.flush()?;
                   SubmitAnyBuilder::SemaphoresWait(a)
               },
               (SubmitAnyBuilder::CommandBuffer(a), SubmitAnyBuilder::SemaphoresWait(b)) => {
                   self.first.flush()?;
                   SubmitAnyBuilder::SemaphoresWait(b)
               },
               (SubmitAnyBuilder::SemaphoresWait(a), SubmitAnyBuilder::QueuePresent(b)) => {
                   self.second.flush()?;
                   SubmitAnyBuilder::SemaphoresWait(a)
               },
               (SubmitAnyBuilder::QueuePresent(a), SubmitAnyBuilder::SemaphoresWait(b)) => {
                   self.first.flush()?;
                   SubmitAnyBuilder::SemaphoresWait(b)
               },
               (SubmitAnyBuilder::SemaphoresWait(a), SubmitAnyBuilder::BindSparse(b)) => {
                   self.second.flush()?;
                   SubmitAnyBuilder::SemaphoresWait(a)
               },
               (SubmitAnyBuilder::BindSparse(a), SubmitAnyBuilder::SemaphoresWait(b)) => {
                   self.first.flush()?;
                   SubmitAnyBuilder::SemaphoresWait(b)
               },
               (SubmitAnyBuilder::CommandBuffer(a), SubmitAnyBuilder::CommandBuffer(b)) => {
                   // TODO: we may want to add debug asserts here
                   let new = a.merge(b);
                   SubmitAnyBuilder::CommandBuffer(new)
               },
               (SubmitAnyBuilder::QueuePresent(a), SubmitAnyBuilder::QueuePresent(b)) => {
                   self.first.flush()?;
                   self.second.flush()?;
                   SubmitAnyBuilder::Empty
               },
               (SubmitAnyBuilder::CommandBuffer(a), SubmitAnyBuilder::QueuePresent(b)) => {
                   unimplemented!()
               },
               (SubmitAnyBuilder::QueuePresent(a), SubmitAnyBuilder::CommandBuffer(b)) => {
                   unimplemented!()
               },
               (SubmitAnyBuilder::BindSparse(a), SubmitAnyBuilder::QueuePresent(b)) => {
                   unimplemented!()
               },
               (SubmitAnyBuilder::QueuePresent(a), SubmitAnyBuilder::BindSparse(b)) => {
                   unimplemented!()
               },
               (SubmitAnyBuilder::BindSparse(a), SubmitAnyBuilder::CommandBuffer(b)) => {
                   unimplemented!()
               },
               (SubmitAnyBuilder::CommandBuffer(a), SubmitAnyBuilder::BindSparse(b)) => {
                   unimplemented!()
               },
               (SubmitAnyBuilder::BindSparse(mut a), SubmitAnyBuilder::BindSparse(b)) => {
                   match a.merge(b) {
                       Ok(()) => SubmitAnyBuilder::BindSparse(a),
                       Err(_) => {
                           // TODO: this happens if both bind sparse have been given a fence already
                           //       annoying, but not impossible, to handle
                           unimplemented!()
                       },
                   }
               },
           })
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        self.first.signal_finished();
        self.second.signal_finished();
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        self.first.queue_change_allowed() && self.second.queue_change_allowed()
    }

    #[inline]
    fn queue(&self) -> Option<Arc<Queue>> {
        match (self.first.queue(), self.second.queue()) {
            (Some(q1), Some(q2)) => if q1.is_same(&q2) {
                Some(q1)
            } else if self.first.queue_change_allowed() {
                Some(q2)
            } else if self.second.queue_change_allowed() {
                Some(q1)
            } else {
                None
            },
            (Some(q), None) => Some(q),
            (None, Some(q)) => Some(q),
            (None, None) => None,
        }
    }

    #[inline]
    fn check_buffer_access(
        &self, buffer: &dyn BufferAccess, exclusive: bool, queue: &Queue)
        -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        let first = self.first.check_buffer_access(buffer, exclusive, queue);
        let second = self.second.check_buffer_access(buffer, exclusive, queue);
        debug_assert!(!exclusive || !(first.is_ok() && second.is_ok()),
                      "Two futures gave exclusive access to the same resource");
        match (first, second) {
            (v, Err(AccessCheckError::Unknown)) => v,
            (Err(AccessCheckError::Unknown), v) => v,
            (Err(AccessCheckError::Denied(e1)), Err(AccessCheckError::Denied(e2))) =>
                Err(AccessCheckError::Denied(e1)),        // TODO: which one?
            (Ok(_), Err(AccessCheckError::Denied(_))) |
            (Err(AccessCheckError::Denied(_)), Ok(_)) => panic!("Contradictory information \
                                                                 between two futures"),
            (Ok(None), Ok(None)) => Ok(None),
            (Ok(Some(a)), Ok(None)) |
            (Ok(None), Ok(Some(a))) => Ok(Some(a)),
            (Ok(Some((a1, a2))), Ok(Some((b1, b2)))) => {
                Ok(Some((a1 | b1, a2 | b2)))
            },
        }
    }

    #[inline]
    fn check_image_access(&self, image: &dyn ImageAccess, layout: ImageLayout, exclusive: bool,
                          queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        let first = self.first
            .check_image_access(image, layout, exclusive, queue);
        let second = self.second
            .check_image_access(image, layout, exclusive, queue);
        debug_assert!(!exclusive || !(first.is_ok() && second.is_ok()),
                      "Two futures gave exclusive access to the same resource");
        match (first, second) {
            (v, Err(AccessCheckError::Unknown)) => v,
            (Err(AccessCheckError::Unknown), v) => v,
            (Err(AccessCheckError::Denied(e1)), Err(AccessCheckError::Denied(e2))) =>
                Err(AccessCheckError::Denied(e1)),        // TODO: which one?
            (Ok(_), Err(AccessCheckError::Denied(_))) |
            (Err(AccessCheckError::Denied(_)), Ok(_)) => panic!("Contradictory information \
                                                                 between two futures"),
            (Ok(None), Ok(None)) => Ok(None),
            (Ok(Some(a)), Ok(None)) |
            (Ok(None), Ok(Some(a))) => Ok(Some(a)),
            (Ok(Some((a1, a2))), Ok(Some((b1, b2)))) => {
                Ok(Some((a1 | b1, a2 | b2)))
            },
        }
    }
}
