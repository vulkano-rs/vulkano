// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error::Error;
use std::sync::Arc;

use buffer::BufferAccess;
use command_buffer::submit::SubmitAnyBuilder;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use sync::AccessFlagBits;
use sync::GpuFuture;
use sync::PipelineStages;

/// A dummy future that represents "now".
#[must_use]
pub struct DummyFuture {
    device: Arc<Device>,
}

impl DummyFuture {
    /// Builds a new dummy future.
    #[inline]
    pub fn new(device: Arc<Device>) -> DummyFuture {
        DummyFuture {
            device: device,
        }
    }
}

unsafe impl GpuFuture for DummyFuture {
    #[inline]
    fn cleanup_finished(&mut self) {
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Box<Error>> {
        Ok(SubmitAnyBuilder::Empty)
    }

    #[inline]
    fn flush(&self) -> Result<(), Box<Error>> {
        Ok(())
    }

    #[inline]
    unsafe fn signal_finished(&self) {
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        true
    }

    #[inline]
    fn queue(&self) -> Option<&Arc<Queue>> {
        None
    }

    #[inline]
    fn check_buffer_access(&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        Err(())
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, exclusive: bool, queue: &Queue)
                           -> Result<Option<(PipelineStages, AccessFlagBits)>, ()>
    {
        Err(())
    }
}

unsafe impl DeviceOwned for DummyFuture {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
