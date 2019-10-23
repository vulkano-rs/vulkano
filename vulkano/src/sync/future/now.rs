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

/// Builds a future that represents "now".
#[inline]
pub fn now(device: Arc<Device>) -> NowFuture {
    NowFuture { device: device }
}

/// A dummy future that represents "now".
pub struct NowFuture {
    device: Arc<Device>,
}

unsafe impl GpuFuture for NowFuture {
    #[inline]
    fn cleanup_finished(&mut self) {
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        Ok(SubmitAnyBuilder::Empty)
    }

    #[inline]
    fn flush(&self) -> Result<(), FlushError> {
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
    fn queue(&self) -> Option<Arc<Queue>> {
        None
    }

    #[inline]
    fn check_buffer_access(
        &self, buffer: &dyn BufferAccess, _: bool, _: &Queue)
        -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        Err(AccessCheckError::Unknown)
    }

    #[inline]
    fn check_image_access(&self, _: &dyn ImageAccess, _: ImageLayout, _: bool, _: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        Err(AccessCheckError::Unknown)
    }
}

unsafe impl DeviceOwned for NowFuture {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
