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
    buffer::Buffer,
    device::{Device, DeviceOwned, Queue},
    image::{Image, ImageLayout},
    swapchain::Swapchain,
    DeviceSize,
};
use std::{ops::Range, sync::Arc};

/// Builds a future that represents "now".
#[inline]
pub fn now(device: Arc<Device>) -> NowFuture {
    NowFuture { device }
}

/// A dummy future that represents "now".
pub struct NowFuture {
    device: Arc<Device>,
}

unsafe impl GpuFuture for NowFuture {
    #[inline]
    fn cleanup_finished(&mut self) {}

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        Ok(SubmitAnyBuilder::Empty)
    }

    #[inline]
    fn flush(&self) -> Result<(), FlushError> {
        Ok(())
    }

    #[inline]
    unsafe fn signal_finished(&self) {}

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
        &self,
        _buffer: &Buffer,
        _range: Range<DeviceSize>,
        _exclusive: bool,
        _queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        Err(AccessCheckError::Unknown)
    }

    #[inline]
    fn check_image_access(
        &self,
        _image: &Image,
        _range: Range<DeviceSize>,
        _exclusive: bool,
        _expected_layout: ImageLayout,
        _queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        Err(AccessCheckError::Unknown)
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        _swapchain: &Swapchain,
        _image_index: u32,
        _before: bool,
    ) -> Result<(), AccessCheckError> {
        Err(AccessCheckError::Unknown)
    }
}

unsafe impl DeviceOwned for NowFuture {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
