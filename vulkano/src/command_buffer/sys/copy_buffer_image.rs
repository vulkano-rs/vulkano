// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::sys::KeepAlive;
use image::traits::ImageContent;

use vk;

/// Prototype for a command that copies data between buffers.
pub struct BufferImageCopyCommand {
    keep_alive_src: Arc<KeepAlive + 'static>,
    keep_alive_dest: Arc<KeepAlive + 'static>,
    device: vk::Device,
    internal_image: vk::Image,
    internal_buffer: vk::Buffer,
}

impl BufferImageCopyCommand {
    /// 
    ///
    /// # Panic
    ///
    /// - Panicks if the image and the buffer were not allocated with the same device.
    ///
    pub fn buffer_to_image<'t, S, P, D, I>(dest: &Arc<D>, ranges: I)
                                           -> Result<BufferImageCopyCommand, BufferImageCopyError>
        where D: ImageContent<P> + Send + Sync + 'static,
              I: IntoIterator<Item = BufferSlice<'t, [P], S>>,
              S: Buffer + Send + Sync + 'static
    {
        if dest.samples() != 1 {
            return Err(BufferImageCopyError::WrongSamplesCount);
        }

        if !dest.inner_image().usage_transfer_dest() {
            return Err(BufferImageCopyError::WrongUsageFlag);
        }

        // FIXME: finish checks

        unimplemented!()

        /*Ok(BufferImageCopyCommand {

        })*/
    }
}

error_ty!{BufferImageCopyError => "Error that can happen when copying between an image and \
                                   a buffer.",
    OutOfRange => "one of regions is out of range of the image or the buffer",
    WrongSamplesCount => "only images with one sample can be used for buffer transfers",
    WrongUsageFlag => "the buffer or image was not created with the correct usage flags",
    OverlappingRegions => "some regions are overlapping",
}
