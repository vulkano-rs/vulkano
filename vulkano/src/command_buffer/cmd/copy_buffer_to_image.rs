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
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;
use vk;

/// Command that copies from a buffer to an image.
#[derive(Debug, Clone)]
pub struct CmdCopyBufferToImage<S, D> {
    // The source buffer.
    buffer: S,
    // Raw source buffer.
    buffer_raw: vk::Buffer,
    // Offset in the source.
    buffer_offset: vk::DeviceSize,
    buffer_row_length: u32,
    buffer_image_height: u32,
    // The destination image.
    destination: D,
    // Raw destination image.
    destination_raw: vk::Image,
    // Layout of the destination image.
    destination_layout: vk::ImageLayout,
    // Offset in the destination.
    destination_offset: [i32; 3],
    destination_aspect_mask: vk::ImageAspectFlags,
    destination_mip_level: u32,
    destination_base_array_layer: u32,
    destination_layer_count: u32,
    // Size.
    extent: [u32; 3],
}

// TODO: add constructor

unsafe impl<S, D> DeviceOwned for CmdCopyBufferToImage<S, D> where S: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

unsafe impl<'a, P, S, D> AddCommand<&'a CmdCopyBufferToImage<S, D>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdCopyBufferToImage<S, D>) -> Self::Out {
        unsafe {
            debug_assert!(command.destination_layout == vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ||
                          command.destination_layout == vk::IMAGE_LAYOUT_GENERAL);

            let region = vk::BufferImageCopy {
                bufferOffset: command.buffer_offset,
                bufferRowLength: command.buffer_row_length,
                bufferImageHeight: command.buffer_image_height,
                imageSubresource: vk::ImageSubresourceLayers {
                    aspectMask: command.destination_aspect_mask,
                    mipLevel: command.destination_mip_level,
                    baseArrayLayer: command.destination_base_array_layer,
                    layerCount: command.destination_layer_count,
                },
                imageOffset: vk::Offset3D {
                    x: command.destination_offset[0],
                    y: command.destination_offset[1],
                    z: command.destination_offset[2],
                },
                imageExtent: vk::Extent3D {
                    width: command.extent[0],
                    height: command.extent[1],
                    depth: command.extent[2],
                },
            };

            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdCopyBufferToImage(cmd, command.buffer_raw, command.destination_raw,
                                    command.destination_layout, 1, &region as *const _);
        }

        self
    }
}

/// Error that can happen when creating a `CmdCopyBufferToImage`.
#[derive(Debug, Copy, Clone)]
pub enum CmdCopyBufferToImageError {
}

impl error::Error for CmdCopyBufferToImageError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
        }
    }
}

impl fmt::Display for CmdCopyBufferToImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
