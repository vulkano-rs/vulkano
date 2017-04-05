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
use buffer::BufferAccess;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use image::ImageAccess;
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

impl<S, D> CmdCopyBufferToImage<S, D> where S: BufferAccess, D: ImageAccess {
    #[inline]
    pub fn new(source: S, destination: D)
               -> Result<CmdCopyBufferToImage<S, D>, CmdCopyBufferToImageError>
    {
        let dims = destination.dimensions().width_height_depth();
        CmdCopyBufferToImage::with_dimensions(source, destination, [0, 0, 0], dims, 0, 1, 0)
    }

    pub fn with_dimensions(source: S, destination: D, offset: [u32; 3], size: [u32; 3],
                           first_layer: u32, num_layers: u32, mipmap: u32)
                           -> Result<CmdCopyBufferToImage<S, D>, CmdCopyBufferToImageError>
    {
        // FIXME: check buffer content format
        // FIXME: check that the buffer is large enough
        // FIXME: check image dimensions

        assert_eq!(source.inner().buffer.device().internal_object(),
                   destination.inner().device().internal_object());

        let (source_raw, src_offset) = {
            let inner = source.inner();
            if !inner.buffer.usage_transfer_src() {
                return Err(CmdCopyBufferToImageError::SourceMissingTransferUsage);
            }
            (inner.buffer.internal_object(), inner.offset)
        };

        if destination.samples() != 1 {
            return Err(CmdCopyBufferToImageError::DestinationMultisampled);
        }

        let destination_raw = {
            let inner = destination.inner();
            if !inner.usage_transfer_dest() {
                return Err(CmdCopyBufferToImageError::DestinationMissingTransferUsage);
            }
            inner.internal_object()
        };

        if source.conflicts_image(0, source.size(), &destination, first_layer, num_layers,
                                  mipmap, 1)
        {
            return Err(CmdCopyBufferToImageError::OverlappingRanges);
        } else {
            debug_assert!(!destination.conflicts_buffer(first_layer, num_layers, mipmap,
                                                        1, &source, 0, source.size()));
        }

        let aspect_mask = if destination.has_color() {
            vk::IMAGE_ASPECT_COLOR_BIT
        } else {
            unimplemented!()        // TODO:
        };

        Ok(CmdCopyBufferToImage {
            buffer: source,
            buffer_raw: source_raw,
            buffer_offset: src_offset as vk::DeviceSize,
            buffer_row_length: 0,
            buffer_image_height: 0,
            destination: destination,
            destination_raw: destination_raw,
            destination_layout: vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,      // FIXME:
            destination_offset: [offset[0] as i32, offset[1] as i32, offset[2] as i32],
            destination_aspect_mask: aspect_mask,
            destination_mip_level: mipmap,
            destination_base_array_layer: first_layer,
            destination_layer_count: num_layers,
            extent: size,
        })
    }
}

impl<S, D> CmdCopyBufferToImage<S, D> {
    /// Returns the source buffer.
    #[inline]
    pub fn source(&self) -> &S {
        &self.buffer
    }

    /// Returns the destination image.
    #[inline]
    pub fn destination(&self) -> &D {
        &self.destination
    }
}

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
    /// The source buffer is missing the transfer source usage.
    SourceMissingTransferUsage,
    /// The destination image is missing the transfer destination usage.
    DestinationMissingTransferUsage,
    /// The destination image has more than one sample per pixel.
    DestinationMultisampled,
    /// The dimensions are out of range of the image.
    OutOfImageRange,
    /// The source and destination are overlapping in memory.
    OverlappingRanges,
}

impl error::Error for CmdCopyBufferToImageError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdCopyBufferToImageError::SourceMissingTransferUsage => {
                "the source buffer is missing the transfer source usage"
            },
            CmdCopyBufferToImageError::DestinationMissingTransferUsage => {
                "the destination image is missing the transfer destination usage"
            },
            CmdCopyBufferToImageError::DestinationMultisampled => {
                "the destination image has more than one sample per pixel"
            },
            CmdCopyBufferToImageError::OutOfImageRange => {
                "the dimensions are out of range of the image"
            },
            CmdCopyBufferToImageError::OverlappingRanges => {
                "the source and destination are overlapping in memory"
            },
        }
    }
}

impl fmt::Display for CmdCopyBufferToImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
