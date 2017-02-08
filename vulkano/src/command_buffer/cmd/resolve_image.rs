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

/// Command that resolves a multisample image into a non-multisample one.
#[derive(Debug, Clone)]
pub struct CmdResolveImage<S, D> {
    // The source image.
    source: S,
    // Raw source image.
    source_raw: vk::Image,
    // Layout of the source image.
    source_layout: vk::ImageLayout,
    // Offset in the source.
    source_offset: [i32; 3],
    source_aspect_mask: vk::ImageAspectFlags,
    source_mip_level: u32,
    source_base_array_layer: u32,
    source_layer_count: u32,
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

unsafe impl<S, D> DeviceOwned for CmdResolveImage<S, D> where S: DeviceOwned {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.source.device()
    }
}

unsafe impl<'a, P, S, D> AddCommand<&'a CmdResolveImage<S, D>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdResolveImage<S, D>) -> Self::Out {
        unsafe {
            debug_assert!(command.source_layout == vk::IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ||
                          command.source_layout == vk::IMAGE_LAYOUT_GENERAL);
            debug_assert!(command.destination_layout == vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ||
                          command.destination_layout == vk::IMAGE_LAYOUT_GENERAL);

            let region = vk::ImageResolve {
                srcSubresource: vk::ImageSubresourceLayers {
                    aspectMask: command.source_aspect_mask,
                    mipLevel: command.source_mip_level,
                    baseArrayLayer: command.source_base_array_layer,
                    layerCount: command.source_layer_count,
                },
                srcOffset: vk::Offset3D {
                    x: command.source_offset[0],
                    y: command.source_offset[1],
                    z: command.source_offset[2],
                },
                dstSubresource: vk::ImageSubresourceLayers {
                    aspectMask: command.destination_aspect_mask,
                    mipLevel: command.destination_mip_level,
                    baseArrayLayer: command.destination_base_array_layer,
                    layerCount: command.destination_layer_count,
                },
                dstOffset: vk::Offset3D {
                    x: command.destination_offset[0],
                    y: command.destination_offset[1],
                    z: command.destination_offset[2],
                },
                extent: vk::Extent3D {
                    width: command.extent[0],
                    height: command.extent[1],
                    depth: command.extent[2],
                },
            };

            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdResolveImage(cmd, command.source_raw, command.source_layout,
                               command.destination_raw, command.destination_layout,
                               1, &region as *const _);
        }

        self
    }
}

/// Error that can happen when creating a `CmdResolveImage`.
#[derive(Debug, Copy, Clone)]
pub enum CmdResolveImageError {
}

impl error::Error for CmdResolveImageError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
        }
    }
}

impl fmt::Display for CmdResolveImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
