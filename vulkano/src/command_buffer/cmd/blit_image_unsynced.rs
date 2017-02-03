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
use smallvec::SmallVec;

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::DeviceOwned;
use format::FormatTy;
use image::Image;
use image::Layout;
use sampler::Filter;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds at the end of it a command that blits from an image to
/// another.
pub struct CmdBlitImageUnsynced<L, S, D>
    where L: CommandsList
{
    // Parent commands list.
    previous: L,
    source: S,
    source_raw: vk::Image,
    source_layout: vk::ImageLayout,
    destination: D,
    destination_raw: vk::Image,
    destination_layout: vk::ImageLayout,
    filter: vk::Filter,
    regions: SmallVec<[vk::ImageBlit; 1]>,
}

impl<L, S, D> CmdBlitImageUnsynced<L, S, D>
    where L: CommandsList, S: Image, D: Image
{
    pub unsafe fn new<I>(previous: L, source: S, source_layout: Layout,
                         destination: D, destination_layout: Layout, filter: Filter,
                         regions: I)
                         -> Result<CmdBlitImageUnsynced<L, S, D>, CmdBlitImageUnsyncedError>
        where I: IntoIterator<Item = BlitRegion>
    {
        // FIXME: check that we're outside a render pass

        if !source.inner().usage_transfer_src() {
            return Err(CmdBlitImageUnsyncedError::SourceUsageMissingSrcTransfer);
        }

        if !destination.inner().usage_transfer_dest() {
            return Err(CmdBlitImageUnsyncedError::DestinationUsageMissingDestTransfer);
        }

        if !source.inner().supports_blit_source() {
            return Err(CmdBlitImageUnsyncedError::SourceFormatDoesntSupportBlits);
        }

        if !destination.inner().supports_blit_destination() {
            return Err(CmdBlitImageUnsyncedError::DestinationFormatDoesntSupportBlits);
        }

        match (source.inner().format().ty(), destination.inner().format().ty()) {
            (FormatTy::Sint, FormatTy::Sint) => (),
            (FormatTy::Sint, _) | (_, FormatTy::Sint) => {
                return Err(CmdBlitImageUnsyncedError::IncompatibleSourceDestinationFormats);
            },
            (FormatTy::Uint, FormatTy::Uint) => (),
            (FormatTy::Uint, _) | (_, FormatTy::Uint) => {
                return Err(CmdBlitImageUnsyncedError::IncompatibleSourceDestinationFormats);
            },
            (FormatTy::Depth, _) | (FormatTy::Stencil, _) | (FormatTy::DepthStencil, _) => {
                if source.inner().format() != destination.inner().format() {
                    return Err(CmdBlitImageUnsyncedError::IncompatibleSourceDestinationFormats);
                }
            },
            _ => ()
        }

        if source.inner().samples() != 1 {
            return Err(CmdBlitImageUnsyncedError::SourceMultisampled);
        }

        if destination.inner().samples() != 1 {
            return Err(CmdBlitImageUnsyncedError::DestinationMultisampled);
        }

        if filter == Filter::Linear && !source.inner().supports_linear_filtering() {
            return Err(CmdBlitImageUnsyncedError::SourceDoesntSupportLinearFilter);
        }

        let raw_regions = {
            let mut raw_regions = SmallVec::new();
            for region in regions {
                // FIXME: check dimensions
                // FIXME: check overlap and document how it works

                raw_regions.push(vk::ImageBlit {
                    srcOffsets: [
                        vk::Offset3D {
                            x: region.source_offset[0] as i32,
                            y: region.source_offset[1] as i32,
                            z: region.source_offset[2] as i32,
                        },
                        vk::Offset3D {
                            x: region.source_offset[0] as i32 + region.source_size[0],
                            y: region.source_offset[1] as i32 + region.source_size[1],
                            z: region.source_offset[2] as i32 + region.source_size[2],
                        },
                    ],
                    srcSubresource: vk::ImageSubresourceLayers {
                        aspectMask: region.aspect as u32,
                        mipLevel: region.source_mipmap,
                        baseArrayLayer: region.source_array_layer,
                        layerCount: region.num_arrays_layers,
                    },
                    dstOffsets: [
                        vk::Offset3D {
                            x: region.destination_offset[0] as i32,
                            y: region.destination_offset[1] as i32,
                            z: region.destination_offset[2] as i32,
                        },
                        vk::Offset3D {
                            x: region.destination_offset[0] as i32 + region.destination_size[0],
                            y: region.destination_offset[1] as i32 + region.destination_size[1],
                            z: region.destination_offset[2] as i32 + region.destination_size[2],
                        },
                    ],
                    dstSubresource: vk::ImageSubresourceLayers {
                        aspectMask: region.aspect as u32,
                        mipLevel: region.destination_mipmap,
                        baseArrayLayer: region.destination_array_layer,
                        layerCount: region.num_arrays_layers,
                    },
                });
            }
            raw_regions
        };

        assert_eq!(source.inner().device().internal_object(),
                   destination.inner().device().internal_object());
        debug_assert!(source_layout == Layout::TransferSrcOptimal ||
                      source_layout == Layout::General);
        debug_assert!(destination_layout == Layout::TransferDstOptimal ||
                      destination_layout == Layout::General);

        let source_raw = source.inner().internal_object();
        let destination_raw = destination.inner().internal_object();

        Ok(CmdBlitImageUnsynced {
            previous: previous,
            source: source,
            source_raw: source_raw,
            source_layout: source_layout as u32,
            destination: destination,
            destination_raw: destination_raw,
            destination_layout: destination_layout as u32,
            filter: filter as vk::Filter,
            regions: raw_regions,
        })
    }
}

impl<L, S, D> CmdBlitImageUnsynced<L, S, D>
    where L: CommandsList, S: Image, D: Image
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        assert_eq!(self.source.inner().device().internal_object(),
                   builder.device().internal_object());

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();
                vk.CmdBlitImage(cmd, self.source_raw, self.source_layout, self.destination_raw,
                                self.destination_layout, self.regions.len() as u32,
                                self.regions.as_ptr(), self.filter);
            }
        }));
    }
}

/// A region of a blit operation.
#[derive(Debug, Copy, Clone)]
pub struct BlitRegion {
    /// Which aspect of the source and destination to blit.
    pub aspect: BlitRegionAspect,
    /// The mipmap of the source image to copy. You can only blit one mipmap at a time.
    pub source_mipmap: u32,
    /// 3D coordinates of the starting pixel in the source image.
    pub source_offset: [u32; 3],
    /// Size of the source area. Can be negative if you want to flip the area.
    pub source_size: [i32; 3],
    /// First array layer of the source image. 
    pub source_array_layer: u32,
    /// The mipmap of the destination image to copy. You can only blit one mipmap at a time.
    pub destination_mipmap: u32,
    /// 3D coordinates of the starting pixel in the destination image.
    pub destination_offset: [u32; 3],
    /// Size of the destination area. Can be negative if you want to flip the area.
    pub destination_size: [i32; 3],
    /// First array layer of the destination image. 
    pub destination_array_layer: u32,
    /// Number of array layers to blit. Each array layer is blit one by one.
    pub num_arrays_layers: u32,
}

// TODO: maybe we can merge this type with other similar types?
#[derive(Debug, Copy, Clone)]
#[repr(u32)]
pub enum BlitRegionAspect {
    Color = vk::IMAGE_ASPECT_COLOR_BIT,
    Depth = vk::IMAGE_ASPECT_DEPTH_BIT,
    Stencil = vk::IMAGE_ASPECT_STENCIL_BIT,
    DepthStencil = vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT,
}

/// Error that can happen when creating a `CmdBlitImageUnsynced`.
#[derive(Debug, Copy, Clone)]
pub enum CmdBlitImageUnsyncedError {
    /// The source image wasn't created with source transfer operations support.
    SourceUsageMissingSrcTransfer,
    /// The destination image wasn't created with destination transfer operations support.
    DestinationUsageMissingDestTransfer,
    /// The format of the source image doesn't support blit operations.
    SourceFormatDoesntSupportBlits,
    /// The format of the destination image doesn't support blit operations.
    DestinationFormatDoesntSupportBlits,
    /// The format of the source and the destination aren't compatible for blit operations.
    // TODO: link to documentation with these rules: 
    // If either of srcImage or dstImage was created with a signed integer VkFormat, the other must also have been created with a signed integer VkFormat
    // If either of srcImage or dstImage was created with an unsigned integer VkFormat, the other must also have been created with an unsigned integer VkFormat
    // If either of srcImage or dstImage was created with a depth/stencil format, the other must have exactly the same format 
    IncompatibleSourceDestinationFormats,
    /// The source image must only have one sample.
    SourceMultisampled,
    /// The destination image must only have one sample.
    DestinationMultisampled,
    /// The filter is `Linear` but the source format doesn't support linear filtering.
    SourceDoesntSupportLinearFilter,
}

impl error::Error for CmdBlitImageUnsyncedError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdBlitImageUnsyncedError::SourceUsageMissingSrcTransfer => {
                "the source image wasn't created with source transfer operations support"
            },
            CmdBlitImageUnsyncedError::DestinationUsageMissingDestTransfer => {
                "the destination image wasn't created with destination transfer operations support"
            },
            CmdBlitImageUnsyncedError::SourceFormatDoesntSupportBlits => {
                "the format of the source image doesn't support blit operations"
            },
            CmdBlitImageUnsyncedError::DestinationFormatDoesntSupportBlits => {
                "the format of the destination image doesn't support blit operations"
            },
            CmdBlitImageUnsyncedError::IncompatibleSourceDestinationFormats => {
                "the format of the source and the destination aren't compatible for blit operations"
            },
            CmdBlitImageUnsyncedError::SourceMultisampled => {
                "the source image must only have one sample"
            },
            CmdBlitImageUnsyncedError::DestinationMultisampled => {
                "the destination image must only have one sample"
            },
            CmdBlitImageUnsyncedError::SourceDoesntSupportLinearFilter => {
                "the filter is `Linear` but the source format doesn't support linear filtering"
            },
        }
    }
}

impl fmt::Display for CmdBlitImageUnsyncedError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
