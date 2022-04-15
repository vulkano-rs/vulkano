// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::{BufferAccess, BufferContents, TypedBufferAccess},
    command_buffer::{
        synced::{Command, Resource, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, CopyError, CopyErrorResource,
    },
    device::DeviceOwned,
    format::Format,
    image::{
        ImageAccess, ImageAspects, ImageLayout, ImageSubresourceLayers, ImageType, SampleCount,
        SampleCounts,
    },
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStages},
    DeviceSize, SafeDeref, Version, VulkanObject,
};
use smallvec::{smallvec, SmallVec};
use std::{
    cmp::{max, min},
    mem::{size_of, size_of_val},
    sync::Arc,
};

/// # Commands to transfer data to a resource, either from the host or from another resource.
///
/// These commands can be called on a transfer queue, in addition to a compute or graphics queue.
impl<L, P> AutoCommandBufferBuilder<L, P> {
    /// Copies data from a buffer to another buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `src_buffer` or `dst_buffer` were not created from the same device
    ///   as `self`.
    #[inline]
    pub fn copy_buffer(
        &mut self,
        copy_buffer_info: impl Into<CopyBufferInfo>,
    ) -> Result<&mut Self, CopyError> {
        let mut copy_buffer_info = copy_buffer_info.into();
        self.validate_copy_buffer(&mut copy_buffer_info)?;

        unsafe {
            self.inner.copy_buffer(copy_buffer_info)?;
        }

        Ok(self)
    }

    fn validate_copy_buffer(&self, copy_buffer_info: &mut CopyBufferInfo) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdCopyBuffer2-renderpass
        if self.render_pass_state.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdCopyBuffer2-commandBuffer-cmdpool
        if !(self.queue_family().explicitly_supports_transfers()
            || self.queue_family().supports_graphics()
            || self.queue_family().supports_compute())
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let CopyBufferInfo {
            src_buffer,
            dst_buffer,
            regions,
            _ne: _,
        } = copy_buffer_info;

        let src_buffer_inner = src_buffer.inner();
        let dst_buffer_inner = dst_buffer.inner();

        // VUID-VkCopyBufferInfo2-commonparent
        assert_eq!(device, src_buffer.device());
        assert_eq!(device, dst_buffer.device());

        // VUID-VkCopyBufferInfo2-srcBuffer-00118
        if !src_buffer.usage().transfer_src {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkCopyBufferInfo2-dstBuffer-00120
        if !dst_buffer.usage().transfer_dst {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        let same_buffer = src_buffer_inner.buffer == dst_buffer_inner.buffer;
        let mut overlap_indices = None;

        for (region_index, region) in regions.iter().enumerate() {
            let &BufferCopy {
                src_offset,
                dst_offset,
                size,
                _ne: _,
            } = region;

            // VUID-VkBufferCopy2-size-01988
            assert!(size != 0);

            // VUID-VkCopyBufferInfo2-srcOffset-00113
            // VUID-VkCopyBufferInfo2-size-00115
            if src_offset + size > src_buffer.size() {
                return Err(CopyError::RegionOutOfBufferBounds {
                    resource: CopyErrorResource::Source,
                    region_index,
                    offset_range_end: src_offset + size,
                    buffer_size: src_buffer.size(),
                });
            }

            // VUID-VkCopyBufferInfo2-dstOffset-00114
            // VUID-VkCopyBufferInfo2-size-00116
            if dst_offset + size > dst_buffer.size() {
                return Err(CopyError::RegionOutOfBufferBounds {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    offset_range_end: dst_offset + size,
                    buffer_size: dst_buffer.size(),
                });
            }

            // VUID-VkCopyBufferInfo2-pRegions-00117
            if same_buffer {
                let src_region_index = region_index;
                let src_range = src_buffer_inner.offset + src_offset
                    ..src_buffer_inner.offset + src_offset + size;

                for (dst_region_index, dst_region) in regions.iter().enumerate() {
                    let &BufferCopy { dst_offset, .. } = dst_region;

                    let dst_range = dst_buffer_inner.offset + dst_offset
                        ..dst_buffer_inner.offset + dst_offset + size;

                    if src_range.start >= dst_range.end || dst_range.start >= src_range.end {
                        // The regions do not overlap
                        continue;
                    }

                    overlap_indices = Some((src_region_index, dst_region_index));
                }
            }
        }

        // VUID-VkCopyBufferInfo2-pRegions-00117
        if let Some((src_region_index, dst_region_index)) = overlap_indices {
            return Err(CopyError::OverlappingRegions {
                src_region_index,
                dst_region_index,
            });
        }

        Ok(())
    }

    /// Copies data from an image to another image.
    ///
    /// There are several restrictions:
    ///
    /// - The number of samples in the source and destination images must be equal.
    /// - The size of the uncompressed element format of the source image must be equal to the
    ///   compressed element format of the destination.
    /// - If you copy between depth, stencil or depth-stencil images, the format of both images
    ///   must match exactly.
    /// - For two-dimensional images, the Z coordinate must be 0 for the image offsets and 1 for
    ///   the extent. Same for the Y coordinate for one-dimensional images.
    /// - For non-array images, the base array layer must be 0 and the number of layers must be 1.
    ///
    /// If `layer_count` is greater than 1, the copy will happen between each individual layer as
    /// if they were separate images.
    ///
    /// # Panics
    ///
    /// - Panics if `src_image` or `dst_image` were not created from the same device
    ///   as `self`.
    pub fn copy_image(
        &mut self,
        mut copy_image_info: CopyImageInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_copy_image(&mut copy_image_info)?;

        unsafe {
            self.inner.copy_image(copy_image_info)?;
        }

        Ok(self)
    }

    fn validate_copy_image(&self, copy_image_info: &mut CopyImageInfo) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdCopyImage2-renderpass
        if self.render_pass_state.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdCopyImage2-commandBuffer-cmdpool
        if !(self.queue_family().explicitly_supports_transfers()
            || self.queue_family().supports_graphics()
            || self.queue_family().supports_compute())
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &mut CopyImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = copy_image_info;

        // VUID-VkCopyImageInfo2-commonparent
        assert_eq!(device, src_image.device());
        assert_eq!(device, dst_image.device());

        let copy_2d_3d_supported =
            device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1;
        let src_image_inner = src_image.inner();
        let dst_image_inner = dst_image.inner();
        let mut src_image_aspects = src_image.format().aspects();
        let mut dst_image_aspects = dst_image.format().aspects();

        // VUID-VkCopyImageInfo2-aspect-06662
        if !src_image.usage().transfer_src {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkCopyImageInfo2-aspect-06663
        if !dst_image.usage().transfer_dst {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            // VUID-VkCopyImageInfo2-srcImage-01995
            if !src_image.format_features().transfer_src {
                return Err(CopyError::MissingFormatFeature {
                    resource: CopyErrorResource::Source,
                    format_feature: "transfer_src",
                });
            }

            // VUID-VkCopyImageInfo2-dstImage-01996
            if !dst_image.format_features().transfer_dst {
                return Err(CopyError::MissingFormatFeature {
                    resource: CopyErrorResource::Destination,
                    format_feature: "transfer_dst",
                });
            }
        }

        // VUID-VkCopyImageInfo2-srcImage-00136
        if src_image.samples() != dst_image.samples() {
            return Err(CopyError::SampleCountMismatch {
                src_sample_count: src_image.samples(),
                dst_sample_count: dst_image.samples(),
            });
        }

        if !(src_image_aspects.color || dst_image_aspects.color) {
            // VUID-VkCopyImageInfo2-srcImage-01548
            if src_image.format() != dst_image.format() {
                return Err(CopyError::FormatsMismatch {
                    src_format: src_image.format(),
                    dst_format: dst_image.format(),
                });
            }
        }

        // VUID-VkCopyImageInfo2-srcImageLayout-01917
        if !matches!(
            src_image_layout,
            ImageLayout::TransferSrcOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Source,
                image_layout: src_image_layout,
            });
        }

        // VUID-VkCopyImageInfo2-dstImageLayout-01395
        if !matches!(
            dst_image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Destination,
                image_layout: dst_image_layout,
            });
        }

        let extent_alignment = match self.queue_family().min_image_transfer_granularity() {
            [0, 0, 0] => None,
            min_image_transfer_granularity => {
                let granularity = move |block_extent: [u32; 3], is_multi_plane: bool| {
                    if is_multi_plane {
                        // Assume planes always have 1x1 blocks
                        min_image_transfer_granularity
                    } else {
                        // "The value returned in minImageTransferGranularity has a unit of
                        // compressed texel blocks for images having a block-compressed format, and
                        // a unit of texels otherwise."
                        [
                            min_image_transfer_granularity[0] * block_extent[0],
                            min_image_transfer_granularity[1] * block_extent[1],
                            min_image_transfer_granularity[2] * block_extent[2],
                        ]
                    }
                };

                Some((
                    granularity(src_image.format().block_extent(), src_image_aspects.plane0),
                    granularity(dst_image.format().block_extent(), dst_image_aspects.plane0),
                ))
            }
        };

        if src_image_aspects.plane0 {
            // VUID-VkCopyImageInfo2-srcImage-01552
            // VUID-VkCopyImageInfo2-srcImage-01553
            src_image_aspects.color = false;
        }

        if dst_image_aspects.plane0 {
            // VUID-VkCopyImageInfo2-dstImage-01554
            // VUID-VkCopyImageInfo2-dstImage-01555
            dst_image_aspects.color = false;
        }

        let same_image = src_image_inner.image == dst_image_inner.image;
        let mut overlap_subresource_indices = None;
        let mut overlap_extent_indices = None;

        for (region_index, region) in regions.iter().enumerate() {
            let &ImageCopy {
                ref src_subresource,
                src_offset,
                ref dst_subresource,
                dst_offset,
                extent,
                _ne,
            } = region;

            let check_subresource = |resource: CopyErrorResource,
                                     image: &dyn ImageAccess,
                                     image_aspects: &ImageAspects,
                                     subresource: &ImageSubresourceLayers|
             -> Result<_, CopyError> {
                // VUID-VkCopyImageInfo2-srcSubresource-01696
                // VUID-VkCopyImageInfo2-dstSubresource-01697
                if subresource.mip_level >= image.mip_levels() {
                    return Err(CopyError::MipLevelsOutOfRange {
                        resource,
                        region_index,
                        mip_levels_range_end: subresource.mip_level + 1,
                        image_mip_levels: image.mip_levels(),
                    });
                }

                // VUID-VkImageSubresourceLayers-layerCount-01700
                assert!(!subresource.array_layers.is_empty());

                // VUID-VkCopyImageInfo2-srcSubresource-01698
                // VUID-VkCopyImageInfo2-dstSubresource-01699
                // VUID-VkCopyImageInfo2-srcImage-04443
                // VUID-VkCopyImageInfo2-dstImage-04444
                if subresource.array_layers.end > image.dimensions().array_layers() {
                    return Err(CopyError::ArrayLayersOutOfRange {
                        resource,
                        region_index,
                        array_layers_range_end: subresource.array_layers.end,
                        image_array_layers: image.dimensions().array_layers(),
                    });
                }

                // VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask
                assert!(subresource.aspects != ImageAspects::none());

                // VUID-VkCopyImageInfo2-aspectMask-00142
                // VUID-VkCopyImageInfo2-aspectMask-00143
                if !image_aspects.contains(&subresource.aspects) {
                    return Err(CopyError::AspectsNotAllowed {
                        resource,
                        region_index,
                        aspects: subresource.aspects,
                        allowed_aspects: *image_aspects,
                    });
                }

                let (subresource_format, subresource_extent) = if image_aspects.plane0 {
                    // VUID-VkCopyImageInfo2-srcImage-01552
                    // VUID-VkCopyImageInfo2-srcImage-01553
                    // VUID-VkCopyImageInfo2-dstImage-01554
                    // VUID-VkCopyImageInfo2-dstImage-01555
                    if subresource.aspects.iter().count() != 1 {
                        return Err(CopyError::MultipleAspectsNotAllowed {
                            resource,
                            region_index,
                            aspects: subresource.aspects,
                        });
                    }

                    if subresource.aspects.plane0 {
                        (
                            image.format().planes()[0],
                            image.dimensions().width_height_depth(),
                        )
                    } else if subresource.aspects.plane1 {
                        (
                            image.format().planes()[1],
                            image
                                .format()
                                .ycbcr_chroma_sampling()
                                .unwrap()
                                .subsampled_extent(image.dimensions().width_height_depth()),
                        )
                    } else {
                        (
                            image.format().planes()[2],
                            image
                                .format()
                                .ycbcr_chroma_sampling()
                                .unwrap()
                                .subsampled_extent(image.dimensions().width_height_depth()),
                        )
                    }
                } else {
                    (
                        image.format(),
                        image
                            .dimensions()
                            .mip_level_dimensions(subresource.mip_level)
                            .unwrap()
                            .width_height_depth(),
                    )
                };

                Ok((subresource_format, subresource_extent))
            };

            let (src_subresource_format, src_subresource_extent) = check_subresource(
                CopyErrorResource::Source,
                src_image,
                &src_image_aspects,
                src_subresource,
            )?;
            let (dst_subresource_format, dst_subresource_extent) = check_subresource(
                CopyErrorResource::Destination,
                dst_image,
                &dst_image_aspects,
                dst_subresource,
            )?;

            if !(src_image_aspects.plane0 || dst_image_aspects.plane0) {
                // VUID-VkCopyImageInfo2-srcImage-01551
                if src_subresource.aspects != dst_subresource.aspects {
                    return Err(CopyError::AspectsMismatch {
                        region_index,
                        src_aspects: src_subresource.aspects,
                        dst_aspects: dst_subresource.aspects,
                    });
                }
            }

            // VUID-VkCopyImageInfo2-srcImage-01548
            // VUID-VkCopyImageInfo2-None-01549
            // Color formats must be size-compatible.
            if src_subresource_format.block_size() != dst_subresource_format.block_size() {
                return Err(CopyError::FormatsNotCompatible {
                    src_format: src_subresource_format,
                    dst_format: dst_subresource_format,
                });
            }

            // TODO:
            // "When copying between compressed and uncompressed formats the extent members
            // represent the texel dimensions of the source image and not the destination."
            let mut src_extent = extent;
            let mut dst_extent = extent;
            let src_layer_count =
                src_subresource.array_layers.end - src_subresource.array_layers.start;
            let dst_layer_count =
                dst_subresource.array_layers.end - dst_subresource.array_layers.start;

            if copy_2d_3d_supported {
                match (
                    src_image.dimensions().image_type(),
                    dst_image.dimensions().image_type(),
                ) {
                    (ImageType::Dim2d, ImageType::Dim3d) => {
                        src_extent[2] = 1;

                        // VUID-vkCmdCopyImage-srcImage-01791
                        if dst_extent[2] != src_layer_count {
                            return Err(CopyError::ArrayLayerCountMismatch {
                                region_index,
                                src_layer_count,
                                dst_layer_count: dst_extent[2],
                            });
                        }
                    }
                    (ImageType::Dim3d, ImageType::Dim2d) => {
                        dst_extent[2] = 1;

                        // VUID-vkCmdCopyImage-dstImage-01792
                        if src_extent[2] != dst_layer_count {
                            return Err(CopyError::ArrayLayerCountMismatch {
                                region_index,
                                src_layer_count: src_extent[2],
                                dst_layer_count,
                            });
                        }
                    }
                    _ => {
                        // VUID-VkImageCopy2-extent-00140
                        if src_layer_count != dst_layer_count {
                            return Err(CopyError::ArrayLayerCountMismatch {
                                region_index,
                                src_layer_count,
                                dst_layer_count,
                            });
                        }
                    }
                }
            } else {
                // VUID-VkImageCopy2-extent-00140
                if src_layer_count != dst_layer_count {
                    return Err(CopyError::ArrayLayerCountMismatch {
                        region_index,
                        src_layer_count,
                        dst_layer_count,
                    });
                }
            };

            if let Some((src_extent_alignment, dst_extent_alignment)) = extent_alignment {
                let check_offset_extent = |resource: CopyErrorResource,
                                           extent_alignment: [u32; 3],
                                           subresource_extent: [u32; 3],
                                           offset: [u32; 3],
                                           extent: [u32; 3]|
                 -> Result<_, CopyError> {
                    for i in 0..3 {
                        // VUID-VkImageCopy2-extent-06668
                        // VUID-VkImageCopy2-extent-06669
                        // VUID-VkImageCopy2-extent-06670
                        assert!(extent[i] != 0);

                        // VUID-VkCopyImageInfo2-srcOffset-00144
                        // VUID-VkCopyImageInfo2-srcOffset-00145
                        // VUID-VkCopyImageInfo2-srcOffset-00147
                        // VUID-VkCopyImageInfo2-dstOffset-00150
                        // VUID-VkCopyImageInfo2-dstOffset-00151
                        // VUID-VkCopyImageInfo2-dstOffset-00153
                        if offset[i] + extent[i] > subresource_extent[i] {
                            return Err(CopyError::RegionOutOfImageBounds {
                                resource,
                                region_index,
                                offset_range_end: [
                                    offset[0] + extent[0],
                                    offset[1] + extent[1],
                                    offset[2] + extent[2],
                                ],
                                subresource_extent,
                            });
                        }

                        // VUID-VkCopyImageInfo2-srcImage-01727
                        // VUID-VkCopyImageInfo2-dstImage-01731
                        // VUID-VkCopyImageInfo2-srcOffset-01783
                        // VUID-VkCopyImageInfo2-dstOffset-01784
                        if offset[i] % extent_alignment[i] != 0 {
                            return Err(CopyError::OffsetNotAlignedForImage {
                                resource,
                                region_index,
                                offset,
                                required_alignment: extent_alignment,
                            });
                        }

                        // VUID-VkCopyImageInfo2-srcImage-01728
                        // VUID-VkCopyImageInfo2-srcImage-01729
                        // VUID-VkCopyImageInfo2-srcImage-01730
                        // VUID-VkCopyImageInfo2-dstImage-01732
                        // VUID-VkCopyImageInfo2-dstImage-01733
                        // VUID-VkCopyImageInfo2-dstImage-01734
                        if offset[i] + extent[i] != subresource_extent[i]
                            && extent[i] % extent_alignment[i] != 0
                        {
                            return Err(CopyError::ExtentNotAlignedForImage {
                                resource,
                                region_index,
                                extent,
                                required_alignment: extent_alignment,
                            });
                        }
                    }

                    Ok(())
                };

                check_offset_extent(
                    CopyErrorResource::Source,
                    src_extent_alignment,
                    src_subresource_extent,
                    src_offset,
                    src_extent,
                )?;
                check_offset_extent(
                    CopyErrorResource::Destination,
                    dst_extent_alignment,
                    dst_subresource_extent,
                    dst_offset,
                    dst_extent,
                )?;

                // VUID-VkCopyImageInfo2-pRegions-00124
                if same_image {
                    let src_region_index = region_index;
                    let src_subresource_axes = [
                        src_image_inner.first_mipmap_level + src_subresource.mip_level
                            ..src_image_inner.first_mipmap_level + src_subresource.mip_level + 1,
                        src_image_inner.first_layer + src_subresource.array_layers.start
                            ..src_image_inner.first_layer + src_subresource.array_layers.end,
                    ];
                    let src_extent_axes = [
                        src_offset[0]..src_offset[0] + extent[0],
                        src_offset[1]..src_offset[1] + extent[1],
                        src_offset[2]..src_offset[2] + extent[2],
                    ];

                    for (dst_region_index, dst_region) in regions.iter().enumerate() {
                        let &ImageCopy {
                            ref dst_subresource,
                            dst_offset,
                            ..
                        } = dst_region;

                        // For a single-plane image, the aspects must always be identical anyway
                        if src_image_aspects.plane0
                            && src_subresource.aspects != dst_subresource.aspects
                        {
                            continue;
                        }

                        let dst_subresource_axes = [
                            dst_image_inner.first_mipmap_level + dst_subresource.mip_level
                                ..dst_image_inner.first_mipmap_level
                                    + dst_subresource.mip_level
                                    + 1,
                            dst_image_inner.first_layer + src_subresource.array_layers.start
                                ..dst_image_inner.first_layer + src_subresource.array_layers.end,
                        ];

                        if src_subresource_axes.iter().zip(dst_subresource_axes).any(
                            |(src_range, dst_range)| {
                                src_range.start >= dst_range.end || dst_range.start >= src_range.end
                            },
                        ) {
                            continue;
                        }

                        // If the subresource axes all overlap, then the source and destination must
                        // have the same layout.
                        overlap_subresource_indices = Some((src_region_index, dst_region_index));

                        let dst_extent_axes = [
                            dst_offset[0]..dst_offset[0] + extent[0],
                            dst_offset[1]..dst_offset[1] + extent[1],
                            dst_offset[2]..dst_offset[2] + extent[2],
                        ];

                        // There is only overlap if all of the axes overlap.
                        if src_extent_axes.iter().zip(dst_extent_axes).any(
                            |(src_range, dst_range)| {
                                src_range.start >= dst_range.end || dst_range.start >= src_range.end
                            },
                        ) {
                            continue;
                        }

                        overlap_extent_indices = Some((src_region_index, dst_region_index));
                    }
                }
            } else {
                // If granularity is `None`, then we can only copy whole subresources.
                let check_offset_extent = |resource: CopyErrorResource,
                                           subresource_extent: [u32; 3],
                                           offset: [u32; 3],
                                           extent: [u32; 3]|
                 -> Result<_, CopyError> {
                    // VUID-VkCopyImageInfo2-srcImage-01727
                    // VUID-VkCopyImageInfo2-dstImage-01731
                    // VUID-vkCmdCopyImage-srcOffset-01783
                    // VUID-vkCmdCopyImage-dstOffset-01784
                    if offset != [0, 0, 0] {
                        return Err(CopyError::OffsetNotAlignedForImage {
                            resource,
                            region_index,
                            offset,
                            required_alignment: subresource_extent,
                        });
                    }

                    // VUID-VkCopyImageInfo2-srcImage-01728
                    // VUID-VkCopyImageInfo2-srcImage-01729
                    // VUID-VkCopyImageInfo2-srcImage-01730
                    // VUID-VkCopyImageInfo2-dstImage-01732
                    // VUID-VkCopyImageInfo2-dstImage-01733
                    // VUID-VkCopyImageInfo2-dstImage-01734
                    if extent != subresource_extent {
                        return Err(CopyError::ExtentNotAlignedForImage {
                            resource,
                            region_index,
                            extent,
                            required_alignment: subresource_extent,
                        });
                    }

                    Ok(())
                };

                check_offset_extent(
                    CopyErrorResource::Source,
                    src_subresource_extent,
                    src_offset,
                    src_extent,
                )?;
                check_offset_extent(
                    CopyErrorResource::Destination,
                    dst_subresource_extent,
                    dst_offset,
                    dst_extent,
                )?;

                // VUID-VkCopyImageInfo2-pRegions-00124
                // A simpler version that assumes the region covers the full extent.
                if same_image {
                    let src_region_index = region_index;
                    let src_axes = [
                        src_image_inner.first_mipmap_level + src_subresource.mip_level
                            ..src_image_inner.first_mipmap_level + src_subresource.mip_level + 1,
                        src_image_inner.first_layer + src_subresource.array_layers.start
                            ..src_image_inner.first_layer + src_subresource.array_layers.end,
                    ];

                    for (dst_region_index, dst_region) in regions.iter().enumerate() {
                        let &ImageCopy {
                            ref dst_subresource,
                            dst_offset,
                            ..
                        } = dst_region;

                        if src_image_aspects.plane0
                            && src_subresource.aspects != dst_subresource.aspects
                        {
                            continue;
                        }

                        let dst_axes = [
                            dst_image_inner.first_mipmap_level + dst_subresource.mip_level
                                ..dst_image_inner.first_mipmap_level
                                    + dst_subresource.mip_level
                                    + 1,
                            dst_image_inner.first_layer + src_subresource.array_layers.start
                                ..dst_image_inner.first_layer + src_subresource.array_layers.end,
                        ];

                        // There is only overlap if all of the axes overlap.
                        if src_axes.iter().zip(dst_axes).any(|(src_range, dst_range)| {
                            src_range.start >= dst_range.end || dst_range.start >= src_range.end
                        }) {
                            continue;
                        }

                        overlap_extent_indices = Some((src_region_index, dst_region_index));
                    }
                }
            }
        }

        // VUID-VkCopyImageInfo2-pRegions-00124
        if let Some((src_region_index, dst_region_index)) = overlap_extent_indices {
            return Err(CopyError::OverlappingRegions {
                src_region_index,
                dst_region_index,
            });
        }

        // VUID-VkCopyImageInfo2-srcImageLayout-00128
        // VUID-VkCopyImageInfo2-dstImageLayout-00133
        if let Some((src_region_index, dst_region_index)) = overlap_subresource_indices {
            if src_image_layout != dst_image_layout {
                return Err(CopyError::OverlappingSubresourcesLayoutMismatch {
                    src_region_index,
                    dst_region_index,
                    src_image_layout,
                    dst_image_layout,
                });
            }
        }

        Ok(())
    }

    /// Copies from a buffer to an image.
    pub fn copy_buffer_to_image(
        &mut self,
        mut copy_buffer_to_image_info: CopyBufferToImageInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_copy_buffer_to_image(&mut copy_buffer_to_image_info)?;

        unsafe {
            self.inner.copy_buffer_to_image(copy_buffer_to_image_info)?;
        }

        Ok(self)
    }

    fn validate_copy_buffer_to_image(
        &self,
        copy_buffer_to_image_info: &mut CopyBufferToImageInfo,
    ) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdCopyBufferToImage2-renderpass
        if self.render_pass_state.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdCopyBufferToImage2-commandBuffer-cmdpool
        if !(self.queue_family().explicitly_supports_transfers()
            || self.queue_family().supports_graphics()
            || self.queue_family().supports_compute())
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &mut CopyBufferToImageInfo {
            src_buffer: ref buffer,
            dst_image: ref image,
            dst_image_layout: image_layout,
            ref regions,
            _ne: _,
        } = copy_buffer_to_image_info;

        // VUID-VkCopyBufferToImageInfo2-commonparent
        assert_eq!(device, buffer.device());
        assert_eq!(device, image.device());

        let buffer_inner = buffer.inner();
        let mut image_aspects = image.format().aspects();

        // VUID-VkCopyBufferToImageInfo2-commandBuffer-04477
        if !self.queue_family().supports_graphics() && !image_aspects.color {
            return Err(CopyError::DepthStencilNotSupportedByQueueFamily);
        }

        // VUID-VkCopyBufferToImageInfo2-srcBuffer-00174
        if !buffer.usage().transfer_src {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkCopyBufferToImageInfo2-dstImage-00177
        if !image.usage().transfer_dst {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            // VUID-VkCopyBufferToImageInfo2-dstImage-01997
            if !image.format_features().transfer_dst {
                return Err(CopyError::MissingFormatFeature {
                    resource: CopyErrorResource::Destination,
                    format_feature: "transfer_dst",
                });
            }
        }

        // VUID-VkCopyBufferToImageInfo2-dstImage-00179
        if image.samples() != SampleCount::Sample1 {
            return Err(CopyError::SampleCountInvalid {
                resource: CopyErrorResource::Destination,
                sample_count: image.samples(),
                allowed_sample_counts: SampleCounts {
                    sample1: true,
                    sample2: false,
                    sample4: false,
                    sample8: false,
                    sample16: false,
                    sample32: false,
                    sample64: false,
                },
            });
        }

        // VUID-VkCopyBufferToImageInfo2-dstImageLayout-01396
        if !matches!(
            image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Destination,
                image_layout,
            });
        }

        let extent_alignment = match self.queue_family().min_image_transfer_granularity() {
            [0, 0, 0] => None,
            min_image_transfer_granularity => {
                let granularity = move |block_extent: [u32; 3], is_multi_plane: bool| {
                    if is_multi_plane {
                        // Assume planes always have 1x1 blocks
                        min_image_transfer_granularity
                    } else {
                        // "The value returned in minImageTransferGranularity has a unit of
                        // compressed texel blocks for images having a block-compressed format, and
                        // a unit of texels otherwise."
                        [
                            min_image_transfer_granularity[0] * block_extent[0],
                            min_image_transfer_granularity[1] * block_extent[1],
                            min_image_transfer_granularity[2] * block_extent[2],
                        ]
                    }
                };

                Some(granularity(
                    image.format().block_extent(),
                    image_aspects.plane0,
                ))
            }
        };

        if image_aspects.plane0 {
            // VUID-VkCopyBufferToImageInfo2-aspectMask-01560
            image_aspects.color = false;
        }

        for (region_index, region) in regions.iter().enumerate() {
            let &BufferImageCopy {
                buffer_offset,
                buffer_row_length,
                buffer_image_height,
                ref image_subresource,
                image_offset,
                image_extent,
                _ne: _,
            } = region;

            // VUID-VkCopyBufferToImageInfo2-imageSubresource-01701
            if image_subresource.mip_level >= image.mip_levels() {
                return Err(CopyError::MipLevelsOutOfRange {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    mip_levels_range_end: image_subresource.mip_level + 1,
                    image_mip_levels: image.mip_levels(),
                });
            }

            // VUID-VkImageSubresourceLayers-layerCount-01700
            // VUID-VkCopyBufferToImageInfo2-baseArrayLayer-00213
            assert!(!image_subresource.array_layers.is_empty());

            // VUID-VkCopyBufferToImageInfo2-imageSubresource-01702
            // VUID-VkCopyBufferToImageInfo2-baseArrayLayer-00213
            if image_subresource.array_layers.end > image.dimensions().array_layers() {
                return Err(CopyError::ArrayLayersOutOfRange {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    array_layers_range_end: image_subresource.array_layers.end,
                    image_array_layers: image.dimensions().array_layers(),
                });
            }

            // VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask
            assert!(image_subresource.aspects != ImageAspects::none());

            // VUID-VkCopyBufferToImageInfo2-aspectMask-00211
            if !image_aspects.contains(&image_subresource.aspects) {
                return Err(CopyError::AspectsNotAllowed {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    aspects: image_subresource.aspects,
                    allowed_aspects: image_aspects,
                });
            }

            // VUID-VkBufferImageCopy2-aspectMask-00212
            // VUID-VkCopyBufferToImageInfo2-aspectMask-01560
            if image_subresource.aspects.iter().count() != 1 {
                return Err(CopyError::MultipleAspectsNotAllowed {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    aspects: image_subresource.aspects,
                });
            }

            let (image_subresource_format, image_subresource_extent) = if image_aspects.plane0 {
                if image_subresource.aspects.plane0 {
                    (
                        image.format().planes()[0],
                        image.dimensions().width_height_depth(),
                    )
                } else if image_subresource.aspects.plane1 {
                    (
                        image.format().planes()[1],
                        image
                            .format()
                            .ycbcr_chroma_sampling()
                            .unwrap()
                            .subsampled_extent(image.dimensions().width_height_depth()),
                    )
                } else {
                    (
                        image.format().planes()[2],
                        image
                            .format()
                            .ycbcr_chroma_sampling()
                            .unwrap()
                            .subsampled_extent(image.dimensions().width_height_depth()),
                    )
                }
            } else {
                (
                    image.format(),
                    image
                        .dimensions()
                        .mip_level_dimensions(image_subresource.mip_level)
                        .unwrap()
                        .width_height_depth(),
                )
            };

            if let Some(extent_alignment) = extent_alignment {
                for i in 0..3 {
                    // VUID-VkBufferImageCopy2-imageExtent-06659
                    // VUID-VkBufferImageCopy2-imageExtent-06660
                    // VUID-VkBufferImageCopy2-imageExtent-06661
                    assert!(image_extent[i] != 0);

                    // VUID-VkCopyBufferToImageInfo2-pRegions-06223
                    // VUID-VkCopyBufferToImageInfo2-pRegions-06224
                    // VUID-VkCopyBufferToImageInfo2-imageOffset-00200
                    if image_offset[i] + image_extent[i] > image_subresource_extent[i] {
                        return Err(CopyError::RegionOutOfImageBounds {
                            resource: CopyErrorResource::Destination,
                            region_index,
                            offset_range_end: [
                                image_offset[0] + image_extent[0],
                                image_offset[1] + image_extent[1],
                                image_offset[2] + image_extent[2],
                            ],
                            subresource_extent: image_subresource_extent,
                        });
                    }

                    // VUID-VkCopyBufferToImageInfo2-imageOffset-01793
                    // VUID-VkCopyBufferToImageInfo2-imageOffset-00205
                    if image_offset[i] % extent_alignment[i] != 0 {
                        return Err(CopyError::OffsetNotAlignedForImage {
                            resource: CopyErrorResource::Destination,
                            region_index,
                            offset: image_offset,
                            required_alignment: extent_alignment,
                        });
                    }

                    // VUID-VkCopyBufferToImageInfo2-imageOffset-01793
                    // VUID-VkCopyBufferToImageInfo2-imageExtent-00207
                    // VUID-VkCopyBufferToImageInfo2-imageExtent-00208
                    // VUID-VkCopyBufferToImageInfo2-imageExtent-00209
                    if image_offset[i] + image_extent[i] != image_subresource_extent[i]
                        && image_extent[i] % extent_alignment[i] != 0
                    {
                        return Err(CopyError::ExtentNotAlignedForImage {
                            resource: CopyErrorResource::Destination,
                            region_index,
                            extent: image_extent,
                            required_alignment: extent_alignment,
                        });
                    }
                }
            } else {
                // If granularity is `None`, then we can only copy whole subresources.

                // VUID-VkCopyBufferToImageInfo2-imageOffset-01793
                if image_offset != [0, 0, 0] {
                    return Err(CopyError::OffsetNotAlignedForImage {
                        resource: CopyErrorResource::Destination,
                        region_index,
                        offset: image_offset,
                        required_alignment: image_subresource_extent,
                    });
                }

                // VUID-VkCopyBufferToImageInfo2-imageOffset-01793
                if image_extent != image_subresource_extent {
                    return Err(CopyError::ExtentNotAlignedForImage {
                        resource: CopyErrorResource::Destination,
                        region_index,
                        extent: image_extent,
                        required_alignment: image_subresource_extent,
                    });
                }
            }

            // VUID-VkBufferImageCopy2-bufferRowLength-00195
            if !(buffer_row_length == 0 || buffer_row_length >= image_extent[0]) {
                return Err(CopyError::BufferRowLengthTooSmall {
                    resource: CopyErrorResource::Source,
                    region_index,
                    row_length: buffer_row_length,
                    min: image_extent[0],
                });
            }

            // VUID-VkBufferImageCopy2-bufferImageHeight-00196
            if !(buffer_image_height == 0 || buffer_image_height >= image_extent[1]) {
                return Err(CopyError::BufferImageHeightTooSmall {
                    resource: CopyErrorResource::Source,
                    region_index,
                    image_height: buffer_image_height,
                    min: image_extent[1],
                });
            }

            let image_subresource_block_extent = image_subresource_format.block_extent();

            // VUID-VkCopyBufferToImageInfo2-bufferRowLength-00203
            if buffer_row_length % image_subresource_block_extent[0] != 0 {
                return Err(CopyError::BufferRowLengthNotAligned {
                    resource: CopyErrorResource::Source,
                    region_index,
                    row_length: buffer_row_length,
                    required_alignment: image_subresource_block_extent[0],
                });
            }

            // VUID-VkCopyBufferToImageInfo2-bufferImageHeight-00204
            if buffer_image_height % image_subresource_block_extent[1] != 0 {
                return Err(CopyError::BufferImageHeightNotAligned {
                    resource: CopyErrorResource::Source,
                    region_index,
                    image_height: buffer_image_height,
                    required_alignment: image_subresource_block_extent[1],
                });
            }

            // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkBufferImageCopy.html#_description
            let image_subresource_block_size = if image_subresource.aspects.stencil {
                1
            } else if image_subresource.aspects.depth {
                match image_subresource_format {
                    Format::D16_UNORM | Format::D16_UNORM_S8_UINT => 2,
                    Format::D32_SFLOAT
                    | Format::D32_SFLOAT_S8_UINT
                    | Format::X8_D24_UNORM_PACK32
                    | Format::D24_UNORM_S8_UINT => 4,
                    _ => unreachable!(),
                }
            } else {
                image_subresource_format.block_size().unwrap()
            };

            // VUID-VkCopyBufferToImageInfo2-pRegions-04725
            // VUID-VkCopyBufferToImageInfo2-pRegions-04726
            if (buffer_row_length / image_subresource_block_extent[0]) as DeviceSize
                * image_subresource_block_size
                > 0x7FFFFFFF
            {
                return Err(CopyError::BufferRowLengthTooLarge {
                    resource: CopyErrorResource::Source,
                    region_index,
                    buffer_row_length,
                });
            }

            let buffer_offset_alignment = if image_aspects.depth || image_aspects.stencil {
                4
            } else {
                let mut buffer_offset_alignment = image_subresource_block_size;

                // VUID-VkCopyBufferToImageInfo2-commandBuffer-04052
                // Make the alignment a multiple of 4.
                if !(self.queue_family().supports_graphics()
                    || self.queue_family().supports_compute())
                {
                    if buffer_offset_alignment % 2 != 0 {
                        buffer_offset_alignment *= 2;
                    }

                    if buffer_offset_alignment % 4 != 0 {
                        buffer_offset_alignment *= 2;
                    }
                }

                buffer_offset_alignment
            };

            // VUID-VkCopyBufferToImageInfo2-bufferOffset-00206
            // VUID-VkCopyBufferToImageInfo2-bufferOffset-01558
            // VUID-VkCopyBufferToImageInfo2-bufferOffset-01559
            // VUID-VkCopyBufferToImageInfo2-srcImage-04053
            if (buffer_inner.offset + buffer_offset) % buffer_offset_alignment != 0 {
                return Err(CopyError::OffsetNotAlignedForBuffer {
                    resource: CopyErrorResource::Source,
                    region_index,
                    offset: buffer_inner.offset + buffer_offset,
                    required_alignment: buffer_offset_alignment,
                });
            }

            let buffer_copy_size = region.buffer_copy_size(image_subresource_format);

            // VUID-VkCopyBufferToImageInfo2-pRegions-00171
            if buffer_offset + buffer_copy_size > buffer.size() {
                return Err(CopyError::RegionOutOfBufferBounds {
                    resource: CopyErrorResource::Source,
                    region_index,
                    offset_range_end: buffer_offset + buffer_copy_size,
                    buffer_size: buffer.size(),
                });
            }
        }

        // VUID-VkCopyBufferToImageInfo2-pRegions-00173
        // Can't occur as long as memory aliasing isn't allowed.

        Ok(())
    }

    /// Copies from an image to a buffer.
    pub fn copy_image_to_buffer(
        &mut self,
        mut copy_image_to_buffer_info: CopyImageToBufferInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_copy_image_to_buffer(&mut copy_image_to_buffer_info)?;

        unsafe {
            self.inner.copy_image_to_buffer(copy_image_to_buffer_info)?;
        }

        Ok(self)
    }

    fn validate_copy_image_to_buffer(
        &self,
        copy_image_to_buffer_info: &mut CopyImageToBufferInfo,
    ) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdCopyImageToBuffer2-renderpass
        if self.render_pass_state.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdCopyImageToBuffer2-commandBuffer-cmdpool
        if !(self.queue_family().explicitly_supports_transfers()
            || self.queue_family().supports_graphics()
            || self.queue_family().supports_compute())
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &mut CopyImageToBufferInfo {
            src_image: ref image,
            src_image_layout: image_layout,
            dst_buffer: ref buffer,
            ref regions,
            _ne: _,
        } = copy_image_to_buffer_info;

        // VUID-VkCopyImageToBufferInfo2-commonparent
        assert_eq!(device, buffer.device());
        assert_eq!(device, image.device());

        let buffer_inner = buffer.inner();
        let mut image_aspects = image.format().aspects();

        // VUID-VkCopyImageToBufferInfo2-srcImage-00186
        if !image.usage().transfer_src {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkCopyImageToBufferInfo2-dstBuffer-00191
        if !buffer.usage().transfer_dst {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            // VUID-VkCopyImageToBufferInfo2-srcImage-01998
            if !image.format_features().transfer_src {
                return Err(CopyError::MissingFormatFeature {
                    resource: CopyErrorResource::Source,
                    format_feature: "transfer_src",
                });
            }
        }

        // VUID-VkCopyImageToBufferInfo2-srcImage-00188
        if image.samples() != SampleCount::Sample1 {
            return Err(CopyError::SampleCountInvalid {
                resource: CopyErrorResource::Source,
                sample_count: image.samples(),
                allowed_sample_counts: SampleCounts {
                    sample1: true,
                    sample2: false,
                    sample4: false,
                    sample8: false,
                    sample16: false,
                    sample32: false,
                    sample64: false,
                },
            });
        }

        // VUID-VkCopyImageToBufferInfo2-srcImageLayout-01397
        if !matches!(
            image_layout,
            ImageLayout::TransferSrcOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Source,
                image_layout,
            });
        }

        let extent_alignment = match self.queue_family().min_image_transfer_granularity() {
            [0, 0, 0] => None,
            min_image_transfer_granularity => {
                let granularity = move |block_extent: [u32; 3], is_multi_plane: bool| {
                    if is_multi_plane {
                        // Assume planes always have 1x1 blocks
                        min_image_transfer_granularity
                    } else {
                        // "The value returned in minImageTransferGranularity has a unit of
                        // compressed texel blocks for images having a block-compressed format, and
                        // a unit of texels otherwise."
                        [
                            min_image_transfer_granularity[0] * block_extent[0],
                            min_image_transfer_granularity[1] * block_extent[1],
                            min_image_transfer_granularity[2] * block_extent[2],
                        ]
                    }
                };

                Some(granularity(
                    image.format().block_extent(),
                    image_aspects.plane0,
                ))
            }
        };

        if image_aspects.plane0 {
            // VUID-VkCopyImageToBufferInfo2-aspectMask-01560
            image_aspects.color = false;
        }

        for (region_index, region) in regions.iter().enumerate() {
            let &BufferImageCopy {
                buffer_offset,
                buffer_row_length,
                buffer_image_height,
                ref image_subresource,
                image_offset,
                image_extent,
                _ne: _,
            } = region;

            // VUID-VkCopyImageToBufferInfo2-imageSubresource-01703
            if image_subresource.mip_level >= image.mip_levels() {
                return Err(CopyError::MipLevelsOutOfRange {
                    resource: CopyErrorResource::Source,
                    region_index,
                    mip_levels_range_end: image_subresource.mip_level + 1,
                    image_mip_levels: image.mip_levels(),
                });
            }

            // VUID-VkImageSubresourceLayers-layerCount-01700
            assert!(!image_subresource.array_layers.is_empty());

            // VUID-VkCopyImageToBufferInfo2-imageSubresource-01704
            // VUID-VkCopyImageToBufferInfo2-baseArrayLayer-00213
            if image_subresource.array_layers.end > image.dimensions().array_layers() {
                return Err(CopyError::ArrayLayersOutOfRange {
                    resource: CopyErrorResource::Source,
                    region_index,
                    array_layers_range_end: image_subresource.array_layers.end,
                    image_array_layers: image.dimensions().array_layers(),
                });
            }

            // VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask
            assert!(image_subresource.aspects != ImageAspects::none());

            // VUID-VkCopyImageToBufferInfo2-aspectMask-00211
            if !image_aspects.contains(&image_subresource.aspects) {
                return Err(CopyError::AspectsNotAllowed {
                    resource: CopyErrorResource::Source,
                    region_index,
                    aspects: image_subresource.aspects,
                    allowed_aspects: image_aspects,
                });
            }

            // VUID-VkBufferImageCopy2-aspectMask-00212
            if image_subresource.aspects.iter().count() != 1 {
                return Err(CopyError::MultipleAspectsNotAllowed {
                    resource: CopyErrorResource::Source,
                    region_index,
                    aspects: image_subresource.aspects,
                });
            }

            let (image_subresource_format, image_subresource_extent) = if image_aspects.plane0 {
                if image_subresource.aspects.plane0 {
                    (
                        image.format().planes()[0],
                        image.dimensions().width_height_depth(),
                    )
                } else if image_subresource.aspects.plane1 {
                    (
                        image.format().planes()[1],
                        image
                            .format()
                            .ycbcr_chroma_sampling()
                            .unwrap()
                            .subsampled_extent(image.dimensions().width_height_depth()),
                    )
                } else {
                    (
                        image.format().planes()[2],
                        image
                            .format()
                            .ycbcr_chroma_sampling()
                            .unwrap()
                            .subsampled_extent(image.dimensions().width_height_depth()),
                    )
                }
            } else {
                (
                    image.format(),
                    image
                        .dimensions()
                        .mip_level_dimensions(image_subresource.mip_level)
                        .unwrap()
                        .width_height_depth(),
                )
            };

            if let Some(extent_alignment) = extent_alignment {
                for i in 0..3 {
                    // VUID-VkBufferImageCopy2-imageExtent-06659
                    // VUID-VkBufferImageCopy2-imageExtent-06660
                    // VUID-VkBufferImageCopy2-imageExtent-06661
                    assert!(image_extent[i] != 0);

                    // VUID-VkCopyImageToBufferInfo2-imageOffset-00197
                    // VUID-VkCopyImageToBufferInfo2-imageOffset-00198
                    // VUID-VkCopyImageToBufferInfo2-imageOffset-00200
                    if image_offset[i] + image_extent[i] > image_subresource_extent[i] {
                        return Err(CopyError::RegionOutOfImageBounds {
                            resource: CopyErrorResource::Source,
                            region_index,
                            offset_range_end: [
                                image_offset[0] + image_extent[0],
                                image_offset[1] + image_extent[1],
                                image_offset[2] + image_extent[2],
                            ],
                            subresource_extent: image_subresource_extent,
                        });
                    }

                    // VUID-VkCopyImageToBufferInfo2-imageOffset-01794
                    // VUID-VkCopyImageToBufferInfo2-imageOffset-00205
                    if image_offset[i] % extent_alignment[i] != 0 {
                        return Err(CopyError::OffsetNotAlignedForImage {
                            resource: CopyErrorResource::Source,
                            region_index,
                            offset: image_offset,
                            required_alignment: extent_alignment,
                        });
                    }

                    // VUID-VkCopyImageToBufferInfo2-imageOffset-01794
                    // VUID-VkCopyImageToBufferInfo2-imageExtent-00207
                    // VUID-VkCopyImageToBufferInfo2-imageExtent-00208
                    // VUID-VkCopyImageToBufferInfo2-imageExtent-00209
                    if image_offset[i] + image_extent[i] != image_subresource_extent[i]
                        && image_extent[i] % extent_alignment[i] != 0
                    {
                        return Err(CopyError::ExtentNotAlignedForImage {
                            resource: CopyErrorResource::Source,
                            region_index,
                            extent: image_extent,
                            required_alignment: extent_alignment,
                        });
                    }
                }
            } else {
                // If granularity is `None`, then we can only copy whole subresources.

                // VUID-VkCopyBufferToImageInfo2-imageOffset-01793
                if image_offset != [0, 0, 0] {
                    return Err(CopyError::OffsetNotAlignedForImage {
                        resource: CopyErrorResource::Source,
                        region_index,
                        offset: image_offset,
                        required_alignment: image_subresource_extent,
                    });
                }

                // VUID-VkCopyBufferToImageInfo2-imageOffset-01793
                if image_extent != image_subresource_extent {
                    return Err(CopyError::ExtentNotAlignedForImage {
                        resource: CopyErrorResource::Source,
                        region_index,
                        extent: image_extent,
                        required_alignment: image_subresource_extent,
                    });
                }
            }

            // VUID-VkBufferImageCopy2-bufferRowLength-00195
            if !(buffer_row_length == 0 || buffer_row_length >= image_extent[0]) {
                return Err(CopyError::BufferRowLengthTooSmall {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    row_length: buffer_row_length,
                    min: image_extent[0],
                });
            }

            // VUID-VkBufferImageCopy2-bufferImageHeight-00196
            if !(buffer_image_height == 0 || buffer_image_height >= image_extent[1]) {
                return Err(CopyError::BufferImageHeightTooSmall {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    image_height: buffer_image_height,
                    min: image_extent[1],
                });
            }

            let image_subresource_block_extent = image_subresource_format.block_extent();

            // VUID-VkCopyImageToBufferInfo2-bufferRowLength-00203
            if buffer_row_length % image_subresource_block_extent[0] != 0 {
                return Err(CopyError::BufferRowLengthNotAligned {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    row_length: buffer_row_length,
                    required_alignment: image_subresource_block_extent[0],
                });
            }

            // VUID-VkCopyImageToBufferInfo2-bufferImageHeight-00204
            if buffer_image_height % image_subresource_block_extent[1] != 0 {
                return Err(CopyError::BufferImageHeightNotAligned {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    image_height: buffer_image_height,
                    required_alignment: image_subresource_block_extent[1],
                });
            }

            // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkBufferImageCopy.html#_description
            let image_subresource_block_size = if image_subresource.aspects.stencil {
                1
            } else if image_subresource.aspects.depth {
                match image_subresource_format {
                    Format::D16_UNORM | Format::D16_UNORM_S8_UINT => 2,
                    Format::D32_SFLOAT
                    | Format::D32_SFLOAT_S8_UINT
                    | Format::X8_D24_UNORM_PACK32
                    | Format::D24_UNORM_S8_UINT => 4,
                    _ => unreachable!(),
                }
            } else {
                image_subresource_format.block_size().unwrap()
            };

            // VUID-VkCopyImageToBufferInfo2-pRegions-04725
            // VUID-VkCopyImageToBufferInfo2-pRegions-04726
            if (buffer_row_length / image_subresource_block_extent[0]) as DeviceSize
                * image_subresource_block_size
                > 0x7FFFFFFF
            {
                return Err(CopyError::BufferRowLengthTooLarge {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    buffer_row_length,
                });
            }

            let buffer_offset_alignment = if image_aspects.depth || image_aspects.stencil {
                4
            } else {
                let mut buffer_offset_alignment = image_subresource_block_size;

                // VUID-VkCopyImageToBufferInfo2-commandBuffer-04052
                // Make the alignment a multiple of 4.
                if !(self.queue_family().supports_graphics()
                    || self.queue_family().supports_compute())
                {
                    if buffer_offset_alignment % 2 != 0 {
                        buffer_offset_alignment *= 2;
                    }

                    if buffer_offset_alignment % 4 != 0 {
                        buffer_offset_alignment *= 2;
                    }
                }

                buffer_offset_alignment
            };

            // VUID-VkCopyImageToBufferInfo2-bufferOffset-01558
            // VUID-VkCopyImageToBufferInfo2-bufferOffset-01559
            // VUID-VkCopyImageToBufferInfo2-bufferOffset-00206
            // VUID-VkCopyImageToBufferInfo2-srcImage-04053
            if (buffer_inner.offset + buffer_offset) % buffer_offset_alignment != 0 {
                return Err(CopyError::OffsetNotAlignedForBuffer {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    offset: buffer_inner.offset + buffer_offset,
                    required_alignment: buffer_offset_alignment,
                });
            }

            let buffer_copy_size = region.buffer_copy_size(image_subresource_format);

            // VUID-VkCopyImageToBufferInfo2-pRegions-00183
            if buffer_offset + buffer_copy_size > buffer.size() {
                return Err(CopyError::RegionOutOfBufferBounds {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    offset_range_end: buffer_offset + buffer_copy_size,
                    buffer_size: buffer.size(),
                });
            }
        }

        // VUID-VkCopyImageToBufferInfo2-pRegions-00184
        // Can't occur as long as memory aliasing isn't allowed.

        Ok(())
    }

    /// Fills a region of a buffer with repeated copies of a value.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatedly written through the entire buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `dst_buffer` was not created from the same device as `self`.
    #[inline]
    pub fn fill_buffer(
        &mut self,
        mut fill_buffer_info: FillBufferInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_fill_buffer(&mut fill_buffer_info)?;

        unsafe {
            self.inner.fill_buffer(fill_buffer_info)?;
        }

        Ok(self)
    }

    fn validate_fill_buffer(&self, fill_buffer_info: &mut FillBufferInfo) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdFillBuffer-renderpass
        if self.render_pass_state.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            // VUID-vkCmdFillBuffer-commandBuffer-cmdpool
            if !(self.queue_family().explicitly_supports_transfers()
                || self.queue_family().supports_graphics()
                || self.queue_family().supports_compute())
            {
                return Err(CopyError::NotSupportedByQueueFamily);
            }
        } else {
            // VUID-vkCmdFillBuffer-commandBuffer-00030
            if !(self.queue_family().supports_graphics() || self.queue_family().supports_compute())
            {
                return Err(CopyError::NotSupportedByQueueFamily);
            }
        }

        let &mut FillBufferInfo {
            data,
            ref dst_buffer,
            dst_offset,
            size,
            _ne: _,
        } = fill_buffer_info;

        let dst_buffer_inner = dst_buffer.inner();

        // VUID-vkCmdFillBuffer-commonparent
        assert_eq!(device, dst_buffer.device());

        // VUID-vkCmdFillBuffer-size-00026
        assert!(size != 0);

        // VUID-vkCmdFillBuffer-dstBuffer-00029
        if !dst_buffer.usage().transfer_dst {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        // VUID-vkCmdFillBuffer-dstOffset-00024
        // VUID-vkCmdFillBuffer-size-00027
        if dst_offset + size > dst_buffer.size() {
            return Err(CopyError::RegionOutOfBufferBounds {
                resource: CopyErrorResource::Destination,
                region_index: 0,
                offset_range_end: dst_offset + size,
                buffer_size: dst_buffer.size(),
            });
        }

        // VUID-vkCmdFillBuffer-dstOffset-00025
        if (dst_buffer_inner.offset + dst_offset) % 4 != 0 {
            return Err(CopyError::OffsetNotAlignedForBuffer {
                resource: CopyErrorResource::Destination,
                region_index: 0,
                offset: dst_buffer_inner.offset + dst_offset,
                required_alignment: 4,
            });
        }

        // VUID-vkCmdFillBuffer-size-00028
        if size % 4 != 0 {
            return Err(CopyError::SizeNotAlignedForBuffer {
                resource: CopyErrorResource::Destination,
                region_index: 0,
                size,
                required_alignment: 4,
            });
        }

        Ok(())
    }

    /// Writes data to a region of a buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `dst_buffer` was not created from the same device as `self`.
    #[inline]
    pub fn update_buffer<B, D, Dd>(
        &mut self,
        data: Dd,
        dst_buffer: Arc<B>,
        dst_offset: DeviceSize,
    ) -> Result<&mut Self, CopyError>
    where
        B: TypedBufferAccess<Content = D> + 'static,
        D: BufferContents + ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        self.validate_update_buffer(data.deref(), &dst_buffer, dst_offset)?;

        unsafe {
            self.inner.update_buffer(data, dst_buffer, dst_offset)?;
        }

        Ok(self)
    }

    fn validate_update_buffer<D>(
        &self,
        data: &D,
        dst_buffer: &dyn BufferAccess,
        dst_offset: DeviceSize,
    ) -> Result<(), CopyError>
    where
        D: ?Sized,
    {
        let device = self.device();

        // VUID-vkCmdUpdateBuffer-renderpass
        if self.render_pass_state.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdUpdateBuffer-commandBuffer-cmdpool
        if !(self.queue_family().explicitly_supports_transfers()
            || self.queue_family().supports_graphics()
            || self.queue_family().supports_compute())
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let dst_buffer_inner = dst_buffer.inner();

        // VUID-vkCmdUpdateBuffer-commonparent
        assert_eq!(device, dst_buffer.device());

        // VUID-vkCmdUpdateBuffer-dataSize-arraylength
        assert!(size_of_val(data) != 0);

        // VUID-vkCmdUpdateBuffer-dstBuffer-00034
        if !dst_buffer.usage().transfer_dst {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        // VUID-vkCmdUpdateBuffer-dstOffset-00032
        // VUID-vkCmdUpdateBuffer-dataSize-00033
        if dst_offset + size_of_val(data) as DeviceSize > dst_buffer.size() {
            return Err(CopyError::RegionOutOfBufferBounds {
                resource: CopyErrorResource::Destination,
                region_index: 0,
                offset_range_end: dst_offset + size_of_val(data) as DeviceSize,
                buffer_size: dst_buffer.size(),
            });
        }

        // VUID-vkCmdUpdateBuffer-dstOffset-00036
        if (dst_buffer_inner.offset + dst_offset) % 4 != 0 {
            return Err(CopyError::OffsetNotAlignedForBuffer {
                resource: CopyErrorResource::Destination,
                region_index: 0,
                offset: dst_buffer_inner.offset + dst_offset,
                required_alignment: 4,
            });
        }

        // VUID-vkCmdUpdateBuffer-dataSize-00037
        if size_of_val(data) > 65536 {
            return Err(CopyError::DataTooLarge {
                size: size_of_val(data) as DeviceSize,
                max: 65536,
            });
        }

        // VUID-vkCmdUpdateBuffer-dataSize-00038
        if size_of_val(data) % 4 != 0 {
            return Err(CopyError::SizeNotAlignedForBuffer {
                resource: CopyErrorResource::Destination,
                region_index: 0,
                size: size_of_val(data) as DeviceSize,
                required_alignment: 4,
            });
        }

        Ok(())
    }
}

/// Computes the minimum required len in elements for buffer with image data in specified
/// format of specified size.
fn required_size_for_format(format: Format, extent: [u32; 3], layer_count: u32) -> DeviceSize {
    let num_blocks = extent
        .into_iter()
        .zip(format.block_extent())
        .map(|(extent, block_extent)| {
            let extent = extent as DeviceSize;
            let block_extent = block_extent as DeviceSize;
            (extent + block_extent - 1) / block_extent
        })
        .product::<DeviceSize>()
        * layer_count as DeviceSize;
    let block_size = format
        .block_size()
        .expect("this format cannot accept pixels");
    num_blocks * block_size
}

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdCopyBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer(
        &mut self,
        copy_buffer_info: CopyBufferInfo,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            copy_buffer_info: CopyBufferInfo,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "copy_buffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_buffer(&self.copy_buffer_info);
            }
        }

        let CopyBufferInfo {
            src_buffer,
            dst_buffer,
            regions,
            _ne: _,
        } = &copy_buffer_info;

        // if its the same image in source and destination, we need to lock it once
        let resources: SmallVec<[_; 8]> = regions
            .iter()
            .flat_map(|region| {
                let &BufferCopy {
                    src_offset,
                    dst_offset,
                    size,
                    _ne: _,
                } = region;

                [
                    (
                        "src_buffer".into(),
                        Resource::Buffer {
                            buffer: src_buffer.clone(),
                            range: src_offset..src_offset + size,
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    transfer: true,
                                    ..PipelineStages::none()
                                },
                                access: AccessFlags {
                                    transfer_read: true,
                                    ..AccessFlags::none()
                                },
                                exclusive: false,
                            },
                        },
                    ),
                    (
                        "dst_buffer".into(),
                        Resource::Buffer {
                            buffer: dst_buffer.clone(),
                            range: dst_offset..dst_offset + size,
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    transfer: true,
                                    ..PipelineStages::none()
                                },
                                access: AccessFlags {
                                    transfer_write: true,
                                    ..AccessFlags::none()
                                },
                                exclusive: true,
                            },
                        },
                    ),
                ]
            })
            .collect();

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd { copy_buffer_info }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdCopyImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image(
        &mut self,
        copy_image_info: CopyImageInfo,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            copy_image_info: CopyImageInfo,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "copy_buffer_to_image"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_image(&self.copy_image_info);
            }
        }

        let &CopyImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = &copy_image_info;

        let resources: SmallVec<[_; 8]> = regions
            .iter()
            .flat_map(|region| {
                let &ImageCopy {
                    ref src_subresource,
                    src_offset,
                    ref dst_subresource,
                    dst_offset,
                    extent,
                    _ne: _,
                } = region;

                [
                    (
                        "src_image".into(),
                        Resource::Image {
                            image: src_image.clone(),
                            subresource_range: src_subresource.clone().into(),
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    transfer: true,
                                    ..PipelineStages::none()
                                },
                                access: AccessFlags {
                                    transfer_read: true,
                                    ..AccessFlags::none()
                                },
                                exclusive: false,
                            },
                            start_layout: src_image_layout,
                            end_layout: src_image_layout,
                        },
                    ),
                    (
                        "dst_image".into(),
                        Resource::Image {
                            image: dst_image.clone(),
                            subresource_range: dst_subresource.clone().into(),
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    transfer: true,
                                    ..PipelineStages::none()
                                },
                                access: AccessFlags {
                                    transfer_write: true,
                                    ..AccessFlags::none()
                                },
                                exclusive: true,
                            },
                            start_layout: dst_image_layout,
                            end_layout: dst_image_layout,
                        },
                    ),
                ]
            })
            .collect();

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd { copy_image_info }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdCopyBufferToImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer_to_image(
        &mut self,
        copy_buffer_to_image_info: CopyBufferToImageInfo,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            copy_buffer_to_image_info: CopyBufferToImageInfo,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "copy_buffer_to_image"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_buffer_to_image(&self.copy_buffer_to_image_info);
            }
        }

        let &CopyBufferToImageInfo {
            ref src_buffer,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = &copy_buffer_to_image_info;

        let resources: SmallVec<[_; 8]> = regions
            .iter()
            .flat_map(|region| {
                let &BufferImageCopy {
                    buffer_offset,
                    buffer_row_length,
                    buffer_image_height,
                    ref image_subresource,
                    image_offset,
                    image_extent,
                    _ne: _,
                } = region;

                [
                    (
                        "src_buffer".into(),
                        Resource::Buffer {
                            buffer: src_buffer.clone(),
                            range: buffer_offset
                                ..buffer_offset + region.buffer_copy_size(dst_image.format()),
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    transfer: true,
                                    ..PipelineStages::none()
                                },
                                access: AccessFlags {
                                    transfer_read: true,
                                    ..AccessFlags::none()
                                },
                                exclusive: false,
                            },
                        },
                    ),
                    (
                        "dst_image".into(),
                        Resource::Image {
                            image: dst_image.clone(),
                            subresource_range: image_subresource.clone().into(),
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    transfer: true,
                                    ..PipelineStages::none()
                                },
                                access: AccessFlags {
                                    transfer_write: true,
                                    ..AccessFlags::none()
                                },
                                exclusive: true,
                            },
                            start_layout: dst_image_layout,
                            end_layout: dst_image_layout,
                        },
                    ),
                ]
            })
            .collect();

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            copy_buffer_to_image_info,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdCopyImageToBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image_to_buffer(
        &mut self,
        copy_image_to_buffer_info: CopyImageToBufferInfo,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            copy_image_to_buffer_info: CopyImageToBufferInfo,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "copy_image_to_buffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_image_to_buffer(&self.copy_image_to_buffer_info);
            }
        }

        let &CopyImageToBufferInfo {
            ref src_image,
            src_image_layout,
            ref dst_buffer,
            ref regions,
            _ne: _,
        } = &copy_image_to_buffer_info;

        let resources: SmallVec<[_; 8]> = regions
            .iter()
            .flat_map(|region| {
                let &BufferImageCopy {
                    buffer_offset,
                    buffer_row_length,
                    buffer_image_height,
                    ref image_subresource,
                    image_offset,
                    image_extent,
                    _ne: _,
                } = region;

                [
                    (
                        "src_image".into(),
                        Resource::Image {
                            image: src_image.clone(),
                            subresource_range: image_subresource.clone().into(),
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    transfer: true,
                                    ..PipelineStages::none()
                                },
                                access: AccessFlags {
                                    transfer_read: true,
                                    ..AccessFlags::none()
                                },
                                exclusive: false,
                            },
                            start_layout: src_image_layout,
                            end_layout: src_image_layout,
                        },
                    ),
                    (
                        "dst_buffer".into(),
                        Resource::Buffer {
                            buffer: dst_buffer.clone(),
                            range: buffer_offset
                                ..buffer_offset + region.buffer_copy_size(src_image.format()),
                            memory: PipelineMemoryAccess {
                                stages: PipelineStages {
                                    transfer: true,
                                    ..PipelineStages::none()
                                },
                                access: AccessFlags {
                                    transfer_write: true,
                                    ..AccessFlags::none()
                                },
                                exclusive: true,
                            },
                        },
                    ),
                ]
            })
            .collect();

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            copy_image_to_buffer_info,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer(
        &mut self,
        fill_buffer_info: FillBufferInfo,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            fill_buffer_info: FillBufferInfo,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "fill_buffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.fill_buffer(&self.fill_buffer_info);
            }
        }

        let &FillBufferInfo {
            data,
            ref dst_buffer,
            dst_offset,
            size,
            _ne: _,
        } = &fill_buffer_info;

        let resources = [(
            "dst_buffer".into(),
            Resource::Buffer {
                buffer: dst_buffer.clone(),
                range: dst_offset..dst_offset + size,
                memory: PipelineMemoryAccess {
                    stages: PipelineStages {
                        transfer: true,
                        ..PipelineStages::none()
                    },
                    access: AccessFlags {
                        transfer_write: true,
                        ..AccessFlags::none()
                    },
                    exclusive: true,
                },
            },
        )];

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd { fill_buffer_info }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    #[inline]
    pub unsafe fn update_buffer<D, Dd>(
        &mut self,
        data: Dd,
        dst_buffer: Arc<dyn BufferAccess>,
        dst_offset: DeviceSize,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        D: BufferContents + ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        struct Cmd<Dd> {
            data: Dd,
            dst_buffer: Arc<dyn BufferAccess>,
            dst_offset: DeviceSize,
        }

        impl<D, Dd> Command for Cmd<Dd>
        where
            D: BufferContents + ?Sized,
            Dd: SafeDeref<Target = D> + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "update_buffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.update_buffer(self.data.deref(), self.dst_buffer.as_ref(), self.dst_offset);
            }
        }

        let resources = [(
            "dst_buffer".into(),
            Resource::Buffer {
                buffer: dst_buffer.clone(),
                range: dst_offset..dst_offset + size_of_val(data.deref()) as DeviceSize,
                memory: PipelineMemoryAccess {
                    stages: PipelineStages {
                        transfer: true,
                        ..PipelineStages::none()
                    },
                    access: AccessFlags {
                        transfer_write: true,
                        ..AccessFlags::none()
                    },
                    exclusive: true,
                },
            },
        )];

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            data,
            dst_buffer,
            dst_offset,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdCopyBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer(&mut self, copy_buffer_info: &CopyBufferInfo) {
        let CopyBufferInfo {
            src_buffer,
            dst_buffer,
            regions,
            _ne: _,
        } = copy_buffer_info;

        if regions.is_empty() {
            return;
        }

        let src_buffer_inner = src_buffer.inner();
        let dst_buffer_inner = dst_buffer.inner();

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3
            || self.device.enabled_extensions().khr_copy_commands2
        {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &BufferCopy {
                        src_offset,
                        dst_offset,
                        size,
                        _ne,
                    } = region;

                    ash::vk::BufferCopy2 {
                        src_offset: src_offset + src_buffer_inner.offset,
                        dst_offset: dst_offset + dst_buffer_inner.offset,
                        size,
                        ..Default::default()
                    }
                })
                .collect();

            let copy_buffer_info = ash::vk::CopyBufferInfo2 {
                src_buffer: src_buffer_inner.buffer.internal_object(),
                dst_buffer: dst_buffer_inner.buffer.internal_object(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                ..Default::default()
            };

            if self.device.api_version() >= Version::V1_3 {
                fns.v1_3.cmd_copy_buffer2(self.handle, &copy_buffer_info);
            } else {
                fns.khr_copy_commands2
                    .cmd_copy_buffer2_khr(self.handle, &copy_buffer_info);
            }
        } else {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &BufferCopy {
                        src_offset,
                        dst_offset,
                        size,
                        _ne,
                    } = region;

                    ash::vk::BufferCopy {
                        src_offset: src_offset + src_buffer_inner.offset,
                        dst_offset: dst_offset + dst_buffer_inner.offset,
                        size,
                    }
                })
                .collect();

            fns.v1_0.cmd_copy_buffer(
                self.handle,
                src_buffer_inner.buffer.internal_object(),
                dst_buffer_inner.buffer.internal_object(),
                regions.len() as u32,
                regions.as_ptr(),
            );
        }
    }

    /// Calls `vkCmdCopyImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image(&mut self, copy_image_info: &CopyImageInfo) {
        let &CopyImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = copy_image_info;

        if regions.is_empty() {
            return;
        }

        let src_image_inner = src_image.inner();
        let dst_image_inner = dst_image.inner();

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3
            || self.device.enabled_extensions().khr_copy_commands2
        {
            let regions: SmallVec<[_; 8]> = regions
                .into_iter()
                .map(|region| {
                    let &ImageCopy {
                        ref src_subresource,
                        src_offset,
                        ref dst_subresource,
                        dst_offset,
                        extent,
                        _ne: _,
                    } = region;

                    let mut src_subresource = src_subresource.clone();
                    src_subresource.array_layers.start += src_image_inner.first_layer;
                    src_subresource.array_layers.end += src_image_inner.first_layer;
                    src_subresource.mip_level += src_image_inner.first_mipmap_level;

                    let mut dst_subresource = dst_subresource.clone();
                    dst_subresource.array_layers.start += dst_image_inner.first_layer;
                    dst_subresource.array_layers.end += dst_image_inner.first_layer;
                    dst_subresource.mip_level += dst_image_inner.first_mipmap_level;

                    ash::vk::ImageCopy2 {
                        src_subresource: src_subresource.into(),
                        src_offset: ash::vk::Offset3D {
                            x: src_offset[0] as i32,
                            y: src_offset[1] as i32,
                            z: src_offset[2] as i32,
                        },
                        dst_subresource: dst_subresource.into(),
                        dst_offset: ash::vk::Offset3D {
                            x: dst_offset[0] as i32,
                            y: dst_offset[1] as i32,
                            z: dst_offset[2] as i32,
                        },
                        extent: ash::vk::Extent3D {
                            width: extent[0],
                            height: extent[1],
                            depth: extent[2],
                        },
                        ..Default::default()
                    }
                })
                .collect();

            let copy_image_info = ash::vk::CopyImageInfo2 {
                src_image: src_image_inner.image.internal_object(),
                src_image_layout: src_image_layout.into(),
                dst_image: dst_image_inner.image.internal_object(),
                dst_image_layout: dst_image_layout.into(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                ..Default::default()
            };

            if self.device.api_version() >= Version::V1_3 {
                fns.v1_3.cmd_copy_image2(self.handle, &copy_image_info);
            } else {
                fns.khr_copy_commands2
                    .cmd_copy_image2_khr(self.handle, &copy_image_info);
            }
        } else {
            let regions: SmallVec<[_; 8]> = regions
                .into_iter()
                .map(|region| {
                    let &ImageCopy {
                        ref src_subresource,
                        src_offset,
                        ref dst_subresource,
                        dst_offset,
                        extent,
                        _ne: _,
                    } = region;

                    let mut src_subresource = src_subresource.clone();
                    src_subresource.array_layers.start += src_image_inner.first_layer;
                    src_subresource.array_layers.end += src_image_inner.first_layer;
                    src_subresource.mip_level += src_image_inner.first_mipmap_level;

                    let mut dst_subresource = dst_subresource.clone();
                    dst_subresource.array_layers.start += dst_image_inner.first_layer;
                    dst_subresource.array_layers.end += dst_image_inner.first_layer;
                    dst_subresource.mip_level += dst_image_inner.first_mipmap_level;

                    ash::vk::ImageCopy {
                        src_subresource: src_subresource.into(),
                        src_offset: ash::vk::Offset3D {
                            x: src_offset[0] as i32,
                            y: src_offset[1] as i32,
                            z: src_offset[2] as i32,
                        },
                        dst_subresource: dst_subresource.into(),
                        dst_offset: ash::vk::Offset3D {
                            x: dst_offset[0] as i32,
                            y: dst_offset[1] as i32,
                            z: dst_offset[2] as i32,
                        },
                        extent: ash::vk::Extent3D {
                            width: extent[0],
                            height: extent[1],
                            depth: extent[2],
                        },
                    }
                })
                .collect();

            fns.v1_0.cmd_copy_image(
                self.handle,
                src_image_inner.image.internal_object(),
                src_image_layout.into(),
                dst_image_inner.image.internal_object(),
                dst_image_layout.into(),
                regions.len() as u32,
                regions.as_ptr(),
            );
        }
    }

    /// Calls `vkCmdCopyBufferToImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer_to_image(
        &mut self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo,
    ) {
        let &CopyBufferToImageInfo {
            ref src_buffer,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = copy_buffer_to_image_info;

        if regions.is_empty() {
            return;
        }

        let src_buffer_inner = src_buffer.inner();
        let dst_image_inner = dst_image.inner();

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3
            || self.device.enabled_extensions().khr_copy_commands2
        {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &BufferImageCopy {
                        buffer_offset,
                        buffer_row_length,
                        buffer_image_height,
                        ref image_subresource,
                        image_offset,
                        image_extent,
                        _ne: _,
                    } = region;

                    let mut image_subresource = image_subresource.clone();
                    image_subresource.array_layers.start += dst_image_inner.first_layer;
                    image_subresource.array_layers.end += dst_image_inner.first_layer;
                    image_subresource.mip_level += dst_image_inner.first_mipmap_level;

                    ash::vk::BufferImageCopy2 {
                        buffer_offset: buffer_offset + src_buffer_inner.offset,
                        buffer_row_length,
                        buffer_image_height,
                        image_subresource: image_subresource.into(),
                        image_offset: ash::vk::Offset3D {
                            x: image_offset[0] as i32,
                            y: image_offset[1] as i32,
                            z: image_offset[2] as i32,
                        },
                        image_extent: ash::vk::Extent3D {
                            width: image_extent[0],
                            height: image_extent[1],
                            depth: image_extent[2],
                        },
                        ..Default::default()
                    }
                })
                .collect();

            let copy_buffer_to_image_info = ash::vk::CopyBufferToImageInfo2 {
                src_buffer: src_buffer_inner.buffer.internal_object(),
                dst_image: dst_image_inner.image.internal_object(),
                dst_image_layout: dst_image_layout.into(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                ..Default::default()
            };

            if self.device.api_version() >= Version::V1_3 {
                fns.v1_3
                    .cmd_copy_buffer_to_image2(self.handle, &copy_buffer_to_image_info);
            } else {
                fns.khr_copy_commands2
                    .cmd_copy_buffer_to_image2_khr(self.handle, &copy_buffer_to_image_info);
            }
        } else {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &BufferImageCopy {
                        buffer_offset,
                        buffer_row_length,
                        buffer_image_height,
                        ref image_subresource,
                        image_offset,
                        image_extent,
                        _ne: _,
                    } = region;

                    let mut image_subresource = image_subresource.clone();
                    image_subresource.array_layers.start += dst_image_inner.first_layer;
                    image_subresource.array_layers.end += dst_image_inner.first_layer;
                    image_subresource.mip_level += dst_image_inner.first_mipmap_level;

                    ash::vk::BufferImageCopy {
                        buffer_offset: buffer_offset + src_buffer_inner.offset,
                        buffer_row_length,
                        buffer_image_height,
                        image_subresource: image_subresource.into(),
                        image_offset: ash::vk::Offset3D {
                            x: image_offset[0] as i32,
                            y: image_offset[1] as i32,
                            z: image_offset[2] as i32,
                        },
                        image_extent: ash::vk::Extent3D {
                            width: image_extent[0],
                            height: image_extent[1],
                            depth: image_extent[2],
                        },
                    }
                })
                .collect();

            fns.v1_0.cmd_copy_buffer_to_image(
                self.handle,
                src_buffer_inner.buffer.internal_object(),
                dst_image_inner.image.internal_object(),
                dst_image_layout.into(),
                regions.len() as u32,
                regions.as_ptr(),
            );
        }
    }

    /// Calls `vkCmdCopyImageToBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image_to_buffer(
        &mut self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo,
    ) {
        let &CopyImageToBufferInfo {
            ref src_image,
            src_image_layout,
            ref dst_buffer,
            ref regions,
            _ne: _,
        } = copy_image_to_buffer_info;

        if regions.is_empty() {
            return;
        }

        let src_image_inner = src_image.inner();
        let dst_buffer_inner = dst_buffer.inner();

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3
            || self.device.enabled_extensions().khr_copy_commands2
        {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &BufferImageCopy {
                        buffer_offset,
                        buffer_row_length,
                        buffer_image_height,
                        ref image_subresource,
                        image_offset,
                        image_extent,
                        _ne: _,
                    } = region;

                    let mut image_subresource = image_subresource.clone();
                    image_subresource.array_layers.start += src_image_inner.first_layer;
                    image_subresource.array_layers.end += src_image_inner.first_layer;
                    image_subresource.mip_level += src_image_inner.first_mipmap_level;

                    ash::vk::BufferImageCopy2 {
                        buffer_offset: buffer_offset + dst_buffer_inner.offset,
                        buffer_row_length,
                        buffer_image_height,
                        image_subresource: image_subresource.into(),
                        image_offset: ash::vk::Offset3D {
                            x: image_offset[0] as i32,
                            y: image_offset[1] as i32,
                            z: image_offset[2] as i32,
                        },
                        image_extent: ash::vk::Extent3D {
                            width: image_extent[0],
                            height: image_extent[1],
                            depth: image_extent[2],
                        },
                        ..Default::default()
                    }
                })
                .collect();

            let copy_image_to_buffer_info = ash::vk::CopyImageToBufferInfo2 {
                src_image: src_image_inner.image.internal_object(),
                src_image_layout: src_image_layout.into(),
                dst_buffer: dst_buffer_inner.buffer.internal_object(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                ..Default::default()
            };

            if self.device.api_version() >= Version::V1_3 {
                fns.v1_3
                    .cmd_copy_image_to_buffer2(self.handle, &copy_image_to_buffer_info);
            } else {
                fns.khr_copy_commands2
                    .cmd_copy_image_to_buffer2_khr(self.handle, &copy_image_to_buffer_info);
            }
        } else {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &BufferImageCopy {
                        buffer_offset,
                        buffer_row_length,
                        buffer_image_height,
                        ref image_subresource,
                        image_offset,
                        image_extent,
                        _ne: _,
                    } = region;
                    let mut image_subresource = image_subresource.clone();
                    image_subresource.array_layers.start += src_image_inner.first_layer;
                    image_subresource.array_layers.end += src_image_inner.first_layer;
                    image_subresource.mip_level += src_image_inner.first_mipmap_level;

                    ash::vk::BufferImageCopy {
                        buffer_offset: buffer_offset + dst_buffer_inner.offset,
                        buffer_row_length,
                        buffer_image_height,
                        image_subresource: image_subresource.into(),
                        image_offset: ash::vk::Offset3D {
                            x: image_offset[0] as i32,
                            y: image_offset[1] as i32,
                            z: image_offset[2] as i32,
                        },
                        image_extent: ash::vk::Extent3D {
                            width: image_extent[0],
                            height: image_extent[1],
                            depth: image_extent[2],
                        },
                    }
                })
                .collect();

            fns.v1_0.cmd_copy_image_to_buffer(
                self.handle,
                src_image_inner.image.internal_object(),
                src_image_layout.into(),
                dst_buffer_inner.buffer.internal_object(),
                regions.len() as u32,
                regions.as_ptr(),
            );
        }
    }

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer(&mut self, fill_buffer_info: &FillBufferInfo) {
        let &FillBufferInfo {
            data,
            ref dst_buffer,
            dst_offset,
            size,
            _ne: _,
        } = fill_buffer_info;

        let dst_buffer_inner = dst_buffer.inner();

        let fns = self.device.fns();
        fns.v1_0.cmd_fill_buffer(
            self.handle,
            dst_buffer_inner.buffer.internal_object(),
            dst_offset,
            size,
            data,
        );
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    #[inline]
    pub unsafe fn update_buffer<D>(
        &mut self,
        data: &D,
        dst_buffer: &dyn BufferAccess,
        dst_offset: DeviceSize,
    ) where
        D: BufferContents + ?Sized,
    {
        let dst_buffer_inner = dst_buffer.inner();

        let fns = self.device.fns();
        fns.v1_0.cmd_update_buffer(
            self.handle,
            dst_buffer_inner.buffer.internal_object(),
            dst_buffer_inner.offset + dst_offset,
            size_of_val(data) as DeviceSize,
            data.as_bytes().as_ptr() as *const _,
        );
    }
}

/// Parameters to copy data from a buffer to another buffer.
///
/// The fields of `regions` represent bytes.
#[derive(Clone, Debug)]
pub struct CopyBufferInfo {
    /// The buffer to copy from.
    ///
    /// There is no default value.
    pub src_buffer: Arc<dyn BufferAccess>,

    /// The buffer to copy to.
    ///
    /// There is no default value.
    pub dst_buffer: Arc<dyn BufferAccess>,

    /// The regions of both buffers to copy between, specified in bytes.
    ///
    /// The default value is a single region, with zero offsets and a `size` equal to the smallest
    /// of the two buffers.
    pub regions: SmallVec<[BufferCopy; 1]>,

    pub _ne: crate::NonExhaustive,
}

impl CopyBufferInfo {
    /// Returns a `CopyBufferInfo` with the specified `src_buffer` and `dst_buffer`.
    #[inline]
    pub fn buffers(src_buffer: Arc<dyn BufferAccess>, dst_buffer: Arc<dyn BufferAccess>) -> Self {
        let region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: min(src_buffer.size(), dst_buffer.size()),
            ..Default::default()
        };

        Self {
            src_buffer,
            dst_buffer,
            regions: smallvec![region],
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to copy data from a buffer to another buffer, with type information.
///
/// The fields of `regions` represent elements of `T`.
#[derive(Clone, Debug)]
pub struct CopyBufferInfoTyped<S, D, T>
where
    S: TypedBufferAccess<Content = [T]>,
    D: TypedBufferAccess<Content = [T]>,
{
    /// The buffer to copy from.
    ///
    /// There is no default value.
    pub src_buffer: Arc<S>,

    /// The buffer to copy to.
    ///
    /// There is no default value.
    pub dst_buffer: Arc<D>,

    /// The regions of both buffers to copy between, specified in elements of `T`.
    ///
    /// The default value is a single region, with zero offsets and a `size` equal to the smallest
    /// of the two buffers.
    pub regions: SmallVec<[BufferCopy; 1]>,

    pub _ne: crate::NonExhaustive,
}

impl<S, D, T> CopyBufferInfoTyped<S, D, T>
where
    S: TypedBufferAccess<Content = [T]>,
    D: TypedBufferAccess<Content = [T]>,
{
    /// Returns a `CopyBufferInfoTyped` with the specified `src_buffer` and `dst_buffer`.
    #[inline]
    pub fn buffers(src_buffer: Arc<S>, dst_buffer: Arc<D>) -> Self {
        let region = BufferCopy {
            size: min(src_buffer.len(), dst_buffer.len()),
            ..Default::default()
        };

        Self {
            src_buffer,
            dst_buffer,
            regions: smallvec![region],
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl<S, D, T> From<CopyBufferInfoTyped<S, D, T>> for CopyBufferInfo
where
    S: TypedBufferAccess<Content = [T]> + 'static,
    D: TypedBufferAccess<Content = [T]> + 'static,
{
    #[inline]
    fn from(typed: CopyBufferInfoTyped<S, D, T>) -> Self {
        let CopyBufferInfoTyped {
            src_buffer,
            dst_buffer,
            mut regions,
            _ne: _,
        } = typed;

        for region in &mut regions {
            region.src_offset *= size_of::<T>() as DeviceSize;
            region.dst_offset *= size_of::<T>() as DeviceSize;
            region.size *= size_of::<T>() as DeviceSize;
        }

        Self {
            src_buffer,
            dst_buffer,
            regions,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A region of data to copy between buffers.
#[derive(Clone, Debug)]
pub struct BufferCopy {
    /// The offset in bytes or elements from the start of `src_buffer` that copying will
    /// start from.
    ///
    /// The default value is `0`.
    pub src_offset: DeviceSize,

    /// The offset in bytes or elements from the start of `dst_buffer` that copying will
    /// start from.
    ///
    /// The default value is `0`.
    pub dst_offset: DeviceSize,

    /// The number of bytes or elements to copy.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: DeviceSize,

    pub _ne: crate::NonExhaustive,
}

impl Default for BufferCopy {
    #[inline]
    fn default() -> Self {
        Self {
            src_offset: 0,
            dst_offset: 0,
            size: 0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to copy data from an image to another image.
#[derive(Clone, Debug)]
pub struct CopyImageInfo {
    /// The image to copy from.
    ///
    /// There is no default value.
    pub src_image: Arc<dyn ImageAccess>,

    /// The layout used for `src_image` during the copy operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferSrcOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferSrcOptimal`].
    pub src_image_layout: ImageLayout,

    /// The image to copy to.
    ///
    /// There is no default value.
    pub dst_image: Arc<dyn ImageAccess>,

    /// The layout used for `dst_image` during the copy operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub dst_image_layout: ImageLayout,

    /// The regions of both images to copy between.
    ///
    /// The default value is a single region, covering the first mip level, and the smallest of the
    /// array layers and extent of the two images. All aspects of each image are selected, or
    /// `plane0` if the image is multi-planar.
    pub regions: SmallVec<[ImageCopy; 1]>,

    pub _ne: crate::NonExhaustive,
}

impl CopyImageInfo {
    /// Returns a `CopyImageInfo` with the specified `src_image` and `dst_image`.
    #[inline]
    pub fn images(src_image: Arc<dyn ImageAccess>, dst_image: Arc<dyn ImageAccess>) -> Self {
        let min_array_layers = src_image
            .dimensions()
            .array_layers()
            .min(dst_image.dimensions().array_layers());
        let region = ImageCopy {
            src_subresource: ImageSubresourceLayers {
                array_layers: 0..min_array_layers,
                ..src_image.subresource_layers()
            },
            dst_subresource: ImageSubresourceLayers {
                array_layers: 0..min_array_layers,
                ..dst_image.subresource_layers()
            },
            extent: {
                let src_extent = src_image.dimensions().width_height_depth();
                let dst_extent = dst_image.dimensions().width_height_depth();

                [
                    src_extent[0].min(dst_extent[0]),
                    src_extent[1].min(dst_extent[1]),
                    src_extent[2].min(dst_extent[2]),
                ]
            },
            ..Default::default()
        };

        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: smallvec![region],
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A region of data to copy between images.
#[derive(Clone, Debug)]
pub struct ImageCopy {
    /// The subresource of `src_image` to copy from.
    ///
    /// The default value is empty, which must be overridden.
    pub src_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of `src_image` that copying will start from.
    ///
    /// The default value is `[0; 3]`.
    pub src_offset: [u32; 3],

    /// The subresource of `dst_image` to copy to.
    ///
    /// The default value is empty, which must be overridden.
    pub dst_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of `dst_image` that copying will start from.
    ///
    /// The default value is `[0; 3]`.
    pub dst_offset: [u32; 3],

    /// The extent of texels to copy.
    ///
    /// The default value is `[0; 3]`, which must be overridden.
    pub extent: [u32; 3],

    pub _ne: crate::NonExhaustive,
}

impl Default for ImageCopy {
    #[inline]
    fn default() -> Self {
        Self {
            src_subresource: ImageSubresourceLayers {
                aspects: ImageAspects::none(),
                mip_level: 0,
                array_layers: 0..0,
            },
            src_offset: [0; 3],
            dst_subresource: ImageSubresourceLayers {
                aspects: ImageAspects::none(),
                mip_level: 0,
                array_layers: 0..0,
            },
            dst_offset: [0; 3],
            extent: [0; 3],
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to copy data from a buffer to an image.
#[derive(Clone, Debug)]
pub struct CopyBufferToImageInfo {
    /// The buffer to copy from.
    ///
    /// There is no default value.
    pub src_buffer: Arc<dyn BufferAccess>,

    /// The image to copy to.
    ///
    /// There is no default value.
    pub dst_image: Arc<dyn ImageAccess>,

    /// The layout used for `dst_image` during the copy operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub dst_image_layout: ImageLayout,

    /// The regions of the buffer and image to copy between.
    ///
    /// The default value is a single region, covering all of the buffer and the first mip level of
    /// the image. All aspects of the image are selected, or `plane0` if the image is multi-planar.
    pub regions: SmallVec<[BufferImageCopy; 1]>,

    pub _ne: crate::NonExhaustive,
}

impl CopyBufferToImageInfo {
    /// Returns a `CopyBufferToImageInfo` with the specified `src_buffer` and
    /// `dst_image`.
    #[inline]
    pub fn buffer_image(
        src_buffer: Arc<dyn BufferAccess>,
        dst_image: Arc<dyn ImageAccess>,
    ) -> Self {
        let region = BufferImageCopy {
            image_subresource: dst_image.subresource_layers(),
            image_extent: dst_image.dimensions().width_height_depth(),
            ..Default::default()
        };

        Self {
            src_buffer,
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: smallvec![region],
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to copy data from an image to a buffer.
#[derive(Clone, Debug)]
pub struct CopyImageToBufferInfo {
    /// The image to copy from.
    ///
    /// There is no default value.
    pub src_image: Arc<dyn ImageAccess>,

    /// The layout used for `src_image` during the copy operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferSrcOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferSrcOptimal`].
    pub src_image_layout: ImageLayout,

    /// The buffer to copy to.
    ///
    /// There is no default value.
    pub dst_buffer: Arc<dyn BufferAccess>,

    /// The regions of the image and buffer to copy between.
    ///
    /// The default value is a single region, covering all of the buffer and the first mip level of
    /// the image. All aspects of the image are selected, or `plane0` if the image is multi-planar.
    pub regions: SmallVec<[BufferImageCopy; 1]>,

    pub _ne: crate::NonExhaustive,
}

impl CopyImageToBufferInfo {
    /// Returns a `CopyImageToBufferInfo` with the specified `src_image` and
    /// `dst_buffer`.
    #[inline]
    pub fn image_buffer(
        src_image: Arc<dyn ImageAccess>,
        dst_buffer: Arc<dyn BufferAccess>,
    ) -> Self {
        let region = BufferImageCopy {
            image_subresource: src_image.subresource_layers(),
            image_extent: src_image.dimensions().width_height_depth(),
            ..Default::default()
        };

        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_buffer,
            regions: smallvec![region],
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A region of data to copy between a buffer and an image.
#[derive(Clone, Debug)]
pub struct BufferImageCopy {
    /// The offset in bytes from the start of the buffer that copying will start from.
    ///
    /// The default value is `0`.
    pub buffer_offset: DeviceSize,

    /// The number of texels between successive rows of image data in the buffer.
    ///
    /// If set to `0`, the width of the image is used.
    ///
    /// The default value is `0`.
    pub buffer_row_length: u32,

    /// The number of rows between successive depth slices of image data in the buffer.
    ///
    /// If set to `0`, the height of the image is used.
    ///
    /// The default value is `0`.
    pub buffer_image_height: u32,

    /// The subresource of the image to copy from/to.
    ///
    /// The default value is empty, which must be overridden.
    pub image_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of the image that copying will start from.
    ///
    /// The default value is `[0; 3]`.
    pub image_offset: [u32; 3],

    /// The extent of texels in the image to copy.
    ///
    /// The default value is `[0; 3]`, which must be overridden.
    pub image_extent: [u32; 3],

    pub _ne: crate::NonExhaustive,
}

impl Default for BufferImageCopy {
    #[inline]
    fn default() -> Self {
        Self {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: ImageSubresourceLayers {
                aspects: ImageAspects::none(),
                mip_level: 0,
                array_layers: 0..0,
            },
            image_offset: [0; 3],
            image_extent: [0; 3],
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl BufferImageCopy {
    // Following
    // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap20.html#copies-buffers-images-addressing
    pub(crate) fn buffer_copy_size(&self, format: Format) -> DeviceSize {
        let &BufferImageCopy {
            buffer_offset: _,
            mut buffer_row_length,
            mut buffer_image_height,
            ref image_subresource,
            image_offset: _,
            mut image_extent,
            _ne: _,
        } = self;

        if buffer_row_length == 0 {
            buffer_row_length = image_extent[0];
        }

        if buffer_image_height == 0 {
            buffer_image_height = image_extent[1];
        }

        // Scale down from texels to texel blocks, rounding up if needed.
        let block_extent = format.block_extent();
        buffer_row_length = (buffer_row_length + block_extent[0] - 1) / block_extent[0];
        buffer_image_height = (buffer_image_height + block_extent[1] - 1) / block_extent[1];

        for i in 0..3 {
            image_extent[i] = (image_extent[i] + block_extent[i] - 1) / block_extent[i];
        }

        // Only one of these is greater than 1, take the greater number.
        image_extent[2] = max(
            image_extent[2],
            image_subresource.array_layers.end - image_subresource.array_layers.start,
        );

        let blocks_to_last_slice = (image_extent[2] as DeviceSize - 1)
            * buffer_image_height as DeviceSize
            * buffer_row_length as DeviceSize;
        let blocks_to_last_row =
            (image_extent[1] as DeviceSize - 1) * buffer_row_length as DeviceSize;
        let num_blocks = blocks_to_last_slice + blocks_to_last_row + image_extent[0] as DeviceSize;

        // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkBufferImageCopy.html#_description
        let block_size = if image_subresource.aspects.stencil {
            1
        } else if image_subresource.aspects.depth {
            match format {
                Format::D16_UNORM | Format::D16_UNORM_S8_UINT => 2,
                Format::D32_SFLOAT
                | Format::D32_SFLOAT_S8_UINT
                | Format::X8_D24_UNORM_PACK32
                | Format::D24_UNORM_S8_UINT => 4,
                _ => unreachable!(),
            }
        } else {
            format.block_size().unwrap()
        };

        num_blocks * block_size
    }
}

/// Parameters to fill a region of a buffer with repeated copies of a value.
#[derive(Clone, Debug)]
pub struct FillBufferInfo {
    /// The data to fill with.
    ///
    /// The default value is `0`.
    pub data: u32,

    /// The buffer to fill.
    ///
    /// There is no default value.
    pub dst_buffer: Arc<dyn BufferAccess>,

    /// The offset in bytes from the start of `dst_buffer` that filling will start from.
    ///
    /// This must be a multiple of 4.
    ///
    /// The default value is `0`.
    pub dst_offset: DeviceSize,

    /// The number of bytes to fill.
    ///
    /// This must be a multiple of 4.
    ///
    /// The default value is the size of `dst_buffer`,
    /// rounded down to the nearest multiple of 4.
    pub size: DeviceSize,

    pub _ne: crate::NonExhaustive,
}

impl FillBufferInfo {
    /// Returns a `FillBufferInfo` with the specified `dst_buffer`.
    #[inline]
    pub fn dst_buffer(dst_buffer: Arc<dyn BufferAccess>) -> Self {
        let size = dst_buffer.size() & !3;

        Self {
            data: 0,
            dst_buffer,
            dst_offset: 0,
            size,
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::Format;

    #[test]
    fn test_required_len_for_format() {
        // issue #1292
        assert_eq!(
            required_size_for_format(Format::BC1_RGB_UNORM_BLOCK, [2048, 2048, 1], 1),
            2097152
        );
        // other test cases
        assert_eq!(
            required_size_for_format(Format::R8G8B8A8_UNORM, [2048, 2048, 1], 1),
            16777216
        );
        assert_eq!(
            required_size_for_format(Format::R4G4_UNORM_PACK8, [512, 512, 1], 1),
            262144
        );
        assert_eq!(
            required_size_for_format(Format::R8G8B8_USCALED, [512, 512, 1], 1),
            786432
        );
        assert_eq!(
            required_size_for_format(Format::R32G32_UINT, [512, 512, 1], 1),
            2097152
        );
        assert_eq!(
            required_size_for_format(Format::R32G32_UINT, [512, 512, 1], 1),
            2097152
        );
        assert_eq!(
            required_size_for_format(Format::ASTC_8x8_UNORM_BLOCK, [512, 512, 1], 1),
            65536
        );
        assert_eq!(
            required_size_for_format(Format::ASTC_12x12_SRGB_BLOCK, [512, 512, 1], 1),
            29584
        );
    }
}
