// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    BlitImageInfo, BufferCopy, BufferImageCopy, CommandBufferBuilder, CopyBufferInfo,
    CopyBufferToImageInfo, CopyError, CopyErrorResource, CopyImageInfo, CopyImageToBufferInfo,
    ImageCopy, ResolveImageInfo,
};
use crate::{
    buffer::BufferUsage,
    command_buffer::{
        allocator::CommandBufferAllocator, ImageBlit, ImageResolve, ResourceInCommand,
        ResourceUseRef,
    },
    device::{DeviceOwned, QueueFlags},
    format::{Format, FormatFeatures, NumericType},
    image::{
        ImageAccess, ImageAspects, ImageDimensions, ImageLayout, ImageSubresourceLayers,
        ImageSubresourceRange, ImageType, ImageUsage, SampleCount, SampleCounts,
    },
    sampler::Filter,
    sync::PipelineStageAccess,
    DeviceSize, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::cmp::{max, min};

impl<L, A> CommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Copies data from a buffer to another buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `src_buffer` or `dst_buffer` were not created from the same device
    ///   as `self`.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers
    ///   that are accessed by the command.
    #[inline]
    pub unsafe fn copy_buffer(
        &mut self,
        copy_buffer_info: impl Into<CopyBufferInfo>,
    ) -> Result<&mut Self, CopyError> {
        let copy_buffer_info = copy_buffer_info.into();
        self.validate_copy_buffer(&copy_buffer_info)?;

        unsafe { Ok(self.copy_buffer_unchecked(copy_buffer_info)) }
    }

    fn validate_copy_buffer(&self, copy_buffer_info: &CopyBufferInfo) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdCopyBuffer2-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdCopyBuffer2-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &CopyBufferInfo {
            ref src_buffer,
            ref dst_buffer,
            ref regions,
            _ne: _,
        } = copy_buffer_info;

        // VUID-VkCopyBufferInfo2-commonparent
        assert_eq!(device, src_buffer.device());
        assert_eq!(device, dst_buffer.device());

        // VUID-VkCopyBufferInfo2-srcBuffer-00118
        if !src_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_SRC)
        {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkCopyBufferInfo2-dstBuffer-00120
        if !dst_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_DST)
        {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        let same_buffer = src_buffer.buffer() == dst_buffer.buffer();
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
                let src_range =
                    src_buffer.offset() + src_offset..src_buffer.offset() + src_offset + size;

                for (dst_region_index, dst_region) in regions.iter().enumerate() {
                    let &BufferCopy { dst_offset, .. } = dst_region;

                    let dst_range =
                        dst_buffer.offset() + dst_offset..dst_buffer.offset() + dst_offset + size;

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

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_buffer_unchecked(
        &mut self,
        copy_buffer_info: impl Into<CopyBufferInfo>,
    ) -> &mut Self {
        let copy_buffer_info = copy_buffer_info.into();
        let CopyBufferInfo {
            src_buffer,
            dst_buffer,
            regions,
            _ne: _,
        } = copy_buffer_info;

        if regions.is_empty() {
            return self;
        }

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
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
                        src_offset: src_offset + src_buffer.offset(),
                        dst_offset: dst_offset + dst_buffer.offset(),
                        size,
                        ..Default::default()
                    }
                })
                .collect();

            let copy_buffer_info = ash::vk::CopyBufferInfo2 {
                src_buffer: src_buffer.buffer().handle(),
                dst_buffer: dst_buffer.buffer().handle(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                ..Default::default()
            };

            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_copy_buffer2)(self.handle(), &copy_buffer_info);
            } else {
                (fns.khr_copy_commands2.cmd_copy_buffer2_khr)(self.handle(), &copy_buffer_info);
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
                        src_offset: src_offset + src_buffer.offset(),
                        dst_offset: dst_offset + dst_buffer.offset(),
                        size,
                    }
                })
                .collect();

            (fns.v1_0.cmd_copy_buffer)(
                self.handle(),
                src_buffer.buffer().handle(),
                dst_buffer.buffer().handle(),
                regions.len() as u32,
                regions.as_ptr(),
            );
        }

        let command_index = self.next_command_index;
        let command_name = "copy_buffer";
        let src_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Source,
            secondary_use_ref: None,
        };
        let dst_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Destination,
            secondary_use_ref: None,
        };

        for region in regions {
            let BufferCopy {
                src_offset,
                dst_offset,
                size,
                _ne: _,
            } = region;

            let mut src_range = src_offset..src_offset + size;
            src_range.start += src_buffer.offset();
            src_range.end += src_buffer.offset();
            self.resources_usage_state.record_buffer_access(
                &src_use_ref,
                src_buffer.buffer(),
                src_range,
                PipelineStageAccess::Copy_TransferRead,
            );

            let mut dst_range = dst_offset..dst_offset + size;
            dst_range.start += dst_buffer.offset();
            dst_range.end += dst_buffer.offset();
            self.resources_usage_state.record_buffer_access(
                &dst_use_ref,
                dst_buffer.buffer(),
                dst_range,
                PipelineStageAccess::Copy_TransferWrite,
            );
        }

        self.resources.push(Box::new(src_buffer));
        self.resources.push(Box::new(dst_buffer));

        self.next_command_index += 1;
        self
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
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn copy_image(
        &mut self,
        copy_image_info: CopyImageInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_copy_image(&copy_image_info)?;

        unsafe { Ok(self.copy_image_unchecked(copy_image_info)) }
    }

    fn validate_copy_image(&self, copy_image_info: &CopyImageInfo) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdCopyImage2-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdCopyImage2-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &CopyImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = copy_image_info;

        // VUID-VkCopyImageInfo2-srcImageLayout-parameter
        src_image_layout.validate_device(device)?;

        // VUID-VkCopyImageInfo2-dstImageLayout-parameter
        dst_image_layout.validate_device(device)?;

        // VUID-VkCopyImageInfo2-commonparent
        assert_eq!(device, src_image.device());
        assert_eq!(device, dst_image.device());

        let copy_2d_3d_supported =
            device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1;
        let src_image_inner = src_image.inner();
        let dst_image_inner = dst_image.inner();
        let mut src_image_aspects = src_image.format().aspects();
        let mut dst_image_aspects = dst_image.format().aspects();

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            // VUID-VkCopyImageInfo2-srcImage-01995
            if !src_image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_SRC)
            {
                return Err(CopyError::MissingFormatFeature {
                    resource: CopyErrorResource::Source,
                    format_feature: "transfer_src",
                });
            }

            // VUID-VkCopyImageInfo2-dstImage-01996
            if !dst_image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_DST)
            {
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

        if !(src_image_aspects.intersects(ImageAspects::COLOR)
            || dst_image_aspects.intersects(ImageAspects::COLOR))
        {
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

        let extent_alignment = match queue_family_properties.min_image_transfer_granularity {
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
                    granularity(
                        src_image.format().block_extent(),
                        src_image_aspects.intersects(ImageAspects::PLANE_0),
                    ),
                    granularity(
                        dst_image.format().block_extent(),
                        dst_image_aspects.intersects(ImageAspects::PLANE_0),
                    ),
                ))
            }
        };

        if src_image_aspects.intersects(ImageAspects::PLANE_0) {
            // VUID-VkCopyImageInfo2-srcImage-01552
            // VUID-VkCopyImageInfo2-srcImage-01553
            src_image_aspects -= ImageAspects::COLOR;
        }

        if dst_image_aspects.intersects(ImageAspects::PLANE_0) {
            // VUID-VkCopyImageInfo2-dstImage-01554
            // VUID-VkCopyImageInfo2-dstImage-01555
            dst_image_aspects -= ImageAspects::COLOR;
        }

        let mut src_image_aspects_used = ImageAspects::empty();
        let mut dst_image_aspects_used = ImageAspects::empty();
        let is_same_image = src_image_inner == dst_image_inner;
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
                                     image_aspects: ImageAspects,
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

                // VUID-VkImageSubresourceLayers-aspectMask-parameter
                subresource.aspects.validate_device(device)?;

                // VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask
                assert!(!subresource.aspects.is_empty());

                // VUID-VkCopyImageInfo2-aspectMask-00142
                // VUID-VkCopyImageInfo2-aspectMask-00143
                if !image_aspects.contains(subresource.aspects) {
                    return Err(CopyError::AspectsNotAllowed {
                        resource,
                        region_index,
                        aspects: subresource.aspects,
                        allowed_aspects: image_aspects,
                    });
                }

                let (subresource_format, subresource_extent) =
                    if image_aspects.intersects(ImageAspects::PLANE_0) {
                        // VUID-VkCopyImageInfo2-srcImage-01552
                        // VUID-VkCopyImageInfo2-srcImage-01553
                        // VUID-VkCopyImageInfo2-dstImage-01554
                        // VUID-VkCopyImageInfo2-dstImage-01555
                        if subresource.aspects.count() != 1 {
                            return Err(CopyError::MultipleAspectsNotAllowed {
                                resource,
                                region_index,
                                aspects: subresource.aspects,
                            });
                        }

                        if subresource.aspects.intersects(ImageAspects::PLANE_0) {
                            (
                                image.format().planes()[0],
                                image.dimensions().width_height_depth(),
                            )
                        } else if subresource.aspects.intersects(ImageAspects::PLANE_1) {
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

            src_image_aspects_used |= src_subresource.aspects;
            dst_image_aspects_used |= dst_subresource.aspects;

            let (src_subresource_format, src_subresource_extent) = check_subresource(
                CopyErrorResource::Source,
                src_image,
                src_image_aspects,
                src_subresource,
            )?;
            let (dst_subresource_format, dst_subresource_extent) = check_subresource(
                CopyErrorResource::Destination,
                dst_image,
                dst_image_aspects,
                dst_subresource,
            )?;

            if !(src_image_aspects.intersects(ImageAspects::PLANE_0)
                || dst_image_aspects.intersects(ImageAspects::PLANE_0))
            {
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
                if is_same_image {
                    let src_region_index = region_index;
                    let src_subresource_axes = [
                        src_subresource.mip_level..src_subresource.mip_level + 1,
                        src_subresource.array_layers.start..src_subresource.array_layers.end,
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
                        if src_image_aspects.intersects(ImageAspects::PLANE_0)
                            && src_subresource.aspects != dst_subresource.aspects
                        {
                            continue;
                        }

                        let dst_subresource_axes = [
                            dst_subresource.mip_level..dst_subresource.mip_level + 1,
                            src_subresource.array_layers.start..src_subresource.array_layers.end,
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
                if is_same_image {
                    let src_region_index = region_index;
                    let src_axes = [
                        src_subresource.mip_level..src_subresource.mip_level + 1,
                        src_subresource.array_layers.start..src_subresource.array_layers.end,
                    ];

                    for (dst_region_index, dst_region) in regions.iter().enumerate() {
                        let &ImageCopy {
                            ref dst_subresource,
                            dst_offset: _,
                            ..
                        } = dst_region;

                        if src_image_aspects.intersects(ImageAspects::PLANE_0)
                            && src_subresource.aspects != dst_subresource.aspects
                        {
                            continue;
                        }

                        let dst_axes = [
                            dst_subresource.mip_level..dst_subresource.mip_level + 1,
                            src_subresource.array_layers.start..src_subresource.array_layers.end,
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

        // VUID-VkCopyImageInfo2-aspect-06662
        if !(src_image_aspects_used - ImageAspects::STENCIL).is_empty()
            && !src_image.usage().intersects(ImageUsage::TRANSFER_SRC)
        {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkCopyImageInfo2-aspect-06663
        if !(dst_image_aspects_used - ImageAspects::STENCIL).is_empty()
            && !dst_image.usage().intersects(ImageUsage::TRANSFER_DST)
        {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        // VUID-VkCopyImageInfo2-aspect-06664
        if src_image_aspects_used.intersects(ImageAspects::STENCIL)
            && !src_image
                .stencil_usage()
                .intersects(ImageUsage::TRANSFER_SRC)
        {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkCopyImageInfo2-aspect-06665
        if dst_image_aspects_used.intersects(ImageAspects::STENCIL)
            && !dst_image
                .stencil_usage()
                .intersects(ImageUsage::TRANSFER_DST)
        {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
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

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_image_unchecked(&mut self, copy_image_info: CopyImageInfo) -> &mut Self {
        let CopyImageInfo {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions,
            _ne: _,
        } = copy_image_info;

        if regions.is_empty() {
            return self;
        }

        let src_image_inner = src_image.inner();
        let dst_image_inner = dst_image.inner();

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &ImageCopy {
                        ref src_subresource,
                        src_offset,
                        ref dst_subresource,
                        dst_offset,
                        extent,
                        _ne: _,
                    } = region;

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
                src_image: src_image_inner.handle(),
                src_image_layout: src_image_layout.into(),
                dst_image: dst_image_inner.handle(),
                dst_image_layout: dst_image_layout.into(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                ..Default::default()
            };

            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_copy_image2)(self.handle(), &copy_image_info);
            } else {
                (fns.khr_copy_commands2.cmd_copy_image2_khr)(self.handle(), &copy_image_info);
            }
        } else {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &ImageCopy {
                        ref src_subresource,
                        src_offset,
                        ref dst_subresource,
                        dst_offset,
                        extent,
                        _ne: _,
                    } = region;

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

            (fns.v1_0.cmd_copy_image)(
                self.handle(),
                src_image_inner.handle(),
                src_image_layout.into(),
                dst_image_inner.handle(),
                dst_image_layout.into(),
                regions.len() as u32,
                regions.as_ptr(),
            );
        }

        let command_index = self.next_command_index;
        let command_name = "copy_image";
        let src_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Source,
            secondary_use_ref: None,
        };
        let dst_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Destination,
            secondary_use_ref: None,
        };

        for region in regions {
            let ImageCopy {
                src_subresource,
                src_offset: _,
                dst_subresource,
                dst_offset: _,
                extent: _,
                _ne: _,
            } = region;

            self.resources_usage_state.record_image_access(
                &src_use_ref,
                src_image_inner,
                ImageSubresourceRange::from(src_subresource),
                PipelineStageAccess::Copy_TransferRead,
                src_image_layout,
            );

            self.resources_usage_state.record_image_access(
                &dst_use_ref,
                dst_image_inner,
                ImageSubresourceRange::from(dst_subresource),
                PipelineStageAccess::Copy_TransferWrite,
                dst_image_layout,
            );
        }

        self.resources.push(Box::new(src_image));
        self.resources.push(Box::new(dst_image));

        self.next_command_index += 1;
        self
    }

    /// Copies from a buffer to an image.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers and images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn copy_buffer_to_image(
        &mut self,
        copy_buffer_to_image_info: CopyBufferToImageInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_copy_buffer_to_image(&copy_buffer_to_image_info)?;

        unsafe { Ok(self.copy_buffer_to_image_unchecked(copy_buffer_to_image_info)) }
    }

    fn validate_copy_buffer_to_image(
        &self,
        copy_buffer_to_image_info: &CopyBufferToImageInfo,
    ) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdCopyBufferToImage2-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdCopyBufferToImage2-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &CopyBufferToImageInfo {
            ref src_buffer,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = copy_buffer_to_image_info;

        // VUID-VkCopyBufferToImageInfo2-dstImageLayout-parameter
        dst_image_layout.validate_device(device)?;

        // VUID-VkCopyBufferToImageInfo2-commonparent
        assert_eq!(device, src_buffer.device());
        assert_eq!(device, dst_image.device());

        let mut image_aspects = dst_image.format().aspects();

        // VUID-VkCopyBufferToImageInfo2-commandBuffer-04477
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
            && !image_aspects.intersects(ImageAspects::COLOR)
        {
            return Err(CopyError::DepthStencilNotSupportedByQueueFamily);
        }

        // VUID-VkCopyBufferToImageInfo2-srcBuffer-00174
        if !src_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_SRC)
        {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkCopyBufferToImageInfo2-dstImage-00177
        if !dst_image.usage().intersects(ImageUsage::TRANSFER_DST) {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            // VUID-VkCopyBufferToImageInfo2-dstImage-01997
            if !dst_image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_DST)
            {
                return Err(CopyError::MissingFormatFeature {
                    resource: CopyErrorResource::Destination,
                    format_feature: "transfer_dst",
                });
            }
        }

        // VUID-VkCopyBufferToImageInfo2-dstImage-00179
        if dst_image.samples() != SampleCount::Sample1 {
            return Err(CopyError::SampleCountInvalid {
                resource: CopyErrorResource::Destination,
                sample_count: dst_image.samples(),
                allowed_sample_counts: SampleCounts::SAMPLE_1,
            });
        }

        // VUID-VkCopyBufferToImageInfo2-dstImageLayout-01396
        if !matches!(
            dst_image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Destination,
                image_layout: dst_image_layout,
            });
        }

        let extent_alignment = match queue_family_properties.min_image_transfer_granularity {
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
                    dst_image.format().block_extent(),
                    image_aspects.intersects(ImageAspects::PLANE_0),
                ))
            }
        };

        if image_aspects.intersects(ImageAspects::PLANE_0) {
            // VUID-VkCopyBufferToImageInfo2-aspectMask-01560
            image_aspects -= ImageAspects::COLOR;
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
            if image_subresource.mip_level >= dst_image.mip_levels() {
                return Err(CopyError::MipLevelsOutOfRange {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    mip_levels_range_end: image_subresource.mip_level + 1,
                    image_mip_levels: dst_image.mip_levels(),
                });
            }

            // VUID-VkImageSubresourceLayers-layerCount-01700
            // VUID-VkCopyBufferToImageInfo2-baseArrayLayer-00213
            assert!(!image_subresource.array_layers.is_empty());

            // VUID-VkCopyBufferToImageInfo2-imageSubresource-01702
            // VUID-VkCopyBufferToImageInfo2-baseArrayLayer-00213
            if image_subresource.array_layers.end > dst_image.dimensions().array_layers() {
                return Err(CopyError::ArrayLayersOutOfRange {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    array_layers_range_end: image_subresource.array_layers.end,
                    image_array_layers: dst_image.dimensions().array_layers(),
                });
            }

            // VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask
            assert!(!image_subresource.aspects.is_empty());

            // VUID-VkCopyBufferToImageInfo2-aspectMask-00211
            if !image_aspects.contains(image_subresource.aspects) {
                return Err(CopyError::AspectsNotAllowed {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    aspects: image_subresource.aspects,
                    allowed_aspects: image_aspects,
                });
            }

            // VUID-VkBufferImageCopy2-aspectMask-00212
            // VUID-VkCopyBufferToImageInfo2-aspectMask-01560
            if image_subresource.aspects.count() != 1 {
                return Err(CopyError::MultipleAspectsNotAllowed {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    aspects: image_subresource.aspects,
                });
            }

            let (image_subresource_format, image_subresource_extent) =
                if image_aspects.intersects(ImageAspects::PLANE_0) {
                    if image_subresource.aspects.intersects(ImageAspects::PLANE_0) {
                        (
                            dst_image.format().planes()[0],
                            dst_image.dimensions().width_height_depth(),
                        )
                    } else if image_subresource.aspects.intersects(ImageAspects::PLANE_1) {
                        (
                            dst_image.format().planes()[1],
                            dst_image
                                .format()
                                .ycbcr_chroma_sampling()
                                .unwrap()
                                .subsampled_extent(dst_image.dimensions().width_height_depth()),
                        )
                    } else {
                        (
                            dst_image.format().planes()[2],
                            dst_image
                                .format()
                                .ycbcr_chroma_sampling()
                                .unwrap()
                                .subsampled_extent(dst_image.dimensions().width_height_depth()),
                        )
                    }
                } else {
                    (
                        dst_image.format(),
                        dst_image
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

            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkBufferImageCopy.html#_description
            let image_subresource_block_size =
                if image_subresource.aspects.intersects(ImageAspects::STENCIL) {
                    1
                } else if image_subresource.aspects.intersects(ImageAspects::DEPTH) {
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

            let buffer_offset_alignment =
                if image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
                    4
                } else {
                    let mut buffer_offset_alignment = image_subresource_block_size;

                    // VUID-VkCopyBufferToImageInfo2-commandBuffer-04052
                    // Make the alignment a multiple of 4.
                    if !queue_family_properties
                        .queue_flags
                        .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
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
            if (src_buffer.offset() + buffer_offset) % buffer_offset_alignment != 0 {
                return Err(CopyError::OffsetNotAlignedForBuffer {
                    resource: CopyErrorResource::Source,
                    region_index,
                    offset: src_buffer.offset() + buffer_offset,
                    required_alignment: buffer_offset_alignment,
                });
            }

            let buffer_copy_size = region.buffer_copy_size(image_subresource_format);

            // VUID-VkCopyBufferToImageInfo2-pRegions-00171
            if buffer_offset + buffer_copy_size > src_buffer.size() {
                return Err(CopyError::RegionOutOfBufferBounds {
                    resource: CopyErrorResource::Source,
                    region_index,
                    offset_range_end: buffer_offset + buffer_copy_size,
                    buffer_size: src_buffer.size(),
                });
            }
        }

        // VUID-VkCopyBufferToImageInfo2-pRegions-00173
        // Can't occur as long as memory aliasing isn't allowed.

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_buffer_to_image_unchecked(
        &mut self,
        copy_buffer_to_image_info: CopyBufferToImageInfo,
    ) -> &mut Self {
        let CopyBufferToImageInfo {
            src_buffer,
            dst_image,
            dst_image_layout,
            regions,
            _ne: _,
        } = copy_buffer_to_image_info;

        if regions.is_empty() {
            return self;
        }

        let dst_image_inner = dst_image.inner();

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
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

                    ash::vk::BufferImageCopy2 {
                        buffer_offset: buffer_offset + src_buffer.offset(),
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
                src_buffer: src_buffer.buffer().handle(),
                dst_image: dst_image_inner.handle(),
                dst_image_layout: dst_image_layout.into(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                ..Default::default()
            };

            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_copy_buffer_to_image2)(self.handle(), &copy_buffer_to_image_info);
            } else {
                (fns.khr_copy_commands2.cmd_copy_buffer_to_image2_khr)(
                    self.handle(),
                    &copy_buffer_to_image_info,
                );
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

                    ash::vk::BufferImageCopy {
                        buffer_offset: buffer_offset + src_buffer.offset(),
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

            (fns.v1_0.cmd_copy_buffer_to_image)(
                self.handle(),
                src_buffer.buffer().handle(),
                dst_image_inner.handle(),
                dst_image_layout.into(),
                regions.len() as u32,
                regions.as_ptr(),
            );
        }

        let command_index = self.next_command_index;
        let command_name = "copy_buffer_to_image";
        let src_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Source,
            secondary_use_ref: None,
        };
        let dst_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Destination,
            secondary_use_ref: None,
        };

        for region in regions {
            let buffer_copy_size = region.buffer_copy_size(dst_image.format());

            let BufferImageCopy {
                buffer_offset,
                buffer_row_length: _,
                buffer_image_height: _,
                image_subresource,
                image_offset: _,
                image_extent: _,
                _ne: _,
            } = region;

            let mut src_range = buffer_offset..buffer_offset + buffer_copy_size;
            src_range.start += src_buffer.offset();
            src_range.end += src_buffer.offset();
            self.resources_usage_state.record_buffer_access(
                &src_use_ref,
                src_buffer.buffer(),
                src_range,
                PipelineStageAccess::Copy_TransferRead,
            );

            self.resources_usage_state.record_image_access(
                &dst_use_ref,
                dst_image_inner,
                ImageSubresourceRange::from(image_subresource),
                PipelineStageAccess::Copy_TransferWrite,
                dst_image_layout,
            );
        }

        self.resources.push(Box::new(src_buffer));
        self.resources.push(Box::new(dst_image));

        self.next_command_index += 1;
        self
    }

    /// Copies from an image to a buffer.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers and images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn copy_image_to_buffer(
        &mut self,
        copy_image_to_buffer_info: CopyImageToBufferInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_copy_image_to_buffer(&copy_image_to_buffer_info)?;

        unsafe { Ok(self.copy_image_to_buffer_unchecked(copy_image_to_buffer_info)) }
    }

    fn validate_copy_image_to_buffer(
        &self,
        copy_image_to_buffer_info: &CopyImageToBufferInfo,
    ) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdCopyImageToBuffer2-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdCopyImageToBuffer2-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &CopyImageToBufferInfo {
            ref src_image,
            src_image_layout,
            ref dst_buffer,
            ref regions,
            _ne: _,
        } = copy_image_to_buffer_info;

        // VUID-VkCopyImageToBufferInfo2-srcImageLayout-parameter
        src_image_layout.validate_device(device)?;

        // VUID-VkCopyImageToBufferInfo2-commonparent
        assert_eq!(device, dst_buffer.device());
        assert_eq!(device, src_image.device());

        let mut image_aspects = src_image.format().aspects();

        // VUID-VkCopyImageToBufferInfo2-srcImage-00186
        if !src_image.usage().intersects(ImageUsage::TRANSFER_SRC) {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkCopyImageToBufferInfo2-dstBuffer-00191
        if !dst_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_DST)
        {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            // VUID-VkCopyImageToBufferInfo2-srcImage-01998
            if !src_image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_SRC)
            {
                return Err(CopyError::MissingFormatFeature {
                    resource: CopyErrorResource::Source,
                    format_feature: "transfer_src",
                });
            }
        }

        // VUID-VkCopyImageToBufferInfo2-srcImage-00188
        if src_image.samples() != SampleCount::Sample1 {
            return Err(CopyError::SampleCountInvalid {
                resource: CopyErrorResource::Source,
                sample_count: src_image.samples(),
                allowed_sample_counts: SampleCounts::SAMPLE_1,
            });
        }

        // VUID-VkCopyImageToBufferInfo2-srcImageLayout-01397
        if !matches!(
            src_image_layout,
            ImageLayout::TransferSrcOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Source,
                image_layout: src_image_layout,
            });
        }

        let extent_alignment = match queue_family_properties.min_image_transfer_granularity {
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
                    src_image.format().block_extent(),
                    image_aspects.intersects(ImageAspects::PLANE_0),
                ))
            }
        };

        if image_aspects.intersects(ImageAspects::PLANE_0) {
            // VUID-VkCopyImageToBufferInfo2-aspectMask-01560
            image_aspects -= ImageAspects::COLOR;
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
            if image_subresource.mip_level >= src_image.mip_levels() {
                return Err(CopyError::MipLevelsOutOfRange {
                    resource: CopyErrorResource::Source,
                    region_index,
                    mip_levels_range_end: image_subresource.mip_level + 1,
                    image_mip_levels: src_image.mip_levels(),
                });
            }

            // VUID-VkImageSubresourceLayers-layerCount-01700
            assert!(!image_subresource.array_layers.is_empty());

            // VUID-VkCopyImageToBufferInfo2-imageSubresource-01704
            // VUID-VkCopyImageToBufferInfo2-baseArrayLayer-00213
            if image_subresource.array_layers.end > src_image.dimensions().array_layers() {
                return Err(CopyError::ArrayLayersOutOfRange {
                    resource: CopyErrorResource::Source,
                    region_index,
                    array_layers_range_end: image_subresource.array_layers.end,
                    image_array_layers: src_image.dimensions().array_layers(),
                });
            }

            // VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask
            assert!(!image_subresource.aspects.is_empty());

            // VUID-VkCopyImageToBufferInfo2-aspectMask-00211
            if !image_aspects.contains(image_subresource.aspects) {
                return Err(CopyError::AspectsNotAllowed {
                    resource: CopyErrorResource::Source,
                    region_index,
                    aspects: image_subresource.aspects,
                    allowed_aspects: image_aspects,
                });
            }

            // VUID-VkBufferImageCopy2-aspectMask-00212
            if image_subresource.aspects.count() != 1 {
                return Err(CopyError::MultipleAspectsNotAllowed {
                    resource: CopyErrorResource::Source,
                    region_index,
                    aspects: image_subresource.aspects,
                });
            }

            let (image_subresource_format, image_subresource_extent) =
                if image_aspects.intersects(ImageAspects::PLANE_0) {
                    if image_subresource.aspects.intersects(ImageAspects::PLANE_0) {
                        (
                            src_image.format().planes()[0],
                            src_image.dimensions().width_height_depth(),
                        )
                    } else if image_subresource.aspects.intersects(ImageAspects::PLANE_1) {
                        (
                            src_image.format().planes()[1],
                            src_image
                                .format()
                                .ycbcr_chroma_sampling()
                                .unwrap()
                                .subsampled_extent(src_image.dimensions().width_height_depth()),
                        )
                    } else {
                        (
                            src_image.format().planes()[2],
                            src_image
                                .format()
                                .ycbcr_chroma_sampling()
                                .unwrap()
                                .subsampled_extent(src_image.dimensions().width_height_depth()),
                        )
                    }
                } else {
                    (
                        src_image.format(),
                        src_image
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

            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkBufferImageCopy.html#_description
            let image_subresource_block_size =
                if image_subresource.aspects.intersects(ImageAspects::STENCIL) {
                    1
                } else if image_subresource.aspects.intersects(ImageAspects::DEPTH) {
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

            let buffer_offset_alignment =
                if image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
                    4
                } else {
                    let mut buffer_offset_alignment = image_subresource_block_size;

                    // VUID-VkCopyImageToBufferInfo2-commandBuffer-04052
                    // Make the alignment a multiple of 4.
                    if !queue_family_properties
                        .queue_flags
                        .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
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
            if (dst_buffer.offset() + buffer_offset) % buffer_offset_alignment != 0 {
                return Err(CopyError::OffsetNotAlignedForBuffer {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    offset: dst_buffer.offset() + buffer_offset,
                    required_alignment: buffer_offset_alignment,
                });
            }

            let buffer_copy_size = region.buffer_copy_size(image_subresource_format);

            // VUID-VkCopyImageToBufferInfo2-pRegions-00183
            if buffer_offset + buffer_copy_size > dst_buffer.size() {
                return Err(CopyError::RegionOutOfBufferBounds {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    offset_range_end: buffer_offset + buffer_copy_size,
                    buffer_size: dst_buffer.size(),
                });
            }
        }

        // VUID-VkCopyImageToBufferInfo2-pRegions-00184
        // Can't occur as long as memory aliasing isn't allowed.

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_image_to_buffer_unchecked(
        &mut self,
        copy_image_to_buffer_info: CopyImageToBufferInfo,
    ) -> &mut Self {
        let CopyImageToBufferInfo {
            src_image,
            src_image_layout,
            dst_buffer,
            regions,
            _ne: _,
        } = copy_image_to_buffer_info;

        if regions.is_empty() {
            return self;
        }

        let src_image_inner = src_image.inner();

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
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

                    ash::vk::BufferImageCopy2 {
                        buffer_offset: buffer_offset + dst_buffer.offset(),
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
                src_image: src_image_inner.handle(),
                src_image_layout: src_image_layout.into(),
                dst_buffer: dst_buffer.buffer().handle(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                ..Default::default()
            };

            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_copy_image_to_buffer2)(self.handle(), &copy_image_to_buffer_info);
            } else {
                (fns.khr_copy_commands2.cmd_copy_image_to_buffer2_khr)(
                    self.handle(),
                    &copy_image_to_buffer_info,
                );
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

                    ash::vk::BufferImageCopy {
                        buffer_offset: buffer_offset + dst_buffer.offset(),
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

            (fns.v1_0.cmd_copy_image_to_buffer)(
                self.handle(),
                src_image_inner.handle(),
                src_image_layout.into(),
                dst_buffer.buffer().handle(),
                regions.len() as u32,
                regions.as_ptr(),
            );
        }

        let command_index = self.next_command_index;
        let command_name = "copy_image_to_buffer";
        let src_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Source,
            secondary_use_ref: None,
        };
        let dst_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Destination,
            secondary_use_ref: None,
        };

        for region in regions {
            let buffer_copy_size = region.buffer_copy_size(src_image.format());

            let BufferImageCopy {
                buffer_offset,
                buffer_row_length: _,
                buffer_image_height: _,
                image_subresource,
                image_offset: _,
                image_extent: _,
                _ne: _,
            } = region;

            self.resources_usage_state.record_image_access(
                &src_use_ref,
                src_image_inner,
                ImageSubresourceRange::from(image_subresource),
                PipelineStageAccess::Copy_TransferRead,
                src_image_layout,
            );

            let mut dst_range = buffer_offset..buffer_offset + buffer_copy_size;
            dst_range.start += dst_buffer.offset();
            dst_range.end += dst_buffer.offset();
            self.resources_usage_state.record_buffer_access(
                &dst_use_ref,
                dst_buffer.buffer(),
                dst_range,
                PipelineStageAccess::Copy_TransferWrite,
            );
        }

        self.resources.push(Box::new(src_image));
        self.resources.push(Box::new(dst_buffer));

        self.next_command_index += 1;
        self
    }

    /// Blits an image to another.
    ///
    /// A *blit* is similar to an image copy operation, except that the portion of the image that
    /// is transferred can be resized. You choose an area of the source and an area of the
    /// destination, and the implementation will resize the area of the source so that it matches
    /// the size of the area of the destination before writing it.
    ///
    /// Blit operations have several restrictions:
    ///
    /// - Blit operations are only allowed on queue families that support graphics operations.
    /// - The format of the source and destination images must support blit operations, which
    ///   depends on the Vulkan implementation. Vulkan guarantees that some specific formats must
    ///   always be supported. See tables 52 to 61 of the specifications.
    /// - Only single-sampled images are allowed.
    /// - You can only blit between two images whose formats belong to the same type. The types
    ///   are: floating-point, signed integers, unsigned integers, depth-stencil.
    /// - If you blit between depth, stencil or depth-stencil images, the format of both images
    ///   must match exactly.
    /// - If you blit between depth, stencil or depth-stencil images, only the `Nearest` filter is
    ///   allowed.
    /// - For two-dimensional images, the Z coordinate must be 0 for the top-left offset and 1 for
    ///   the bottom-right offset. Same for the Y coordinate for one-dimensional images.
    /// - For non-array images, the base array layer must be 0 and the number of layers must be 1.
    ///
    /// If `layer_count` is greater than 1, the blit will happen between each individual layer as
    /// if they were separate images.
    ///
    /// # Panic
    ///
    /// - Panics if the source or the destination was not created with `device`.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn blit_image(
        &mut self,
        blit_image_info: BlitImageInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_blit_image(&blit_image_info)?;

        unsafe { Ok(self.blit_image_unchecked(blit_image_info)) }
    }

    fn validate_blit_image(&self, blit_image_info: &BlitImageInfo) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdBlitImage2-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBlitImage2-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &BlitImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            filter,
            _ne: _,
        } = blit_image_info;

        // VUID-VkBlitImageInfo2-srcImageLayout-parameter
        src_image_layout.validate_device(device)?;

        // VUID-VkBlitImageInfo2-dstImageLayout-parameter
        dst_image_layout.validate_device(device)?;

        // VUID-VkBlitImageInfo2-filter-parameter
        filter.validate_device(device)?;

        let src_image_inner = src_image.inner();
        let dst_image_inner = dst_image.inner();

        // VUID-VkBlitImageInfo2-commonparent
        assert_eq!(device, src_image.device());
        assert_eq!(device, dst_image.device());

        let src_image_aspects = src_image.format().aspects();
        let dst_image_aspects = dst_image.format().aspects();
        let src_image_type = src_image.dimensions().image_type();
        let dst_image_type = dst_image.dimensions().image_type();

        // VUID-VkBlitImageInfo2-srcImage-00219
        if !src_image.usage().intersects(ImageUsage::TRANSFER_SRC) {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkBlitImageInfo2-dstImage-00224
        if !dst_image.usage().intersects(ImageUsage::TRANSFER_DST) {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        // VUID-VkBlitImageInfo2-srcImage-01999
        if !src_image
            .format_features()
            .intersects(FormatFeatures::BLIT_SRC)
        {
            return Err(CopyError::MissingFormatFeature {
                resource: CopyErrorResource::Source,
                format_feature: "blit_src",
            });
        }

        // VUID-VkBlitImageInfo2-dstImage-02000
        if !dst_image
            .format_features()
            .intersects(FormatFeatures::BLIT_DST)
        {
            return Err(CopyError::MissingFormatFeature {
                resource: CopyErrorResource::Destination,
                format_feature: "blit_dst",
            });
        }

        // VUID-VkBlitImageInfo2-srcImage-06421
        if src_image.format().ycbcr_chroma_sampling().is_some() {
            return Err(CopyError::FormatNotSupported {
                resource: CopyErrorResource::Source,
                format: src_image.format(),
            });
        }

        // VUID-VkBlitImageInfo2-dstImage-06422
        if dst_image.format().ycbcr_chroma_sampling().is_some() {
            return Err(CopyError::FormatNotSupported {
                resource: CopyErrorResource::Destination,
                format: src_image.format(),
            });
        }

        if !(src_image_aspects.intersects(ImageAspects::COLOR)
            && dst_image_aspects.intersects(ImageAspects::COLOR))
        {
            // VUID-VkBlitImageInfo2-srcImage-00231
            if src_image.format() != dst_image.format() {
                return Err(CopyError::FormatsMismatch {
                    src_format: src_image.format(),
                    dst_format: dst_image.format(),
                });
            }
        } else {
            // VUID-VkBlitImageInfo2-srcImage-00229
            // VUID-VkBlitImageInfo2-srcImage-00230
            if !matches!(
                (
                    src_image.format().type_color().unwrap(),
                    dst_image.format().type_color().unwrap()
                ),
                (
                    NumericType::SFLOAT
                        | NumericType::UFLOAT
                        | NumericType::SNORM
                        | NumericType::UNORM
                        | NumericType::SSCALED
                        | NumericType::USCALED
                        | NumericType::SRGB,
                    NumericType::SFLOAT
                        | NumericType::UFLOAT
                        | NumericType::SNORM
                        | NumericType::UNORM
                        | NumericType::SSCALED
                        | NumericType::USCALED
                        | NumericType::SRGB,
                ) | (NumericType::SINT, NumericType::SINT)
                    | (NumericType::UINT, NumericType::UINT)
            ) {
                return Err(CopyError::FormatsNotCompatible {
                    src_format: src_image.format(),
                    dst_format: dst_image.format(),
                });
            }
        }

        // VUID-VkBlitImageInfo2-srcImage-00233
        if src_image.samples() != SampleCount::Sample1 {
            return Err(CopyError::SampleCountInvalid {
                resource: CopyErrorResource::Destination,
                sample_count: dst_image.samples(),
                allowed_sample_counts: SampleCounts::SAMPLE_1,
            });
        }

        // VUID-VkBlitImageInfo2-dstImage-00234
        if dst_image.samples() != SampleCount::Sample1 {
            return Err(CopyError::SampleCountInvalid {
                resource: CopyErrorResource::Destination,
                sample_count: dst_image.samples(),
                allowed_sample_counts: SampleCounts::SAMPLE_1,
            });
        }

        // VUID-VkBlitImageInfo2-srcImageLayout-01398
        if !matches!(
            src_image_layout,
            ImageLayout::TransferSrcOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Source,
                image_layout: src_image_layout,
            });
        }

        // VUID-VkBlitImageInfo2-dstImageLayout-01399
        if !matches!(
            dst_image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Destination,
                image_layout: dst_image_layout,
            });
        }

        // VUID-VkBlitImageInfo2-srcImage-00232
        if !src_image_aspects.intersects(ImageAspects::COLOR) && filter != Filter::Nearest {
            return Err(CopyError::FilterNotSupportedByFormat);
        }

        match filter {
            Filter::Nearest => (),
            Filter::Linear => {
                // VUID-VkBlitImageInfo2-filter-02001
                if !src_image
                    .format_features()
                    .intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR)
                {
                    return Err(CopyError::FilterNotSupportedByFormat);
                }
            }
            Filter::Cubic => {
                // VUID-VkBlitImageInfo2-filter-02002
                if !src_image
                    .format_features()
                    .intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_CUBIC)
                {
                    return Err(CopyError::FilterNotSupportedByFormat);
                }

                // VUID-VkBlitImageInfo2-filter-00237
                if !matches!(src_image.dimensions(), ImageDimensions::Dim2d { .. }) {
                    return Err(CopyError::FilterNotSupportedForImageType);
                }
            }
        }

        let is_same_image = src_image_inner == dst_image_inner;
        let mut overlap_subresource_indices = None;
        let mut overlap_extent_indices = None;

        for (region_index, region) in regions.iter().enumerate() {
            let &ImageBlit {
                ref src_subresource,
                src_offsets,
                ref dst_subresource,
                dst_offsets,
                _ne: _,
            } = region;

            let check_subresource = |resource: CopyErrorResource,
                                     image: &dyn ImageAccess,
                                     image_aspects: ImageAspects,
                                     subresource: &ImageSubresourceLayers|
             -> Result<_, CopyError> {
                // VUID-VkBlitImageInfo2-srcSubresource-01705
                // VUID-VkBlitImageInfo2-dstSubresource-01706
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

                // VUID-VkBlitImageInfo2-srcSubresource-01707
                // VUID-VkBlitImageInfo2-dstSubresource-01708
                // VUID-VkBlitImageInfo2-srcImage-00240
                if subresource.array_layers.end > image.dimensions().array_layers() {
                    return Err(CopyError::ArrayLayersOutOfRange {
                        resource,
                        region_index,
                        array_layers_range_end: subresource.array_layers.end,
                        image_array_layers: image.dimensions().array_layers(),
                    });
                }

                // VUID-VkImageSubresourceLayers-aspectMask-parameter
                subresource.aspects.validate_device(device)?;

                // VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask
                assert!(!subresource.aspects.is_empty());

                // VUID-VkBlitImageInfo2-aspectMask-00241
                // VUID-VkBlitImageInfo2-aspectMask-00242
                if !image_aspects.contains(subresource.aspects) {
                    return Err(CopyError::AspectsNotAllowed {
                        resource,
                        region_index,
                        aspects: subresource.aspects,
                        allowed_aspects: image_aspects,
                    });
                }

                Ok(image
                    .dimensions()
                    .mip_level_dimensions(subresource.mip_level)
                    .unwrap()
                    .width_height_depth())
            };

            let src_subresource_extent = check_subresource(
                CopyErrorResource::Source,
                src_image,
                src_image_aspects,
                src_subresource,
            )?;
            let dst_subresource_extent = check_subresource(
                CopyErrorResource::Destination,
                dst_image,
                dst_image_aspects,
                dst_subresource,
            )?;

            // VUID-VkImageBlit2-aspectMask-00238
            if src_subresource.aspects != dst_subresource.aspects {
                return Err(CopyError::AspectsMismatch {
                    region_index,
                    src_aspects: src_subresource.aspects,
                    dst_aspects: dst_subresource.aspects,
                });
            }

            let src_layer_count =
                src_subresource.array_layers.end - src_subresource.array_layers.start;
            let dst_layer_count =
                dst_subresource.array_layers.end - dst_subresource.array_layers.start;

            // VUID-VkImageBlit2-layerCount-00239
            // VUID-VkBlitImageInfo2-srcImage-00240
            if src_layer_count != dst_layer_count {
                return Err(CopyError::ArrayLayerCountMismatch {
                    region_index,
                    src_layer_count,
                    dst_layer_count,
                });
            }

            let check_offset_extent = |resource: CopyErrorResource,
                                       image_type: ImageType,
                                       subresource_extent: [u32; 3],
                                       offsets: [[u32; 3]; 2]|
             -> Result<_, CopyError> {
                match image_type {
                    ImageType::Dim1d => {
                        // VUID-VkBlitImageInfo2-srcImage-00245
                        // VUID-VkBlitImageInfo2-dstImage-00250
                        if !(offsets[0][1] == 0 && offsets[1][1] == 1) {
                            return Err(CopyError::OffsetsInvalidForImageType {
                                resource,
                                region_index,
                                offsets: [offsets[0][1], offsets[1][1]],
                            });
                        }

                        // VUID-VkBlitImageInfo2-srcImage-00247
                        // VUID-VkBlitImageInfo2-dstImage-00252
                        if !(offsets[0][2] == 0 && offsets[1][2] == 1) {
                            return Err(CopyError::OffsetsInvalidForImageType {
                                resource,
                                region_index,
                                offsets: [offsets[0][2], offsets[1][2]],
                            });
                        }
                    }
                    ImageType::Dim2d => {
                        // VUID-VkBlitImageInfo2-srcImage-00247
                        // VUID-VkBlitImageInfo2-dstImage-00252
                        if !(offsets[0][2] == 0 && offsets[1][2] == 1) {
                            return Err(CopyError::OffsetsInvalidForImageType {
                                resource,
                                region_index,
                                offsets: [offsets[0][2], offsets[1][2]],
                            });
                        }
                    }
                    ImageType::Dim3d => (),
                }

                let offset_range_end = [
                    max(offsets[0][0], offsets[1][0]),
                    max(offsets[0][1], offsets[1][1]),
                    max(offsets[0][2], offsets[1][2]),
                ];

                for i in 0..3 {
                    // VUID-VkBlitImageInfo2-srcOffset-00243
                    // VUID-VkBlitImageInfo2-srcOffset-00244
                    // VUID-VkBlitImageInfo2-srcOffset-00246
                    // VUID-VkBlitImageInfo2-dstOffset-00248
                    // VUID-VkBlitImageInfo2-dstOffset-00249
                    // VUID-VkBlitImageInfo2-dstOffset-00251
                    if offset_range_end[i] > subresource_extent[i] {
                        return Err(CopyError::RegionOutOfImageBounds {
                            resource,
                            region_index,
                            offset_range_end,
                            subresource_extent,
                        });
                    }
                }

                Ok(())
            };

            check_offset_extent(
                CopyErrorResource::Source,
                src_image_type,
                src_subresource_extent,
                src_offsets,
            )?;
            check_offset_extent(
                CopyErrorResource::Destination,
                dst_image_type,
                dst_subresource_extent,
                dst_offsets,
            )?;

            // VUID-VkBlitImageInfo2-pRegions-00217
            if is_same_image {
                let src_region_index = region_index;
                let src_subresource_axes = [
                    src_subresource.mip_level..src_subresource.mip_level + 1,
                    src_subresource.array_layers.start..src_subresource.array_layers.end,
                ];
                let src_extent_axes = [
                    min(src_offsets[0][0], src_offsets[1][0])
                        ..max(src_offsets[0][0], src_offsets[1][0]),
                    min(src_offsets[0][1], src_offsets[1][1])
                        ..max(src_offsets[0][1], src_offsets[1][1]),
                    min(src_offsets[0][2], src_offsets[1][2])
                        ..max(src_offsets[0][2], src_offsets[1][2]),
                ];

                for (dst_region_index, dst_region) in regions.iter().enumerate() {
                    let &ImageBlit {
                        ref dst_subresource,
                        dst_offsets,
                        ..
                    } = dst_region;

                    let dst_subresource_axes = [
                        dst_subresource.mip_level..dst_subresource.mip_level + 1,
                        src_subresource.array_layers.start..src_subresource.array_layers.end,
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
                        min(dst_offsets[0][0], dst_offsets[1][0])
                            ..max(dst_offsets[0][0], dst_offsets[1][0]),
                        min(dst_offsets[0][1], dst_offsets[1][1])
                            ..max(dst_offsets[0][1], dst_offsets[1][1]),
                        min(dst_offsets[0][2], dst_offsets[1][2])
                            ..max(dst_offsets[0][2], dst_offsets[1][2]),
                    ];

                    if src_extent_axes
                        .iter()
                        .zip(dst_extent_axes)
                        .any(|(src_range, dst_range)| {
                            src_range.start >= dst_range.end || dst_range.start >= src_range.end
                        })
                    {
                        continue;
                    }

                    // If the extent axes *also* overlap, then that's an error.
                    overlap_extent_indices = Some((src_region_index, dst_region_index));
                }
            }
        }

        // VUID-VkBlitImageInfo2-pRegions-00217
        if let Some((src_region_index, dst_region_index)) = overlap_extent_indices {
            return Err(CopyError::OverlappingRegions {
                src_region_index,
                dst_region_index,
            });
        }

        // VUID-VkBlitImageInfo2-srcImageLayout-00221
        // VUID-VkBlitImageInfo2-dstImageLayout-00226
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

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn blit_image_unchecked(&mut self, blit_image_info: BlitImageInfo) -> &mut Self {
        let BlitImageInfo {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions,
            filter,
            _ne: _,
        } = blit_image_info;

        if regions.is_empty() {
            return self;
        }

        let src_image_inner = src_image.inner();
        let dst_image_inner = dst_image.inner();

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &ImageBlit {
                        ref src_subresource,
                        src_offsets,
                        ref dst_subresource,
                        dst_offsets,
                        _ne: _,
                    } = region;

                    ash::vk::ImageBlit2 {
                        src_subresource: src_subresource.into(),
                        src_offsets: [
                            ash::vk::Offset3D {
                                x: src_offsets[0][0] as i32,
                                y: src_offsets[0][1] as i32,
                                z: src_offsets[0][2] as i32,
                            },
                            ash::vk::Offset3D {
                                x: src_offsets[1][0] as i32,
                                y: src_offsets[1][1] as i32,
                                z: src_offsets[1][2] as i32,
                            },
                        ],
                        dst_subresource: dst_subresource.into(),
                        dst_offsets: [
                            ash::vk::Offset3D {
                                x: dst_offsets[0][0] as i32,
                                y: dst_offsets[0][1] as i32,
                                z: dst_offsets[0][2] as i32,
                            },
                            ash::vk::Offset3D {
                                x: dst_offsets[1][0] as i32,
                                y: dst_offsets[1][1] as i32,
                                z: dst_offsets[1][2] as i32,
                            },
                        ],
                        ..Default::default()
                    }
                })
                .collect();

            let blit_image_info = ash::vk::BlitImageInfo2 {
                src_image: src_image_inner.handle(),
                src_image_layout: src_image_layout.into(),
                dst_image: dst_image_inner.handle(),
                dst_image_layout: dst_image_layout.into(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                filter: filter.into(),
                ..Default::default()
            };

            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_blit_image2)(self.handle(), &blit_image_info);
            } else {
                (fns.khr_copy_commands2.cmd_blit_image2_khr)(self.handle(), &blit_image_info);
            }
        } else {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &ImageBlit {
                        ref src_subresource,
                        src_offsets,
                        ref dst_subresource,
                        dst_offsets,
                        _ne: _,
                    } = region;

                    ash::vk::ImageBlit {
                        src_subresource: src_subresource.into(),
                        src_offsets: [
                            ash::vk::Offset3D {
                                x: src_offsets[0][0] as i32,
                                y: src_offsets[0][1] as i32,
                                z: src_offsets[0][2] as i32,
                            },
                            ash::vk::Offset3D {
                                x: src_offsets[1][0] as i32,
                                y: src_offsets[1][1] as i32,
                                z: src_offsets[1][2] as i32,
                            },
                        ],
                        dst_subresource: dst_subresource.into(),
                        dst_offsets: [
                            ash::vk::Offset3D {
                                x: dst_offsets[0][0] as i32,
                                y: dst_offsets[0][1] as i32,
                                z: dst_offsets[0][2] as i32,
                            },
                            ash::vk::Offset3D {
                                x: dst_offsets[1][0] as i32,
                                y: dst_offsets[1][1] as i32,
                                z: dst_offsets[1][2] as i32,
                            },
                        ],
                    }
                })
                .collect();

            (fns.v1_0.cmd_blit_image)(
                self.handle(),
                src_image_inner.handle(),
                src_image_layout.into(),
                dst_image_inner.handle(),
                dst_image_layout.into(),
                regions.len() as u32,
                regions.as_ptr(),
                filter.into(),
            );
        }

        let command_index = self.next_command_index;
        let command_name = "blit_image";
        let src_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Source,
            secondary_use_ref: None,
        };
        let dst_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Destination,
            secondary_use_ref: None,
        };

        for region in regions {
            let ImageBlit {
                src_subresource,
                src_offsets: _,
                dst_subresource,
                dst_offsets: _,
                _ne: _,
            } = region;

            self.resources_usage_state.record_image_access(
                &src_use_ref,
                src_image_inner,
                ImageSubresourceRange::from(src_subresource),
                PipelineStageAccess::Blit_TransferRead,
                src_image_layout,
            );

            self.resources_usage_state.record_image_access(
                &dst_use_ref,
                dst_image_inner,
                ImageSubresourceRange::from(dst_subresource),
                PipelineStageAccess::Blit_TransferWrite,
                dst_image_layout,
            );
        }

        self.resources.push(Box::new(src_image));
        self.resources.push(Box::new(dst_image));

        self.next_command_index += 1;
        self
    }

    /// Resolves a multisampled image into a single-sampled image.
    ///
    /// # Panics
    ///
    /// - Panics if `src_image` or `dst_image` were not created from the same device
    ///   as `self`.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn resolve_image(
        &mut self,
        resolve_image_info: ResolveImageInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_resolve_image(&resolve_image_info)?;

        unsafe { Ok(self.resolve_image_unchecked(resolve_image_info)) }
    }

    fn validate_resolve_image(
        &self,
        resolve_image_info: &ResolveImageInfo,
    ) -> Result<(), CopyError> {
        let device = self.device();

        // VUID-vkCmdResolveImage2-renderpass
        if self.builder_state.render_pass.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdResolveImage2-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &ResolveImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = resolve_image_info;

        // VUID-VkResolveImageInfo2-srcImageLayout-parameter
        src_image_layout.validate_device(device)?;

        // VUID-VkResolveImageInfo2-dstImageLayout-parameter
        dst_image_layout.validate_device(device)?;

        // VUID-VkResolveImageInfo2-commonparent
        assert_eq!(device, src_image.device());
        assert_eq!(device, dst_image.device());

        let src_image_type = src_image.dimensions().image_type();
        let dst_image_type = dst_image.dimensions().image_type();

        // VUID-VkResolveImageInfo2-srcImage-00257
        if src_image.samples() == SampleCount::Sample1 {
            return Err(CopyError::SampleCountInvalid {
                resource: CopyErrorResource::Source,
                sample_count: dst_image.samples(),
                allowed_sample_counts: SampleCounts::SAMPLE_2
                    | SampleCounts::SAMPLE_4
                    | SampleCounts::SAMPLE_8
                    | SampleCounts::SAMPLE_16
                    | SampleCounts::SAMPLE_32
                    | SampleCounts::SAMPLE_64,
            });
        }

        // VUID-VkResolveImageInfo2-dstImage-00259
        if dst_image.samples() != SampleCount::Sample1 {
            return Err(CopyError::SampleCountInvalid {
                resource: CopyErrorResource::Destination,
                sample_count: dst_image.samples(),
                allowed_sample_counts: SampleCounts::SAMPLE_1,
            });
        }

        // VUID-VkResolveImageInfo2-dstImage-02003
        if !dst_image
            .format_features()
            .intersects(FormatFeatures::COLOR_ATTACHMENT)
        {
            return Err(CopyError::MissingFormatFeature {
                resource: CopyErrorResource::Destination,
                format_feature: "color_attachment",
            });
        }

        // VUID-VkResolveImageInfo2-srcImage-01386
        if src_image.format() != dst_image.format() {
            return Err(CopyError::FormatsMismatch {
                src_format: src_image.format(),
                dst_format: dst_image.format(),
            });
        }

        // VUID-VkResolveImageInfo2-srcImageLayout-01400
        if !matches!(
            src_image_layout,
            ImageLayout::TransferSrcOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Source,
                image_layout: src_image_layout,
            });
        }

        // VUID-VkResolveImageInfo2-dstImageLayout-01401
        if !matches!(
            dst_image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Destination,
                image_layout: dst_image_layout,
            });
        }

        // Should be guaranteed by the requirement that formats match, and that the destination
        // image format features support color attachments.
        debug_assert!(
            src_image.format().aspects().intersects(ImageAspects::COLOR)
                && dst_image.format().aspects().intersects(ImageAspects::COLOR)
        );

        for (region_index, region) in regions.iter().enumerate() {
            let &ImageResolve {
                ref src_subresource,
                src_offset,
                ref dst_subresource,
                dst_offset,
                extent,
                _ne: _,
            } = region;

            let check_subresource = |resource: CopyErrorResource,
                                     image: &dyn ImageAccess,
                                     subresource: &ImageSubresourceLayers|
             -> Result<_, CopyError> {
                // VUID-VkResolveImageInfo2-srcSubresource-01709
                // VUID-VkResolveImageInfo2-dstSubresource-01710
                if subresource.mip_level >= image.mip_levels() {
                    return Err(CopyError::MipLevelsOutOfRange {
                        resource,
                        region_index,
                        mip_levels_range_end: subresource.mip_level + 1,
                        image_mip_levels: image.mip_levels(),
                    });
                }

                // VUID-VkImageSubresourceLayers-layerCount-01700
                // VUID-VkResolveImageInfo2-srcImage-04446
                // VUID-VkResolveImageInfo2-srcImage-04447
                assert!(!subresource.array_layers.is_empty());

                // VUID-VkResolveImageInfo2-srcSubresource-01711
                // VUID-VkResolveImageInfo2-dstSubresource-01712
                // VUID-VkResolveImageInfo2-srcImage-04446
                // VUID-VkResolveImageInfo2-srcImage-04447
                if subresource.array_layers.end > image.dimensions().array_layers() {
                    return Err(CopyError::ArrayLayersOutOfRange {
                        resource: CopyErrorResource::Destination,
                        region_index,
                        array_layers_range_end: subresource.array_layers.end,
                        image_array_layers: image.dimensions().array_layers(),
                    });
                }

                // VUID-VkImageSubresourceLayers-aspectMask-parameter
                subresource.aspects.validate_device(device)?;

                // VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask
                // VUID-VkImageResolve2-aspectMask-00266
                if subresource.aspects != (ImageAspects::COLOR) {
                    return Err(CopyError::AspectsNotAllowed {
                        resource,
                        region_index,
                        aspects: subresource.aspects,
                        allowed_aspects: ImageAspects::COLOR,
                    });
                }

                Ok(image
                    .dimensions()
                    .mip_level_dimensions(subresource.mip_level)
                    .unwrap()
                    .width_height_depth())
            };

            let src_subresource_extent =
                check_subresource(CopyErrorResource::Source, src_image, src_subresource)?;
            let dst_subresource_extent =
                check_subresource(CopyErrorResource::Destination, dst_image, dst_subresource)?;

            let src_layer_count =
                src_subresource.array_layers.end - src_subresource.array_layers.start;
            let dst_layer_count =
                dst_subresource.array_layers.end - dst_subresource.array_layers.start;

            // VUID-VkImageResolve2-layerCount-00267
            // VUID-VkResolveImageInfo2-srcImage-04446
            // VUID-VkResolveImageInfo2-srcImage-04447
            if src_layer_count != dst_layer_count {
                return Err(CopyError::ArrayLayerCountMismatch {
                    region_index,
                    src_layer_count,
                    dst_layer_count,
                });
            }

            // No VUID, but it makes sense?
            assert!(extent[0] != 0 && extent[1] != 0 && extent[2] != 0);

            let check_offset_extent = |resource: CopyErrorResource,
                                       _image_type: ImageType,
                                       subresource_extent: [u32; 3],
                                       offset: [u32; 3]|
             -> Result<_, CopyError> {
                for i in 0..3 {
                    // No VUID, but makes sense?
                    assert!(extent[i] != 0);

                    // VUID-VkResolveImageInfo2-srcOffset-00269
                    // VUID-VkResolveImageInfo2-srcOffset-00270
                    // VUID-VkResolveImageInfo2-srcOffset-00272
                    // VUID-VkResolveImageInfo2-dstOffset-00274
                    // VUID-VkResolveImageInfo2-dstOffset-00275
                    // VUID-VkResolveImageInfo2-dstOffset-00277
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
                }

                Ok(())
            };

            check_offset_extent(
                CopyErrorResource::Source,
                src_image_type,
                src_subresource_extent,
                src_offset,
            )?;
            check_offset_extent(
                CopyErrorResource::Destination,
                dst_image_type,
                dst_subresource_extent,
                dst_offset,
            )?;
        }

        // VUID-VkResolveImageInfo2-pRegions-00255
        // Can't occur as long as memory aliasing isn't allowed, because `src_image` and
        // `dst_image` must have different sample counts and therefore can never be the same image.

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn resolve_image_unchecked(
        &mut self,
        resolve_image_info: ResolveImageInfo,
    ) -> &mut Self {
        let ResolveImageInfo {
            src_image,
            src_image_layout,
            dst_image,
            dst_image_layout,
            regions,
            _ne: _,
        } = resolve_image_info;

        if regions.is_empty() {
            return self;
        }

        let src_image_inner = src_image.inner();
        let dst_image_inner = dst_image.inner();

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3
            || self.device().enabled_extensions().khr_copy_commands2
        {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &ImageResolve {
                        ref src_subresource,
                        src_offset,
                        ref dst_subresource,
                        dst_offset,
                        extent,
                        _ne: _,
                    } = region;

                    ash::vk::ImageResolve2 {
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

            let resolve_image_info = ash::vk::ResolveImageInfo2 {
                src_image: src_image_inner.handle(),
                src_image_layout: src_image_layout.into(),
                dst_image: dst_image_inner.handle(),
                dst_image_layout: dst_image_layout.into(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                ..Default::default()
            };

            if self.device().api_version() >= Version::V1_3 {
                (fns.v1_3.cmd_resolve_image2)(self.handle(), &resolve_image_info);
            } else {
                (fns.khr_copy_commands2.cmd_resolve_image2_khr)(self.handle(), &resolve_image_info);
            }
        } else {
            let regions: SmallVec<[_; 8]> = regions
                .iter()
                .map(|region| {
                    let &ImageResolve {
                        ref src_subresource,
                        src_offset,
                        ref dst_subresource,
                        dst_offset,
                        extent,
                        _ne: _,
                    } = region;

                    ash::vk::ImageResolve {
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

            (fns.v1_0.cmd_resolve_image)(
                self.handle(),
                src_image_inner.handle(),
                src_image_layout.into(),
                dst_image_inner.handle(),
                dst_image_layout.into(),
                regions.len() as u32,
                regions.as_ptr(),
            );
        }

        let command_index = self.next_command_index;
        let command_name = "resolve_image";
        let src_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Source,
            secondary_use_ref: None,
        };
        let dst_use_ref = ResourceUseRef {
            command_index,
            command_name,
            resource_in_command: ResourceInCommand::Destination,
            secondary_use_ref: None,
        };

        for region in regions {
            let ImageResolve {
                src_subresource,
                src_offset: _,
                dst_subresource,
                dst_offset: _,
                extent: _,
                _ne: _,
            } = region;

            self.resources_usage_state.record_image_access(
                &src_use_ref,
                src_image_inner,
                ImageSubresourceRange::from(src_subresource),
                PipelineStageAccess::Resolve_TransferRead,
                src_image_layout,
            );

            self.resources_usage_state.record_image_access(
                &dst_use_ref,
                dst_image_inner,
                ImageSubresourceRange::from(dst_subresource),
                PipelineStageAccess::Resolve_TransferWrite,
                dst_image_layout,
            );
        }

        self.resources.push(Box::new(src_image));
        self.resources.push(Box::new(dst_image));

        self.next_command_index += 1;
        self
    }
}
