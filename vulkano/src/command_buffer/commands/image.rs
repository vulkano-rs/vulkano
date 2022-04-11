// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    command_buffer::{
        synced::{Command, Resource, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, CopyError, CopyErrorResource,
    },
    device::DeviceOwned,
    format::{ClearColorValue, ClearDepthStencilValue, NumericType},
    image::{
        ImageAccess, ImageAspects, ImageDimensions, ImageLayout, ImageSubresourceLayers,
        ImageSubresourceRange, ImageType, SampleCount, SampleCounts,
    },
    sampler::Filter,
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStages},
    Version, VulkanObject,
};
use smallvec::{smallvec, SmallVec};
use std::{
    cmp::{max, min},
    sync::Arc,
};

/// # Commands that operate on images.
///
/// Unlike transfer commands, these require a graphics queue, except for `clear_color_image`, which
/// can also be called on a compute queue.
impl<L, P> AutoCommandBufferBuilder<L, P> {
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
    pub fn blit_image(
        &mut self,
        mut blit_image_info: BlitImageInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_blit_image(&mut blit_image_info)?;

        unsafe {
            self.inner.blit_image(blit_image_info)?;
        }

        Ok(self)
    }

    fn validate_blit_image(&self, blit_image_info: &mut BlitImageInfo) -> Result<(), CopyError> {
        // VUID-vkCmdBlitImage2-renderpass
        if self.render_pass_state.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdBlitImage2-commandBuffer-cmdpool
        if !self.queue_family().supports_graphics() {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &mut BlitImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            filter,
            _ne: _,
        } = blit_image_info;

        let device = self.device();
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
        if !src_image.usage().transfer_src {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Source,
                usage: "transfer_src",
            });
        }

        // VUID-VkBlitImageInfo2-dstImage-00224
        if !dst_image.usage().transfer_dst {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        // VUID-VkBlitImageInfo2-srcImage-01999
        if !src_image.format_features().blit_src {
            return Err(CopyError::MissingFormatFeature {
                resource: CopyErrorResource::Source,
                format_feature: "blit_src",
            });
        }

        // VUID-VkBlitImageInfo2-dstImage-02000
        if !dst_image.format_features().blit_dst {
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

        if !(src_image_aspects.color && dst_image_aspects.color) {
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

        // VUID-VkBlitImageInfo2-dstImage-00234
        if dst_image.samples() != SampleCount::Sample1 {
            return Err(CopyError::SampleCountInvalid {
                resource: CopyErrorResource::Destination,
                sample_count: dst_image.samples(),
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
        if !src_image_aspects.color && filter != Filter::Nearest {
            return Err(CopyError::FilterNotSupportedByFormat);
        }

        match filter {
            Filter::Nearest => (),
            Filter::Linear => {
                // VUID-VkBlitImageInfo2-filter-02001
                if !src_image.format_features().sampled_image_filter_linear {
                    return Err(CopyError::FilterNotSupportedByFormat);
                }
            }
            Filter::Cubic => {
                if !device.enabled_extensions().ext_filter_cubic {
                    return Err(CopyError::ExtensionNotEnabled {
                        extension: "ext_filter_cubic",
                        reason: "the specified filter was Cubic",
                    });
                }

                // VUID-VkBlitImageInfo2-filter-02002
                if !src_image.format_features().sampled_image_filter_cubic {
                    return Err(CopyError::FilterNotSupportedByFormat);
                }

                // VUID-VkBlitImageInfo2-filter-00237
                if !matches!(src_image.dimensions(), ImageDimensions::Dim2d { .. }) {
                    return Err(CopyError::FilterNotSupportedForImageType);
                }
            }
        }

        let same_image = src_image_inner.image == dst_image_inner.image;
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
                                     image_aspects: &ImageAspects,
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

                // VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask
                assert!(subresource.aspects != ImageAspects::none());

                // VUID-VkBlitImageInfo2-aspectMask-00241
                // VUID-VkBlitImageInfo2-aspectMask-00242
                if !image_aspects.contains(&subresource.aspects) {
                    return Err(CopyError::AspectsNotAllowed {
                        resource,
                        region_index,
                        aspects: subresource.aspects,
                        allowed_aspects: *image_aspects,
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
                &src_image_aspects,
                src_subresource,
            )?;
            let dst_subresource_extent = check_subresource(
                CopyErrorResource::Destination,
                dst_image,
                &dst_image_aspects,
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
            if same_image {
                let src_region_index = region_index;
                let src_subresource_axes = [
                    src_image_inner.first_mipmap_level + src_subresource.mip_level
                        ..src_image_inner.first_mipmap_level + src_subresource.mip_level + 1,
                    src_image_inner.first_layer + src_subresource.array_layers.start
                        ..src_image_inner.first_layer + src_subresource.array_layers.end,
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
                        dst_image_inner.first_mipmap_level + dst_subresource.mip_level
                            ..dst_image_inner.first_mipmap_level + dst_subresource.mip_level + 1,
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

        Ok(())
    }

    /// Clears a color image with a specific value.
    pub fn clear_color_image(
        &mut self,
        mut clear_info: ClearColorImageInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_clear_color_image(&mut clear_info)?;

        unsafe {
            self.inner.clear_color_image(clear_info)?;
        }

        Ok(self)
    }

    fn validate_clear_color_image(
        &self,
        clear_info: &mut ClearColorImageInfo,
    ) -> Result<(), CopyError> {
        // VUID-vkCmdClearColorImage-renderpass
        if self.render_pass_state.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdClearColorImage-commandBuffer-cmdpool
        if !(self.queue_family().supports_graphics() || self.queue_family().supports_compute()) {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &mut ClearColorImageInfo {
            ref image,
            image_layout,
            clear_value,
            ref regions,
            _ne: _,
        } = clear_info;

        let device = self.device();

        // VUID-vkCmdClearColorImage-commonparent
        assert_eq!(device, image.device());

        // VUID-vkCmdClearColorImage-image-00002
        if !image.usage().transfer_dst {
            return Err(CopyError::MissingUsage {
                resource: CopyErrorResource::Destination,
                usage: "transfer_dst",
            });
        }

        // VUID-vkCmdClearColorImage-image-01993
        if !image.format_features().transfer_dst {
            return Err(CopyError::MissingFormatFeature {
                resource: CopyErrorResource::Destination,
                format_feature: "transfer_dst",
            });
        }

        let image_aspects = image.format().aspects();

        // VUID-vkCmdClearColorImage-image-00007
        if image_aspects.depth || image_aspects.stencil {
            return Err(CopyError::FormatNotSupported {
                resource: CopyErrorResource::Destination,
                format: image.format(),
            });
        }

        // VUID-vkCmdClearColorImage-image-00007
        if image.format().compression().is_some() {
            return Err(CopyError::FormatNotSupported {
                resource: CopyErrorResource::Destination,
                format: image.format(),
            });
        }

        // VUID-vkCmdClearColorImage-image-01545
        if image.format().ycbcr_chroma_sampling().is_some() {
            return Err(CopyError::FormatNotSupported {
                resource: CopyErrorResource::Destination,
                format: image.format(),
            });
        }

        // VUID-vkCmdClearColorImage-imageLayout-01394
        if !matches!(
            image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Destination,
                image_layout,
            });
        }

        for (region_index, subresource_range) in regions.iter().enumerate() {
            // VUID-VkImageSubresourceRange-aspectMask-requiredbitmask
            assert!(subresource_range.aspects != ImageAspects::none());

            // VUID-vkCmdClearColorImage-aspectMask-02498
            if !image_aspects.contains(&subresource_range.aspects) {
                return Err(CopyError::AspectsNotAllowed {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    aspects: subresource_range.aspects,
                    allowed_aspects: image_aspects,
                });
            }

            // VUID-VkImageSubresourceRange-levelCount-01720
            assert!(!subresource_range.mip_levels.is_empty());

            // VUID-vkCmdClearColorImage-baseMipLevel-01470
            // VUID-vkCmdClearColorImage-pRanges-01692
            if subresource_range.mip_levels.end > image.mip_levels() {
                return Err(CopyError::MipLevelsOutOfRange {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    mip_levels_range_end: subresource_range.mip_levels.end,
                    image_mip_levels: image.dimensions().array_layers(),
                });
            }

            // VUID-VkImageSubresourceRange-layerCount-01721
            assert!(!subresource_range.array_layers.is_empty());

            // VUID-vkCmdClearDepthStencilImage-baseArrayLayer-01476
            // VUID-vkCmdClearDepthStencilImage-pRanges-01695
            if subresource_range.array_layers.end > image.dimensions().array_layers() {
                return Err(CopyError::ArrayLayersOutOfRange {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    array_layers_range_end: subresource_range.array_layers.end,
                    image_array_layers: image.dimensions().array_layers(),
                });
            }
        }

        Ok(())
    }

    /// Clears a depth/stencil image with a specific value.
    pub fn clear_depth_stencil_image(
        &mut self,
        mut clear_info: ClearDepthStencilImageInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_clear_depth_stencil_image(&mut clear_info)?;

        unsafe {
            self.inner.clear_depth_stencil_image(clear_info)?;
        }

        Ok(self)
    }

    fn validate_clear_depth_stencil_image(
        &self,
        clear_info: &mut ClearDepthStencilImageInfo,
    ) -> Result<(), CopyError> {
        // VUID-vkCmdClearDepthStencilImage-renderpass
        if self.render_pass_state.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdClearDepthStencilImage-commandBuffer-cmdpool
        if !self.queue_family().supports_graphics() {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &mut ClearDepthStencilImageInfo {
            ref image,
            image_layout,
            clear_value,
            ref regions,
            _ne: _,
        } = clear_info;

        let device = self.device();

        // VUID-vkCmdClearDepthStencilImage-commonparent
        assert_eq!(device, image.device());

        // VUID-vkCmdClearDepthStencilImage-pRanges-02659
        // VUID-vkCmdClearDepthStencilImage-pRanges-02660
        if !image.usage().transfer_dst {
            if !image.usage().transfer_dst {
                return Err(CopyError::MissingUsage {
                    resource: CopyErrorResource::Destination,
                    usage: "transfer_dst",
                });
            }
        }

        // VUID-vkCmdClearDepthStencilImage-image-01994
        if !image.format_features().transfer_dst {
            return Err(CopyError::MissingFormatFeature {
                resource: CopyErrorResource::Destination,
                format_feature: "transfer_dst",
            });
        }

        let image_aspects = image.format().aspects();

        // VUID-vkCmdClearDepthStencilImage-image-00014
        if !(image_aspects.depth || image_aspects.stencil) {
            return Err(CopyError::FormatNotSupported {
                resource: CopyErrorResource::Destination,
                format: image.format(),
            });
        }

        // VUID-vkCmdClearDepthStencilImage-imageLayout-00012
        if !matches!(
            image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(CopyError::ImageLayoutInvalid {
                resource: CopyErrorResource::Destination,
                image_layout,
            });
        }

        // VUID-VkClearDepthStencilValue-depth-00022
        if !device.enabled_extensions().ext_depth_range_unrestricted
            && !(0.0..=1.0).contains(&clear_value.depth)
        {
            return Err(CopyError::ExtensionNotEnabled {
                extension: "ext_depth_range_unrestricted",
                reason: "clear_value.depth was not between 0.0 and 1.0 inclusive",
            });
        }

        for (region_index, subresource_range) in regions.iter().enumerate() {
            // VUID-VkImageSubresourceRange-aspectMask-requiredbitmask
            assert!(subresource_range.aspects != ImageAspects::none());

            // VUID-vkCmdClearDepthStencilImage-aspectMask-02824
            // VUID-vkCmdClearDepthStencilImage-image-02825
            // VUID-vkCmdClearDepthStencilImage-image-02826
            if !image_aspects.contains(&subresource_range.aspects) {
                return Err(CopyError::AspectsNotAllowed {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    aspects: subresource_range.aspects,
                    allowed_aspects: image_aspects,
                });
            }

            // VUID-VkImageSubresourceRange-levelCount-01720
            assert!(!subresource_range.mip_levels.is_empty());

            // VUID-vkCmdClearDepthStencilImage-baseMipLevel-01474
            // VUID-vkCmdClearDepthStencilImage-pRanges-01694
            if subresource_range.mip_levels.end > image.mip_levels() {
                return Err(CopyError::MipLevelsOutOfRange {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    mip_levels_range_end: subresource_range.mip_levels.end,
                    image_mip_levels: image.dimensions().array_layers(),
                });
            }

            // VUID-VkImageSubresourceRange-layerCount-01721
            assert!(!subresource_range.array_layers.is_empty());

            // VUID-vkCmdClearDepthStencilImage-baseArrayLayer-01476
            // VUID-vkCmdClearDepthStencilImage-pRanges-01695
            if subresource_range.array_layers.end > image.dimensions().array_layers() {
                return Err(CopyError::ArrayLayersOutOfRange {
                    resource: CopyErrorResource::Destination,
                    region_index,
                    array_layers_range_end: subresource_range.array_layers.end,
                    image_array_layers: image.dimensions().array_layers(),
                });
            }
        }

        Ok(())
    }

    /// Resolves a multisampled image into a single-sampled image.
    ///
    /// # Panics
    ///
    /// - Panics if `src_image` or `dst_image` were not created from the same device
    ///   as `self`.
    pub fn resolve_image(
        &mut self,
        mut resolve_image_info: ResolveImageInfo,
    ) -> Result<&mut Self, CopyError> {
        self.validate_resolve_image(&mut resolve_image_info)?;

        unsafe {
            self.inner.resolve_image(resolve_image_info)?;
        }

        Ok(self)
    }

    fn validate_resolve_image(
        &self,
        resolve_image_info: &mut ResolveImageInfo,
    ) -> Result<(), CopyError> {
        // VUID-vkCmdResolveImage2-renderpass
        if self.render_pass_state.is_some() {
            return Err(CopyError::ForbiddenInsideRenderPass);
        }

        // VUID-vkCmdResolveImage2-commandBuffer-cmdpool
        if !self.queue_family().supports_graphics() {
            return Err(CopyError::NotSupportedByQueueFamily);
        }

        let &mut ResolveImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = resolve_image_info;

        let device = self.device();

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
                allowed_sample_counts: SampleCounts {
                    sample1: false,
                    sample2: true,
                    sample4: true,
                    sample8: true,
                    sample16: true,
                    sample32: true,
                    sample64: true,
                },
            });
        }

        // VUID-VkResolveImageInfo2-dstImage-00259
        if dst_image.samples() != SampleCount::Sample1 {
            return Err(CopyError::SampleCountInvalid {
                resource: CopyErrorResource::Destination,
                sample_count: dst_image.samples(),
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

        // VUID-VkResolveImageInfo2-dstImage-02003
        if !dst_image.format_features().color_attachment {
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
        debug_assert!(src_image.format().aspects().color && dst_image.format().aspects().color);

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

                // VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask
                // VUID-VkImageResolve2-aspectMask-00266
                if subresource.aspects
                    != (ImageAspects {
                        color: true,
                        ..ImageAspects::none()
                    })
                {
                    return Err(CopyError::AspectsNotAllowed {
                        resource,
                        region_index,
                        aspects: subresource.aspects,
                        allowed_aspects: ImageAspects {
                            color: true,
                            ..ImageAspects::none()
                        },
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
                                       image_type: ImageType,
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

        Ok(())
    }
}

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdBlitImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn blit_image(
        &mut self,
        blit_image_info: BlitImageInfo,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            blit_image_info: BlitImageInfo,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "blit_image"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.blit_image(&self.blit_image_info);
            }
        }

        let &BlitImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            filter,
            _ne: _,
        } = &blit_image_info;

        // if its the same image in source and destination, we need to lock it once
        let src_key = (
            src_image.conflict_key(),
            src_image.current_mip_levels_access(),
            src_image.current_array_layers_access(),
        );
        let dst_key = (
            dst_image.conflict_key(),
            dst_image.current_mip_levels_access(),
            dst_image.current_array_layers_access(),
        );

        let resources: SmallVec<[_; 8]> = regions
            .iter()
            .flat_map(|region| {
                let &ImageBlit {
                    ref src_subresource,
                    src_offsets,
                    ref dst_subresource,
                    dst_offsets,
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

        self.commands.push(Box::new(Cmd { blit_image_info }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdClearColorImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    pub unsafe fn clear_color_image(
        &mut self,
        clear_info: ClearColorImageInfo,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            clear_info: ClearColorImageInfo,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "clear_color_image"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.clear_color_image(&self.clear_info);
            }
        }

        let &ClearColorImageInfo {
            ref image,
            image_layout,
            clear_value,
            ref regions,
            _ne: _,
        } = &clear_info;

        let resources: SmallVec<[_; 8]> = regions
            .iter()
            .cloned()
            .flat_map(|subresource_range| {
                [(
                    "image".into(),
                    Resource::Image {
                        image: image.clone(),
                        subresource_range,
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
                        start_layout: image_layout,
                        end_layout: image_layout,
                    },
                )]
            })
            .collect();

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd { clear_info }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdClearDepthStencilImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    pub unsafe fn clear_depth_stencil_image(
        &mut self,
        clear_info: ClearDepthStencilImageInfo,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            clear_info: ClearDepthStencilImageInfo,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "clear_depth_stencil_image"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.clear_depth_stencil_image(&self.clear_info);
            }
        }

        let &ClearDepthStencilImageInfo {
            ref image,
            image_layout,
            clear_value,
            ref regions,
            _ne: _,
        } = &clear_info;

        let resources: SmallVec<[_; 8]> = regions
            .iter()
            .cloned()
            .flat_map(|subresource_range| {
                [(
                    "image".into(),
                    Resource::Image {
                        image: image.clone(),
                        subresource_range,
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
                        start_layout: image_layout,
                        end_layout: image_layout,
                    },
                )]
            })
            .collect();

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd { clear_info }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdResolveImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn resolve_image(
        &mut self,
        resolve_image_info: ResolveImageInfo,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            resolve_image_info: ResolveImageInfo,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "resolve_image"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.resolve_image(&self.resolve_image_info);
            }
        }

        let &ResolveImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = &resolve_image_info;

        let resources: SmallVec<[_; 8]> = regions
            .iter()
            .flat_map(|region| {
                let &ImageResolve {
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

        self.commands.push(Box::new(Cmd { resolve_image_info }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdBlitImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn blit_image(&mut self, blit_image_info: &BlitImageInfo) {
        let &BlitImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            filter,
            _ne,
        } = blit_image_info;

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
                    let &ImageBlit {
                        ref src_subresource,
                        src_offsets,
                        ref dst_subresource,
                        dst_offsets,
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
                src_image: src_image_inner.image.internal_object(),
                src_image_layout: src_image_layout.into(),
                dst_image: dst_image_inner.image.internal_object(),
                dst_image_layout: dst_image_layout.into(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                filter: filter.into(),
                ..Default::default()
            };

            if self.device.api_version() >= Version::V1_3 {
                fns.v1_3.cmd_blit_image2(self.handle, &blit_image_info);
            } else {
                fns.khr_copy_commands2
                    .cmd_blit_image2_khr(self.handle, &blit_image_info);
            }
        } else {
            let regions: SmallVec<[_; 8]> = regions
                .into_iter()
                .map(|region| {
                    let &ImageBlit {
                        ref src_subresource,
                        src_offsets,
                        ref dst_subresource,
                        dst_offsets,
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

            fns.v1_0.cmd_blit_image(
                self.handle,
                src_image_inner.image.internal_object(),
                src_image_layout.into(),
                dst_image_inner.image.internal_object(),
                dst_image_layout.into(),
                regions.len() as u32,
                regions.as_ptr(),
                filter.into(),
            );
        }
    }

    /// Calls `vkCmdClearColorImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    pub unsafe fn clear_color_image(&mut self, clear_info: &ClearColorImageInfo) {
        let &ClearColorImageInfo {
            ref image,
            image_layout,
            clear_value,
            ref regions,
            _ne: _,
        } = clear_info;

        if regions.is_empty() {
            return;
        }

        let clear_value = clear_value.into();
        let ranges: SmallVec<[_; 8]> = regions
            .iter()
            .cloned()
            .map(ash::vk::ImageSubresourceRange::from)
            .collect();

        let fns = self.device.fns();
        fns.v1_0.cmd_clear_color_image(
            self.handle,
            image.inner().image.internal_object(),
            image_layout.into(),
            &clear_value,
            ranges.len() as u32,
            ranges.as_ptr(),
        );
    }

    /// Calls `vkCmdClearDepthStencilImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    pub unsafe fn clear_depth_stencil_image(&mut self, clear_info: &ClearDepthStencilImageInfo) {
        let &ClearDepthStencilImageInfo {
            ref image,
            image_layout,
            clear_value,
            ref regions,
            _ne: _,
        } = clear_info;

        if regions.is_empty() {
            return;
        }

        let clear_value = clear_value.into();
        let ranges: SmallVec<[_; 8]> = regions
            .iter()
            .cloned()
            .map(ash::vk::ImageSubresourceRange::from)
            .collect();

        let fns = self.device.fns();
        fns.v1_0.cmd_clear_depth_stencil_image(
            self.handle,
            image.inner().image.internal_object(),
            image_layout.into(),
            &clear_value,
            ranges.len() as u32,
            ranges.as_ptr(),
        );
    }

    /// Calls `vkCmdResolveImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn resolve_image(&mut self, resolve_image_info: &ResolveImageInfo) {
        let &ResolveImageInfo {
            ref src_image,
            src_image_layout,
            ref dst_image,
            dst_image_layout,
            ref regions,
            _ne: _,
        } = resolve_image_info;

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
                    let &ImageResolve {
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
                src_image: src_image_inner.image.internal_object(),
                src_image_layout: src_image_layout.into(),
                dst_image: dst_image_inner.image.internal_object(),
                dst_image_layout: dst_image_layout.into(),
                region_count: regions.len() as u32,
                p_regions: regions.as_ptr(),
                ..Default::default()
            };

            if self.device.api_version() >= Version::V1_3 {
                fns.v1_3
                    .cmd_resolve_image2(self.handle, &resolve_image_info);
            } else {
                fns.khr_copy_commands2
                    .cmd_resolve_image2_khr(self.handle, &resolve_image_info);
            }
        } else {
            let regions: SmallVec<[_; 8]> = regions
                .into_iter()
                .map(|region| {
                    let &ImageResolve {
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

            fns.v1_0.cmd_resolve_image(
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
}

/// Parameters to blit image data.
#[derive(Clone, Debug)]
pub struct BlitImageInfo {
    /// The image to blit from.
    ///
    /// There is no default value.
    pub src_image: Arc<dyn ImageAccess>,

    /// The layout used for `src_image` during the blit operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferSrcOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferSrcOptimal`].
    pub src_image_layout: ImageLayout,

    /// The image to blit to.
    ///
    /// There is no default value.
    pub dst_image: Arc<dyn ImageAccess>,

    /// The layout used for `dst_image` during the blit operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub dst_image_layout: ImageLayout,

    /// The regions of both images to blit between.
    ///
    /// The default value is a single region, covering the first mip level, and the smallest of the
    /// array layers of the two images. The whole extent of each image is covered, scaling if
    /// necessary. All aspects of each image are selected, or `plane0` if the image is multi-planar.
    pub regions: SmallVec<[ImageBlit; 1]>,

    /// The filter to use for sampling `src_image` when the `src_extent` and
    /// `dst_extent` of a region are not the same size.
    ///
    /// The default value is [`Filter::Nearest`].
    pub filter: Filter,

    pub _ne: crate::NonExhaustive,
}

impl BlitImageInfo {
    /// Returns a `BlitImageInfo` with the specified `src_image` and `dst_image`.
    #[inline]
    pub fn images(src_image: Arc<dyn ImageAccess>, dst_image: Arc<dyn ImageAccess>) -> Self {
        let min_array_layers = src_image
            .dimensions()
            .array_layers()
            .min(dst_image.dimensions().array_layers());
        let region = ImageBlit {
            src_subresource: ImageSubresourceLayers {
                array_layers: 0..min_array_layers,
                ..src_image.subresource_layers()
            },
            src_offsets: [[0; 3], src_image.dimensions().width_height_depth()],
            dst_subresource: ImageSubresourceLayers {
                array_layers: 0..min_array_layers,
                ..dst_image.subresource_layers()
            },
            dst_offsets: [[0; 3], dst_image.dimensions().width_height_depth()],
            ..Default::default()
        };

        Self {
            src_image,
            src_image_layout: ImageLayout::TransferSrcOptimal,
            dst_image,
            dst_image_layout: ImageLayout::TransferDstOptimal,
            regions: smallvec![region],
            filter: Filter::Nearest,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A region of data to blit between images.
#[derive(Clone, Debug)]
pub struct ImageBlit {
    /// The subresource of `src_image` to blit from.
    ///
    /// The default value is empty, which must be overridden.
    pub src_subresource: ImageSubresourceLayers,

    /// The offsets from the zero coordinate of `src_image`, defining two corners of the region
    /// to blit from.
    /// If the ordering of the two offsets differs between source and destination, the image will
    /// be flipped.
    ///
    /// The default value is `[[0; 3]; 2]`, which must be overridden.
    pub src_offsets: [[u32; 3]; 2],

    /// The subresource of `dst_image` to blit to.
    ///
    /// The default value is empty, which must be overridden.
    pub dst_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of `dst_image` defining two corners of the
    /// region to blit to.
    /// If the ordering of the two offsets differs between source and destination, the image will
    /// be flipped.
    ///
    /// The default value is `[[0; 3]; 2]`, which must be overridden.
    pub dst_offsets: [[u32; 3]; 2],

    pub _ne: crate::NonExhaustive,
}

impl Default for ImageBlit {
    #[inline]
    fn default() -> Self {
        Self {
            src_subresource: ImageSubresourceLayers {
                aspects: ImageAspects::none(),
                mip_level: 0,
                array_layers: 0..0,
            },
            src_offsets: [[0; 3]; 2],
            dst_subresource: ImageSubresourceLayers {
                aspects: ImageAspects::none(),
                mip_level: 0,
                array_layers: 0..0,
            },
            dst_offsets: [[0; 3]; 2],
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to clear a color image.
#[derive(Clone, Debug)]
pub struct ClearColorImageInfo {
    /// The image to clear.
    ///
    /// There is no default value.
    pub image: Arc<dyn ImageAccess>,

    /// The layout used for `image` during the clear operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub image_layout: ImageLayout,

    /// The color value to clear the image to.
    ///
    /// The default value is `ClearColorValue::Float([0.0; 4])`.
    pub clear_value: ClearColorValue,

    /// The subresource ranges of `image` to clear.
    ///
    /// The default value is a single region, covering the whole image.
    pub regions: SmallVec<[ImageSubresourceRange; 1]>,

    pub _ne: crate::NonExhaustive,
}

impl ClearColorImageInfo {
    /// Returns a `ClearColorImageInfo` with the specified `image`.
    #[inline]
    pub fn image(image: Arc<dyn ImageAccess>) -> Self {
        let range = image.subresource_range();

        Self {
            image,
            image_layout: ImageLayout::TransferDstOptimal,
            clear_value: ClearColorValue::Float([0.0; 4]),
            regions: smallvec![range],
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to clear a depth/stencil image.
#[derive(Clone, Debug)]
pub struct ClearDepthStencilImageInfo {
    /// The image to clear.
    ///
    /// There is no default value.
    pub image: Arc<dyn ImageAccess>,

    /// The layout used for `image` during the clear operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub image_layout: ImageLayout,

    /// The depth/stencil values to clear the image to.
    ///
    /// The default value is zero for both.
    pub clear_value: ClearDepthStencilValue,

    /// The subresource ranges of `image` to clear.
    ///
    /// The default value is a single region, covering the whole image.
    pub regions: SmallVec<[ImageSubresourceRange; 1]>,

    pub _ne: crate::NonExhaustive,
}

impl ClearDepthStencilImageInfo {
    /// Returns a `ClearDepthStencilImageInfo` with the specified `image`.
    #[inline]
    pub fn image(image: Arc<dyn ImageAccess>) -> Self {
        let range = image.subresource_range();

        Self {
            image,
            image_layout: ImageLayout::TransferDstOptimal,
            clear_value: ClearDepthStencilValue::default(),
            regions: smallvec![range],
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to resolve image data.
#[derive(Clone, Debug)]
pub struct ResolveImageInfo {
    /// The multisampled image to resolve from.
    ///
    /// There is no default value.
    pub src_image: Arc<dyn ImageAccess>,

    /// The layout used for `src_image` during the resolve operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferSrcOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferSrcOptimal`].
    pub src_image_layout: ImageLayout,

    /// The non-multisampled image to resolve into.
    ///
    /// There is no default value.
    pub dst_image: Arc<dyn ImageAccess>,

    /// The layout used for `dst_image` during the resolve operation.
    ///
    /// The following layouts are allowed:
    /// - [`ImageLayout::TransferDstOptimal`]
    /// - [`ImageLayout::General`]
    ///
    /// The default value is [`ImageLayout::TransferDstOptimal`].
    pub dst_image_layout: ImageLayout,

    /// The regions of both images to resolve between.
    ///
    /// The default value is a single region, covering the first mip level, and the smallest of the
    /// array layers and extent of the two images. All aspects of each image are selected, or
    /// `plane0` if the image is multi-planar.
    pub regions: SmallVec<[ImageResolve; 1]>,

    pub _ne: crate::NonExhaustive,
}

impl ResolveImageInfo {
    /// Returns a `ResolveImageInfo` with the specified `src_image` and `dst_image`.
    #[inline]
    pub fn images(src_image: Arc<dyn ImageAccess>, dst_image: Arc<dyn ImageAccess>) -> Self {
        let min_array_layers = src_image
            .dimensions()
            .array_layers()
            .min(dst_image.dimensions().array_layers());
        let region = ImageResolve {
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

/// A region of data to resolve between images.
#[derive(Clone, Debug)]
pub struct ImageResolve {
    /// The subresource of `src_image` to resolve from.
    ///
    /// The default value is empty, which must be overridden.
    pub src_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of `src_image` that resolving will start from.
    ///
    /// The default value is `[0; 3]`.
    pub src_offset: [u32; 3],

    /// The subresource of `dst_image` to resolve into.
    ///
    /// The default value is empty, which must be overridden.
    pub dst_subresource: ImageSubresourceLayers,

    /// The offset from the zero coordinate of `dst_image` that resolving will start from.
    ///
    /// The default value is `[0; 3]`.
    pub dst_offset: [u32; 3],

    /// The extent of texels to resolve.
    ///
    /// The default value is `[0; 3]`, which must be overridden.
    pub extent: [u32; 3],

    pub _ne: crate::NonExhaustive,
}

impl Default for ImageResolve {
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
