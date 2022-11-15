// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    ClearColorImageInfo, ClearDepthStencilImageInfo, ClearError, CommandBufferBuilder,
    FillBufferInfo,
};
use crate::{
    buffer::{BufferAccess, BufferContents, BufferUsage, TypedBufferAccess},
    command_buffer::allocator::CommandBufferAllocator,
    device::{DeviceOwned, QueueFlags},
    format::FormatFeatures,
    image::{ImageAccess, ImageAspects, ImageLayout, ImageUsage},
    DeviceSize, RequiresOneOf, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::{mem::size_of_val, sync::Arc};

impl<L, A> CommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Clears a color image with a specific value.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn clear_color_image(
        &mut self,
        clear_info: ClearColorImageInfo,
    ) -> Result<&mut Self, ClearError> {
        self.validate_clear_color_image(&clear_info)?;

        unsafe { Ok(self.clear_color_image_unchecked(clear_info)) }
    }

    fn validate_clear_color_image(
        &self,
        clear_info: &ClearColorImageInfo,
    ) -> Result<(), ClearError> {
        let device = self.device();

        // VUID-vkCmdClearColorImage-renderpass
        if self.current_state.render_pass.is_some() {
            return Err(ClearError::ForbiddenInsideRenderPass);
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdClearColorImage-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(ClearError::NotSupportedByQueueFamily);
        }

        let &ClearColorImageInfo {
            ref image,
            image_layout,
            clear_value: _,
            ref regions,
            _ne: _,
        } = clear_info;

        // VUID-vkCmdClearColorImage-imageLayout-parameter
        image_layout.validate_device(device)?;

        // VUID-vkCmdClearColorImage-commonparent
        assert_eq!(device, image.device());

        // VUID-vkCmdClearColorImage-image-00002
        if !image.usage().intersects(ImageUsage::TRANSFER_DST) {
            return Err(ClearError::MissingUsage {
                usage: "transfer_dst",
            });
        }

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            // VUID-vkCmdClearColorImage-image-01993
            if !image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_DST)
            {
                return Err(ClearError::MissingFormatFeature {
                    format_feature: "transfer_dst",
                });
            }
        }

        let image_aspects = image.format().aspects();

        // VUID-vkCmdClearColorImage-image-00007
        if image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            return Err(ClearError::FormatNotSupported {
                format: image.format(),
            });
        }

        // VUID-vkCmdClearColorImage-image-00007
        if image.format().compression().is_some() {
            return Err(ClearError::FormatNotSupported {
                format: image.format(),
            });
        }

        // VUID-vkCmdClearColorImage-image-01545
        if image.format().ycbcr_chroma_sampling().is_some() {
            return Err(ClearError::FormatNotSupported {
                format: image.format(),
            });
        }

        // VUID-vkCmdClearColorImage-imageLayout-01394
        if !matches!(
            image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(ClearError::ImageLayoutInvalid { image_layout });
        }

        for (region_index, subresource_range) in regions.iter().enumerate() {
            // VUID-VkImageSubresourceRange-aspectMask-parameter
            subresource_range.aspects.validate_device(device)?;

            // VUID-VkImageSubresourceRange-aspectMask-requiredbitmask
            assert!(!subresource_range.aspects.is_empty());

            // VUID-vkCmdClearColorImage-aspectMask-02498
            if !image_aspects.contains(subresource_range.aspects) {
                return Err(ClearError::AspectsNotAllowed {
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
                return Err(ClearError::MipLevelsOutOfRange {
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
                return Err(ClearError::ArrayLayersOutOfRange {
                    region_index,
                    array_layers_range_end: subresource_range.array_layers.end,
                    image_array_layers: image.dimensions().array_layers(),
                });
            }
        }

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_color_image_unchecked(
        &mut self,
        clear_info: ClearColorImageInfo,
    ) -> &mut Self {
        let ClearColorImageInfo {
            image,
            image_layout,
            clear_value,
            regions,
            _ne: _,
        } = clear_info;

        if regions.is_empty() {
            return self;
        }

        let clear_value = clear_value.into();
        let ranges: SmallVec<[_; 8]> = regions
            .iter()
            .cloned()
            .map(ash::vk::ImageSubresourceRange::from)
            .collect();

        let fns = self.device().fns();
        (fns.v1_0.cmd_clear_color_image)(
            self.handle(),
            image.inner().image.handle(),
            image_layout.into(),
            &clear_value,
            ranges.len() as u32,
            ranges.as_ptr(),
        );

        self.resources.push(Box::new(image));

        // TODO: sync state update

        self
    }

    /// Clears a depth/stencil image with a specific value.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all images
    ///   that are accessed by the command.
    /// - All images that are accessed by the command must be in the expected image layout.
    #[inline]
    pub unsafe fn clear_depth_stencil_image(
        &mut self,
        clear_info: ClearDepthStencilImageInfo,
    ) -> Result<&mut Self, ClearError> {
        self.validate_clear_depth_stencil_image(&clear_info)?;

        unsafe { Ok(self.clear_depth_stencil_image_unchecked(clear_info)) }
    }

    fn validate_clear_depth_stencil_image(
        &self,
        clear_info: &ClearDepthStencilImageInfo,
    ) -> Result<(), ClearError> {
        let device = self.device();

        // VUID-vkCmdClearDepthStencilImage-renderpass
        if self.current_state.render_pass.is_some() {
            return Err(ClearError::ForbiddenInsideRenderPass);
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdClearDepthStencilImage-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(ClearError::NotSupportedByQueueFamily);
        }

        let &ClearDepthStencilImageInfo {
            ref image,
            image_layout,
            clear_value,
            ref regions,
            _ne: _,
        } = clear_info;

        // VUID-vkCmdClearDepthStencilImage-imageLayout-parameter
        image_layout.validate_device(device)?;

        // VUID-vkCmdClearDepthStencilImage-commonparent
        assert_eq!(device, image.device());

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            // VUID-vkCmdClearDepthStencilImage-image-01994
            if !image
                .format_features()
                .intersects(FormatFeatures::TRANSFER_DST)
            {
                return Err(ClearError::MissingFormatFeature {
                    format_feature: "transfer_dst",
                });
            }
        }

        let image_aspects = image.format().aspects();

        // VUID-vkCmdClearDepthStencilImage-image-00014
        if !image_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            return Err(ClearError::FormatNotSupported {
                format: image.format(),
            });
        }

        // VUID-vkCmdClearDepthStencilImage-imageLayout-00012
        if !matches!(
            image_layout,
            ImageLayout::TransferDstOptimal | ImageLayout::General
        ) {
            return Err(ClearError::ImageLayoutInvalid { image_layout });
        }

        // VUID-VkClearDepthStencilValue-depth-00022
        if !device.enabled_extensions().ext_depth_range_unrestricted
            && !(0.0..=1.0).contains(&clear_value.depth)
        {
            return Err(ClearError::RequirementNotMet {
                required_for: "`clear_info.clear_value.depth` is not between `0.0` and `1.0` \
                    inclusive",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["ext_depth_range_unrestricted"],
                    ..Default::default()
                },
            });
        }

        let mut image_aspects_used = ImageAspects::empty();

        for (region_index, subresource_range) in regions.iter().enumerate() {
            // VUID-VkImageSubresourceRange-aspectMask-parameter
            subresource_range.aspects.validate_device(device)?;

            // VUID-VkImageSubresourceRange-aspectMask-requiredbitmask
            assert!(!subresource_range.aspects.is_empty());

            // VUID-vkCmdClearDepthStencilImage-aspectMask-02824
            // VUID-vkCmdClearDepthStencilImage-image-02825
            // VUID-vkCmdClearDepthStencilImage-image-02826
            if !image_aspects.contains(subresource_range.aspects) {
                return Err(ClearError::AspectsNotAllowed {
                    region_index,
                    aspects: subresource_range.aspects,
                    allowed_aspects: image_aspects,
                });
            }

            image_aspects_used |= subresource_range.aspects;

            // VUID-VkImageSubresourceRange-levelCount-01720
            assert!(!subresource_range.mip_levels.is_empty());

            // VUID-vkCmdClearDepthStencilImage-baseMipLevel-01474
            // VUID-vkCmdClearDepthStencilImage-pRanges-01694
            if subresource_range.mip_levels.end > image.mip_levels() {
                return Err(ClearError::MipLevelsOutOfRange {
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
                return Err(ClearError::ArrayLayersOutOfRange {
                    region_index,
                    array_layers_range_end: subresource_range.array_layers.end,
                    image_array_layers: image.dimensions().array_layers(),
                });
            }
        }

        // VUID-vkCmdClearDepthStencilImage-pRanges-02658
        // VUID-vkCmdClearDepthStencilImage-pRanges-02659
        if image_aspects_used.intersects(ImageAspects::STENCIL)
            && !image.stencil_usage().intersects(ImageUsage::TRANSFER_DST)
        {
            return Err(ClearError::MissingUsage {
                usage: "transfer_dst",
            });
        }

        // VUID-vkCmdClearDepthStencilImage-pRanges-02660
        if !(image_aspects_used - ImageAspects::STENCIL).is_empty()
            && !image.usage().intersects(ImageUsage::TRANSFER_DST)
        {
            return Err(ClearError::MissingUsage {
                usage: "transfer_dst",
            });
        }

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn clear_depth_stencil_image_unchecked(
        &mut self,
        clear_info: ClearDepthStencilImageInfo,
    ) -> &mut Self {
        let ClearDepthStencilImageInfo {
            image,
            image_layout,
            clear_value,
            regions,
            _ne: _,
        } = clear_info;

        if regions.is_empty() {
            return self;
        }

        let clear_value = clear_value.into();
        let ranges: SmallVec<[_; 8]> = regions
            .iter()
            .cloned()
            .map(ash::vk::ImageSubresourceRange::from)
            .collect();

        let fns = self.device().fns();
        (fns.v1_0.cmd_clear_depth_stencil_image)(
            self.handle(),
            image.inner().image.handle(),
            image_layout.into(),
            &clear_value,
            ranges.len() as u32,
            ranges.as_ptr(),
        );

        self.resources.push(Box::new(image));

        // TODO: sync state update

        self
    }

    /// Fills a region of a buffer with repeated copies of a value.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatedly written through the entire buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `dst_buffer` was not created from the same device as `self`.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers
    ///   that are accessed by the command.
    #[inline]
    pub unsafe fn fill_buffer(
        &mut self,
        fill_buffer_info: FillBufferInfo,
    ) -> Result<&mut Self, ClearError> {
        self.validate_fill_buffer(&fill_buffer_info)?;

        unsafe { Ok(self.fill_buffer_unchecked(fill_buffer_info)) }
    }

    fn validate_fill_buffer(&self, fill_buffer_info: &FillBufferInfo) -> Result<(), ClearError> {
        let device = self.device();

        // VUID-vkCmdFillBuffer-renderpass
        if self.current_state.render_pass.is_some() {
            return Err(ClearError::ForbiddenInsideRenderPass);
        }

        let queue_family_properties = self.queue_family_properties();

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1 {
            // VUID-vkCmdFillBuffer-commandBuffer-cmdpool
            if !queue_family_properties
                .queue_flags
                .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
            {
                return Err(ClearError::NotSupportedByQueueFamily);
            }
        } else {
            // VUID-vkCmdFillBuffer-commandBuffer-00030
            if !queue_family_properties
                .queue_flags
                .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
            {
                return Err(ClearError::NotSupportedByQueueFamily);
            }
        }

        let &FillBufferInfo {
            data: _,
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
        if !dst_buffer.usage().intersects(BufferUsage::TRANSFER_DST) {
            return Err(ClearError::MissingUsage {
                usage: "transfer_dst",
            });
        }

        // VUID-vkCmdFillBuffer-dstOffset-00024
        // VUID-vkCmdFillBuffer-size-00027
        if dst_offset + size > dst_buffer.size() {
            return Err(ClearError::RegionOutOfBufferBounds {
                region_index: 0,
                offset_range_end: dst_offset + size,
                buffer_size: dst_buffer.size(),
            });
        }

        // VUID-vkCmdFillBuffer-dstOffset-00025
        if (dst_buffer_inner.offset + dst_offset) % 4 != 0 {
            return Err(ClearError::OffsetNotAlignedForBuffer {
                region_index: 0,
                offset: dst_buffer_inner.offset + dst_offset,
                required_alignment: 4,
            });
        }

        // VUID-vkCmdFillBuffer-size-00028
        if size % 4 != 0 {
            return Err(ClearError::SizeNotAlignedForBuffer {
                region_index: 0,
                size,
                required_alignment: 4,
            });
        }

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn fill_buffer_unchecked(&mut self, fill_buffer_info: FillBufferInfo) -> &mut Self {
        let FillBufferInfo {
            data,
            dst_buffer,
            dst_offset,
            size,
            _ne: _,
        } = fill_buffer_info;

        let dst_buffer_inner = dst_buffer.inner();

        let fns = self.device().fns();
        (fns.v1_0.cmd_fill_buffer)(
            self.handle(),
            dst_buffer_inner.buffer.handle(),
            dst_offset,
            size,
            data,
        );

        self.resources.push(Box::new(dst_buffer));

        // TODO: sync state update

        self
    }

    /// Writes data to a region of a buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `dst_buffer` was not created from the same device as `self`.
    ///
    /// # Safety
    ///
    /// - Appropriate synchronization must be provided for all buffers
    ///   that are accessed by the command.
    #[inline]
    pub unsafe fn update_buffer<B, D>(
        &mut self,
        data: &D,
        dst_buffer: Arc<B>,
        dst_offset: DeviceSize,
    ) -> Result<&mut Self, ClearError>
    where
        B: TypedBufferAccess<Content = D> + 'static,
        D: BufferContents + ?Sized,
    {
        self.validate_update_buffer(data, &dst_buffer, dst_offset)?;

        unsafe { Ok(self.update_buffer_unchecked(data, dst_buffer, dst_offset)) }
    }

    fn validate_update_buffer(
        &self,
        data: &impl ?Sized,
        dst_buffer: &dyn BufferAccess,
        dst_offset: DeviceSize,
    ) -> Result<(), ClearError> {
        let device = self.device();

        // VUID-vkCmdUpdateBuffer-renderpass
        if self.current_state.render_pass.is_some() {
            return Err(ClearError::ForbiddenInsideRenderPass);
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdUpdateBuffer-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::TRANSFER | QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(ClearError::NotSupportedByQueueFamily);
        }

        let dst_buffer_inner = dst_buffer.inner();

        // VUID-vkCmdUpdateBuffer-commonparent
        assert_eq!(device, dst_buffer.device());

        // VUID-vkCmdUpdateBuffer-dataSize-arraylength
        assert!(size_of_val(data) != 0);

        // VUID-vkCmdUpdateBuffer-dstBuffer-00034
        if !dst_buffer.usage().intersects(BufferUsage::TRANSFER_DST) {
            return Err(ClearError::MissingUsage {
                usage: "transfer_dst",
            });
        }

        // VUID-vkCmdUpdateBuffer-dstOffset-00032
        // VUID-vkCmdUpdateBuffer-dataSize-00033
        if dst_offset + size_of_val(data) as DeviceSize > dst_buffer.size() {
            return Err(ClearError::RegionOutOfBufferBounds {
                region_index: 0,
                offset_range_end: dst_offset + size_of_val(data) as DeviceSize,
                buffer_size: dst_buffer.size(),
            });
        }

        // VUID-vkCmdUpdateBuffer-dstOffset-00036
        if (dst_buffer_inner.offset + dst_offset) % 4 != 0 {
            return Err(ClearError::OffsetNotAlignedForBuffer {
                region_index: 0,
                offset: dst_buffer_inner.offset + dst_offset,
                required_alignment: 4,
            });
        }

        // VUID-vkCmdUpdateBuffer-dataSize-00037
        if size_of_val(data) > 65536 {
            return Err(ClearError::DataTooLarge {
                size: size_of_val(data) as DeviceSize,
                max: 65536,
            });
        }

        // VUID-vkCmdUpdateBuffer-dataSize-00038
        if size_of_val(data) % 4 != 0 {
            return Err(ClearError::SizeNotAlignedForBuffer {
                region_index: 0,
                size: size_of_val(data) as DeviceSize,
                required_alignment: 4,
            });
        }

        // TODO: sync check

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_buffer_unchecked(
        &mut self,
        data: &(impl BufferContents + ?Sized),
        dst_buffer: Arc<dyn BufferAccess>,
        dst_offset: DeviceSize,
    ) -> &mut Self {
        let dst_buffer_inner = dst_buffer.inner();

        let fns = self.device().fns();
        (fns.v1_0.cmd_update_buffer)(
            self.handle(),
            dst_buffer_inner.buffer.handle(),
            dst_buffer_inner.offset + dst_offset,
            size_of_val(data) as DeviceSize,
            data.as_bytes().as_ptr() as *const _,
        );

        self.resources.push(Box::new(dst_buffer));

        // TODO: sync state update

        self
    }
}
