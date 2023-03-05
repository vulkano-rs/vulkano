// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::{BufferContents, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator,
        synced::{Command, Resource, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, ResourceInCommand, ResourceUseRef,
    },
    device::{DeviceOwned, QueueFlags},
    format::{ClearColorValue, ClearDepthStencilValue, Format, FormatFeatures},
    image::{ImageAccess, ImageAspects, ImageLayout, ImageSubresourceRange, ImageUsage},
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStages},
    DeviceSize, RequirementNotMet, RequiresOneOf, SafeDeref, Version, VulkanObject,
};
use smallvec::{smallvec, SmallVec};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::size_of_val,
    sync::Arc,
};

/// # Commands to fill resources with new data.
impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Clears a color image with a specific value.
    pub fn clear_color_image(
        &mut self,
        clear_info: ClearColorImageInfo,
    ) -> Result<&mut Self, ClearError> {
        self.validate_clear_color_image(&clear_info)?;

        unsafe {
            self.inner.clear_color_image(clear_info)?;
        }

        Ok(self)
    }

    fn validate_clear_color_image(
        &self,
        clear_info: &ClearColorImageInfo,
    ) -> Result<(), ClearError> {
        let device = self.device();

        // VUID-vkCmdClearColorImage-renderpass
        if self.render_pass_state.is_some() {
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

        Ok(())
    }

    /// Clears a depth/stencil image with a specific value.
    pub fn clear_depth_stencil_image(
        &mut self,
        clear_info: ClearDepthStencilImageInfo,
    ) -> Result<&mut Self, ClearError> {
        self.validate_clear_depth_stencil_image(&clear_info)?;

        unsafe {
            self.inner.clear_depth_stencil_image(clear_info)?;
        }

        Ok(self)
    }

    fn validate_clear_depth_stencil_image(
        &self,
        clear_info: &ClearDepthStencilImageInfo,
    ) -> Result<(), ClearError> {
        let device = self.device();

        // VUID-vkCmdClearDepthStencilImage-renderpass
        if self.render_pass_state.is_some() {
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
    pub fn fill_buffer(
        &mut self,
        dst_buffer: Subbuffer<[u32]>,
        data: u32,
    ) -> Result<&mut Self, ClearError> {
        self.validate_fill_buffer(&dst_buffer, data)?;

        unsafe {
            self.inner.fill_buffer(dst_buffer, data)?;
        }

        Ok(self)
    }

    fn validate_fill_buffer(
        &self,
        dst_buffer: &Subbuffer<[u32]>,
        _data: u32,
    ) -> Result<(), ClearError> {
        let device = self.device();

        // VUID-vkCmdFillBuffer-renderpass
        if self.render_pass_state.is_some() {
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

        // VUID-vkCmdFillBuffer-commonparent
        assert_eq!(device, dst_buffer.device());

        // VUID-vkCmdFillBuffer-size-00026
        // Guaranteed by `Subbuffer`

        // VUID-vkCmdFillBuffer-dstBuffer-00029
        if !dst_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_DST)
        {
            return Err(ClearError::MissingUsage {
                usage: "transfer_dst",
            });
        }

        // VUID-vkCmdFillBuffer-dstOffset-00024
        // VUID-vkCmdFillBuffer-size-00027
        // Guaranteed by `Subbuffer`

        // VUID-vkCmdFillBuffer-dstOffset-00025
        // VUID-vkCmdFillBuffer-size-00028
        // Guaranteed because we take `Subbuffer<[u32]>`

        Ok(())
    }

    /// Writes data to a region of a buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `dst_buffer` was not created from the same device as `self`.
    pub fn update_buffer<D, Dd>(
        &mut self,
        dst_buffer: Subbuffer<D>,
        data: Dd,
    ) -> Result<&mut Self, ClearError>
    where
        D: BufferContents + ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        self.validate_update_buffer(
            dst_buffer.as_bytes(),
            size_of_val(data.deref()) as DeviceSize,
        )?;

        unsafe {
            self.inner.update_buffer(dst_buffer, data)?;
        }

        Ok(self)
    }

    fn validate_update_buffer(
        &self,
        dst_buffer: &Subbuffer<[u8]>,
        data_size: DeviceSize,
    ) -> Result<(), ClearError> {
        let device = self.device();

        // VUID-vkCmdUpdateBuffer-renderpass
        if self.render_pass_state.is_some() {
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

        // VUID-vkCmdUpdateBuffer-commonparent
        assert_eq!(device, dst_buffer.device());

        // VUID-vkCmdUpdateBuffer-dataSize-arraylength
        assert!(data_size != 0);

        // VUID-vkCmdUpdateBuffer-dstBuffer-00034
        if !dst_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::TRANSFER_DST)
        {
            return Err(ClearError::MissingUsage {
                usage: "transfer_dst",
            });
        }

        // VUID-vkCmdUpdateBuffer-dstOffset-00032
        // VUID-vkCmdUpdateBuffer-dataSize-00033
        if data_size > dst_buffer.size() {
            return Err(ClearError::RegionOutOfBufferBounds {
                region_index: 0,
                offset_range_end: data_size,
                buffer_size: dst_buffer.size(),
            });
        }

        // VUID-vkCmdUpdateBuffer-dstOffset-00036
        if dst_buffer.offset() % 4 != 0 {
            return Err(ClearError::OffsetNotAlignedForBuffer {
                region_index: 0,
                offset: dst_buffer.offset(),
                required_alignment: 4,
            });
        }

        // VUID-vkCmdUpdateBuffer-dataSize-00037
        if data_size > 65536 {
            return Err(ClearError::DataTooLarge {
                size: data_size,
                max: 65536,
            });
        }

        // VUID-vkCmdUpdateBuffer-dataSize-00038
        if data_size % 4 != 0 {
            return Err(ClearError::SizeNotAlignedForBuffer {
                region_index: 0,
                size: data_size,
                required_alignment: 4,
            });
        }

        Ok(())
    }
}

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdClearColorImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
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
            clear_value: _,
            ref regions,
            _ne: _,
        } = &clear_info;

        let command_index = self.commands.len();
        let command_name = "clear_color_image";
        let resources: SmallVec<[_; 8]> = regions
            .iter()
            .cloned()
            .flat_map(|subresource_range| {
                [(
                    ResourceUseRef {
                        command_index,
                        command_name,
                        resource_in_command: ResourceInCommand::Destination,
                        secondary_use_ref: None,
                    },
                    Resource::Image {
                        image: image.clone(),
                        subresource_range,
                        memory: PipelineMemoryAccess {
                            stages: PipelineStages::ALL_TRANSFER,
                            access: AccessFlags::TRANSFER_WRITE,
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
    #[inline]
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
            clear_value: _,
            ref regions,
            _ne: _,
        } = &clear_info;

        let command_index = self.commands.len();
        let command_name = "clear_depth_stencil_image";
        let resources: SmallVec<[_; 8]> = regions
            .iter()
            .cloned()
            .flat_map(|subresource_range| {
                [(
                    ResourceUseRef {
                        command_index,
                        command_name,
                        resource_in_command: ResourceInCommand::Destination,
                        secondary_use_ref: None,
                    },
                    Resource::Image {
                        image: image.clone(),
                        subresource_range,
                        memory: PipelineMemoryAccess {
                            stages: PipelineStages::ALL_TRANSFER,
                            access: AccessFlags::TRANSFER_WRITE,
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

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer(
        &mut self,
        dst_buffer: Subbuffer<[u32]>,
        data: u32,
    ) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            dst_buffer: Subbuffer<[u32]>,
            data: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "fill_buffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.fill_buffer(&self.dst_buffer, self.data);
            }
        }

        let command_index = self.commands.len();
        let command_name = "fill_buffer";
        let resources = [(
            ResourceUseRef {
                command_index,
                command_name,
                resource_in_command: ResourceInCommand::Destination,
                secondary_use_ref: None,
            },
            Resource::Buffer {
                buffer: dst_buffer.as_bytes().clone(),
                range: 0..dst_buffer.size(),
                memory: PipelineMemoryAccess {
                    stages: PipelineStages::ALL_TRANSFER,
                    access: AccessFlags::TRANSFER_WRITE,
                    exclusive: true,
                },
            },
        )];

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd { dst_buffer, data }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    pub unsafe fn update_buffer<D, Dd>(
        &mut self,
        dst_buffer: Subbuffer<D>,
        data: Dd,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        D: BufferContents + ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        struct Cmd<D: ?Sized, Dd> {
            dst_buffer: Subbuffer<D>,
            data: Dd,
        }

        impl<D, Dd> Command for Cmd<D, Dd>
        where
            D: BufferContents + ?Sized,
            Dd: SafeDeref<Target = D> + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "update_buffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.update_buffer(&self.dst_buffer, self.data.deref());
            }
        }

        let command_index = self.commands.len();
        let command_name = "update_buffer";
        let resources = [(
            ResourceUseRef {
                command_index,
                command_name,
                resource_in_command: ResourceInCommand::Destination,
                secondary_use_ref: None,
            },
            Resource::Buffer {
                buffer: dst_buffer.as_bytes().clone(),
                range: 0..size_of_val(data.deref()) as DeviceSize,
                memory: PipelineMemoryAccess {
                    stages: PipelineStages::ALL_TRANSFER,
                    access: AccessFlags::TRANSFER_WRITE,
                    exclusive: true,
                },
            },
        )];

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd { dst_buffer, data }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdClearColorImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
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
        (fns.v1_0.cmd_clear_color_image)(
            self.handle,
            image.inner().image.handle(),
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
    #[inline]
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
        (fns.v1_0.cmd_clear_depth_stencil_image)(
            self.handle,
            image.inner().image.handle(),
            image_layout.into(),
            &clear_value,
            ranges.len() as u32,
            ranges.as_ptr(),
        );
    }

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer(&mut self, dst_buffer: &Subbuffer<[u32]>, data: u32) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_fill_buffer)(
            self.handle,
            dst_buffer.buffer().handle(),
            dst_buffer.offset(),
            dst_buffer.size(),
            data,
        );
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    pub unsafe fn update_buffer<D>(&mut self, dst_buffer: &Subbuffer<D>, data: &D)
    where
        D: BufferContents + ?Sized,
    {
        let fns = self.device.fns();
        (fns.v1_0.cmd_update_buffer)(
            self.handle,
            dst_buffer.buffer().handle(),
            dst_buffer.offset(),
            size_of_val(data) as DeviceSize,
            data as *const _ as *const _,
        );
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

/// Error that can happen when recording a clear command.
#[derive(Clone, Debug)]
pub enum ClearError {
    SyncCommandBufferBuilderError(SyncCommandBufferBuilderError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// Operation forbidden inside of a render pass.
    ForbiddenInsideRenderPass,

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,

    /// The end of the range of accessed array layers of the subresource range of a region is
    /// greater than the number of array layers in the image.
    ArrayLayersOutOfRange {
        region_index: usize,
        array_layers_range_end: u32,
        image_array_layers: u32,
    },

    /// The aspects of the subresource range of a region contain aspects that are not present
    /// in the image, or that are not allowed.
    AspectsNotAllowed {
        region_index: usize,
        aspects: ImageAspects,
        allowed_aspects: ImageAspects,
    },

    /// The provided data has a size larger than the maximum allowed.
    DataTooLarge {
        size: DeviceSize,
        max: DeviceSize,
    },

    /// The format of an image is not supported for this operation.
    FormatNotSupported {
        format: Format,
    },

    /// A specified image layout is not valid for this operation.
    ImageLayoutInvalid {
        image_layout: ImageLayout,
    },

    /// The end of the range of accessed mip levels of the subresource range of a region is greater
    /// than the number of mip levels in the image.
    MipLevelsOutOfRange {
        region_index: usize,
        mip_levels_range_end: u32,
        image_mip_levels: u32,
    },

    /// An image does not have a required format feature.
    MissingFormatFeature {
        format_feature: &'static str,
    },

    /// A resource did not have a required usage enabled.
    MissingUsage {
        usage: &'static str,
    },

    /// The buffer offset of a region is not a multiple of the required buffer alignment.
    OffsetNotAlignedForBuffer {
        region_index: usize,
        offset: DeviceSize,
        required_alignment: DeviceSize,
    },

    /// The end of the range of accessed byte offsets of a region is greater than the size of the
    /// buffer.
    RegionOutOfBufferBounds {
        region_index: usize,
        offset_range_end: DeviceSize,
        buffer_size: DeviceSize,
    },

    /// The buffer size of a region is not a multiple of the required buffer alignment.
    SizeNotAlignedForBuffer {
        region_index: usize,
        size: DeviceSize,
        required_alignment: DeviceSize,
    },
}

impl Error for ClearError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SyncCommandBufferBuilderError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for ClearError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::SyncCommandBufferBuilderError(_) => write!(f, "a SyncCommandBufferBuilderError"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::ForbiddenInsideRenderPass => {
                write!(f, "operation forbidden inside of a render pass")
            }
            Self::NotSupportedByQueueFamily => {
                write!(f, "the queue family doesn't allow this operation")
            }
            Self::ArrayLayersOutOfRange {
                region_index,
                array_layers_range_end,
                image_array_layers,
            } => write!(
                f,
                "the end of the range of accessed array layers ({}) of the subresource range of \
                region {} is greater than the number of array layers in the image ({})",
                array_layers_range_end, region_index, image_array_layers,
            ),
            Self::AspectsNotAllowed {
                region_index,
                aspects,
                allowed_aspects,
            } => write!(
                f,
                "the aspects ({:?}) of the subresource range of region {} contain aspects that \
                are not present in the image, or that are not allowed ({:?})",
                aspects, region_index, allowed_aspects,
            ),
            Self::DataTooLarge { size, max } => write!(
                f,
                "the provided data has a size ({}) greater than the maximum allowed ({})",
                size, max,
            ),
            Self::FormatNotSupported { format } => write!(
                f,
                "the format of the image ({:?}) is not supported for this operation",
                format,
            ),
            Self::ImageLayoutInvalid { image_layout } => write!(
                f,
                "the specified image layout {:?} is not valid for this operation",
                image_layout,
            ),
            Self::MipLevelsOutOfRange {
                region_index,
                mip_levels_range_end,
                image_mip_levels,
            } => write!(
                f,
                "the end of the range of accessed mip levels ({}) of the subresource range of \
                region {} is not less than the number of mip levels in the image ({})",
                mip_levels_range_end, region_index, image_mip_levels,
            ),
            Self::MissingFormatFeature { format_feature } => write!(
                f,
                "the image does not have the required format feature {}",
                format_feature,
            ),
            Self::MissingUsage { usage } => write!(
                f,
                "the resource did not have the required usage {} enabled",
                usage,
            ),
            Self::OffsetNotAlignedForBuffer {
                region_index,
                offset,
                required_alignment,
            } => write!(
                f,
                "the buffer offset ({}) of region {} is not a multiple of the required \
                buffer alignment ({})",
                offset, region_index, required_alignment,
            ),
            Self::RegionOutOfBufferBounds {
                region_index,
                offset_range_end,
                buffer_size,
            } => write!(
                f,
                "the end of the range of accessed byte offsets ({}) of region {} is greater \
                than the size of the buffer ({})",
                offset_range_end, region_index, buffer_size,
            ),
            Self::SizeNotAlignedForBuffer {
                region_index,
                size,
                required_alignment,
            } => write!(
                f,
                "the buffer size ({}) of region {} is not a multiple of the required buffer \
                alignment ({})",
                size, region_index, required_alignment,
            ),
        }
    }
}

impl From<SyncCommandBufferBuilderError> for ClearError {
    fn from(err: SyncCommandBufferBuilderError) -> Self {
        Self::SyncCommandBufferBuilderError(err)
    }
}

impl From<RequirementNotMet> for ClearError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}
